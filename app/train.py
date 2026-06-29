"""데이터 전처리, 모델 학습, 평가, 저장을 수행하는 모듈."""

from __future__ import annotations

import os
import pickle
import random
from typing import Dict, Tuple

import numpy as np
import torch

torch.set_num_threads(1)  # 작은 실습 데이터에서는 CPU 스레드를 1개로 제한하여 실행 환경별 지연을 줄인다.
torch.backends.mkldnn.enabled = False  # 일부 CPU 환경에서 LSTM 연산이 오래 멈추는 문제를 피하기 위해 MKLDNN을 비활성화한다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import Config
from app.data import load_sample_data
from app.model import TextLSTMClassifier
from app.preprocess import build_vocab, clean_text, encode_labels, pad_sequences, texts_to_sequences

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 폰트 지정 (Windows)
plt.rcParams["axes.unicode_minus"] = False      # 마이너스 기호 깨짐 방지


def set_seed(seed: int) -> None:
    """학습 결과가 최대한 동일하게 재현되도록 난수를 고정한다."""

    random.seed(seed)              # 파이썬 random 모듈의 난수를 고정한다.
    np.random.seed(seed)           # NumPy 난수를 고정한다.
    torch.manual_seed(seed)        # PyTorch CPU 난수를 고정한다.


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


def train_model(config: Config) -> Tuple[TextLSTMClassifier, Dict[str, object]]:
    """샘플 BBC 기사 데이터를 사용해 LSTM 문서 분류 모델을 학습한다."""

    set_seed(config.random_state)                                      # 실험 재현성을 위해 난수를 고정한다.
    raw_texts, labels = load_sample_data()                             # 기사 문장과 카테고리 라벨을 불러온다.
    cleaned_texts = [clean_text(text) for text in raw_texts]           # 각 기사 문장에 대해 정제와 제외어 처리를 수행한다.
    vocab = build_vocab(cleaned_texts, config.max_vocab)               # 학습 데이터 기준으로 단어 사전을 만든다.
    sequences = texts_to_sequences(cleaned_texts, vocab)               # 정제된 문장을 정수 토큰 시퀀스로 변환한다.
    x = pad_sequences(sequences, config.max_len)                       # 모든 기사 길이를 동일하게 패딩한다.
    y, label_to_id, id_to_label = encode_labels(labels)                # 문자열 카테고리를 정수 라벨로 변환한다.

    from collections import Counter
    print("라벨 분포:", Counter(y_train if 'y' not in dir() else y))

    x_train, x_test, y_train, y_test = train_test_split(               # 학습용 입력, 평가용 입력, 학습용 정답, 평가용 정답으로 사분할한다.
        x, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train)) # NumPy 배열을 PyTorch 학습 데이터셋으로 변환한다.
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))    # 평가 데이터도 PyTorch 데이터셋으로 변환한다.
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True) # 학습 데이터를 미니배치 단위로 섞어 공급한다.
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)                 # 평가 데이터는 순서대로 공급한다.

    model = TextLSTMClassifier(                                      # LSTM 기반 텍스트 분류 모델을 생성한다.
        vocab_size=len(vocab), embed_dim=config.embed_dim, hidden_dim=config.hidden_dim, num_classes=len(label_to_id)
    )
    criterion = nn.CrossEntropyLoss()                                # 다중 분류 손실함수를 생성한다.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) # Adam 최적화 알고리즘을 설정한다.

    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, config.epochs + 1):
        # ---- 학습 단계 ----
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        # ---- 검증 단계 (test 데이터로 loss만 계산, 가중치 갱신 X) ----
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_total += loss.item()
        val_loss = val_total / len(test_loader)

        # ---- 기록 & 출력 ----
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        print(f"Epoch {epoch:02d}/{config.epochs} - train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

    model.eval()                                                      # 모델을 평가 모드로 전환한다.
    all_preds = []                                                    # 전체 평가 예측값을 저장할 리스트이다.
    all_targets = []                                                  # 전체 평가 정답값을 저장할 리스트이다.
    with torch.no_grad():                                             # 평가 과정에서는 기울기 계산을 하지 않는다.
        for batch_x, batch_y in test_loader:                          # 평가 데이터를 미니배치 단위로 꺼낸다.
            logits = model(batch_x)                                   # 각 평가 샘플의 클래스 점수를 계산한다.
            preds = torch.argmax(logits, dim=1)                       # 가장 점수가 높은 클래스를 예측 라벨로 선택한다.
            all_preds.extend(preds.tolist())                          # 예측 라벨을 파이썬 리스트에 추가한다.
            all_targets.extend(batch_y.tolist())                      # 정답 라벨을 파이썬 리스트에 추가한다.

    accuracy = accuracy_score(all_targets, all_preds)                 # 전체 평가 정확도를 계산한다.
    print(f"평가 정확도: {accuracy:.4f}")                              # 평가 정확도를 출력한다.
    print(classification_report(all_targets, all_preds, target_names=[id_to_label[i] for i in range(len(id_to_label))], zero_division=0)) # 상세 평가표를 출력한다.

    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)     # 모델 저장 폴더가 없으면 생성한다.
    torch.save(model.state_dict(), config.model_path)                  # 학습된 모델 가중치를 파일로 저장한다.
    with open(config.model_path.replace(".pt", "_meta.pkl"), "wb") as f: # 전처리와 예측에 필요한 부가 정보를 저장할 파일을 연다.
        pickle.dump({"vocab": vocab, "label_to_id": label_to_id, "id_to_label": id_to_label, "config": config}, f) # 사전과 라벨 정보를 저장한다.

    metadata = {"vocab": vocab, "label_to_id": label_to_id, "id_to_label": id_to_label, "accuracy": accuracy} # 호출자에게 반환할 메타데이터를 구성한다.

    plot_confusion_matrix(all_targets, all_preds, class_names=["경제/기업", "사회", "연예"])
    plot_loss_history(train_loss_history, val_loss_history)

    return model, metadata                                            # 학습된 모델과 메타데이터를 반환한다.


def plot_loss_history(train_history, val_history):
    epochs = range(1, len(train_history) + 1)
    plt.figure()
    plt.plot(epochs, train_history, marker="o", label="train_loss")
    plt.plot(epochs, val_history, marker="o", label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training / Validation Loss per Epoch")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("loss_history.png", dpi=150); plt.show()