'''텍스트 정제, 토큰화, 패딩, 라벨 인코딩을 담당하는 모듈.'''

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple, Sequence

import numpy as np


STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "after", "as", "by", "at", "during",
}


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """영문 기사 문장에서 특수문자와 불필요한 단어를 제거한다."""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [w for w in tokens if  w not in STOP_WORDS]
    return " ".join(tokens)


def build_vocab(texts: Sequence[str], max_vocab: int) -> Dict[str, int]:
    """학습 데이터에서 자주 등장한 단어를 정수 인덱스로 매핑하는 사전을 만든다."""

    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(text.split())
    most_common = counter.most_common(max_vocab - 2)
    vocab = {"<PAD>": 0, "<OOV>": 1}
    for index, (word, _) in enumerate(most_common, start=2):
        vocab[word] = index
    return vocab


def texts_to_sequences(texts: Sequence[str], vocab: Dict[str, int]) -> List[List[int]]:
    """문장 목록을 정수 토큰 시퀀스 목록으로 변환한다."""

    sequences: List[List[int]] = []
    for text in texts:
        seq = [vocab.get(word, vocab["<OOV>"]) for word in text.split()]
        sequences.append(seq)
    return sequences


def pad_sequences(sequences: Sequence[Sequence[int]], max_len: int) -> np.ndarray:
    """서로 다른 길이의 정수 시퀀스를 동일한 길이의 2차원 배열로 맞춘다."""

    padded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        truncated = list(seq)[-max_len:]
        padded[i, -len(truncated):] = truncated if truncated else []
    return padded


def encode_labels(labels: Sequence[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """문자열 라벨을 정수 라벨로 변환하고 양방향 라벨 사전을 반환한다."""

    label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    encoded = np.array([label_to_id[label] for label in labels], dtype=np.int64)
    return encoded, label_to_id , id_to_label
