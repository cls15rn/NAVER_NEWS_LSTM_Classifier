

from __future__ import annotations

import pickle
from typing import Dict, Tuple

import torch

from app.config import Config
from app.model import TextLSTMClassifier
from app.preprocess import clean_text, pad_sequences, texts_to_sequences


def load_artifacts(config: Config) -> Tuple[TextLSTMClassifier, Dict[str, object]]:


    meta_path = config.model_path.replace('.pt', '_meta.pkl')
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    model = TextLSTMClassifier(
        vocab_size = len(metadata['vocab']),
        embed_dim=metadata["config"].embed_dim,
        hidden_dim=metadata["config"].hidden_dim,
        num_classes=len(metadata["label_to_id"])
    )
    model.load_state_dict(torch.load(config.model_path, map_location="cpu"))
    model.eval()
    return model, metadata


def predict_text(text: str, model: TextLSTMClassifier, metadata: Dict[str, object], config: Config) -> str:

    cleaned = clean_text(text)
    sequence = texts_to_sequences([cleaned], metadata["vocab"])[0]
    padded = pad_sequences([sequence], config.max_len)
    x = torch.tensor(padded, dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        pred_id = int(torch.argmax(logits, dim=1).item())
    return metadata["id_to_label"][pred_id]




