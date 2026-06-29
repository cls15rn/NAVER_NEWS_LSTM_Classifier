'''PyTorch 기반 LSTM 기사 분류 모델 정의 모듈.'''
from __future__ import annotations

import torch
from torch import nn

class TextLSTMClassifier(nn.Module):


    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes:int) -> None:


        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        last_hidden = hidden[-1]
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)
        return logits




