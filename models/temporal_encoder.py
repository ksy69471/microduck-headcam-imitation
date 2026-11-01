"""Temporal Transformer over per-frame vision features."""

from __future__ import annotations

import torch
from torch import nn


class TemporalTransformer(nn.Module):
    def __init__(self, dim: int = 128, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.pos = nn.Parameter(torch.zeros(1, 512, dim))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        t = x.shape[1]
        x = x + self.pos[:, :t]
        return self.encoder(x)
