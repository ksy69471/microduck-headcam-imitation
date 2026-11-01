"""Small CNN vision encoder for Video-to-Motion."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 128, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [0,1]
        h = self.net(x)
        h = h.flatten(1)
        return self.proj(h)
