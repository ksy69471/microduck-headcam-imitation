"""Decode temporal features to joint / root / contact trajectories."""

from __future__ import annotations

import torch
from torch import nn


class MotionDecoder(nn.Module):
    def __init__(self, dim: int, n_joints: int):
        super().__init__()
        self.n_joints = n_joints
        self.joint = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_joints))
        self.root_pos = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
        self.root_quat = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4))
        self.contact = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))  # L/R feet logits

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, T, D)
        quat = self.root_quat(x)
        quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
        return {
            "joint": self.joint(x),
            "root_pos": self.root_pos(x),
            "root_quat": quat,
            "contact_logits": self.contact(x),
        }
