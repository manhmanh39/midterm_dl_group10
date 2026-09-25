"""Shared sequential detection head cho ca 3 model.

Output: (B, G, G, A, 5 + C), raw logits.
Head duoc giu sequential de Model 1 van ro rang la simple/sequential CNN.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from scripts.config import BOXES_PER_CELL, NUM_CLASSES


class DetectionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = NUM_CLASSES,
        boxes_per_cell: int = BOXES_PER_CELL,
        hidden: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.boxes_per_cell = boxes_per_cell
        self.pred_dim = 5 + num_classes
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, boxes_per_cell * self.pred_dim, 1),
        )
        self._init_biases()

    def _init_biases(self):
        with torch.no_grad():
            bias = self.head[-1].bias.view(self.boxes_per_cell, self.pred_dim)
            bias.zero_()
            # objectness prior ~= 0.01
            bias[:, 0] = -2.2

    def initialize_box_prior(self, median_w: float, median_h: float):
        """Khoi tao width/height logits gan median bbox size cua train set."""
        eps = 1e-4
        w = min(max(float(median_w), eps), 1 - eps)
        h = min(max(float(median_h), eps), 1 - eps)
        w_logit = math.log(w / (1 - w))
        h_logit = math.log(h / (1 - h))
        with torch.no_grad():
            bias = self.head[-1].bias.view(self.boxes_per_cell, self.pred_dim)
            bias[:, 3] = w_logit
            bias[:, 4] = h_logit

    def forward(self, feat):
        x = self.head(feat)
        B, _, G, G2 = x.shape
        if G != G2:
            raise ValueError(f"Detection feature map phai vuong, nhan {G}x{G2}")
        x = x.view(B, self.boxes_per_cell, self.pred_dim, G, G)
        return x.permute(0, 3, 4, 1, 2).contiguous()
