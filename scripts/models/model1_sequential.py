"""Model 1 - Simple Sequential CNN Object Detector.

Rubric-safe: cac stage chinh luan phien Conv -> Pool, khong residual, khong branch.
Tong stride = 16 de giu spatial resolution cho lesion nho.
"""
import torch.nn as nn

from scripts.config import BOXES_PER_CELL, NUM_CLASSES
from scripts.src.detection_head import DetectionHead


class _ConvStage(nn.Sequential):
    def __init__(self, cin, cout, pool=True):
        layers = [
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        super().__init__(*layers)


class SequentialCNNDetector(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, boxes_per_cell: int = BOXES_PER_CELL):
        super().__init__()
        self.backbone = nn.Sequential(
            _ConvStage(3, 32, pool=True),    # /2
            _ConvStage(32, 64, pool=True),   # /4
            _ConvStage(64, 128, pool=True),  # /8
            _ConvStage(128, 256, pool=True), # /16
            _ConvStage(256, 512, pool=False),
        )
        self.head = DetectionHead(512, num_classes, boxes_per_cell)

    def forward(self, x):
        return self.head(self.backbone(x))
