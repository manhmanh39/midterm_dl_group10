"""
Model 1 - Sequential CNN Object Detector

Backbone: TUAN TU thuan (Conv -> BN -> ReLU -> Pool) x 5, khong nhanh re,
khong skip connection. Day la kien truc "phang" don gian nhat trong 3 model.

Backbone giam kich thuoc anh tu IMAGE_SIZE ve GRID_SIZE = IMAGE_SIZE / 32
(5 lan pool, moi lan /2 -> tong /32), sau do DetectionHead du bao truc tiep
tren feature map (khong Flatten) - day la diem khac biet co ban voi CNN
classification thong thuong.
"""

import torch.nn as nn

from scripts.config import NUM_CLASSES
from scripts.src.detection_head import DetectionHead


class SequentialCNNDetector(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # 5 khoi Conv-BN-ReLU-Pool xep TUAN TU, tong stride = 2^5 = 32
        self.backbone = nn.Sequential(
            # Block 1: 3 -> 32,  stride tich luy /2
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32 -> 64, /4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64 -> 128, /8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 128 -> 256, /16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 5: 256 -> 512, /32
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.head = DetectionHead(in_channels=512, num_classes=num_classes)

    def forward(self, x):
        feat = self.backbone(x)      # (B, 512, G, G)
        return self.head(feat)       # (B, G, G, 5+num_classes)