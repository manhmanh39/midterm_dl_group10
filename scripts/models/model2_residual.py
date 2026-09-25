"""Model 2 - Complex CNN: sequential stages + parallel branches.

Moi ParallelBlock co 4 nhanh chay song song tren cung input roi CONCAT.
Cac block lai duoc xep tuan tu theo depth. Train from scratch.
Tong stride = 16.
"""
import torch
import torch.nn as nn

from scripts.config import BOXES_PER_CELL, NUM_CLASSES
from scripts.src.detection_head import DetectionHead


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, kernel_size=3, padding=1, dilation=1):
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )


class ParallelBlock(nn.Module):
    def __init__(self, in_channels, ch_1x1, ch_3x3_reduce, ch_3x3,
                 ch_dilated_reduce, ch_dilated, pool_proj):
        super().__init__()
        self.branch1 = ConvBNAct(in_channels, ch_1x1, kernel_size=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBNAct(in_channels, ch_3x3_reduce, kernel_size=1, padding=0),
            ConvBNAct(ch_3x3_reduce, ch_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBNAct(in_channels, ch_dilated_reduce, kernel_size=1, padding=0),
            ConvBNAct(ch_dilated_reduce, ch_dilated, kernel_size=3, padding=2, dilation=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBNAct(in_channels, pool_proj, kernel_size=1, padding=0),
        )
        self.out_channels = ch_1x1 + ch_3x3 + ch_dilated + pool_proj

    def forward(self, x):
        return torch.cat([
            self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)
        ], dim=1)


class NonSequentialCNNDetector(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, boxes_per_cell: int = BOXES_PER_CELL):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # /2
        )

        self.block1 = ParallelBlock(32, 16, 16, 32, 16, 32, 16)   # 96
        self.pool1 = nn.MaxPool2d(2, 2)                             # /4

        self.block2 = ParallelBlock(self.block1.out_channels, 32, 32, 64, 32, 64, 32)  # 192
        self.pool2 = nn.MaxPool2d(2, 2)                             # /8

        self.block3 = ParallelBlock(self.block2.out_channels, 64, 64, 128, 64, 128, 64) # 384
        self.pool3 = nn.MaxPool2d(2, 2)                             # /16

        self.block4 = ParallelBlock(self.block3.out_channels, 128, 128, 256, 128, 256, 128) # 768
        self.head = DetectionHead(self.block4.out_channels, num_classes, boxes_per_cell)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        return self.head(x)
