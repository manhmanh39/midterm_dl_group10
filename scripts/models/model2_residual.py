"""
Model 2 - Non-Sequential (Parallel) CNN Object Detector

Backbone: ket hop TUAN TU (thu tu cac stage) va SONG SONG (nhieu nhanh
trong moi InceptionBlock chay dong thoi tren cung input, roi CONCAT - khac
biet ro voi residual add). Day la "Complex CNN" theo yeu cau mon hoc.

Tong stride: stem (/2) + 4 stage (moi stage 1 InceptionBlock + 1 MaxPool /2)
= /2 * /2^4 = /32, khop voi model1 va config.STRIDE=32 -> cung grid size,
co the so sanh cong bang giua 2 model.
"""

import torch
import torch.nn as nn

from scripts.config import NUM_CLASSES
from scripts.src.detection_head import DetectionHead


class InceptionBlock(nn.Module):
    """
    Khoi tich chap da nhanh SONG SONG (lay cam hung Inception/GoogLeNet).
    4 nhanh xu ly CUNG MOT input DONG THOI, sau do ghep (concat) theo
    chieu channel. Khac biet co ban voi residual block: o day CA 4 nhanh
    deu la nhanh dac trung chinh (khong co nhanh "shortcut" cong lai),
    va phep ket hop la CONCAT (tang so channel) chu khong phai ADD
    (giu nguyen so channel).
    """

    def __init__(self, in_channels, ch_1x1, ch_3x3_reduce, ch_3x3, ch_5x5_reduce, ch_5x5, pool_proj):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch_1x1, kernel_size=1),
            nn.BatchNorm2d(ch_1x1),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch_3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch_3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_3x3_reduce, ch_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_3x3),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch_5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch_5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_5x5_reduce, ch_5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_5x5, ch_5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_5x5),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True),
        )

        self.out_channels = ch_1x1 + ch_3x3 + ch_5x5 + pool_proj

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class NonSequentialCNNDetector(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Stem: /2
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Stage 1: InceptionBlock (song song) + Pool -> /4
        self.inception1 = InceptionBlock(
            in_channels=32, ch_1x1=16, ch_3x3_reduce=16, ch_3x3=32,
            ch_5x5_reduce=8, ch_5x5=16, pool_proj=16,
        )  # out = 80
        self.pool1 = nn.MaxPool2d(2, 2)

        # Stage 2: -> /8
        self.inception2 = InceptionBlock(
            in_channels=self.inception1.out_channels, ch_1x1=32, ch_3x3_reduce=32,
            ch_3x3=64, ch_5x5_reduce=16, ch_5x5=32, pool_proj=32,
        )  # out = 160
        self.pool2 = nn.MaxPool2d(2, 2)

        # Stage 3: -> /16
        self.inception3 = InceptionBlock(
            in_channels=self.inception2.out_channels, ch_1x1=64, ch_3x3_reduce=64,
            ch_3x3=128, ch_5x5_reduce=32, ch_5x5=64, pool_proj=64,
        )  # out = 320
        self.pool3 = nn.MaxPool2d(2, 2)

        # Stage 4: -> /32 (them 1 stage nua de dat dung tong stride 32,
        # khop voi model1 va grid size cua config)
        self.inception4 = InceptionBlock(
            in_channels=self.inception3.out_channels, ch_1x1=128, ch_3x3_reduce=128,
            ch_3x3=256, ch_5x5_reduce=64, ch_5x5=128, pool_proj=128,
        )  # out = 640
        self.pool4 = nn.MaxPool2d(2, 2)

        self.head = DetectionHead(in_channels=self.inception4.out_channels, num_classes=num_classes)

    def forward(self, x):
        x = self.stem(x)              # /2

        x = self.inception1(x)        # song song: 4 nhanh -> concat
        x = self.pool1(x)              # tuan tu: /4

        x = self.inception2(x)
        x = self.pool2(x)              # /8

        x = self.inception3(x)
        x = self.pool3(x)              # /16

        x = self.inception4(x)
        x = self.pool4(x)              # /32

        return self.head(x)            # (B, G, G, 5+num_classes)

    def trainable_parameter_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total