"""Model 1 - Sequential CNN Object Detector (5 block x [Conv-BN-ReLU]x2 + MaxPool, stride 32)."""
import torch.nn as nn

from scripts.config import NUM_CLASSES
from scripts.src.detection_head import DetectionHead


def _block(cin, cout):
    return [
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    ]


class SequentialCNNDetector(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = nn.Sequential(
            *_block(3, 32), *_block(32, 64), *_block(64, 128), *_block(128, 256), *_block(256, 512),
        )
        self.head = DetectionHead(in_channels=512, num_classes=num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))

    def trainable_parameter_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total