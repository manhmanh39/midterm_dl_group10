"""DetectionHead dung chung: (B,C,G,G) -> (B,G,G,5+num_classes) raw logits."""
import torch.nn as nn

from scripts.config import NUM_CLASSES


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = NUM_CLASSES, hidden: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 5 + num_classes, 1),
        )
        # prior objectness ~0.01: tranh collapse ve "khong co gi"
        self.head[-1].bias.data[0] = -4.6

    def forward(self, feat):
        return self.head(feat).permute(0, 2, 3, 1).contiguous()