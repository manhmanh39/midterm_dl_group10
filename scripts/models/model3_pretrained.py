"""Model 3 - Improved pretrained ResNet-50 detector.

Improvements vs plain pretrained backbone:
- neutralize RGB-specific conv1 weighting for engineered X-ray 3 channels
- fuse layer3 (/16) + upsampled layer4 (/32) with a light FPN-like neck
- staged fine-tuning is controlled from train.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from scripts.config import BOXES_PER_CELL, NUM_CLASSES
from scripts.src.detection_head import DetectionHead


class PretrainedDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = False,
        unfreeze_from_layer: str = "layer3",
        boxes_per_cell: int = BOXES_PER_CELL,
    ):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Engineered channels are not literal RGB. Average pretrained RGB filters
        # then repeat, preserving edge/texture priors without channel-color bias.
        with torch.no_grad():
            w = resnet.conv1.weight.data
            resnet.conv1.weight.data.copy_(w.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1))

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # /4
        self.layer1 = resnet.layer1   # /4
        self.layer2 = resnet.layer2   # /8
        self.layer3 = resnet.layer3   # /16, 1024
        self.layer4 = resnet.layer4   # /32, 2048

        self.lateral3 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU(inplace=True)
        )
        self.lateral4 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.SiLU(inplace=True),
        )
        self.head = DetectionHead(256, num_classes, boxes_per_cell)

        self._stage_modules = {
            "conv1": [self.stem],
            "layer1": [self.layer1],
            "layer2": [self.layer2],
            "layer3": [self.layer3],
            "layer4": [self.layer4],
        }
        self._stage_order = ["conv1", "layer1", "layer2", "layer3", "layer4"]
        if freeze_backbone:
            self.set_backbone_trainable(None)
        else:
            self.set_backbone_trainable(unfreeze_from_layer)

    def backbone_parameters(self):
        for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            yield from module.parameters()

    def set_backbone_trainable(self, unfreeze_from_layer: str | None):
        for p in self.backbone_parameters():
            p.requires_grad = False
        if unfreeze_from_layer is None:
            return
        if unfreeze_from_layer not in self._stage_order:
            raise ValueError(f"unfreeze_from_layer phai thuoc {self._stage_order}")
        start = self._stage_order.index(unfreeze_from_layer)
        for name in self._stage_order[start:]:
            for module in self._stage_modules[name]:
                for p in module.parameters():
                    p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # Frozen pretrained stages keep BN running stats fixed.
            for modules in self._stage_modules.values():
                if not any(p.requires_grad for m in modules for p in m.parameters()):
                    for m in modules:
                        m.eval()
        return self

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        p3 = self.lateral3(c3)
        p4 = F.interpolate(self.lateral4(c4), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.fuse(torch.cat([p3, p4], dim=1))
        return self.head(feat)

    def trainable_parameter_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
