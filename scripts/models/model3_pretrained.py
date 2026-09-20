"""Model 3 - ResNet-50 pretrained ImageNet + DetectionHead (stride 32)."""
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from scripts.config import NUM_CLASSES
from scripts.src.detection_head import DetectionHead


class PretrainedDetector(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = False, unfreeze_from_layer: str = "layer3"):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self._named_stages = {
            "conv1": [resnet.conv1, resnet.bn1],
            "layer1": [resnet.layer1], "layer2": [resnet.layer2],
            "layer3": [resnet.layer3], "layer4": [resnet.layer4],
        }

        for p in self.backbone.parameters():
            p.requires_grad = False
        if not freeze_backbone:
            order = ["conv1", "layer1", "layer2", "layer3", "layer4"]
            if unfreeze_from_layer not in order:
                raise ValueError(f"unfreeze_from_layer phai thuoc {order}, nhan '{unfreeze_from_layer}'")
            for name in order[order.index(unfreeze_from_layer):]:
                for m in self._named_stages[name]:
                    for p in m.parameters():
                        p.requires_grad = True

        self.head = DetectionHead(in_channels=2048, num_classes=num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:  # stage bi freeze -> BN o eval, khong cap nhat running stats
            for mods in self._named_stages.values():
                if not any(p.requires_grad for m in mods for p in m.parameters()):
                    for m in mods:
                        m.eval()
        return self

    def forward(self, x):
        return self.head(self.backbone(x))

    def trainable_parameter_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total