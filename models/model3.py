from typing import Literal
import torch
import torch.nn as nn
import torchvision.models as models


class TransferLearningModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 15,
        backbone_name: Literal[
            "resnet18", "resnet50", "mobilenet_v3", "efficientnet_b0", "convnext_tiny"
        ] = "resnet50",
        pretrained: bool = True,
        freeze_base: bool = True,
        dropout: float = 0.3,
        image_size: int = 224,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.image_size = image_size

        if backbone_name == "resnet18":
            w = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=w)
            in_f = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

        elif backbone_name == "resnet50":
            w = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=w)
            in_f = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

        elif backbone_name == "mobilenet_v3":
            w = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=w)
            in_f = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_f, 256), nn.Hardswish(), nn.Dropout(dropout), nn.Linear(256, num_classes)
            )

        elif backbone_name == "efficientnet_b0":
            w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=w)
            in_f = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

        elif backbone_name == "convnext_tiny":
            w = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.backbone = models.convnext_tiny(weights=w)
            in_f = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(in_f, num_classes)

        else:
            raise ValueError(f"Không hỗ trợ backbone: {backbone_name}")

        if freeze_base:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        head = getattr(self.backbone, "fc", None) or getattr(self.backbone, "classifier", None)
        if head is not None:
            for p in head.parameters():
                p.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_model3_transfer(
    num_classes: int = 15,
    backbone_name: str = "resnet50",
    pretrained: bool = True,
    freeze_base: bool = True,
    image_size: int = 224,
    dropout: float = 0.3,
) -> TransferLearningModel:
    return TransferLearningModel(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=pretrained,
        freeze_base=freeze_base,
        image_size=image_size,
        dropout=dropout,
    )


if __name__ == "__main__":
    model = get_model3_transfer(num_classes=15, backbone_name="resnet18", pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"[Model 3] Output shape: {out.shape}")
    assert out.shape == (2, 15)
    print("[Model 3] OK")