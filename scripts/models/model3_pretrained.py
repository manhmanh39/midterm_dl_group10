"""
Model 3 - Pretrained Backbone Object Detector (Transfer Learning / Fine-tuning)

Dung ResNet-50 pretrained ImageNet lam feature extractor. KHAC voi ban
classification truoc (dung avgpool+fc), o day GIU LAI feature map cuoi
(truoc avgpool) de DetectionHead du bao tren tung cell cua grid - dung
dung cach ma cac detector thuc te (Faster R-CNN, RetinaNet, YOLO) dung
backbone pretrained: lay feature map, khong lay vector da pool.

ResNet-50 co tong stride tu nhien la 32 (conv1 /2, maxpool /2, layer1 giu
nguyen /4, layer2 /8, layer3 /16, layer4 /32) -> grid size dau ra CHINH
XAC bang IMAGE_SIZE/32, khop voi config.GRID_SIZE va voi model1/model2,
dam bao 3 model co the so sanh cong bang (cung grid size, cung detection
head, cung loss function).

Ho tro 2 che do:
    - freeze_backbone=True  : dong bang toan bo ResNet-50, chi train head
      (transfer learning thuan).
    - freeze_backbone=False : fine-tune tu unfreeze_from_layer tro di.
"""

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from scripts.config import NUM_CLASSES
from scripts.src.detection_head import DetectionHead


class PretrainedDetector(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = True, unfreeze_from_layer: str = "layer4"):
        """
        Parameters
        ----------
        num_classes : so luong lop benh
        freeze_backbone : True -> dong bang toan bo backbone (transfer learning thuan)
        unfreeze_from_layer : chi dung khi freeze_backbone=False. Mo khoa tu
            layer nay tro di (theo thu tu conv1 -> layer1 -> layer2 -> layer3 -> layer4).
        """
        super().__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Bo avgpool va fc, chi giu phan feature extractor (conv1 -> layer4)
        # de output ra feature map (B, 2048, G, G) thay vi vector da pool.
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        # Luu tham chieu ten layer de dieu khien freeze/unfreeze
        self._named_stages = {
            "conv1": [resnet.conv1, resnet.bn1],
            "layer1": [resnet.layer1],
            "layer2": [resnet.layer2],
            "layer3": [resnet.layer3],
            "layer4": [resnet.layer4],
        }

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

            layer_order = ["conv1", "layer1", "layer2", "layer3", "layer4"]
            if unfreeze_from_layer not in layer_order:
                raise ValueError(
                    f"unfreeze_from_layer phai la mot trong {layer_order}, "
                    f"nhan duoc '{unfreeze_from_layer}'"
                )
            start_idx = layer_order.index(unfreeze_from_layer)
            for name in layer_order[start_idx:]:
                for module in self._named_stages[name]:
                    for param in module.parameters():
                        param.requires_grad = True

        # ResNet-50 layer4 output = 2048 channels
        self.head = DetectionHead(in_channels=2048, num_classes=num_classes)

    def forward(self, x):
        feat = self.backbone(x)   # (B, 2048, G, G), G = image_size/32
        return self.head(feat)    # (B, G, G, 5+num_classes)

    def trainable_parameter_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total