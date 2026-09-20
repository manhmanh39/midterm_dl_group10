"""
DetectionHead - lop head chung, dung lai cho ca 3 model (model1/model2/model3).

Nhan feature map (B, C, G, G) tu backbone (G = grid_size, da duoc thiet ke
dung bang cach chon so tang pooling phu hop voi STRIDE trong config.py),
tra ve prediction map (B, G, G, 5 + NUM_CLASSES):
    5 = [objectness_logit, tx, ty, tw, th]
    NUM_CLASSES = class logits (dung BCEWithLogitsLoss - multi-label khong
    can thiet trong detection 1-object/cell, nhung van dung BCE cho on dinh
    va don gian hoa so voi Softmax + CrossEntropy)

Quan trong: head nay hoat dong tren feature map CON KHONG GIAM CHIEU thanh
vector (khac voi Model classification cu dung Flatten+Linear). Day chinh la
diem khac biet co ban giua classification/localization (1 vector/anh) va
detection thuc su (1 prediction/CELL cua grid).
"""

import torch
import torch.nn as nn

from scripts.config import NUM_CLASSES


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # 2 lop conv 1x1 dong vai tro nhu "per-cell fully-connected" -
        # moi vi tri (i,j) tren feature map duoc xu ly doc lap, giu nguyen
        # cau truc khong gian cua grid (khong Flatten).
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 5 + num_classes, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, C, G, G)
        return: (B, G, G, 5 + num_classes) - raw logits (chua qua sigmoid),
                loss function se tu ap dung sigmoid/BCE phu hop.
        """
        out = self.head(feat)                    # (B, 5+num_classes, G, G)
        out = out.permute(0, 2, 3, 1).contiguous()  # (B, G, G, 5+num_classes)
        return out