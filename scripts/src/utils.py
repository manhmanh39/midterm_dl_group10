"""
Loss va cac metric cho object detection anchor-free 1-scale (kieu YOLOv1).

CombinedLocalizationLoss (ten giu nguyen de tuong thich train.py/evaluate.py)
gio day tinh loss tren TOAN BO GRID, khong con la 1 bbox/anh nhu truoc:

    L = LAMBDA_COORD  * L_coord   (chi tren cell co object)
      + LAMBDA_OBJ    * L_obj     (chi tren cell co object)
      + LAMBDA_NOOBJ  * L_noobj   (chi tren cell KHONG co object)
      + LAMBDA_CLASS  * L_class   (chi tren cell co object)

calculate_iou gio tinh mAP-style: IoU trung binh giua predicted box va GT box
tren cac cell CO object thuc su (dung de theo doi qua trinh train, khong
phai metric mAP day du - mAP thuc te nen tinh rieng bang scripts/src/metrics.py
sau khi decode+NMS toan bo tap du lieu).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from scripts.config import LAMBDA_COORD, LAMBDA_OBJ, LAMBDA_NOOBJ, LAMBDA_CLASS


class CombinedLocalizationLoss(nn.Module):
    def __init__(
        self,
        lambda_coord: float = LAMBDA_COORD,
        lambda_obj: float = LAMBDA_OBJ,
        lambda_noobj: float = LAMBDA_NOOBJ,
        lambda_class: float = LAMBDA_CLASS,
    ):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

        self.bce_obj = nn.BCEWithLogitsLoss(reduction="sum")
        self.bce_class = nn.BCEWithLogitsLoss(reduction="sum")
        self.mse_coord = nn.MSELoss(reduction="sum")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, G, G, 5 + num_classes), pred la RAW LOGITS
        (chua qua sigmoid), target da la gia tri thuc (0/1 cho objectness
        va one-hot class, [0,1] cho tx,ty,tw,th).
        """
        obj_mask = target[..., 0] == 1.0       # (B, G, G) bool - cell co object
        noobj_mask = ~obj_mask

        batch_size = pred.shape[0]

        # --- Objectness loss ---
        obj_loss = self.bce_obj(pred[..., 0][obj_mask], target[..., 0][obj_mask]) if obj_mask.any() else pred.sum() * 0.0
        noobj_loss = self.bce_obj(pred[..., 0][noobj_mask], target[..., 0][noobj_mask]) if noobj_mask.any() else pred.sum() * 0.0

        if obj_mask.any():
            # --- Coordinate loss (chi tren cell co object) ---
            pred_box = pred[..., 1:5][obj_mask]      # (N, 4) raw logits cho tx,ty,tw,th
            target_box = target[..., 1:5][obj_mask]  # (N, 4)

            # tx, ty di qua sigmoid de ep ve [0,1] (offset trong cell)
            pred_tx_ty = torch.sigmoid(pred_box[:, 0:2])
            # tw, th di qua sigmoid de ep ve [0,1] (vi target cung la [0,1]
            # normalized theo ca anh, khong dung exp() nhu YOLOv2/v3 vi
            # khong co anchor prior o day)
            pred_tw_th = torch.sigmoid(pred_box[:, 2:4])

            pred_box_decoded = torch.cat([pred_tx_ty, pred_tw_th], dim=1)
            coord_loss = self.mse_coord(pred_box_decoded, target_box)

            # --- Classification loss (chi tren cell co object) ---
            pred_cls = pred[..., 5:][obj_mask]
            target_cls = target[..., 5:][obj_mask]
            class_loss = self.bce_class(pred_cls, target_cls)
        else:
            coord_loss = pred.sum() * 0.0
            class_loss = pred.sum() * 0.0

        total = (
            self.lambda_coord * coord_loss
            + self.lambda_obj * obj_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_class * class_loss
        )
        return total / batch_size


def _xywh_to_xyxy_grid(tx, ty, tw, th, gx, gy, grid_size):
    """Chuyen (tx,ty offset-trong-cell, tw,th normalized-ca-anh) -> (x1,y1,x2,y2) normalized ca anh."""
    cx = (gx + tx) / grid_size
    cy = (gy + ty) / grid_size
    x1 = cx - tw / 2
    y1 = cy - th / 2
    x2 = cx + tw / 2
    y2 = cy + th / 2
    return x1, y1, x2, y2


@torch.no_grad()
def calculate_iou(pred: torch.Tensor, target: torch.Tensor, grid_size: int) -> float:
    """
    IoU trung binh giua predicted box va GT box, CHI tren cac cell co
    object thuc su theo target (dung de theo doi tien trinh train qua tung
    epoch - khong phai mAP day du).
    """
    obj_mask = target[..., 0] == 1.0
    if not obj_mask.any():
        return 0.0

    B, G, _, _ = target.shape
    device = pred.device
    gy_grid, gx_grid = torch.meshgrid(
        torch.arange(G, device=device), torch.arange(G, device=device), indexing="ij"
    )
    gy_grid = gy_grid.unsqueeze(0).expand(B, -1, -1)[obj_mask].float()
    gx_grid = gx_grid.unsqueeze(0).expand(B, -1, -1)[obj_mask].float()

    pred_box = pred[..., 1:5][obj_mask]
    pred_tx_ty = torch.sigmoid(pred_box[:, 0:2])
    pred_tw_th = torch.sigmoid(pred_box[:, 2:4])

    target_box = target[..., 1:5][obj_mask]

    px1, py1, px2, py2 = _xywh_to_xyxy_grid(
        pred_tx_ty[:, 0], pred_tx_ty[:, 1], pred_tw_th[:, 0], pred_tw_th[:, 1], gx_grid, gy_grid, G
    )
    tx1, ty1, tx2, ty2 = _xywh_to_xyxy_grid(
        target_box[:, 0], target_box[:, 1], target_box[:, 2], target_box[:, 3], gx_grid, gy_grid, G
    )

    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    p_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    t_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = p_area + t_area - inter_area

    iou = torch.where(union > 0, inter_area / union, torch.zeros_like(union))
    return iou.mean().item()


def plot_history(train_losses, val_losses, train_ious, val_ious, model_name: str, out_dir: str = "checkpoints"):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_losses, label="Train Loss", marker="o")
    axes[0].plot(epochs, val_losses, label="Val Loss", marker="o")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_ious, label="Train IoU (obj cells)", marker="o")
    axes[1].plot(epochs, val_ious, label="Val IoU (obj cells)", marker="o")
    axes[1].set_title(f"{model_name} - IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name}_history.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"--> Da luu bieu do tai {out_path}")