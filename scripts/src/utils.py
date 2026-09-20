"""Loss (focal objectness + GIoU/MSE box + softmax CE class) va metric theo doi train."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

from scripts.config import LAMBDA_COORD, LAMBDA_OBJ, LAMBDA_CLASS


class CombinedLocalizationLoss(nn.Module):
    def __init__(self, lambda_coord=LAMBDA_COORD, lambda_obj=LAMBDA_OBJ,
                 lambda_class=LAMBDA_CLASS, focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.lambda_coord, self.lambda_obj, self.lambda_class = lambda_coord, lambda_obj, lambda_class
        self.gamma, self.alpha = focal_gamma, focal_alpha

    def forward(self, pred, target):
        G = pred.shape[1]
        obj_mask = target[..., 0] > 0.5
        num_pos = obj_mask.sum().clamp(min=1).float()

        logit, tgt = pred[..., 0], target[..., 0]
        p = torch.sigmoid(logit)
        ce = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
        p_t = p * tgt + (1 - p) * (1 - tgt)
        a_t = self.alpha * tgt + (1 - self.alpha) * (1 - tgt)
        obj_loss = (a_t * (1 - p_t) ** self.gamma * ce).sum() / num_pos

        if obj_mask.any():
            pb, tb = pred[..., 1:5][obj_mask], target[..., 1:5][obj_mask]
            pxy, pwh = torch.sigmoid(pb[:, :2]), torch.sigmoid(pb[:, 2:])
            txy, twh = tb[:, :2], tb[:, 2:]

            def to_xyxy(xy, wh):
                wh = wh * G
                return torch.cat([xy - wh / 2, xy + wh / 2], dim=1)

            giou = generalized_box_iou_loss(to_xyxy(pxy, pwh), to_xyxy(txy, twh), reduction="sum")
            coord_loss = (giou + F.mse_loss(pxy, txy, reduction="sum")) / num_pos

            cls_target = target[..., 5:][obj_mask].argmax(dim=-1)
            cls_loss = F.cross_entropy(pred[..., 5:][obj_mask], cls_target, reduction="sum") / num_pos
        else:
            coord_loss = cls_loss = pred.sum() * 0.0

        return self.lambda_coord * coord_loss + self.lambda_obj * obj_loss + self.lambda_class * cls_loss


def _xywh_to_xyxy_grid(tx, ty, tw, th, gx, gy, grid_size):
    cx = (gx + tx) / grid_size
    cy = (gy + ty) / grid_size
    return cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2


@torch.no_grad()
def calculate_iou(pred: torch.Tensor, target: torch.Tensor, grid_size: int) -> float:
    """IoU trung binh tren cac cell co object (chi de theo doi train)."""
    obj_mask = target[..., 0] > 0.5
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
    pxy = torch.sigmoid(pred_box[:, 0:2])
    pwh = torch.sigmoid(pred_box[:, 2:4])
    tb = target[..., 1:5][obj_mask]

    px1, py1, px2, py2 = _xywh_to_xyxy_grid(pxy[:, 0], pxy[:, 1], pwh[:, 0], pwh[:, 1], gx_grid, gy_grid, G)
    tx1, ty1, tx2, ty2 = _xywh_to_xyxy_grid(tb[:, 0], tb[:, 1], tb[:, 2], tb[:, 3], gx_grid, gy_grid, G)

    iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = iw * ih
    union = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0) + (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0) - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return iou.mean().item()


def plot_history(train_losses, val_losses, train_ious, val_ious, model_name: str, out_dir: str = "checkpoints"):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_losses, label="Train Loss", marker="o")
    axes[0].plot(epochs, val_losses, label="Val Loss", marker="o")
    axes[0].set_title(f"{model_name} - Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, train_ious, label="Train IoU (obj cells)", marker="o")
    axes[1].plot(epochs, val_ious, label="Val IoU (obj cells)", marker="o")
    axes[1].set_title(f"{model_name} - IoU"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name}_history.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"--> Da luu bieu do tai {out_path}")