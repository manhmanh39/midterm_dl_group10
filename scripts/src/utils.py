"""Detection loss + train diagnostics."""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.config import (
    FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTHING,
    LAMBDA_BOX, LAMBDA_CLASS, LAMBDA_OBJ,
)


def _ciou_loss_elementwise(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor, eps: float = 1e-7):
    """Complete-IoU loss for aligned box pairs, coordinates normalized 0..1."""
    px1, py1, px2, py2 = pred_xyxy.unbind(dim=1)
    tx1, ty1, tx2, ty2 = target_xyxy.unbind(dim=1)

    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)
    union = pw * ph + tw * th - inter
    iou = inter / union.clamp(min=eps)

    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    rho2 = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)

    ex1 = torch.minimum(px1, tx1)
    ey1 = torch.minimum(py1, ty1)
    ex2 = torch.maximum(px2, tx2)
    ey2 = torch.maximum(py2, ty2)
    c2 = (ex2 - ex1).pow(2) + (ey2 - ey1).pow(2) + eps

    v = (4.0 / math.pi**2) * (torch.atan(tw / th) - torch.atan(pw / ph)).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
    ciou = iou - rho2 / c2 - alpha * v
    return 1.0 - ciou


class CombinedLocalizationLoss(nn.Module):
    def __init__(
        self,
        lambda_box=LAMBDA_BOX,
        lambda_obj=LAMBDA_OBJ,
        lambda_class=LAMBDA_CLASS,
        focal_gamma=FOCAL_GAMMA,
        focal_alpha=FOCAL_ALPHA,
        class_weights: torch.Tensor | None = None,
        label_smoothing=LABEL_SMOOTHING,
        # backward-compatible name
        lambda_coord=None,
    ):
        super().__init__()
        if lambda_coord is not None:
            lambda_box = lambda_coord
        self.lambda_box = float(lambda_box)
        self.lambda_obj = float(lambda_obj)
        self.lambda_class = float(lambda_class)
        self.gamma = float(focal_gamma)
        self.alpha = float(focal_alpha)
        self.label_smoothing = float(label_smoothing)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.detach().float().clone())

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
        G = pred.shape[1]
        obj_mask = target[..., 0] > 0.5
        num_pos = obj_mask.sum().clamp(min=1).float()

        # Objectness focal BCE with separate positive/negative normalization.
        # The previous implementation divided ALL ~G*G*A slots by num_pos.
        # On a 32x32x3 grid this can make easy negatives dominate the gradient
        # by hundreds of times when an image has only a few lesions.
        logit = pred[..., 0]
        tgt = target[..., 0]
        prob = torch.sigmoid(logit)
        bce = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
        p_t = prob * tgt + (1.0 - prob) * (1.0 - tgt)
        alpha_t = self.alpha * tgt + (1.0 - self.alpha) * (1.0 - tgt)
        focal = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        pos_obj = focal[obj_mask].sum() / num_pos
        neg_mask = ~obj_mask
        num_neg = neg_mask.sum().clamp(min=1).float()
        neg_obj = focal[neg_mask].sum() / num_neg
        # Keep negative suppression important, but never let it swamp positives.
        obj_loss = pos_obj + 0.50 * neg_obj

        if not obj_mask.any():
            zero = pred.sum() * 0.0
            return self.lambda_obj * obj_loss + zero

        pos_idx = obj_mask.nonzero(as_tuple=False)  # [N, b, gy, gx, slot]
        pb = pred[..., 1:5][obj_mask]
        tb = target[..., 1:5][obj_mask]
        pxy = torch.sigmoid(pb[:, :2])
        pwh = torch.sigmoid(pb[:, 2:4]).clamp(min=1e-4, max=1.0)
        txy = tb[:, :2]
        twh = tb[:, 2:4].clamp(min=1e-4, max=1.0)

        gy = pos_idx[:, 1].float()
        gx = pos_idx[:, 2].float()
        pcx = (gx + pxy[:, 0]) / G
        pcy = (gy + pxy[:, 1]) / G
        tcx = (gx + txy[:, 0]) / G
        tcy = (gy + txy[:, 1]) / G

        pbox = torch.stack([
            pcx - pwh[:, 0] / 2, pcy - pwh[:, 1] / 2,
            pcx + pwh[:, 0] / 2, pcy + pwh[:, 1] / 2,
        ], dim=1)
        tbox = torch.stack([
            tcx - twh[:, 0] / 2, tcy - twh[:, 1] / 2,
            tcx + twh[:, 0] / 2, tcy + twh[:, 1] / 2,
        ], dim=1)
        box_loss = _ciou_loss_elementwise(pbox, tbox).mean()

        cls_target = target[..., 5:][obj_mask].argmax(dim=-1)
        cls_loss = F.cross_entropy(
            pred[..., 5:][obj_mask], cls_target,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * obj_loss
            + self.lambda_class * cls_loss
        )


@torch.no_grad()
def calculate_iou(pred: torch.Tensor, target: torch.Tensor, grid_size: int | None = None) -> float:
    """Mean IoU on positive target slots; only train diagnostic."""
    del grid_size
    obj_mask = target[..., 0] > 0.5
    if not obj_mask.any():
        return 0.0
    G = pred.shape[1]
    pos_idx = obj_mask.nonzero(as_tuple=False)
    pb = pred[..., 1:5][obj_mask]
    tb = target[..., 1:5][obj_mask]
    pxy, pwh = torch.sigmoid(pb[:, :2]), torch.sigmoid(pb[:, 2:4])
    txy, twh = tb[:, :2], tb[:, 2:4]
    gy = pos_idx[:, 1].float()
    gx = pos_idx[:, 2].float()

    pcx, pcy = (gx + pxy[:, 0]) / G, (gy + pxy[:, 1]) / G
    tcx, tcy = (gx + txy[:, 0]) / G, (gy + txy[:, 1]) / G
    pbox = torch.stack([pcx-pwh[:,0]/2, pcy-pwh[:,1]/2, pcx+pwh[:,0]/2, pcy+pwh[:,1]/2], 1)
    tbox = torch.stack([tcx-twh[:,0]/2, tcy-twh[:,1]/2, tcx+twh[:,0]/2, tcy+twh[:,1]/2], 1)

    ix1 = torch.maximum(pbox[:, 0], tbox[:, 0]); iy1 = torch.maximum(pbox[:, 1], tbox[:, 1])
    ix2 = torch.minimum(pbox[:, 2], tbox[:, 2]); iy2 = torch.minimum(pbox[:, 3], tbox[:, 3])
    inter = (ix2-ix1).clamp(min=0) * (iy2-iy1).clamp(min=0)
    pa = (pbox[:,2]-pbox[:,0]).clamp(min=0) * (pbox[:,3]-pbox[:,1]).clamp(min=0)
    ta = (tbox[:,2]-tbox[:,0]).clamp(min=0) * (tbox[:,3]-tbox[:,1]).clamp(min=0)
    return (inter / (pa + ta - inter).clamp(min=1e-8)).mean().item()


def plot_history(train_losses, val_losses, train_ious, val_ious, model_name: str, out_dir: str = "checkpoints"):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_losses, label="Train Loss", marker="o")
    axes[0].plot(epochs, val_losses, label="Val Loss", marker="o")
    axes[0].set_title(f"{model_name} - Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, train_ious, label="Train IoU", marker="o")
    axes[1].plot(epochs, val_ious, label="Val IoU", marker="o")
    axes[1].set_title(f"{model_name} - IoU"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name}_history.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"--> Da luu bieu do tai {out_path}")
