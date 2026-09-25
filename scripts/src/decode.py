"""Prediction map (B,G,G,A,5+C) -> boxes + class-aware NMS."""
from __future__ import annotations

import torch
import torchvision

from scripts.config import MAX_DET


def decode_predictions(pred, conf_threshold=0.01, nms_iou_threshold=0.45, max_det=MAX_DET):
    if pred.ndim != 5:
        raise ValueError(f"Expected pred (B,G,G,A,D), got {tuple(pred.shape)}")
    B, G, G2, A, D = pred.shape
    if G != G2 or D < 6:
        raise ValueError(f"Invalid detection output shape: {tuple(pred.shape)}")
    device = pred.device

    obj = torch.sigmoid(pred[..., 0])
    xy = torch.sigmoid(pred[..., 1:3])
    wh = torch.sigmoid(pred[..., 3:5]).clamp(min=1e-4, max=1.0)
    cls_prob = torch.softmax(pred[..., 5:], dim=-1)

    gy, gx = torch.meshgrid(
        torch.arange(G, device=device), torch.arange(G, device=device), indexing="ij"
    )
    gx = gx.view(1, G, G, 1).float()
    gy = gy.view(1, G, G, 1).float()
    cx = (gx + xy[..., 0]) / G
    cy = (gy + xy[..., 1]) / G
    boxes = torch.stack([
        cx - wh[..., 0] / 2, cy - wh[..., 1] / 2,
        cx + wh[..., 0] / 2, cy + wh[..., 1] / 2,
    ], dim=-1).clamp(0, 1)

    cls_conf, cls_id = cls_prob.max(dim=-1)
    score = obj * cls_conf

    results = []
    for b in range(B):
        m = score[b] >= conf_threshold
        bx = boxes[b][m]
        sc = score[b][m]
        lb = cls_id[b][m]
        if sc.numel() == 0:
            results.append({
                "boxes": torch.zeros((0, 4), device=device),
                "scores": torch.zeros((0,), device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
            })
            continue
        keep = torchvision.ops.batched_nms(bx, sc, lb, nms_iou_threshold)[:max_det]
        results.append({"boxes": bx[keep], "scores": sc[keep], "labels": lb[keep]})
    return results
