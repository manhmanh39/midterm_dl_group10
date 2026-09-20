"""
decode.py - chuyen prediction map (B, G, G, 5+num_classes) thanh danh sach
box thuc su (x1,y1,x2,y2,conf,class_id) cho tung anh, kem Non-Max Suppression
(NMS) de loai bo cac box trung lap qua muc - buoc BAT BUOC voi detection
anchor-free vi nhieu cell lien ke co the cung du bao cho 1 object.
"""

from __future__ import annotations

import torch
import torchvision


def decode_predictions(
    pred: torch.Tensor,
    conf_threshold: float = 0.5,
    nms_iou_threshold: float = 0.45,
):
    """
    Parameters
    ----------
    pred : (B, G, G, 5 + num_classes) RAW LOGITS (chua qua sigmoid), dung
           output truc tiep tu model.
    conf_threshold : nguong objectness (sau sigmoid) de giu box
    nms_iou_threshold : nguong IoU cho NMS

    Returns
    -------
    list co B phan tu, moi phan tu la dict:
        {
          "boxes": Tensor (N, 4) - (x1,y1,x2,y2) normalized [0,1],
          "scores": Tensor (N,) - objectness * class_confidence,
          "labels": Tensor (N,) - class_id
        }
    """
    B, G, _, C = pred.shape
    num_classes = C - 5
    device = pred.device

    obj_conf = torch.sigmoid(pred[..., 0])              # (B, G, G)
    tx_ty = torch.sigmoid(pred[..., 1:3])                # (B, G, G, 2)
    tw_th = torch.sigmoid(pred[..., 3:5])                # (B, G, G, 2)
    class_probs = torch.sigmoid(pred[..., 5:])           # (B, G, G, num_classes)

    gy_grid, gx_grid = torch.meshgrid(
        torch.arange(G, device=device), torch.arange(G, device=device), indexing="ij"
    )
    gy_grid = gy_grid.unsqueeze(0).expand(B, -1, -1).float()  # (B, G, G)
    gx_grid = gx_grid.unsqueeze(0).expand(B, -1, -1).float()

    cx = (gx_grid + tx_ty[..., 0]) / G
    cy = (gy_grid + tx_ty[..., 1]) / G
    w = tw_th[..., 0]
    h = tw_th[..., 1]

    x1 = (cx - w / 2).clamp(0, 1)
    y1 = (cy - h / 2).clamp(0, 1)
    x2 = (cx + w / 2).clamp(0, 1)
    y2 = (cy + h / 2).clamp(0, 1)

    class_conf, class_id = class_probs.max(dim=-1)   # (B, G, G) moi
    final_score = obj_conf * class_conf              # (B, G, G)

    results = []
    for b in range(B):
        mask = final_score[b] > conf_threshold
        if mask.sum() == 0:
            results.append({
                "boxes": torch.zeros((0, 4), device=device),
                "scores": torch.zeros((0,), device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
            })
            continue

        boxes_b = torch.stack([x1[b][mask], y1[b][mask], x2[b][mask], y2[b][mask]], dim=1)
        scores_b = final_score[b][mask]
        labels_b = class_id[b][mask]

        # NMS phai chay RIENG cho tung class (khong loai bo box cua 2 class
        # khac nhau chi vi chong lan vi tri - vd Nodule/Mass va Lung Opacity
        # co the cung nam trong 1 vung)
        keep_indices = []
        for c in labels_b.unique():
            cls_mask = labels_b == c
            idxs = torch.nonzero(cls_mask, as_tuple=True)[0]
            keep = torchvision.ops.nms(boxes_b[idxs], scores_b[idxs], nms_iou_threshold)
            keep_indices.append(idxs[keep])

        if keep_indices:
            keep_indices = torch.cat(keep_indices)
        else:
            keep_indices = torch.zeros((0,), dtype=torch.long, device=device)

        results.append({
            "boxes": boxes_b[keep_indices],
            "scores": scores_b[keep_indices],
            "labels": labels_b[keep_indices],
        })

    return results