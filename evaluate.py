"""
evaluate.py - Danh gia detector tren TestLoader voi Single-Pass Traversal va Split Guard.
Dung cho Final-Test sau khi da khoa protocol_lock.json va vuot qua global_preflight_check().
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.config import (
    CLASS_NAMES,
    CONF_THRESHOLD,
    IMAGE_SIZE,
    LAMBDA_COORD,
    NMS_IOU_THRESHOLD,
    NUM_CLASSES,
    STRIDE,
    get_grid_size,
    get_processed_data_root,
)
from scripts.data.prepared_loader import get_test_dataloader
from scripts.models.factory import build_model
from scripts.src.decode import decode_predictions
from scripts.src.metrics import _iou_1_to_n, compute_voc_ap
from scripts.src.utils import CombinedLocalizationLoss, calculate_iou


def evaluate_model(
    model_name: str,
    checkpoint_path: str,
    data_root: str = "data/dataset_202601",
    image_size: int = IMAGE_SIZE,
    batch_size: int = 16,
    num_workers: int = 4,
    conf_threshold: float = CONF_THRESHOLD,
    nms_iou_threshold: float = NMS_IOU_THRESHOLD,
    match_iou_threshold: float = 0.5,
    min_score: float = 0.01,
    lock_token: Optional[str] = None,
    output_dir: str = "outputs",
) -> Dict:
    """
    Danh gia Test Set dung 1 pass duy nhat (1 traversal qua TestLoader).
    Bat buoc phai co lock_token tu global_preflight_check() de mo TestLoader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint khong ton tai: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    image_size = ckpt.get("image_size", image_size)
    grid_size = ckpt.get("grid_size", get_grid_size(image_size=image_size, stride=STRIDE))

    # Xay dung model voi pretrained=False de KHONG tai lai ImageNet
    model = build_model(
        model_name,
        num_classes=NUM_CLASSES,
        freeze_backbone=ckpt.get("freeze_backbone", False),
        unfreeze_from_layer=ckpt.get("unfreeze_from_layer", "layer3"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Split Semantics Guard: Bat buoc co lock_token hop le
    loader, ds = get_test_dataloader(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        lock_token=lock_token,
    )

    criterion = CombinedLocalizationLoss(lambda_coord=ckpt.get("hparams", {}).get("lambda_coord", LAMBDA_COORD))

    tot_loss = 0.0
    tot_iou = 0.0
    n_batches = 0
    preds_dict = {}
    gts_dict = {}
    idx = 0

    print(f"[evaluate] Bat dau Single-Pass Traversal qua TestLoader ({len(ds)} anh)...")
    # Traversal DUY NHAT 1 lan qua TestLoader
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            targets = batch[1].to(device)
            image_ids = batch[2] if len(batch) > 2 else None

            preds = model(imgs)
            tot_loss += criterion(preds, targets).item()
            tot_iou += calculate_iou(preds, targets, grid_size)
            n_batches += 1

            # Decode predictions de danh gia mAP
            dec = decode_predictions(preds, min_score=min_score, nms_iou_threshold=nms_iou_threshold)
            for i, d in enumerate(dec):
                img_id = image_ids[i] if (image_ids is not None and i < len(image_ids)) else f"img_{idx}"
                preds_dict[img_id] = {k: v.cpu().numpy() for k, v in d.items()}

                if hasattr(ds, "get_raw_boxes_by_id"):
                    try:
                        raw = ds.get_raw_boxes_by_id(img_id)
                    except Exception:
                        raw = ds.get_raw_boxes(idx) if idx < len(ds) else []
                elif hasattr(ds, "get_raw_boxes") and idx < len(ds):
                    raw = ds.get_raw_boxes(idx)
                else:
                    raw = []

                if raw:
                    a = np.array(raw, dtype=np.float32)
                    lab = a[:, 0].astype(np.int64)
                    cx, cy, w, h = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
                    bx = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
                else:
                    bx, lab = np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

                gts_dict[img_id] = (bx, lab)
                idx += 1

    # Tinh VOC continuous all-points mAP@0.5 va operating metrics (P/R/F1)
    ap = np.zeros(NUM_CLASSES)
    P = np.zeros(NUM_CLASSES)
    R = np.zeros(NUM_CLASSES)
    F1 = np.zeros(NUM_CLASSES)
    n_gt_c = np.zeros(NUM_CLASSES, dtype=int)
    TPs = FPs = FNs = 0

    for c in range(NUM_CLASSES):
        gt_by_img = {img_id: g[0][g[1] == c] for img_id, g in gts_dict.items() if (g[1] == c).any()}
        n_gt = sum(len(v) for v in gt_by_img.values())
        n_gt_c[c] = n_gt

        det_for_c = []
        for img_id, p_ in preds_dict.items():
            for b, s_, l in zip(p_["boxes"], p_["scores"], p_["labels"]):
                if int(l) == c:
                    det_for_c.append((float(s_), img_id, b))

        det_for_c.sort(key=lambda x: -x[0])
        used = {img_id: np.zeros(len(v), bool) for img_id, v in gt_by_img.items()}
        tp = np.zeros(len(det_for_c))

        for k, (_, img_id, b) in enumerate(det_for_c):
            if img_id not in gt_by_img:
                continue
            ious = _iou_1_to_n(b, gt_by_img[img_id])
            j = int(ious.argmax())
            if ious[j] >= match_iou_threshold and not used[img_id][j]:
                used[img_id][j] = True
                tp[k] = 1.0

        scores = np.array([d[0] for d in det_for_c])
        ctp, cfp = np.cumsum(tp), np.cumsum(1.0 - tp)

        if n_gt > 0 and len(det_for_c) > 0:
            rec = ctp / n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            ap[c] = compute_voc_ap(rec, prec)

        n_keep = int((scores >= conf_threshold).sum()) if len(det_for_c) else 0
        TP = int(ctp[n_keep - 1]) if n_keep > 0 else 0
        FP = n_keep - TP
        FN = n_gt - TP

        P[c] = TP / max(TP + FP, 1)
        R[c] = TP / max(TP + FN, 1)
        F1[c] = 2 * P[c] * R[c] / max(P[c] + R[c], 1e-8)
        TPs += TP
        FPs += FP
        FNs += FN

    valid = n_gt_c > 0
    mp = TPs / max(TPs + FPs, 1)
    mr = TPs / max(TPs + FNs, 1)
    total_preds = sum(len(p["boxes"]) for p in preds_dict.values())
    mean_preds = float(total_preds / max(len(preds_dict), 1))

    mean_loss = tot_loss / max(n_batches, 1)
    mean_iou = tot_iou / max(n_batches, 1)
    map50 = float(ap[valid].mean()) if valid.any() else 0.0

    det = {
        "loss": round(mean_loss, 4),
        "iou": round(mean_iou, 4),
        "map50": round(map50, 4),
        "macro_precision": round(float(P[valid].mean()) if valid.any() else 0.0, 4),
        "macro_recall": round(float(R[valid].mean()) if valid.any() else 0.0, 4),
        "macro_f1": round(float(F1[valid].mean()) if valid.any() else 0.0, 4),
        "micro_precision": round(float(mp), 4),
        "micro_recall": round(float(mr), 4),
        "micro_f1": round(float(2 * mp * mr / max(mp + mr, 1e-8)), 4),
        "mean_preds_per_image": round(mean_preds, 2),
        "ap_per_class": ap.tolist(),
        "precision_per_class": P.tolist(),
        "recall_per_class": R.tolist(),
        "f1_per_class": F1.tolist(),
        "n_gt_per_class": n_gt_c.tolist(),
    }

    print("=" * 75)
    print(f" {model_name.upper()} TEST SET (Post-Freeze Single-Pass) | {len(ds)} anh")
    print("=" * 75)
    print(f"  Loss: {det['loss']:.4f} | IoU: {det['iou']:.4f} | mAP@0.5: {det['map50']:.4f}")
    print(f"  Macro P/R/F1: {det['macro_precision']:.3f} / {det['macro_recall']:.3f} / {det['macro_f1']:.3f}")
    print(f"  Micro P/R/F1: {det['micro_precision']:.3f} / {det['micro_recall']:.3f} / {det['micro_f1']:.3f}")
    print(f"  Preds/anh: {det['mean_preds_per_image']:.2f} (eval_min_score={min_score})")

    os.makedirs(output_dir, exist_ok=True)
    stem = Path(checkpoint_path).stem
    out_json = os.path.join(output_dir, f"eval_{stem}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(det, f, indent=2)

    return det


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model1", choices=["model1", "model2", "model3"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="data/dataset_202601")
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD)
    p.add_argument("--nms_iou_threshold", type=float, default=NMS_IOU_THRESHOLD)
    p.add_argument("--match_iou_threshold", type=float, default=0.5)
    p.add_argument("--min_score", type=float, default=0.01)
    p.add_argument("--lock_token", type=str, default=None, help="Token tu global_preflight_check()")
    p.add_argument("--output_dir", default="outputs")
    a = p.parse_args()

    evaluate_model(
        model_name=a.model,
        checkpoint_path=a.checkpoint,
        data_root=a.data_root,
        image_size=a.image_size,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        conf_threshold=a.conf_threshold,
        nms_iou_threshold=a.nms_iou_threshold,
        match_iou_threshold=a.match_iou_threshold,
        min_score=a.min_score,
        lock_token=a.lock_token,
        output_dir=a.output_dir,
    )