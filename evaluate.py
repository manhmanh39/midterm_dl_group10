"""Evaluate detector on FULL raw GT: loss/IoU + mAP@0.5 + P/R/F1."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.config import (
    BOXES_PER_CELL, CLASS_NAMES, CONF_THRESHOLD, IMAGE_SIZE, LAMBDA_BOX,
    LAMBDA_CLASS, LAMBDA_OBJ, MAX_DET, MIN_AP_SCORE, NMS_IOU_THRESHOLD,
    NUM_CLASSES, get_grid_size,
)
from scripts.models.factory import build_model
from scripts.src.dataset import VinBigDataDetectionDataset, collate_fn
from scripts.src.metrics import evaluate_detections
from scripts.src.utils import CombinedLocalizationLoss, calculate_iou


def evaluate_model(
    model_name,
    checkpoint_path,
    data_dir="data/processed/dataset_202601/test",
    image_size=IMAGE_SIZE,
    batch_size=16,
    num_workers=20,
    conf_threshold=CONF_THRESHOLD,
    nms_iou_threshold=NMS_IOU_THRESHOLD,
    match_iou_threshold=0.5,
    output_dir="outputs",
    max_det=MAX_DET,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    image_size = ckpt.get("image_size", image_size)
    grid_size = ckpt.get("grid_size", get_grid_size(image_size=image_size))
    boxes_per_cell = ckpt.get("boxes_per_cell", BOXES_PER_CELL)
    if ckpt.get("model_name", model_name) != model_name:
        raise ValueError(
            f"Checkpoint la {ckpt.get('model_name')}, nhung --model={model_name}. "
            "Hay eval dung model architecture da train."
        )

    model = build_model(
        model_name, NUM_CLASSES,
        freeze_backbone=ckpt.get("freeze_backbone", False),
        unfreeze_from_layer="layer3",
        boxes_per_cell=boxes_per_cell,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    ds = VinBigDataDetectionDataset(
        data_dir, image_size, grid_size, boxes_per_cell, augment=False
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    hp = ckpt.get("hparams", {})
    cw = ckpt.get("class_weights")
    class_weights = torch.tensor(cw, dtype=torch.float32) if cw is not None else None
    criterion = CombinedLocalizationLoss(
        lambda_box=hp.get("lambda_box", hp.get("lambda_coord", LAMBDA_BOX)),
        lambda_obj=hp.get("lambda_obj", LAMBDA_OBJ),
        lambda_class=hp.get("lambda_class", LAMBDA_CLASS),
        focal_gamma=hp.get("focal_gamma", 2.0),
        class_weights=class_weights,
    ).to(device)

    tot_loss = tot_iou = 0.0
    n = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model(imgs)
            tot_loss += criterion(preds, targets).item()
            tot_iou += calculate_iou(preds, targets)
            n += 1

    det = evaluate_detections(
        model, ds, loader, device, NUM_CLASSES,
        conf_threshold=conf_threshold,
        nms_iou=nms_iou_threshold,
        iou_thr=match_iou_threshold,
        min_score=MIN_AP_SCORE,
        max_det=max_det,
    )

    print("=" * 78)
    print(
        f" {model_name.upper()} | {data_dir} ({len(ds)} images) | "
        f"grid={grid_size} slots={boxes_per_cell} | IoU match={match_iou_threshold}"
    )
    print("=" * 78)
    print(f"  Loss {tot_loss/max(n,1):.4f} | IoU(positive slots) {tot_iou/max(n,1):.4f}")
    print(f"  mAP@0.5 {det['map50']:.4f}")
    print(f"  Macro P/R/F1: {det['macro_precision']:.3f} / {det['macro_recall']:.3f} / {det['macro_f1']:.3f}")
    print(f"  Micro P/R/F1: {det['micro_precision']:.3f} / {det['micro_recall']:.3f} / {det['micro_f1']:.3f}")
    print(f"  Preds/image: {det['mean_preds_per_image']:.2f} @ conf>={conf_threshold}")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {name:<20} AP={det['ap_per_class'][i]:.3f} "
            f"P={det['precision_per_class'][i]:.3f} R={det['recall_per_class'][i]:.3f} "
            f"F1={det['f1_per_class'][i]:.3f} (GT={det['n_gt_per_class'][i]})"
        )

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(NUM_CLASSES), det["ap_per_class"])
    plt.xticks(np.arange(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    plt.ylabel("AP@0.5")
    plt.title(f"Per-class AP - {model_name} (mAP={det['map50']:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"per_class_ap_{model_name}.png"), dpi=200)
    plt.close()

    result = {"loss": tot_loss/max(n,1), "iou": tot_iou/max(n,1), **det}
    with open(os.path.join(output_dir, f"eval_{model_name}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["model1", "model2", "model3"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default="data/processed/dataset_202601/test")
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=20)
    p.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD)
    p.add_argument("--nms_iou_threshold", type=float, default=NMS_IOU_THRESHOLD)
    p.add_argument("--match_iou_threshold", type=float, default=0.5)
    p.add_argument("--max_det", type=int, default=MAX_DET)
    p.add_argument("--output_dir", default="outputs")
    a = p.parse_args()
    evaluate_model(
        a.model, a.checkpoint, a.data_dir, a.image_size, a.batch_size,
        a.num_workers, a.conf_threshold, a.nms_iou_threshold,
        a.match_iou_threshold, a.output_dir, a.max_det,
    )
