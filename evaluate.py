"""evaluate.py - Danh gia detector: Loss/IoU + mAP@0.5 + P/R/F1 (GT doc tu file label).

    python evaluate.py --model model1 --checkpoint checkpoints/model1_seed202601_best.pth
"""
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

from scripts.config import (NUM_CLASSES, CLASS_NAMES, IMAGE_SIZE, get_grid_size,
                            CONF_THRESHOLD, NMS_IOU_THRESHOLD, LAMBDA_COORD)
from scripts.src.dataset import VinBigDataDetectionDataset, collate_fn
from scripts.src.utils import CombinedLocalizationLoss, calculate_iou
from scripts.src.metrics import evaluate_detections
from scripts.models.factory import build_model


def evaluate_model(model_name, checkpoint_path, data_dir="data/processed/dataset_202601/test",
                   image_size=IMAGE_SIZE, batch_size=16, num_workers=8,
                   conf_threshold=CONF_THRESHOLD, nms_iou_threshold=NMS_IOU_THRESHOLD,
                   match_iou_threshold=0.5, output_dir="outputs"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    image_size = ckpt.get("image_size", image_size)
    grid_size = ckpt.get("grid_size", get_grid_size(image_size=image_size))
    model = build_model(model_name, NUM_CLASSES, ckpt.get("freeze_backbone", False),
                        ckpt.get("unfreeze_from_layer", "layer3")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = VinBigDataDetectionDataset(data_dir, image_size, grid_size, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    criterion = CombinedLocalizationLoss(lambda_coord=ckpt.get("hparams", {}).get("lambda_coord", LAMBDA_COORD))

    tot_loss = tot_iou = 0.0
    n = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            tot_loss += criterion(preds, targets).item()
            tot_iou += calculate_iou(preds, targets, grid_size)
            n += 1

    det = evaluate_detections(model, ds, loader, device, NUM_CLASSES, conf_threshold,
                              nms_iou_threshold, match_iou_threshold, min_score=0.01)

    print("=" * 70)
    print(f" {model_name.upper()} | {data_dir} ({len(ds)} anh) | IoU match = {match_iou_threshold}")
    print("=" * 70)
    print(f"  Loss {tot_loss / n:.4f} | IoU(obj cells) {tot_iou / n:.4f}")
    print(f"  mAP@0.5 {det['map50']:.4f}")
    print(f"  Macro P/R/F1: {det['macro_precision']:.3f} / {det['macro_recall']:.3f} / {det['macro_f1']:.3f}")
    print(f"  Micro P/R/F1: {det['micro_precision']:.3f} / {det['micro_recall']:.3f} / {det['micro_f1']:.3f}")
    print(f"  Preds/anh: {det['mean_preds_per_image']:.2f}  (conf>={conf_threshold})")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<20} AP={det['ap_per_class'][i]:.3f} P={det['precision_per_class'][i]:.3f} "
              f"R={det['recall_per_class'][i]:.3f} F1={det['f1_per_class'][i]:.3f} (GT={det['n_gt_per_class'][i]})")

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(NUM_CLASSES), det["ap_per_class"])
    plt.xticks(np.arange(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    plt.ylabel("AP@0.5"); plt.title(f"Per-class AP - {model_name} (mAP={det['map50']:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"per_class_ap_{model_name}.png"), dpi=200)
    plt.close()

    result = {"loss": tot_loss / n, "iou": tot_iou / n, **det}
    with open(os.path.join(output_dir, f"eval_{model_name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model1", choices=["model1", "model2", "model3"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default="data/processed/dataset_202601/test")
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD)
    p.add_argument("--nms_iou_threshold", type=float, default=NMS_IOU_THRESHOLD)
    p.add_argument("--match_iou_threshold", type=float, default=0.5)
    p.add_argument("--output_dir", default="outputs")
    a = p.parse_args()
    evaluate_model(a.model, a.checkpoint, a.data_dir, a.image_size, a.batch_size, a.num_workers,
                   a.conf_threshold, a.nms_iou_threshold, a.match_iou_threshold, a.output_dir)