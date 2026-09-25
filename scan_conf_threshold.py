"""
scan_conf_threshold.py - Quet conf_threshold tren tap VAL de tim nguong
cho Macro F1 cao nhat, roi dung nguong do de bao cao ket qua tren test.

Chay:
    python scan_conf_threshold.py --model model1 \
        --checkpoint checkpoints/model1_seed202601_best.pth \
        --data_dir data/processed/dataset_202601/val
"""

from __future__ import annotations

import argparse
import numpy as np

from evaluate import evaluate_model
from scripts.config import IMAGE_SIZE, NMS_IOU_THRESHOLD


def scan(
    model_name: str,
    checkpoint_path: str,
    data_dir: str,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 16,
    num_workers: int = 8,
    nms_iou_threshold: float = NMS_IOU_THRESHOLD,
    match_iou_threshold: float = 0.5,
    output_dir: str = "outputs",
    thresholds=None,
):
    if thresholds is None:
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    results = []
    print(f"[scan] Quet {len(thresholds)} nguong conf_threshold cho model '{model_name}'...")
    for thr in thresholds:
        print(f"\n--- conf_threshold = {thr} ---")
        metrics = evaluate_model(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            conf_threshold=thr,
            nms_iou_threshold=nms_iou_threshold,
            match_iou_threshold=match_iou_threshold,
            output_dir=output_dir,
        )
        results.append((thr, metrics["macro_f1"], metrics["macro_precision"], metrics["macro_recall"]))

    print("\n" + "=" * 70)
    print(f"BANG TONG KET QUET NGUONG - {model_name.upper()}")
    print("=" * 70)
    print(f"{'conf_thr':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    for thr, f1, p, r in results:
        print(f"{thr:>10.2f} {f1:>8.4f} {p:>10.4f} {r:>8.4f}")

    best_thr, best_f1, _, _ = max(results, key=lambda x: x[1])
    print("-" * 70)
    print(f"[scan] Nguong tot nhat theo Macro F1: conf_threshold={best_thr} (F1={best_f1:.4f})")
    print("=" * 70)

    return best_thr, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quet conf_threshold toi uu tren tap val")
    parser.add_argument("--model", type=str, required=True, choices=["model1", "model2", "model3"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="Nen dung tap VAL, khong dung test")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--nms_iou_threshold", type=float, default=NMS_IOU_THRESHOLD)
    parser.add_argument("--match_iou_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()
    scan(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        nms_iou_threshold=args.nms_iou_threshold,
        match_iou_threshold=args.match_iou_threshold,
        output_dir=args.output_dir,
    )