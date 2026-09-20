"""
main.py - Dieu phoi toan bo pipeline OBJECT DETECTION VinBigData Chest X-ray.

Chay:
    python main.py --model all --mode all --epochs 30 --batch_size 16
    python main.py --model model1 --mode train --epochs 30
    python main.py --model model3 --mode eval --checkpoint checkpoints/model3_best.pth
    python main.py --model model3 --mode all --no_freeze_backbone --unfreeze_from_layer layer4

Ghi chu: buoc chuan bi du lieu (DICOM -> PNG 3-kenh + nhan YOLO) KHONG nam
trong main.py nay - phai chay rieng TRUOC (mot lan duy nhat, ton nhieu thoi
gian xu ly anh):
    python -m scripts.data.prepare_dataset --data_root data/raw --output_root data/processed/prepared_dataset

main.py chi kiem tra dataset da prepared hop le hay chua (dung
scripts/data/prepared_loader.py) roi moi chay train/eval, tranh truong hop
train that bai giua chung vi thieu du lieu.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

# evaluate.py (trong scripts/) dung import bare (from config..., from src...),
# nen them scripts/ vao sys.path truoc khi import.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from scripts.config import IMAGE_SIZE, get_processed_data_root
from scripts.data.prepared_loader import is_prepared_dataset, describe_dataset
from train import train
from evaluate import evaluate_model  # noqa: E402  (lay tu scripts/evaluate.py nho sys.path o tren)


def run_pipeline(
    model_names: List[str],
    epochs: int,
    batch_size: int,
    lr: float,
    mode: str = "all",
    data_root: str = None,
    image_size: int = IMAGE_SIZE,
    freeze_backbone: bool = True,
    unfreeze_from_layer: str = "layer4",
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "outputs",
    checkpoint_path: str = None,
):
    data_root = data_root or get_processed_data_root()
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    print("*" * 80)
    print("   DU AN PHAT HIEN BAT THUONG X-QUANG PHOI (VINBIGDATA CHEST X-RAY)   ")
    print("   Bai toan: OBJECT DETECTION (anchor-free, 1-scale kieu YOLO)        ")
    print("*" * 80)
    print(f"Model: {model_names} | Mode: {mode.upper()} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"Data root: {data_root}")
    print("*" * 80)

    print("\n>>> BUOC 1: KIEM TRA DATASET DA PREPARED...")
    if not is_prepared_dataset(data_root):
        print(
            f"[LOI] '{data_root}' khong phai dataset hop le "
            f"(thieu dataset.yaml hoac thieu anh .png trong train/val/test).\n"
            f"Hay chay truoc:\n"
            f"    python -m scripts.data.prepare_dataset --data_root <duong_dan_dicom> --output_root {data_root}"
        )
        sys.exit(1)

    stats = describe_dataset(data_root)
    for split, s in stats.items():
        print(
            f"  [{split:<5}] {s['n_images']:>5} anh | {s['n_boxes']:>6} box | "
            f"No finding: {s['n_no_finding']:>5} ({s['pct_no_finding']:.1f}%) | "
            f"TB box/anh: {s['avg_boxes_per_image']:.2f}"
        )
    print("[OK] Dataset san sang.\n")

    summary_results = []

    for model_name in model_names:
        print("\n" + "#" * 80)
        print(f"### TIEN TRINH CHO MODEL: {model_name.upper()} ###")
        print("#" * 80)

        model_ckpt_dir = checkpoint_dir

        if mode in ("train", "all"):
            print(f"\n>>> BUOC 2: HUAN LUYEN [{model_name}]...")
            train(
                model_name=model_name,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                train_dir=train_dir,
                val_dir=val_dir,
                image_size=image_size,
                freeze_backbone=freeze_backbone,
                unfreeze_from_layer=unfreeze_from_layer,
                checkpoint_dir=model_ckpt_dir,
            )

        if mode in ("eval", "all"):
            print(f"\n>>> BUOC 3: DANH GIA [{model_name}]...")
            ckpt_path = checkpoint_path or os.path.join(model_ckpt_dir, f"{model_name}_best.pth")
            eval_metrics = evaluate_model(
                model_name=model_name,
                checkpoint_path=os.path.abspath(ckpt_path),
                test_dir=os.path.abspath(test_dir),
                batch_size=batch_size,
                output_dir=os.path.abspath(output_dir),
            )
            summary_results.append({"model": model_name, **eval_metrics})

    if summary_results and len(summary_results) > 1:
        print("\n" + "=" * 80)
        print("BANG TONG KET SO SANH")
        print("=" * 80)
        header = f"{'Model':<10} {'Loss':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}"
        print(header)
        print("-" * len(header))
        for res in summary_results:
            print(
                f"{res['model']:<10} {res['loss']:>8.4f} {res['iou']:>8.4f} "
                f"{res['macro_precision']:>10.4f} {res['macro_recall']:>8.4f} {res['macro_f1']:>8.4f}"
            )
        print("=" * 80)

        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "model_comparison_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        print(f"\n[main] Da luu bang tong ket tai: {summary_path}")

    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chay pipeline object detection VinBigData")
    parser.add_argument("--model", type=str, default="all", choices=["model1", "model2", "model3", "all"])
    parser.add_argument("--mode", type=str, default="all", choices=["all", "train", "eval"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_root", type=str, default=None,
                         help="Thu muc dataset da prepared (mac dinh: scripts.config.get_processed_data_root())")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--freeze_backbone", dest="freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--unfreeze_from_layer", type=str, default="layer4",
                         choices=["conv1", "layer1", "layer2", "layer3", "layer4"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Chi dung khi --mode eval va muon chi dinh checkpoint cu the (chi ap dung cho 1 model)")

    args = parser.parse_args()
    selected_models = ["model1", "model2", "model3"] if args.model == "all" else [args.model]

    run_pipeline(
        model_names=selected_models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mode=args.mode,
        data_root=args.data_root,
        image_size=args.image_size,
        freeze_backbone=args.freeze_backbone,
        unfreeze_from_layer=args.unfreeze_from_layer,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
    )