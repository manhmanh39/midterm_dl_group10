"""main.py - Dieu phoi pipeline detection VinBigData (train + eval).

Chuan bi du lieu chay rieng TRUOC:
    python -m scripts.data.prepare_dataset --zip_path <zip> --data_root data/raw --output_root data/processed/dataset_202601

    python main.py --model all --mode all --epochs 40
    python main.py --model model1 --mode train --epochs 15
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from scripts.config import IMAGE_SIZE, get_processed_data_root
from scripts.data.prepared_loader import is_prepared_dataset, describe_dataset
from train import train_pipeline
from evaluate import evaluate_model


def run_pipeline(
    model_names: List[str], epochs: int, batch_size=None, lr=None, mode: str = "all",
    data_root: str = None, image_size: int = IMAGE_SIZE, freeze_backbone: bool = False,
    unfreeze_from_layer: str = "layer3", checkpoint_dir: str = "checkpoints",
    output_dir: str = "outputs", checkpoint_path: str = None, seed: int = 202601, num_workers: int = 8,
):
    data_root = data_root or get_processed_data_root()
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    print("*" * 80)
    print(f"Model: {model_names} | Mode: {mode.upper()} | Epochs: {epochs} | Seed: {seed}")
    print(f"Data root: {data_root}")
    print("*" * 80)

    if not is_prepared_dataset(data_root):
        print(f"[LOI] '{data_root}' khong hop le. Hay chay truoc:\n"
              f"    python -m scripts.data.prepare_dataset --output_root {data_root}")
        sys.exit(1)

    for split, s in describe_dataset(data_root).items():
        print(f"  [{split:<5}] {s['n_images']:>5} anh | {s['n_boxes']:>6} box | "
              f"No finding: {s['n_no_finding']:>5} ({s['pct_no_finding']:.1f}%) | "
              f"TB box/anh: {s['avg_boxes_per_image']:.2f}")

    summary_results = []
    for model_name in model_names:
        print("\n" + "#" * 80 + f"\n### MODEL: {model_name.upper()} ###\n" + "#" * 80)

        if mode in ("train", "all"):
            train_pipeline(
                model_name=model_name, seed=seed, epochs=epochs, batch_size=batch_size, lr=lr,
                train_dir=train_dir, val_dir=val_dir, image_size=image_size,
                freeze_backbone=freeze_backbone, unfreeze_from_layer=unfreeze_from_layer,
                checkpoint_dir=checkpoint_dir, num_workers=num_workers,
            )

        if mode in ("eval", "all"):
            ckpt_path = checkpoint_path or os.path.join(checkpoint_dir, f"{model_name}_seed{seed}_best.pth")
            m = evaluate_model(
                model_name=model_name, checkpoint_path=os.path.abspath(ckpt_path),
                data_dir=os.path.abspath(test_dir), batch_size=batch_size or 16,
                num_workers=num_workers, output_dir=os.path.abspath(output_dir),
            )
            summary_results.append({"model": model_name, **{k: v for k, v in m.items() if not isinstance(v, list)}})

    if len(summary_results) > 1:
        header = f"{'Model':<10} {'Loss':>8} {'IoU':>8} {'mAP50':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}"
        print("\n" + "=" * len(header) + "\n" + header + "\n" + "-" * len(header))
        for r in summary_results:
            print(f"{r['model']:<10} {r['loss']:>8.4f} {r['iou']:>8.4f} {r['map50']:>8.4f} "
                  f"{r['macro_precision']:>8.4f} {r['macro_recall']:>8.4f} {r['macro_f1']:>8.4f}")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "model_comparison_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        print(f"\n[main] Da luu: {path}")
    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["model1", "model2", "model3", "all"])
    parser.add_argument("--mode", type=str, default="all", choices=["all", "train", "eval"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=202601)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze_from_layer", type=str, default="layer3",
                        choices=["conv1", "layer1", "layer2", "layer3", "layer4"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Chi dung voi --mode eval cho 1 model")
    args = parser.parse_args()

    models = ["model1", "model2", "model3"] if args.model == "all" else [args.model]
    run_pipeline(
        model_names=models, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, mode=args.mode,
        data_root=args.data_root, image_size=args.image_size, freeze_backbone=args.freeze_backbone,
        unfreeze_from_layer=args.unfreeze_from_layer, checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir, checkpoint_path=args.checkpoint, seed=args.seed,
        num_workers=args.num_workers,
    )