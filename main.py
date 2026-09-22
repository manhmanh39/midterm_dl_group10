"""
main.py - Dieu phoi pipeline detection VinBigData P0/P1.
Tach biet ro rang 2 giai doan:
  1. Giai doan DEVELOP (--phase develop): Chi su dung Train/Val loaders, tim model tot nhat.
  2. Giai doan FINAL-TEST (--phase final-test): Preflight check 9 checkpoints, mo TestLoader 1 pass duy nhat.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from scripts.config import IMAGE_SIZE, get_processed_data_root
from scripts.data.prepared_loader import describe_dataset, is_prepared_dataset
from scripts.experiment_config import (
    CANONICAL_MODELS,
    CANONICAL_SEEDS,
    generate_protocol_lock,
    global_preflight_check,
)
from evaluate import evaluate_model
from train import train_pipeline


def run_pipeline(
    model_names: List[str],
    phase: str = "develop",
    epochs: int = 40,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    data_root: Optional[str] = None,
    image_size: int = IMAGE_SIZE,
    freeze_backbone: bool = False,
    unfreeze_from_layer: str = "layer3",
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "outputs",
    checkpoint_path: Optional[str] = None,
    seed: int = 202601,
    num_workers: int = 4,
    lock_token: Optional[str] = None,
):
    data_root = data_root or "data/dataset_202601"

    print("*" * 80)
    print(f"VinBigData Detection Protocol | Phase: {phase.upper()} | Models: {model_names} | Seed: {seed}")
    print(f"Data Root: {data_root} | Epochs: {epochs}")
    print("*" * 80)

    if not is_prepared_dataset(data_root):
        print(f"[LOI] '{data_root}' khong hop le. Can dataset.yaml va 3 splits (train/val/test/images).")
        sys.exit(1)

    for split, s in describe_dataset(data_root).items():
        print(f"  [{split:<5}] {s['n_images']:>5} anh | {s['n_boxes']:>6} box | "
              f"No finding: {s['n_no_finding']:>5} ({s['pct_no_finding']:.1f}%) | "
              f"TB box/anh: {s['avg_boxes_per_image']:.2f}")

    if phase == "develop":
        print("\n[PHASE: DEVELOP] Huan luyen tren Train + Val. TestLoader bi khoa tuyet doi.")
        develop_results = []
        for model_name in model_names:
            print(f"\n>>> Huan luyen {model_name.upper()} (Seed: {seed}) <<<")
            res = train_pipeline(
                model_name=model_name,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                data_root=data_root,
                image_size=image_size,
                freeze_backbone=freeze_backbone,
                unfreeze_from_layer=unfreeze_from_layer,
                checkpoint_dir=checkpoint_dir,
                num_workers=num_workers,
            )
            develop_results.append({"model": model_name, "seed": seed, **res})
        return develop_results

    elif phase == "final-test":
        print("\n[PHASE: FINAL-TEST] Chay danh gia Post-Freeze tren TestLoader.")
        lock_file = os.path.join(output_dir, "protocol_lock.json")

        if not lock_token:
            print("[final-test] Chua co lock_token. Dang thuc hien global_preflight_check()...")
            _, lock_token = global_preflight_check(lock_path=lock_file, data_root=data_root)

        test_results = []
        for model_name in model_names:
            ckpt = checkpoint_path or os.path.join(checkpoint_dir, f"{model_name}_seed{seed}_best.pth")
            print(f"\n>>> Danh gia Final-Test cho {model_name.upper()} tu {ckpt} <<<")
            m = evaluate_model(
                model_name=model_name,
                checkpoint_path=ckpt,
                data_root=data_root,
                image_size=image_size,
                batch_size=batch_size or 16,
                num_workers=num_workers,
                lock_token=lock_token,
                output_dir=output_dir,
            )
            test_results.append({
                "model": model_name,
                "seed": seed,
                **{k: v for k, v in m.items() if not isinstance(v, list)}
            })

        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, f"test_summary_{phase}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)
        print(f"[final-test] Da luu ket qua: {summary_path}")
        return test_results

    else:
        raise ValueError(f"Phase '{phase}' khong hop le. Hay chon 'develop' hoac 'final-test'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VinBigData Detection P0/P1 Main Orchestrator")
    parser.add_argument("--model", type=str, default="all", choices=["model1", "model2", "model3", "all"])
    parser.add_argument("--phase", type=str, default="develop", choices=["develop", "final-test"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=202601)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_root", type=str, default="data/dataset_202601")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze_from_layer", type=str, default="layer3",
                        choices=["conv1", "layer1", "layer2", "layer3", "layer4"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Chi dinh checkpoint cho final-test")
    parser.add_argument("--lock_token", type=str, default=None, help="Token tu preflight check")
    args = parser.parse_args()

    models = CANONICAL_MODELS if args.model == "all" else [args.model]
    run_pipeline(
        model_names=models,
        phase=args.phase,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_root=args.data_root,
        image_size=args.image_size,
        freeze_backbone=args.freeze_backbone,
        unfreeze_from_layer=args.unfreeze_from_layer,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        num_workers=args.num_workers,
        lock_token=args.lock_token,
    )