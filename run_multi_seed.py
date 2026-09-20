"""
run_multi_seed.py - Chay training nhieu seed khac nhau de kiem tra do on dinh
(reproducibility) cua model OBJECT DETECTION.

Voi moi seed: train model tu dau, roi danh gia tren tap test bang
scripts/evaluate.py::evaluate_model (Precision/Recall/F1/IoU). Sau khi chay
het cac seed, tinh mean/std cho tung metric va luu bao cao JSON.

Chay:
    python run_multi_seed.py --model model1 --n_runs 5
    python run_multi_seed.py --model model3 --n_runs 3 --base_seed 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# evaluate.py (trong scripts/) dung import bare (from config..., from src...),
# nen can them thu muc scripts/ vao sys.path DAU TIEN de "import evaluate"
# lay dung file scripts/evaluate.py, va cac import "from config...",
# "from src..." o BEN TRONG evaluate.py cung tu resolve dung theo sys.path nay.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from scripts.config import IMAGE_SIZE
from train import train
from evaluate import evaluate_model  # noqa: E402  (lay tu scripts/evaluate.py nho sys.path o tren)


def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_multi_seed(
    model_name: str = "model1",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    train_dir: str = "data/processed/prepared_dataset/train",
    val_dir: str = "data/processed/prepared_dataset/val",
    test_dir: str = "data/processed/prepared_dataset/test",
    image_size: int = IMAGE_SIZE,
    base_seed: int = 42,
    n_runs: int = 5,
    output_dir: str = "outputs",
):
    seeds = [base_seed + i for i in range(n_runs)]
    print(f"[multi_seed] Se chay {n_runs} lan voi seeds: {seeds}")

    all_results = []

    for run_idx, seed in enumerate(seeds, start=1):
        print("\n" + "#" * 80)
        print(f"### LAN CHAY {run_idx}/{n_runs}  -  SEED = {seed} ###")
        print("#" * 80)

        _set_seed(seed)

        checkpoint_dir = os.path.join("checkpoints", f"seed_{seed}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        train(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            train_dir=train_dir,
            val_dir=val_dir,
            image_size=image_size,
            checkpoint_dir=checkpoint_dir,
        )

        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")

        # evaluate.py dung duong dan tuong doi tinh tu thu muc scripts/,
        # nen truyen duong dan tuyet doi de tranh nham lan.
        metrics = evaluate_model(
            model_name=model_name,
            checkpoint_path=os.path.abspath(checkpoint_path),
            test_dir=os.path.abspath(test_dir),
            batch_size=batch_size,
            output_dir=os.path.abspath(output_dir),
        )
        metrics["seed"] = seed
        all_results.append(metrics)

    print("\n" + "=" * 80)
    print(f"KET QUA TONG HOP QUA {n_runs} SEED KHAC NHAU ({model_name.upper()})")
    print("=" * 80)

    numeric_keys = [
        k for k, v in all_results[0].items()
        if isinstance(v, (int, float)) and k != "seed"
    ]
    summary = {}
    for key in numeric_keys:
        values = [r[key] for r in all_results]
        mean_v, std_v = float(np.mean(values)), float(np.std(values))
        summary[key] = {"mean": mean_v, "std": std_v, "values": values}
        print(f"  {key:<20} mean={mean_v:.4f}  std={std_v:.4f}   ({[round(v, 4) for v in values]})")

    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"multi_seed_report_{model_name}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "seeds": seeds,
                "per_run": [{k: v for k, v in r.items() if not isinstance(v, list) or k == "seed"} for r in all_results],
                "summary": summary,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n[multi_seed] Da luu bao cao chi tiet tai: {report_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chay training nhieu seed de kiem tra reproducibility")
    parser.add_argument("--model", type=str, default="model1", choices=["model1", "model2", "model3"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_dir", type=str, default="data/processed/prepared_dataset/train")
    parser.add_argument("--val_dir", type=str, default="data/processed/prepared_dataset/val")
    parser.add_argument("--test_dir", type=str, default="data/processed/prepared_dataset/test")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()
    run_multi_seed(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        image_size=args.image_size,
        base_seed=args.base_seed,
        n_runs=args.n_runs,
        output_dir=args.output_dir,
    )