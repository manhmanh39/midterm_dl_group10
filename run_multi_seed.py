"""run_multi_seed.py - Train nhieu seed, danh gia tren test, tinh mean/std.

    python run_multi_seed.py --model model1 --n_runs 5
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from scripts.config import IMAGE_SIZE
from train import train_pipeline
from evaluate import evaluate_model


def run_multi_seed(
    model_name="model1", epochs=40, batch_size=None, lr=None,
    train_dir="data/processed/dataset_202601/train", val_dir="data/processed/dataset_202601/val",
    test_dir="data/processed/dataset_202601/test", image_size=IMAGE_SIZE,
    base_seed=42, n_runs=5, output_dir="outputs",
):
    seeds = [base_seed + i for i in range(n_runs)]
    print(f"[multi_seed] {n_runs} lan, seeds: {seeds}")
    all_results = []

    for run_idx, seed in enumerate(seeds, start=1):
        print("\n" + "#" * 80 + f"\n### LAN {run_idx}/{n_runs} - SEED {seed} ###\n" + "#" * 80)
        ckpt_dir = os.path.join("checkpoints", f"seed_{seed}")
        res = train_pipeline(model_name=model_name, seed=seed, epochs=epochs, batch_size=batch_size, lr=lr,
                             train_dir=train_dir, val_dir=val_dir, image_size=image_size,
                             checkpoint_dir=ckpt_dir)
        metrics = evaluate_model(model_name=model_name, checkpoint_path=os.path.abspath(res["best_ckpt"]),
                                 data_dir=os.path.abspath(test_dir), batch_size=batch_size or 16,
                                 output_dir=os.path.abspath(output_dir))
        metrics = {k: v for k, v in metrics.items() if not isinstance(v, list)}
        metrics["seed"] = seed
        all_results.append(metrics)

    print("\n" + "=" * 80 + f"\nTONG HOP {n_runs} SEED ({model_name.upper()})\n" + "=" * 80)
    keys = [k for k, v in all_results[0].items() if isinstance(v, (int, float)) and k != "seed"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in all_results]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
        print(f"  {k:<22} mean={summary[k]['mean']:.4f} std={summary[k]['std']:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"multi_seed_report_{model_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "seeds": seeds, "per_run": all_results, "summary": summary},
                  f, indent=2, ensure_ascii=False)
    print(f"\n[multi_seed] Da luu: {path}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model1", choices=["model1", "model2", "model3"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--train_dir", default="data/processed/dataset_202601/train")
    p.add_argument("--val_dir", default="data/processed/dataset_202601/val")
    p.add_argument("--test_dir", default="data/processed/dataset_202601/test")
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--output_dir", default="outputs")
    a = p.parse_args()
    run_multi_seed(a.model, a.epochs, a.batch_size, a.lr, a.train_dir, a.val_dir, a.test_dir,
                   a.image_size, a.base_seed, a.n_runs, a.output_dir)