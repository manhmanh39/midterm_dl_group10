"""
run_multi_seed.py - Dieu phoi thuc nghiem da seed Canonical (3 models x 3 seeds)
Seeds: [202601, 202602, 202603] | Models: [model1, model2, model3]
Cac giai doan:
  --phase develop: Huan luyen tren Train/Val (khong dong den Test)
  --phase lock: Tao protocol_lock.json va kiem tra preflight check
  --phase final-test: Danh gia 1-pass tren TestLoader voi permit (tuyet doi khong sinh lai lock)
  --phase all: Chay tuan tu develop -> lock -> final-test
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from scripts.config import IMAGE_SIZE, get_processed_data_root
from scripts.experiment_config import (
    CANONICAL_MODELS,
    CANONICAL_SEEDS,
    PreflightPermit,
    generate_protocol_lock,
    global_preflight_check,
)
from evaluate import evaluate_model
from train import train_pipeline


def run_canonical_multi_seed(
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    phase: str = "develop",
    epochs: int = 40,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    data_root: Optional[str] = None,
    image_size: int = IMAGE_SIZE,
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "outputs",
    num_workers: int = 4,
    eval_every: int = 1,
    data_mode: str = "real",
) -> Dict:
    models = models or CANONICAL_MODELS
    seeds = seeds or CANONICAL_SEEDS

    # Deduplication check
    if len(models) != len(set(models)):
        raise ValueError(f"Duplicate model in models: {models}")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seed in seeds: {seeds}")

    # Forbid phase='all' in real mode
    if phase == "all" and data_mode == "real":
        raise ValueError(
            "GIAO THUC CANONICAL: O che do 'real', cam tuyet doi chay '--phase all' mot mach!\n"
            "Quy trinh bat buoc phai tuan thu 3 buoc ro rang voi khoang dung kiem dinh:\n"
            "  Buoc 1: python run_multi_seed.py --model all --phase develop --data_mode real\n"
            "  Buoc 2: python run_multi_seed.py --model all --phase lock --data_mode real\n"
            "  Buoc 3: python run_multi_seed.py --model all --phase final-test --data_mode real\n"
            "Co '--phase all' chi danh rieng cho data_mode='demo' de phuc vu smoke testing tu dong."
        )

    data_root = data_root or get_processed_data_root()
    os.makedirs(output_dir, exist_ok=True)
    lock_path = os.path.join(output_dir, "protocol_lock.json")

    print("=" * 80)
    print(f"CANONICAL DETECTION PROTOCOL | Phase: {phase.upper()} | Mode: {data_mode.upper()}")
    print(f"Models: {models} | Seeds: {seeds}")
    print(f"Data root: {data_root} | Lock path: {lock_path}")
    print("=" * 80)

    # 1. Phase Develop
    if phase in ("develop", "all"):
        print("\n>>> BAT DAU PHASE 1: DEVELOP (Train + Val) <<<")
        for m in models:
            for s in seeds:
                print(f"\n--- Model: {m.upper()} | Seed: {s} ---")
                ckpt_dir = os.path.join(checkpoint_dir, f"seed_{s}")
                train_pipeline(
                    model_name=m,
                    seed=s,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    data_root=data_root,
                    image_size=image_size,
                    checkpoint_dir=ckpt_dir,
                    num_workers=num_workers,
                    eval_every=eval_every,
                    data_mode=data_mode,
                )

    # 2. Phase Lock & Preflight
    permit: Optional[PreflightPermit] = None

    if phase in ("lock", "all"):
        print("\n>>> BAT DAU PHASE 2: PROTOCOL LOCK GENERATION & PREFLIGHT CHECK <<<")
        generate_protocol_lock(
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            lock_path=lock_path,
            models=models,
            seeds=seeds,
            data_mode=data_mode,
        )
        passed, permit = global_preflight_check(lock_path=lock_path, data_root=data_root, data_mode=data_mode)
        if not passed:
            raise RuntimeError("Global preflight check failed! Khong the cap protocol_lock_token.")

    elif phase == "final-test":
        print("\n>>> PHASE FINAL-TEST: Doc protocol_lock.json co san (KHONG duoc phep sinh lai lock) <<<")
        if not os.path.isfile(lock_path):
            raise RuntimeError(
                f"Protocol lock file khong ton tai tai: {lock_path}!\n"
                f"Giai doan final-test khong duoc phep tu dong sinh lai lock de tranh ghi de artifact da sua. "
                f"Hay chay '--phase lock' truoc khi test."
            )
        passed, permit = global_preflight_check(lock_path=lock_path, data_root=data_root, data_mode=data_mode)
        if not passed:
            raise RuntimeError("Global preflight check failed! Khong the cap protocol_lock_token.")

    # 3. Phase Final-Test (Post-freeze 1 pass per model-seed)
    if phase in ("final-test", "all"):
        print("\n>>> BAT DAU PHASE 3: FINAL-TEST (Post-Freeze 1-Pass TestLoader) <<<")
        all_results: Dict[str, List[Dict]] = {m: [] for m in models}

        for m in models:
            for s in seeds:
                ckpt = os.path.join(checkpoint_dir, f"seed_{s}", f"{m}_seed{s}_best.pth")
                if not os.path.exists(ckpt):
                    ckpt = os.path.join(checkpoint_dir, f"{m}_seed{s}_best.pth")

                print(f"\n[Final-Test] Model: {m.upper()} | Seed: {s} | Ckpt: {ckpt}")
                metrics = evaluate_model(
                    model_name=m,
                    checkpoint_path=ckpt,
                    data_root=data_root,
                    image_size=image_size,
                    batch_size=batch_size or 16,
                    num_workers=num_workers,
                    lock_token=permit,
                    lock_path=lock_path,
                    output_dir=output_dir,
                    data_mode=data_mode,
                )
                scalar_metrics = {k: v for k, v in metrics.items() if not isinstance(v, list)}
                scalar_metrics["seed"] = s
                all_results[m].append(scalar_metrics)

        # Tong hop thong ke da seed (Mean +- Sample Std ddof=1)
        summary_by_model = {}
        for m in models:
            res_list = all_results[m]
            if not res_list:
                continue
            metric_keys = [k for k, v in res_list[0].items() if isinstance(v, (int, float)) and k != "seed"]
            summary = {}
            for k in metric_keys:
                vals = [r[k] for r in res_list]
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                summary[k] = {
                    "mean": round(mean_val, 4),
                    "sample_std_ddof1": round(std_val, 4),
                    "values": vals,
                }
            summary_by_model[m] = summary

        summary_file = os.path.join(output_dir, "canonical_multi_seed_test_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "protocol_version": "P0_P1_DETECTION_HARDENED_V1",
                "data_mode": data_mode,
                "models": models,
                "seeds": seeds,
                "per_run": all_results,
                "summary": summary_by_model,
            }, f, indent=2)
        print(f"\n[run_multi_seed] Da luu tong hop ket qua tai: {summary_file}")
        return summary_by_model

    return {}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Canonical Multi-Seed Detection Runner")
    p.add_argument("--model", default="all", choices=["model1", "model2", "model3", "all"])
    p.add_argument("--phase", default="develop", choices=["develop", "lock", "final-test", "all"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--data_root", default=None, help="Mac dinh lay tu get_processed_data_root()")
    p.add_argument("--data_mode", default="real", choices=["real", "demo"])
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=1)
    a = p.parse_args()

    target_models = CANONICAL_MODELS if a.model == "all" else [a.model]
    run_canonical_multi_seed(
        models=target_models,
        seeds=CANONICAL_SEEDS,
        phase=a.phase,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        data_root=a.data_root,
        image_size=a.image_size,
        checkpoint_dir=a.checkpoint_dir,
        output_dir=a.output_dir,
        num_workers=a.num_workers,
        eval_every=a.eval_every,
        data_mode=a.data_mode,
    )