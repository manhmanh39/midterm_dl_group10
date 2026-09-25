"""Main entrypoint for the three-model object-detection experiment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from evaluate import evaluate_model
from scripts.config import IMAGE_SIZE, get_processed_data_root
from scripts.data.prepared_loader import describe_dataset, is_prepared_dataset
from train import train_pipeline


def run_pipeline(
    model_names: List[str], epochs: int, batch_size=None, lr=None, mode: str = "all",
    data_root: str | None = None, image_size: int = IMAGE_SIZE,
    freeze_backbone: bool = False, checkpoint_dir: str = "checkpoints",
    output_dir: str = "outputs", checkpoint_path: str | None = None,
    seed: int = 202601, num_workers: int = 20,
    negative_ratio: float = 1.0, eval_every: int = 2,
):
    data_root = data_root or get_processed_data_root()
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    print("*" * 80)
    print(f"Models={model_names} | Mode={mode.upper()} | Epochs={epochs} | Seed={seed}")
    print(f"Data root={data_root}")
    print("*" * 80)

    if not is_prepared_dataset(data_root):
        print(
            f"[ERROR] '{data_root}' is not a prepared dataset. Run:\n"
            f"python -m scripts.data.prepare_dataset --output_root {data_root} --overwrite"
        )
        sys.exit(1)

    for split, s in describe_dataset(data_root).items():
        print(
            f"  [{split:<5}] {s['n_images']:>5} images | {s['n_boxes']:>6} boxes | "
            f"No finding={s['n_no_finding']:>5} ({s['pct_no_finding']:.1f}%) | "
            f"avg boxes/image={s['avg_boxes_per_image']:.2f}"
        )

    summary_results = []
    for model_name in model_names:
        print("\n" + "#" * 80 + f"\n### MODEL: {model_name.upper()} ###\n" + "#" * 80)
        train_result = None
        if mode in ("train", "all"):
            train_result = train_pipeline(
                model_name=model_name, seed=seed, epochs=epochs,
                batch_size=batch_size, lr=lr,
                train_dir=train_dir, val_dir=val_dir,
                image_size=image_size, freeze_backbone=freeze_backbone,
                checkpoint_dir=checkpoint_dir, num_workers=num_workers,
                negative_ratio=negative_ratio, eval_every=eval_every,
            )

        if mode in ("eval", "all"):
            ckpt = checkpoint_path
            if ckpt is None and train_result is not None:
                ckpt = train_result["best_ckpt"]
            if ckpt is None:
                ckpt = os.path.join(checkpoint_dir, f"{model_name}_seed{seed}_best.pth")
            m = evaluate_model(
                model_name=model_name,
                checkpoint_path=os.path.abspath(ckpt),
                data_dir=os.path.abspath(test_dir),
                batch_size=batch_size or 16,
                num_workers=num_workers,
                output_dir=os.path.abspath(output_dir),
            )
            summary_results.append({
                "model": model_name,
                **{k: v for k, v in m.items() if not isinstance(v, list)},
            })

    if len(summary_results) > 1:
        header = f"{'Model':<10} {'Loss':>8} {'IoU':>8} {'mAP50':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}"
        print("\n" + "=" * len(header) + "\n" + header + "\n" + "-" * len(header))
        for r in summary_results:
            print(
                f"{r['model']:<10} {r['loss']:>8.4f} {r['iou']:>8.4f} {r['map50']:>8.4f} "
                f"{r['macro_precision']:>8.4f} {r['macro_recall']:>8.4f} {r['macro_f1']:>8.4f}"
            )
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "model_comparison_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        print(f"\n[main] Saved {path}")
    return summary_results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all", choices=["model1", "model2", "model3", "all"])
    p.add_argument("--mode", default="all", choices=["all", "train", "eval"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=202601)
    p.add_argument("--num_workers", type=int, default=20)
    p.add_argument("--data_root", default=None)
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--negative_ratio", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=2)
    a = p.parse_args()

    models = ["model1", "model2", "model3"] if a.model == "all" else [a.model]
    run_pipeline(
        model_names=models, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        mode=a.mode, data_root=a.data_root, image_size=a.image_size,
        freeze_backbone=a.freeze_backbone, checkpoint_dir=a.checkpoint_dir,
        output_dir=a.output_dir, checkpoint_path=a.checkpoint,
        seed=a.seed, num_workers=a.num_workers,
        negative_ratio=a.negative_ratio, eval_every=a.eval_every,
    )
