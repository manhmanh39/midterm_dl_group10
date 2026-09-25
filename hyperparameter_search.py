"""Optuna search for the refactored detection pipeline.

Output names match train.py exactly:
  outputs/best_hparams_model1.json
  outputs/best_hparams_model2.json
  outputs/best_hparams_model3.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random

import numpy as np
import optuna
from optuna.trial import TrialState
import torch
from torch.utils.data import DataLoader, Subset
import tqdm

from scripts.config import (
    BOXES_PER_CELL, IMAGE_SIZE, MAX_DET, MIN_AP_SCORE, NUM_CLASSES, get_grid_size,
)
from scripts.models.factory import build_model, make_optimizer
from scripts.models.model3_pretrained import PretrainedDetector
from scripts.src.dataset import VinBigDataDetectionDataset, collate_fn
from scripts.src.metrics import evaluate_detections
from scripts.src.utils import CombinedLocalizationLoss


def _seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def objective(
    trial, model_name, train_dir, val_dir, search_epochs, image_size,
    sample_ratio=0.3, num_workers=20,
):
    _seed_everything(202601 + trial.number)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = get_grid_size(image_size=image_size)

    if model_name == "model3":
        lr = trial.suggest_float("lr", 3e-5, 8e-4, log=True)
        batch_choices = [8, 16, 32]
    else:
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_choices = [8, 16, 32]
    batch_size = trial.suggest_categorical("batch_size", batch_choices)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "sgd"])
    lambda_box = trial.suggest_float("lambda_box", 3.0, 8.0)
    lambda_obj = trial.suggest_float("lambda_obj", 0.5, 2.0)
    lambda_class = trial.suggest_float("lambda_class", 0.5, 2.0)
    focal_gamma = trial.suggest_float("focal_gamma", 1.5, 3.0)

    train_ds = VinBigDataDetectionDataset(
        train_dir, image_size, grid_size, BOXES_PER_CELL, augment=True
    )
    val_ds = VinBigDataDetectionDataset(
        val_dir, image_size, grid_size, BOXES_PER_CELL, augment=False
    )
    class_weights = train_ds.class_weights().to(device)

    g = torch.Generator().manual_seed(42)
    n_sub = max(batch_size, int(len(train_ds) * sample_ratio))
    sub_idx = torch.randperm(len(train_ds), generator=g)[:n_sub].tolist()
    train_loader = DataLoader(
        Subset(train_ds, sub_idx), batch_size=batch_size, shuffle=True, drop_last=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    try:
        model = build_model(
            model_name, NUM_CLASSES, freeze_backbone=False,
            unfreeze_from_layer="layer4" if model_name == "model3" else "layer3",
            boxes_per_cell=BOXES_PER_CELL,
        ).to(device)
        mw, mh = train_ds.box_size_prior()
        model.head.initialize_box_prior(mw, mh)

        criterion = CombinedLocalizationLoss(
            lambda_box=lambda_box,
            lambda_obj=lambda_obj,
            lambda_class=lambda_class,
            focal_gamma=focal_gamma,
            class_weights=class_weights,
        ).to(device)
        optimizer = make_optimizer(model, optimizer_name, lr, weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(search_epochs, 1), eta_min=1e-6
        )
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

        best = 0.0
        for epoch in range(1, search_epochs + 1):
            if isinstance(model, PretrainedDetector):
                model.set_backbone_trainable(None if epoch <= 2 else "layer4")
            model.train()
            for imgs, targets in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    loss = criterion(model(imgs), targets)
                if device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            scheduler.step()

            metrics = evaluate_detections(
                model, val_ds, val_loader, device, NUM_CLASSES,
                min_score=MIN_AP_SCORE, max_det=MAX_DET,
            )
            m = metrics["map50"]
            best = max(best, m)
            trial.report(m, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best

    except (torch.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
            raise
        print(f"[Optuna] prune OOM: {e}")
        raise optuna.TrialPruned()
    finally:
        if "model" in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_search(
    model_name="model1", n_trials=25, search_epochs=6,
    train_dir="data/processed/dataset_202601/train",
    val_dir="data/processed/dataset_202601/val",
    image_size=IMAGE_SIZE, sample_ratio=0.3,
    output_dir="outputs", num_workers=20,
):
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    print(
        f"[optuna] {model_name}: {n_trials} trials, {search_epochs} epochs/trial, "
        f"{sample_ratio*100:.0f}% train proxy"
    )
    study.optimize(
        lambda t: objective(
            t, model_name, train_dir, val_dir, search_epochs,
            image_size, sample_ratio, num_workers,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    if not done:
        print("[optuna] Khong co trial nao complete.")
        return study

    print(f"Trials={len(study.trials)} complete={len(done)} pruned={len(pruned)}")
    print(f"Best val mAP50={study.best_value:.4f}\nBest params={study.best_params}")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"best_hparams_{model_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"best_value_map50": study.best_value, "best_params": study.best_params},
            f, indent=2, ensure_ascii=False,
        )
    print(f"[optuna] Saved {out_path}")
    return study


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model1", choices=["model1", "model2", "model3"])
    p.add_argument("--n_trials", type=int, default=25)
    p.add_argument("--search_epochs", type=int, default=6)
    p.add_argument("--train_dir", default="data/processed/dataset_202601/train")
    p.add_argument("--val_dir", default="data/processed/dataset_202601/val")
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    p.add_argument("--sample_ratio", type=float, default=0.3)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--num_workers", type=int, default=4)
    a = p.parse_args()
    run_search(
        a.model, a.n_trials, a.search_epochs, a.train_dir, a.val_dir,
        a.image_size, a.sample_ratio, a.output_dir, a.num_workers,
    )
