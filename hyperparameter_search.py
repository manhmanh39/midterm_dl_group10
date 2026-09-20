"""hyperparameter_search.py - Optuna toi uu val mAP@0.5 (tren tap con cua val).

    python hyperparameter_search.py --model model1 --n_trials 30 --search_epochs 8 --num_workers 8
"""
from __future__ import annotations

import argparse
import gc
import json
import os

import optuna
from optuna.trial import TrialState
import torch
from torch.utils.data import DataLoader, Subset

from scripts.config import NUM_CLASSES, IMAGE_SIZE, get_grid_size
from scripts.src.dataset import VinBigDataDetectionDataset, collate_fn
from scripts.src.utils import CombinedLocalizationLoss
from scripts.src.metrics import evaluate_detections
from scripts.models.factory import build_model, make_optimizer


def objective(trial, model_name, train_dir, val_dir, search_epochs, image_size,
              sample_ratio=0.3, num_workers=4, val_images=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = get_grid_size(image_size=image_size)

    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32] if model_name == "model3" else [8, 16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    lambda_coord = trial.suggest_float("lambda_coord", 1.0, 10.0)

    train_ds = VinBigDataDetectionDataset(train_dir, image_size, grid_size, augment=True)
    val_ds = VinBigDataDetectionDataset(val_dir, image_size, grid_size, augment=False)
    g = torch.Generator().manual_seed(42)
    sub = torch.randperm(len(train_ds), generator=g)[: max(1, int(len(train_ds) * sample_ratio))].tolist()
    train_loader = DataLoader(Subset(train_ds, sub), batch_size=batch_size, shuffle=True, drop_last=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=True)
    max_val_batches = max(1, val_images // batch_size)

    try:
        model = build_model(model_name, NUM_CLASSES, False, "layer3").to(device)
        criterion = CombinedLocalizationLoss(lambda_coord=lambda_coord)
        optimizer = make_optimizer(model, optimizer_name, lr, weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=search_epochs)

        best = 0.0
        for epoch in range(1, search_epochs + 1):
            model.train()
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            scheduler.step()

            m = evaluate_detections(model, val_ds, val_loader, device, NUM_CLASSES,
                                    min_score=0.05, max_batches=max_val_batches)["map50"]
            best = max(best, m)
            trial.report(m, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best
    except (torch.OutOfMemoryError, ValueError) as e:
        print(f"[Optuna] bo trial: {e}")
        raise optuna.TrialPruned()
    finally:
        if "model" in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_search(model_name="model1", n_trials=30, search_epochs=8,
               train_dir="data/processed/dataset_202601/train", val_dir="data/processed/dataset_202601/val",
               image_size=IMAGE_SIZE, sample_ratio=0.3, output_dir="outputs", num_workers=4):
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    print(f"[optuna] {n_trials} trials cho '{model_name}' ({search_epochs} epoch/trial, "
          f"{int(sample_ratio * 100)}% train, {num_workers} workers)")
    study.optimize(
        lambda t: objective(t, model_name, train_dir, val_dir, search_epochs, image_size, sample_ratio, num_workers),
        n_trials=n_trials, show_progress_bar=True,
    )

    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    if not done:
        print("[optuna] Khong co trial nao hoan thanh.")
        return study
    print(f"Trials: {len(study.trials)} (hoan thanh {len(done)}, prune {pruned.__len__()})")
    print(f"Best val mAP50: {study.best_value:.4f}\nBest params: {study.best_params}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"best_hparams_{model_name}.json")
    with open(out_path, "w") as f:
        json.dump({"best_value_map50": study.best_value, "best_params": study.best_params}, f, indent=2)
    print(f"[optuna] Da luu: {out_path}")
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model1", choices=["model1", "model2", "model3"])
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--search_epochs", type=int, default=8)
    parser.add_argument("--train_dir", type=str, default="data/processed/dataset_202601/train")
    parser.add_argument("--val_dir", type=str, default="data/processed/dataset_202601/val")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--sample_ratio", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_workers", type=int, default=8)
    a = parser.parse_args()
    run_search(a.model, a.n_trials, a.search_epochs, a.train_dir, a.val_dir, a.image_size,
               a.sample_ratio, a.output_dir, a.num_workers)