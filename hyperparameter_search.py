"""
hyperparameter_search.py - Quet sieu tham so (Optuna) cho model OBJECT DETECTION.

Muc tieu toi uu: Val IoU trung binh tren cac cell co object (proxy nhanh,
khong ton chi phi decode+NMS+matching nhu evaluate.py that su) sau
`search_epochs` epoch huan luyen tren mot phan du lieu train (sample_ratio)
de tang toc qua trinh quet.

Chay:
    python hyperparameter_search.py --model model1 --n_trials 30 --search_epochs 5 --num_workers 8
"""

from __future__ import annotations

import argparse
import gc
import json

import optuna
from optuna.trial import TrialState
import torch
from torch.utils.data import DataLoader, Subset

from scripts.config import NUM_CLASSES, IMAGE_SIZE, get_grid_size
from scripts.src.dataset import VinBigDataDetectionDataset, collate_fn
from scripts.src.utils import CombinedLocalizationLoss, calculate_iou
from scripts.models.model1_sequential import SequentialCNNDetector
from scripts.models.model2_residual import NonSequentialCNNDetector
from scripts.models.model3_pretrained import PretrainedDetector


def build_model(model_name: str, num_classes: int, freeze_backbone: bool = True):
    if model_name == "model1":
        return SequentialCNNDetector(num_classes=num_classes)
    elif model_name == "model2":
        return NonSequentialCNNDetector(num_classes=num_classes)
    elif model_name == "model3":
        return PretrainedDetector(num_classes=num_classes, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Model khong hop le: {model_name}")


def _quick_validate(model, loader, criterion, device, grid_size):
    model.eval()
    total_loss, total_iou, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            total_loss += loss.item()
            total_iou += calculate_iou(preds, targets, grid_size)
            n_batches += 1
    return total_loss / max(n_batches, 1), total_iou / max(n_batches, 1)


def objective(
    trial: optuna.Trial,
    model_name: str,
    train_dir: str,
    val_dir: str,
    search_epochs: int,
    image_size: int,
    sample_ratio: float = 0.3,
    num_workers: int = 4,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = get_grid_size(image_size=image_size)

    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    if model_name == "model3":
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    else:
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    lambda_coord = trial.suggest_float("lambda_coord", 1.0, 10.0)

    train_dataset = VinBigDataDetectionDataset(train_dir, image_size=image_size, grid_size=grid_size, augment=True)
    val_dataset = VinBigDataDetectionDataset(val_dir, image_size=image_size, grid_size=grid_size, augment=False)

    n_sub = max(1, int(len(train_dataset) * sample_ratio))
    sub_indices = torch.randperm(len(train_dataset))[:n_sub]
    fast_train_dataset = Subset(train_dataset, sub_indices.tolist())

    # Them num_workers va pin_memory de tang toc nap du lieu
    train_loader = DataLoader(
        fast_train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True
    )

    try:
        model = build_model(model_name, NUM_CLASSES).to(device)
        criterion = CombinedLocalizationLoss(lambda_coord=lambda_coord)

        params = filter(lambda p: p.requires_grad, model.parameters())
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=search_epochs)

        best_iou = 0.0
        for epoch in range(1, search_epochs + 1):
            model.train()
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            scheduler.step()

            _, val_iou = _quick_validate(model, val_loader, criterion, device, grid_size)
            best_iou = max(best_iou, val_iou)

            trial.report(val_iou, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_iou

    except (torch.OutOfMemoryError, ValueError) as e:
        print(f"  [Optuna Warning] Trial bi bo qua do loi: {e}")
        raise optuna.TrialPruned()
    finally:
        if "model" in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_search(
    model_name: str = "model1",
    n_trials: int = 30,
    search_epochs: int = 5,
    train_dir: str = "data/processed/prepared_dataset/train",
    val_dir: str = "data/processed/prepared_dataset/val",
    image_size: int = IMAGE_SIZE,
    sample_ratio: float = 0.3,
    output_dir: str = "outputs",
    num_workers: int = 4,
):
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print(
        f"[optuna] Bat dau quet {n_trials} trials cho model '{model_name}' "
        f"({search_epochs} epoch/trial, {int(sample_ratio*100)}% du lieu train, {num_workers} workers)..."
    )
    study.optimize(
        lambda trial: objective(trial, model_name, train_dir, val_dir, search_epochs, image_size, sample_ratio, num_workers),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    print("\n" + "=" * 70)
    print(f"KET QUA QUET SIEU THAM SO - {model_name.upper()}")
    print("=" * 70)
    print(f"  Tong so trial: {len(study.trials)}  (hoan thanh: {len(completed)}, bi prune: {len(pruned)})")
    print(f"  Best Val IoU: {study.best_value:.4f}")
    print("  Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print("=" * 70)

    import os
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"best_hparams_{model_name}.json")
    with open(out_path, "w") as f:
        json.dump({"best_value_iou": study.best_value, "best_params": study.best_params}, f, indent=2)
    print(f"\n[optuna] Da luu bo sieu tham so tot nhat tai: {out_path}")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quet sieu tham so bang Optuna cho object detection")
    parser.add_argument("--model", type=str, default="model1", choices=["model1", "model2", "model3"])
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--search_epochs", type=int, default=5)
    parser.add_argument("--train_dir", type=str, default="data/processed/prepared_dataset/train")
    parser.add_argument("--val_dir", type=str, default="data/processed/prepared_dataset/val")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--sample_ratio", type=float, default=0.3,
                        help="Ty le du lieu train dung de quet (mac dinh 0.3 = 30%)")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_workers", type=int, default=8, help="So luong workers cho DataLoader")

    args = parser.parse_args()
    run_search(
        model_name=args.model,
        n_trials=args.n_trials,
        search_epochs=args.search_epochs,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        image_size=args.image_size,
        sample_ratio=args.sample_ratio,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )