import argparse
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config
from seed_utils import set_seed
from data_loader import get_dataloaders
from models import get_model
from experiment_config import (
    DEVELOP_DIR,
    get_git_commit,
    git_worktree_is_dirty,
    compute_dataset_fingerprint,
    resolve_experiment_config,
)


def _build_optimizer(params, optimizer_name: str, lr: float, weight_decay: float):
    opt_type = optimizer_name.lower().strip()
    if opt_type == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_type in ("adamw", "adam"):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' không được hỗ trợ. Chọn 'adamw' hoặc 'sgd'.")


def _build_scheduler(optimizer, epochs: int):
    if config.LR_SCHEDULER == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=config.LR_PATIENCE, factor=0.5)
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


def _compute_batch_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    if config.IS_MULTILABEL:
        preds = (torch.sigmoid(outputs) >= config.MULTILABEL_THRESHOLD).float()
        correct = (preds == labels).sum().item()
        total = labels.numel()
    else:
        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
    return correct, total


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="  [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        c, t = _compute_batch_metrics(outputs, labels)
        correct += c
        total += t
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    n = max(1, len(loader.dataset))
    return running_loss / n, correct / total


def validate_one_epoch(model, loader, criterion, device) -> Tuple[float, float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="  [Val]  ", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            c, t = _compute_batch_metrics(outputs, labels)
            correct += c
            total += t

            if config.IS_MULTILABEL:
                all_probs.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    n = max(1, len(loader.dataset))
    val_loss, val_acc = running_loss / n, correct / total

    val_auc = float("nan")
    if config.IS_MULTILABEL and all_probs:
        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        try:
            val_auc = roc_auc_score(y_true, y_prob, average="macro")
        except ValueError:
            val_auc = float("nan")

    return val_loss, val_acc, val_auc


def train(
    model_name: str = "transfer",
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    optimizer: str = "adamw",
    weight_decay: float = config.WEIGHT_DECAY,
    dropout: Optional[float] = None,
    data_dir: Optional[str] = None,
    save_dir: Optional[Path] = None,
    device: torch.device = config.DEVICE,
    backbone_name: str = "resnet50",
    seed: int = config.SEED,
    use_tuned: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Huấn luyện mô hình thuần túy và lưu Checkpoint tốt nhất theo Validation Metric.
    Trả về đường dẫn tới file best checkpoint (best.pth).
    """
    if "optimizer_name" in kwargs:
        optimizer = kwargs["optimizer_name"]
    if "lr" in kwargs:
        learning_rate = kwargs["lr"]

    set_seed(seed)

    if save_dir is None:
        save_dir = DEVELOP_DIR / model_name / f"seed{seed}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"BẮT ĐẦU HUẤN LUYỆN: {model_name.upper()} (Seed={seed})")
    print(f"Tiêu chí chọn Best Model: {config.BEST_METRIC.upper()}")
    print(
        f"Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate:.6f} | "
        f"Opt: {optimizer} | WD: {weight_decay:.2e} | Dropout: {dropout}"
    )
    print(f"Multi-label: {config.IS_MULTILABEL} | Device: {device}")
    print("=" * 75)

    train_loader, val_loader, _, class_names, pos_weight = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, seed=seed
    )

    model_kwargs: Dict[str, Any] = {}
    is_transfer = model_name.lower().strip() in ("transfer", "model3", "base")
    if is_transfer:
        model_kwargs["backbone_name"] = backbone_name
        model_kwargs["freeze_base"] = True
    if dropout is not None:
        model_kwargs["dropout"] = dropout

    model = get_model(model_name, num_classes=len(class_names), **model_kwargs).to(device)

    if config.IS_MULTILABEL:
        pw = pos_weight.to(device) if pos_weight is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer_obj = _build_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        optimizer_name=optimizer,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = _build_scheduler(optimizer_obj, epochs)

    metric_mode = "min" if config.BEST_METRIC == "loss" else "max"
    best_metric_value = float("inf") if metric_mode == "min" else float("-inf")
    best_epoch = 1

    best_checkpoint_path = save_dir / "best.pth"
    dataset_meta = compute_dataset_fingerprint(data_dir=data_dir)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        if is_transfer and epoch == config.FREEZE_EPOCHS + 1:
            print(f"  >>> Mở khóa backbone (unfreeze) tại epoch {epoch}, LR -> {config.UNFREEZE_LR}")
            model.unfreeze_backbone()
            optimizer_obj = _build_optimizer(
                model.parameters(),
                optimizer_name=optimizer,
                lr=config.UNFREEZE_LR,
                weight_decay=weight_decay,
            )
            scheduler = _build_scheduler(optimizer_obj, epochs - epoch + 1)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_obj, device)
        val_loss, val_acc, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        dur = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({dur:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}% - Val AUC: {val_auc:.4f}", end=""
        )

        current_value = {"loss": val_loss, "acc": val_acc, "auc": val_auc}[config.BEST_METRIC]
        is_better = (
            (metric_mode == "min" and current_value < best_metric_value) or
            (metric_mode == "max" and current_value > best_metric_value)
        )

        if is_better or epoch == 1:
            best_metric_value = current_value
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_obj.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "num_classes": len(class_names),
                "class_names": class_names,
                "is_multilabel": config.IS_MULTILABEL,
                "backbone_name": backbone_name if is_transfer else None,
                "seed": seed,
                "best_metric": config.BEST_METRIC,
                "optimizer": optimizer,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "provenance": {
                    "git_commit": get_git_commit(),
                    "git_dirty": git_worktree_is_dirty(),
                    "dataset_fingerprint": dataset_meta["dataset_fingerprint"],
                    "model_name": model_name,
                    "seed": seed,
                },
            }, best_checkpoint_path)
            print(f" -> [ĐÃ LƯU BEST tại: {best_checkpoint_path.name}]")
        else:
            print()

    total_time = time.time() - start_time
    print("-" * 75)
    print(f"Huấn luyện hoàn tất trong {total_time:.2f}s! Best Epoch: {best_epoch} (Metric: {best_metric_value:.4f})")
    print("-" * 75)

    return best_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình VinBigData")
    parser.add_argument("--model", type=str, default="transfer", choices=["simple", "complex", "transfer"])
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use_tuned", action="store_true", default=False)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)

    args = parser.parse_args()

    resolved = resolve_experiment_config(args.model, cli_args=args, use_tuned=args.use_tuned)

    train(
        model_name=args.model,
        epochs=resolved["epochs"],
        batch_size=resolved["batch_size"],
        learning_rate=resolved["learning_rate"],
        optimizer=resolved["optimizer"],
        weight_decay=resolved["weight_decay"],
        dropout=resolved["dropout"],
        data_dir=args.data_dir,
        backbone_name=resolved["backbone"],
        seed=args.seed,
        use_tuned=args.use_tuned,
    )