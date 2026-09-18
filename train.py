import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config
from seed_utils import set_seed
from data_loader import get_dataloaders
from models import get_model


def _build_scheduler(optimizer, epochs: int):
    if config.LR_SCHEDULER == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=config.LR_PATIENCE, factor=0.5)
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


def _compute_batch_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    if config.IS_MULTILABEL:
        preds = (torch.sigmoid(outputs) > config.MULTILABEL_THRESHOLD).float()
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

    n = len(loader.dataset)
    return running_loss / n, correct / total


def validate_one_epoch(model, loader, criterion, device) -> Tuple[float, float, float]:
    """Trả về (val_loss, val_acc, val_auc). val_auc = nan nếu không tính được."""
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

    n = len(loader.dataset)
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
    data_dir: str = None,
    save_dir: Path = config.CHECKPOINT_DIR,
    device: torch.device = config.DEVICE,
    backbone_name: str = "resnet50",
    seed: int = config.SEED,
) -> Dict[str, List[float]]:
    set_seed(seed)

    print("=" * 70)
    print(f"BẮT ĐẦU HUẤN LUYỆN: {model_name.upper()}  (seed={seed})")
    print(f"Tiêu chí chọn Best Model: {config.BEST_METRIC.upper()}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate} | Multi-label: {config.IS_MULTILABEL}")
    print("=" * 70)

    train_loader, val_loader, _, class_names, pos_weight = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, seed=seed
    )

    model_kwargs = {}
    is_transfer = model_name in ("transfer", "model3", "base")
    if is_transfer:
        model_kwargs = {"backbone_name": backbone_name, "freeze_base": True}
    model = get_model(model_name, num_classes=len(class_names), **model_kwargs).to(device)

    if config.IS_MULTILABEL:
        pw = pos_weight.to(device) if pos_weight is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = _build_scheduler(optimizer, epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}

    metric_mode = "min" if config.BEST_METRIC == "loss" else "max"
    best_metric_value = float("inf") if metric_mode == "min" else float("-inf")

    best_checkpoint_path = save_dir / f"{model_name}_best.pth"
    last_checkpoint_path = save_dir / f"{model_name}_last.pth"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        if is_transfer and epoch == config.FREEZE_EPOCHS + 1:
            print(f"  >>> Mở khóa backbone (unfreeze) tại epoch {epoch}, LR -> {config.UNFREEZE_LR}")
            model.unfreeze_backbone()
            optimizer = AdamW(model.parameters(), lr=config.UNFREEZE_LR, weight_decay=config.WEIGHT_DECAY)
            scheduler = _build_scheduler(optimizer, epochs - epoch + 1)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

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
            torch.save({
                "epoch": epoch, "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss, "val_acc": val_acc, "val_auc": val_auc,
                "num_classes": len(class_names), "class_names": class_names,
                "is_multilabel": config.IS_MULTILABEL,
                "backbone_name": backbone_name if is_transfer else None,
                "seed": seed, "best_metric": config.BEST_METRIC,
            }, best_checkpoint_path)
            print(f" -> [ĐÃ LƯU BEST theo {config.BEST_METRIC.upper()}]")
        else:
            print()

    torch.save({
        "epoch": epochs, "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss, "val_acc": val_acc, "val_auc": val_auc,
        "num_classes": len(class_names),
        "is_multilabel": config.IS_MULTILABEL,
        "backbone_name": backbone_name if is_transfer else None,
    }, last_checkpoint_path)

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"Huấn luyện hoàn tất trong {total_time:.2f}s!")
    print(f"Best {config.BEST_METRIC.upper()}: {best_metric_value:.4f}  (tại: {best_checkpoint_path})")
    print("-" * 70)

    return history


import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân loại X-quang VinBigData")
    parser.add_argument("--model", type=str, default="transfer", choices=["simple", "complex", "transfer"])
    parser.add_argument("--backbone", type=str, default="resnet50",
                         choices=["resnet18", "resnet50", "mobilenet_v3", "efficientnet_b0", "convnext_tiny"])
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)

    args = parser.parse_args()

    hparam_path = config.OUTPUT_DIR / f"best_hparams_{args.model}.json"
    final_batch_size = args.batch_size
    final_lr = args.lr

    if hparam_path.exists():
        with open(hparam_path, "r") as f:
            best_data = json.load(f)
            best_params = best_data["best_params"]
            print(f"\n[train] 🎯 Đã tìm thấy file tuning của Optuna tại: {hparam_path}")
            print(f"[train] 📌 Các tham số tốt nhất được áp dụng tự động: {best_params}")
            
            if final_batch_size is None:
                final_batch_size = best_params.get("batch_size", config.BATCH_SIZE)
            if final_lr is None:
                final_lr = best_params.get("lr", config.LEARNING_RATE)
    else:
        print(f"\n[train] ⚠️ Không tìm thấy file hparam ({hparam_path}), dùng thông số mặc định.")
        if final_batch_size is None: final_batch_size = config.BATCH_SIZE
        if final_lr is None: final_lr = config.LEARNING_RATE

    train(
        model_name=args.model, 
        epochs=args.epochs, 
        batch_size=final_batch_size,
        learning_rate=final_lr, 
        data_dir=args.data_dir, 
        backbone_name=args.backbone,
        seed=args.seed,
    )