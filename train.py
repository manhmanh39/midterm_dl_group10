"""
train.py - Huấn luyện mô hình thuần túy, lưu Best Checkpoint và Lưu Lịch sử Huấn luyện (Training History).
Hỗ trợ:
  - P0.4: Phân định rõ siêu tham số transfer learning
  - P0.5: Đo lường chính xác đồng thời Exact Match Accuracy và Per-Label Accuracy
  - P0.6: Lưu đầy đủ History epoch-by-epoch vào JSON và CSV
  - P0.7: Đóng gói Provenance đầy đủ (Commit, Dirty check, Dataset Fingerprint, Hyperparameters)
"""

import argparse
import csv
from datetime import datetime
import json
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


def _compute_batch_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    P0.5: Tính toán song song:
      1. Per-label accuracy: Đúng từng phần tử nhãn trên toàn bộ nhãn (tp + tn) / (num_classes * batch_size)
      2. Exact-match accuracy: Toàn bộ 15 nhãn của một mẫu phải khớp 100%
    """
    if config.IS_MULTILABEL:
        preds = (torch.sigmoid(outputs) >= config.MULTILABEL_THRESHOLD).float()
        per_label_correct = (preds == labels).sum().item()
        total_labels = labels.numel()
        exact_match_correct = (preds == labels).all(dim=1).sum().item()
        total_samples = labels.size(0)
    else:
        _, preds = torch.max(outputs, 1)
        per_label_correct = (preds == labels).sum().item()
        total_labels = labels.size(0)
        exact_match_correct = per_label_correct
        total_samples = total_labels
    return per_label_correct, total_labels, exact_match_correct, total_samples


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    c_label, t_label = 0, 0
    c_exact, t_exact = 0, 0

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
        c_lbl, t_lbl, c_ex, t_ex = _compute_batch_metrics(outputs, labels)
        c_label += c_lbl
        t_label += t_lbl
        c_exact += c_ex
        t_exact += t_ex
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    n = max(1, len(loader.dataset))
    per_label_acc = c_label / max(1, t_label)
    exact_match_acc = c_exact / max(1, t_exact)
    return running_loss / n, per_label_acc, exact_match_acc


def validate_one_epoch(model, loader, criterion, device) -> Tuple[float, float, float, float]:
    model.eval()
    running_loss = 0.0
    c_label, t_label = 0, 0
    c_exact, t_exact = 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="  [Val]  ", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            c_lbl, t_lbl, c_ex, t_ex = _compute_batch_metrics(outputs, labels)
            c_label += c_lbl
            t_label += t_lbl
            c_exact += c_ex
            t_exact += t_ex

            if config.IS_MULTILABEL:
                all_probs.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    n = max(1, len(loader.dataset))
    val_loss = running_loss / n
    val_per_label_acc = c_label / max(1, t_label)
    val_exact_match_acc = c_exact / max(1, t_exact)

    val_auc = float("nan")
    if config.IS_MULTILABEL and all_probs:
        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        try:
            val_auc = roc_auc_score(y_true, y_prob, average="macro")
        except ValueError:
            val_auc = float("nan")

    return val_loss, val_per_label_acc, val_exact_match_acc, val_auc


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
    parameter_sources: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Path:
    """
    Huấn luyện mô hình thuần túy, lưu Checkpoint tốt nhất và Lưu Toàn Bộ History.
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

    dataset_meta = compute_dataset_fingerprint(data_dir=data_dir)
    if not dataset_meta["is_demo_data"] and git_worktree_is_dirty():
        raise RuntimeError("[FAIL CLOSED] Working tree của Git đang có thay đổi chưa commit!")

    train_loader, val_loader, _, class_names, pos_weight = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, seed=seed
    )

    model_kwargs: Dict[str, Any] = {}
    is_transfer = model_name.lower().strip() in ("transfer", "model3", "base")
    if is_transfer:
        model_kwargs["backbone_name"] = backbone_name
        model_kwargs["pretrained"] = True
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

    start_time = time.time()
    history_records: List[Dict[str, Any]] = []

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

        # Lấy LR thực tế dùng cho epoch này trước khi train và trước khi scheduler step
        current_lr = float(optimizer_obj.param_groups[0]["lr"])

        train_loss, train_per_label_acc, train_exact_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_obj, device
        )
        val_loss, val_per_label_acc, val_exact_acc, val_auc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        dur = time.time() - epoch_start

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({dur:.1f}s) | "
            f"Train Loss: {train_loss:.4f} (Per-Label: {train_per_label_acc*100:.1f}%, Exact: {train_exact_acc*100:.1f}%) | "
            f"Val Loss: {val_loss:.4f} (Per-Label: {val_per_label_acc*100:.1f}%, Exact: {val_exact_acc*100:.1f}%) - Val AUC: {val_auc:.4f}",
            end="",
        )

        # P0.6: Ghi nhận lịch sử huấn luyện
        history_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_per_label_acc": round(train_per_label_acc, 4),
            "train_exact_match_acc": round(train_exact_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_per_label_acc": round(val_per_label_acc, 4),
            "val_exact_match_acc": round(val_exact_acc, 4),
            "val_auc": round(val_auc, 4) if not np.isnan(val_auc) else None,
            "learning_rate": current_lr,
            "backbone_frozen": bool(is_transfer and epoch <= config.FREEZE_EPOCHS),
            "duration_sec": round(dur, 2),
        })

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        current_value = {"loss": val_loss, "acc": val_per_label_acc, "auc": val_auc}[config.BEST_METRIC]
        is_better = (
            (metric_mode == "min" and current_value < best_metric_value)
            or (metric_mode == "max" and current_value > best_metric_value)
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
                "val_per_label_acc": val_per_label_acc,
                "val_exact_match_acc": val_exact_acc,
                "val_auc": val_auc,
                "num_classes": len(class_names),
                "class_names": class_names,
                "is_multilabel": config.IS_MULTILABEL,
                "backbone_name": backbone_name if is_transfer else None,
                "seed": seed,
                "best_metric": config.BEST_METRIC,
                # P0.7: Provenance hoàn chỉnh có cấu trúc
                "provenance": {
                    "timestamp": datetime.now().isoformat(),
                    "git_commit": get_git_commit(),
                    "git_dirty": git_worktree_is_dirty(),
                    "dataset_fingerprint": dataset_meta["dataset_fingerprint"],
                    "data_dir": str(dataset_meta.get("data_dir")),
                    "model_name": model_name,
                    "seed": seed,
                    "device": str(device),
                    "device_name": torch.cuda.get_device_name(device) if (device.type == "cuda" and torch.cuda.is_available()) else "CPU",
                    "is_multilabel": config.IS_MULTILABEL,
                    "parameter_sources": parameter_sources or kwargs.get("source", {}),
                    "hyperparameters": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "optimizer": optimizer,
                        "weight_decay": weight_decay,
                        "dropout": dropout,
                        "backbone_name": backbone_name if is_transfer else None,
                        "best_metric": config.BEST_METRIC,
                    },
                    "dataset_checksums": {
                        "manifest_sha256": dataset_meta.get("manifest_sha256"),
                        "train_annotations_sha256": dataset_meta.get("train_annotations_sha256"),
                        "val_annotations_sha256": dataset_meta.get("val_annotations_sha256"),
                        "test_annotations_sha256": dataset_meta.get("test_annotations_sha256"),
                    },
                },
            }, best_checkpoint_path)
            print(f" -> [ĐÃ LƯU BEST tại: {best_checkpoint_path.name}]")
        else:
            print()

    total_time = time.time() - start_time
    print("-" * 75)
    print(f"Huấn luyện hoàn tất trong {total_time:.2f}s! Best Epoch: {best_epoch} (Metric: {best_metric_value:.4f})")
    print("-" * 75)

    # P0.6: LƯU LỊCH SỬ HUẤN LUYỆN RA JSON VÀ CSV
    history_json_path = save_dir / "history.json"
    history_csv_path = save_dir / "history.csv"

    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history_records, f, indent=2, ensure_ascii=False)

    if history_records:
        with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_records[0].keys()))
            writer.writeheader()
            for r in history_records:
                writer.writerow(r)

    # Copy một bản vào outputs/ với tên chuẩn: history_{model_name}_seed{seed}.csv
    outputs_csv = config.OUTPUT_DIR / f"history_{model_name}_seed{seed}.csv"
    outputs_json = config.OUTPUT_DIR / f"history_{model_name}_seed{seed}.json"
    try:
        import shutil
        shutil.copy2(history_csv_path, outputs_csv)
        shutil.copy2(history_json_path, outputs_json)
    except Exception:
        pass

    print(f"[history] 💾 Đã lưu lịch sử huấn luyện ({len(history_records)} epochs) tại:")
    print(f"  - {history_csv_path}")
    print(f"  - {history_json_path}")

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