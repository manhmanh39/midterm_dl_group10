import argparse
import gc
import json
import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

import config
from data_loader import get_develop_dataloaders
from models import get_model
from seed_utils import set_seed
from metrics_utils import compute_safe_macro_auc


def _quick_validate(model, loader, criterion, device, data_mode: str = config.DEFAULT_DATA_MODE):
    model.eval()
    running_loss, n = 0.0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            n += images.size(0)
            all_probs.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    safe_res = compute_safe_macro_auc(y_true, y_prob, class_indices=range(config.NO_FINDING_CLASS_ID))
    valid_classes = safe_res["valid_classes"]

    if data_mode == "real" and valid_classes == 0:
        raise RuntimeError("[FAIL CLOSED] Optuna validation Macro-AUC-14 has 0 valid classes on real data! Caller aborts.")

    auc = safe_res["macro_auc"] if valid_classes > 0 else 0.0
    return running_loss / n, auc


def objective(
    trial: optuna.Trial,
    model_name: str,
    data_dir: Optional[str],
    search_epochs: int,
    sample_ratio: float = 0.2,
    data_mode: str = config.DEFAULT_DATA_MODE,
) -> float:
    set_seed(config.SEED)

    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    if model_name == "complex":
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
    else:
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "sgd"])
    dropout = trial.suggest_float("dropout", 0.2, 0.5)

    device = config.DEVICE
    train_loader, val_loader, class_names, pos_weight = get_develop_dataloaders(
        data_dir=data_dir, data_mode=data_mode, batch_size=batch_size, seed=config.SEED
    )

    dataset_len = len(train_loader.dataset)
    subset_size = max(1, int(dataset_len * sample_ratio))
    sub_indices = torch.randperm(dataset_len)[:subset_size]
    fast_train_dataset = Subset(train_loader.dataset, sub_indices)

    fast_train_loader = DataLoader(
        fast_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model_kwargs = (
        {"dropout": dropout}
        if model_name in ("simple", "complex")
        else {
            "backbone_name": "resnet50",
            "freeze_base": True,
            "dropout": dropout,
        }
    )

    try:
        model = get_model(model_name, num_classes=len(class_names), **model_kwargs).to(
            device
        )

        pw = pos_weight.to(device) if pos_weight is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        if optimizer_name == "adamw":
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
            )

        scheduler = CosineAnnealingLR(optimizer, T_max=search_epochs)

        best_auc = 0.0
        for epoch in range(1, search_epochs + 1):
            model.train()
            for images, labels in fast_train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

            _, val_auc = _quick_validate(model, val_loader, criterion, device, data_mode=data_mode)
            best_auc = max(best_auc, val_auc)

            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_auc

    except (torch.OutOfMemoryError, ValueError) as e:
        print(f"  [Optuna Warning] Trial bị bỏ qua do lỗi: {e}")
        raise optuna.TrialPruned()
    finally:
        if 'model' in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_search(
    model_name="transfer",
    n_trials=30,
    search_epochs=5,
    data_dir=None,
    sample_ratio=0.2,
    data_mode=config.DEFAULT_DATA_MODE,
):
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
    )

    print(
        f"[optuna] Bắt đầu quét {n_trials} trials cho model '{model_name}' (mode={data_mode}) "
        f"({search_epochs} epoch/trial, sử dụng {int(sample_ratio*100)}% dữ liệu train để tăng tốc)..."
    )
    study.optimize(
        lambda trial: objective(
            trial, model_name, data_dir, search_epochs, sample_ratio, data_mode=data_mode
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    print("\n" + "=" * 70)
    print(f"KẾT QUẢ QUÉT SIÊU THAM SỐ — {model_name.upper()}")
    print("=" * 70)
    print(
        f"  Tổng số trial: {len(study.trials)}  (hoàn thành: {len(completed)}, bị prune: {len(pruned)})"
    )
    print(f"  Best Val AUC: {study.best_value:.4f}")
    print(f"  Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print("=" * 70)

    out_path = config.OUTPUT_DIR / f"best_hparams_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"best_value_auc": study.best_value, "best_params": study.best_params},
            f,
            indent=2,
        )
    print(f"\n[optuna] Đã lưu bộ siêu tham số tốt nhất tại: {out_path}")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quét siêu tham số bằng Optuna")
    parser.add_argument(
        "--model",
        type=str,
        default="transfer",
        choices=["simple", "complex", "transfer"],
    )
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--search_epochs", type=int, default=5)
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.2,
        help="Tỷ lệ dữ liệu train dùng để quét (mặc định 0.2 = 20%)",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--data_mode", type=str, default=config.DEFAULT_DATA_MODE, choices=["real", "demo"])

    args = parser.parse_args()
    run_search(
        model_name=args.model,
        n_trials=args.n_trials,
        search_epochs=args.search_epochs,
        data_dir=args.data_dir,
        sample_ratio=args.sample_ratio,
        data_mode=args.data_mode,
    )