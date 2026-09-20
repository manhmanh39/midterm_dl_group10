"""
train.py - Huan luyen model OBJECT DETECTION (anchor-free, 1-scale kieu YOLO).
Ho tro co dinh seed (multi-seed reproducibility), tu dong load best hyperparameters 
tu Optuna va quan ly backbone linh hoat.

Chay:
    python train.py --model model1 --seed 202601 --epochs 30 --num_workers 16
    python train.py --model model3 --epochs 30 --freeze_backbone --num_workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.config import NUM_CLASSES, IMAGE_SIZE, GRID_SIZE, get_grid_size
from scripts.src.dataset import VinBigDataDetectionDataset, collate_fn
from scripts.src.utils import CombinedLocalizationLoss, calculate_iou, plot_history
from scripts.models.model1_sequential import SequentialCNNDetector
from scripts.models.model2_residual import NonSequentialCNNDetector
from scripts.models.model3_pretrained import PretrainedDetector


def set_seed(seed: int):
    """Co dinh seed toan dien de dam bao tinh tai lap (reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(model_name: str, num_classes: int, freeze_backbone: bool, unfreeze_from_layer: str):
    if model_name == "model1":
        return SequentialCNNDetector(num_classes=num_classes)
    elif model_name == "model2":
        return NonSequentialCNNDetector(num_classes=num_classes)
    elif model_name == "model3":
        return PretrainedDetector(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_from_layer=unfreeze_from_layer,
        )
    else:
        raise ValueError(f"Model khong hop le: {model_name}. Chon model1 / model2 / model3.")


def run_one_epoch(model, loader, criterion, optimizer, device, grid_size, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            if train:
                optimizer.zero_grad()

            preds = model(imgs)
            loss = criterion(preds, targets)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            total_iou += calculate_iou(preds, targets, grid_size)
            num_batches += 1

    return total_loss / max(num_batches, 1), total_iou / max(num_batches, 1)


def train_pipeline(
    model_name: str = "model1",
    seed: int = 202601,
    epochs: int = 30,
    batch_size: int = None,  # Se duoc ghi de boi Optuna neu co
    lr: float = None,        # Se duoc ghi de boi Optuna neu co
    weight_decay: float = 1e-4,
    lambda_coord: float = 5.0,
    train_dir: str = "data/processed/dataset_202601/train",
    val_dir: str = "data/processed/dataset_202601/val",
    image_size: int = IMAGE_SIZE,
    freeze_backbone: bool = True,
    unfreeze_from_layer: str = "layer4",
    checkpoint_dir: str = "checkpoints",
    num_workers: int = 16,
):
    # 1. Co dinh seed cho toan bo qua trinh
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device} | Seed: {seed} | Model: {model_name}")

    # 2. Tu dong load hyperparameter tot nhat tu Optuna (neu ton tai)
    hparam_path = os.path.join("outputs", f"best_hparams_{model_name}.json")
    if os.path.exists(hparam_path):
        with open(hparam_path, "r") as f:
            hdata = json.load(f)
            params = hdata.get("best_params", {})
        print(f"[train] Da load thanh cong Optuna hyperparameters tu {hparam_path}: {params}")
        lr = params.get("lr", lr if lr else 1e-3)
        # batch_size = params.get("batch_size", batch_size if batch_size else 16)
        batch_size = 32
        weight_decay = params.get("weight_decay", weight_decay)
        lambda_coord = params.get("lambda_coord", lambda_coord)
    else:
        print(f"[train Warning] Khong tim thay {hparam_path}, su dung tham so mac dinh.")
        lr = lr if lr else 1e-3
        batch_size = batch_size if batch_size else 16

    grid_size = get_grid_size(image_size=image_size)
    print(f"[train] image_size={image_size}, grid_size={grid_size}x{grid_size} | batch_size={batch_size}, lr={lr}")

    # 3. Chuan bi Dataset & DataLoader
    train_dataset = VinBigDataDetectionDataset(train_dir, image_size=image_size, grid_size=grid_size, augment=True)
    val_dataset = VinBigDataDetectionDataset(val_dir, image_size=image_size, grid_size=grid_size, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
    )

    # 4. Khoi tao Model, Loss, Optimizer, Scheduler
    model = build_model(model_name, NUM_CLASSES, freeze_backbone, unfreeze_from_layer).to(device)

    if hasattr(model, "trainable_parameter_summary"):
        trainable, total = model.trainable_parameter_summary()
        print(f"[train] Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    criterion = CombinedLocalizationLoss(lambda_coord=lambda_coord)
    
    optimizer_cls = torch.optim.Adam if params.get("optimizer", "adam") == "adam" else torch.optim.SGD
    optimizer = optimizer_cls(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    train_losses, val_losses, train_ious, val_ious = [], [], [], []

    print(f"\n[train] Bat dau huan luyen '{model_name}' (Seed: {seed}) trong {epochs} epoch...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_iou = run_one_epoch(model, train_loader, criterion, optimizer, device, grid_size, train=True)
        val_loss, val_iou = run_one_epoch(model, val_loader, criterion, optimizer, device, grid_size, train=False)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        dt = time.time() - t0
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} - Val IoU: {val_iou:.4f}",
            end="",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_seed{seed}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_iou,
                "num_classes": NUM_CLASSES,
                "image_size": image_size,
                "grid_size": grid_size,
                "freeze_backbone": freeze_backbone,
                "unfreeze_from_layer": unfreeze_from_layer,
                "best_params": params,
            }, ckpt_path)
            print(f" -> [DA LUU BEST tai {ckpt_path}]")
        else:
            print()

    last_ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_seed{seed}_last.pth")
    torch.save({
        "epoch": epochs,
        "model_name": model_name,
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "num_classes": NUM_CLASSES,
        "image_size": image_size,
        "grid_size": grid_size,
    }, last_ckpt_path)

    plot_history(train_losses, val_losses, train_ious, val_ious, f"{model_name}_seed{seed}", out_dir=checkpoint_dir)

    print(f"[train] Hoan tat. Best Val Loss: {best_val_loss:.4f}")
    return {"train_loss": train_losses, "val_loss": val_losses, "train_iou": train_ious, "val_iou": val_ious}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huan luyen model object detection VinBigData")
    parser.add_argument("--model", type=str, default="model1", choices=["model1", "model2", "model3"])
    parser.add_argument("--seed", type=int, default=1, help="Seed co dinh cho qua trinh train")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--train_dir", type=str, default="data/processed/dataset_202601/train")
    parser.add_argument("--val_dir", type=str, default="data/processed/dataset_202601/val")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--freeze_backbone", dest="freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--unfreeze_from_layer", type=str, default="layer4",
                        choices=["conv1", "layer1", "layer2", "layer3", "layer4"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()

    train_pipeline(
        model_name=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        image_size=args.image_size,
        freeze_backbone=args.freeze_backbone,
        unfreeze_from_layer=args.unfreeze_from_layer,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
    )