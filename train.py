"""
train.py - Huan luyen detector VinBigData Object Detection P0/P1.
Best checkpoint chon theo val mAP@0.5 (PASCAL VOC 2010+ all-points).
Chi su dung TrainLoader va ValLoader (tuyet doi khong dong vao Test).
Luu history.json, history.csv, va SHA-256 sidecar file sau khi train.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.config import CONF_THRESHOLD, IMAGE_SIZE, NUM_CLASSES, STRIDE, get_grid_size
from scripts.data.prepared_loader import get_develop_dataloaders
from scripts.experiment_config import get_git_commit, write_checkpoint_sha256
from scripts.models.factory import build_model, make_optimizer
from scripts.src.metrics import evaluate_detections
from scripts.src.utils import CombinedLocalizationLoss, calculate_iou, plot_history


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_one_epoch(model, loader, criterion, optimizer, device, grid_size, train, scaler=None):
    model.train() if train else model.eval()
    tot_loss = tot_iou = 0.0
    n = 0

    phase = "Train" if train else "Val"
    pbar = tqdm(loader, desc=f"[{phase}]", leave=False, dynamic_ncols=True)

    with torch.enable_grad() if train else torch.no_grad():
        for batch in pbar:
            imgs = batch[0].to(device)
            targets = batch[1].to(device)

            if train:
                optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                preds = model(imgs)
                loss = criterion(preds, targets)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            tot_loss += loss.item()
            tot_iou += calculate_iou(preds, targets, grid_size)
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return tot_loss / max(n, 1), tot_iou / max(n, 1)


def train_pipeline(
    model_name: str = "model1",
    seed: int = 202601,
    epochs: int = 40,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    lambda_coord: Optional[float] = None,
    optimizer_name: Optional[str] = None,
    data_root: str = "data/dataset_202601",
    image_size: int = IMAGE_SIZE,
    freeze_backbone: bool = False,
    unfreeze_from_layer: str = "layer3",
    checkpoint_dir: str = "checkpoints",
    num_workers: int = 4,
    use_optuna: bool = True,
    warmup_epochs: int = 2,
    eval_every: int = 1,
) -> Dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {}
    hp = os.path.join("outputs", f"best_hparams_{model_name}.json")
    if use_optuna and os.path.exists(hp):
        try:
            with open(hp) as f:
                params = json.load(f).get("best_params", {})
            print(f"[train] Optuna params: {params}")
        except Exception:
            pass

    lr = lr if lr is not None else params.get("lr", 3e-4 if model_name == "model3" else 1e-3)
    batch_size = batch_size if batch_size is not None else params.get("batch_size", 16)
    weight_decay = weight_decay if weight_decay is not None else params.get("weight_decay", 1e-4)
    lambda_coord = lambda_coord if lambda_coord is not None else params.get("lambda_coord", 5.0)
    optimizer_name = optimizer_name or params.get("optimizer", "adam")

    grid_size = get_grid_size(image_size=image_size, stride=STRIDE)
    print(f"[train] {device} | seed={seed} | {model_name} | bs={batch_size} lr={lr:.2e} "
          f"wd={weight_decay:.1e} opt={optimizer_name} lambda_coord={lambda_coord:.2f} eval_every={eval_every}")

    # Giai doan Develop: Chi lay train va val tu data_root, tuyet doi khong dong vao test
    train_loader, val_loader, train_ds, val_ds = get_develop_dataloaders(
        data_root=data_root, batch_size=batch_size, image_size=image_size, num_workers=num_workers, augment=True
    )

    model = build_model(model_name, NUM_CLASSES, freeze_backbone, unfreeze_from_layer, pretrained=True).to(device)
    if hasattr(model, "trainable_parameter_summary"):
        t, tot = model.trainable_parameter_summary()
        print(f"[train] Trainable parameters: {t:,} / {tot:,}")

    criterion = CombinedLocalizationLoss(lambda_coord=lambda_coord)
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[warmup_epochs])

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(checkpoint_dir, f"{model_name}_seed{seed}_best.pth")
    best_map = -1.0

    history_records = []
    tl, vl, ti, vi = [], [], [], []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_iou = run_one_epoch(model, train_loader, criterion, optimizer, device, grid_size, True, scaler)
        va_loss, va_iou = run_one_epoch(model, val_loader, criterion, optimizer, device, grid_size, False, scaler)

        do_eval = (epoch % eval_every == 0) or (epoch == epochs)
        det = evaluate_detections(
            model, val_ds, val_loader, device, NUM_CLASSES,
            conf_threshold=CONF_THRESHOLD, min_score=0.01
        ) if do_eval else None

        scheduler.step()
        tl.append(tr_loss)
        vl.append(va_loss)
        ti.append(tr_iou)
        vi.append(va_iou)

        current_map = det["map50"] if det else None
        current_macro_f1 = det["macro_f1"] if det else None
        is_best = False

        if det is not None and det["map50"] > best_map:
            best_map = det["map50"]
            is_best = True
            # Luu Checkpoint Provenance (khong tu luu SHA vao ben trong)
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "seed": seed,
                "git_commit": get_git_commit(),
                "model_state_dict": model.state_dict(),
                "val_loss": va_loss,
                "val_iou": va_iou,
                "val_map50": best_map,
                "num_classes": NUM_CLASSES,
                "image_size": image_size,
                "grid_size": grid_size,
                "freeze_backbone": freeze_backbone,
                "unfreeze_from_layer": unfreeze_from_layer,
                "hparams": {
                    "lr": lr,
                    "batch_size": batch_size,
                    "weight_decay": weight_decay,
                    "lambda_coord": lambda_coord,
                    "optimizer": optimizer_name,
                },
            }, best_ckpt)
            # Tinh SHA-256 tren dia va luu sidecar
            write_checkpoint_sha256(best_ckpt)

        record = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "val_loss": round(va_loss, 4),
            "train_iou": round(tr_iou, 4),
            "val_iou": round(va_iou, 4),
            "val_map50": round(current_map, 4) if current_map is not None else None,
            "macro_f1": round(current_macro_f1, 4) if current_macro_f1 is not None else None,
            "is_best": is_best,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        history_records.append(record)

        msg = (f"Epoch [{epoch:02d}/{epochs}] ({time.time() - t0:.0f}s) | "
               f"Train {tr_loss:.3f} | Val {va_loss:.3f} IoU {va_iou:.3f}")
        if det is not None:
            msg += f" | mAP50 {det['map50']:.4f} F1 {det['macro_f1']:.3f}"
            if is_best:
                msg += "  <- BEST"
        print(msg)

    # Luu history JSON va CSV
    os.makedirs("outputs", exist_ok=True)
    history_json = os.path.join("outputs", f"history_{model_name}_seed{seed}.json")
    history_csv = os.path.join("outputs", f"history_{model_name}_seed{seed}.csv")

    with open(history_json, "w", encoding="utf-8") as f:
        json.dump(history_records, f, indent=2)

    if history_records:
        with open(history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_records[0].keys()))
            writer.writeheader()
            writer.writerows(history_records)

    plot_history(tl, vl, ti, vi, f"{model_name}_seed{seed}", out_dir=checkpoint_dir)
    print(f"[train] Hoan thanh. Best val mAP50 = {best_map:.4f} -> {best_ckpt}")
    return {
        "best_ckpt": best_ckpt,
        "best_val_map50": best_map,
        "history_json": history_json,
        "history_csv": history_csv,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model1", choices=["model1", "model2", "model3"])
    p.add_argument("--seed", type=int, default=202601)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--data_root", default="data/dataset_202601")
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--unfreeze_from_layer", default="layer3",
                   choices=["conv1", "layer1", "layer2", "layer3", "layer4"])
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=1, help="tinh val mAP moi N epoch (mac dinh 1)")
    p.add_argument("--no_optuna", action="store_true")
    a = p.parse_args()
    train_pipeline(
        model_name=a.model,
        seed=a.seed,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        data_root=a.data_root,
        image_size=a.image_size,
        freeze_backbone=a.freeze_backbone,
        unfreeze_from_layer=a.unfreeze_from_layer,
        checkpoint_dir=a.checkpoint_dir,
        num_workers=a.num_workers,
        use_optuna=not a.no_optuna,
        eval_every=a.eval_every,
    )