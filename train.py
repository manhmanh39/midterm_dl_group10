"""train.py - Huan luyen detector. Best checkpoint chon theo val mAP@0.5 (KHONG theo loss).

    python train.py --model model1 --epochs 15 --no_optuna
    python train.py --model model3 --epochs 40
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
from tqdm import tqdm

from scripts.config import NUM_CLASSES, IMAGE_SIZE, CONF_THRESHOLD, get_grid_size
from scripts.src.dataset import VinBigDataDetectionDataset, collate_fn
from scripts.src.utils import CombinedLocalizationLoss, calculate_iou, plot_history
from scripts.src.metrics import evaluate_detections
from scripts.models.factory import build_model, make_optimizer


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            if train:
                optimizer.zero_grad()
            
            # Kich hoat Automatic Mixed Precision (AMP) de tang toc do train tren GPU
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
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
    model_name="model1", seed=202601, epochs=40,
    batch_size=None, lr=None, weight_decay=None, lambda_coord=None, optimizer_name=None,
    train_dir="data/processed/dataset_202601/train", val_dir="data/processed/dataset_202601/val",
    image_size=IMAGE_SIZE, freeze_backbone=False, unfreeze_from_layer="layer3",
    checkpoint_dir="checkpoints", num_workers=16, use_optuna=True, warmup_epochs=2, eval_every=2,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {}
    hp = os.path.join("outputs", f"best_hparams_{model_name}.json")
    if use_optuna and os.path.exists(hp):
        with open(hp) as f:
            params = json.load(f).get("best_params", {})
        print(f"[train] Optuna params: {params}")

    # Uu tien: CLI > Optuna JSON > mac dinh
    lr = lr if lr is not None else params.get("lr", 3e-4 if model_name == "model3" else 1e-3)
    batch_size = batch_size if batch_size is not None else params.get("batch_size", 16)
    weight_decay = weight_decay if weight_decay is not None else params.get("weight_decay", 1e-4)
    lambda_coord = lambda_coord if lambda_coord is not None else params.get("lambda_coord", 5.0)
    optimizer_name = optimizer_name or params.get("optimizer", "adam")

    grid_size = get_grid_size(image_size=image_size)
    print(f"[train] {device} | seed={seed} | {model_name} | bs={batch_size} lr={lr:.2e} "
          f"wd={weight_decay:.1e} opt={optimizer_name} lambda_coord={lambda_coord:.2f}")

    train_ds = VinBigDataDetectionDataset(train_dir, image_size, grid_size, augment=True)
    val_ds = VinBigDataDetectionDataset(val_dir, image_size, grid_size, augment=False)
    pin = torch.cuda.is_available()
    dl_kw = dict(num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin)
    if num_workers > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kw)

    model = build_model(model_name, NUM_CLASSES, freeze_backbone, unfreeze_from_layer).to(device)
    if hasattr(model, "trainable_parameter_summary"):
        t, tot = model.trainable_parameter_summary()
        print(f"[train] Trainable: {t:,}/{tot:,}")

    criterion = CombinedLocalizationLoss(lambda_coord=lambda_coord)
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay)
    
    # Khoi tao GradScaler cho AMP
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[warmup_epochs])

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(checkpoint_dir, f"{model_name}_seed{seed}_best.pth")
    best_map = -1.0
    tl, vl, ti, vi = [], [], [], []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_iou = run_one_epoch(model, train_loader, criterion, optimizer, device, grid_size, True, scaler)
        va_loss, va_iou = run_one_epoch(model, val_loader, criterion, optimizer, device, grid_size, False, scaler)
        
        do_eval = (epoch % eval_every == 0) or epoch == epochs
        det = evaluate_detections(model, val_ds, val_loader, device, NUM_CLASSES,
                                    conf_threshold=CONF_THRESHOLD, min_score=0.05) if do_eval else None
        scheduler.step()
        tl.append(tr_loss); vl.append(va_loss); ti.append(tr_iou); vi.append(va_iou)

        # Xay dung thong bao log co ban
        msg = (f"Epoch [{epoch:02d}/{epochs}] ({time.time() - t0:.0f}s) | "
               f"Train {tr_loss:.3f} | Val {va_loss:.3f} IoU {va_iou:.3f}")

        # Chi xu ly mAP va luu checkpoint khi co ket qua evaluation
        if det is not None:
            msg += f" | mAP50 {det['map50']:.4f} F1 {det['micro_f1']:.3f} preds/img {det['mean_preds_per_image']:.2f}"
            
            if det["map50"] > best_map:
                best_map = det["map50"]
                torch.save({
                    "epoch": epoch, "model_name": model_name, "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "val_loss": va_loss, "val_iou": va_iou, "val_map50": best_map,
                    "num_classes": NUM_CLASSES, "image_size": image_size, "grid_size": grid_size,
                    "freeze_backbone": freeze_backbone, "unfreeze_from_layer": unfreeze_from_layer,
                    "hparams": {"lr": lr, "batch_size": batch_size, "weight_decay": weight_decay,
                                "lambda_coord": lambda_coord, "optimizer": optimizer_name},
                }, best_ckpt)
                msg += "  <- BEST"
                
        print(msg)

    plot_history(tl, vl, ti, vi, f"{model_name}_seed{seed}", out_dir=checkpoint_dir)
    print(f"[train] Xong. Best val mAP50 = {best_map:.4f} -> {best_ckpt}")
    return {"best_ckpt": best_ckpt, "best_val_map50": best_map}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model1", choices=["model1", "model2", "model3"])
    p.add_argument("--seed", type=int, default=202601)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--train_dir", default="data/processed/dataset_202601/train")
    p.add_argument("--val_dir", default="data/processed/dataset_202601/val")
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--unfreeze_from_layer", default="layer3",
                   choices=["conv1", "layer1", "layer2", "layer3", "layer4"])
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--eval_every", type=int, default=2, help="tinh mAP moi N epoch")
    p.add_argument("--no_optuna", action="store_true")
    a = p.parse_args()
    train_pipeline(a.model, a.seed, a.epochs, a.batch_size, a.lr, None, None, None,
                   a.train_dir, a.val_dir, a.image_size, a.freeze_backbone, a.unfreeze_from_layer,
                   a.checkpoint_dir, a.num_workers, use_optuna=not a.no_optuna, eval_every=a.eval_every)