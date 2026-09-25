"""Train detector; best checkpoint selected by validation mAP@0.5."""
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

from scripts.config import (
    BOXES_PER_CELL, CONF_THRESHOLD, FOCAL_ALPHA, FOCAL_GAMMA, IMAGE_SIZE,
    LAMBDA_BOX, LAMBDA_CLASS, LAMBDA_OBJ, MAX_DET, MIN_AP_SCORE, NUM_CLASSES,
    get_grid_size,
)
from scripts.models.factory import build_model, make_optimizer
from scripts.models.model3_pretrained import PretrainedDetector
from scripts.src.dataset import VinBigDataDetectionDataset, collate_fn
from scripts.src.metrics import evaluate_detections
from scripts.src.utils import CombinedLocalizationLoss, calculate_iou, plot_history


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_one_epoch(model, loader, criterion, optimizer, device, train, scaler=None):
    model.train() if train else model.eval()
    tot_loss = tot_iou = 0.0
    n = 0
    phase = "Train" if train else "Val"
    pbar = tqdm(loader, desc=f"[{phase}]", leave=False, dynamic_ncols=True)
    use_amp = device.type == "cuda"

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                preds = model(imgs)
                loss = criterion(preds, targets)

            if train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

            tot_loss += float(loss.item())
            tot_iou += calculate_iou(preds.detach(), targets)
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return tot_loss / max(n, 1), tot_iou / max(n, 1)


def _set_model3_stage(model, epoch, freeze_backbone, head_only_epochs, layer4_until_epoch):
    if not isinstance(model, PretrainedDetector):
        return None
    if freeze_backbone:
        stage = None
    elif epoch <= head_only_epochs:
        stage = None
    elif epoch <= layer4_until_epoch:
        stage = "layer4"
    else:
        stage = "layer3"
    model.set_backbone_trainable(stage)
    return "frozen" if stage is None else f"unfreeze:{stage}+"


def train_pipeline(
    model_name="model1", seed=202601, epochs=80,
    batch_size=None, lr=None, weight_decay=None,
    lambda_box=None, lambda_obj=None, lambda_class=None,
    optimizer_name=None, focal_gamma=None,
    train_dir="data/processed/dataset_202601/train",
    val_dir="data/processed/dataset_202601/val",
    image_size=IMAGE_SIZE, freeze_backbone=False, unfreeze_from_layer="layer3",
    checkpoint_dir="checkpoints", num_workers=20, use_optuna=False,
    warmup_epochs=2, eval_every=2, negative_ratio=1.0,
    head_only_epochs=5, layer4_until_epoch=15,
):
    del unfreeze_from_layer  # staged policy below is the default final protocol
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {}
    hp = os.path.join("outputs", f"best_hparams_{model_name}.json")
    if use_optuna and os.path.exists(hp):
        with open(hp, encoding="utf-8") as f:
            params = json.load(f).get("best_params", {})
        print(f"[train] Load Optuna params: {params}")

    lr = lr if lr is not None else params.get("lr", 1e-4 if model_name == "model3" else 5e-4)
    batch_size = batch_size if batch_size is not None else params.get("batch_size", 8)
    weight_decay = weight_decay if weight_decay is not None else params.get("weight_decay", 1e-4)
    lambda_box = lambda_box if lambda_box is not None else params.get("lambda_box", LAMBDA_BOX)
    lambda_obj = lambda_obj if lambda_obj is not None else params.get("lambda_obj", LAMBDA_OBJ)
    lambda_class = lambda_class if lambda_class is not None else params.get("lambda_class", LAMBDA_CLASS)
    focal_gamma = focal_gamma if focal_gamma is not None else params.get("focal_gamma", FOCAL_GAMMA)
    optimizer_name = optimizer_name or params.get("optimizer", "adamw")

    grid_size = get_grid_size(image_size=image_size)
    print(
        f"[train] {device} | seed={seed} | {model_name} | image={image_size} grid={grid_size} "
        f"slots={BOXES_PER_CELL} | bs={batch_size} lr={lr:.2e} wd={weight_decay:.1e} "
        f"opt={optimizer_name} | box/obj/cls={lambda_box:.2f}/{lambda_obj:.2f}/{lambda_class:.2f}"
    )

    train_ds = VinBigDataDetectionDataset(
        train_dir, image_size, grid_size, BOXES_PER_CELL, augment=True
    )
    val_ds = VinBigDataDetectionDataset(
        val_dir, image_size, grid_size, BOXES_PER_CELL, augment=False
    )

    assignment = train_ds.target_assignment_stats()
    print(
        "[train] Target capacity: "
        f"{assignment['representable_boxes']}/{assignment['total_boxes']} "
        f"({assignment['representable_ratio']*100:.3f}%), overflow={assignment['overflow_boxes']}, "
        f"max same-cell={assignment['max_boxes_same_cell']}"
    )

    class_weights = train_ds.class_weights()
    print(f"[train] Class weights: {[round(float(x), 3) for x in class_weights]}")
    sampler = train_ds.make_balanced_sampler(negative_ratio=negative_ratio)

    pin = device.type == "cuda"
    dl_kw = dict(
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
        worker_init_fn=seed_worker,
    )
    if num_workers > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=4)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
        generator=generator,
        **dl_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **dl_kw
    )

    model = build_model(
        model_name, NUM_CLASSES, freeze_backbone=freeze_backbone,
        unfreeze_from_layer="layer3", boxes_per_cell=BOXES_PER_CELL,
    ).to(device)
    median_w, median_h = train_ds.box_size_prior()
    model.head.initialize_box_prior(median_w, median_h)
    print(f"[train] Box prior median w/h = {median_w:.4f}/{median_h:.4f}")

    if hasattr(model, "trainable_parameter_summary"):
        t, tot = model.trainable_parameter_summary()
        print(f"[train] Initial trainable: {t:,}/{tot:,}")

    criterion = CombinedLocalizationLoss(
        lambda_box=lambda_box,
        lambda_obj=lambda_obj,
        lambda_class=lambda_class,
        focal_gamma=focal_gamma,
        focal_alpha=FOCAL_ALPHA,
        class_weights=class_weights,
    ).to(device)
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.05, total_iters=warmup_epochs
        )
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warm, cos], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=1e-6
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(checkpoint_dir, f"{model_name}_seed{seed}_best.pth")
    best_map = -1.0
    tl, vl, ti, vi = [], [], [], []
    last_stage = None

    for epoch in range(1, epochs + 1):
        stage = _set_model3_stage(
            model, epoch, freeze_backbone, head_only_epochs, layer4_until_epoch
        )
        if stage is not None and stage != last_stage:
            print(f"[train] Model3 stage @ epoch {epoch}: {stage}")
            last_stage = stage

        t0 = time.time()
        tr_loss, tr_iou = run_one_epoch(
            model, train_loader, criterion, optimizer, device, True, scaler
        )
        va_loss, va_iou = run_one_epoch(
            model, val_loader, criterion, optimizer, device, False, scaler
        )

        do_eval = (epoch % eval_every == 0) or epoch == epochs
        det = None
        if do_eval:
            det = evaluate_detections(
                model, val_ds, val_loader, device, NUM_CLASSES,
                conf_threshold=CONF_THRESHOLD,
                min_score=MIN_AP_SCORE,
                max_det=MAX_DET,
            )
        scheduler.step()
        tl.append(tr_loss); vl.append(va_loss); ti.append(tr_iou); vi.append(va_iou)

        msg = (
            f"Epoch [{epoch:02d}/{epochs}] ({time.time()-t0:.0f}s) | "
            f"Train {tr_loss:.3f} IoU {tr_iou:.3f} | Val {va_loss:.3f} IoU {va_iou:.3f}"
        )
        if det is not None:
            msg += (
                f" | mAP50 {det['map50']:.4f} F1 {det['micro_f1']:.3f} "
                f"preds/img {det['mean_preds_per_image']:.1f}"
            )
            if det["map50"] > best_map:
                best_map = det["map50"]
                torch.save({
                    "epoch": epoch,
                    "model_name": model_name,
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "val_loss": va_loss,
                    "val_iou": va_iou,
                    "val_map50": best_map,
                    "num_classes": NUM_CLASSES,
                    "image_size": image_size,
                    "grid_size": grid_size,
                    "boxes_per_cell": BOXES_PER_CELL,
                    "freeze_backbone": freeze_backbone,
                    "class_weights": class_weights.tolist(),
                    "target_assignment_stats": assignment,
                    "hparams": {
                        "lr": lr,
                        "batch_size": batch_size,
                        "weight_decay": weight_decay,
                        "lambda_box": lambda_box,
                        "lambda_obj": lambda_obj,
                        "lambda_class": lambda_class,
                        "focal_gamma": focal_gamma,
                        "optimizer": optimizer_name,
                        "negative_ratio": negative_ratio,
                        "head_only_epochs": head_only_epochs,
                        "layer4_until_epoch": layer4_until_epoch,
                    },
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
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--train_dir", default="data/processed/dataset_202601/train")
    p.add_argument("--val_dir", default="data/processed/dataset_202601/val")
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--negative_ratio", type=float, default=1.0)
    p.add_argument("--no_optuna", action="store_true")
    a = p.parse_args()
    train_pipeline(
        model_name=a.model, seed=a.seed, epochs=a.epochs,
        batch_size=a.batch_size, lr=a.lr,
        train_dir=a.train_dir, val_dir=a.val_dir,
        image_size=a.image_size, freeze_backbone=a.freeze_backbone,
        checkpoint_dir=a.checkpoint_dir, num_workers=a.num_workers,
        use_optuna=not a.no_optuna, eval_every=a.eval_every,
        negative_ratio=a.negative_ratio,
    )
