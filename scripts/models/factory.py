"""Factory chung cho train/evaluate/HPO."""
from __future__ import annotations

import torch

from scripts.config import BOXES_PER_CELL
from scripts.models.model1_sequential import SequentialCNNDetector
from scripts.models.model2_residual import NonSequentialCNNDetector
from scripts.models.model3_pretrained import PretrainedDetector


def build_model(
    model_name,
    num_classes,
    freeze_backbone=False,
    unfreeze_from_layer="layer3",
    boxes_per_cell=BOXES_PER_CELL,
):
    if model_name == "model1":
        return SequentialCNNDetector(num_classes, boxes_per_cell)
    if model_name == "model2":
        return NonSequentialCNNDetector(num_classes, boxes_per_cell)
    if model_name == "model3":
        return PretrainedDetector(num_classes, freeze_backbone, unfreeze_from_layer, boxes_per_cell)
    raise ValueError(f"Model khong hop le: {model_name}")


def make_optimizer(model, name, lr, weight_decay, backbone_lr_mult=0.1):
    name = name.lower()
    if isinstance(model, PretrainedDetector):
        bb_ids = {id(p) for p in model.backbone_parameters()}
        bb = [p for p in model.parameters() if id(p) in bb_ids]
        other = [p for p in model.parameters() if id(p) not in bb_ids]
        # Include frozen params so staged unfreeze later automatically joins optimizer.
        groups = [
            {"params": other, "lr": lr},
            {"params": bb, "lr": lr * backbone_lr_mult},
        ]
    else:
        groups = [{"params": list(model.parameters()), "lr": lr}]

    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    if name in {"adam", "adamw"}:
        return torch.optim.AdamW(groups, weight_decay=weight_decay)
    raise ValueError(f"Optimizer khong ho tro: {name}")
