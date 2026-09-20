"""build_model + make_optimizer dung chung cho train / evaluate / optuna."""
import torch

from scripts.models.model1_sequential import SequentialCNNDetector
from scripts.models.model2_residual import NonSequentialCNNDetector
from scripts.models.model3_pretrained import PretrainedDetector


def build_model(model_name, num_classes, freeze_backbone=False, unfreeze_from_layer="layer3"):
    if model_name == "model1":
        return SequentialCNNDetector(num_classes)
    if model_name == "model2":
        return NonSequentialCNNDetector(num_classes)
    if model_name == "model3":
        return PretrainedDetector(num_classes, freeze_backbone, unfreeze_from_layer)
    raise ValueError(f"Model khong hop le: {model_name}")


def make_optimizer(model, name, lr, weight_decay, backbone_lr_mult=0.1):
    pre = isinstance(model, PretrainedDetector)
    bb, other = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (bb if pre and n.startswith("backbone.") else other).append(p)
    groups = [{"params": other, "lr": lr}]
    if bb:
        groups.append({"params": bb, "lr": lr * backbone_lr_mult})
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)