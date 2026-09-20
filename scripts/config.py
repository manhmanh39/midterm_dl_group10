"""Cau hinh chung cho pipeline OBJECT DETECTION (anchor-free, 1-scale)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 512
STRIDE = 32
GRID_SIZE = IMAGE_SIZE // STRIDE

# score = objectness * class_prob (softmax) nen nguong phai thap
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45

LAMBDA_COORD = 5.0
LAMBDA_OBJ = 1.0
LAMBDA_CLASS = 1.0


def get_grid_size(image_size: int = IMAGE_SIZE, stride: int = STRIDE) -> int:
    if image_size % stride != 0:
        raise ValueError(f"image_size ({image_size}) phai chia het cho stride ({stride})")
    return image_size // stride


def get_data_root() -> str:
    return os.environ.get("VINBIGDATA_ROOT", "data/raw")


def get_processed_data_root() -> str:
    return os.environ.get("VINBIGDATA_PROCESSED", "data/processed/dataset_202601")


@dataclass
class DataConfig:
    root: str = field(default_factory=get_data_root)
    processed_root: str = field(default_factory=get_processed_data_root)
    image_size: int = IMAGE_SIZE
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    include_no_finding: bool = True
    wbf_iou_thr: float = 0.3

    @property
    def train_csv(self) -> Path:
        return Path(self.root) / "train.csv"

    @property
    def train_img_dir(self) -> Path:
        return Path(self.root) / "train"