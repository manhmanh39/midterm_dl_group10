"""Cau hinh chung cho pipeline OBJECT DETECTION.

Hard constraints cua bai:
- model1: simple sequential CNN, train from scratch
- model2: complex CNN co nhanh song song, train from scratch
- model3: pretrained/base network + transfer learning/fine-tuning

Detection formulation dung chung de so sanh cong bang:
- input 512x512
- stride 16 -> grid 32x32
- 3 prediction slots/cell de giam collision target
"""
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
STRIDE = 16
GRID_SIZE = IMAGE_SIZE // STRIDE
BOXES_PER_CELL = 3

# score = objectness * class probability
CONF_THRESHOLD = 0.10
NMS_IOU_THRESHOLD = 0.45
MIN_AP_SCORE = 0.001
MAX_DET = 300

LAMBDA_BOX = 5.0
LAMBDA_OBJ = 1.0
LAMBDA_CLASS = 1.0
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.02

# Backward-compatible alias for old code/checkpoints.
LAMBDA_COORD = LAMBDA_BOX


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
    seed: int = 202601
    include_no_finding: bool = True
    wbf_iou_thr: float = 0.35
    overwrite_images: bool = False

    @property
    def train_csv(self) -> Path:
        return Path(self.root) / "train.csv"

    @property
    def train_img_dir(self) -> Path:
        return Path(self.root) / "train"
