"""
Cau hinh chung cho pipeline OBJECT DETECTION (anchor-free, 1-scale, kieu YOLO don gian).

Bai toan: VinBigData VinDr-CXR - phat hien nhieu bat thuong/anh (multi-object
detection), khong con la classification 1-bbox/anh nhu ban truoc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 14 lop bat thuong (class_id 0-13). "No finding" KHONG nam trong detection
# head (anh No finding don gian la anh khong co bbox nao ca - negative image).
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]
NUM_CLASSES = len(CLASS_NAMES)  # 14

# ---------------------------------------------------------------------------
# Tham so detection head (anchor-free, 1-scale)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 512          # anh dau vao vuong, chia het cho STRIDE
STRIDE = 32               # tong stride cua backbone (4 lan pool x2 = /16,
                          # + 1 lan pool trong stem = /32 -> grid = IMAGE_SIZE/32)
GRID_SIZE = IMAGE_SIZE // STRIDE   # so cell moi chieu, vd 512/32 = 16 -> 16x16

# Nguong khi decode prediction -> box that (dung luc inference/evaluate)
CONF_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45

# He so trong so cac thanh phan loss (theo tinh than YOLOv1/v3: box loss
# nhan trong so lon hon vi so luong cell "co object" it hon nhieu so voi
# cell "khong co object", can can bang lai anh huong cua tung phan)
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5
LAMBDA_OBJ = 1.0
LAMBDA_CLASS = 1.0


def get_grid_size(image_size: int = IMAGE_SIZE, stride: int = STRIDE) -> int:
    if image_size % stride != 0:
        raise ValueError(
            f"image_size ({image_size}) phai chia het cho stride ({stride}) "
            "de dam bao kich thuoc grid la so nguyen."
        )
    return image_size // stride


def get_data_root() -> str:
    return os.environ.get("VINBIGDATA_ROOT", "data/raw")


def get_processed_data_root() -> str:
    return os.environ.get("VINBIGDATA_PROCESSED", "data/processed/prepared_dataset")


@dataclass
class DataConfig:
    root: str = field(default_factory=get_data_root)
    processed_root: str = field(default_factory=get_processed_data_root)
    image_size: int = IMAGE_SIZE
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    include_no_finding: bool = True   # giu anh No finding lam negative sample
    wbf_iou_thr: float = 0.3

    @property
    def train_csv(self) -> Path:
        return Path(self.root) / "train.csv"

    @property
    def train_img_dir(self) -> Path:
        return Path(self.root) / "train"