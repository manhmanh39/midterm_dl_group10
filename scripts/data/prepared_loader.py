"""
scripts/data/prepared_loader.py - Kiem tra, audit cell collisions va cung cap DataLoaders
cho VinBigData Object Detection voi Split Semantics Guard nghiem ngat.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from scripts.config import (
    IMAGE_SIZE,
    STRIDE,
    get_grid_size,
    get_processed_data_root,
)
from scripts.experiment_config import assert_test_access_allowed
from scripts.src.dataset import VinBigDataDetectionDataset, collate_fn


def is_prepared_dataset(root: str | Path) -> bool:
    """Kiem tra dataset da prepared hop le: co du 3 split, moi split co images/."""
    root = Path(root)
    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        return False
    for split in ("train", "val", "test"):
        img_dir = root / split / "images"
        if not img_dir.is_dir() or not any(img_dir.glob("*.png")):
            return False
    return True


def describe_split(root: str | Path, split: str) -> Dict[str, float]:
    """
    Dem nhanh so anh, so box, ty le anh No finding (0 box) trong 1 split.
    Khong doc noi dung anh - chi doc file .txt nhan de dem dong.
    """
    root = Path(root)
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"

    image_files = sorted(img_dir.glob("*.png"))
    n_images = len(image_files)
    n_boxes = 0
    n_no_finding = 0

    for img_path in image_files:
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            n_no_finding += 1
            continue
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            n_no_finding += 1
            continue
        n_lines = len(text.splitlines())
        n_boxes += n_lines

    return {
        "n_images": n_images,
        "n_boxes": n_boxes,
        "n_no_finding": n_no_finding,
        "avg_boxes_per_image": n_boxes / n_images if n_images else 0.0,
        "pct_no_finding": n_no_finding / n_images * 100 if n_images else 0.0,
    }


def describe_dataset(
    root: str | Path,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, Dict[str, float]]:
    """
    Mo ta cac split duoc yeu cau. Trong giai doan Develop, chi duoc phep
    truyen splits=('train', 'val') de tranh leakage thong ke tap test truoc freeze.
    """
    result = {}
    for split in splits:
        result[split] = describe_split(root, split)
    return result


def audit_cell_collisions(dataset: VinBigDataDetectionDataset, grid_size: Optional[int] = None) -> Dict[str, float]:
    """
    Audit va cham toa do bounding box tren luoi YOLO GxG (16x16).
    Dem so box cung roi vao 1 cell trong cung 1 anh, dan den viec chi co box lon hon duoc giu lai.
    """
    G = grid_size or dataset.grid_size
    total_images = len(dataset)
    total_boxes = 0
    collided_boxes = 0
    images_with_collision = 0
    max_boxes_in_cell = 0

    for idx in range(total_images):
        boxes = dataset.get_raw_boxes(idx)
        if not boxes:
            continue
        total_boxes += len(boxes)
        cell_counts: Dict[Tuple[int, int], int] = {}
        for cls_id, cx, cy, w, h in boxes:
            gx = min(max(int(cx * G), 0), G - 1)
            gy = min(max(int(cy * G), 0), G - 1)
            cell_counts[(gy, gx)] = cell_counts.get((gy, gx), 0) + 1

        img_collisions = 0
        for (gy, gx), cnt in cell_counts.items():
            if cnt > max_boxes_in_cell:
                max_boxes_in_cell = cnt
            if cnt > 1:
                img_collisions += (cnt - 1)

        if img_collisions > 0:
            images_with_collision += 1
            collided_boxes += img_collisions

    collision_rate_boxes = (collided_boxes / total_boxes) if total_boxes > 0 else 0.0
    collision_rate_images = (images_with_collision / total_images) if total_images > 0 else 0.0

    return {
        "grid_size": G,
        "total_images": total_images,
        "total_boxes": total_boxes,
        "collided_boxes": collided_boxes,
        "images_with_collision": images_with_collision,
        "collision_rate_boxes": float(collision_rate_boxes),
        "collision_rate_images": float(collision_rate_images),
        "max_boxes_in_cell": max_boxes_in_cell,
    }


def get_develop_dataloaders(
    data_root: str | Path,
    batch_size: int = 16,
    image_size: int = IMAGE_SIZE,
    num_workers: int = 4,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, VinBigDataDetectionDataset, VinBigDataDetectionDataset]:
    """
    Tra ve TrainLoader va ValLoader cho giai doan Develop.
    CAM TUYET DOI dong cham hoac tao TestLoader o ham nay.
    """
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Khong tim thay thu muc train/val trong {data_root}")

    grid_size = get_grid_size(image_size=image_size, stride=STRIDE)
    train_ds = VinBigDataDetectionDataset(str(train_dir), image_size=image_size, grid_size=grid_size, augment=augment)
    val_ds = VinBigDataDetectionDataset(str(val_dir), image_size=image_size, grid_size=grid_size, augment=False)

    pin = torch.cuda.is_available()
    dl_kw = dict(num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin)
    if num_workers > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kw)

    return train_loader, val_loader, train_ds, val_ds


def get_test_dataloader(
    data_root: str | Path,
    batch_size: int = 16,
    image_size: int = IMAGE_SIZE,
    num_workers: int = 4,
    lock_token: Optional[Union[str, object]] = None,
    lock_path: str | Path = "outputs/protocol_lock.json",
) -> Tuple[DataLoader, VinBigDataDetectionDataset]:
    """
    Split Semantics Guard:
    Chi mo TestLoader khi cung cap protocol_lock_token/PreflightPermit hop le tu global_preflight_check().
    Dong thoi tai kiem tra dataset fingerprint tren dia de ngan ngua thay doi muon.
    """
    assert_test_access_allowed(lock_token, lock_path=lock_path, dataset_root=data_root)

    root = Path(data_root)
    test_dir = root / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Khong tim thay thu muc test trong {data_root}")

    grid_size = get_grid_size(image_size=image_size, stride=STRIDE)
    test_ds = VinBigDataDetectionDataset(str(test_dir), image_size=image_size, grid_size=grid_size, augment=False)

    pin = torch.cuda.is_available()
    dl_kw = dict(num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin)
    if num_workers > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=2)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dl_kw)
    return test_loader, test_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiem tra va audit dataset detection da prepared")
    parser.add_argument("--root", type=str, default=get_processed_data_root())
    parser.add_argument("--audit_collision", action="store_true", help="Chay audit cell collisions")
    args = parser.parse_args()

    if not is_prepared_dataset(args.root):
        print(f"'{args.root}' khong phai dataset da prepared hop le "
              f"(thieu dataset.yaml hoac thieu anh .png trong 1 trong 3 split).")
    else:
        stats = describe_dataset(args.root)
        print(f"Dataset tai: {args.root}\n")
        for split, s in stats.items():
            print(
                f"  [{split:<5}] {s['n_images']:>5} anh | {s['n_boxes']:>6} box | "
                f"No finding: {s['n_no_finding']:>5} ({s['pct_no_finding']:.1f}%) | "
                f"TB box/anh: {s['avg_boxes_per_image']:.2f}"
            )

        if args.audit_collision:
            print("\nDang audit cell collisions tren tap Train...")
            grid_size = get_grid_size(IMAGE_SIZE, STRIDE)
            train_ds = VinBigDataDetectionDataset(str(Path(args.root) / "train"), IMAGE_SIZE, grid_size, augment=False)
            aud = audit_cell_collisions(train_ds)
            print(f"  Grid: {aud['grid_size']}x{aud['grid_size']}")
            print(f"  Tong so box: {aud['total_boxes']}")
            print(f"  So box bi va cham: {aud['collided_boxes']} ({aud['collision_rate_boxes']*100:.2f}%)")
            print(f"  So anh co va cham: {aud['images_with_collision']} ({aud['collision_rate_images']*100:.2f}%)")
            print(f"  Max box/cell: {aud['max_boxes_in_cell']}")