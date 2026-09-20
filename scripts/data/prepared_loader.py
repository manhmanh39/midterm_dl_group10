"""
prepared_loader.py - Kiem tra va mo ta nhanh mot dataset da duoc chuan bi
(processed) boi scripts/data/prepare_dataset.py, dung cho object detection.

Khac voi ban classification cu (khong con tinh 1 vector nhan/anh hay
pos_weight cho BCE theo class); o day dataset la thu muc images/ + labels/
(YOLO format, nhieu dong/anh), nen ham chinh o day chi:
    - kiem tra cau truc thu muc co hop le khong (is_prepared_dataset)
    - dem so anh / so box / ty le anh No finding cho tung split, de nguoi
      dung nhanh chong nam duoc dataset truoc khi train (khong lam nhiem
      vu cua Dataset/DataLoader - viec do da giao cho
      scripts/src/dataset.py::VinBigDataDetectionDataset)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from scripts.config import get_processed_data_root


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
        text = label_path.read_text().strip()
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


def describe_dataset(root: str | Path) -> Dict[str, Dict[str, float]]:
    """Mo ta ca 3 split (train/val/test) cua 1 dataset da prepared."""
    result = {}
    for split in ("train", "val", "test"):
        result[split] = describe_split(root, split)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiem tra nhanh dataset detection da prepared")
    parser.add_argument("--root", type=str, default=get_processed_data_root())
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