from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import config

NO_FINDING_CLASS_ID = config.NO_FINDING_CLASS_ID


def is_prepared_dataset(root: str | Path) -> bool:
    root = Path(root)
    manifest = root / "manifest.json"
    train_imgs = root / "train" / "images"
    return manifest.exists() and train_imgs.is_dir() and any(train_imgs.glob("*.png"))


def _load_manifest(root: Path) -> dict:
    with open(root / "manifest.json") as f:
        return json.load(f)


def _no_finding_vector(num_classes: int) -> np.ndarray:
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[NO_FINDING_CLASS_ID] = 1.0
    return vec


def _build_split_labels(
    root: Path,
    split: str,
    image_ids: List[str],
    num_classes: int = 15,
) -> Dict[str, np.ndarray]:
    labels: Dict[str, np.ndarray] = {
        img_id: _no_finding_vector(num_classes) for img_id in image_ids
    }

    csv_path = root / split / f"{split}_boxes.csv"
    if not csv_path.exists():
        print(f"[prepared_loader] Không tìm thấy {csv_path} — toàn bộ '{split}' sẽ mang nhãn No finding.")
        return labels

    df = pd.read_csv(csv_path)
    if df.empty:
        return labels

    for image_id, group in df.groupby("image_id"):
        if image_id not in labels:
            continue
        vec = np.zeros(num_classes, dtype=np.float32)
        for cls_id in group["class_id"].unique():
            cls_id = int(cls_id)
            if 0 <= cls_id < NO_FINDING_CLASS_ID:
                vec[cls_id] = 1.0
        if vec.sum() == 0:
            vec = _no_finding_vector(num_classes)
        labels[image_id] = vec

    return labels


def build_prepared_samples(
    root: str | Path,
    num_classes: int = 15,
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]:
    root = Path(root)
    manifest = _load_manifest(root)

    results = []
    for split in ["train", "val", "test"]:
        image_ids = manifest.get(split, [])
        img_dir = root / split / "images"

        label_map = _build_split_labels(root, split, image_ids, num_classes=num_classes)

        samples: List[Tuple[str, np.ndarray]] = []
        missing = 0
        for img_id in image_ids:
            img_path = img_dir / f"{img_id}.png"
            if not img_path.exists():
                missing += 1
                continue
            samples.append((str(img_path), label_map[img_id]))

        if missing:
            pct = missing / max(len(image_ids), 1) * 100
            print(
                f"[prepared_loader] CẢNH BÁO {split}: {missing}/{len(image_ids)} ảnh "
                f"({pct:.1f}%) có trong manifest nhưng KHÔNG tìm thấy file PNG "
                f"— kiểm tra lại bước tải/giải nén nếu tỷ lệ này cao bất thường."
            )

        n_no_finding = sum(1 for _, lbl in samples if lbl[NO_FINDING_CLASS_ID] == 1.0)
        n_abnormal = len(samples) - n_no_finding
        pct_nf = n_no_finding / max(len(samples), 1) * 100
        print(
            f"[prepared_loader] {split}: {len(samples)} ảnh hợp lệ  "
            f"({n_abnormal} có bất thường, {n_no_finding} No finding = {pct_nf:.1f}%)"
        )

        results.append(samples)

    return tuple(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kiểm thử prepared_loader")
    parser.add_argument("--root", type=str, default="data/processed")
    args = parser.parse_args()

    if not is_prepared_dataset(args.root):
        print(f"'{args.root}' không phải dataset đã prepared hợp lệ.")
    else:
        train_s, val_s, test_s = build_prepared_samples(args.root, num_classes=config.NUM_CLASSES)
        print(f"\nVí dụ mẫu train đầu tiên: {train_s[0][0]}")
        print(f"Label: {train_s[0][1]}")