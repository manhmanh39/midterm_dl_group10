from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional

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
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"[FAIL CLOSED] Không tìm thấy manifest.json tại: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
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
    strict: bool = True,
) -> Dict[str, np.ndarray]:
    csv_path = root / split / f"{split}_boxes.csv"
    if not csv_path.exists():
        if strict:
            raise FileNotFoundError(f"[FAIL CLOSED] Không tìm thấy file annotations bắt buộc: {csv_path}")
        print(f"[prepared_loader] Không tìm thấy {csv_path} — toàn bộ '{split}' sẽ mang nhãn No finding.")
        return {img_id: _no_finding_vector(num_classes) for img_id in image_ids}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        if strict:
            raise ValueError(f"[FAIL CLOSED] Lỗi đọc CSV annotations tại {csv_path}: {e}") from e
        return {img_id: _no_finding_vector(num_classes) for img_id in image_ids}

    if not df.empty:
        required_cols = {"image_id", "class_id"}
        if not required_cols.issubset(df.columns):
            if strict:
                raise ValueError(f"[FAIL CLOSED] File {csv_path} thiếu cột bắt buộc ({required_cols - set(df.columns)})")

        for cid in df["class_id"].dropna():
            try:
                icid = int(cid)
            except (ValueError, TypeError) as e:
                if strict:
                    raise ValueError(f"[FAIL CLOSED] Class_id không thể ép kiểu int: {cid}") from e
                continue
            if icid < 0 or icid >= num_classes:
                if strict:
                    raise ValueError(f"[FAIL CLOSED] File {csv_path} chứa class_id không hợp lệ: {icid} (num_classes={num_classes})")

    labels: Dict[str, np.ndarray] = {
        img_id: _no_finding_vector(num_classes) for img_id in image_ids
    }

    if df.empty:
        return labels

    for image_id, group in df.groupby("image_id"):
        image_id_str = str(image_id)
        if image_id_str not in labels:
            continue
        vec = np.zeros(num_classes, dtype=np.float32)
        for cls_id in group["class_id"].unique():
            cls_id = int(cls_id)
            if 0 <= cls_id < NO_FINDING_CLASS_ID:
                vec[cls_id] = 1.0
        if vec.sum() == 0:
            vec = _no_finding_vector(num_classes)
        labels[image_id_str] = vec

    return labels


def build_prepared_samples(
    root: str | Path,
    splits: Sequence[str] = ("train", "val"),
    num_classes: int = 15,
    strict: bool = True,
) -> Tuple[List[Tuple[str, np.ndarray]], ...]:
    root = Path(root)
    manifest = _load_manifest(root)

    results = []
    for split in splits:
        raw_ids = manifest.get(split, [])
        if isinstance(raw_ids, list) and raw_ids:
            image_ids = raw_ids
        elif isinstance(raw_ids, dict):
            image_ids = raw_ids.get("image_ids", [])
        elif "splits" in manifest and split in manifest["splits"]:
            image_ids = manifest["splits"][split].get("image_ids", [])
        else:
            image_ids = []

        if not image_ids and (root / split / "images").exists():
            image_ids = [p.stem for p in (root / split / "images").glob("*.png")]

        if strict and not image_ids:
            raise ValueError(f"[FAIL CLOSED] Manifest không chứa image_ids nào cho split '{split}'!")

        img_dir = root / split / "images"
        if strict and not img_dir.exists():
            raise FileNotFoundError(f"[FAIL CLOSED] Thư mục ảnh của split '{split}' không tồn tại: {img_dir}")

        label_map = _build_split_labels(root, split, image_ids, num_classes=num_classes, strict=strict)

        samples: List[Tuple[str, np.ndarray]] = []
        missing = 0
        for img_id in image_ids:
            img_path = img_dir / f"{img_id}.png"
            if not img_path.exists():
                missing += 1
                if strict:
                    raise FileNotFoundError(f"[FAIL CLOSED] Thiếu file ảnh cho ID '{img_id}' tại: {img_path}")
                continue
            samples.append((str(img_path), label_map[img_id]))

        if missing and not strict:
            pct = missing / max(len(image_ids), 1) * 100
            print(
                f"[prepared_loader] CẢNH BÁO {split}: {missing}/{len(image_ids)} ảnh "
                f"({pct:.1f}%) có trong manifest nhưng KHÔNG tìm thấy file PNG."
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
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args()

    if not is_prepared_dataset(args.root):
        print(f"'{args.root}' không phải dataset đã prepared hợp lệ.")
    else:
        train_s, val_s, test_s = build_prepared_samples(
            args.root, splits=("train", "val", "test"), num_classes=config.NUM_CLASSES, strict=args.strict
        )
        print(f"\nVí dụ mẫu train đầu tiên: {train_s[0][0]}")
        print(f"Label: {train_s[0][1]}")