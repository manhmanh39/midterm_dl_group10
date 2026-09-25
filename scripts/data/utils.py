"""DICOM -> anh 3-kenh PNG; train.csv -> nhan YOLO detection."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

NO_FINDING_LABEL = "No finding"


def _apply_voi_lut_and_invert(dicom_obj) -> np.ndarray:
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    arr = apply_voi_lut(dicom_obj.pixel_array, dicom_obj).astype(np.float32)
    if str(getattr(dicom_obj, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = np.amax(arr) - arr
    return arr


def _normalize_to_8bit(arr: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """Robust percentile normalization, it nhay voi extreme pixels hon min-max."""
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    vals = arr[finite]
    lo, hi = np.percentile(vals, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
    arr = np.clip(arr, lo, hi)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return np.round(arr * 255.0).astype(np.uint8)


def dicom_to_3channel_8bit(dicom_path, image_size: int = 512) -> np.ndarray:
    """3 channels dung chung cho ca 3 model: raw, CLAHE, Laplacian."""
    import pydicom

    dcm = pydicom.dcmread(str(dicom_path))
    raw = _normalize_to_8bit(_apply_voi_lut_and_invert(dcm))
    raw = cv2.resize(raw, (image_size, image_size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(raw)
    lap = cv2.Laplacian(clahe, cv2.CV_32F, ksize=3)
    lap = _normalize_to_8bit(np.abs(lap), low_pct=0.0, high_pct=99.5)
    return np.stack([raw, clahe, lap], axis=-1)


def load_annotations(train_csv) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    ann_df = df[df["class_name"] != NO_FINDING_LABEL].copy()
    for col in ["x_min", "y_min", "x_max", "y_max"]:
        ann_df[col] = pd.to_numeric(ann_df[col], errors="coerce")
    ann_df = ann_df.dropna(subset=["x_min", "y_min", "x_max", "y_max"])
    return ann_df[(ann_df["x_max"] > ann_df["x_min"]) & (ann_df["y_max"] > ann_df["y_min"])]


def _iou_xyxy(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    iw = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    ih = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _weighted_box_fusion(boxes, iou_thr: float):
    """Equal-weight WBF-like fusion cho annotation cung class.

    VinBigData co nhieu radiologist; khong co confidence score tin cay de lam weight,
    nen moi box duoc gan weight=1. Cluster cap nhat theo fused box hien tai thay vi
    chi so voi box dau tien.
    """
    if not boxes:
        return []

    clusters: list[list[np.ndarray]] = []
    fused_boxes: list[np.ndarray] = []

    # Box lon truoc giup cluster on dinh hon khi annotation gan nhau.
    ordered = sorted(
        [np.asarray(b, dtype=np.float64) for b in boxes],
        key=lambda b: -max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]),
    )

    for box in ordered:
        best_idx, best_iou = -1, -1.0
        for i, fused in enumerate(fused_boxes):
            iou = _iou_xyxy(box, fused)
            if iou > best_iou:
                best_idx, best_iou = i, iou
        if best_idx >= 0 and best_iou >= iou_thr:
            clusters[best_idx].append(box)
            fused_boxes[best_idx] = np.mean(np.stack(clusters[best_idx], axis=0), axis=0)
        else:
            clusters.append([box])
            fused_boxes.append(box.copy())

    return [tuple(b.tolist()) for b in fused_boxes]


def build_yolo_dataset(
    ann_df: pd.DataFrame,
    image_ids: list,
    img_dir,
    label_dir,
    sizes: dict,
    iou_thr: float = 0.35,
    image_size: int = 512,
):
    """Tao label YOLO normalized tu annotation pixel goc."""
    del img_dir  # kept in API for backward compatibility
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    grouped = ann_df.groupby("image_id")

    for img_id in image_ids:
        lines = []
        if img_id in grouped.groups:
            rows = grouped.get_group(img_id)
            ow, oh = sizes[img_id]
            sx, sy = image_size / float(ow), image_size / float(oh)

            for class_id, class_rows in rows.groupby("class_id"):
                boxes = [
                    (float(r["x_min"]) * sx, float(r["y_min"]) * sy,
                     float(r["x_max"]) * sx, float(r["y_max"]) * sy)
                    for _, r in class_rows.iterrows()
                ]
                for x1, y1, x2, y2 in _weighted_box_fusion(boxes, iou_thr=iou_thr):
                    x1, x2 = np.clip([x1, x2], 0, image_size)
                    y1, y2 = np.clip([y1, y2], 0, image_size)
                    w, h = x2 - x1, y2 - y1
                    if w <= 1 or h <= 1:
                        continue
                    lines.append(
                        f"{int(class_id)} {(x1 + x2) / 2 / image_size:.6f} "
                        f"{(y1 + y2) / 2 / image_size:.6f} "
                        f"{w / image_size:.6f} {h / image_size:.6f}"
                    )
        (label_dir / f"{img_id}.txt").write_text("\n".join(lines))


def write_yolo_yaml(yaml_path, dataset_path, train_img_dir, val_img_dir, test_img_dir, class_names):
    lines = [
        f"path: {dataset_path}", f"train: {train_img_dir}", f"val: {val_img_dir}",
        f"test: {test_img_dir}", "", f"nc: {len(class_names)}", "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    Path(yaml_path).write_text("\n".join(lines) + "\n")
