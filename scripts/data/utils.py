"""DICOM -> anh 3-kenh PNG; train.csv -> nhan YOLO (nhieu dong/anh)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

NO_FINDING_LABEL = "No finding"


def _apply_voi_lut_and_invert(dicom_obj) -> np.ndarray:
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    arr = apply_voi_lut(dicom_obj.pixel_array, dicom_obj)
    if str(getattr(dicom_obj, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = np.amax(arr) - arr
    return arr.astype(np.float32)


def _normalize_to_8bit(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val
    return (arr * 255.0).astype(np.uint8)


def dicom_to_3channel_8bit(dicom_path, image_size: int = 512) -> np.ndarray:
    import pydicom

    dcm = pydicom.dcmread(str(dicom_path))
    arr8 = _normalize_to_8bit(_apply_voi_lut_and_invert(dcm))
    arr8 = cv2.resize(arr8, (image_size, image_size), interpolation=cv2.INTER_AREA)
    ch_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(arr8)
    lap = _normalize_to_8bit(np.absolute(cv2.Laplacian(ch_clahe, cv2.CV_64F, ksize=3)))
    return np.stack([arr8, ch_clahe, lap], axis=-1)


def load_annotations(train_csv) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    ann_df = df[df["class_name"] != NO_FINDING_LABEL].copy()
    for col in ["x_min", "y_min", "x_max", "y_max"]:
        ann_df[col] = pd.to_numeric(ann_df[col], errors="coerce")
    return ann_df.dropna(subset=["x_min", "y_min", "x_max", "y_max"])


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
    """Gop box CUNG class co IoU > iou_thr thanh 1 box (trung binh toa do)."""
    if not boxes:
        return []
    boxes = list(boxes)
    used = [False] * len(boxes)
    fused = []
    for i in range(len(boxes)):
        if used[i]:
            continue
        cluster = [boxes[i]]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if not used[j] and _iou_xyxy(boxes[i], boxes[j]) > iou_thr:
                cluster.append(boxes[j])
                used[j] = True
        fused.append(tuple(np.array(cluster, dtype=np.float64).mean(axis=0)))
    return fused


def build_yolo_dataset(
    ann_df: pd.DataFrame,
    image_ids: list,
    img_dir,
    label_dir,
    sizes: dict,               # {image_id: (orig_w, orig_h)} lay tu DICOM
    iou_thr: float = 0.3,
    image_size: int = 512,
):
    """FIX BUG GOC: train.csv khong co orig_width/height -> phai scale bang kich thuoc DICOM that."""
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
                        f"{int(class_id)} {(x1 + x2) / 2 / image_size:.6f} {(y1 + y2) / 2 / image_size:.6f} "
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