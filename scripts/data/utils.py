"""
Ham xu ly du lieu cho prepare_dataset.py:
    DICOM -> anh 3-kenh (raw, CLAHE, Laplacian) -> PNG
    train.csv -> gop bbox trung lap (WBF) -> ghi nhan YOLO format (nhieu dong/anh)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

NO_FINDING_LABEL = "No finding"


# ---------------------------------------------------------------------------
# Anh: DICOM -> 3-channel 8-bit
# ---------------------------------------------------------------------------

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


def dicom_to_3channel_8bit(dicom_path: str | Path, image_size: int = 512) -> np.ndarray:
    """Tra ve anh 3-kenh (H, W, 3): [raw, CLAHE, Laplacian]."""
    import pydicom

    dcm = pydicom.dcmread(str(dicom_path))
    arr = _apply_voi_lut_and_invert(dcm)
    arr8 = _normalize_to_8bit(arr)
    arr8 = cv2.resize(arr8, (image_size, image_size), interpolation=cv2.INTER_AREA)

    ch_raw = arr8

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch_clahe = clahe.apply(arr8)

    laplacian = cv2.Laplacian(ch_clahe, cv2.CV_64F, ksize=3)
    laplacian = np.absolute(laplacian)
    ch_laplacian = _normalize_to_8bit(laplacian)

    return np.stack([ch_raw, ch_clahe, ch_laplacian], axis=-1)


# ---------------------------------------------------------------------------
# Nhan: doc annotation, gop bbox bang WBF, ghi YOLO format (NHIEU dong/anh)
# ---------------------------------------------------------------------------

def load_annotations(train_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    ann_df = df[df["class_name"] != NO_FINDING_LABEL].copy()
    for col in ["x_min", "y_min", "x_max", "y_max"]:
        ann_df[col] = pd.to_numeric(ann_df[col], errors="coerce")
    return ann_df.dropna(subset=["x_min", "y_min", "x_max", "y_max"])


def _iou_xyxy(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _weighted_box_fusion(boxes, iou_thr: float):
    """Gop cac box CUNG class co IoU > iou_thr thanh 1 box (trung binh toa do)."""
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
            if used[j]:
                continue
            if _iou_xyxy(boxes[i], boxes[j]) > iou_thr:
                cluster.append(boxes[j])
                used[j] = True
        arr = np.array(cluster, dtype=np.float64)
        fused.append(tuple(arr.mean(axis=0)))

    return fused


def build_yolo_dataset(
    ann_df: pd.DataFrame,
    image_ids: list[str],
    img_dir: str | Path,
    label_dir: str | Path,
    iou_thr: float = 0.3,
    image_size: int = 512,
):
    """
    Voi moi image_id: gop bbox cung class bang WBF, ghi TOAN BO cac box
    con lai (co the nhieu dong) vao file .txt YOLO-format - GIU NGUYEN
    tinh chat multi-object, khong rut gon ve 1 box/anh nhu ban classification.
    """
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    has_orig_dims = {"orig_width", "orig_height"}.issubset(ann_df.columns)
    grouped = ann_df.groupby("image_id")

    for img_id in image_ids:
        lines = []

        if img_id in grouped.groups:
            rows = grouped.get_group(img_id)

            if has_orig_dims:
                ow = float(rows["orig_width"].iloc[0])
                oh = float(rows["orig_height"].iloc[0])
            else:
                ow = oh = float(image_size)

            scale_x = image_size / ow
            scale_y = image_size / oh

            for class_id, class_rows in rows.groupby("class_id"):
                boxes = []
                for _, r in class_rows.iterrows():
                    x1 = float(r["x_min"]) * scale_x
                    y1 = float(r["y_min"]) * scale_y
                    x2 = float(r["x_max"]) * scale_x
                    y2 = float(r["y_max"]) * scale_y
                    boxes.append((x1, y1, x2, y2))

                fused_boxes = _weighted_box_fusion(boxes, iou_thr=iou_thr)

                for x1, y1, x2, y2 in fused_boxes:
                    x1, x2 = np.clip([x1, x2], 0, image_size)
                    y1, y2 = np.clip([y1, y2], 0, image_size)
                    w = x2 - x1
                    h = y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                    cx = (x1 + x2) / 2 / image_size
                    cy = (y1 + y2) / 2 / image_size
                    nw = w / image_size
                    nh = h / image_size
                    lines.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # Neu img_id khong co annotation nao (anh "No finding"), ghi file
        # rong -> dataset se hieu la anh negative (0 box), dung de train
        # objectness=0 tren toan bo grid, giup giam false positive.
        label_path = label_dir / f"{img_id}.txt"
        label_path.write_text("\n".join(lines))


def write_yolo_yaml(yaml_path, dataset_path, train_img_dir, val_img_dir, test_img_dir, class_names):
    yaml_path = Path(yaml_path)
    lines = [
        f"path: {dataset_path}",
        f"train: {train_img_dir}",
        f"val: {val_img_dir}",
        f"test: {test_img_dir}",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n")