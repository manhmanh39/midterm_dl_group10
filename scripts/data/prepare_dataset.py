"""Chuan bi VinBigData DICOM -> PNG + YOLO labels + multilabel-stratified split."""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pydicom
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm

from scripts.config import CLASS_NAMES, NUM_CLASSES, DataConfig
from scripts.data.utils import (
    _apply_voi_lut_and_invert,
    _normalize_to_8bit,
    build_yolo_dataset,
    load_annotations,
    write_yolo_yaml,
)


def process_single_image_mp(args_tuple):
    """Return (img_id, orig_w, orig_h); None dimensions neu loi."""
    img_id, zip_path, img_out_dir, image_size, overwrite = args_tuple
    out_path = Path(img_out_dir) / f"{img_id}.png"
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            with z.open(f"train/{img_id}.dicom") as f:
                dicom_bytes = f.read()

        exists = out_path.exists() and not overwrite
        dcm_meta = pydicom.dcmread(io.BytesIO(dicom_bytes), stop_before_pixels=True)
        h, w = int(dcm_meta.Rows), int(dcm_meta.Columns)
        if exists:
            return img_id, w, h

        dcm = pydicom.dcmread(io.BytesIO(dicom_bytes))
        raw = _normalize_to_8bit(_apply_voi_lut_and_invert(dcm))
        raw = cv2.resize(raw, (image_size, image_size), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(raw)
        lap = cv2.Laplacian(clahe, cv2.CV_32F, ksize=3)
        lap = _normalize_to_8bit(np.abs(lap), low_pct=0.0, high_pct=99.5)
        img3 = np.stack([raw, clahe, lap], axis=-1)
        cv2.imwrite(str(out_path), cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
        return img_id, w, h
    except Exception as e:
        print(f"[prepare] LOI {img_id}: {e}")
        return img_id, None, None


def _build_multilabel_matrix(image_ids, ann_df):
    """14 disease labels + 1 pseudo-label No finding de stratify."""
    index = {img_id: i for i, img_id in enumerate(image_ids)}
    y = np.zeros((len(image_ids), NUM_CLASSES + 1), dtype=np.int8)
    for row in ann_df[["image_id", "class_id"]].itertuples(index=False):
        i = index.get(row.image_id)
        if i is not None:
            y[i, int(row.class_id)] = 1
    no_finding = y[:, :NUM_CLASSES].sum(axis=1) == 0
    y[no_finding, NUM_CLASSES] = 1
    return y


def _stratified_split(image_ids, ann_df, val_split, test_split, seed):
    ids = np.asarray(image_ids)
    y = _build_multilabel_matrix(image_ids, ann_df)
    holdout = val_split + test_split
    if not (0.0 < holdout < 1.0):
        raise ValueError("val_split + test_split phai nam trong (0, 1)")

    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
    train_idx, temp_idx = next(first.split(ids, y))
    train_ids = ids[train_idx]
    temp_ids = ids[temp_idx]
    temp_y = y[temp_idx]

    rel_test = test_split / holdout
    second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed + 1)
    val_rel, test_rel = next(second.split(temp_ids, temp_y))
    return train_ids.tolist(), temp_ids[val_rel].tolist(), temp_ids[test_rel].tolist()


def prepare_dataset(cfg: DataConfig, zip_path: str, num_workers: int = 20):
    output_root = Path(cfg.processed_root)
    zip_file_path = Path(zip_path)
    if not zip_file_path.exists():
        raise FileNotFoundError(f"Khong tim thay file ZIP tai: {zip_file_path}")

    print(f"[prepare] Doc annotation tu {cfg.train_csv}")
    ann_df = load_annotations(cfg.train_csv)
    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    ann_df = ann_df[ann_df["class_name"].isin(class_to_id)].copy()
    ann_df["class_id"] = ann_df["class_name"].map(class_to_id).astype(int)
    all_image_ids = sorted(ann_df["image_id"].unique().tolist())

    print("[prepare] Quet danh sach file trong ZIP...")
    with zipfile.ZipFile(zip_file_path, "r") as z:
        all_dicom_ids = [Path(p).stem for p in z.namelist() if p.startswith("train/") and p.endswith(".dicom")]

    if cfg.include_no_finding and all_dicom_ids:
        no_finding_ids = sorted(set(all_dicom_ids) - set(all_image_ids))
        image_ids = sorted(set(all_image_ids) | set(no_finding_ids))
        print(f"[prepare] Anh co bbox: {len(all_image_ids)}, No finding: {len(no_finding_ids)}")
    else:
        image_ids = all_image_ids
        print(f"[prepare] Chi dung anh co bbox: {len(image_ids)}")

    train_ids, val_ids, test_ids = _stratified_split(
        image_ids, ann_df, cfg.val_split, cfg.test_split, cfg.seed
    )
    print(f"[prepare] Stratified split: Train={len(train_ids)} Val={len(val_ids)} Test={len(test_ids)}")

    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_dir = output_root / split_name
        img_out_dir = split_dir / "images"
        lbl_out_dir = split_dir / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        # Split strategy changed to multilabel stratification. Remove stale files
        # from previous preparations to prevent train/val/test leakage.
        expected = set(split_ids)
        for old in img_out_dir.glob("*.png"):
            if old.stem not in expected:
                old.unlink()
        for old in lbl_out_dir.glob("*.txt"):
            if old.stem not in expected:
                old.unlink()

        print(f"\n[prepare] Xu ly split '{split_name}' ({len(split_ids)} anh)...")
        tasks = [
            (i, str(zip_file_path), str(img_out_dir), cfg.image_size, cfg.overwrite_images)
            for i in split_ids
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
            results = list(tqdm(
                ex.map(process_single_image_mp, tasks, chunksize=16),
                total=len(tasks), desc=f"  {split_name}"
            ))

        sizes = {i: (w, h) for i, w, h in results if w is not None}
        failed = [i for i, w, _ in results if w is None]
        if failed:
            print(f"[prepare] Bo {len(failed)} anh loi: {failed[:5]}...")
        valid_ids = [i for i in split_ids if i in sizes]

        build_yolo_dataset(
            ann_df, valid_ids, img_out_dir, lbl_out_dir, sizes=sizes,
            iou_thr=cfg.wbf_iou_thr, image_size=cfg.image_size,
        )

    write_yolo_yaml(
        output_root / "dataset.yaml", dataset_path=str(output_root.resolve()),
        train_img_dir="train/images", val_img_dir="val/images", test_img_dir="test/images",
        class_names=CLASS_NAMES,
    )
    print(f"\n[prepare] Hoan tat! Dataset tai: {output_root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, default="data/vinbigdata-chest-xray-abnormalities-detection(2).zip")
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--output_root", type=str, default="data/processed/dataset_202601")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=202601)
    parser.add_argument("--wbf_iou_thr", type=float, default=0.35)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true", help="Tao lai PNG voi preprocessing moi")
    parser.add_argument("--no_include_no_finding", dest="include_no_finding", action="store_false", default=True)
    args = parser.parse_args()

    cfg = DataConfig(
        root=args.data_root, processed_root=args.output_root, image_size=args.image_size,
        val_split=args.val_split, test_split=args.test_split, seed=args.seed,
        wbf_iou_thr=args.wbf_iou_thr, include_no_finding=args.include_no_finding,
        overwrite_images=args.overwrite,
    )
    prepare_dataset(cfg, zip_path=args.zip_path, num_workers=args.num_workers)
