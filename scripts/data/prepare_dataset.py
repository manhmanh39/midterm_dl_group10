"""Chuan bi VinBigData tu file ZIP -> PNG 3-kenh + nhan YOLO (dung kich thuoc DICOM that)."""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pydicom
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from scripts.config import CLASS_NAMES, DataConfig
from scripts.data.utils import (
    _apply_voi_lut_and_invert, _normalize_to_8bit, load_annotations,
    build_yolo_dataset, write_yolo_yaml,
)


def process_single_image_mp(args_tuple):
    """Tra ve (img_id, orig_w, orig_h); (img_id, None, None) neu loi."""
    img_id, zip_path, img_out_dir, image_size = args_tuple
    out_path = Path(img_out_dir) / f"{img_id}.png"
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            with z.open(f"train/{img_id}.dicom") as f:
                dicom_bytes = f.read()

        exists = out_path.exists()
        dcm = pydicom.dcmread(io.BytesIO(dicom_bytes), stop_before_pixels=exists)
        h, w = int(dcm.Rows), int(dcm.Columns)
        if exists:
            return img_id, w, h

        dcm = pydicom.dcmread(io.BytesIO(dicom_bytes))
        arr8 = _normalize_to_8bit(_apply_voi_lut_and_invert(dcm))
        arr8 = cv2.resize(arr8, (image_size, image_size), interpolation=cv2.INTER_AREA)
        ch_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(arr8)
        lap = _normalize_to_8bit(np.absolute(cv2.Laplacian(ch_clahe, cv2.CV_64F, ksize=3)))
        img3 = np.stack([arr8, ch_clahe, lap], axis=-1)
        cv2.imwrite(str(out_path), cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
        return img_id, w, h
    except Exception as e:
        print(f"[prepare] LOI {img_id}: {e}")
        return img_id, None, None


def prepare_dataset(cfg: DataConfig, zip_path: str, num_workers: int = 4):
    output_root = Path(cfg.processed_root)
    zip_file_path = Path(zip_path)
    if not zip_file_path.exists():
        raise FileNotFoundError(f"Khong tim thay file ZIP tai: {zip_file_path}")

    print(f"[prepare] Doc annotation tu {cfg.train_csv}")
    ann_df = load_annotations(cfg.train_csv)
    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    ann_df = ann_df[ann_df["class_name"].isin(class_to_id)].copy()
    ann_df["class_id"] = ann_df["class_name"].map(class_to_id)
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

    train_ids, temp_ids = train_test_split(image_ids, test_size=cfg.val_split + cfg.test_split, random_state=cfg.seed)
    rel_test = cfg.test_split / (cfg.val_split + cfg.test_split)
    val_ids, test_ids = train_test_split(temp_ids, test_size=rel_test, random_state=cfg.seed)
    print(f"[prepare] Train={len(train_ids)} Val={len(val_ids)} Test={len(test_ids)} | workers={num_workers}")

    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_dir = output_root / split_name
        img_out_dir = split_dir / "images"
        lbl_out_dir = split_dir / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[prepare] Xu ly split '{split_name}' ({len(split_ids)} anh)...")
        tasks = [(i, str(zip_file_path), str(img_out_dir), cfg.image_size) for i in split_ids]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
            results = list(tqdm(ex.map(process_single_image_mp, tasks, chunksize=16),
                                total=len(tasks), desc=f"  {split_name}"))

        sizes = {i: (w, h) for i, w, h in results if w is not None}
        failed = [i for i, w, _ in results if w is None]
        if failed:
            print(f"[prepare] Bo {len(failed)} anh loi: {failed[:5]}...")
        split_ids = [i for i in split_ids if i in sizes]

        build_yolo_dataset(ann_df, split_ids, img_out_dir, lbl_out_dir, sizes=sizes,
                           iou_thr=cfg.wbf_iou_thr, image_size=cfg.image_size)

    write_yolo_yaml(
        output_root / "dataset.yaml", dataset_path=str(output_root.resolve()),
        train_img_dir="train/images", val_img_dir="val/images", test_img_dir="test/images",
        class_names=CLASS_NAMES,
    )
    print(f"\n[prepare] Hoan tat! Dataset tai: {output_root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, default="data/vinbigdata-chest-xray-abnormalities-detection(2).zip")
    parser.add_argument("--data_root", type=str, default="data/raw", help="Thu muc chua train.csv")
    parser.add_argument("--output_root", type=str, default="data/processed/dataset_202601")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=202601)
    parser.add_argument("--wbf_iou_thr", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--no_include_no_finding", dest="include_no_finding", action="store_false", default=True)
    args = parser.parse_args()

    cfg = DataConfig(
        root=args.data_root, processed_root=args.output_root, image_size=args.image_size,
        val_split=args.val_split, test_split=args.test_split, seed=args.seed,
        wbf_iou_thr=args.wbf_iou_thr, include_no_finding=args.include_no_finding,
    )
    prepare_dataset(cfg, zip_path=args.zip_path, num_workers=args.num_workers)