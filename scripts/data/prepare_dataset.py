"""
prepare_dataset.py - Chuan bi du lieu VinBigData truc tiep tu file ZIP (True Multiprocessing)
thanh dataset da xu ly (anh PNG 3-kenh + nhan YOLO-format, nhieu box/anh).
"""

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
    _apply_voi_lut_and_invert,
    _normalize_to_8bit,
    load_annotations,
    build_yolo_dataset,
    write_yolo_yaml,
)


def process_single_image_mp(args_tuple):
    """Ham xu ly 1 anh cho ProcessPoolExecutor (True Multiprocessing, ne tranh GIL)."""
    img_id, zip_path, img_out_dir, image_size = args_tuple
    out_path = Path(img_out_dir) / f"{img_id}.png"
    if out_path.exists():
        return True

    zip_internal_path = f"train/{img_id}.dicom"
    try:
        # Moi process tu mo 1 handle rieng den file zip de doc doc lap, an toan luong/tien trinh
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open(zip_internal_path) as f:
                dicom_bytes = f.read()
        
        dcm = pydicom.dcmread(io.BytesIO(dicom_bytes))
        arr = _apply_voi_lut_and_invert(dcm)
        arr8 = _normalize_to_8bit(arr)
        arr8 = cv2.resize(arr8, (image_size, image_size), interpolation=cv2.INTER_AREA)

        ch_raw = arr8
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ch_clahe = clahe.apply(arr8)

        laplacian = cv2.Laplacian(ch_clahe, cv2.CV_64F, ksize=3)
        laplacian = np.absolute(laplacian)
        ch_laplacian = _normalize_to_8bit(laplacian)

        img3 = np.stack([ch_raw, ch_clahe, ch_laplacian], axis=-1)
        cv2.imwrite(str(out_path), cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        return False


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

    print(f"[prepare] Dang quet danh sach file trong file ZIP...")
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        zip_namelist = z.namelist()
        all_dicom_ids = [Path(p).stem for p in zip_namelist if p.startswith('train/') and p.endswith('.dicom')]

    if cfg.include_no_finding and all_dicom_ids:
        no_finding_ids = sorted(set(all_dicom_ids) - set(all_image_ids))
        image_ids = sorted(set(all_image_ids) | set(no_finding_ids))
        print(f"[prepare] Anh co bbox: {len(all_image_ids)}, anh No finding: {len(no_finding_ids)}")
    else:
        image_ids = all_image_ids
        print(f"[prepare] Chi dung anh co bbox: {len(image_ids)}")

    train_ids, temp_ids = train_test_split(image_ids, test_size=cfg.val_split + cfg.test_split, random_state=cfg.seed)
    rel_test = cfg.test_split / (cfg.val_split + cfg.test_split)
    val_ids, test_ids = train_test_split(temp_ids, test_size=rel_test, random_state=cfg.seed)

    print(f"[prepare] Train={len(train_ids)}  Val={len(val_ids)}  Test={len(test_ids)}")
    print(f"[prepare] Su dung ProcessPoolExecutor voi num_workers = {num_workers} (True Multiprocessing)...")

    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_dir = output_root / split_name
        img_out_dir = split_dir / "images"
        lbl_out_dir = split_dir / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[prepare] Xu ly anh cho split '{split_name}' ({len(split_ids)} anh)...")
        
        tasks = [(img_id, str(zip_file_path), str(img_out_dir), cfg.image_size) for img_id in split_ids]

        # Dung ProcessPoolExecutor de tan dung tat ca CPU cores, tranh GIL
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(
                executor.map(process_single_image_mp, tasks),
                total=len(tasks),
                desc=f"  {split_name}"
            ))

        print(f"[prepare] Ghi nhan (labels) cho split '{split_name}'...")
        build_yolo_dataset(
            ann_df, split_ids, img_out_dir, lbl_out_dir,
            iou_thr=cfg.wbf_iou_thr, image_size=cfg.image_size,
        )

    yaml_path = output_root / "dataset.yaml"
    write_yolo_yaml(
        yaml_path,
        dataset_path=str(output_root.resolve()),
        train_img_dir="train/images",
        val_img_dir="val/images",
        test_img_dir="test/images",
        class_names=CLASS_NAMES,
    )
    print(f"\n[prepare] Hoan tat! Dataset san sang tai: {output_root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuan bi dataset VinBigData truc tiep tu file ZIP")
    parser.add_argument("--zip_path", type=str, default="data/vinbigdata-chest-xray-abnormalities-detection(2).zip")
    parser.add_argument("--output_root", type=str, default="data/processed/dataset_202601")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=202601)
    parser.add_argument("--wbf_iou_thr", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=18, help="So luong tien trinh chay song song")
    parser.add_argument("--no_include_no_finding", dest="include_no_finding", action="store_false", default=True)

    args = parser.parse_args()
    cfg = DataConfig(
        processed_root=args.output_root,
        image_size=args.image_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        wbf_iou_thr=args.wbf_iou_thr,
        include_no_finding=args.include_no_finding,
    )
    prepare_dataset(cfg, zip_path=args.zip_path, num_workers=args.num_workers)