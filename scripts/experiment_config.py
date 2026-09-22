"""
scripts/experiment_config.py - Quan ly cau hinh thuc nghiem, Provenance, Dataset Fingerprint,
Protocol Lock, va Test Split Access Guard cho VinBigData Object Detection P0/P1.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.config import (
    CLASS_NAMES,
    CONF_THRESHOLD,
    IMAGE_SIZE,
    NMS_IOU_THRESHOLD,
    NUM_CLASSES,
    STRIDE,
    get_grid_size,
    get_processed_data_root,
)

CANONICAL_SEEDS: List[int] = [202601, 202602, 202603]
CANONICAL_MODELS: List[str] = ["model1", "model2", "model3"]

# Bi-phase threshold policy
EVAL_MIN_SCORE: float = 0.01       # Dung cho PR curve va continuous VOC all-points mAP@0.5
BASELINE_CONF_THRESHOLD: float = 0.25 # P1 baseline operating point
BASELINE_NMS_IOU: float = 0.45        # P1 baseline NMS IoU
MATCH_IOU_THRESHOLD: float = 0.5      # IoU threshold de coi la True Positive

_PROTOCOL_SECRET: bytes = b"vinbigdata_detection_p0_p1_lock_salt_2026"


def get_git_commit() -> str:
    """Lay SHA commit git hien tai hoac fallback neu khong co git repo."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "git_commit_unknown"


def compute_file_sha256(path: str | Path) -> str:
    """Tinh ma SHA-256 cua file tren dia theo block 64KB."""
    h = hashlib.sha256()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File khong ton tai de tinh SHA: {path}")
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def write_checkpoint_sha256(checkpoint_path: str | Path) -> str:
    """
    Tinh SHA-256 cua file checkpoint da luu tren dia va ghi ra file sidecar {path}.sha256.
    Tranh nghich ly de quy: SHA duoc tinh tu file tren dia, khong tu luu vao noi dung file.
    """
    ckpt_path = Path(checkpoint_path)
    sha = compute_file_sha256(ckpt_path)
    sidecar_path = ckpt_path.parent / f"{ckpt_path.name}.sha256"
    sidecar_path.write_text(f"{sha}  {ckpt_path.name}\n", encoding="utf-8")
    return sha


def compute_split_fingerprint(split_dir: Path) -> str:
    """
    Tinh fingerprint cho 1 split (train/val/test) gom images/ va labels/.
    Bao gom: danh sach sorted ten file, kich thuoc, va noi dung labels.
    """
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Thieu thu muc anh trong split: {img_dir}")
    if not lbl_dir.is_dir():
        raise FileNotFoundError(f"Thieu thu muc nhan trong split: {lbl_dir}")

    h = hashlib.sha256()
    image_files = sorted(img_dir.glob("*.png"))
    for img_p in image_files:
        h.update(img_p.name.encode("utf-8"))
        h.update(str(img_p.stat().st_size).encode("utf-8"))

        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            raise FileNotFoundError(
                f"Missing label file: {lbl_p}. Empty vs missing label contract requires "
                f"a label file for every image (0-byte file represents No-Finding)."
            )
        h.update(lbl_p.name.encode("utf-8"))
        lbl_bytes = lbl_p.read_bytes()
        h.update(hashlib.sha256(lbl_bytes).digest())

    return h.hexdigest()


def compute_dataset_fingerprint(data_root: str | Path) -> str:
    """
    Canonical Dataset Fingerprint:
    Bao gom: SHA cua train, val, test splits, danh sach ten lop va so luong lop.
    """
    root = Path(data_root)
    train_sha = compute_split_fingerprint(root / "train")
    val_sha = compute_split_fingerprint(root / "val")
    test_sha = compute_split_fingerprint(root / "test")

    payload = (
        f"train={train_sha}|val={val_sha}|test={test_sha}|"
        f"classes={','.join(CLASS_NAMES)}|nc={NUM_CLASSES}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_protocol_lock(
    data_root: str | Path,
    checkpoint_dir: str | Path = "checkpoints",
    lock_path: str | Path = "outputs/protocol_lock.json",
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> Dict:
    """
    Sinh file protocol_lock.json khoa toan bo 9 checkpoints (SHA256 tinh tu dia),
    dataset fingerprint, evaluation parameters, git commit.
    """
    models = models or CANONICAL_MODELS
    seeds = seeds or CANONICAL_SEEDS
    data_root = Path(data_root)
    checkpoint_dir = Path(checkpoint_dir)
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[protocol_lock] Tinh dataset fingerprint tai: {data_root}")
    ds_fingerprint = compute_dataset_fingerprint(data_root)

    checkpoints_meta = {}
    for m in models:
        for s in seeds:
            key = f"{m}_seed{s}"
            ckpt_name = f"{m}_seed{s}_best.pth"
            ckpt_path = checkpoint_dir / ckpt_name
            if not ckpt_path.is_file():
                # Co the nam trong checkpoint_dir / f"seed_{s}" / ckpt_name
                alt_path = checkpoint_dir / f"seed_{s}" / ckpt_name
                if alt_path.is_file():
                    ckpt_path = alt_path
                else:
                    raise FileNotFoundError(
                        f"Khong tim thay checkpoint de khoa protocol: {ckpt_path} hoac {alt_path}"
                    )

            sha = write_checkpoint_sha256(ckpt_path)
            checkpoints_meta[key] = {
                "model": m,
                "seed": s,
                "relative_path": str(ckpt_path.as_posix()),
                "sha256": sha,
                "file_size_bytes": ckpt_path.stat().st_size,
            }

    lock_data = {
        "protocol_version": "P0_P1_DETECTION_V1",
        "timestamp_utc": str(Path(lock_path).stat().st_mtime if lock_path.exists() else ""),
        "git_commit": get_git_commit(),
        "dataset_root": str(data_root.as_posix()),
        "dataset_fingerprint": ds_fingerprint,
        "models": models,
        "seeds": seeds,
        "expected_checkpoints_count": len(models) * len(seeds),
        "checkpoints": checkpoints_meta,
        "evaluation_config": {
            "image_size": IMAGE_SIZE,
            "stride": STRIDE,
            "grid_size": get_grid_size(IMAGE_SIZE, STRIDE),
            "eval_min_score": EVAL_MIN_SCORE,
            "baseline_conf_threshold": BASELINE_CONF_THRESHOLD,
            "baseline_nms_iou": BASELINE_NMS_IOU,
            "match_iou_threshold": MATCH_IOU_THRESHOLD,
            "voc_all_points_ap": True,
            "num_classes": NUM_CLASSES,
        },
    }

    lock_path.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")
    print(f"[protocol_lock] Da tao va khoa protocol tai: {lock_path}")
    return lock_data


def _make_lock_token(lock_payload_str: str) -> str:
    """Tao token xac thuc cho protocol lock."""
    return hmac.new(_PROTOCOL_SECRET, lock_payload_str.encode("utf-8"), hashlib.sha256).hexdigest()


def global_preflight_check(
    lock_path: str | Path = "outputs/protocol_lock.json",
    data_root: Optional[str | Path] = None,
) -> Tuple[bool, str]:
    """
    Kiem tra toan bo 9 checkpoints va dataset fingerprint theo protocol_lock.json.
    Neu hop le 100%, cap `protocol_lock_token` de mo quyen truy cap TestLoader.
    Neu bat ky dieu kien nao that bai, nem RuntimeError (Fail-Closed).
    """
    lock_file = Path(lock_path)
    if not lock_file.is_file():
        raise RuntimeError(
            f"Global preflight check that bai: Khong tim thay {lock_file}. "
            f"Hay chay generate_protocol_lock() sau khi train xong develop phase."
        )

    lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
    ds_root = Path(data_root or lock_data["dataset_root"])

    print(f"[preflight] Xac thuc Dataset Fingerprint...")
    current_ds_fp = compute_dataset_fingerprint(ds_root)
    if current_ds_fp != lock_data["dataset_fingerprint"]:
        raise RuntimeError(
            f"Dataset Fingerprint mismatch! Data da bi thay doi sau khi khoa lock.\n"
            f"Expected: {lock_data['dataset_fingerprint']}\n"
            f"Actual:   {current_ds_fp}"
        )

    print(f"[preflight] Xac thuc 9 checkpoints va sidecar SHA256...")
    for key, info in lock_data["checkpoints"].items():
        ckpt_p = Path(info["relative_path"])
        if not ckpt_p.is_file():
            raise RuntimeError(f"Preflight check that bai: Checkpoint bi thieu: {ckpt_p}")

        disk_sha = compute_file_sha256(ckpt_p)
        if disk_sha != info["sha256"]:
            raise RuntimeError(
                f"Checkpoint SHA256 mismatch cho {key} ({ckpt_p})!\n"
                f"Lock SHA:   {info['sha256']}\n"
                f"Actual SHA: {disk_sha}"
            )

        sidecar_p = ckpt_p.parent / f"{ckpt_p.name}.sha256"
        if not sidecar_p.is_file():
            raise RuntimeError(f"Sidecar SHA256 file bi thieu cho {ckpt_p}: {sidecar_p}")

        sidecar_content = sidecar_p.read_text(encoding="utf-8").strip().split()[0]
        if sidecar_content != disk_sha:
            raise RuntimeError(
                f"Sidecar SHA256 mismatch cho {ckpt_p}!\n"
                f"Sidecar SHA: {sidecar_content}\n"
                f"Disk SHA:    {disk_sha}"
            )

    token_payload = f"{lock_data['dataset_fingerprint']}:{len(lock_data['checkpoints'])}:{lock_data['git_commit']}"
    token = _make_lock_token(token_payload)
    print(f"[preflight] Global Preflight Passed! Cap protocol_lock_token: {token[:16]}...")
    return True, token


def verify_protocol_lock_token(
    token: Optional[str],
    lock_path: str | Path = "outputs/protocol_lock.json",
) -> bool:
    """Kiem tra xem token co hop le voi protocol_lock.json hien tai khong."""
    if not token or not isinstance(token, str):
        return False
    lock_file = Path(lock_path)
    if not lock_file.is_file():
        return False
    try:
        lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
        token_payload = f"{lock_data['dataset_fingerprint']}:{len(lock_data['checkpoints'])}:{lock_data['git_commit']}"
        expected_token = _make_lock_token(token_payload)
        return hmac.compare_digest(token, expected_token)
    except Exception:
        return False


def assert_test_access_allowed(
    token: Optional[str],
    lock_path: str | Path = "outputs/protocol_lock.json",
):
    """
    Split Semantics Guard:
    Cam tuyet doi truy cap TestLoader neu khong co token xac thuc tu preflight check.
    """
    if not verify_protocol_lock_token(token, lock_path=lock_path):
        raise RuntimeError(
            "TEST SET ACCESS DENIED! (Split Semantics Guard Fail-Closed)\n"
            "Ban khong the tao hoac load TestLoader trong giai doan Develop hoac khi chua vuot qua "
            "global_preflight_check() voi protocol_lock_token hop le."
        )
