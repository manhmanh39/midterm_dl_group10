"""
scripts/experiment_config.py - Quan ly cau hinh thuc nghiem, Provenance, Dataset Fingerprint,
Protocol Lock, va Test Split Access Guard cho VinBigData Object Detection P0/P1.
(Hardened implementation theo 18 chi muc kiem toan).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple, Union

from scripts.config import (
    BASELINE_CONF_THRESHOLD,
    BASELINE_NMS_IOU,
    CLASS_NAMES,
    EVAL_MIN_SCORE,
    IMAGE_SIZE,
    MATCH_IOU_THRESHOLD,
    MAX_DET,
    NMS_IOU_THRESHOLD,
    NUM_CLASSES,
    PREPROCESSING_IDENTITY,
    STRIDE,
    get_grid_size,
    get_processed_data_root,
)

CANONICAL_SEEDS: List[int] = [202601, 202602, 202603]
CANONICAL_MODELS: List[str] = ["model1", "model2", "model3"]

_PROTOCOL_SECRET: bytes = b"vinbigdata_detection_p0_p1_hardened_salt_2026"


def get_git_commit() -> str:
    """Lay SHA commit git HEAD hien tai hoac tra ve git_commit_unknown."""
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


def is_git_clean(ignore_runtime_outputs: bool = True) -> bool:
    """
    Kiem tra git working tree co hoan toan sach khong (khong co staged/unstaged changes).
    Mac dinh bo qua runtime artifacts sinh ra trong qua trinh train/lock/eval o thu muc outputs/.
    """
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        if not ignore_runtime_outputs:
            return len(lines) == 0

        unclean = []
        for line in lines:
            path_part = line[3:].strip().replace("\\", "/")
            if path_part.startswith("outputs/"):
                if any(path_part.startswith(f"outputs/{prefix}") for prefix in [
                    "protocol_lock", "canonical_multi_seed", "develop", "history", "eval",
                ]) or path_part.endswith((".sha256", ".png", ".log", ".tmp")):
                    continue
            unclean.append(line)
        return len(unclean) == 0
    except Exception:
        return False


def write_protocol_lock_sha256(lock_path: str | Path) -> str:
    """
    Tinh SHA-256 cua file protocol_lock.json va ghi ra sidecar {lock_path}.sha256.
    Khoa tinh bat bien tuyet doi (Full-Lock Immutability).
    """
    p = Path(lock_path)
    sha = compute_file_sha256(p)
    sidecar_p = p.parent / f"{p.name}.sha256"
    sidecar_p.write_text(f"{sha}  {p.name}\n", encoding="utf-8")
    return sha


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
    - Bắt buộc quan hệ song ánh tuyệt đối: set(image_stems) == set(label_stems).
    - Băm đầy đủ từng byte dữ liệu của toàn bộ ảnh .png và nhãn .txt.
    """
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Thieu thu muc anh trong split: {img_dir}")
    if not lbl_dir.is_dir():
        raise FileNotFoundError(f"Thieu thu muc nhan trong split: {lbl_dir}")

    image_files = sorted(img_dir.glob("*.png"))
    label_files = sorted(lbl_dir.glob("*.txt"))

    img_stems = set(p.stem for p in image_files)
    lbl_stems = set(p.stem for p in label_files)

    # Enforce exact bijection
    if img_stems != lbl_stems:
        missing_lbl = img_stems - lbl_stems
        orphan_lbl = lbl_stems - img_stems
        raise RuntimeError(
            f"Dataset integrity violation tai {split_dir}: Danh sach anh va nhan khong song anh!\n"
            f"  - So anh thieu file nhan: {len(missing_lbl)} (vi du: {list(missing_lbl)[:3]})\n"
            f"  - So file nhan mo coi: {len(orphan_lbl)} (vi du: {list(orphan_lbl)[:3]})"
        )

    h = hashlib.sha256()
    for img_p in image_files:
        # Băm relative path + kích thước + toàn bộ raw bytes của ảnh PNG
        h.update(f"img:{img_p.name}:{img_p.stat().st_size}".encode("utf-8"))
        with open(img_p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)

        # Băm relative path + kích thước + toàn bộ raw bytes của nhãn TXT
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        h.update(f"lbl:{lbl_p.name}:{lbl_p.stat().st_size}".encode("utf-8"))
        with open(lbl_p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)

    return h.hexdigest()


def compute_dataset_fingerprint(data_root: str | Path) -> str:
    """
    Canonical Dataset Fingerprint:
    SHA256 băm toàn bộ bytes của train, val, test, cùng danh sách 14 tên lớp bệnh học.
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


@dataclass
class PreflightPermit:
    """
    Giấy phép truy cập TestLoader cấp từ global_preflight_check().
    Cung cấp cơ chế dynamic re-verification chống thay đổi artifact sau khi cấp.
    """
    token: str
    lock_path: str
    lock_sha256: str
    dataset_root: str
    dataset_fingerprint: str
    issued_at_utc: str
    data_mode: str
    checkpoints: Dict[str, Dict]

    def verify(self, recheck_dataset: bool = True) -> bool:
        """Kiem tra permit hop le va tuy chon tai kiem tra fingerprint dataset tren dia."""
        if not verify_protocol_lock_token(self.token, lock_path=self.lock_path):
            return False
        # Full-Lock Immutability verification
        if not Path(self.lock_path).is_file():
            return False
        current_lock_sha = compute_file_sha256(self.lock_path)
        if current_lock_sha != self.lock_sha256:
            return False
        if recheck_dataset:
            cur_fp = compute_dataset_fingerprint(self.dataset_root)
            if cur_fp != self.dataset_fingerprint:
                return False
        return True


def _make_lock_token(lock_payload_str: str) -> str:
    """Tao token xac thuc HMAC cho protocol lock."""
    return hmac.new(_PROTOCOL_SECRET, lock_payload_str.encode("utf-8"), hashlib.sha256).hexdigest()


def generate_protocol_lock(
    data_root: str | Path,
    checkpoint_dir: str | Path = "checkpoints",
    lock_path: str | Path = "outputs/protocol_lock.json",
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    data_mode: str = "real",
) -> Dict:
    """
    Sinh file protocol_lock.json khoa toan bo 9 checkpoints (SHA256 tinh tu dia),
    dataset fingerprint, evaluation parameters, git commit, va timestamp UTC chuan.
    Chi duoc goi o phase=lock (hoac chuoi all).
    """
    models = models or CANONICAL_MODELS
    seeds = seeds or CANONICAL_SEEDS
    data_root = Path(data_root)
    checkpoint_dir = Path(checkpoint_dir)
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # 0. Deduplication check cho ca real va demo
    if len(models) != len(set(models)):
        raise ValueError(f"Protocol Lock error: Duplicate model found in models list: {models}")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Protocol Lock error: Duplicate seed found in seeds list: {seeds}")

    # 1. Enforce Exact Canonical 3x3 o che do real
    if data_mode == "real":
        if sorted(list(models)) != sorted(CANONICAL_MODELS):
            raise ValueError(
                f"Protocol Lock error (Real mode): Bat buoc dung chinh xac 3 canonical models "
                f"{CANONICAL_MODELS} khong trung lap.\nReceived models: {models}"
            )
        if sorted(list(seeds)) != sorted(CANONICAL_SEEDS):
            raise ValueError(
                f"Protocol Lock error (Real mode): Bat buoc dung chinh xac 3 canonical seeds "
                f"{CANONICAL_SEEDS} khong trung lap.\nReceived seeds: {seeds}"
            )

        # 2. Check Git HEAD & Working tree clean
        commit = get_git_commit()
        if commit == "git_commit_unknown":
            raise RuntimeError("Protocol Lock error (Real mode): Git commit khong xac dinh!")
        if not is_git_clean():
            raise RuntimeError(
                "Protocol Lock error (Real mode): Git working tree dang bi dirty (chua commit)! "
                "Hay commit toan bo thay doi truoc khi khoa protocol."
            )
    else:
        commit = get_git_commit()

    print(f"[protocol_lock] Tinh toan Dataset Fingerprint tren {data_root} (che do {data_mode})...")
    ds_fingerprint = compute_dataset_fingerprint(data_root)

    checkpoints_meta = {}
    for m in models:
        for s in seeds:
            key = f"{m}_seed{s}"
            ckpt_name = f"{m}_seed{s}_best.pth"
            ckpt_path = checkpoint_dir / ckpt_name
            if not ckpt_path.is_file():
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
        "protocol_version": "P0_P1_DETECTION_HARDENED_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_clean_at_lock": is_git_clean(),
        "data_mode": data_mode,
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
            "max_det": MAX_DET,
            "voc_all_points_ap": True,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "preprocessing_identity": PREPROCESSING_IDENTITY,
        },
    }

    lock_path.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")
    lock_sha = write_protocol_lock_sha256(lock_path)
    print(f"[protocol_lock] Da tao va khoa protocol thanh cong tai: {lock_path} (SHA: {lock_sha[:16]}...)")
    return lock_data


def global_preflight_check(
    lock_path: str | Path = "outputs/protocol_lock.json",
    data_root: Optional[str | Path] = None,
    data_mode: str = "real",
) -> Tuple[bool, PreflightPermit]:
    """
    Kiem tra toan bo checkpoints, dataset fingerprint, git status theo protocol_lock.json.
    Chi doc protocol_lock.json hien co, tuyet doi khong tu tao hoac ghi de lock!
    Neu hop le 100%, cap PreflightPermit de mo quyen truy cap TestLoader.
    """
    lock_file = Path(lock_path)
    if not lock_file.is_file():
        raise RuntimeError(
            f"Global preflight check that bai: Khong tim thay {lock_file}.\n"
            f"Giai doan final-test yeu cau file lock da duoc sinh truoc tu phase=lock."
        )

    # 0. Kiem tra Full-Lock Immutability qua Sidecar SHA256
    sidecar_p = lock_file.parent / f"{lock_file.name}.sha256"
    if not sidecar_p.is_file():
        raise RuntimeError(
            f"SECURITY ALERT (P1 Blocker): Thieu sidecar SHA256 cho file lock tai: {sidecar_p}!\n"
            f"Moi protocol_lock.json bat buoc phai co file sidecar SHA256 song hanh de dam bao tinh bat bien."
        )
    sidecar_sha = sidecar_p.read_text(encoding="utf-8").strip().split()[0]
    disk_lock_sha = compute_file_sha256(lock_file)
    if sidecar_sha != disk_lock_sha:
        raise RuntimeError(
            f"SECURITY ALERT (P1 Blocker): Protocol lock immutability violated! File {lock_file} da bi sua doi!\n"
            f"Sidecar SHA: {sidecar_sha}\n"
            f"Disk SHA:    {disk_lock_sha}"
        )

    lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
    ds_root = Path(data_root or lock_data["dataset_root"])

    # Anti-bypass: Neu lock la real thi khong cho phep dung data_mode=demo de bypass
    if lock_data.get("data_mode") == "real" and data_mode == "demo":
        raise ValueError(
            "SECURITY ALERT: Khong the su dung data_mode='demo' voi protocol_lock duoc khoa o che do 'real'! "
            "Day la vi pham quy tac bao mat giao thuc (Anti-Bypass Guard)."
        )
    mode = lock_data.get("data_mode", "real") if lock_data.get("data_mode") == "real" else (data_mode or "real")

    # 1. Kiem tra Git HEAD va Clean neu la real mode
    if mode == "real":
        current_commit = get_git_commit()
        if current_commit == "git_commit_unknown":
            raise RuntimeError("Preflight check that bai (Real mode): Git commit khong xac dinh!")
        if current_commit != lock_data["git_commit"]:
            raise RuntimeError(
                f"Git HEAD mismatch! Code da bi thay doi sau khi khoa protocol.\n"
                f"Locked commit:  {lock_data['git_commit']}\n"
                f"Current commit: {current_commit}"
            )
        if not is_git_clean():
            raise RuntimeError("Git working tree dang co uncommitted changes! Can git clean truoc khi test.")

    # 2. Xac thuc Dataset Fingerprint
    print(f"[preflight] Xac thuc Dataset Fingerprint tai: {ds_root}")
    current_ds_fp = compute_dataset_fingerprint(ds_root)
    if current_ds_fp != lock_data["dataset_fingerprint"]:
        raise RuntimeError(
            f"Dataset Fingerprint mismatch! Dữ liệu đã bị thay đổi sau khi khóa protocol.\n"
            f"Expected: {lock_data['dataset_fingerprint']}\n"
            f"Actual:   {current_ds_fp}"
        )

    # 3. Xac thuc tat ca Checkpoints va Sidecars
    print(f"[preflight] Xac thuc {len(lock_data['checkpoints'])} checkpoints va sidecar SHA256...")
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

    token_payload = f"{disk_lock_sha}:{lock_data['dataset_fingerprint']}:{lock_data['git_commit']}"
    token = _make_lock_token(token_payload)

    permit = PreflightPermit(
        token=token,
        lock_path=str(lock_file.as_posix()),
        lock_sha256=disk_lock_sha,
        dataset_root=str(ds_root.as_posix()),
        dataset_fingerprint=lock_data["dataset_fingerprint"],
        issued_at_utc=datetime.now(timezone.utc).isoformat(),
        data_mode=mode,
        checkpoints=lock_data["checkpoints"],
    )

    print(f"[preflight] Global Preflight Passed! Cap PreflightPermit hop le ({token[:16]}...).")
    return True, permit


def verify_protocol_lock_token(
    token_or_permit: Optional[Union[str, PreflightPermit]],
    lock_path: str | Path = "outputs/protocol_lock.json",
) -> bool:
    """Kiem tra token hoac permit co hop le voi protocol_lock.json hien tai khong."""
    if not token_or_permit:
        return False
    token_str = token_or_permit.token if isinstance(token_or_permit, PreflightPermit) else token_or_permit
    lock_file = Path(lock_path)
    if not lock_file.is_file():
        return False
    try:
        lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
        disk_lock_sha = compute_file_sha256(lock_file)
        token_payload = f"{disk_lock_sha}:{lock_data['dataset_fingerprint']}:{lock_data['git_commit']}"
        expected_token = _make_lock_token(token_payload)
        if not hmac.compare_digest(token_str, expected_token):
            return False
        if isinstance(token_or_permit, PreflightPermit):
            if token_or_permit.lock_sha256 != disk_lock_sha:
                return False
        return True
    except Exception:
        return False


def assert_test_access_allowed(
    token_or_permit: Optional[Union[str, PreflightPermit]],
    lock_path: str | Path = "outputs/protocol_lock.json",
    dataset_root: Optional[str | Path] = None,
):
    """
    Split Semantics Guard:
    Cam tuyet doi truy cap TestLoader neu khong co permit/token hop le hoac du lieu bi thay doi.
    """
    lock_file = Path(lock_path)
    if not lock_file.is_file():
        raise RuntimeError("TEST SET ACCESS DENIED! Protocol lock khong ton tai.")

    # Kiem tra sidecar immutability
    sidecar_p = lock_file.parent / f"{lock_file.name}.sha256"
    if not sidecar_p.is_file():
        raise RuntimeError("TEST SET ACCESS DENIED! Protocol lock sidecar .sha256 bi thieu.")
    if sidecar_p.read_text(encoding="utf-8").strip().split()[0] != compute_file_sha256(lock_file):
        raise RuntimeError("TEST SET ACCESS DENIED! Protocol lock da bi thay doi (sidecar mismatch).")

    if not verify_protocol_lock_token(token_or_permit, lock_path=lock_path):
        raise RuntimeError(
            "TEST SET ACCESS DENIED! (Split Semantics Guard Fail-Closed)\n"
            "Ban khong the tao hoac load TestLoader trong giai doan Develop hoac khi chua vuot qua "
            "global_preflight_check() voi PreflightPermit hop le."
        )

    if isinstance(token_or_permit, PreflightPermit):
        if not token_or_permit.verify(recheck_dataset=False):
            raise RuntimeError(
                "TEST SET ACCESS DENIED! PreflightPermit da bi vo hieu hoa (lock file thay doi sau khi cap)."
            )

    # Dynamic re-check dataset fingerprint neu co dataset_root hoac permit
    root_to_check = dataset_root or (token_or_permit.dataset_root if isinstance(token_or_permit, PreflightPermit) else None)
    if root_to_check and Path(root_to_check).is_dir():
        lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
        cur_fp = compute_dataset_fingerprint(root_to_check)
        if cur_fp != lock_data["dataset_fingerprint"]:
            raise RuntimeError(
                "TEST SET ACCESS DENIED! (Data tampering detected after preflight check)\n"
                f"Dataset fingerprint da bi thay doi sau khi cap permit!\n"
                f"Expected: {lock_data['dataset_fingerprint']}\n"
                f"Actual:   {cur_fp}"
            )
