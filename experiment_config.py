"""
experiment_config.py - Quản lý cấu hình thử nghiệm, dataset fingerprint, 
git commit provenance, protocol_lock.json và kiểm soát nghiêm ngặt cơ chế Fail-Closed.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import torch
import config


# Các thư mục artifact phân cấp chuẩn
DEVELOP_DIR = config.OUTPUT_DIR / "develop"
FINAL_TEST_DIR = config.OUTPUT_DIR / "final_test"
BENCHMARK_DIR = config.OUTPUT_DIR / "benchmarks"


def get_develop_dir(data_mode: Optional[str] = None, is_smoke: bool = False) -> Path:
    """Trả về thư mục develop được phân tách theo namespace (demo / real / smoke)."""
    if DEVELOP_DIR != config.OUTPUT_DIR / "develop":
        return DEVELOP_DIR
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    prefix = "smoke" if is_smoke else "develop"
    mode_str = "demo" if mode == "demo" else "real"
    target = config.OUTPUT_DIR / prefix / mode_str
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_final_test_dir(data_mode: Optional[str] = None) -> Path:
    """Trả về thư mục final_test được phân tách theo namespace (demo / real)."""
    if FINAL_TEST_DIR != config.OUTPUT_DIR / "final_test":
        return FINAL_TEST_DIR
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    mode_str = "demo" if mode == "demo" else "real"
    target = config.OUTPUT_DIR / "final_test" / mode_str
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_benchmark_dir(data_mode: Optional[str] = None, is_smoke: bool = False) -> Path:
    """Trả về thư mục benchmark được phân tách theo namespace (demo / real / smoke/real)."""
    if BENCHMARK_DIR != config.OUTPUT_DIR / "benchmarks":
        return BENCHMARK_DIR
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    prefix = Path("smoke") / "real" if is_smoke else Path("demo" if mode == "demo" else "real")
    target = config.OUTPUT_DIR / "benchmarks" / prefix
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_git_commit(allow_fallback: bool = False) -> str:
    """Lấy mã băm Git commit hiện tại (HEAD) của repository. Fail-closed nếu thất bại."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(config.BASE_DIR),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        if not commit:
            raise RuntimeError("git rev-parse HEAD trả về kết quả rỗng.")
        return commit
    except Exception as e:
        if allow_fallback:
            return "uncommitted_workspace"
        raise RuntimeError(
            f"[FAIL CLOSED] Không thể xác định Git commit tại '{config.BASE_DIR}'! "
            f"Giao thức yêu cầu môi trường Git hợp lệ."
        ) from e


def git_worktree_is_dirty(allow_fallback: bool = False) -> bool:
    """Kiểm tra xem working tree có file nào bị modified/uncommitted hay không. Fail-closed nếu thất bại."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(config.BASE_DIR),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return len(status) > 0
    except Exception as e:
        if allow_fallback:
            return False
        raise RuntimeError(
            f"[FAIL CLOSED] Không thể kiểm tra trạng thái Git working tree tại '{config.BASE_DIR}'! "
            f"Giao thức yêu cầu xác minh working tree sạch."
        ) from e


def compute_file_sha256(filepath: Optional[Path]) -> Optional[str]:
    """Tính toán mã băm SHA256 của một tệp theo từng block 64KB."""
    if filepath is None or not Path(filepath).exists():
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_fingerprint(
    data_dir: Optional[str] = None,
    data_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tạo định danh dataset_fingerprint toàn vẹn theo chuẩn canonical representation:
      SHA256("archive=" + archive_hash + "|manifest=" + manifest_hash +
             "|train=" + train_hash + "|val=" + val_hash + "|test=" + test_hash)
    Fail-closed khi data_mode == 'real' nếu path sai hoặc thiếu manifest/CSV.
    """
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    if mode not in ("real", "demo"):
        raise ValueError(f"Invalid data_mode '{mode}'. Phải là 'real' hoặc 'demo'.")

    if mode == "real":
        target_dir = Path(data_dir) if data_dir else config.PROCESSED_DATA_DIR
        if not target_dir.exists():
            raise FileNotFoundError(f"[FAIL CLOSED] Thư mục dữ liệu thật không tồn tại: {target_dir}")
        manifest_file = target_dir / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"[FAIL CLOSED] Thiếu manifest.json tại: {manifest_file}")
        is_demo = False
    else:
        target_dir = Path(data_dir) if data_dir else config.DEMO_DATA_DIR
        is_demo = True

    # 1. Tìm source archive (nếu có zip trong data)
    archive_files = list(config.DATA_DIR.glob("*.zip")) if config.DATA_DIR.exists() else []
    archive_hash = compute_file_sha256(archive_files[0]) if archive_files else "none"

    manifest_file = target_dir / "manifest.json"
    manifest_hash = compute_file_sha256(manifest_file) if manifest_file.exists() else "none"

    train_boxes = target_dir / "train" / "train_boxes.csv"
    train_hash = compute_file_sha256(train_boxes) if train_boxes.exists() else "none"

    val_boxes = target_dir / "val" / "val_boxes.csv"
    val_hash = compute_file_sha256(val_boxes) if val_boxes.exists() else "none"

    test_boxes = target_dir / "test" / "test_boxes.csv"
    test_hash = compute_file_sha256(test_boxes) if test_boxes.exists() else "none"

    if mode == "real":
        if manifest_hash == "none":
            raise FileNotFoundError(f"[FAIL CLOSED] Thiếu manifest.json cho real data tại: {manifest_file}")
        if train_hash == "none":
            raise FileNotFoundError(f"[FAIL CLOSED] Thiếu train_boxes.csv cho real data tại: {train_boxes}")
        if val_hash == "none":
            raise FileNotFoundError(f"[FAIL CLOSED] Thiếu val_boxes.csv cho real data tại: {val_boxes}")
        if test_hash == "none":
            raise FileNotFoundError(f"[FAIL CLOSED] Thiếu test_boxes.csv cho real data tại: {test_boxes}")

    canonical_str = (
        f"archive={archive_hash}|manifest={manifest_hash}|"
        f"train={train_hash}|val={val_hash}|test={test_hash}"
    )
    dataset_fingerprint = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    return {
        "dataset_fingerprint": dataset_fingerprint,
        "is_demo_data": is_demo,
        "data_mode": mode,
        "data_dir": str(target_dir),
        "source_archive_sha256": archive_hash,
        "manifest_sha256": manifest_hash,
        "train_annotations_sha256": train_hash,
        "val_annotations_sha256": val_hash,
        "test_annotations_sha256": test_hash,
    }


def generate_protocol_lock(
    model_names: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    data_dir: Optional[str] = None,
    data_mode: Optional[str] = None,
    develop_dir: Optional[Path] = None,
) -> Path:
    """
    Sinh file protocol_lock.json khóa toàn bộ 9 bộ thí nghiệm
    sau khi phase develop hoàn tất và TRƯỚC KHI mở bất kỳ final-test nào.
    Kiểm tra set equality đúng 9 cặp canonical (3 models x 3 seeds).
    """
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    model_names = list(model_names) if model_names is not None else list(config.CANONICAL_MODELS)
    seeds = list(seeds) if seeds is not None else list(config.CANONICAL_SEEDS)

    dataset_meta = compute_dataset_fingerprint(data_dir=data_dir, data_mode=mode)
    if not dataset_meta.get("is_demo_data", True) and git_worktree_is_dirty():
        raise RuntimeError(
            "[FAIL CLOSED] Working tree của Git đang có thay đổi chưa commit!\n"
            "Giao thức P1 bắt buộc working tree phải sạch trước khi tạo protocol_lock.json."
        )

    dev_dir = Path(develop_dir) if develop_dir is not None else get_develop_dir(data_mode=mode)
    dev_dir.mkdir(parents=True, exist_ok=True)
    lock_file = dev_dir / "protocol_lock.json"

    experiments = []
    missing_items = []

    for model in model_names:
        for seed in seeds:
            ckpt = dev_dir / model / f"seed{seed}" / "best.pth"
            thresh = dev_dir / model / f"seed{seed}" / "calibrated_thresholds.json"

            if not ckpt.exists():
                missing_items.append(f"Checkpoint thiếu: {ckpt}")
            if not thresh.exists():
                missing_items.append(f"Thresholds thiếu: {thresh}")

            if ckpt.exists() and thresh.exists():
                experiments.append({
                    "model": model,
                    "seed": seed,
                    "checkpoint_rel_path": str(ckpt.relative_to(dev_dir)),
                    "checkpoint_sha256": compute_file_sha256(ckpt),
                    "threshold_rel_path": str(thresh.relative_to(dev_dir)),
                    "threshold_sha256": compute_file_sha256(thresh),
                })

    if missing_items:
        raise FileNotFoundError(
            f"[PROTOCOL LOCK ERROR] Không thể tạo protocol_lock.json vì thiếu {len(missing_items)} artifacts:\n"
            + "\n".join(f"  - {m}" for m in missing_items)
        )

    # P1.4: Kiểm tra set equality và exact-nine đúng ma trận canonical 3x3
    expected_pairs = {(m, s) for m in config.CANONICAL_MODELS for s in config.CANONICAL_SEEDS}
    pairs = [(exp["model"], exp["seed"]) for exp in experiments]
    actual_pairs = set(pairs)
    if mode == "real" or (len(model_names) == 3 and len(seeds) == 3):
        if len(pairs) != 9 or len(actual_pairs) != 9 or actual_pairs != expected_pairs:
            raise ValueError(
                f"[PROTOCOL LOCK ERROR] Thí nghiệm không khớp chính xác ma trận canonical 3x3 ({len(expected_pairs)} cặp không trùng lặp)!\n"
                f"  Tổng số pairs: {len(pairs)}, unique: {len(actual_pairs)}\n"
                f"  Thiếu: {sorted(list(expected_pairs - actual_pairs))}\n"
                f"  Thừa/Lệch: {sorted(list(actual_pairs - expected_pairs))}"
            )

    lock_data = {
        "protocol_version": "P1.Locked",
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "git_dirty": git_worktree_is_dirty(),
        "data_mode": mode,
        "dataset_fingerprint": dataset_meta["dataset_fingerprint"],
        "dataset_meta": dataset_meta,
        "total_experiments": len(experiments),
        "experiments": experiments,
    }

    with open(lock_file, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2, ensure_ascii=False)

    print(f"\n[protocol_lock] 🔒 ĐÃ KHÓA TOÀN BỘ {len(experiments)} THÍ NGHIỆM TẠI: {lock_file}")
    return lock_file


def global_preflight_check(
    data_dir: Optional[str] = None,
    lock_file: Optional[Path] = None,
    enforce_clean_git: bool = False,
    data_mode: Optional[str] = None,
    develop_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    GLOBAL PREFLIGHT TRƯỚC KHI MỞ TEST SET:
    Kiểm tra toàn bộ các checkpoints và thresholds đã bị khóa trong protocol_lock.json.
    Nếu bất kỳ model/seed nào thiếu hoặc bị sai lệch hash -> ABORT (FAIL CLOSED).
    Enforce set equality đúng 9 cặp canonical trên real data.
    """
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    dev_dir = Path(develop_dir) if develop_dir is not None else get_develop_dir(data_mode=mode)
    if lock_file is None:
        candidate = dev_dir / "protocol_lock.json"
        if candidate.exists():
            lock_file = candidate
        elif (DEVELOP_DIR / "protocol_lock.json").exists():
            lock_file = DEVELOP_DIR / "protocol_lock.json"
        else:
            lock_file = candidate
    lock_file = Path(lock_file)

    if not lock_file.exists():
        raise FileNotFoundError(
            f"\n[FAIL CLOSED] Không tìm thấy {lock_file}!\n"
            "Giao thức P1 yêu cầu PHẢI hoàn tất phase develop cho toàn bộ models/seeds "
            "và đóng băng trước khi mở Final-Test."
        )

    with open(lock_file, "r", encoding="utf-8") as f:
        lock_data = json.load(f)

    # 1. Kiểm tra Git Clean Tree và Git Commit Match (nếu không phải demo data)
    current_dataset_meta = compute_dataset_fingerprint(data_dir=data_dir, data_mode=mode)
    is_demo = current_dataset_meta.get("is_demo_data", True)

    if enforce_clean_git or not is_demo:
        if git_worktree_is_dirty():
            raise RuntimeError(
                "[FAIL CLOSED] Working tree của Git đang có thay đổi chưa commit!\n"
                "Giao thức P1 bắt buộc working tree phải sạch (git status clean) khi chạy Final-Test trên real data."
            )
        locked_commit = lock_data.get("git_commit")
        if not locked_commit:
            raise ValueError("[FAIL CLOSED] protocol_lock.json thiếu trường bắt buộc 'git_commit'!")

        current_commit = get_git_commit()
        if current_commit != locked_commit:
            raise RuntimeError(
                f"[FAIL CLOSED] Git commit hiện tại ({current_commit}) không khớp với commit đã khóa trong protocol_lock.json ({locked_commit})!\n"
                "Giao thức P1 yêu cầu mã nguồn phải ở đúng commit đã khóa khi chạy Final-Test."
            )

        if lock_data.get("git_dirty") is not False:
            raise RuntimeError(
                f"[FAIL CLOSED] protocol_lock.json được tạo khi Git working tree bị dirty (git_dirty={lock_data.get('git_dirty')})!"
            )

    # 2. Kiểm tra Dataset Fingerprint
    if lock_data.get("dataset_fingerprint") != current_dataset_meta["dataset_fingerprint"]:
        raise ValueError(
            f"[FAIL CLOSED] Dataset đã bị thay đổi sau khi đóng băng:\n"
            f"  Locked fingerprint : {lock_data.get('dataset_fingerprint')}\n"
            f"  Current fingerprint: {current_dataset_meta['dataset_fingerprint']}"
        )

    # 3. Kiểm tra toàn bộ các thí nghiệm trong lock
    experiments = lock_data.get("experiments", [])
    if not experiments:
        raise ValueError("[FAIL CLOSED] protocol_lock.json rỗng!")

    # P1.4: Set equality & exact-nine 9 cặp canonical trên real data
    if mode == "real":
        expected_pairs = {(m, s) for m in config.CANONICAL_MODELS for s in config.CANONICAL_SEEDS}
        pairs = [(exp["model"], exp["seed"]) for exp in experiments]
        actual_pairs = set(pairs)
        if len(pairs) != 9 or len(actual_pairs) != 9 or actual_pairs != expected_pairs:
            raise ValueError(
                f"[FAIL CLOSED] protocol_lock.json không chứa đúng 9 cặp canonical (3 models x 3 seeds, không trùng lặp)!\n"
                f"  Tổng số pairs: {len(pairs)}, unique: {len(actual_pairs)}\n"
                f"  Thiếu: {sorted(list(expected_pairs - actual_pairs))}\n"
                f"  Thừa/Lệch: {sorted(list(actual_pairs - expected_pairs))}"
            )

    root_dev = lock_file.parent
    for exp in experiments:
        ckpt_path = root_dev / exp["checkpoint_rel_path"]
        thresh_path = root_dev / exp["threshold_rel_path"]

        if not ckpt_path.exists():
            raise FileNotFoundError(f"[FAIL CLOSED] Thiếu checkpoint: {ckpt_path}")
        if not thresh_path.exists():
            raise FileNotFoundError(f"[FAIL CLOSED] Thiếu threshold: {thresh_path}")

        current_ckpt_hash = compute_file_sha256(ckpt_path)
        if current_ckpt_hash != exp["checkpoint_sha256"]:
            raise ValueError(
                f"[FAIL CLOSED] Checkpoint {ckpt_path.name} bị sửa đổi sau khi lock!\n"
                f"  Locked: {exp['checkpoint_sha256']}\n"
                f"  Actual: {current_ckpt_hash}"
            )

        current_thresh_hash = compute_file_sha256(thresh_path)
        if current_thresh_hash != exp["threshold_sha256"]:
            raise ValueError(
                f"[FAIL CLOSED] Threshold {thresh_path.name} bị sửa đổi sau khi lock!\n"
                f"  Locked: {exp['threshold_sha256']}\n"
                f"  Actual: {current_thresh_hash}"
            )

        # ALL-9 INTERNAL PROVENANCE SEMANTIC BINDING TRƯỚC KHI MỞ TESTLOADER
        verify_fail_closed_provenance(
            checkpoint_path=ckpt_path,
            threshold_path=thresh_path,
            current_dataset_meta=current_dataset_meta,
            expected_model=exp.get("model"),
            expected_seed=exp.get("seed"),
            expected_git_commit=lock_data.get("git_commit"),
        )

    print(f"[preflight] ✅ GLOBAL PREFLIGHT PASS: Toàn bộ {len(experiments)} thí nghiệm hợp lệ 100%!")
    return lock_data


def verify_fail_closed_provenance(
    checkpoint_path: Path,
    threshold_path: Path,
    current_dataset_meta: Dict[str, Any],
    expected_model: Optional[str] = None,
    expected_seed: Optional[int] = None,
    expected_git_commit: Optional[str] = None,
) -> None:
    """Kiểm tra chéo fail-closed cho 1 checkpoint đơn lẻ kèm semantic binding với protocol lock."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"[FAIL CLOSED] Không tìm thấy checkpoint tại: {checkpoint_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"[FAIL CLOSED] Không tìm thấy calibrated thresholds tại: {threshold_path}")

    actual_ckpt_sha256 = compute_file_sha256(checkpoint_path)

    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    th_prov = threshold_data.get("provenance", {})
    ck_prov = checkpoint_data.get("provenance", {})

    REQUIRED_PROV_FIELDS = [
        "checkpoint_sha256",
        "dataset_fingerprint",
        "git_commit",
        "model_name",
        "seed",
    ]
    for field in REQUIRED_PROV_FIELDS:
        if not th_prov.get(field):
            raise ValueError(f"[FAIL CLOSED] Threshold provenance thiếu trường bắt buộc: '{field}'")
        ck_val = ck_prov.get(field) if ck_prov.get(field) is not None else checkpoint_data.get(field)
        if field != "checkpoint_sha256" and (ck_val is None or ck_val == ""):
            raise ValueError(f"[FAIL CLOSED] Checkpoint provenance thiếu trường bắt buộc: '{field}'")

    th_model = str(th_prov["model_name"]).lower().strip()
    ck_model = str(checkpoint_data.get("model_name", ck_prov.get("model_name"))).lower().strip()
    if th_model != ck_model:
        raise ValueError(f"[FAIL CLOSED] Sai lệch Model Name: {th_model} != {ck_model}")

    if expected_model is not None:
        exp_m = str(expected_model).lower().strip()
        if th_model != exp_m or ck_model != exp_m:
            raise ValueError(
                f"[FAIL CLOSED] Semantic mismatch với lock: expected_model='{exp_m}', nhưng th_model='{th_model}', ck_model='{ck_model}'"
            )

    th_seed = th_prov["seed"]
    ck_seed = checkpoint_data.get("seed", ck_prov.get("seed"))
    if th_seed != ck_seed:
        raise ValueError(f"[FAIL CLOSED] Sai lệch Seed: {th_seed} != {ck_seed}")

    if expected_seed is not None:
        exp_s = int(expected_seed)
        if int(th_seed) != exp_s or int(ck_seed) != exp_s:
            raise ValueError(
                f"[FAIL CLOSED] Semantic mismatch với lock: expected_seed={exp_s}, nhưng th_seed={th_seed}, ck_seed={ck_seed}"
            )

    th_ckpt_hash = th_prov["checkpoint_sha256"]
    if th_ckpt_hash != actual_ckpt_sha256:
        raise ValueError(f"[FAIL CLOSED] Checkpoint bị sửa đổi sau khi calibrate! (Thresh: {th_ckpt_hash} != Actual: {actual_ckpt_sha256})")

    curr_fingerprint = current_dataset_meta["dataset_fingerprint"]
    th_fingerprint = th_prov["dataset_fingerprint"]
    if th_fingerprint != curr_fingerprint:
        raise ValueError(f"[FAIL CLOSED] Dataset đã bị thay đổi sau khi calibrate! (Thresh FP: {th_fingerprint} != Current: {curr_fingerprint})")

    ck_fingerprint = ck_prov.get("dataset_fingerprint")
    if ck_fingerprint != curr_fingerprint:
        raise ValueError(f"[FAIL CLOSED] Dataset của Checkpoint không khớp Dataset hiện tại! (Ckpt FP: {ck_fingerprint} != Current: {curr_fingerprint})")

    th_commit = th_prov.get("git_commit")
    ck_commit = ck_prov.get("git_commit")
    if th_commit != ck_commit:
        raise ValueError(f"[FAIL CLOSED] Git commit không khớp giữa Checkpoint ({ck_commit}) và Thresholds ({th_commit})!")

    if expected_git_commit is not None:
        if th_commit != expected_git_commit or ck_commit != expected_git_commit:
            raise ValueError(
                f"[FAIL CLOSED] Semantic mismatch với lock: expected_git_commit='{expected_git_commit}', nhưng th_commit='{th_commit}', ck_commit='{ck_commit}'"
            )

    if not current_dataset_meta.get("is_demo_data", True):
        if ck_prov.get("git_dirty") is not False:
            raise RuntimeError(f"[FAIL CLOSED] Checkpoint {checkpoint_path.name} được huấn luyện khi Git working tree bị dirty (git_dirty={ck_prov.get('git_dirty')})!")
        if th_prov.get("git_dirty") is not False:
            raise RuntimeError(f"[FAIL CLOSED] Thresholds {threshold_path.name} được calibrate khi Git working tree bị dirty (git_dirty={th_prov.get('git_dirty')})!")

    print(f"[provenance] ✅ Verification PASS cho {th_model} (seed={th_seed}).")


def load_best_hparams(model_name: str, output_dir: Path = config.OUTPUT_DIR) -> Optional[dict]:
    """Đọc tham số tối ưu từ best_hparams_{model_name}.json nếu có."""
    alias_map = {
        "model1": "simple", "model1_simple": "simple",
        "model2": "complex", "model2_complex": "complex",
        "model3": "transfer", "base": "transfer",
    }
    canonical = alias_map.get(model_name.lower().strip(), model_name.lower().strip())
    hparam_path = output_dir / f"best_hparams_{canonical}.json"

    if hparam_path.exists():
        try:
            with open(hparam_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("best_params", {})
        except Exception:
            return None
    return None


def resolve_experiment_config(
    model_name: str,
    cli_args: Optional[argparse.Namespace] = None,
    use_tuned: bool = False,
    **overrides: Any,
) -> Dict[str, Any]:
    """Bộ giải quyết cấu hình tập trung: CLI > Tuned > Defaults."""
    canonical_model = model_name.lower().strip()
    dropout_default = 0.3 if canonical_model in ("transfer", "model3", "base") else 0.4

    resolved = {
        "epochs": config.DEFAULT_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "optimizer": "adamw",
        "weight_decay": config.WEIGHT_DECAY,
        "dropout": dropout_default,
        "backbone": "resnet50",
    }
    source = {k: "default" for k in resolved}

    if use_tuned:
        tuned_params = load_best_hparams(canonical_model) or {}
        if tuned_params:
            for param, key in [
                ("batch_size", "batch_size"),
                ("learning_rate", "lr"),
                ("optimizer", "optimizer"),
                ("weight_decay", "weight_decay"),
                ("dropout", "dropout"),
            ]:
                if key in tuned_params:
                    resolved[param] = tuned_params[key]
                    source[param] = "tuned"

    if cli_args is not None:
        for p in ["epochs", "batch_size", "optimizer", "weight_decay", "dropout", "backbone"]:
            v = getattr(cli_args, p, None)
            if v is not None:
                resolved[p] = v
                source[p] = "cli"
        if getattr(cli_args, "lr", None) is not None:
            resolved["learning_rate"] = cli_args.lr
            source["learning_rate"] = "cli"

    for k, v in overrides.items():
        if v is not None:
            norm_k = "learning_rate" if k == "lr" else ("optimizer" if k == "optimizer_name" else k)
            if norm_k in resolved:
                resolved[norm_k] = v
                source[norm_k] = "override"

    resolved["use_tuned"] = use_tuned
    resolved["source"] = source
    return resolved


def save_calibrated_thresholds(
    threshold_dict: Dict[str, Any],
    model_name: str,
    seed: int,
    checkpoint_path: Path,
    dataset_meta: Dict[str, Any],
    save_dir: Optional[Path] = None,
) -> Path:
    """Lưu thresholds calibrated vào cùng thư mục với checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if save_dir is None:
        save_dir = checkpoint_path.parent
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_meta.get("is_demo_data", True) and git_worktree_is_dirty():
        raise RuntimeError(
            "[FAIL CLOSED] Working tree của Git đang có thay đổi chưa commit!\n"
            "Giao thức P1 bắt buộc working tree phải sạch khi calibrate thresholds trên real data."
        )

    ckpt_hash = compute_file_sha256(checkpoint_path)

    device_obj = config.DEVICE
    device_name = torch.cuda.get_device_name(device_obj) if (device_obj.type == "cuda" and torch.cuda.is_available()) else "CPU"

    provenance = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "git_dirty": git_worktree_is_dirty(),
        "model_name": model_name,
        "seed": seed,
        "device": str(device_obj),
        "device_name": device_name,
        "is_multilabel": config.IS_MULTILABEL,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_hash,
        "dataset_fingerprint": dataset_meta["dataset_fingerprint"],
        "data_dir": dataset_meta["data_dir"],
        "data_mode": dataset_meta.get("data_mode", config.DEFAULT_DATA_MODE),
    }

    out_data = {
        "provenance": provenance,
        "thresholds": [round(float(t), 6) for t in threshold_dict["thresholds"]],
        "per_class_calibration": threshold_dict["per_class_calibration"],
    }

    out_file = save_dir / "calibrated_thresholds.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"[calibration] 💾 Đã lưu Calibrated Thresholds tại: {out_file}")
    return out_file


def save_final_test_artifacts(
    metrics_summary: Dict[str, Any],
    per_class_table: List[Dict[str, Any]],
    model_name: str,
    seed: int,
    provenance: Dict[str, Any],
    final_test_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Lưu kết quả final test vào outputs/final_test/{data_mode}/{model_name}_seed{seed}_metrics.json & .csv."""
    if final_test_dir is None:
        data_mode = provenance.get("data_mode")
        final_test_dir = get_final_test_dir(data_mode=data_mode)
    final_test_dir = Path(final_test_dir)
    final_test_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{model_name}_seed{seed}"
    json_path = final_test_dir / f"{base_name}_metrics.json"
    csv_per_class = final_test_dir / f"{base_name}_per_class.csv"

    # 1. Lưu JSON
    full_data = {
        "provenance": provenance,
        "summary_metrics": metrics_summary,
        "per_class_metrics": per_class_table,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)

    # 2. Lưu Per-Class CSV
    headers = [
        "class_id", "class_name", "support_positive", "support_negative",
        "roc_auc", "ap", "optimal_threshold",
        "f1_fixed", "f1_calibrated",
        "accuracy_fixed", "accuracy_calibrated",
        "sensitivity_fixed", "sensitivity_calibrated",
        "specificity_fixed", "specificity_calibrated",
    ]
    with open(csv_per_class, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in per_class_table:
            writer.writerow(row)

    print(f"[final_test] 💾 Đã lưu kết quả Test tại: {json_path.name} & {csv_per_class.name}")
    return json_path, csv_per_class
