"""
run_multi_seed.py - Thực thi quy trình Multi-Seed (3-5 seeds) theo đúng Giao thức Đóng băng P1.
Tách biệt hoàn toàn:
  1. DEVELOP: Huấn luyện, Calibrate và Đóng băng toàn bộ 9 artifacts (3 models x 3 seeds) -> protocol_lock.json
  2. FINAL-TEST: Global Preflight kiểm tra toàn vẹn đủ 9 artifacts -> Tạo Test DataLoader duy nhất -> Đánh giá 9 mô hình -> Bảng so sánh ddof=1.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import config
from data_loader import get_develop_dataloaders, get_test_dataloader
from models import get_model
from train import train
from eval import evaluate_frozen_model
from benchmark_utils import benchmark_model_forward, save_benchmark
from metrics_utils import collect_predictions, calibrate_thresholds_from_pr_curve
from experiment_config import (
    DEVELOP_DIR,
    compute_dataset_fingerprint,
    git_worktree_is_dirty,
    resolve_experiment_config,
    save_calibrated_thresholds,
    generate_protocol_lock,
    global_preflight_check,
)
from compare_models import generate_comparison_table


DEFAULT_SEEDS = list(config.CANONICAL_SEEDS)
DEFAULT_MODELS = list(config.CANONICAL_MODELS)


def run_multi_seed_develop(
    models: List[str] = DEFAULT_MODELS,
    seeds: List[int] = DEFAULT_SEEDS,
    cli_args: Optional[argparse.Namespace] = None,
    data_dir: Optional[str] = None,
    data_mode: Optional[str] = None,
) -> Path:
    """
    PHASE 1 (DEVELOP): Huấn luyện toàn bộ models x seeds, calibrate trên validation,
    sinh benchmark và khóa manifest bằng protocol_lock.json.
    Tuyệt đối KHÔNG mở hoặc tạo Test DataLoader!
    """
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    print("\n" + "=" * 80)
    print(f"      PHASE 1: MULTI-SEED DEVELOPMENT & CALIBRATION (mode={mode})      ")
    print("=" * 80)
    print(f"Mô hình : {models}")
    print(f"Seeds   : {seeds} (Tổng cộng: {len(models) * len(seeds)} thí nghiệm)")
    print("=" * 80)

    dataset_meta = compute_dataset_fingerprint(data_dir=data_dir, data_mode=mode)
    if not dataset_meta["is_demo_data"] and git_worktree_is_dirty():
        raise RuntimeError(
            "[FAIL CLOSED] Working tree của Git đang có thay đổi chưa commit!\n"
            "Giao thức P1 bắt buộc working tree phải sạch (git status clean) khi chạy develop trên dữ liệu thật."
        )

    device = config.DEVICE

    for model_name in models:
        # Benchmark model forward 1 lần cho kiến trúc (hằng số kiến trúc)
        model_benchmarked = False

        for seed in seeds:
            print("\n" + "#" * 80)
            print(f"### [DEVELOP] MÔ HÌNH: {model_name.upper()} | SEED: {seed} ###")
            print("#" * 80)

            # 1. Resolve Config
            resolved = resolve_experiment_config(
                model_name=model_name,
                cli_args=cli_args,
                use_tuned=getattr(cli_args, "use_tuned", False) if cli_args else False,
            )

            # 2. Train
            ckpt_path = train(
                model_name=model_name,
                epochs=resolved["epochs"],
                batch_size=resolved["batch_size"],
                learning_rate=resolved["learning_rate"],
                optimizer=resolved["optimizer"],
                weight_decay=resolved["weight_decay"],
                dropout=resolved["dropout"],
                data_dir=data_dir,
                backbone_name=resolved["backbone"],
                seed=seed,
                use_tuned=resolved["use_tuned"],
                parameter_sources=resolved.get("source"),
                data_mode=mode,
            )

            # 3. Load Best Checkpoint & Calibrate on Validation Only
            print(f">>> [DEVELOP] Nạp Checkpoint {ckpt_path.name} để Calibrate Ngưỡng...")
            _, val_loader, class_names, _ = get_develop_dataloaders(
                data_dir=data_dir, data_mode=mode, batch_size=resolved["batch_size"], seed=seed
            )

            ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
            model_kwargs = {}
            if resolved.get("backbone"):
                model_kwargs["backbone_name"] = resolved["backbone"]
            if resolved.get("dropout") is not None:
                model_kwargs["dropout"] = resolved["dropout"]
            model_kwargs["pretrained"] = False
            model_kwargs["freeze_base"] = False

            model = get_model(model_name, num_classes=len(class_names), **model_kwargs).to(device)
            model.load_state_dict(ckpt_data["model_state_dict"])
            model.eval()

            y_val_true, y_val_prob, _ = collect_predictions(model, val_loader, device=device)
            calib_res = calibrate_thresholds_from_pr_curve(y_val_true, y_val_prob, class_names=class_names)
            save_calibrated_thresholds(
                threshold_dict=calib_res,
                model_name=model_name,
                seed=seed,
                checkpoint_path=ckpt_path,
                dataset_meta=dataset_meta,
            )

            if not model_benchmarked:
                print(f">>> [DEVELOP] Chạy Standardized Inference Benchmark cho {model_name}...")
                bench_data = benchmark_model_forward(model, device=device)
                save_benchmark(bench_data, model_name=model_name)
                model_benchmarked = True

    # 4. KHÓA TẤT CẢ VÀ TẠO MANIFEST protocol_lock.json
    print("\n" + "=" * 80)
    print("🔒 TIẾN HÀNH ĐÓNG BĂNG TOÀN BỘ CÁC THÍ NGHIỆM VÀ TẠO protocol_lock.json...")
    print("=" * 80)
    lock_file = generate_protocol_lock(model_names=models, seeds=seeds, data_dir=data_dir, data_mode=mode)
    print(f"✅ ĐÃ KHÓA THÀNH CÔNG: {lock_file}")
    return lock_file


def run_multi_seed_final_test(
    models: List[str] = DEFAULT_MODELS,
    seeds: List[int] = DEFAULT_SEEDS,
    data_dir: Optional[str] = None,
    batch_size: int = config.BATCH_SIZE,
    data_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    PHASE 2 (FINAL-TEST):
    1. Global Preflight Check trên toàn bộ 9 artifacts.
    2. Nếu PASS 100% -> Tạo Test DataLoader duy nhất.
    3. Đánh giá tuần tự từng model và seed.
    4. Sinh bảng so sánh tổng hợp với mean ± std (ddof=1).
    """
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    print("\n" + "=" * 80)
    print(f"      PHASE 2: GLOBAL PREFLIGHT & FINAL TEST EVALUATION (mode={mode})      ")
    print("=" * 80)

    # 1. BẮT BUỘC: GLOBAL PREFLIGHT CHECK TRÊN TOÀN BỘ 9 THÍ NGHIỆM
    print(">>> [FINAL-TEST] Kiểm tra Preflight toàn bộ artifacts trước khi tạo DataLoader...")
    lock_file = DEVELOP_DIR / "protocol_lock.json"
    global_preflight_check(data_dir=data_dir, lock_file=lock_file, data_mode=mode, enforce_clean_git=(mode == "real"))

    # 2. CHỈ TẠO TEST DATALOADER SAU KHI PREFLIGHT ĐÃ PASS
    print("\n>>> [FINAL-TEST] Toàn bộ 9 thí nghiệm đã hợp lệ. Khởi tạo Test DataLoader...")
    test_loader, class_names = get_test_dataloader(data_dir=data_dir, data_mode=mode, batch_size=batch_size)

    device = config.DEVICE

    # 3. Đánh giá toàn bộ các mô hình đã đóng băng
    for model_name in models:
        for seed in seeds:
            print("\n" + "-" * 70)
            print(f"Đánh giá Test: {model_name.upper()} (Seed = {seed})")
            print("-" * 70)

            ckpt_path = DEVELOP_DIR / model_name / f"seed{seed}" / "best.pth"
            th_path = DEVELOP_DIR / model_name / f"seed{seed}" / "calibrated_thresholds.json"

            evaluate_frozen_model(
                model_name=model_name,
                checkpoint_path=ckpt_path,
                threshold_path=th_path,
                test_loader=test_loader,
                class_names=class_names,
                seed=seed,
                device=device,
                data_dir=data_dir,
                save_artifacts=True,
                data_mode=mode,
            )

    # 4. Sinh bảng so sánh tổng hợp
    print("\n" + "=" * 80)
    print("TỔNG HỢP KẾT QUẢ ĐA SEED VÀ XUẤT BẢNG SO SÁNH")
    print("=" * 80)
    summary_table = generate_comparison_table(models=models, seeds=seeds)
    return summary_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Seed Protocol Pipeline (P1 Fully Locked)")
    parser.add_argument("--phase", type=str, default="develop", choices=["develop", "final-test", "all"],
                        help="'develop' (huấn luyện & đóng băng), 'final-test' (chạy test), 'all' (chỉ demo)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=["simple", "complex", "transfer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use_tuned", action="store_true", default=False)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--data_mode", type=str, default=config.DEFAULT_DATA_MODE, choices=["real", "demo"])

    args = parser.parse_args()

    # Guard chống Data Leakage trên Real Data
    if args.phase == "all" and args.data_mode == "real":
        raise ValueError(
            "\n[BẢO VỆ GIAO THỨC] CẤM dùng '--phase all' trên dataset thật để tránh rò rỉ dữ liệu test.\n"
            "Quy trình khoa học bắt buộc:\n"
            "  1. python run_multi_seed.py --phase develop --data_mode real ...\n"
            "  2. python run_multi_seed.py --phase final-test --data_mode real ..."
        )

    if args.phase in ("develop", "all"):
        run_multi_seed_develop(
            models=args.models,
            seeds=args.seeds,
            cli_args=args,
            data_dir=args.data_dir,
            data_mode=args.data_mode,
        )

    if args.phase in ("final-test", "all"):
        run_multi_seed_final_test(
            models=args.models,
            seeds=args.seeds,
            data_dir=args.data_dir,
            batch_size=args.batch_size or config.BATCH_SIZE,
            data_mode=args.data_mode,
        )