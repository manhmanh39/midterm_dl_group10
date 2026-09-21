"""
main.py - Điểm khởi chạy chính của Pipeline Phân loại VinBigData Chest X-ray.
Phân tách nghiêm ngặt:
  - Phase Develop : Train -> Val Calibration -> Standardized Benchmark (Tuyệt đối không chạm tập Test)
  - Phase Final-Test: Global Preflight -> Single-pass Frozen Inference -> Comparison Table
  - Guard: Cấm sử dụng '--phase all' trên tập dữ liệu thật để loại bỏ hoàn toàn nguy cơ Test Leakage.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import config
from data_loader import get_dataloaders
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
    global_preflight_check,
)
from compare_models import generate_comparison_table


def run_develop_phase(
    model_names: List[str],
    cli_args: argparse.Namespace,
    seed: int = config.SEED,
    data_dir: Optional[str] = None,
) -> None:
    """
    PHASE 1: DEVELOP (Độc lập hoàn toàn với Test Set)
    """
    print("\n" + "=" * 80)
    print(f"BẮT ĐẦU GIAI ĐOẠN DEVELOP (Seed = {seed})")
    print("=" * 80)

    dataset_meta = compute_dataset_fingerprint(data_dir=data_dir)
    print(f"[develop] Dataset Fingerprint: {dataset_meta['dataset_fingerprint']}")
    print(f"[develop] Demo Mode          : {dataset_meta['is_demo_data']}\n")

    # Kiểm tra clean git tree nếu chạy trên real data
    if not dataset_meta["is_demo_data"] and git_worktree_is_dirty():
        raise RuntimeError(
            "[FAIL CLOSED] Working tree của Git đang có thay đổi chưa commit!\n"
            "Giao thức P1 bắt buộc working tree phải sạch (git status clean) khi chạy develop trên dữ liệu thật."
        )

    device = config.DEVICE

    for model_name in model_names:
        print("\n" + "#" * 80)
        print(f"### [DEVELOP] TIẾN HÀNH CHO MÔ HÌNH: {model_name.upper()} ###")
        print("#" * 80)

        # 1. Giải quyết cấu hình
        resolved = resolve_experiment_config(
            model_name=model_name,
            cli_args=cli_args,
            use_tuned=cli_args.use_tuned,
        )

        # 2. Huấn luyện và lưu Checkpoint tốt nhất trên Validation
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
            use_tuned=cli_args.use_tuned,
        )

        # 3. Nạp lại Checkpoint tốt nhất để Calibrate trên tập Validation
        print(f"\n>>> [DEVELOP] Nạp Checkpoint tốt nhất để Calibrate Ngưỡng: {ckpt_path.name}")
        _, val_loader, _, class_names, _ = get_dataloaders(
            data_dir=data_dir, batch_size=resolved["batch_size"], seed=seed
        )

        checkpoint_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_kwargs = {}
        if resolved.get("backbone"):
            model_kwargs["backbone_name"] = resolved["backbone"]
        if resolved.get("dropout") is not None:
            model_kwargs["dropout"] = resolved["dropout"]

        model = get_model(model_name, num_classes=len(class_names), **model_kwargs).to(device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model.eval()

        print(f">>> [DEVELOP] Thu thập Validation Predictions để tìm T* theo PR Curve...")
        y_val_true, y_val_prob, _ = collect_predictions(model, val_loader, device=device)

        # 4. Hiệu chuẩn ngưỡng theo PR Curve
        calib_res = calibrate_thresholds_from_pr_curve(y_val_true, y_val_prob, class_names=class_names)
        save_calibrated_thresholds(
            threshold_dict=calib_res,
            model_name=model_name,
            seed=seed,
            checkpoint_path=ckpt_path,
            dataset_meta=dataset_meta,
        )

        # 5. Chạy Standardized Inference Benchmark
        print(f">>> [DEVELOP] Chạy Standardized Inference Benchmark (BS=1 Latency & BS=16 Throughput)...")
        bench_data = benchmark_model_forward(model, device=device)
        save_benchmark(bench_data, model_name=model_name)

    print("\n" + "=" * 80)
    print("✅ GIAI ĐOẠN DEVELOP HOÀN TẤT CHO TẤT CẢ MÔ HÌNH ĐÃ CHỌN!")
    print("   (Không có bất kỳ dữ liệu Test nào bị rò rỉ hay truy cập trong pha này)")
    print("=" * 80)


def run_final_test_phase(
    model_names: List[str],
    seed: int = config.SEED,
    data_dir: Optional[str] = None,
    batch_size: int = config.BATCH_SIZE,
) -> None:
    """
    PHASE 2: FINAL TEST (Đánh giá các mô hình đã đóng băng hoàn toàn)
    """
    print("\n" + "=" * 80)
    print(f"BẮT ĐẦU GIAI ĐOẠN FINAL TEST (Seed = {seed})")
    print("=" * 80)

    # 1. BẮT BUỘC: GLOBAL PREFLIGHT CHECK TRƯỚC KHI MỞ TEST DATALOADER
    print(">>> [FINAL-TEST] Thực hiện Global Preflight Check trên toàn bộ artifacts đã khóa...")
    lock_file = DEVELOP_DIR / "protocol_lock.json"
    if lock_file.exists():
        global_preflight_check(data_dir=data_dir, lock_file=lock_file)
    else:
        # Nếu chưa tạo manifest 9 seed, kiểm tra dataset và working tree
        dataset_meta = compute_dataset_fingerprint(data_dir=data_dir)
        if not dataset_meta["is_demo_data"] and git_worktree_is_dirty():
            raise RuntimeError(
                "[FAIL CLOSED] Working tree của Git đang có thay đổi chưa commit!\n"
                "Giao thức P1 bắt buộc working tree phải sạch khi chạy Final-Test trên real data."
            )

    # 2. CHỈ TẠO TEST DATALOADER SAU KHI PREFLIGHT ĐÃ HOÀN TOÀN HỢP LỆ
    print("\n>>> [FINAL-TEST] Tạo Test DataLoader độc lập...")
    _, _, test_loader, class_names, _ = get_dataloaders(data_dir=data_dir, batch_size=batch_size, seed=seed)

    test_results = []
    device = config.DEVICE

    for model_name in model_names:
        ckpt_path = DEVELOP_DIR / model_name / f"seed{seed}" / "best.pth"
        th_path = DEVELOP_DIR / model_name / f"seed{seed}" / "calibrated_thresholds.json"

        if not ckpt_path.exists() or not th_path.exists():
            raise FileNotFoundError(
                f"[FINAL-TEST ERROR] Thiếu artifact cho {model_name} (seed={seed})!\n"
                f"  Checkpoint: {ckpt_path}\n"
                f"  Thresholds: {th_path}"
            )

        metrics = evaluate_frozen_model(
            model_name=model_name,
            checkpoint_path=ckpt_path,
            threshold_path=th_path,
            test_loader=test_loader,
            class_names=class_names,
            seed=seed,
            device=device,
            data_dir=data_dir,
            save_artifacts=True,
        )
        test_results.append(metrics)

    # 3. Xuất bảng so sánh tổng hợp
    try:
        generate_comparison_table(models=model_names, seeds=[seed])
    except Exception as e:
        print(f"[main] Lưu ý khi tạo bảng so sánh: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline phân loại X-quang VinBigData theo Giao thức P1")
    parser.add_argument(
        "--phase",
        type=str,
        default="develop",
        choices=["develop", "final-test", "all"],
        help="Phase thực thi: 'develop' (không mở test), 'final-test' (chạy test đóng băng), 'all' (chỉ cho demo/CI)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["simple", "complex", "transfer", "all"],
    )
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use_tuned", action="store_true", default=False)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)

    args = parser.parse_args()
    selected_models = ["simple", "complex", "transfer"] if args.model == "all" else [args.model]

    # KHÓA BẢO VỆ CHỐNG DATA LEAKAGE TRÊN DATA THẬT
    dataset_meta = compute_dataset_fingerprint(data_dir=args.data_dir)
    if args.phase == "all" and not dataset_meta["is_demo_data"]:
        raise ValueError(
            "\n[BẢO VỆ GIAO THỨC] CẤM dùng '--phase all' trên dataset thật để tránh rò rỉ dữ liệu test.\n"
            "Quy trình khoa học bắt buộc:\n"
            "  1. python main.py --phase develop --model ...\n"
            "  2. python main.py --phase final-test --model ..."
        )

    if args.phase in ("develop", "all"):
        run_develop_phase(
            model_names=selected_models,
            cli_args=args,
            seed=args.seed,
            data_dir=args.data_dir,
        )

    if args.phase in ("final-test", "all"):
        run_final_test_phase(
            model_names=selected_models,
            seed=args.seed,
            data_dir=args.data_dir,
            batch_size=args.batch_size or config.BATCH_SIZE,
        )