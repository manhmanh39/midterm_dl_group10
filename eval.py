"""
eval.py - Đánh giá mô hình đã đóng băng (Frozen Model Evaluation) theo Giao thức P1.
Thực thi suy luận 1-pass trên tập Test độc lập, kiểm tra chặt chẽ tính toàn vẹn (Fail-Closed Provenance),
áp dụng song song Ngưỡng cố định (Fixed-0.5) và Ngưỡng hiệu chuẩn (Calibrated T*),
đo lường mâu thuẫn No-Finding và báo cáo chỉ số 14 pathologies riêng biệt.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_loader import get_test_dataloader
from models import get_model
from experiment_config import (
    DEVELOP_DIR,
    FINAL_TEST_DIR,
    compute_file_sha256,
    compute_dataset_fingerprint,
    get_git_commit,
    git_worktree_is_dirty,
    global_preflight_check,
    verify_fail_closed_provenance,
    save_final_test_artifacts,
)
from metrics_utils import (
    apply_thresholds,
    measure_no_finding_inconsistency,
    enforce_no_finding_consistency,
    compute_per_class_table,
    compute_macro_metrics,
)


def evaluate_frozen_model(
    model_name: str,
    checkpoint_path: Path,
    threshold_path: Path,
    test_loader: DataLoader,
    class_names: List[str],
    seed: int,
    device: torch.device = config.DEVICE,
    data_dir: Optional[str] = None,
    save_artifacts: bool = True,
    data_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Thực thi 1-pass đánh giá trên tập Test cho một mô hình đã đóng băng.
    """
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    checkpoint_path = Path(checkpoint_path)
    threshold_path = Path(threshold_path)

    # 1. Kiểm tra Provenance nội bộ
    dataset_meta = compute_dataset_fingerprint(data_dir=data_dir, data_mode=mode)
    verify_fail_closed_provenance(checkpoint_path, threshold_path, dataset_meta)

    # 2. Đọc threshold calibrated đã lưu
    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_json = json.load(f)
    calibrated_thresholds = np.array(threshold_json["thresholds"], dtype=np.float32)

    # 3. Nạp model từ frozen checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", len(class_names))
    backbone_name = checkpoint.get("backbone_name")
    dropout = checkpoint.get("dropout")

    model_kwargs: Dict[str, Any] = {}
    if backbone_name:
        model_kwargs["backbone_name"] = backbone_name
    if dropout is not None:
        model_kwargs["dropout"] = dropout
    model_kwargs["pretrained"] = False
    model_kwargs["freeze_base"] = False

    model = get_model(model_name, num_classes=num_classes, **model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\n[eval] 🧪 Bắt đầu 1-pass Test Inference cho {model_name.upper()} (Seed={seed})...")

    # 4. Thu thập toàn bộ predictions & ground truth
    all_probs = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc=f"  [Test {model_name}]", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # 5. Phân nhánh nhị phân hóa: Fixed 0.5 vs Calibrated T*
    fixed_thresholds = np.full(num_classes, 0.5, dtype=np.float32)
    y_pred_fixed_raw = apply_thresholds(y_prob, fixed_thresholds)
    y_pred_cal_raw = apply_thresholds(y_prob, calibrated_thresholds)

    # 6. Đo lường mâu thuẫn No-Finding nguyên bản (Raw Inconsistency)
    incons_fixed_raw = measure_no_finding_inconsistency(y_pred_fixed_raw)
    incons_cal_raw = measure_no_finding_inconsistency(y_pred_cal_raw)

    # 7. Áp dụng quy tắc hiệu chỉnh nhất quán No-Finding (Adjusted Consistency)
    y_pred_fixed_adj = enforce_no_finding_consistency(y_pred_fixed_raw)
    y_pred_cal_adj = enforce_no_finding_consistency(y_pred_cal_raw)

    # 8. Tính bảng Per-Class Table & Macro Metrics
    per_class_table = compute_per_class_table(y_true, y_prob, calibrated_thresholds, class_names)
    macro_metrics = compute_macro_metrics(per_class_table)

    # 9. Bổ sung các chỉ số Inconsistency, Exact Match, và Per-Label Accuracy
    exact_match_fixed = float((y_true == y_pred_fixed_raw).all(axis=1).mean())
    exact_match_cal = float((y_true == y_pred_cal_raw).all(axis=1).mean())
    per_label_fixed = float((y_true == y_pred_fixed_raw).mean())
    per_label_cal = float((y_true == y_pred_cal_raw).mean())

    macro_metrics.update({
        "model_name": model_name,
        "seed": seed,
        "exact_match_accuracy_fixed": round(exact_match_fixed, 4),
        "exact_match_accuracy_calibrated": round(exact_match_cal, 4),
        "per_label_accuracy_fixed": round(per_label_fixed, 4),
        "per_label_accuracy_calibrated": round(per_label_cal, 4),
        "contradiction_rate_raw_fixed": round(incons_fixed_raw["contradiction_rate"], 4),
        "empty_diagnosis_rate_raw_fixed": round(incons_fixed_raw["empty_diagnosis_rate"], 4),
        "total_inconsistency_rate_raw_fixed": round(incons_fixed_raw["total_inconsistency_rate"], 4),
        "contradiction_rate_raw_calibrated": round(incons_cal_raw["contradiction_rate"], 4),
        "empty_diagnosis_rate_raw_calibrated": round(incons_cal_raw["empty_diagnosis_rate"], 4),
        "total_inconsistency_rate_raw_calibrated": round(incons_cal_raw["total_inconsistency_rate"], 4),
        "total_inconsistency_rate_adjusted": 0.0,
    })

    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    device_name = torch.cuda.get_device_name(device_obj) if (device_obj.type == "cuda" and torch.cuda.is_available()) else "CPU"

    # 10. Tạo Provenance đầy đủ
    from datetime import datetime
    provenance = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "git_dirty": git_worktree_is_dirty(),
        "model_name": model_name,
        "seed": seed,
        "device": str(device_obj),
        "device_name": device_name,
        "is_multilabel": config.IS_MULTILABEL,
        "checkpoint_sha256": compute_file_sha256(checkpoint_path),
        "threshold_sha256": compute_file_sha256(threshold_path),
        "dataset_fingerprint": dataset_meta["dataset_fingerprint"],
        "data_dir": dataset_meta.get("data_dir"),
    }

    # 11. Lưu Artifacts
    if save_artifacts:
        save_final_test_artifacts(
            metrics_summary=macro_metrics,
            per_class_table=per_class_table,
            model_name=model_name,
            seed=seed,
            provenance=provenance,
        )

    # In kết quả tóm tắt
    print(f"\n{'='*70}")
    print(f"KẾT QUẢ FINAL TEST: {model_name.upper()} (Seed={seed})")
    print(f"{'='*70}")
    print(f"  Macro-14 ROC-AUC              : {macro_metrics['macro_auc_14']} (valid={macro_metrics['valid_classes_auc_14']}/14)")
    print(f"  Macro-14 Average Precision (AP): {macro_metrics['macro_ap_14']} (valid={macro_metrics['valid_classes_ap_14']}/14)")
    print(f"  Macro-14 F1 (Fixed 0.5)        : {macro_metrics['macro_f1_14_fixed']}")
    print(f"  Macro-14 F1 (Calibrated T*)    : {macro_metrics['macro_f1_14_calibrated']}")
    print(f"  Per-Label Acc (Fixed 0.5)      : {macro_metrics['per_label_accuracy_fixed']*100:.2f}%")
    print(f"  Per-Label Acc (Calibrated T*)  : {macro_metrics['per_label_accuracy_calibrated']*100:.2f}%")
    print(f"  Exact Match Acc (Fixed 0.5)    : {macro_metrics['exact_match_accuracy_fixed']*100:.2f}%")
    print(f"  Exact Match Acc (Calibrated T*): {macro_metrics['exact_match_accuracy_calibrated']*100:.2f}%")
    print(f"  Mean Sens (Calibrated T*)      : {macro_metrics['mean_sensitivity_14_calibrated']}")
    print(f"  Mean Spec (Calibrated T*)      : {macro_metrics['mean_specificity_14_calibrated']}")
    print(f"  Raw Inconsistency (Fixed)      : {macro_metrics['total_inconsistency_rate_raw_fixed']*100:.2f}%")
    print(f"  Raw Inconsistency (Calibrated) : {macro_metrics['total_inconsistency_rate_raw_calibrated']*100:.2f}%")
    print(f"  Adjusted Inconsistency         : {macro_metrics['total_inconsistency_rate_adjusted']*100:.2f}% (Invariant Enforced)")
    print(f"{'='*70}")

    return macro_metrics


def evaluate_model(
    model_name: str = "transfer",
    checkpoint_path: Optional[str] = None,
    threshold_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
    device: torch.device = config.DEVICE,
    enforce_preflight: bool = True,
    data_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Hàm entry point kiểm tra tính toàn vẹn Fail-Closed trước khi tạo DataLoader.
    """
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    if checkpoint_path is None:
        ckpt_file = DEVELOP_DIR / model_name / f"seed{seed}" / "best.pth"
    else:
        ckpt_file = Path(checkpoint_path)

    if threshold_path is None:
        th_file = DEVELOP_DIR / model_name / f"seed{seed}" / "calibrated_thresholds.json"
    else:
        th_file = Path(threshold_path)

    # GLOBAL PREFLIGHT CHECK TRƯỚC KHI MỞ TEST DATALOADER
    dataset_meta = compute_dataset_fingerprint(data_dir=data_dir, data_mode=mode)
    is_demo = dataset_meta.get("is_demo_data", True)

    if not is_demo:
        if not enforce_preflight:
            raise RuntimeError(
                "[FAIL CLOSED] Giao thức cấm bỏ qua preflight (--skip_preflight) khi chạy Final-Test trên tập dữ liệu thật!"
            )
        lock_file = DEVELOP_DIR / "protocol_lock.json"
        if not lock_file.exists():
            raise FileNotFoundError(
                f"\n[FAIL CLOSED] Không tìm thấy file khóa giao thức bắt buộc: {lock_file}!\n"
                "Trên tập dữ liệu thật, Giao thức P1 yêu cầu PHẢI hoàn tất phase develop cho toàn bộ "
                "9 thí nghiệm (3 models x 3 seeds) và đóng băng qua protocol_lock.json TRƯỚC KHI mở Final-Test DataLoader."
            )
        global_preflight_check(data_dir=data_dir, lock_file=lock_file, enforce_clean_git=True, data_mode=mode)
    else:
        # Trong môi trường demo/debug data
        if enforce_preflight:
            lock_file = DEVELOP_DIR / "protocol_lock.json"
            if lock_file.exists():
                global_preflight_check(data_dir=data_dir, lock_file=lock_file, enforce_clean_git=False, data_mode=mode)
            else:
                verify_fail_closed_provenance(ckpt_file, th_file, dataset_meta)
        else:
            verify_fail_closed_provenance(ckpt_file, th_file, dataset_meta)

    # TẠO TEST DATALOADER CHỈ KHI PREFLIGHT ĐÃ PASS 100%
    test_loader, class_names = get_test_dataloader(data_dir=data_dir, data_mode=mode, batch_size=batch_size)

    return evaluate_frozen_model(
        model_name=model_name,
        checkpoint_path=ckpt_file,
        threshold_path=th_file,
        test_loader=test_loader,
        class_names=class_names,
        seed=seed,
        device=device,
        data_dir=data_dir,
        save_artifacts=True,
        data_mode=mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P1 Final Test Evaluation")
    parser.add_argument("--model", type=str, default="transfer", choices=["simple", "complex", "transfer"])
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--data_mode", type=str, default=config.DEFAULT_DATA_MODE, choices=["real", "demo"])
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--skip_preflight", action="store_true", default=False)

    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        threshold_path=args.threshold,
        data_dir=args.data_dir,
        data_mode=args.data_mode,
        batch_size=args.batch_size,
        seed=args.seed,
        enforce_preflight=not args.skip_preflight,
    )