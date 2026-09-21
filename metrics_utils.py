"""
metrics_utils.py - Các thuật toán đo lường, threshold calibration, 
báo cáo 14-pathology và kiểm soát nhất quán No-Finding theo Giao thức Chuẩn P1.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch
from tqdm import tqdm

import config


def safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Tính Average Precision (AP) an toàn theo protocol policy của dự án.
    Nếu ground-truth chỉ có 1 class (toàn 0 hoặc toàn 1), tự động trả về NaN.
    """
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Tính ROC-AUC an toàn theo protocol policy của dự án.
    Nếu ground-truth chỉ có 1 class (toàn 0 hoặc toàn 1), tự động trả về NaN.
    """
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def compute_safe_macro_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_indices: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    P1.6: Tính Macro-AUC an toàn cho danh sách class_indices (mặc định 14 pathologies 0..13).
    - Trả về {'macro_auc': float, 'valid_classes': int, 'per_class_auc': Dict[int, Optional[float]]}
    - Nếu valid_classes == 0: macro_auc = float('nan'), valid_classes = 0.
    - Nếu có một số lớp thiếu support: nanmean trên các lớp hợp lệ và báo valid_classes.
    """
    if y_true.ndim != 2 or y_prob.ndim != 2:
        raise ValueError("y_true and y_prob must be 2-dimensional arrays.")
    if class_indices is None:
        class_indices = list(range(config.NO_FINDING_CLASS_ID))

    valid_aucs: List[float] = []
    per_class: Dict[int, Optional[float]] = {}

    for c in class_indices:
        auc = safe_roc_auc(y_true[:, c], y_prob[:, c])
        if not np.isnan(auc):
            valid_aucs.append(auc)
            per_class[c] = round(float(auc), 4)
        else:
            per_class[c] = None

    if len(valid_aucs) == 0:
        return {
            "macro_auc": float("nan"),
            "valid_classes": 0,
            "per_class_auc": per_class,
        }

    macro_auc = float(np.mean(valid_aucs))
    return {
        "macro_auc": round(macro_auc, 4),
        "valid_classes": len(valid_aucs),
        "per_class_auc": per_class,
    }


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str = "[Inference]",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Thu thập xác suất (sigmoid) và nhãn thực tế từ DataLoader trong 1 lần forward duy nhất.
    Trả về: (y_true, y_prob, average_loss)
    """
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if config.IS_MULTILABEL
        else torch.nn.CrossEntropyLoss()
    )

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs) if config.IS_MULTILABEL else torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / max(1, len(loader.dataset))
    return y_true, y_prob, float(avg_loss)


def calibrate_thresholds_from_pr_curve(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    P1.5: Calibration threshold trên tập validation dựa trên Precision-Recall curve.
    - Validate y_val_true chỉ chứa {0, 1}
    - Validate y_val_prob hữu hạn (finite)
    - Fallback về 0.5 (val_f1=None, status='fallback') CHỈ khi n_pos == 0 hoặc n_neg == 0.
    - Nếu n_pos > 0 và n_neg > 0: bất kỳ trường hợp nào như thresholds rỗng, f1 non-positive/NaN,
      hoặc threshold không finite đều raise ValueError.
    - KHÔNG clip về [0.01, 0.99].
    - Deterministic tie-breaking (chọn ngưỡng gần 0.5 nhất khi F1 bằng nhau).
    """
    if y_val_true.ndim != 2 or y_val_prob.ndim != 2:
        raise ValueError("y_val_true and y_val_prob must be 2-dimensional arrays.")
    if y_val_true.shape != y_val_prob.shape:
        raise ValueError(f"Shape mismatch: y_val_true {y_val_true.shape} != y_val_prob {y_val_prob.shape}")

    # Validate binary values
    if not np.all(np.isin(y_val_true, [0, 1])):
        raise ValueError("y_val_true must contain only binary values 0 and 1.")

    # Validate finite probabilities
    if not np.all(np.isfinite(y_val_prob)):
        raise ValueError("y_val_prob contains non-finite values (NaN or Inf).")

    num_classes = y_val_true.shape[1]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    threshold_vector = np.full(num_classes, 0.5, dtype=np.float32)
    per_class_calibrations: List[Dict[str, Any]] = []

    for c in range(num_classes):
        y_c = y_val_true[:, c]
        p_c = y_val_prob[:, c]

        n_pos = int((y_c == 1.0).sum())
        n_neg = int((y_c == 0.0).sum())

        # Fallback duy nhất khi thiếu positive hoặc negative support
        if n_pos == 0 or n_neg == 0:
            reason = "no_positive_samples" if n_pos == 0 else "no_negative_samples"
            per_class_calibrations.append({
                "class_id": c,
                "class_name": class_names[c],
                "status": "fallback",
                "reason": reason,
                "threshold": 0.5,
                "val_f1": None,
                "support_positive": n_pos,
                "support_negative": n_neg,
                "val_precision": None,
                "val_recall": None,
            })
            threshold_vector[c] = 0.5
            continue

        precision, recall, thresholds = precision_recall_curve(y_c, p_c)

        if len(thresholds) == 0:
            raise ValueError(
                f"[CALIBRATION ERROR] PR curve thresholds array is empty for class {c} ({class_names[c]}) "
                f"despite valid positive ({n_pos}) and negative ({n_neg}) support!"
            )

        prec_candidates = precision[:-1]
        rec_candidates = recall[:-1]

        denom = prec_candidates + rec_candidates
        f1 = np.where(denom > 1e-12, (2.0 * prec_candidates * rec_candidates) / (denom + 1e-12), 0.0)

        max_f1 = np.nanmax(f1)
        if np.isnan(max_f1) or max_f1 <= 0.0:
            raise ValueError(
                f"[CALIBRATION ERROR] Abnormal PR curve: max F1 is {max_f1} for class {c} ({class_names[c]}) "
                f"despite valid positive ({n_pos}) and negative ({n_neg}) support!"
            )

        # Deterministic tie-breaking: Chọn vị trí gần 0.5 nhất
        candidate_indices = np.flatnonzero(np.isclose(f1, max_f1, rtol=1e-7, atol=1e-9))
        closest_idx = candidate_indices[np.argmin(np.abs(thresholds[candidate_indices] - 0.5))]
        optimal_thresh = float(thresholds[closest_idx])
        if not np.isfinite(optimal_thresh):
            raise ValueError(
                f"[CALIBRATION ERROR] Optimal threshold is non-finite ({optimal_thresh}) for class {c} ({class_names[c]})!"
            )

        chosen_f1 = float(f1[closest_idx])
        chosen_prec = float(prec_candidates[closest_idx])
        chosen_rec = float(rec_candidates[closest_idx])

        # Strict: Không clip ngưỡng
        threshold_vector[c] = optimal_thresh

        per_class_calibrations.append({
            "class_id": c,
            "class_name": class_names[c],
            "status": "calibrated",
            "reason": None,
            "threshold": round(optimal_thresh, 6),
            "val_f1": round(chosen_f1, 4),
            "support_positive": n_pos,
            "support_negative": n_neg,
            "val_precision": round(chosen_prec, 4),
            "val_recall": round(chosen_rec, 4),
        })

    return {
        "thresholds": threshold_vector,
        "per_class_calibration": per_class_calibrations,
    }


def apply_thresholds(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Áp dụng semantics nhất quán trên toàn bộ dự án: prob >= threshold."""
    return (y_prob >= thresholds).astype(np.float32)


def measure_no_finding_inconsistency(preds: np.ndarray) -> Dict[str, float]:
    """
    Đo đạc độc lập 2 loại mâu thuẫn nhãn No-Finding trên 15 classes:
      - contradiction: có ít nhất 1 pathology (0..13) == 1 VÀ No Finding (14) == 1
      - empty_diagnosis: toàn bộ pathologies (0..13) == 0 VÀ No Finding (14) == 0
      - total: union của contradiction và empty_diagnosis
    """
    n_samples = len(preds)
    if n_samples == 0:
        return {
            "contradiction_rate": 0.0,
            "empty_diagnosis_rate": 0.0,
            "total_inconsistency_rate": 0.0,
        }

    pathology_preds = preds[:, :config.NO_FINDING_CLASS_ID]
    no_finding_preds = preds[:, config.NO_FINDING_CLASS_ID]

    has_pathology = (pathology_preds.sum(axis=1) >= 1.0)
    has_no_finding = (no_finding_preds >= 1.0)

    contradiction = has_pathology & has_no_finding
    empty_diagnosis = (~has_pathology) & (~has_no_finding)
    total_inconsistent = contradiction | empty_diagnosis

    return {
        "contradiction_rate": float(contradiction.mean()),
        "empty_diagnosis_rate": float(empty_diagnosis.mean()),
        "total_inconsistency_rate": float(total_inconsistent.mean()),
    }


def enforce_no_finding_consistency(preds: np.ndarray) -> np.ndarray:
    """
    CONSISTENCY-ADJUSTED: Áp dụng quy tắc suy diễn:
    Nếu có bất kỳ pathology nào -> No Finding = 0; ngược lại No Finding = 1.
    Đảm bảo invariant assert total_inconsistency_rate == 0.0.
    """
    adjusted = preds.copy()
    pathology_preds = adjusted[:, :config.NO_FINDING_CLASS_ID]
    has_pathology = (pathology_preds.sum(axis=1) >= 1.0)

    adjusted[:, config.NO_FINDING_CLASS_ID] = np.where(has_pathology, 0.0, 1.0)

    # Invariant assertion
    inconsistency = measure_no_finding_inconsistency(adjusted)
    assert inconsistency["total_inconsistency_rate"] == 0.0, "Consistency policy invariant violated!"
    return adjusted


def compute_per_class_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    calibrated_thresholds: np.ndarray,
    class_names: List[str],
) -> List[Dict[str, Any]]:
    """
    P1.9: Tạo bảng chi tiết từng lớp:
      - Threshold-independent: support_pos, support_neg, roc_auc, ap
      - Fixed-0.5: f1_fixed, sensitivity_fixed, specificity_fixed
      - Calibrated T*: f1_calibrated, sensitivity_calibrated, specificity_calibrated
    Gán NaN khi mathematically undefined.
    """
    num_classes = y_true.shape[1]
    fixed_thresholds = np.full(num_classes, 0.5, dtype=np.float32)

    preds_fixed = apply_thresholds(y_prob, fixed_thresholds)
    preds_calibrated = apply_thresholds(y_prob, calibrated_thresholds)

    per_class_records: List[Dict[str, Any]] = []

    for c in range(num_classes):
        yt = y_true[:, c]
        yp = y_prob[:, c]
        pf = preds_fixed[:, c]
        pc = preds_calibrated[:, c]

        support_pos = int((yt == 1.0).sum())
        support_neg = int((yt == 0.0).sum())

        # Rank metrics (Threshold-Independent)
        roc_auc = safe_roc_auc(yt, yp)
        ap = safe_average_precision(yt, yp)

        def _calc_binary_metrics(y_real, y_pred, pos_count, neg_count):
            tp = int(((y_real == 1.0) & (y_pred == 1.0)).sum())
            fp = int(((y_real == 0.0) & (y_pred == 1.0)).sum())
            fn = int(((y_real == 1.0) & (y_pred == 0.0)).sum())
            tn = int(((y_real == 0.0) & (y_pred == 0.0)).sum())

            acc = float((tp + tn) / max(1, (tp + tn + fp + fn)))
            sens = float(tp / (tp + fn)) if pos_count > 0 else float("nan")
            spec = float(tn / (tn + fp)) if neg_count > 0 else float("nan")

            if pos_count == 0:
                f1 = float("nan")
            else:
                denom = 2 * tp + fp + fn
                f1 = float((2.0 * tp) / denom) if denom > 0 else 0.0

            return f1, sens, spec, acc

        f1_fixed, sens_fixed, spec_fixed, acc_fixed = _calc_binary_metrics(yt, pf, support_pos, support_neg)
        f1_cal, sens_cal, spec_cal, acc_cal = _calc_binary_metrics(yt, pc, support_pos, support_neg)

        per_class_records.append({
            "class_id": c,
            "class_name": class_names[c],
            "support_positive": support_pos,
            "support_negative": support_neg,
            "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
            "ap": round(ap, 4) if not np.isnan(ap) else None,
            "optimal_threshold": round(float(calibrated_thresholds[c]), 4),
            "f1_fixed": round(f1_fixed, 4) if not np.isnan(f1_fixed) else None,
            "f1_calibrated": round(f1_cal, 4) if not np.isnan(f1_cal) else None,
            "accuracy_fixed": round(acc_fixed, 4),
            "accuracy_calibrated": round(acc_cal, 4),
            "sensitivity_fixed": round(sens_fixed, 4) if not np.isnan(sens_fixed) else None,
            "sensitivity_calibrated": round(sens_cal, 4) if not np.isnan(sens_cal) else None,
            "specificity_fixed": round(spec_fixed, 4) if not np.isnan(spec_fixed) else None,
            "specificity_calibrated": round(spec_cal, 4) if not np.isnan(spec_cal) else None,
        })

    return per_class_records


def compute_macro_metrics(per_class_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tính các chỉ số tổng hợp (Macro Metrics) tách riêng cho:
      - PRIMARY: 14 Pathologies (indices 0..13)
      - SECONDARY: All 15 Classes (indices 0..14)
    Báo cáo kèm số lớp hợp lệ (valid classes count) cho từng metric.
    """
    pathology_records = per_class_records[:config.NO_FINDING_CLASS_ID]
    all_records = per_class_records

    def _nanmean_with_count(records, key):
        vals = [r[key] for r in records if r[key] is not None and not np.isnan(r[key])]
        mean_val = float(np.mean(vals)) if len(vals) > 0 else float("nan")
        return mean_val, len(vals)

    # 14 Pathologies
    macro_auc_14, valid_auc_14 = _nanmean_with_count(pathology_records, "roc_auc")
    macro_ap_14, valid_ap_14 = _nanmean_with_count(pathology_records, "ap")
    macro_f1_fixed_14, valid_f1_fixed_14 = _nanmean_with_count(pathology_records, "f1_fixed")
    macro_f1_cal_14, valid_f1_cal_14 = _nanmean_with_count(pathology_records, "f1_calibrated")
    macro_acc_fixed_14, valid_acc_fixed_14 = _nanmean_with_count(pathology_records, "accuracy_fixed")
    macro_acc_cal_14, valid_acc_cal_14 = _nanmean_with_count(pathology_records, "accuracy_calibrated")
    mean_sens_fixed_14, valid_sens_fixed_14 = _nanmean_with_count(pathology_records, "sensitivity_fixed")
    mean_sens_cal_14, valid_sens_cal_14 = _nanmean_with_count(pathology_records, "sensitivity_calibrated")
    mean_spec_fixed_14, valid_spec_fixed_14 = _nanmean_with_count(pathology_records, "specificity_fixed")
    mean_spec_cal_14, valid_spec_cal_14 = _nanmean_with_count(pathology_records, "specificity_calibrated")

    # All 15 Classes
    macro_auc_15, valid_auc_15 = _nanmean_with_count(all_records, "roc_auc")
    macro_ap_15, valid_ap_15 = _nanmean_with_count(all_records, "ap")
    macro_f1_fixed_15, valid_f1_fixed_15 = _nanmean_with_count(all_records, "f1_fixed")
    macro_f1_cal_15, valid_f1_cal_15 = _nanmean_with_count(all_records, "f1_calibrated")
    macro_acc_fixed_15, valid_acc_fixed_15 = _nanmean_with_count(all_records, "accuracy_fixed")
    macro_acc_cal_15, valid_acc_cal_15 = _nanmean_with_count(all_records, "accuracy_calibrated")

    return {
        # PRIMARY: 14 Pathologies
        "macro_auc_14": round(macro_auc_14, 4) if not np.isnan(macro_auc_14) else None,
        "valid_classes_auc_14": valid_auc_14,
        "macro_ap_14": round(macro_ap_14, 4) if not np.isnan(macro_ap_14) else None,
        "valid_classes_ap_14": valid_ap_14,
        "macro_f1_14_fixed": round(macro_f1_fixed_14, 4) if not np.isnan(macro_f1_fixed_14) else None,
        "valid_classes_f1_14_fixed": valid_f1_fixed_14,
        "macro_f1_14_calibrated": round(macro_f1_cal_14, 4) if not np.isnan(macro_f1_cal_14) else None,
        "valid_classes_f1_14_calibrated": valid_f1_cal_14,
        "macro_accuracy_14_fixed": round(macro_acc_fixed_14, 4) if not np.isnan(macro_acc_fixed_14) else None,
        "valid_classes_accuracy_14_fixed": valid_acc_fixed_14,
        "macro_accuracy_14_calibrated": round(macro_acc_cal_14, 4) if not np.isnan(macro_acc_cal_14) else None,
        "valid_classes_accuracy_14_calibrated": valid_acc_cal_14,
        "mean_sensitivity_14_fixed": round(mean_sens_fixed_14, 4) if not np.isnan(mean_sens_fixed_14) else None,
        "valid_classes_sensitivity_14_fixed": valid_sens_fixed_14,
        "mean_sensitivity_14_calibrated": round(mean_sens_cal_14, 4) if not np.isnan(mean_sens_cal_14) else None,
        "valid_classes_sensitivity_14_calibrated": valid_sens_cal_14,
        "mean_specificity_14_fixed": round(mean_spec_fixed_14, 4) if not np.isnan(mean_spec_fixed_14) else None,
        "valid_classes_specificity_14_fixed": valid_spec_fixed_14,
        "mean_specificity_14_calibrated": round(mean_spec_cal_14, 4) if not np.isnan(mean_spec_cal_14) else None,
        "valid_classes_specificity_14_calibrated": valid_spec_cal_14,

        # SECONDARY: 15 Classes
        "macro_auc_15": round(macro_auc_15, 4) if not np.isnan(macro_auc_15) else None,
        "valid_classes_auc_15": valid_auc_15,
        "macro_ap_15": round(macro_ap_15, 4) if not np.isnan(macro_ap_15) else None,
        "valid_classes_ap_15": valid_ap_15,
        "macro_f1_15_fixed": round(macro_f1_fixed_15, 4) if not np.isnan(macro_f1_fixed_15) else None,
        "valid_classes_f1_15_fixed": valid_f1_fixed_15,
        "macro_f1_15_calibrated": round(macro_f1_cal_15, 4) if not np.isnan(macro_f1_cal_15) else None,
        "valid_classes_f1_15_calibrated": valid_f1_cal_15,
        "macro_accuracy_15_fixed": round(macro_acc_fixed_15, 4) if not np.isnan(macro_acc_fixed_15) else None,
        "valid_classes_accuracy_15_fixed": valid_acc_fixed_15,
        "macro_accuracy_15_calibrated": round(macro_acc_cal_15, 4) if not np.isnan(macro_acc_cal_15) else None,
        "valid_classes_accuracy_15_calibrated": valid_acc_cal_15,
    }
