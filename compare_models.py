"""
compare_models.py - Tổng hợp kết quả đa seed (Multi-seed aggregation) và sinh bảng so sánh toàn diện theo Giao thức P1.
Tách bạch:
  - Chất lượng mô hình: Mean ± Seed Sample Std (ddof=1)
  - Runtime Latency: Mean ± Runtime Std (ddof=1)
  - Architectural Complexity: Hằng số cố định (Params, Size)
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import config
from experiment_config import FINAL_TEST_DIR, BENCHMARK_DIR


def compute_sample_stats(values: List[Optional[float]]) -> Dict[str, Any]:
    """Tính mean và sample std (ddof=1), bỏ qua None."""
    valid_vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not valid_vals:
        return {"mean": None, "std": None, "str": "N/A"}
    mean_val = float(np.mean(valid_vals))
    std_val = float(np.std(valid_vals, ddof=1)) if len(valid_vals) > 1 else 0.0
    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "str": f"{mean_val:.4f} ± {std_val:.4f}" if len(valid_vals) > 1 else f"{mean_val:.4f}",
    }


# [P1 - Item 12: Tạo comparison table]
def generate_comparison_table(
    models: List[str] = ["simple", "complex", "transfer"],
    seeds: List[int] = [202601, 202602, 202603],
    output_dir: Path = config.OUTPUT_DIR,
    data_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    [P1 - Item 12: Tạo comparison table]
    Thu thập artifacts từ outputs/final_test và outputs/benchmarks trên toàn bộ các seeds,
    tính toán Mean ± Sample Standard Deviation (ddof=1) và sinh bảng so sánh toàn diện
    ra Markdown (protocol_comparison_table.md), CSV và JSON.
    """
    from experiment_config import get_final_test_dir, get_benchmark_dir
    mode = data_mode if data_mode is not None else config.DEFAULT_DATA_MODE
    test_dir = get_final_test_dir(data_mode=mode)
    bench_dir = get_benchmark_dir(data_mode=mode)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_display_names = {
        "simple": "Simple CNN (Baseline)",
        "complex": "Complex CNN (Multi-branch)",
        "transfer": "Transfer Learning (ResNet-50)",
    }

    aggregated_data = {}

    for model in models:
        test_runs = []
        for s in seeds:
            metric_file = test_dir / f"{model}_seed{s}_metrics.json"
            if mode == "real":
                # Trên dữ liệu thật: KHÔNG legacy fallback! Fail closed nếu thiếu metric file
                if not metric_file.exists():
                    raise FileNotFoundError(
                        f"[FAIL CLOSED] Không tìm thấy file metrics cho {model} seed {s} tại: {metric_file}\n"
                        "Giao thức P1 cấm legacy fallback khi đánh giá trên dữ liệu thật."
                    )
                with open(metric_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    summary = data.get("summary_metrics", data)
                    test_runs.append(summary)
            else:
                # Trên môi trường demo: kiểm tra test_dir trước, sau đó fallback về FINAL_TEST_DIR legacy
                if not metric_file.exists() and (FINAL_TEST_DIR / f"{model}_seed{s}_metrics.json").exists():
                    metric_file = FINAL_TEST_DIR / f"{model}_seed{s}_metrics.json"
                if metric_file.exists():
                    with open(metric_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        summary = data.get("summary_metrics", data)
                        test_runs.append(summary)

        # Benchmark
        bench_file = bench_dir / f"{model}.json"
        if not bench_file.exists() and (BENCHMARK_DIR / f"{model}.json").exists():
            bench_file = BENCHMARK_DIR / f"{model}.json"
        bench_data = {}
        if bench_file.exists():
            with open(bench_file, "r", encoding="utf-8") as f:
                bench_data = json.load(f)

        if not test_runs:
            print(f"[compare] ⚠️ Không tìm thấy kết quả final-test cho mô hình '{model}'!")
            continue

        n_seeds = len(test_runs)

        # [P1 - Item 8: Thêm PR-AUC] & [P1 - Item 9: Report macro metric của 14 pathologies riêng]
        auc_14 = compute_sample_stats([r.get("macro_auc_14") for r in test_runs])
        ap_14 = compute_sample_stats([r.get("macro_ap_14") for r in test_runs])
        f1_fixed_14 = compute_sample_stats([r.get("macro_f1_14_fixed") for r in test_runs])
        f1_cal_14 = compute_sample_stats([r.get("macro_f1_14_calibrated") for r in test_runs])
        sens_cal_14 = compute_sample_stats([r.get("mean_sensitivity_14_calibrated") for r in test_runs])
        spec_cal_14 = compute_sample_stats([r.get("mean_specificity_14_calibrated") for r in test_runs])

        # [P0 - Item 5: Sửa exact_match_accuracy] Phân định rõ Exact Match vs Per-Label
        per_label_fixed = compute_sample_stats([r.get("per_label_accuracy_fixed") for r in test_runs])
        per_label_cal = compute_sample_stats([r.get("per_label_accuracy_calibrated") for r in test_runs])
        exact_match_fixed = compute_sample_stats([r.get("exact_match_accuracy_fixed") for r in test_runs])
        exact_match_cal = compute_sample_stats([r.get("exact_match_accuracy_calibrated") for r in test_runs])

        # Inconsistency
        incons_raw_fixed = compute_sample_stats([r.get("total_inconsistency_rate_raw_fixed") for r in test_runs])
        incons_raw_cal = compute_sample_stats([r.get("total_inconsistency_rate_raw_calibrated") for r in test_runs])

        # Benchmark details
        complexity = bench_data.get("complexity", {})
        latency_info = bench_data.get("latency", {})
        throughput_info = bench_data.get("throughput", {})

        total_params = complexity.get("total_params_m", "N/A")
        weights_mb = complexity.get("weights_size_mb", "N/A")
        latency_mean = latency_info.get("mean_ms", "N/A")
        latency_std = latency_info.get("std_ms", 0.0)
        latency_str = f"{latency_mean} ± {latency_std}" if latency_mean != "N/A" else "N/A"
        throughput_fps = throughput_info.get("fps", "N/A")

        aggregated_data[model] = {
            "display_name": model_display_names.get(model, model),
            "num_seeds": n_seeds,
            "seeds": seeds[:n_seeds],
            "macro_auc_14": auc_14,
            "macro_ap_14": ap_14,
            "macro_f1_14_fixed": f1_fixed_14,
            "macro_f1_14_calibrated": f1_cal_14,
            "per_label_accuracy_fixed": per_label_fixed,
            "per_label_accuracy_calibrated": per_label_cal,
            "exact_match_accuracy_fixed": exact_match_fixed,
            "exact_match_accuracy_calibrated": exact_match_cal,
            "mean_sensitivity_14_calibrated": sens_cal_14,
            "mean_specificity_14_calibrated": spec_cal_14,
            "inconsistency_raw_fixed": incons_raw_fixed,
            "inconsistency_raw_calibrated": incons_raw_cal,
            "inconsistency_adjusted": "0.00% (Guaranteed)",
            "total_params_m": total_params,
            "weights_size_mb": weights_mb,
            "latency_ms_bs1": latency_str,
            "throughput_fps_bs16": throughput_fps,
        }

    # Sinh Markdown Table
    md_lines = [
        "# BẢNG SO SÁNH TỔNG HỢP MÔ HÌNH THEO GIAO THỨC P1 (LOCKED PROTOCOL)",
        "",
        "> **Quy chuẩn thực nghiệm:**",
        "> - Phân tách tập dữ liệu: Train / Val / Test độc lập (Zero Test Leakage).",
        "> - Primary Metrics: Macro-14 Pathologies (loại trừ No Finding khỏi macro primary).",
        "> - Thống kê chất lượng: Mean ± Sample Std (ddof=1) qua các seeds độc lập.",
        "> - Inference Benchmark: Đo lường chuẩn hóa batch size 1 (Latency) và batch size 16 (Standardized Throughput).",
        "> - Phân định rõ: Per-Label Accuracy (từng nhãn độc lập) vs Exact Match Accuracy (toàn bộ 15 nhãn đồng thời).",
        "> - Không gọi nhầm Average Precision (AP) là PR-AUC.",
        "",
        "## 1. Hiệu Năng Phân Loại Bệnh Học (14 Pathologies Primary Metrics)",
        "",
        "| Mô hình | Số Seeds | Macro-14 ROC-AUC | Macro-14 AP | Macro-14 F1 (Fixed 0.5) | Macro-14 F1 (Calibrated T*) | Per-Label Acc (Cal T*) | Exact Match Acc (Cal T*) | Sens (Cal T*) | Spec (Cal T*) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model, d in aggregated_data.items():
        md_lines.append(
            f"| **{d['display_name']}** | {d['num_seeds']} | "
            f"{d['macro_auc_14']['str']} | {d['macro_ap_14']['str']} | "
            f"{d['macro_f1_14_fixed']['str']} | {d['macro_f1_14_calibrated']['str']} | "
            f"{d['per_label_accuracy_calibrated']['str']} | {d['exact_match_accuracy_calibrated']['str']} | "
            f"{d['mean_sensitivity_14_calibrated']['str']} | {d['mean_specificity_14_calibrated']['str']} |"
        )

    md_lines.extend([
        "",
        "## 2. Tính Nhất Quán Nhãn No-Finding (Consistency Policy)",
        "",
        "| Mô hình | Raw Inconsistency (Fixed 0.5) | Raw Inconsistency (Calibrated T*) | Adjusted Inconsistency |",
        "| :--- | :---: | :---: | :---: |",
    ])

    for model, d in aggregated_data.items():
        raw_fix_str = f"{d['inconsistency_raw_fixed']['mean']*100:.2f}%" if d['inconsistency_raw_fixed']['mean'] is not None else "N/A"
        raw_cal_str = f"{d['inconsistency_raw_calibrated']['mean']*100:.2f}%" if d['inconsistency_raw_calibrated']['mean'] is not None else "N/A"
        md_lines.append(
            f"| **{d['display_name']}** | {raw_fix_str} | {raw_cal_str} | {d['inconsistency_adjusted']} |"
        )

    md_lines.extend([
        "",
        "## 3. Độ Phức Tạp & Hiệu Năng Tính Toán Chuẩn Hóa (Standardized Benchmark)",
        "",
        "| Mô hình | Params (M) | Weights (MB) | Latency BS=1 (ms) [mean ± std] | Standardized Throughput BS=16 (FPS) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for model, d in aggregated_data.items():
        md_lines.append(
            f"| **{d['display_name']}** | {d['total_params_m']} | {d['weights_size_mb']} | "
            f"{d['latency_ms_bs1']} | {d['throughput_fps_bs16']} |"
        )

    md_content = "\n".join(md_lines) + "\n"

    # Lưu Markdown
    md_path = output_dir / "protocol_comparison_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Lưu JSON
    json_path = output_dir / "protocol_comparison_table.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_data, f, indent=2, ensure_ascii=False)

    # Lưu CSV
    csv_path = output_dir / "protocol_comparison_table.csv"
    csv_rows = []
    for model, d in aggregated_data.items():
        csv_rows.append({
            "model": model,
            "display_name": d["display_name"],
            "num_seeds": d["num_seeds"],
            "macro_auc_14_mean": d["macro_auc_14"]["mean"],
            "macro_auc_14_std": d["macro_auc_14"]["std"],
            "macro_ap_14_mean": d["macro_ap_14"]["mean"],
            "macro_ap_14_std": d["macro_ap_14"]["std"],
            "macro_f1_14_fixed_mean": d["macro_f1_14_fixed"]["mean"],
            "macro_f1_14_calibrated_mean": d["macro_f1_14_calibrated"]["mean"],
            "per_label_acc_calibrated_mean": d["per_label_accuracy_calibrated"]["mean"],
            "exact_match_acc_calibrated_mean": d["exact_match_accuracy_calibrated"]["mean"],
            "sens_14_calibrated_mean": d["mean_sensitivity_14_calibrated"]["mean"],
            "spec_14_calibrated_mean": d["mean_specificity_14_calibrated"]["mean"],
            "inconsistency_raw_fixed_mean": d["inconsistency_raw_fixed"]["mean"],
            "inconsistency_raw_cal_mean": d["inconsistency_raw_calibrated"]["mean"],
            "total_params_m": d["total_params_m"],
            "weights_size_mb": d["weights_size_mb"],
            "latency_mean_ms": d["latency_ms_bs1"],
            "throughput_fps_bs16": d["throughput_fps_bs16"],
        })

    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)

    print(f"\n[compare] 📊 Đã xuất bảng so sánh tổng hợp tại:")
    print(f"  - Markdown: {md_path}")
    print(f"  - CSV     : {csv_path}")
    print(f"  - JSON    : {json_path}")

    return aggregated_data


if __name__ == "__main__":
    generate_comparison_table()
