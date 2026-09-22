"""
compare_models.py - Tong hop so sanh toan dien 3 Kien truc Detector (VinBigData Detection P0/P1)
Ket hop do dac Complexity (RAM state_dict_size_mib, Parameters), Benchmark (BS=1 Latency, BS=16 FPS),
va Test Metrics da seed (mAP@0.5, Macro F1, Loss) theo chuan Mau N-1 (ddof=1).
Xuat ket qua ra: Markdown, CSV, va JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from scripts.benchmark_utils import benchmark_detection_inference, measure_model_complexity
from scripts.config import IMAGE_SIZE, NUM_CLASSES
from scripts.experiment_config import CANONICAL_MODELS, CANONICAL_SEEDS
from scripts.models.factory import build_model


def run_model_benchmarks(
    models: Optional[List[str]] = None,
    image_size: int = IMAGE_SIZE,
    device_str: Optional[str] = None,
) -> Dict[str, Dict]:
    models = models or CANONICAL_MODELS
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    benchmarks = {}

    print("=" * 80)
    print(f"BENCHMARK COMPLEXITY & INFERENCE ({device})")
    print("=" * 80)

    for m in models:
        print(f"--> Benchmark {m.upper()}...")
        model = build_model(m, num_classes=NUM_CLASSES, pretrained=False).to(device)
        complexity = measure_model_complexity(model, img_size=image_size)
        speed = benchmark_detection_inference(model, device=device, img_size=image_size, num_warmup=5, num_runs=20)
        benchmarks[m] = {**complexity, **speed}
        bs = speed.get("throughput_batch_size", 16)
        print(f"    Params: {complexity['total_params']:,} | RAM: {complexity['state_dict_size_mib']:.2f} MiB")
        print(f"    BS=1 Latency: {speed['bs1_latency_median_ms']:.2f} ms | BS={bs} FPS: {speed['throughput_fps']:.1f} ({speed.get('benchmark_device', 'unknown')})")

    return benchmarks


def build_comparison_table(
    summary_path: str = "outputs/canonical_multi_seed_test_summary.json",
    output_dir: str = "outputs",
    models: Optional[List[str]] = None,
    benchmarks: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    models = models or CANONICAL_MODELS
    os.makedirs(output_dir, exist_ok=True)

    test_data = {}
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            test_data = data.get("summary", {})

    benchmarks = benchmarks or run_model_benchmarks(models)

    table_rows = []
    for m in models:
        bs = benchmarks[m].get("throughput_batch_size", 16)
        row = {
            "Model Architecture": m,
            "Total Params": f"{benchmarks[m]['total_params']:,}",
            "RAM (state_dict MiB)": f"{benchmarks[m]['state_dict_size_mib']:.2f}",
            "Latency BS=1 (ms)": f"{benchmarks[m]['bs1_latency_median_ms']:.2f}",
            f"Throughput BS={bs} (FPS)": f"{benchmarks[m]['throughput_fps']:.1f}",
        }

        if m in test_data:
            m_sum = test_data[m]
            map_mean = m_sum.get("map50", {}).get("mean", 0.0)
            map_std = m_sum.get("map50", {}).get("sample_std_ddof1", 0.0)
            f1_mean = m_sum.get("macro_f1", {}).get("mean", 0.0)
            f1_std = m_sum.get("macro_f1", {}).get("sample_std_ddof1", 0.0)
            loss_mean = m_sum.get("loss", {}).get("mean", 0.0)

            row["mAP@0.5 (Mean ± Std)"] = f"{map_mean:.4f} ± {map_std:.4f}"
            row["Macro F1 (Mean ± Std)"] = f"{f1_mean:.4f} ± {f1_std:.4f}"
            row["Test Loss (Mean)"] = f"{loss_mean:.4f}"
        else:
            row["mAP@0.5 (Mean ± Std)"] = "N/A (Pending Final-Test)"
            row["Macro F1 (Mean ± Std)"] = "N/A (Pending Final-Test)"
            row["Test Loss (Mean)"] = "N/A"

        table_rows.append(row)

    # 1. In ra Markdown
    headers = list(table_rows[0].keys())
    md_lines = [
        "# VinBigData Detection P0/P1 Model Comparison Table",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---:"] * len(headers)) + " |",
    ]
    for r in table_rows:
        md_lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")

    md_content = "\n".join(md_lines) + "\n"
    md_file = os.path.join(output_dir, "detection_comparison_table.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 2. In ra CSV
    csv_file = os.path.join(output_dir, "detection_comparison_table.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(table_rows)

    # 3. In ra JSON
    json_file = os.path.join(output_dir, "detection_comparison_table.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2)

    print("\n" + md_content)
    print(f"[compare_models] Da luu: {md_file}, {csv_file}, {json_file}")
    return table_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sanh tong the 3 detector")
    parser.add_argument("--summary", default="outputs/canonical_multi_seed_test_summary.json")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    build_comparison_table(summary_path=args.summary, output_dir=args.output_dir)
