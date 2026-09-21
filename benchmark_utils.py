"""
benchmark_utils.py - Đo đạc hiệu năng tính toán chuẩn hóa (Standardized Model-Forward Benchmark).
Đo thuần model inference trên in-memory tensor, tách bạch Latency (batch=1) và Standardized Throughput (batch=16).
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import config


def get_model_complexity(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Tính toán các chỉ số độ phức tạp kiến trúc (Hằng số cố định, không aggregate std qua seed):
      - Total Params (M): Tổng số tham số (triệu)
      - Trainable Params (M): Số tham số có requires_grad
      - Parameter Only Bytes: Dung lượng tensor chỉ tính parameters
      - State Dict Size Bytes: Dung lượng toàn bộ tensor trong state_dict (bao gồm buffers như BatchNorm running_mean, running_var)
      - State Dict Size MiB: Dung lượng (MiB) của state_dict
      - Weights Size MB: Alias cho State Dict Size MiB (backwards compatibility)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_only_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    state_dict_bytes = sum(
        t.numel() * t.element_size()
        for t in model.state_dict().values()
        if torch.is_tensor(t)
    )
    state_dict_size_mib = state_dict_bytes / (1024.0 * 1024.0)

    return {
        "total_params_m": round(total_params / 1e6, 4),
        "total_params_raw": total_params,
        "trainable_params_m": round(trainable_params / 1e6, 4),
        "parameter_only_bytes": parameter_only_bytes,
        "state_dict_size_bytes": state_dict_bytes,
        "state_dict_size_mib": round(state_dict_size_mib, 2),
        "weights_size_mb": round(state_dict_size_mib, 2),
    }


def benchmark_model_forward(
    model: torch.nn.Module,
    device: torch.device,
    image_size: Tuple[int, int] = config.IMAGE_SIZE,
    warmup: int = 10,
    runs: int = 100,
    throughput_batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Chạy standardized benchmark suy luận:
      1. Latency (batch=1): Đo thời gian xử lý 1 ảnh (ms/img) -> mean ± runtime_std (ddof=1)
      2. Standardized Throughput at BS=16: Đo số ảnh/giây (FPS)
    Sử dụng torch.inference_mode() và đồng bộ CUDA (nếu có).
    """
    model.eval()
    is_cuda = (device.type == "cuda" and torch.cuda.is_available())

    # --- 1. Latency Benchmark (Batch Size = 1) ---
    dummy_single = torch.randn(1, 3, image_size[0], image_size[1], device=device)

    with torch.inference_mode():
        # Warmup
        for _ in range(warmup):
            _ = model(dummy_single)
        if is_cuda:
            torch.cuda.synchronize()

        latencies_ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy_single)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latency_mean = float(np.mean(latencies_ms))
    latency_std = float(np.std(latencies_ms, ddof=1))

    # --- 2. Standardized Throughput at BS=16 ---
    dummy_batch = torch.randn(throughput_batch_size, 3, image_size[0], image_size[1], device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy_batch)
        if is_cuda:
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy_batch)
            if is_cuda:
                torch.cuda.synchronize()
        total_time_sec = time.perf_counter() - t_start

    total_images_processed = runs * throughput_batch_size
    throughput_fps = float(total_images_processed / max(1e-6, total_time_sec))

    hardware_info = {
        "device": str(device),
        "cuda_available": is_cuda,
        "device_name": torch.cuda.get_device_name(0) if is_cuda else "CPU",
        "pytorch_version": torch.__version__,
    }

    complexity = get_model_complexity(model)

    return {
        "complexity": complexity,
        "latency": {
            "batch_size": 1,
            "mean_ms": round(latency_mean, 2),
            "std_ms": round(latency_std, 2),
            "warmup": warmup,
            "runs": runs,
        },
        "throughput": {
            "batch_size": throughput_batch_size,
            "fps": round(throughput_fps, 2),
            "total_images": total_images_processed,
            "total_time_sec": round(total_time_sec, 3),
        },
        "environment": hardware_info,
    }


def save_benchmark(
    benchmark_data: Dict[str, Any],
    model_name: str,
    output_dir: Path = config.OUTPUT_DIR / "benchmarks",
) -> Path:
    """Lưu kết quả benchmark vào outputs/benchmarks/{model_name}.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / f"{model_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print(f"[benchmark] 💾 Đã lưu benchmark tại: {out_file}")
    return out_file
