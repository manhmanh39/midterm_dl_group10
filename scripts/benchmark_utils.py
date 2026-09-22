"""
scripts/benchmark_utils.py - Do dac do phuc tap mo hinh (parameters, RAM state_dict_size_mib)
va benchmark hieu nang inference (BS=1 latency median/mean/p95, BS=16 throughput FPS).
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from scripts.config import IMAGE_SIZE


def measure_model_complexity(model: nn.Module, img_size: int = IMAGE_SIZE) -> Dict[str, float]:
    """
    Do dac so luong tham so va dung luong bo nho RAM that su cua model (state_dict_size_mib).
    Khong do dung luong file .pth tren dia vi file .pth chua ca overhead zip/pickle/metadata.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params

    # Tinh exact RAM footprint cua parameters va buffers
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes
    state_dict_size_mib = total_bytes / (1024.0 * 1024.0)

    # Uoc tinh FLOPs neu co thu vien ho tro
    flops_g = 0.0
    try:
        from thop import profile
        dummy = torch.randn(1, 3, img_size, img_size)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        flops_g = flops / 1e9
    except Exception:
        flops_g = 0.0

    return {
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params": total_params,
        "state_dict_size_bytes": total_bytes,
        "state_dict_size_mib": round(state_dict_size_mib, 3),
        "flops_giga": round(flops_g, 2) if flops_g > 0 else None,
    }


@torch.no_grad()
def benchmark_detection_inference(
    model: nn.Module,
    device: torch.device,
    img_size: int = IMAGE_SIZE,
    num_warmup: int = 10,
    num_runs: int = 50,
) -> Dict[str, float]:
    """
    Do BS=1 Latency (ms) va BS=16 Throughput (FPS) chuan xac voi torch.cuda.synchronize().
    Tu dong toi uu so vong chay tren CPU de tranh qua tai CPU khi benchmark.
    """
    model.eval()
    model.to(device)

    is_cpu = (device.type == "cpu")
    warmup_bs1 = 2 if is_cpu else num_warmup
    actual_runs_bs1 = 5 if is_cpu else num_runs
    warmup_bs16 = 1 if is_cpu else max(2, num_warmup // 2)
    runs_bs16 = 2 if is_cpu else max(10, num_runs // 2)

    # 1. Benchmark BS=1 Latency
    dummy_bs1 = torch.randn(1, 3, img_size, img_size, device=device)

    # Warmup
    for _ in range(warmup_bs1):
        _ = model(dummy_bs1)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(actual_runs_bs1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(dummy_bs1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latency_arr = np.array(latencies_ms)
    lat_mean = float(np.mean(latency_arr))
    lat_median = float(np.median(latency_arr))
    lat_p95 = float(np.percentile(latency_arr, 95))

    # 2. Benchmark BS=16 Throughput
    # Neu tren CPU, dung bs=4 de do throughput ma khong gay Memory/Compute spike
    test_bs = 4 if is_cpu else 16
    dummy_test_bs = torch.randn(test_bs, 3, img_size, img_size, device=device)

    for _ in range(warmup_bs16):
        _ = model(dummy_test_bs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    for _ in range(runs_bs16):
        _ = model(dummy_test_bs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start
    throughput_fps = float((test_bs * runs_bs16) / max(total_time, 1e-6))

    return {
        "benchmark_device": device.type,
        "throughput_batch_size": test_bs,
        "throughput_fps": round(throughput_fps, 1),
        f"bs{test_bs}_throughput_fps": round(throughput_fps, 1),
        "bs1_latency_median_ms": round(lat_median, 2),
        "bs1_latency_mean_ms": round(lat_mean, 2),
        "bs1_latency_p95_ms": round(lat_p95, 2),
    }
