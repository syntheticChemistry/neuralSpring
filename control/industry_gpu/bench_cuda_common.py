# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared CUDA benchmark utilities for industry GPU parity tests.

All industry GPU benchmarks use PyTorch as a thin frontend to cuBLAS,
cuDNN, and cuFFT.  This module provides deterministic seeding, CUDA
synchronization, warmup, and median timing in one place.
"""

import time

import torch

WARMUP = 50
ITERATIONS = 200
DEVICE = "cuda"


def require_cuda():
    """Exit cleanly if CUDA is not available."""
    if not torch.cuda.is_available():
        print("SKIP — no CUDA device")
        raise SystemExit(0)
    torch.cuda.synchronize()


def seed_all(seed: int = 42):
    """Pin all RNG sources for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bench_cuda(func, warmup=WARMUP, iters=ITERATIONS):
    """Time a CUDA kernel with proper synchronization.

    Returns median wall-clock time in microseconds.
    """
    torch.cuda.synchronize()
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        func()
        torch.cuda.synchronize()
        timings.append(time.perf_counter_ns() - t0)

    timings.sort()
    return timings[len(timings) // 2] / 1000.0


def emit(tag: str, median_us: float):
    """Print a machine-readable timing line."""
    print(f"{tag}_US={median_us:.1f}")
