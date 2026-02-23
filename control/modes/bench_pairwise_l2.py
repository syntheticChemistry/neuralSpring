# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark: Pairwise L2 distance (10 vectors × 8 dimensions)."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def bench_fn(func, warmup=WARMUP, iters=ITERATIONS):
    for _ in range(warmup):
        func()
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        func()
        timings.append(time.perf_counter_ns() - t0)
    timings.sort()
    return timings[len(timings) // 2] / 1000.0


if __name__ == "__main__":
    n, dim = 10, 8
    features = np.arange(n * dim, dtype=np.float64).reshape(n, dim) * 0.1

    def run():
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(l2_distance(features[i], features[j]))
        return dists

    median_us = bench_fn(run)

    print(f"PAIRWISE_L2_10x8_US={median_us:.1f}")
    print()
    print(f"Python/NumPy pairwise L2 benchmark — NumPy {np.__version__}")
    print(f"  Config: {n} vectors × {dim} dimensions ({n*(n-1)//2} pairs)")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
