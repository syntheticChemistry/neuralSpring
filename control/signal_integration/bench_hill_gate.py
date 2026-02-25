# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark: Two-input Hill function grid (50×50)."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200


def two_input_hill(
    cdg: float,
    ai: float,
    vmax: float = 1.0,
    k1: float = 0.5,
    k2: float = 0.3,
    n1: float = 2.0,
    n2: float = 2.0,
) -> float:
    h1 = (cdg**n1) / (k1**n1 + cdg**n1 + 1e-30)
    h2 = (ai**n2) / (k2**n2 + ai**n2 + 1e-30)
    return vmax * h1 * h2


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
    nx, ny = 50, 50
    cdg_vals = [i * 0.1 for i in range(nx)]
    ai_vals = [i * 0.1 for i in range(ny)]

    def run():
        out = []
        for cdg in cdg_vals:
            for ai in ai_vals:
                out.append(two_input_hill(cdg, ai))
        return out

    median_us = bench_fn(run)

    print(f"HILL_GATE_50x50_US={median_us:.1f}")
    print()
    print(f"Python two-input Hill gate benchmark — NumPy {np.__version__}")
    print(f"  Config: {nx}×{ny} grid = {nx * ny} evaluations")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
