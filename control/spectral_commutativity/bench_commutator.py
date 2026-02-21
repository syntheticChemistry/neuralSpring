#!/usr/bin/env python3
"""Benchmark: Commutator Frobenius norm (64×64 matrices)."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200


def commutator_frobenius_norm(A, B):
    """||[A,B]||_F = ||AB - BA||_F — same math as spectral_commutativity.py."""
    return np.linalg.norm(A @ B - B @ A, "fro")


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
    rng = np.random.default_rng(42)
    n = 64
    A = rng.standard_normal((n, n)).astype(np.float64)
    B = rng.standard_normal((n, n)).astype(np.float64)

    def run():
        commutator_frobenius_norm(A, B)

    median_us = bench_fn(run)

    print(f"COMMUTATOR_64x64_US={median_us:.1f}")
    print()
    print(f"Python/NumPy commutator benchmark — NumPy {np.__version__}")
    print(f"  Config: {n}×{n} matrices")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
