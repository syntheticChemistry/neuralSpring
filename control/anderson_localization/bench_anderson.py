# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark: Anderson Hamiltonian eigensolve + mean IPR (N=64, W=4.0)."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200

PHI = (1 + np.sqrt(5)) / 2
T_HOPPING = 1.0


def anderson_hamiltonian_random(n, t, w, seed=42):
    rng = np.random.default_rng(seed)
    h = np.zeros((n, n))
    v = rng.uniform(-w / 2, w / 2, n)
    for i in range(n):
        h[i, i] = v[i]
    for i in range(n - 1):
        h[i, i + 1] = h[i + 1, i] = -t
    return h


def ipr(psi):
    p = np.abs(psi) ** 2
    return float(np.sum(p * p))


def mean_ipr(eigenvectors):
    n = eigenvectors.shape[0]
    return float(np.mean([ipr(eigenvectors[:, k]) for k in range(n)]))


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
    N = 64
    W = 4.0

    h = anderson_hamiltonian_random(N, T_HOPPING, W, seed=42)

    def run():
        eigenvalues, eigenvectors = np.linalg.eigh(h)
        _ = mean_ipr(eigenvectors)

    median_us = bench_fn(run)

    print(f"ANDERSON_IPR_64_US={median_us:.1f}")
    print()
    print(f"Python/NumPy Anderson localization benchmark — NumPy {np.__version__}")
    print(f"  Config: N={N}, W={W}, t={T_HOPPING}")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
