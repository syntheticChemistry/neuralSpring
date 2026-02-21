#!/usr/bin/env python3
"""Benchmark: HMM forward algorithm (scaled, T=5000, N=3, M=4)."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200


def hmm_forward(pi, A, B, obs):
    """Scaled forward algorithm — same math as hmm_phylo.py."""
    T = len(obs)
    N = pi.shape[0]
    alpha = np.zeros((T, N), dtype=np.float64)
    scales = np.zeros(T, dtype=np.float64)

    alpha[0] = pi * B[:, obs[0]]
    scales[0] = alpha[0].sum()
    alpha[0] /= scales[0]

    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
        scales[t] = alpha[t].sum()
        if scales[t] > 0:
            alpha[t] /= scales[t]

    log_likelihood = np.sum(np.log(scales + 1e-300))
    return alpha, log_likelihood


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
    N, M, T = 3, 4, 5000

    A = rng.dirichlet(np.ones(N) * 10, size=N).astype(np.float64)
    B = rng.dirichlet(np.ones(M) * 2, size=N).astype(np.float64)
    pi = rng.dirichlet(np.ones(N) * 5).astype(np.float64)
    obs = rng.integers(0, M, size=T)

    def run():
        hmm_forward(pi, A, B, obs)

    median_us = bench_fn(run)

    print(f"HMM_FORWARD_3x5000_US={median_us:.1f}")
    print()
    print(f"Python/NumPy HMM forward benchmark — NumPy {np.__version__}")
    print(f"  Config: N={N} states, M={M} symbols, T={T} observations")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
