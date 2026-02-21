#!/usr/bin/env python3
"""Benchmark: Pairwise Hamming distance matrix (20 sequences × 500 sites)."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200


def hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Proportion of differing sites — same math as sate_alignment.py."""
    if len(a) != len(b):
        return 1.0
    diff = np.sum(a != b)
    return diff / len(a)


def pairwise_hamming(seqs: list[np.ndarray]) -> np.ndarray:
    """Compute N×N pairwise Hamming distance matrix."""
    n = len(seqs)
    L = len(seqs[0])
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(seqs[i], seqs[j])
            D[i, j] = d
            D[j, i] = d
    return D


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
    n_seqs, seq_len = 20, 500

    seqs = [rng.integers(0, 4, size=seq_len) for _ in range(n_seqs)]

    def run():
        pairwise_hamming(seqs)

    median_us = bench_fn(run)

    print(f"HAMMING_20x500_US={median_us:.1f}")
    print()
    print(f"Python/NumPy Hamming distance benchmark — NumPy {np.__version__}")
    print(f"  Config: {n_seqs} sequences × {seq_len} sites (DNA: 0-3)")
    print(f"  {n_seqs * (n_seqs - 1) // 2} pairwise distances")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
