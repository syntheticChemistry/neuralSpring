# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark: Pairwise Jaccard distance matrix (30 genomes × 500 genes)."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200


def jaccard_distance_matrix(pa):
    """Pairwise Jaccard distance between genomes (columns) — same math as pangenome_selection.py."""
    n = pa.shape[1]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            intersection = np.sum(pa[:, i] * pa[:, j])
            union = np.sum(np.maximum(pa[:, i], pa[:, j]))
            d = 1.0 - intersection / union if union > 0 else 0.0
            dist[i, j] = d
            dist[j, i] = d
    return dist


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
    n_genomes, n_genes = 30, 500
    pa = (rng.random((n_genes, n_genomes)) < 0.5).astype(np.float64)

    def run():
        jaccard_distance_matrix(pa)

    median_us = bench_fn(run)

    print(f"JACCARD_30x500_US={median_us:.1f}")
    print()
    print(f"Python/NumPy Jaccard distance benchmark — NumPy {np.__version__}")
    print(f"  Config: {n_genomes} genomes × {n_genes} genes")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
