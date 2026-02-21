#!/usr/bin/env python3
"""Benchmark: NK fitness landscape evaluation (N=10, K=2, 1000 genotypes)."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200


class NKLandscape:
    """NK fitness landscape — same math as counterdiabatic_evolution.py."""

    def __init__(self, n: int, k: int, seed: int = 42):
        self.n = n
        self.k = k
        rng = np.random.default_rng(seed)
        self.neighbors = np.zeros((n, k), dtype=int)
        for i in range(n):
            candidates = [j for j in range(n) if j != i]
            self.neighbors[i] = rng.choice(candidates, size=k, replace=False)
        self.tables = {}
        for i in range(n):
            n_entries = 2 ** (k + 1)
            self.tables[i] = rng.uniform(0, 1, n_entries)

    def fitness(self, genotype: np.ndarray) -> float:
        """Compute fitness of a binary genotype vector."""
        total = 0.0
        for i in range(self.n):
            bits = [genotype[i]] + [genotype[j] for j in self.neighbors[i]]
            idx = sum(b * (2**p) for p, b in enumerate(bits))
            total += self.tables[i][idx]
        return total / self.n


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
    N, K = 10, 2
    n_genotypes = 1000

    landscape = NKLandscape(N, K, seed=42)
    genotypes = rng.integers(0, 2, size=(n_genotypes, N)).astype(np.int64)

    def run():
        for g in range(n_genotypes):
            landscape.fitness(genotypes[g])

    median_us = bench_fn(run)

    print(f"NK_FITNESS_10x2_1000_US={median_us:.1f}")
    print()
    print(f"Python/NumPy NK fitness benchmark — NumPy {np.__version__}")
    print(f"  Config: N={N}, K={K}, {n_genotypes} genotypes")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
