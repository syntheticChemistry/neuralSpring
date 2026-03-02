# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark: Multi-niche batch fitness evaluation (N=20, 4 niches, 200 genotypes)."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200


class MultiNicheLandscape:
    """Multi-niche fitness landscape — same math as eco_dynamics.py."""

    def __init__(self, n_loci, n_niches, niche_width=0.15, seed=42):
        self.n_loci = n_loci
        self.n_niches = n_niches
        rng = np.random.default_rng(seed)
        self.niche_optima = rng.integers(0, 2, (n_niches, n_loci))
        self.niche_capacity = np.ones(n_niches)
        self.niche_width = np.full(n_niches, niche_width)

    def batch_fitness(self, population, frequency_dependent=False):
        dists = np.array(
            [
                np.sum(population != self.niche_optima[i], axis=1) / self.n_loci
                for i in range(self.n_niches)
            ]
        ).T
        niche_fits = self.niche_capacity[np.newaxis, :] * np.exp(
            -(dists**2) / (2 * self.niche_width[np.newaxis, :] ** 2)
        )
        if frequency_dependent:
            occupancy = np.sum(dists < 0.25, axis=0).astype(float)
            crowding = 1.0 / (1.0 + 0.05 * occupancy)
            niche_fits = niche_fits * crowding[np.newaxis, :]
        return np.max(niche_fits, axis=1)


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
    N_LOCI = 20
    N_NICHES = 4
    POP_SIZE = 200

    landscape = MultiNicheLandscape(N_LOCI, N_NICHES, niche_width=0.15, seed=42)
    population = rng.integers(0, 2, size=(POP_SIZE, N_LOCI)).astype(np.int64)

    def run():
        landscape.batch_fitness(population, frequency_dependent=True)

    median_us = bench_fn(run)

    print(f"ECO_BATCH_FITNESS_20x200x4_US={median_us:.1f}")
    print()
    print(f"Python/NumPy eco batch fitness benchmark — NumPy {np.__version__}")
    print(f"  Config: N_LOCI={N_LOCI}, N_NICHES={N_NICHES}, POP_SIZE={POP_SIZE}")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
