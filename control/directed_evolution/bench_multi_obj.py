# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark: Multi-objective fitness (100 genomes × 30 loci × 3 objectives)."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200


def multi_objective_fitness(genotype: np.ndarray, n_objectives: int = 3) -> np.ndarray:
    n = len(genotype)
    chunk = n // n_objectives
    fitnesses = np.zeros(n_objectives)
    for i in range(n_objectives):
        start = i * chunk
        end = start + chunk if i < n_objectives - 1 else n
        segment = genotype[start:end]
        fitnesses[i] = np.mean(segment) + 0.1 * np.std(segment)
    return fitnesses


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
    pop_size, genome_len, n_obj = 100, 30, 3
    population = rng.random((pop_size, genome_len))

    def run():
        results = []
        for i in range(pop_size):
            results.append(multi_objective_fitness(population[i], n_obj))
        return results

    median_us = bench_fn(run)

    print(f"MULTI_OBJ_FITNESS_100x30x3_US={median_us:.1f}")
    print()
    print(f"Python/NumPy multi-objective fitness benchmark — NumPy {np.__version__}")
    print(f"  Config: {pop_size} genomes × {genome_len} loci × {n_obj} objectives")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
