# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark: Replicator dynamics (2-strategy PD, 10000 steps, dt=0.001)."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200


def replicator_trajectory(x0, payoff, n_steps, dt):
    """Replicator dynamics — same math as game_theory.py."""
    x = x0.copy()
    for _ in range(n_steps):
        fitness = payoff @ x
        avg_fitness = x @ fitness
        dx = x * (fitness - avg_fitness)
        x = x + dt * dx
        x = np.maximum(x, 0.0)
        x /= x.sum()
    return x


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
    b, c = 3.0, 1.0
    payoff = np.array([[b - c, -c], [b, 0.0]], dtype=np.float64)
    x0 = np.array([0.5, 0.5], dtype=np.float64)
    n_steps, dt = 10000, 0.001

    def run():
        replicator_trajectory(x0, payoff, n_steps, dt)

    median_us = bench_fn(run)

    print(f"REPLICATOR_10000_US={median_us:.1f}")
    print()
    print(f"Python/NumPy replicator dynamics benchmark — NumPy {np.__version__}")
    print(f"  Config: 2-strategy PD (b={b}, c={c}), {n_steps} steps, dt={dt}")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
