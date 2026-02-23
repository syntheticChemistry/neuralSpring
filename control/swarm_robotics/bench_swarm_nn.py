# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark: Swarm NN forward pass (20 controllers × 50 evaluations)."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def neural_forward(params: np.ndarray, sense: float) -> int:
    n_in, n_h, n_out = 1, 4, 5
    w1 = params[:4].reshape(n_in, n_h)
    b1 = params[4:8]
    w2 = params[8:28].reshape(n_h, n_out)
    b2 = params[28:33]
    h = sigmoid(sense * w1 + b1)
    out = sigmoid(h @ w2 + b2)
    return int(np.argmax(out))


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
    rng = np.random.default_rng(123)
    n_ctrl, n_eval = 20, 50
    all_params = [rng.random(33) for _ in range(n_ctrl)]
    inputs = [i / n_eval for i in range(n_eval)]

    def run():
        actions = []
        for params in all_params:
            for sense in inputs:
                actions.append(neural_forward(params, sense))
        return actions

    median_us = bench_fn(run)

    print(f"SWARM_NN_20x50_US={median_us:.1f}")
    print()
    print(f"Python/NumPy swarm NN benchmark — NumPy {np.__version__}")
    print(f"  Config: {n_ctrl} controllers × {n_eval} evaluations = {n_ctrl*n_eval} forward passes")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
