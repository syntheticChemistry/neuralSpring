# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark: RK4 integration of 3-variable GRN ODE (2000 steps, dt=0.01)."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200


def hill_activation(x: float, a: float, K: float, n: float) -> float:
    """Activation: a * x^n / (K^n + x^n)."""
    kn = K**n
    xn = x**n if x > 0 else 0.0
    return a * xn / (kn + xn + 1e-20)


def hill_repression(x: float, a: float, K: float, n: float) -> float:
    """Repression: a * K^n / (K^n + x^n)."""
    kn = K**n
    xn = x**n if x > 0 else 0.0
    return a * kn / (kn + xn + 1e-20)


def grn_rhs(state: np.ndarray, signal: float, params: dict) -> np.ndarray:
    """RHS of GRN ODE — same math as regulatory_network.py (3-variable variant)."""
    biofilm, motility, virulence = state
    alpha_b, alpha_m, alpha_v = params["alpha"]
    K, n = params["K"], params["n"]
    gamma = params["gamma"]

    db = alpha_b * hill_activation(signal, 1.0, K, n) * hill_repression(
        motility, 1.0, K, n
    ) - gamma * biofilm
    dm = alpha_m * hill_repression(biofilm, 1.0, K, n) * hill_activation(
        virulence, 1.0, K, n
    ) - gamma * motility
    dv = alpha_v * hill_activation(biofilm, 1.0, K, n) - gamma * virulence
    return np.array([db, dm, dv])


def rk4_step(
    x: np.ndarray, signal: float, params: dict, dt: float
) -> np.ndarray:
    """Single RK4 step."""
    k1 = grn_rhs(x, signal, params)
    k2 = grn_rhs(x + 0.5 * dt * k1, signal, params)
    k3 = grn_rhs(x + 0.5 * dt * k2, signal, params)
    k4 = grn_rhs(x + dt * k3, signal, params)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_grn(
    x0: np.ndarray,
    signal: float,
    params: dict,
    n_steps: int = 2000,
    dt: float = 0.01,
) -> np.ndarray:
    """Integrate GRN ODE for n_steps RK4 steps."""
    x = x0.copy()
    for _ in range(n_steps):
        x = rk4_step(x, signal, params, dt)
        x = np.maximum(x, 0.0)
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
    x0 = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    signal = 0.5
    params = {
        "alpha": [1.0, 1.0, 1.0],
        "K": 0.5,
        "n": 2.0,
        "gamma": 0.1,
    }
    n_steps, dt = 2000, 0.01

    def run():
        integrate_grn(x0, signal, params, n_steps=n_steps, dt=dt)

    median_us = bench_fn(run)

    print(f"RK4_GRN_2000_US={median_us:.1f}")
    print()
    print(f"Python/NumPy RK4 GRN benchmark — NumPy {np.__version__}")
    print(f"  Config: 3-variable GRN (biofilm, motility, virulence)")
    print(f"  {n_steps} RK4 steps, dt={dt}, 4 RHS evals/step = {n_steps * 4} calls")
    print(f"  Median: {median_us:.1f} µs over {ITERATIONS} iterations (warmup={WARMUP})")
