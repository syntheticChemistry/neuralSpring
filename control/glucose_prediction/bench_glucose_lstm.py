# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark: LSTM reservoir forward + autocorrelation (Paper 026, Chuna glucose)."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

WARMUP = 10
ITERATIONS = 200

SEQ_LEN = 24
HIDDEN = 24
N_DAYS = 7
SAMPLES_PER_DAY = 288  # 5-min intervals


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def lstm_step(x_val, h_prev, c_prev, w_i, w_h, b):
    """Single LSTM cell forward — same math as sequence.rs::lstm_cell."""
    hs = h_prev.shape[0]
    gates = w_i @ np.array([x_val]) + w_h @ h_prev + b
    f_gate = sigmoid(gates[:hs])
    i_gate = sigmoid(gates[hs : 2 * hs])
    g_gate = np.tanh(gates[2 * hs : 3 * hs])
    o_gate = sigmoid(gates[3 * hs :])
    c_new = f_gate * c_prev + i_gate * g_gate
    h_new = o_gate * np.tanh(c_new)
    return h_new, c_new


def lstm_reservoir(data, w_i, w_h, b, hidden_size, seq_len):
    """Run LSTM over a window, return [mean, std, last] features."""
    h = np.zeros(hidden_size, dtype=np.float64)
    c = np.zeros(hidden_size, dtype=np.float64)
    all_h = []
    for val in data[:seq_len]:
        h, c = lstm_step(val, h, c, w_i, w_h, b)
        all_h.append(h.copy())
    arr = np.array(all_h)
    mean_feat = arr.mean(axis=0)
    std_feat = arr.std(axis=0)
    return np.concatenate([mean_feat, std_feat, h])


def autocorrelation(series, max_lag):
    """Normalized autocorrelation — same math as glucose_prediction.rs."""
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    acor = np.empty(max_lag, dtype=np.float64)
    for lag in range(max_lag):
        cov = np.sum((series[: n - lag] - mean) * (series[lag:] - mean)) / n
        acor[lag] = cov / max(var, 1e-30)
    return acor


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
    hs = HIDDEN
    n_samples = N_DAYS * SAMPLES_PER_DAY

    # Synthetic CGM signal (simplified)
    glucose = 120.0 + 15.0 * np.sin(
        2 * np.pi * np.arange(n_samples) / SAMPLES_PER_DAY
    ) + rng.normal(0, 5, n_samples)
    glucose = np.clip(glucose, 40, 400)

    # LSTM weights (random, consistent seed)
    w_i = rng.standard_normal((4 * hs, 1)).astype(np.float64) * 0.5
    w_h = rng.standard_normal((4 * hs, hs)).astype(np.float64) * 0.1
    b = np.zeros(4 * hs, dtype=np.float64)
    b[hs : 2 * hs] = 1.0  # forget bias

    window = glucose[:SEQ_LEN]

    # Benchmark LSTM reservoir forward
    def run_lstm():
        lstm_reservoir(window, w_i, w_h, b, hs, SEQ_LEN)

    lstm_us = bench_fn(run_lstm)

    # Benchmark autocorrelation
    max_lag = 100

    def run_acor():
        autocorrelation(glucose[:500], max_lag)

    acor_us = bench_fn(run_acor)

    # Combined benchmark (reservoir + acor)
    combined_us = lstm_us + acor_us

    print(f"GLUCOSE_LSTM_RESERVOIR_US={lstm_us:.1f}")
    print(f"GLUCOSE_AUTOCORRELATION_US={acor_us:.1f}")
    print(f"GLUCOSE_COMBINED_US={combined_us:.1f}")
    print()
    print(f"Python/NumPy LSTM glucose benchmark — NumPy {np.__version__}")
    print(f"  LSTM reservoir: {lstm_us:.1f} µs (hs={hs}, seq_len={SEQ_LEN})")
    print(f"  Autocorrelation: {acor_us:.1f} µs (n=500, max_lag={max_lag})")
    print(f"  Combined: {combined_us:.1f} µs")
