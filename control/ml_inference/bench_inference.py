# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark NumPy ML inference for neuralSpring 3-way comparison.

Measures MLP and Transformer encoder block forward pass times using
the same weights/inputs as the BarraCUDA benchmarks.

Usage:
    python control/ml_inference/bench_inference.py
"""

import json
import math
import os
import time

import numpy as np

WARMUP = 10
ITERATIONS = 200

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(name):
    with open(os.path.join(SCRIPT_DIR, name)) as f:
        return json.load(f)


# ── MLP ─────────────────────────────────────────────────────────────


def mlp_forward(x, weights, biases):
    hidden = x
    for i in range(len(weights) - 1):
        hidden = hidden @ weights[i] + biases[i]
        hidden = np.maximum(hidden, 0.0)  # ReLU
    logits = hidden @ weights[-1] + biases[-1]
    exp_l = np.exp(logits - np.max(logits))
    return exp_l / np.sum(exp_l)


def bench_mlp():
    b = load_json("mlp_baseline.json")
    x = np.array(b["input"], dtype=np.float32)
    weights = [
        np.array(w, dtype=np.float32).reshape(s)
        for w, s in zip(b["weights"], b["weight_shapes"], strict=True)
    ]
    biases = [np.array(bi, dtype=np.float32) for bi in b["biases"]]

    # Correctness
    out = mlp_forward(x, weights, biases)
    pred = int(np.argmax(out))
    print(f"  Predicted: {pred} (expected {b['predicted_class']})")
    print(f"  Max diff:  {np.max(np.abs(out - np.array(b['output']))):.6e}")

    # Warmup
    for _ in range(WARMUP):
        mlp_forward(x, weights, biases)

    # Benchmark
    timings = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter_ns()
        mlp_forward(x, weights, biases)
        timings.append(time.perf_counter_ns() - t0)

    timings.sort()
    median_us = timings[len(timings) // 2] / 1000
    min_us = timings[0] / 1000
    max_us = timings[-1] / 1000
    mean_us = sum(timings) / len(timings) / 1000
    print(f"\n  MLP Forward ({ITERATIONS} iterations):")
    print(f"    Median:     {median_us:.1f}µs")
    print(f"    Mean:       {mean_us:.1f}µs")
    print(f"    Min:        {min_us:.1f}µs")
    print(f"    Max:        {max_us:.1f}µs")
    print(f"    Throughput: {1_000_000 / median_us:.0f} inferences/sec")
    return median_us, min_us, max_us


# ── Transformer ──────────────────────────────────────────────────────


def layer_norm(t, eps=1e-5):
    mean = t.mean(axis=-1, keepdims=True)
    var = t.var(axis=-1, keepdims=True)
    return (t - mean) / np.sqrt(var + eps)


def gelu(t):
    return 0.5 * t * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (t + 0.044715 * t**3)))


def sdpa(q, k, v, d_k):
    scores = (q @ k.T) / math.sqrt(d_k)
    exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    return attn @ v


def transformer_forward(x, w, cfg):
    seq_len = cfg["seq_len"]
    n_heads = cfg["n_heads"]
    d_model = cfg["d_model"]
    d_head = d_model // n_heads
    eps = cfg["epsilon"]

    normed1 = layer_norm(x, eps)
    q_full = normed1 @ w["w_q"]
    k_full = normed1 @ w["w_k"]
    v_full = normed1 @ w["w_v"]

    q_h = q_full.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    k_h = k_full.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    v_h = v_full.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

    heads = [sdpa(q_h[h], k_h[h], v_h[h], d_head) for h in range(n_heads)]
    attn_out = np.concatenate(heads, axis=-1)
    attn_proj = attn_out @ w["w_o"]

    after_attn = x + attn_proj
    normed2 = layer_norm(after_attn, eps)
    ffn_h = gelu(normed2 @ w["w_ff1"] + w["b_ff1"])
    ffn_out = ffn_h @ w["w_ff2"] + w["b_ff2"]
    return after_attn + ffn_out


def bench_transformer():
    b = load_json("transformer_baseline.json")
    cfg = b["config"]
    d = cfg["d_model"]
    d_ff = cfg["d_ff"]

    x = np.array(b["input"], dtype=np.float32).reshape(b["input_shape"])
    w = {
        "w_q": np.array(b["weights"]["w_q"], dtype=np.float32).reshape(d, d),
        "w_k": np.array(b["weights"]["w_k"], dtype=np.float32).reshape(d, d),
        "w_v": np.array(b["weights"]["w_v"], dtype=np.float32).reshape(d, d),
        "w_o": np.array(b["weights"]["w_o"], dtype=np.float32).reshape(d, d),
        "w_ff1": np.array(b["weights"]["w_ff1"], dtype=np.float32).reshape(d, d_ff),
        "b_ff1": np.array(b["weights"]["b_ff1"], dtype=np.float32),
        "w_ff2": np.array(b["weights"]["w_ff2"], dtype=np.float32).reshape(d_ff, d),
        "b_ff2": np.array(b["weights"]["b_ff2"], dtype=np.float32),
    }

    # Correctness
    out = transformer_forward(x, w, cfg)
    expected = np.array(b["output"], dtype=np.float32).reshape(b["input_shape"])
    print(f"  Max diff:  {np.max(np.abs(out - expected)):.6e}")

    # Warmup
    for _ in range(WARMUP):
        transformer_forward(x, w, cfg)

    # Benchmark
    timings = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter_ns()
        transformer_forward(x, w, cfg)
        timings.append(time.perf_counter_ns() - t0)

    timings.sort()
    median_us = timings[len(timings) // 2] / 1000
    min_us = timings[0] / 1000
    max_us = timings[-1] / 1000
    mean_us = sum(timings) / len(timings) / 1000
    print(f"\n  Transformer Block ({ITERATIONS} iterations):")
    print(f"    Median:     {median_us:.1f}µs")
    print(f"    Mean:       {mean_us:.1f}µs")
    print(f"    Min:        {min_us:.1f}µs")
    print(f"    Max:        {max_us:.1f}µs")
    print(f"    Throughput: {1_000_000 / median_us:.0f} blocks/sec")
    return median_us, min_us, max_us


if __name__ == "__main__":
    print("=" * 60)
    print("Python/NumPy ML Inference Benchmark")
    print(f"NumPy {np.__version__}, Warmup={WARMUP}, Iterations={ITERATIONS}")
    print("=" * 60)

    print("\n── MLP (4→64→64→10) ──")
    mlp_med, mlp_min, mlp_max = bench_mlp()

    print("\n── Transformer Block (d=32, h=4, seq=8) ──")
    tfm_med, tfm_min, tfm_max = bench_transformer()

    print("\n" + "=" * 60)
    print("Summary (median):")
    print(f"  MLP:         {mlp_med:.1f}µs ({1_000_000 / mlp_med:.0f} inf/s)")
    print(f"  Transformer: {tfm_med:.1f}µs ({1_000_000 / tfm_med:.0f} blocks/s)")
    print("=" * 60)

    # Machine-readable output for Rust benchmark harness
    print(f"MLP_MEDIAN_US={mlp_med:.1f}")
    print(f"TRANSFORMER_MEDIAN_US={tfm_med:.1f}")
