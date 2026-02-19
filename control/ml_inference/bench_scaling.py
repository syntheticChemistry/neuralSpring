#!/usr/bin/env python3
"""Scaling benchmark: NumPy MLP and Transformer at multiple model sizes.

Outputs machine-readable lines for the Rust harness to parse:
    SCALE_MLP_<name>_US=<median_us>
    SCALE_TF_<name>_US=<median_us>

When BENCH_SINGLE_THREAD=1 is set, also runs with OPENBLAS_NUM_THREADS=1
to establish a fair single-core comparison against BarraCUDA CPU shaders.
"""

import math
import os
import time
import numpy as np

WARMUP = 10
ITERATIONS = 200

SCALES = [
    {"name": "tiny",    "input": 4,    "hidden": 64,   "output": 10,  "seq": 8,   "d": 32,   "heads": 4,  "dff": 128},
    {"name": "small",   "input": 32,   "hidden": 128,  "output": 32,  "seq": 32,  "d": 128,  "heads": 8,  "dff": 512},
    {"name": "medium",  "input": 128,  "hidden": 512,  "output": 128, "seq": 64,  "d": 256,  "heads": 8,  "dff": 1024},
    {"name": "large",   "input": 256,  "hidden": 1024, "output": 256, "seq": 128, "d": 512,  "heads": 8,  "dff": 2048},
    {"name": "xlarge",  "input": 512,  "hidden": 2048, "output": 512, "seq": 256, "d": 1024, "heads": 16, "dff": 4096},
]


def mlp_forward(x, weights, biases):
    h = x
    for i in range(len(weights) - 1):
        h = h @ weights[i] + biases[i]
        h = np.maximum(h, 0.0)
    logits = h @ weights[-1] + biases[-1]
    exp_l = np.exp(logits - np.max(logits))
    return exp_l / np.sum(exp_l)


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
    seq_len = cfg["seq"]
    n_heads = cfg["heads"]
    d_model = cfg["d"]
    d_head = d_model // n_heads

    normed1 = layer_norm(x)
    q_full = normed1 @ w["wq"]
    k_full = normed1 @ w["wk"]
    v_full = normed1 @ w["wv"]

    q_h = q_full.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    k_h = k_full.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    v_h = v_full.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

    heads = [sdpa(q_h[h], k_h[h], v_h[h], d_head) for h in range(n_heads)]
    attn_out = np.concatenate(heads, axis=-1)
    attn_proj = attn_out @ w["wo"]

    after_attn = x + attn_proj
    normed2 = layer_norm(after_attn)
    ffn_h = gelu(normed2 @ w["wff1"] + w["bff1"])
    ffn_out = ffn_h @ w["wff2"] + w["bff2"]
    return after_attn + ffn_out


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


def make_random(shape, seed):
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32) * 0.1


def run_all_scales():
    """Run benchmarks for all scales, return list of (name, mlp_us, tf_us)."""
    results = []
    for s in SCALES:
        inp, hid, out = s["input"], s["hidden"], s["output"]

        x_mlp = make_random((1, inp), 1)
        w0 = make_random((inp, hid), 2)
        b0 = make_random((1, hid), 3)
        w1 = make_random((hid, hid), 4)
        b1 = make_random((1, hid), 5)
        w2 = make_random((hid, out), 6)
        b2 = make_random((1, out), 7)
        weights = [w0, w1, w2]
        biases = [b0, b1, b2]
        mlp_us = bench_fn(lambda: mlp_forward(x_mlp, weights, biases))

        d, dff, seq = s["d"], s["dff"], s["seq"]
        x_tf = make_random((seq, d), 10)
        w_tf = {
            "wq": make_random((d, d), 11),
            "wk": make_random((d, d), 12),
            "wv": make_random((d, d), 13),
            "wo": make_random((d, d), 14),
            "wff1": make_random((d, dff), 15),
            "bff1": make_random((1, dff), 16),
            "wff2": make_random((dff, d), 17),
            "bff2": make_random((1, d), 18),
        }
        tf_us = bench_fn(lambda: transformer_forward(x_tf, w_tf, s))
        results.append((s["name"], mlp_us, tf_us))
    return results


if __name__ == "__main__":
    print(f"Python/NumPy Scaling Benchmark — NumPy {np.__version__}")

    # Multi-threaded baseline (default OpenBLAS)
    mt_results = run_all_scales()
    for name, mlp_us, tf_us in mt_results:
        print(f"SCALE_MLP_{name.upper()}_US={mlp_us:.1f}")
        print(f"SCALE_TF_{name.upper()}_US={tf_us:.1f}")

    # Single-threaded baseline (fair comparison against single-core shader)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    st_results = run_all_scales()
    for name, mlp_us, tf_us in st_results:
        print(f"SCALE_MLP_1T_{name.upper()}_US={mlp_us:.1f}")
        print(f"SCALE_TF_1T_{name.upper()}_US={tf_us:.1f}")

    # Human-readable summary
    print()
    hdr = f"{'Scale':>8}  {'MLP(mt)':>10}  {'MLP(1t)':>10}  {'TF(mt)':>10}  {'TF(1t)':>10}"
    print(hdr)
    print("-" * len(hdr))
    for (name, m_mt, t_mt), (_, m_st, t_st) in zip(mt_results, st_results):
        print(f"{name:>8}  {m_mt:>10.1f}  {m_st:>10.1f}  {t_mt:>10.1f}  {t_st:>10.1f}")
