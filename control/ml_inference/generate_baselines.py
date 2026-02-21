# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Generate ML inference baselines for neuralSpring BarraCUDA validation.

Produces JSON files with hardcoded weights, inputs, and expected outputs
for an MLP and a transformer encoder block.  All randomness uses seed=42
for exact reproducibility.

Usage:
    python control/ml_inference/generate_baselines.py

Provenance:
    PyTorch 2.2+, Python 3.10+, seed=42, Xavier uniform init.
"""

import json
import math

import numpy as np

np.random.seed(42)


# ── MLP: input(4) -> 64 -> 64 -> 10 with ReLU ──────────────────────────


def generate_mlp_baseline():
    """3-layer MLP with ReLU activations, softmax output."""
    dims = [4, 64, 64, 10]

    weights = []
    biases = []
    for i in range(len(dims) - 1):
        fan_in, fan_out = dims[i], dims[i + 1]
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        w = np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)
        b = np.zeros(fan_out, dtype=np.float32)
        weights.append(w)
        biases.append(b)

    x = np.array([0.5, -0.3, 0.8, -0.1], dtype=np.float32)

    hidden = x
    for i in range(len(weights) - 1):
        hidden = hidden @ weights[i] + biases[i]
        hidden = np.maximum(hidden, 0.0)  # ReLU

    logits = hidden @ weights[-1] + biases[-1]

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    return {
        "architecture": "MLP: 4 -> 64 (ReLU) -> 64 (ReLU) -> 10 (softmax)",
        "seed": 42,
        "input": x.tolist(),
        "weights": [w.flatten().tolist() for w in weights],
        "weight_shapes": [[w.shape[0], w.shape[1]] for w in weights],
        "biases": [b.tolist() for b in biases],
        "logits": logits.tolist(),
        "output": probs.tolist(),
        "predicted_class": int(np.argmax(probs)),
    }


# ── Transformer encoder block ───────────────────────────────────────────


def generate_transformer_baseline():
    """Single pre-norm transformer encoder block.

    Config: d_model=32, n_heads=4, d_ff=128, seq_len=8.
    Pre-norm: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual.
    """
    d_model = 32
    n_heads = 4
    d_head = d_model // n_heads  # 8
    d_ff = 128
    seq_len = 8
    eps = 1e-5

    def xavier(fan_in, fan_out):
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

    # Attention weights: Q, K, V projections + output
    w_q = xavier(d_model, d_model)
    w_k = xavier(d_model, d_model)
    w_v = xavier(d_model, d_model)
    w_o = xavier(d_model, d_model)

    # FFN weights
    w_ff1 = xavier(d_model, d_ff)
    b_ff1 = np.zeros(d_ff, dtype=np.float32)
    w_ff2 = xavier(d_ff, d_model)
    b_ff2 = np.zeros(d_model, dtype=np.float32)

    # LayerNorm parameters (identity: gamma=1, beta=0)
    ln1_gamma = np.ones(d_model, dtype=np.float32)
    ln1_beta = np.zeros(d_model, dtype=np.float32)
    ln2_gamma = np.ones(d_model, dtype=np.float32)
    ln2_beta = np.zeros(d_model, dtype=np.float32)

    # Input: random [seq_len, d_model]
    x = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1

    def layer_norm(t, gamma, beta, eps=1e-5):
        mean = t.mean(axis=-1, keepdims=True)
        var = t.var(axis=-1, keepdims=True)
        return gamma * (t - mean) / np.sqrt(var + eps) + beta

    def gelu(t):
        return 0.5 * t * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (t + 0.044715 * t**3)))

    def sdpa(q, k, v, d_k):
        scores = (q @ k.T) / math.sqrt(d_k)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        return attn @ v

    # ── Forward pass ──
    # Pre-norm attention
    normed1 = layer_norm(x, ln1_gamma, ln1_beta, eps)

    q_full = normed1 @ w_q  # [seq, d_model]
    k_full = normed1 @ w_k
    v_full = normed1 @ w_v

    # Multi-head: split into heads, apply attention, concat
    q_heads = q_full.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)  # [heads, seq, d_head]
    k_heads = k_full.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    v_heads = v_full.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

    head_outputs = []
    for h in range(n_heads):
        head_outputs.append(sdpa(q_heads[h], k_heads[h], v_heads[h], d_head))

    # Concat heads: [seq, d_model]
    attn_out = np.concatenate(head_outputs, axis=-1)
    attn_proj = attn_out @ w_o

    # Residual 1
    after_attn = x + attn_proj

    # Pre-norm FFN
    normed2 = layer_norm(after_attn, ln2_gamma, ln2_beta, eps)
    ffn_hidden = gelu(normed2 @ w_ff1 + b_ff1)
    ffn_out = ffn_hidden @ w_ff2 + b_ff2

    # Residual 2
    output = after_attn + ffn_out

    return {
        "architecture": "Pre-norm transformer encoder block",
        "config": {
            "d_model": d_model,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "seq_len": seq_len,
            "epsilon": eps,
        },
        "seed": 42,
        "input": x.flatten().tolist(),
        "input_shape": [seq_len, d_model],
        "weights": {
            "w_q": w_q.flatten().tolist(),
            "w_k": w_k.flatten().tolist(),
            "w_v": w_v.flatten().tolist(),
            "w_o": w_o.flatten().tolist(),
            "w_ff1": w_ff1.flatten().tolist(),
            "b_ff1": b_ff1.tolist(),
            "w_ff2": w_ff2.flatten().tolist(),
            "b_ff2": b_ff2.tolist(),
        },
        "output": output.flatten().tolist(),
        "output_shape": [seq_len, d_model],
        "after_attention": after_attn.flatten().tolist(),
    }


if __name__ == "__main__":
    import os

    out_dir = os.path.dirname(os.path.abspath(__file__))

    mlp = generate_mlp_baseline()
    with open(os.path.join(out_dir, "mlp_baseline.json"), "w") as f:
        json.dump(mlp, f, indent=2)
    print(f"MLP: predicted class={mlp['predicted_class']}, top prob={max(mlp['output']):.4f}")

    tfm = generate_transformer_baseline()
    with open(os.path.join(out_dir, "transformer_baseline.json"), "w") as f:
        json.dump(tfm, f, indent=2)
    print(
        f"Transformer: output shape={tfm['output_shape']}, "
        f"output norm={np.linalg.norm(tfm['output']):.4f}"
    )

    print("Baselines written.")
