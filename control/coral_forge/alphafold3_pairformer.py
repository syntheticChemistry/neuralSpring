# SPDX-License-Identifier: AGPL-3.0-or-later
#
# alphafold3_pairformer.py — NumPy reference for Pairformer block (nF-03 Phase B)
#
# The Pairformer is AlphaFold3's simplified Evoformer that operates on pair
# representations only (no MSA track). One block:
#
#   1. Triangle multiplicative update — outgoing (Algorithm 11)
#   2. Triangle multiplicative update — incoming (Algorithm 12)
#   3. Triangle attention (starting/ending, Algorithms 13-14)
#   4. Pair transition FFN (Linear → GELU → Linear)
#   5. Timestep conditioning (diffusion time embedding)
#
# ~90% reuse of existing Evoformer primitives from nF-02.
#
# Reference: Abramson et al. Nature 630:493-500 (2024)
#
# Usage: python3 control/coral_forge/alphafold3_pairformer.py
# Output: control/coral_forge/pairformer_baselines.json

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evoformer_primitives import (
    gelu,
    layer_norm,
    softmax,
    sdpa_scores,
    triangle_mul_outgoing,
    triangle_mul_incoming,
    triangle_attention_scores,
)
from alphafold3_diffusion import pair_transition_ffn

SEED = 42
N_RES = 8
D_PAIR = 4
N_HEADS = 2
HEAD_DIM = 4
D_HIDDEN = 16


# ═══════════════════════════════════════════════════════════════════
# Timestep embedding
# ═══════════════════════════════════════════════════════════════════

def sinusoidal_embedding(t, d_model):
    """Sinusoidal timestep embedding (Vaswani et al. 2017)."""
    emb = np.zeros(d_model, dtype=np.float64)
    for i in range(d_model):
        if i % 2 == 0:
            emb[i] = math.sin(t / (10000.0 ** (i / d_model)))
        else:
            emb[i] = math.cos(t / (10000.0 ** ((i - 1) / d_model)))
    return emb


def condition_pair_with_timestep(pair_repr, t_emb, w_cond, b_cond):
    """
    Add timestep conditioning to pair representation.
    t_emb: [d_model] → Linear → [d_pair] → broadcast-add to pair[i,j,:].
    """
    n, _, d = pair_repr.shape
    # Project timestep embedding to pair dimension
    cond = t_emb @ w_cond + b_cond  # [d_pair]
    return pair_repr + cond.reshape(1, 1, d)


# ═══════════════════════════════════════════════════════════════════
# Pairformer Block
# ═══════════════════════════════════════════════════════════════════

def pairformer_block(pair, weights, t_emb=None):
    """
    One Pairformer block iteration.

    Steps:
      1. LayerNorm → Triangle multiplicative outgoing
      2. LayerNorm → Triangle multiplicative incoming
      3. LayerNorm → Triangle attention (with pair bias)
      4. LayerNorm → Pair transition FFN
      5. (Optional) Timestep conditioning
    """
    n, _, d = pair.shape

    # 1. Triangle multiplicative outgoing (Algorithm 11)
    gamma_ln1 = weights["ln1_gamma"]
    beta_ln1 = weights["ln1_beta"]
    normed = layer_norm(pair.reshape(-1, d), gamma_ln1, beta_ln1).reshape(n, n, d)
    proj_a = np.einsum("ijd,dk->ijk", normed, weights["tri_out_wa"])
    proj_b = np.einsum("ijd,dk->ijk", normed, weights["tri_out_wb"])
    gate = 1.0 / (1.0 + np.exp(-np.einsum("ijd,dk->ijk", normed, weights["tri_out_wg"])))
    tri_out = triangle_mul_outgoing(proj_a, proj_b)
    pair = pair + gate * tri_out

    # 2. Triangle multiplicative incoming (Algorithm 12)
    normed = layer_norm(pair.reshape(-1, d), gamma_ln1, beta_ln1).reshape(n, n, d)
    proj_a_in = np.einsum("ijd,dk->ijk", normed, weights["tri_in_wa"])
    proj_b_in = np.einsum("ijd,dk->ijk", normed, weights["tri_in_wb"])
    gate_in = 1.0 / (1.0 + np.exp(-np.einsum("ijd,dk->ijk", normed, weights["tri_in_wg"])))
    tri_in = triangle_mul_incoming(proj_a_in, proj_b_in)
    pair = pair + gate_in * tri_in

    # 3. Triangle attention (Algorithms 13-14)
    # Per-row attention over the pair representation
    normed = layer_norm(pair.reshape(-1, d), gamma_ln1, beta_ln1).reshape(n, n, d)
    h = weights["n_heads"]
    hd = weights["head_dim"]
    # query/key/value: [row, col, head, head_dim] = [R, N, H, D]
    q = np.einsum("ijd,dhk->ijhk", normed,
                  weights["tri_attn_wq"].reshape(d, h, hd))
    k = np.einsum("ijd,dhk->ijhk", normed,
                  weights["tri_attn_wk"].reshape(d, h, hd))
    v = np.einsum("ijd,dhk->ijhk", normed,
                  weights["tri_attn_wv"].reshape(d, h, hd))
    # Pair bias: pair_repr → [H, N, N] via a learned projection
    # For simplicity, use normed pair[:,:,0:H] as bias
    bias = np.zeros((h, n, n))
    for hi in range(min(h, d)):
        bias[hi] = normed[:, :, hi]
    scores = triangle_attention_scores(q, k, bias)  # [R, H, N, N]
    # Softmax over last (key) dimension
    scores_flat = scores.reshape(-1, scores.shape[-1])
    attn = softmax(scores_flat).reshape(scores.shape)
    # Apply: weighted sum of values per (row, head)
    # attn: [R, H, j, k], v: [R, k, H, D] → [R, j, H, D]
    attended = np.einsum("rhjk,rkhd->rjhd", attn, v)
    # Merge heads and truncate/project to d_pair
    merged = attended.reshape(n, n, h * hd)[:, :, :d]
    pair = pair + merged

    # 4. Pair transition FFN
    normed = layer_norm(pair.reshape(-1, d), gamma_ln1, beta_ln1).reshape(n, n, d)
    ffn_out = pair_transition_ffn(
        normed, weights["ffn_w1"], weights["ffn_b1"],
        weights["ffn_w2"], weights["ffn_b2"]
    )
    pair = pair + ffn_out

    # 5. Timestep conditioning (if provided)
    if t_emb is not None:
        cond = t_emb @ weights["cond_w"] + weights["cond_b"]
        pair = pair + cond.reshape(1, 1, d)

    return pair


# ═══════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════

def run_tests():
    rng = np.random.default_rng(SEED)
    results = {}
    n_pass = 0
    n_fail = 0

    def check(name, condition, detail=""):
        nonlocal n_pass, n_fail
        if condition:
            n_pass += 1
            print(f"  [PASS] {name}")
        else:
            n_fail += 1
            print(f"  [FAIL] {name}: {detail}")

    # ─── Timestep embedding ─────────────────────────────────────────
    print("\n--- Timestep Embedding ---\n")

    t_emb_0 = sinusoidal_embedding(0.0, D_PAIR)
    t_emb_25 = sinusoidal_embedding(25.0, D_PAIR)
    t_emb_49 = sinusoidal_embedding(49.0, D_PAIR)
    check("tsemb: shape", t_emb_0.shape == (D_PAIR,))
    check("tsemb: different timesteps → different embeddings",
          not np.allclose(t_emb_0, t_emb_25))
    check("tsemb: all finite", np.all(np.isfinite(t_emb_0)))

    results["t_emb_0"] = t_emb_0.tolist()
    results["t_emb_25"] = t_emb_25.tolist()
    results["t_emb_49"] = t_emb_49.tolist()

    # ─── Timestep conditioning ──────────────────────────────────────
    print("\n--- Timestep Conditioning ---\n")

    pair_repr = rng.standard_normal((N_RES, N_RES, D_PAIR))
    w_cond = rng.standard_normal((D_PAIR, D_PAIR)) * 0.1
    b_cond = np.zeros(D_PAIR)
    conditioned = condition_pair_with_timestep(pair_repr, t_emb_25, w_cond, b_cond)
    check("cond: shape preserved", conditioned.shape == pair_repr.shape)
    check("cond: output differs from input", not np.allclose(conditioned, pair_repr))
    check("cond: all residue pairs get same shift",
          np.allclose(conditioned[0, 0] - pair_repr[0, 0],
                      conditioned[3, 5] - pair_repr[3, 5]))

    results["pair_repr"] = pair_repr.tolist()
    results["w_cond"] = w_cond.tolist()
    results["b_cond"] = b_cond.tolist()
    results["conditioned"] = conditioned.tolist()

    # ─── Pairformer block ───────────────────────────────────────────
    print("\n--- Pairformer Block ---\n")

    # Generate weights
    pair_input = rng.standard_normal((N_RES, N_RES, D_PAIR)) * 0.1
    weights = {
        "ln1_gamma": np.ones(D_PAIR),
        "ln1_beta": np.zeros(D_PAIR),
        "tri_out_wa": rng.standard_normal((D_PAIR, D_PAIR)) * 0.1,
        "tri_out_wb": rng.standard_normal((D_PAIR, D_PAIR)) * 0.1,
        "tri_out_wg": rng.standard_normal((D_PAIR, D_PAIR)) * 0.1,
        "tri_in_wa": rng.standard_normal((D_PAIR, D_PAIR)) * 0.1,
        "tri_in_wb": rng.standard_normal((D_PAIR, D_PAIR)) * 0.1,
        "tri_in_wg": rng.standard_normal((D_PAIR, D_PAIR)) * 0.1,
        "n_heads": N_HEADS,
        "head_dim": HEAD_DIM,
        "tri_attn_wq": rng.standard_normal((D_PAIR, N_HEADS * HEAD_DIM)) * 0.1,
        "tri_attn_wk": rng.standard_normal((D_PAIR, N_HEADS * HEAD_DIM)) * 0.1,
        "tri_attn_wv": rng.standard_normal((D_PAIR, N_HEADS * HEAD_DIM)) * 0.1,
        "ffn_w1": rng.standard_normal((D_PAIR, D_HIDDEN)) * 0.1,
        "ffn_b1": np.zeros(D_HIDDEN),
        "ffn_w2": rng.standard_normal((D_HIDDEN, D_PAIR)) * 0.1,
        "ffn_b2": np.zeros(D_PAIR),
        "cond_w": w_cond,
        "cond_b": b_cond,
    }

    # Without timestep conditioning
    out_no_cond = pairformer_block(pair_input, weights, t_emb=None)
    check("pf: output shape", out_no_cond.shape == pair_input.shape)
    check("pf: output differs from input", not np.allclose(out_no_cond, pair_input))
    check("pf: output finite", np.all(np.isfinite(out_no_cond)))

    # With timestep conditioning
    out_with_cond = pairformer_block(pair_input, weights, t_emb=t_emb_25)
    check("pf+cond: output shape", out_with_cond.shape == pair_input.shape)
    check("pf+cond: conditioning changes output",
          not np.allclose(out_no_cond, out_with_cond))

    results["pair_input"] = pair_input.tolist()
    results["pf_out_no_cond"] = out_no_cond.tolist()
    results["pf_out_with_cond"] = out_with_cond.tolist()

    # Save all weights for Rust validation
    for k, v in weights.items():
        if isinstance(v, np.ndarray):
            results[f"w_{k}"] = v.tolist()
        else:
            results[f"w_{k}"] = v

    # ─── Multi-block iteration ──────────────────────────────────────
    print("\n--- Multi-Block Iteration (3 blocks) ---\n")

    pair_evolving = pair_input.copy()
    for block_idx in range(3):
        t = 49 - block_idx * 20  # Decreasing timestep
        t_emb = sinusoidal_embedding(float(t), D_PAIR)
        pair_evolving = pairformer_block(pair_evolving, weights, t_emb=t_emb)

    check("multi: 3 blocks completed without NaN",
          np.all(np.isfinite(pair_evolving)))
    check("multi: output differs from single block",
          not np.allclose(pair_evolving, out_with_cond))
    check("multi: representation norm bounded",
          np.linalg.norm(pair_evolving) < 1000.0,
          f"norm = {np.linalg.norm(pair_evolving):.4f}")

    results["multi_block_out"] = pair_evolving.tolist()

    # ─── Summary ────────────────────────────────────────────────────
    print(f"\n=== alphafold3_pairformer: {n_pass}/{n_pass + n_fail} PASS, {n_fail} FAIL ===")

    out_path = Path(__file__).parent / "pairformer_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Baselines written to {out_path}")

    return n_fail == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
