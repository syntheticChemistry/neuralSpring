# SPDX-License-Identifier: AGPL-3.0-or-later
#
# evoformer_primitives.py — NumPy reference implementations of Evoformer ops
#
# Generates deterministic baselines for coralForge Phase B validation.
# These primitives form the computational core of AlphaFold2's Evoformer:
#
#   - GELU activation
#   - Layer normalization
#   - Row-wise softmax
#   - Scaled dot-product attention (scores, apply, full)
#   - Triangle multiplicative update (outgoing, Algorithm 11)
#   - Triangle multiplicative update (incoming, Algorithm 12)
#   - Triangle attention scores (with pair bias, Algorithms 13-14)
#
# Reference: Jumper et al. "Highly accurate protein structure prediction
# with AlphaFold" Nature 596:583-589 (2021)
#
# Usage: python3 control/coral_forge/evoformer_primitives.py
# Output: control/coral_forge/evoformer_baselines.json

import json
import math
import sys
from pathlib import Path

import numpy as np

SEED = 42
N_RES = 8
CHANNELS = 4
N_HEADS = 2
HEAD_DIM = 4
HIDDEN_DIM = 16
BATCH = 1

# ═══════════════════════════════════════════════════════════════════
# Primitives
# ═══════════════════════════════════════════════════════════════════

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """Layer normalization along last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along given axis."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def sdpa_scores(query: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention scores: Q @ K^T / sqrt(d_k).

    query: [B, H, Sq, D]
    key:   [B, H, Skv, D]
    returns: [B, H, Sq, Skv]
    """
    d_k = query.shape[-1]
    return np.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(d_k)


def attention_apply(weights: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Weighted sum: output[b,h,q,d] = sum_k weights[b,h,q,k] * V[b,h,k,d].

    weights: [B, H, Sq, Skv]
    value:   [B, H, Skv, D]
    returns: [B, H, Sq, D]
    """
    return np.einsum("bhqk,bhkd->bhqd", weights, value)


def sdpa_full(query: np.ndarray, key: np.ndarray,
              value: np.ndarray) -> np.ndarray:
    """Full scaled dot-product attention: softmax(QK^T / sqrt(d)) @ V."""
    scores = sdpa_scores(query, key)
    weights = softmax(scores, axis=-1)
    return attention_apply(weights, value)


def triangle_mul_outgoing(proj_a: np.ndarray,
                          proj_b: np.ndarray) -> np.ndarray:
    """Algorithm 11: outgoing edges.

    proj_a: [N, N, C]  (gated projections for row i, edge i→k)
    proj_b: [N, N, C]  (gated projections for row j, edge j→k)
    output[i,j,c] = sum_k proj_a[i,k,c] * proj_b[j,k,c]

    Contracts over shared outgoing index k.
    """
    return np.einsum("ikc,jkc->ijc", proj_a, proj_b)


def triangle_mul_incoming(proj_a: np.ndarray,
                          proj_b: np.ndarray) -> np.ndarray:
    """Algorithm 12: incoming edges.

    proj_a: [N, N, C]  (gated projections for edge k→i)
    proj_b: [N, N, C]  (gated projections for edge k→j)
    output[i,j,c] = sum_k proj_a[k,i,c] * proj_b[k,j,c]

    Contracts over shared incoming index k.
    """
    return np.einsum("kic,kjc->ijc", proj_a, proj_b)


def triangle_attention_scores(
    query: np.ndarray,
    key: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Algorithms 13-14: triangle self-attention scores with pair bias.

    For each row i of the pair representation:
      logit[row, h, j, k] = sum_d Q[row,j,h,d]*K[row,k,h,d]/sqrt(D) + bias[h,j,k]

    query: [R, N, H, D]
    key:   [R, N, H, D]
    bias:  [H, N, N]
    returns: [R, H, N, N]
    """
    d = query.shape[-1]
    scores = np.einsum("rjhd,rkhd->rhjk", query, key) / math.sqrt(d)
    return scores + bias[np.newaxis, :, :, :]


def outer_product_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer product mean: MSA → pair representation.

    a: [N_seq, N_res, C_a]
    b: [N_seq, N_res, C_b]
    returns: [N_res, N_res, C_a * C_b]
    """
    outer = np.einsum("sia,sjb->sijab", a, b)   # [S, N, N, Ca, Cb]
    mean_outer = outer.mean(axis=0)              # [N, N, Ca, Cb]
    n_res = a.shape[1]
    c_a, c_b = a.shape[2], b.shape[2]
    return mean_outer.reshape(n_res, n_res, c_a * c_b)


def msa_row_attention_scores(
    query: np.ndarray,
    key: np.ndarray,
    pair_bias: np.ndarray,
) -> np.ndarray:
    """MSA row attention scores with pair bias.

    query/key: [N_seq, N_res, H, D]
    pair_bias: [H, N_res, N_res]
    returns:   [N_seq, H, N_res, N_res]
    """
    d = query.shape[-1]
    scores = np.einsum("sihd,sjhd->shij", query, key) / math.sqrt(d)
    return scores + pair_bias[np.newaxis, :, :, :]


def msa_row_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    pair_bias: np.ndarray,
) -> np.ndarray:
    """Full MSA row attention: scores → softmax → apply.

    query/key/value: [N_seq, N_res, H, D]
    pair_bias:       [H, N_res, N_res]
    returns:         [N_seq, N_res, H, D]
    """
    scores = msa_row_attention_scores(query, key, pair_bias)
    weights = softmax(scores, axis=-1)
    return np.einsum("shij,sjhd->sihd", weights, value)


def msa_col_attention_scores(
    query: np.ndarray,
    key: np.ndarray,
) -> np.ndarray:
    """MSA column attention scores (no pair bias).

    query/key: [N_seq, N_res, H, D]
    returns:   [N_res, H, N_seq, N_seq]
    """
    d = query.shape[-1]
    return np.einsum("srhd,trhd->rhst", query, key) / math.sqrt(d)


def msa_col_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    """Full MSA column attention: scores → softmax → apply.

    query/key/value: [N_seq, N_res, H, D]
    returns:         [N_seq, N_res, H, D]
    """
    scores = msa_col_attention_scores(query, key)
    weights = softmax(scores, axis=-1)
    return np.einsum("rhst,trhd->srhd", weights, value)


# ═══════════════════════════════════════════════════════════════════
# Structure Module primitives
# ═══════════════════════════════════════════════════════════════════

def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def apply_frame(rot: np.ndarray, trans: np.ndarray,
                point: np.ndarray) -> np.ndarray:
    """Apply rigid-body frame: R @ point + t."""
    return rot @ point + trans


def ipa_scores(
    q_scalar: np.ndarray,
    k_scalar: np.ndarray,
    pair_bias: np.ndarray,
    q_points: np.ndarray,
    k_points: np.ndarray,
    frames_rot: np.ndarray,
    frames_trans: np.ndarray,
    w_l: float, w_c: float, w_p: float, gamma: float,
) -> np.ndarray:
    """IPA attention scores (Algorithm 22).

    q_scalar/k_scalar: [N, H, D]
    pair_bias:         [H, N, N]
    q_points/k_points: [N, H, P, 3]
    frames_rot:        [N, 3, 3]
    frames_trans:      [N, 3]
    returns:           [H, N, N]
    """
    N, H, D = q_scalar.shape
    P = q_points.shape[2]
    scale = math.sqrt(D)

    # Scalar term: w_L * Q·K / sqrt(d)
    scalar = np.einsum("ihd,jhd->hij", q_scalar, k_scalar) / scale
    scalar *= w_l

    # Pair bias term: w_C * bias
    pair_term = w_c * pair_bias

    # Point distance term
    point_term = np.zeros((H, N, N))
    for i in range(N):
        Ri, ti = frames_rot[i], frames_trans[i]
        for j in range(N):
            Rj, tj = frames_rot[j], frames_trans[j]
            dist_sq = 0.0
            for p in range(P):
                for h in range(H):
                    qp_global = Ri @ q_points[i, h, p] + ti
                    kp_global = Rj @ k_points[j, h, p] + tj
                    diff = qp_global - kp_global
                    dist_sq_hp = float(np.dot(diff, diff))
                    point_term[h, i, j] += (-gamma / 2.0) * dist_sq_hp
    point_term *= w_p

    return scalar + pair_term + point_term


def backbone_update(
    delta_quats: np.ndarray,
    delta_trans: np.ndarray,
    current_rot: np.ndarray,
    current_trans: np.ndarray,
) -> tuple:
    """Update backbone frames by composing with delta transforms.

    delta_quats:    [N, 4] — quaternion updates
    delta_trans:    [N, 3]
    current_rot:    [N, 3, 3]
    current_trans:  [N, 3]
    returns:        (new_rot [N,3,3], new_trans [N,3])
    """
    N = delta_quats.shape[0]
    new_rot = np.zeros((N, 3, 3))
    new_trans = np.zeros((N, 3))
    for i in range(N):
        dr = quat_to_rotation(delta_quats[i])
        new_rot[i] = current_rot[i] @ dr
        new_trans[i] = current_rot[i] @ delta_trans[i] + current_trans[i]
    return new_rot, new_trans


def torsion_angles(
    single: np.ndarray,
    weights: list,
) -> np.ndarray:
    """Predict torsion angles via ResNet + unit circle normalization.

    Architecture: Linear → ResNet → ResNet → Linear → normalize.

    single:  [N, C_s]
    weights: list of (W, b) tuples for each layer
    returns: [N, 7, 2] (sin, cos for 7 angles, unit-normalized)
    """
    proj_in_w, proj_in_b = weights[0]
    r1_w1, r1_b1 = weights[1]
    r1_w2, r1_b2 = weights[2]
    r2_w1, r2_b1 = weights[3]
    r2_w2, r2_b2 = weights[4]
    proj_out_w, proj_out_b = weights[5]

    h = single @ proj_in_w + proj_in_b

    # ResNet block 1
    h_skip = h.copy()
    h = np.maximum(0, h @ r1_w1 + r1_b1)
    h = h @ r1_w2 + r1_b2 + h_skip

    # ResNet block 2
    h_skip = h.copy()
    h = np.maximum(0, h @ r2_w1 + r2_b1)
    h = h @ r2_w2 + r2_b2 + h_skip

    raw = h @ proj_out_w + proj_out_b
    raw = raw.reshape(-1, 7, 2)

    # Normalize to unit circle
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return raw / norms


# ═══════════════════════════════════════════════════════════════════
# Baseline generation
# ═══════════════════════════════════════════════════════════════════

def generate_baselines() -> dict:
    rng = np.random.default_rng(SEED)
    baselines = {}

    # -- GELU --
    x_gelu = rng.standard_normal(16).astype(np.float64)
    baselines["gelu_input"] = x_gelu.tolist()
    baselines["gelu_output"] = gelu(x_gelu).tolist()

    # -- Layer Norm --
    x_ln = rng.standard_normal((N_RES, HIDDEN_DIM)).astype(np.float64)
    gamma = rng.uniform(0.5, 1.5, HIDDEN_DIM).astype(np.float64)
    beta = rng.uniform(-0.1, 0.1, HIDDEN_DIM).astype(np.float64)
    baselines["layer_norm_input"] = x_ln.tolist()
    baselines["layer_norm_gamma"] = gamma.tolist()
    baselines["layer_norm_beta"] = beta.tolist()
    baselines["layer_norm_output"] = layer_norm(x_ln, gamma, beta).tolist()

    # -- Softmax --
    x_sm = rng.standard_normal((4, 8)).astype(np.float64)
    baselines["softmax_input"] = x_sm.tolist()
    baselines["softmax_output"] = softmax(x_sm).tolist()

    # -- SDPA --
    q = rng.standard_normal((BATCH, N_HEADS, N_RES, HEAD_DIM)).astype(np.float64)
    k = rng.standard_normal((BATCH, N_HEADS, N_RES, HEAD_DIM)).astype(np.float64)
    v = rng.standard_normal((BATCH, N_HEADS, N_RES, HEAD_DIM)).astype(np.float64)
    baselines["sdpa_query"] = q.tolist()
    baselines["sdpa_key"] = k.tolist()
    baselines["sdpa_value"] = v.tolist()
    baselines["sdpa_scores"] = sdpa_scores(q, k).tolist()
    baselines["sdpa_output"] = sdpa_full(q, k, v).tolist()

    # -- Triangle mul outgoing (Algorithm 11) --
    proj_a = rng.standard_normal((N_RES, N_RES, CHANNELS)).astype(np.float64)
    proj_b = rng.standard_normal((N_RES, N_RES, CHANNELS)).astype(np.float64)
    baselines["tri_out_proj_a"] = proj_a.tolist()
    baselines["tri_out_proj_b"] = proj_b.tolist()
    baselines["tri_out_output"] = triangle_mul_outgoing(proj_a, proj_b).tolist()

    # -- Triangle mul incoming (Algorithm 12) --
    proj_a_in = rng.standard_normal((N_RES, N_RES, CHANNELS)).astype(np.float64)
    proj_b_in = rng.standard_normal((N_RES, N_RES, CHANNELS)).astype(np.float64)
    baselines["tri_in_proj_a"] = proj_a_in.tolist()
    baselines["tri_in_proj_b"] = proj_b_in.tolist()
    baselines["tri_in_output"] = triangle_mul_incoming(proj_a_in, proj_b_in).tolist()

    # -- Triangle attention scores (Algorithms 13-14) --
    tri_q = rng.standard_normal((N_RES, N_RES, N_HEADS, HEAD_DIM)).astype(np.float64)
    tri_k = rng.standard_normal((N_RES, N_RES, N_HEADS, HEAD_DIM)).astype(np.float64)
    tri_bias = rng.standard_normal((N_HEADS, N_RES, N_RES)).astype(np.float64)
    baselines["tri_attn_query"] = tri_q.tolist()
    baselines["tri_attn_key"] = tri_k.tolist()
    baselines["tri_attn_bias"] = tri_bias.tolist()
    baselines["tri_attn_scores"] = triangle_attention_scores(
        tri_q, tri_k, tri_bias
    ).tolist()

    # -- Outer product mean --
    N_SEQ = 6
    C_A = 3
    C_B = 2
    opm_a = rng.standard_normal((N_SEQ, N_RES, C_A)).astype(np.float64)
    opm_b = rng.standard_normal((N_SEQ, N_RES, C_B)).astype(np.float64)
    baselines["opm_a"] = opm_a.tolist()
    baselines["opm_b"] = opm_b.tolist()
    baselines["opm_output"] = outer_product_mean(opm_a, opm_b).tolist()
    baselines["opm_n_seq"] = N_SEQ
    baselines["opm_c_a"] = C_A
    baselines["opm_c_b"] = C_B

    # -- MSA row attention --
    msa_q = rng.standard_normal((N_SEQ, N_RES, N_HEADS, HEAD_DIM)).astype(np.float64)
    msa_k = rng.standard_normal((N_SEQ, N_RES, N_HEADS, HEAD_DIM)).astype(np.float64)
    msa_v = rng.standard_normal((N_SEQ, N_RES, N_HEADS, HEAD_DIM)).astype(np.float64)
    msa_bias = rng.standard_normal((N_HEADS, N_RES, N_RES)).astype(np.float64) * 0.1
    baselines["msa_row_query"] = msa_q.tolist()
    baselines["msa_row_key"] = msa_k.tolist()
    baselines["msa_row_value"] = msa_v.tolist()
    baselines["msa_row_pair_bias"] = msa_bias.tolist()
    baselines["msa_row_scores"] = msa_row_attention_scores(
        msa_q, msa_k, msa_bias
    ).tolist()
    baselines["msa_row_output"] = msa_row_attention(
        msa_q, msa_k, msa_v, msa_bias
    ).tolist()
    baselines["msa_n_seq"] = N_SEQ

    # -- MSA column attention --
    msa_col_q = rng.standard_normal((N_SEQ, N_RES, N_HEADS, HEAD_DIM)).astype(np.float64)
    msa_col_k = rng.standard_normal((N_SEQ, N_RES, N_HEADS, HEAD_DIM)).astype(np.float64)
    msa_col_v = rng.standard_normal((N_SEQ, N_RES, N_HEADS, HEAD_DIM)).astype(np.float64)
    baselines["msa_col_query"] = msa_col_q.tolist()
    baselines["msa_col_key"] = msa_col_k.tolist()
    baselines["msa_col_value"] = msa_col_v.tolist()
    baselines["msa_col_scores"] = msa_col_attention_scores(
        msa_col_q, msa_col_k
    ).tolist()
    baselines["msa_col_output"] = msa_col_attention(
        msa_col_q, msa_col_k, msa_col_v
    ).tolist()

    # -- Structure Module: frame operations --
    IPA_N_POINTS = 3
    IPA_N_RES = 4
    IPA_N_HEADS = 2
    IPA_HEAD_DIM = 4

    # Random frames (rotation from random quaternion + random translation)
    raw_quats = rng.standard_normal((IPA_N_RES, 4)).astype(np.float64)
    frames_rot = np.zeros((IPA_N_RES, 3, 3))
    frames_trans = rng.standard_normal((IPA_N_RES, 3)).astype(np.float64)
    for i in range(IPA_N_RES):
        frames_rot[i] = quat_to_rotation(raw_quats[i])

    baselines["ipa_frames_rot"] = frames_rot.tolist()
    baselines["ipa_frames_trans"] = frames_trans.tolist()
    baselines["ipa_n_res"] = IPA_N_RES
    baselines["ipa_n_heads"] = IPA_N_HEADS
    baselines["ipa_head_dim"] = IPA_HEAD_DIM
    baselines["ipa_n_points"] = IPA_N_POINTS

    # IPA inputs
    ipa_q = rng.standard_normal((IPA_N_RES, IPA_N_HEADS, IPA_HEAD_DIM)).astype(np.float64)
    ipa_k = rng.standard_normal((IPA_N_RES, IPA_N_HEADS, IPA_HEAD_DIM)).astype(np.float64)
    ipa_bias = rng.standard_normal((IPA_N_HEADS, IPA_N_RES, IPA_N_RES)).astype(np.float64) * 0.1
    ipa_qp = rng.standard_normal((IPA_N_RES, IPA_N_HEADS, IPA_N_POINTS, 3)).astype(np.float64)
    ipa_kp = rng.standard_normal((IPA_N_RES, IPA_N_HEADS, IPA_N_POINTS, 3)).astype(np.float64)

    baselines["ipa_q_scalar"] = ipa_q.tolist()
    baselines["ipa_k_scalar"] = ipa_k.tolist()
    baselines["ipa_pair_bias"] = ipa_bias.tolist()
    baselines["ipa_q_points"] = ipa_qp.tolist()
    baselines["ipa_k_points"] = ipa_kp.tolist()

    w_l, w_c, w_p, gamma_val = 1.0, 1.0, 1.0, 0.5
    baselines["ipa_w_l"] = w_l
    baselines["ipa_w_c"] = w_c
    baselines["ipa_w_p"] = w_p
    baselines["ipa_gamma"] = gamma_val

    ipa_scores_out = ipa_scores(
        ipa_q, ipa_k, ipa_bias, ipa_qp, ipa_kp,
        frames_rot, frames_trans, w_l, w_c, w_p, gamma_val,
    )
    baselines["ipa_scores"] = ipa_scores_out.tolist()

    # Backbone update
    delta_quats = rng.standard_normal((IPA_N_RES, 4)).astype(np.float64)
    delta_quats[:, 0] += 1.0  # bias toward identity
    delta_trans_vals = rng.standard_normal((IPA_N_RES, 3)).astype(np.float64) * 0.1
    baselines["backbone_delta_quats"] = delta_quats.tolist()
    baselines["backbone_delta_trans"] = delta_trans_vals.tolist()

    new_rot, new_trans = backbone_update(
        delta_quats, delta_trans_vals, frames_rot, frames_trans,
    )
    baselines["backbone_new_rot"] = new_rot.tolist()
    baselines["backbone_new_trans"] = new_trans.tolist()

    # -- Torsion angle prediction --
    TORSION_N = 4
    TORSION_C_S = 8
    TORSION_C_H = 6
    torsion_single = rng.standard_normal((TORSION_N, TORSION_C_S)).astype(np.float64)
    baselines["torsion_single"] = torsion_single.tolist()
    baselines["torsion_n_res"] = TORSION_N
    baselines["torsion_c_single"] = TORSION_C_S
    baselines["torsion_c_hidden"] = TORSION_C_H

    torsion_weights = []
    torsion_weights_flat = []

    def make_wb(in_d, out_d):
        w = rng.standard_normal((in_d, out_d)).astype(np.float64) * 0.1
        b = rng.standard_normal(out_d).astype(np.float64) * 0.01
        torsion_weights_flat.extend(w.flatten().tolist())
        torsion_weights_flat.extend(b.tolist())
        return (w, b)

    torsion_weights.append(make_wb(TORSION_C_S, TORSION_C_H))  # proj_in
    for _ in range(4):  # 2 ResNet blocks × 2 layers each
        torsion_weights.append(make_wb(TORSION_C_H, TORSION_C_H))
    torsion_weights.append(make_wb(TORSION_C_H, 14))  # proj_out

    baselines["torsion_weights"] = torsion_weights_flat

    torsion_out = torsion_angles(torsion_single, torsion_weights)
    baselines["torsion_output"] = torsion_out.tolist()

    # -- Metadata --
    baselines["seed"] = SEED
    baselines["n_res"] = N_RES
    baselines["channels"] = CHANNELS
    baselines["n_heads"] = N_HEADS
    baselines["head_dim"] = HEAD_DIM
    baselines["hidden_dim"] = HIDDEN_DIM

    return baselines


def main():
    baselines = generate_baselines()
    out_path = Path(__file__).parent / "evoformer_baselines.json"
    with open(out_path, "w") as f:
        json.dump(baselines, f, indent=2)

    n_checks = 0
    checks = []

    # GELU
    checks.append(("GELU output length", len(baselines["gelu_output"]) == 16))
    n_checks += 1

    # Layer norm: output shape matches input
    ln_out = np.array(baselines["layer_norm_output"])
    checks.append(("LayerNorm shape", ln_out.shape == (N_RES, HIDDEN_DIM)))
    # Row means should be close to beta (after normalization)
    checks.append(("LayerNorm finite", np.all(np.isfinite(ln_out))))
    n_checks += 2

    # Softmax: rows sum to 1
    sm_out = np.array(baselines["softmax_output"])
    row_sums = sm_out.sum(axis=-1)
    checks.append(("Softmax rows sum to 1",
                    np.allclose(row_sums, 1.0, atol=1e-12)))
    n_checks += 1

    # SDPA: output finite
    sdpa_out = np.array(baselines["sdpa_output"])
    checks.append(("SDPA output finite", np.all(np.isfinite(sdpa_out))))
    n_checks += 1

    # Triangle mul outgoing: check dimensions
    tri_out = np.array(baselines["tri_out_output"])
    checks.append(("TriMul outgoing shape",
                    tri_out.shape == (N_RES, N_RES, CHANNELS)))
    checks.append(("TriMul outgoing finite", np.all(np.isfinite(tri_out))))
    n_checks += 2

    # Triangle mul incoming
    tri_in = np.array(baselines["tri_in_output"])
    checks.append(("TriMul incoming shape",
                    tri_in.shape == (N_RES, N_RES, CHANNELS)))
    checks.append(("TriMul incoming finite", np.all(np.isfinite(tri_in))))
    n_checks += 2

    # Triangle attention
    tri_attn = np.array(baselines["tri_attn_scores"])
    checks.append(("TriAttn scores shape",
                    tri_attn.shape == (N_RES, N_HEADS, N_RES, N_RES)))
    checks.append(("TriAttn scores finite", np.all(np.isfinite(tri_attn))))
    n_checks += 2

    # Outer product mean
    opm = np.array(baselines["opm_output"])
    N_SEQ = baselines["opm_n_seq"]
    C_A = baselines["opm_c_a"]
    C_B = baselines["opm_c_b"]
    checks.append(("OPM shape", opm.shape == (N_RES, N_RES, C_A * C_B)))
    checks.append(("OPM finite", np.all(np.isfinite(opm))))
    n_checks += 2

    # MSA row attention
    msa_row_out = np.array(baselines["msa_row_output"])
    msa_n_seq = baselines["msa_n_seq"]
    checks.append(("MSA row attn shape",
                    msa_row_out.shape == (msa_n_seq, N_RES, N_HEADS, HEAD_DIM)))
    checks.append(("MSA row attn finite", np.all(np.isfinite(msa_row_out))))
    n_checks += 2

    # MSA column attention
    msa_col_out = np.array(baselines["msa_col_output"])
    checks.append(("MSA col attn shape",
                    msa_col_out.shape == (msa_n_seq, N_RES, N_HEADS, HEAD_DIM)))
    checks.append(("MSA col attn finite", np.all(np.isfinite(msa_col_out))))
    n_checks += 2

    # Structure Module: IPA scores
    ipa_s = np.array(baselines["ipa_scores"])
    IPA_N = baselines["ipa_n_res"]
    IPA_H = baselines["ipa_n_heads"]
    checks.append(("IPA scores shape", ipa_s.shape == (IPA_H, IPA_N, IPA_N)))
    checks.append(("IPA scores finite", np.all(np.isfinite(ipa_s))))
    n_checks += 2

    # Backbone update
    bb_rot = np.array(baselines["backbone_new_rot"])
    bb_trans = np.array(baselines["backbone_new_trans"])
    checks.append(("Backbone rot shape", bb_rot.shape == (IPA_N, 3, 3)))
    checks.append(("Backbone trans shape", bb_trans.shape == (IPA_N, 3)))
    # Rotation matrices should be approximately orthogonal
    ortho_ok = True
    for i in range(IPA_N):
        rrt = bb_rot[i] @ bb_rot[i].T
        if not np.allclose(rrt, np.eye(3), atol=1e-10):
            ortho_ok = False
    checks.append(("Backbone rot orthogonal", ortho_ok))
    n_checks += 3

    # Torsion angle prediction
    torsion_out = np.array(baselines["torsion_output"])
    T_N = baselines["torsion_n_res"]
    checks.append(("Torsion shape", torsion_out.shape == (T_N, 7, 2)))
    checks.append(("Torsion finite", np.all(np.isfinite(torsion_out))))
    unit_norms = np.linalg.norm(torsion_out, axis=-1)
    checks.append(("Torsion unit circle", np.allclose(unit_norms, 1.0, atol=1e-10)))
    n_checks += 3

    # -- Report --
    passed = sum(1 for _, ok in checks if ok)
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
    print(f"\n=== evoformer_primitives: {passed}/{n_checks} PASS ===")

    print(f"\nBaselines written to {out_path}")
    print(f"  GELU: {len(baselines['gelu_output'])} values")
    print(f"  LayerNorm: {N_RES}x{HIDDEN_DIM}")
    print(f"  Softmax: 4x8")
    print(f"  SDPA: B={BATCH}, H={N_HEADS}, N={N_RES}, D={HEAD_DIM}")
    print(f"  TriMul out/in: {N_RES}x{N_RES}x{CHANNELS}")
    print(f"  TriAttn: R={N_RES}, H={N_HEADS}, N={N_RES}, D={HEAD_DIM}")
    print(f"  OPM: S={baselines['opm_n_seq']}, N={N_RES}, Ca={baselines['opm_c_a']}, Cb={baselines['opm_c_b']}")
    print(f"  MSA row: S={baselines['msa_n_seq']}, N={N_RES}, H={N_HEADS}, D={HEAD_DIM}")
    print(f"  MSA col: S={baselines['msa_n_seq']}, N={N_RES}, H={N_HEADS}, D={HEAD_DIM}")
    print(f"  IPA: N={baselines['ipa_n_res']}, H={baselines['ipa_n_heads']}, D={baselines['ipa_head_dim']}, P={baselines['ipa_n_points']}")
    print(f"  Backbone: N={baselines['ipa_n_res']}")
    print(f"  Torsion: N={baselines['torsion_n_res']}, C_s={baselines['torsion_c_single']}, C_h={baselines['torsion_c_hidden']}")

    sys.exit(0 if passed == n_checks else 1)


if __name__ == "__main__":
    main()
