# SPDX-License-Identifier: AGPL-3.0-or-later
# Provenance: see src/provenance/experiments.rs — ALPHAFOLD2_EVOFORMER_PROVENANCE
#
# alphafold2_evoformer_block.py — Full Evoformer block iteration (Algorithm 6)
#
# nF-02: Jumper et al. "Highly accurate protein structure prediction with
# AlphaFold" Nature 596:583-589 (2021)
#
# Validates the COMPLETE Evoformer block pipeline: all primitives composed
# into one iteration of the Evoformer stack, plus Structure Module IPA +
# backbone update + torsion angle prediction.
#
# Pipeline (one block):
#   1. MSA row attention with pair bias      (Algorithm 7)
#   2. MSA column attention                  (Algorithm 8)
#   3. Outer product mean update             (Algorithm 10)
#   4. Triangle multiplicative outgoing      (Algorithm 11)
#   5. Triangle multiplicative incoming      (Algorithm 12)
#   6. Triangle attention starting/ending    (Algorithms 13-14)
#   7. IPA scores → softmax → weighted sum   (Algorithm 22)
#   8. Backbone frame update
#   9. Torsion angle prediction
#
# Usage: python3 control/coral_forge/alphafold2_evoformer_block.py
# Output: control/coral_forge/evoformer_block_baselines.json

import json
import math
import sys
from pathlib import Path

import numpy as np

from evoformer_primitives import (
    gelu,
    layer_norm,
    softmax,
    sdpa_scores,
    sdpa_full,
    triangle_mul_outgoing,
    triangle_mul_incoming,
    triangle_attention_scores,
    outer_product_mean,
    msa_row_attention,
    msa_col_attention,
    ipa_scores,
    backbone_update,
    torsion_angles,
    quat_to_rotation,
)

SEED = 42
N_SEQ = 4
N_RES = 6
N_HEADS = 2
HEAD_DIM = 4
CHANNELS = 4
C_MSA = N_HEADS * HEAD_DIM  # MSA representation dim
C_PAIR = CHANNELS            # pair representation channels
IPA_N_POINTS = 2


def project(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Linear projection (no bias) for attention heads."""
    return x @ w


def evoformer_block(
    msa: np.ndarray,
    pair: np.ndarray,
    rng: np.random.Generator,
) -> tuple:
    """One full Evoformer block (simplified Algorithm 6).

    msa:  [N_seq, N_res, C_msa]
    pair: [N_res, N_res, C_pair]

    Returns updated (msa, pair).
    """
    s, n, c_m = msa.shape
    c_p = pair.shape[2]
    h = N_HEADS
    d = HEAD_DIM

    # ── Step 1: MSA row attention with pair bias ──────────────────
    # Project Q/K/V for row attention: [N_seq, N_res, H, D]
    w_q = rng.standard_normal((c_m, h * d)).astype(np.float64) * 0.1
    w_k = rng.standard_normal((c_m, h * d)).astype(np.float64) * 0.1
    w_v = rng.standard_normal((c_m, h * d)).astype(np.float64) * 0.1

    q_row = (msa @ w_q).reshape(s, n, h, d)
    k_row = (msa @ w_k).reshape(s, n, h, d)
    v_row = (msa @ w_v).reshape(s, n, h, d)

    # Pair bias: project pair channels to H heads
    w_bias = rng.standard_normal((c_p, h)).astype(np.float64) * 0.1
    pair_bias = np.einsum("ijc,ch->hij", pair, w_bias)

    msa_row_out = msa_row_attention(q_row, k_row, v_row, pair_bias)
    # Reshape back and project output
    w_o_row = rng.standard_normal((h * d, c_m)).astype(np.float64) * 0.1
    msa = msa + msa_row_out.reshape(s, n, h * d) @ w_o_row

    # ── Step 2: MSA column attention ─────────────────────────────
    w_q_col = rng.standard_normal((c_m, h * d)).astype(np.float64) * 0.1
    w_k_col = rng.standard_normal((c_m, h * d)).astype(np.float64) * 0.1
    w_v_col = rng.standard_normal((c_m, h * d)).astype(np.float64) * 0.1

    q_col = (msa @ w_q_col).reshape(s, n, h, d)
    k_col = (msa @ w_k_col).reshape(s, n, h, d)
    v_col = (msa @ w_v_col).reshape(s, n, h, d)

    msa_col_out = msa_col_attention(q_col, k_col, v_col)
    w_o_col = rng.standard_normal((h * d, c_m)).astype(np.float64) * 0.1
    msa = msa + msa_col_out.reshape(s, n, h * d) @ w_o_col

    # ── Step 3: Outer product mean ───────────────────────────────
    c_opm_a, c_opm_b = 2, 2
    w_opm_a = rng.standard_normal((c_m, c_opm_a)).astype(np.float64) * 0.1
    w_opm_b = rng.standard_normal((c_m, c_opm_b)).astype(np.float64) * 0.1
    opm_a = msa @ w_opm_a   # [S, N, c_opm_a]
    opm_b = msa @ w_opm_b   # [S, N, c_opm_b]
    opm_out = outer_product_mean(opm_a, opm_b)  # [N, N, c_opm_a*c_opm_b]

    # Project OPM output back to pair channels
    w_opm_proj = rng.standard_normal((c_opm_a * c_opm_b, c_p)).astype(np.float64) * 0.1
    pair = pair + opm_out @ w_opm_proj

    # ── Step 4: Triangle multiplicative outgoing (Algorithm 11) ──
    w_tri_a_out = rng.standard_normal((c_p, c_p)).astype(np.float64) * 0.1
    w_tri_b_out = rng.standard_normal((c_p, c_p)).astype(np.float64) * 0.1
    proj_a_out = pair @ w_tri_a_out  # [N, N, C]
    proj_b_out = pair @ w_tri_b_out
    tri_out = triangle_mul_outgoing(proj_a_out, proj_b_out)
    w_tri_proj_out = rng.standard_normal((c_p, c_p)).astype(np.float64) * 0.1
    pair = pair + tri_out @ w_tri_proj_out

    # ── Step 5: Triangle multiplicative incoming (Algorithm 12) ──
    w_tri_a_in = rng.standard_normal((c_p, c_p)).astype(np.float64) * 0.1
    w_tri_b_in = rng.standard_normal((c_p, c_p)).astype(np.float64) * 0.1
    proj_a_in = pair @ w_tri_a_in
    proj_b_in = pair @ w_tri_b_in
    tri_in = triangle_mul_incoming(proj_a_in, proj_b_in)
    w_tri_proj_in = rng.standard_normal((c_p, c_p)).astype(np.float64) * 0.1
    pair = pair + tri_in @ w_tri_proj_in

    # ── Step 6: Triangle attention (Algorithms 13-14) ────────────
    w_tq = rng.standard_normal((c_p, h * d)).astype(np.float64) * 0.1
    w_tk = rng.standard_normal((c_p, h * d)).astype(np.float64) * 0.1
    pair_flat = pair.reshape(n * n, c_p)
    tri_q = (pair_flat @ w_tq).reshape(n, n, h, d)
    tri_k = (pair_flat @ w_tk).reshape(n, n, h, d)
    w_tri_bias = rng.standard_normal((c_p, h)).astype(np.float64) * 0.1
    tri_bias = np.einsum("ijc,ch->hij", pair, w_tri_bias)
    tri_attn_scores = triangle_attention_scores(tri_q, tri_k, tri_bias)

    return msa, pair, tri_attn_scores


def structure_module_step(
    single: np.ndarray,
    pair: np.ndarray,
    frames_rot: np.ndarray,
    frames_trans: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """One Structure Module iteration.

    single: [N_res, C_single]  (first row of MSA)
    pair:   [N_res, N_res, C_pair]
    """
    n = single.shape[0]
    c_s = single.shape[1]
    c_p = pair.shape[2]
    h = N_HEADS
    d = HEAD_DIM
    p = IPA_N_POINTS

    # IPA projections
    w_iq = rng.standard_normal((c_s, h * d)).astype(np.float64) * 0.1
    w_ik = rng.standard_normal((c_s, h * d)).astype(np.float64) * 0.1
    w_iv = rng.standard_normal((c_s, h * d)).astype(np.float64) * 0.1

    q_scalar = (single @ w_iq).reshape(n, h, d)
    k_scalar = (single @ w_ik).reshape(n, h, d)
    v_scalar = (single @ w_iv).reshape(n, h, d)

    # Pair bias for IPA
    w_ipa_bias = rng.standard_normal((c_p, h)).astype(np.float64) * 0.1
    pair_bias = np.einsum("ijc,ch->hij", pair, w_ipa_bias)

    # Point projections
    w_qp = rng.standard_normal((c_s, h * p * 3)).astype(np.float64) * 0.1
    w_kp = rng.standard_normal((c_s, h * p * 3)).astype(np.float64) * 0.1
    q_points = (single @ w_qp).reshape(n, h, p, 3)
    k_points = (single @ w_kp).reshape(n, h, p, 3)

    w_l, w_c, w_p, gamma = 1.0, 1.0, 1.0, 0.5

    ipa_s = ipa_scores(
        q_scalar, k_scalar, pair_bias, q_points, k_points,
        frames_rot, frames_trans, w_l, w_c, w_p, gamma,
    )

    # Backbone update
    delta_quats = rng.standard_normal((n, 4)).astype(np.float64) * 0.1
    delta_quats[:, 0] += 1.0
    delta_trans = rng.standard_normal((n, 3)).astype(np.float64) * 0.1
    new_rot, new_trans = backbone_update(
        delta_quats, delta_trans, frames_rot, frames_trans,
    )

    # Torsion angle prediction
    c_hidden = 6
    torsion_weights = []
    torsion_flat = []

    def make_wb(in_d, out_d):
        w = rng.standard_normal((in_d, out_d)).astype(np.float64) * 0.1
        b = rng.standard_normal(out_d).astype(np.float64) * 0.01
        torsion_flat.extend(w.flatten().tolist())
        torsion_flat.extend(b.tolist())
        return (w, b)

    torsion_weights.append(make_wb(c_s, c_hidden))
    for _ in range(4):
        torsion_weights.append(make_wb(c_hidden, c_hidden))
    torsion_weights.append(make_wb(c_hidden, 14))

    torsion_out = torsion_angles(single, torsion_weights)

    return {
        "ipa_scores": ipa_s,
        "new_rot": new_rot,
        "new_trans": new_trans,
        "torsion_output": torsion_out,
        "torsion_weights": torsion_flat,
        "delta_quats": delta_quats,
        "delta_trans": delta_trans,
        "q_scalar": q_scalar,
        "k_scalar": k_scalar,
        "pair_bias": pair_bias,
        "q_points": q_points,
        "k_points": k_points,
    }


def generate_baselines() -> dict:
    rng = np.random.default_rng(SEED)
    baselines = {}

    # Initial MSA and pair representations
    msa = rng.standard_normal((N_SEQ, N_RES, C_MSA)).astype(np.float64)
    pair = rng.standard_normal((N_RES, N_RES, C_PAIR)).astype(np.float64)

    baselines["msa_input"] = msa.tolist()
    baselines["pair_input"] = pair.tolist()
    baselines["n_seq"] = N_SEQ
    baselines["n_res"] = N_RES
    baselines["n_heads"] = N_HEADS
    baselines["head_dim"] = HEAD_DIM
    baselines["channels"] = C_PAIR
    baselines["c_msa"] = C_MSA
    baselines["seed"] = SEED

    # Run Evoformer block
    msa_out, pair_out, tri_attn_scores_out = evoformer_block(msa, pair, rng)

    baselines["msa_output"] = msa_out.tolist()
    baselines["pair_output"] = pair_out.tolist()
    baselines["tri_attn_scores"] = tri_attn_scores_out.tolist()

    # Structure Module with first-row single representation
    single = msa_out[0]  # [N_res, C_msa]

    # Initialize identity backbone frames
    frames_rot = np.zeros((N_RES, 3, 3))
    for i in range(N_RES):
        frames_rot[i] = np.eye(3)
    frames_trans = np.zeros((N_RES, 3))

    baselines["single_repr"] = single.tolist()
    baselines["init_frames_rot"] = frames_rot.tolist()
    baselines["init_frames_trans"] = frames_trans.tolist()

    sm_result = structure_module_step(
        single, pair_out, frames_rot, frames_trans, rng,
    )

    baselines["sm_ipa_scores"] = sm_result["ipa_scores"].tolist()
    baselines["sm_new_rot"] = sm_result["new_rot"].tolist()
    baselines["sm_new_trans"] = sm_result["new_trans"].tolist()
    baselines["sm_torsion_output"] = sm_result["torsion_output"].tolist()
    baselines["sm_torsion_weights"] = sm_result["torsion_weights"]
    baselines["sm_delta_quats"] = sm_result["delta_quats"].tolist()
    baselines["sm_delta_trans"] = sm_result["delta_trans"].tolist()
    baselines["sm_q_scalar"] = sm_result["q_scalar"].tolist()
    baselines["sm_k_scalar"] = sm_result["k_scalar"].tolist()
    baselines["sm_pair_bias"] = sm_result["pair_bias"].tolist()
    baselines["sm_q_points"] = sm_result["q_points"].tolist()
    baselines["sm_k_points"] = sm_result["k_points"].tolist()
    baselines["sm_ipa_n_points"] = IPA_N_POINTS

    return baselines


def main():
    baselines = generate_baselines()
    out_path = Path(__file__).parent / "evoformer_block_baselines.json"
    with open(out_path, "w") as f:
        json.dump(baselines, f, indent=2)

    checks = []

    # Evoformer block outputs
    msa_out = np.array(baselines["msa_output"])
    pair_out = np.array(baselines["pair_output"])
    tri_s = np.array(baselines["tri_attn_scores"])

    checks.append(("Evoformer MSA shape",
                    msa_out.shape == (N_SEQ, N_RES, C_MSA)))
    checks.append(("Evoformer MSA finite", np.all(np.isfinite(msa_out))))
    checks.append(("Evoformer pair shape",
                    pair_out.shape == (N_RES, N_RES, C_PAIR)))
    checks.append(("Evoformer pair finite", np.all(np.isfinite(pair_out))))
    checks.append(("Evoformer MSA changed",
                    not np.allclose(msa_out, np.array(baselines["msa_input"]))))
    checks.append(("Evoformer pair changed",
                    not np.allclose(pair_out, np.array(baselines["pair_input"]))))
    checks.append(("TriAttn scores finite", np.all(np.isfinite(tri_s))))
    checks.append(("TriAttn scores shape",
                    tri_s.shape == (N_RES, N_HEADS, N_RES, N_RES)))

    # Structure Module outputs
    ipa_s = np.array(baselines["sm_ipa_scores"])
    checks.append(("IPA scores finite", np.all(np.isfinite(ipa_s))))
    checks.append(("IPA scores shape", ipa_s.shape == (N_HEADS, N_RES, N_RES)))

    new_rot = np.array(baselines["sm_new_rot"])
    new_trans = np.array(baselines["sm_new_trans"])
    checks.append(("Backbone rot shape", new_rot.shape == (N_RES, 3, 3)))
    checks.append(("Backbone trans shape", new_trans.shape == (N_RES, 3)))
    ortho = all(
        np.allclose(new_rot[i] @ new_rot[i].T, np.eye(3), atol=1e-10)
        for i in range(N_RES)
    )
    checks.append(("Backbone rot orthogonal", ortho))

    torsion = np.array(baselines["sm_torsion_output"])
    checks.append(("Torsion shape", torsion.shape == (N_RES, 7, 2)))
    checks.append(("Torsion finite", np.all(np.isfinite(torsion))))
    norms = np.linalg.norm(torsion, axis=-1)
    checks.append(("Torsion unit circle", np.allclose(norms, 1.0, atol=1e-10)))

    # Pair representation should have non-trivial updates
    pair_input = np.array(baselines["pair_input"])
    pair_delta = np.abs(pair_out - pair_input).max()
    checks.append(("Pair delta > 0.1", pair_delta > 0.1))

    # MSA representation should have non-trivial updates
    msa_input = np.array(baselines["msa_input"])
    msa_delta = np.abs(msa_out - msa_input).max()
    checks.append(("MSA delta > 0.01", msa_delta > 0.01))

    # IPA diagonal should differ from off-diagonal (self vs. neighbor)
    ipa_diag = np.array([ipa_s[h, i, i] for h in range(N_HEADS) for i in range(N_RES)])
    ipa_off = np.array([ipa_s[h, i, j] for h in range(N_HEADS)
                        for i in range(N_RES) for j in range(N_RES) if i != j])
    checks.append(("IPA diag != off-diag mean",
                    abs(ipa_diag.mean() - ipa_off.mean()) > 1e-6))

    # Report
    passed = sum(1 for _, ok in checks if ok)
    n_checks = len(checks)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n=== alphafold2_evoformer_block (nF-02): {passed}/{n_checks} PASS ===")

    print(f"\nBaselines written to {out_path}")
    sys.exit(0 if passed == n_checks else 1)


if __name__ == "__main__":
    main()
