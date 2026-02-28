// SPDX-License-Identifier: AGPL-3.0-or-later

//! nF-01 Phase B: coralForge Evoformer primitive validation.
//!
//! Loads Python-generated baselines from `evoformer_baselines.json` and
//! validates that Rust CPU implementations reproduce them within cross-language
//! tolerance (1e-10 for f64→f64, tighter for exact arithmetic).
//!
//! ## Provenance
//!
//! Python baseline: `control/coral_forge/evoformer_primitives.py`
//! Reference: Jumper et al. Nature 596:583-589 (2021), Algorithms 11-14
//!
//! ## Experiments
//!
//! | Check | Primitive | What it validates |
//! |-------|-----------|-------------------|
//! | nF-B01 | GELU | Pointwise activation parity |
//! | nF-B02 | LayerNorm | Row normalization parity |
//! | nF-B03 | Softmax | Row-wise probability parity |
//! | nF-B04 | SDPA scores | QKᵀ/√d dot-product parity |
//! | nF-B05 | SDPA full | End-to-end attention parity |
//! | nF-B06 | TriMul outgoing | Algorithm 11 contraction |
//! | nF-B07 | TriMul incoming | Algorithm 12 contraction |
//! | nF-B08 | TriAttn scores | Algorithms 13-14 biased attention |
//! | nF-B09 | OPM | Outer product mean (MSA → pair) |
//! | nF-B10 | MSA row attn | Row attention with pair bias |
//! | nF-B11 | MSA col attn | Column attention across sequences |
//! | nF-B12 | IPA scores | Invariant Point Attention (SE(3)-equivariant) |
//! | nF-B13 | Backbone update | Frame composition (quat → rotation) |
//! | nF-B14 | Torsion angles | ResNet → unit circle (sin, cos) |

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines
)]

use neural_spring::coral_forge;
use neural_spring::coral_forge::structure;
use neural_spring::coral_forge::structure::IpaConfig;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

const BASELINE_JSON: &str = include_str!("../../control/coral_forge/evoformer_baselines.json");

fn flat_f64(val: &serde_json::Value) -> Vec<f64> {
    match val {
        serde_json::Value::Array(arr) => arr.iter().flat_map(flat_f64).collect(),
        serde_json::Value::Number(n) => vec![n.as_f64().unwrap_or(0.0)],
        _ => vec![],
    }
}

fn main() {
    let mut h = ValidationHarness::new("coral_forge");

    let Ok(baselines) = serde_json::from_str::<serde_json::Value>(BASELINE_JSON) else {
        h.check_bool("JSON parse", false);
        h.finish();
    };

    let n_res = baselines["n_res"].as_u64().unwrap_or(8) as usize;
    let channels = baselines["channels"].as_u64().unwrap_or(4) as usize;
    let n_heads = baselines["n_heads"].as_u64().unwrap_or(2) as usize;
    let head_dim = baselines["head_dim"].as_u64().unwrap_or(4) as usize;
    let hidden_dim = baselines["hidden_dim"].as_u64().unwrap_or(16) as usize;

    h.check_bool("metadata loaded", n_res > 0 && channels > 0);

    // ── nF-B01: GELU ────────────────────────────────────────────
    let gelu_in = flat_f64(&baselines["gelu_input"]);
    let gelu_expected = flat_f64(&baselines["gelu_output"]);
    let gelu_rust = coral_forge::gelu_vec(&gelu_in);

    h.check_bool("GELU length match", gelu_rust.len() == gelu_expected.len());
    for (i, (r, e)) in gelu_rust.iter().zip(gelu_expected.iter()).enumerate() {
        h.check_abs(&format!("GELU[{i}]"), *r, *e, tolerances::CROSS_LANGUAGE);
    }

    // ── nF-B02: Layer Norm ──────────────────────────────────────
    let ln_in = flat_f64(&baselines["layer_norm_input"]);
    let gamma = flat_f64(&baselines["layer_norm_gamma"]);
    let beta = flat_f64(&baselines["layer_norm_beta"]);
    let ln_expected = flat_f64(&baselines["layer_norm_output"]);
    let ln_rust = coral_forge::layer_norm(&ln_in, n_res, hidden_dim, &gamma, &beta, 1e-5);

    h.check_bool("LayerNorm length match", ln_rust.len() == ln_expected.len());
    let mut ln_max_diff = 0.0_f64;
    for (r, e) in ln_rust.iter().zip(ln_expected.iter()) {
        ln_max_diff = ln_max_diff.max((r - e).abs());
    }
    h.check_abs(
        "LayerNorm max diff",
        ln_max_diff,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    // ── nF-B03: Softmax ─────────────────────────────────────────
    let sm_in = flat_f64(&baselines["softmax_input"]);
    let sm_expected = flat_f64(&baselines["softmax_output"]);
    let sm_rows = 4;
    let sm_cols = 8;
    let sm_rust = coral_forge::softmax_rows(&sm_in, sm_rows, sm_cols);

    h.check_bool("Softmax length match", sm_rust.len() == sm_expected.len());
    let mut sm_max_diff = 0.0_f64;
    for (r, e) in sm_rust.iter().zip(sm_expected.iter()) {
        sm_max_diff = sm_max_diff.max((r - e).abs());
    }
    h.check_abs(
        "Softmax max diff",
        sm_max_diff,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    for row in 0..sm_rows {
        let sum: f64 = sm_rust[row * sm_cols..(row + 1) * sm_cols].iter().sum();
        h.check_abs(
            &format!("Softmax row {row} sum"),
            sum,
            1.0,
            tolerances::EXACT_F64,
        );
    }

    // ── nF-B04: SDPA scores ─────────────────────────────────────
    let sdpa_q = flat_f64(&baselines["sdpa_query"]);
    let sdpa_k = flat_f64(&baselines["sdpa_key"]);
    let scores_expected = flat_f64(&baselines["sdpa_scores"]);
    let scores_rust =
        coral_forge::sdpa_scores(&sdpa_q, &sdpa_k, 1, n_heads, n_res, n_res, head_dim);

    h.check_bool(
        "SDPA scores length",
        scores_rust.len() == scores_expected.len(),
    );
    let mut scores_max_diff = 0.0_f64;
    for (r, e) in scores_rust.iter().zip(scores_expected.iter()) {
        scores_max_diff = scores_max_diff.max((r - e).abs());
    }
    h.check_abs(
        "SDPA scores max diff",
        scores_max_diff,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    // ── nF-B05: SDPA full ───────────────────────────────────────
    let sdpa_v = flat_f64(&baselines["sdpa_value"]);
    let sdpa_expected = flat_f64(&baselines["sdpa_output"]);
    let sdpa_rust = coral_forge::sdpa_full(
        &sdpa_q, &sdpa_k, &sdpa_v, 1, n_heads, n_res, n_res, head_dim,
    );

    h.check_bool("SDPA output length", sdpa_rust.len() == sdpa_expected.len());
    let mut sdpa_max_diff = 0.0_f64;
    for (r, e) in sdpa_rust.iter().zip(sdpa_expected.iter()) {
        sdpa_max_diff = sdpa_max_diff.max((r - e).abs());
    }
    h.check_abs(
        "SDPA output max diff",
        sdpa_max_diff,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    // ── nF-B06: Triangle mul outgoing (Algorithm 11) ────────────
    let tri_out_a = flat_f64(&baselines["tri_out_proj_a"]);
    let tri_out_b = flat_f64(&baselines["tri_out_proj_b"]);
    let tri_out_expected = flat_f64(&baselines["tri_out_output"]);
    let tri_out_rust = coral_forge::triangle_mul_outgoing(&tri_out_a, &tri_out_b, n_res, channels);

    h.check_bool(
        "TriMul outgoing length",
        tri_out_rust.len() == tri_out_expected.len(),
    );
    let mut tri_out_max = 0.0_f64;
    for (r, e) in tri_out_rust.iter().zip(tri_out_expected.iter()) {
        tri_out_max = tri_out_max.max((r - e).abs());
    }
    h.check_abs(
        "TriMul outgoing max diff",
        tri_out_max,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    // Determinism check
    let tri_out_rust2 = coral_forge::triangle_mul_outgoing(&tri_out_a, &tri_out_b, n_res, channels);
    h.check_bool(
        "TriMul outgoing determinism",
        tri_out_rust
            .iter()
            .zip(tri_out_rust2.iter())
            .all(|(a, b)| (a - b).abs() == 0.0),
    );

    // ── nF-B07: Triangle mul incoming (Algorithm 12) ────────────
    let tri_in_a = flat_f64(&baselines["tri_in_proj_a"]);
    let tri_in_b = flat_f64(&baselines["tri_in_proj_b"]);
    let tri_in_expected = flat_f64(&baselines["tri_in_output"]);
    let tri_in_rust = coral_forge::triangle_mul_incoming(&tri_in_a, &tri_in_b, n_res, channels);

    h.check_bool(
        "TriMul incoming length",
        tri_in_rust.len() == tri_in_expected.len(),
    );
    let mut tri_in_max = 0.0_f64;
    for (r, e) in tri_in_rust.iter().zip(tri_in_expected.iter()) {
        tri_in_max = tri_in_max.max((r - e).abs());
    }
    h.check_abs(
        "TriMul incoming max diff",
        tri_in_max,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    // ── nF-B08: Triangle attention scores (Algorithms 13-14) ────
    let tri_attn_q = flat_f64(&baselines["tri_attn_query"]);
    let tri_attn_k = flat_f64(&baselines["tri_attn_key"]);
    let tri_attn_bias = flat_f64(&baselines["tri_attn_bias"]);
    let tri_attn_expected = flat_f64(&baselines["tri_attn_scores"]);
    let tri_attn_rust = coral_forge::triangle_attention_scores(
        &tri_attn_q,
        &tri_attn_k,
        &tri_attn_bias,
        n_res,
        n_res,
        n_heads,
        head_dim,
    );

    h.check_bool(
        "TriAttn scores length",
        tri_attn_rust.len() == tri_attn_expected.len(),
    );
    let mut tri_attn_max = 0.0_f64;
    for (r, e) in tri_attn_rust.iter().zip(tri_attn_expected.iter()) {
        tri_attn_max = tri_attn_max.max((r - e).abs());
    }
    h.check_abs(
        "TriAttn scores max diff",
        tri_attn_max,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    // ── nF-B09: Outer product mean ─────────────────────────────
    let opm_n_seq = baselines["opm_n_seq"].as_u64().unwrap_or(6) as usize;
    let opm_c_a = baselines["opm_c_a"].as_u64().unwrap_or(3) as usize;
    let opm_c_b = baselines["opm_c_b"].as_u64().unwrap_or(2) as usize;
    let opm_a = flat_f64(&baselines["opm_a"]);
    let opm_b = flat_f64(&baselines["opm_b"]);
    let opm_expected = flat_f64(&baselines["opm_output"]);
    let opm_rust =
        coral_forge::outer_product_mean(&opm_a, &opm_b, opm_n_seq, n_res, opm_c_a, opm_c_b);

    h.check_bool("OPM length", opm_rust.len() == opm_expected.len());
    let mut opm_max = 0.0_f64;
    for (r, e) in opm_rust.iter().zip(opm_expected.iter()) {
        opm_max = opm_max.max((r - e).abs());
    }
    h.check_abs("OPM max diff", opm_max, 0.0, tolerances::CROSS_LANGUAGE);
    h.check_bool("OPM finite", opm_rust.iter().all(|v| v.is_finite()));

    // ── nF-B10: MSA row attention ────────────────────────────────
    let msa_n_seq = baselines["msa_n_seq"].as_u64().unwrap_or(6) as usize;
    let msa_row_q = flat_f64(&baselines["msa_row_query"]);
    let msa_row_k = flat_f64(&baselines["msa_row_key"]);
    let msa_row_v = flat_f64(&baselines["msa_row_value"]);
    let msa_row_bias = flat_f64(&baselines["msa_row_pair_bias"]);
    let msa_row_scores_expected = flat_f64(&baselines["msa_row_scores"]);
    let msa_row_out_expected = flat_f64(&baselines["msa_row_output"]);

    let msa_row_scores_rust = coral_forge::msa_row_attention_scores(
        &msa_row_q,
        &msa_row_k,
        &msa_row_bias,
        msa_n_seq,
        n_res,
        n_heads,
        head_dim,
    );
    h.check_bool(
        "MSA row scores length",
        msa_row_scores_rust.len() == msa_row_scores_expected.len(),
    );
    let mut msa_row_scores_max = 0.0_f64;
    for (r, e) in msa_row_scores_rust
        .iter()
        .zip(msa_row_scores_expected.iter())
    {
        msa_row_scores_max = msa_row_scores_max.max((r - e).abs());
    }
    h.check_abs(
        "MSA row scores max diff",
        msa_row_scores_max,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    let msa_row_out_rust = coral_forge::msa_row_attention(
        &msa_row_q,
        &msa_row_k,
        &msa_row_v,
        &msa_row_bias,
        msa_n_seq,
        n_res,
        n_heads,
        head_dim,
    );
    h.check_bool(
        "MSA row output length",
        msa_row_out_rust.len() == msa_row_out_expected.len(),
    );
    let mut msa_row_out_max = 0.0_f64;
    for (r, e) in msa_row_out_rust.iter().zip(msa_row_out_expected.iter()) {
        msa_row_out_max = msa_row_out_max.max((r - e).abs());
    }
    h.check_abs(
        "MSA row output max diff",
        msa_row_out_max,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "MSA row output finite",
        msa_row_out_rust.iter().all(|v| v.is_finite()),
    );

    // ── nF-B11: MSA column attention ─────────────────────────────
    let msa_col_q = flat_f64(&baselines["msa_col_query"]);
    let msa_col_k = flat_f64(&baselines["msa_col_key"]);
    let msa_col_v = flat_f64(&baselines["msa_col_value"]);
    let msa_col_scores_expected = flat_f64(&baselines["msa_col_scores"]);
    let msa_col_out_expected = flat_f64(&baselines["msa_col_output"]);

    let msa_col_scores_rust = coral_forge::msa_col_attention_scores(
        &msa_col_q, &msa_col_k, msa_n_seq, n_res, n_heads, head_dim,
    );
    h.check_bool(
        "MSA col scores length",
        msa_col_scores_rust.len() == msa_col_scores_expected.len(),
    );
    let mut msa_col_scores_max = 0.0_f64;
    for (r, e) in msa_col_scores_rust
        .iter()
        .zip(msa_col_scores_expected.iter())
    {
        msa_col_scores_max = msa_col_scores_max.max((r - e).abs());
    }
    h.check_abs(
        "MSA col scores max diff",
        msa_col_scores_max,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    let msa_col_out_rust = coral_forge::msa_col_attention(
        &msa_col_q, &msa_col_k, &msa_col_v, msa_n_seq, n_res, n_heads, head_dim,
    );
    h.check_bool(
        "MSA col output length",
        msa_col_out_rust.len() == msa_col_out_expected.len(),
    );
    let mut msa_col_out_max = 0.0_f64;
    for (r, e) in msa_col_out_rust.iter().zip(msa_col_out_expected.iter()) {
        msa_col_out_max = msa_col_out_max.max((r - e).abs());
    }
    h.check_abs(
        "MSA col output max diff",
        msa_col_out_max,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "MSA col output finite",
        msa_col_out_rust.iter().all(|v| v.is_finite()),
    );

    // ── nF-B12: IPA scores ──────────────────────────────────────
    let ipa_n = baselines["ipa_n_res"].as_u64().unwrap_or(4) as usize;
    let ipa_h = baselines["ipa_n_heads"].as_u64().unwrap_or(2) as usize;
    let ipa_d = baselines["ipa_head_dim"].as_u64().unwrap_or(4) as usize;
    let ipa_p = baselines["ipa_n_points"].as_u64().unwrap_or(3) as usize;
    let ipa_w_l = baselines["ipa_w_l"].as_f64().unwrap_or(1.0);
    let ipa_w_c = baselines["ipa_w_c"].as_f64().unwrap_or(1.0);
    let ipa_w_p = baselines["ipa_w_p"].as_f64().unwrap_or(1.0);
    let ipa_gamma = baselines["ipa_gamma"].as_f64().unwrap_or(0.5);

    let ipa_q = flat_f64(&baselines["ipa_q_scalar"]);
    let ipa_k = flat_f64(&baselines["ipa_k_scalar"]);
    let ipa_bias = flat_f64(&baselines["ipa_pair_bias"]);
    let ipa_qp = flat_f64(&baselines["ipa_q_points"]);
    let ipa_kp = flat_f64(&baselines["ipa_k_points"]);

    // Build flat frames: rotation (row-major 3x3) + translation (3)
    let ipa_rot = flat_f64(&baselines["ipa_frames_rot"]);
    let ipa_trans = flat_f64(&baselines["ipa_frames_trans"]);
    let mut ipa_frames = Vec::with_capacity(ipa_n * 12);
    for i in 0..ipa_n {
        for r in 0..9 {
            ipa_frames.push(ipa_rot[i * 9 + r]);
        }
        for t in 0..3 {
            ipa_frames.push(ipa_trans[i * 3 + t]);
        }
    }

    let ipa_scores_expected = flat_f64(&baselines["ipa_scores"]);
    let ipa_cfg = IpaConfig {
        n_res: ipa_n,
        n_heads: ipa_h,
        head_dim: ipa_d,
        n_points: ipa_p,
        w_l: ipa_w_l,
        w_c: ipa_w_c,
        w_p: ipa_w_p,
        gamma: ipa_gamma,
    };
    let ipa_scores_rust = structure::ipa_scores(
        &ipa_q,
        &ipa_k,
        &ipa_bias,
        &ipa_qp,
        &ipa_kp,
        &ipa_frames,
        &ipa_cfg,
    );
    h.check_bool(
        "IPA scores length",
        ipa_scores_rust.len() == ipa_scores_expected.len(),
    );
    let mut ipa_scores_max = 0.0_f64;
    for (r, e) in ipa_scores_rust.iter().zip(ipa_scores_expected.iter()) {
        ipa_scores_max = ipa_scores_max.max((r - e).abs());
    }
    h.check_abs(
        "IPA scores max diff",
        ipa_scores_max,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "IPA scores finite",
        ipa_scores_rust.iter().all(|v| v.is_finite()),
    );

    // ── nF-B13: Backbone update ───────────────────────────────
    let bb_dq = flat_f64(&baselines["backbone_delta_quats"]);
    let bb_dt = flat_f64(&baselines["backbone_delta_trans"]);
    let bb_expected_rot = flat_f64(&baselines["backbone_new_rot"]);
    let bb_expected_trans = flat_f64(&baselines["backbone_new_trans"]);

    let bb_updated = structure::backbone_update(&bb_dq, &bb_dt, &ipa_frames, ipa_n);
    h.check_bool("Backbone output length", bb_updated.len() == ipa_n * 12);

    let mut bb_rot_max = 0.0_f64;
    let mut bb_trans_max = 0.0_f64;
    for i in 0..ipa_n {
        for r in 0..9 {
            let diff = (bb_updated[i * 12 + r] - bb_expected_rot[i * 9 + r]).abs();
            bb_rot_max = bb_rot_max.max(diff);
        }
        for t in 0..3 {
            let diff = (bb_updated[i * 12 + 9 + t] - bb_expected_trans[i * 3 + t]).abs();
            bb_trans_max = bb_trans_max.max(diff);
        }
    }
    h.check_abs(
        "Backbone rot max diff",
        bb_rot_max,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "Backbone trans max diff",
        bb_trans_max,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "Backbone output finite",
        bb_updated.iter().all(|v| v.is_finite()),
    );

    // ── nF-B14: Torsion angle prediction ────────────────────────
    let torsion_n = baselines["torsion_n_res"].as_u64().unwrap_or(4) as usize;
    let torsion_cs = baselines["torsion_c_single"].as_u64().unwrap_or(8) as usize;
    let torsion_ch = baselines["torsion_c_hidden"].as_u64().unwrap_or(6) as usize;

    let torsion_single = flat_f64(&baselines["torsion_single"]);
    let torsion_weights = flat_f64(&baselines["torsion_weights"]);
    let torsion_expected = flat_f64(&baselines["torsion_output"]);

    let torsion_rust = structure::torsion_angles(
        &torsion_single,
        &torsion_weights,
        torsion_n,
        torsion_cs,
        torsion_ch,
    );
    h.check_bool(
        "Torsion output length",
        torsion_rust.len() == torsion_expected.len(),
    );
    let mut torsion_max = 0.0_f64;
    for (r, e) in torsion_rust.iter().zip(torsion_expected.iter()) {
        torsion_max = torsion_max.max((r - e).abs());
    }
    h.check_abs(
        "Torsion max diff",
        torsion_max,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool("Torsion finite", torsion_rust.iter().all(|v| v.is_finite()));
    // Verify unit circle normalization
    let mut unit_ok = true;
    for i in 0..torsion_n {
        for a in 0..7 {
            let s = torsion_rust[i * 14 + a * 2];
            let c = torsion_rust[i * 14 + a * 2 + 1];
            let r = s.hypot(c);
            if (r - 1.0).abs() > 1e-10 {
                unit_ok = false;
            }
        }
    }
    h.check_bool("Torsion unit circle", unit_ok);

    // ── Summary invariants ──────────────────────────────────────
    h.check_bool("All GELU finite", gelu_rust.iter().all(|v| v.is_finite()));
    h.check_bool(
        "All LayerNorm finite",
        ln_rust.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "All TriMul outgoing finite",
        tri_out_rust.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "All TriMul incoming finite",
        tri_in_rust.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "All TriAttn finite",
        tri_attn_rust.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "All SDPA output finite",
        sdpa_rust.iter().all(|v| v.is_finite()),
    );

    h.finish();
}
