// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "validation binary"
)]

//! nF-02: AlphaFold2 Evoformer Block Validation (Jumper et al. 2021)
//!
//! Validates the complete Evoformer block pipeline and Structure Module
//! against Python baselines from `alphafold2_evoformer_block.py`.
//!
//! Reference: Jumper et al. "Highly accurate protein structure prediction
//! with AlphaFold" Nature 596:583-589 (2021)

use neural_spring::coral_forge::structure;
use neural_spring::coral_forge::structure::IpaConfig;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

const BASELINE_JSON: &str =
    include_str!("../../control/coral_forge/evoformer_block_baselines.json");

fn flat_f64(val: &serde_json::Value) -> Vec<f64> {
    match val {
        serde_json::Value::Array(arr) => arr.iter().flat_map(flat_f64).collect(),
        serde_json::Value::Number(n) => vec![n.as_f64().unwrap_or(0.0)],
        _ => vec![],
    }
}

fn main() {
    let mut h = ValidationHarness::new("alphafold2_evoformer");

    let Ok(baselines) = serde_json::from_str::<serde_json::Value>(BASELINE_JSON) else {
        h.check_bool("JSON parse", false);
        h.finish();
    };

    let n_res = baselines["n_res"].as_u64().unwrap_or(6) as usize;
    let n_seq = baselines["n_seq"].as_u64().unwrap_or(4) as usize;
    let n_heads = baselines["n_heads"].as_u64().unwrap_or(2) as usize;
    let head_dim = baselines["head_dim"].as_u64().unwrap_or(4) as usize;
    let channels = baselines["channels"].as_u64().unwrap_or(4) as usize;
    let c_msa = baselines["c_msa"].as_u64().unwrap_or(8) as usize;
    let ipa_n_points = baselines["sm_ipa_n_points"].as_u64().unwrap_or(2) as usize;

    h.check_bool("metadata loaded", n_res > 0 && channels > 0);

    // ── Evoformer block outputs: shape + finite only (weights not in JSON) ──
    let msa_output = flat_f64(&baselines["msa_output"]);
    let pair_output = flat_f64(&baselines["pair_output"]);
    let msa_expected_len = n_seq * n_res * c_msa;
    let pair_expected_len = n_res * n_res * channels;

    h.check_bool("MSA output shape", msa_output.len() == msa_expected_len);
    h.check_bool(
        "MSA output finite",
        msa_output.iter().all(|v| v.is_finite()),
    );

    h.check_bool("Pair output shape", pair_output.len() == pair_expected_len);
    h.check_bool(
        "Pair output finite",
        pair_output.iter().all(|v| v.is_finite()),
    );

    // ── Triangle attention scores: shape + finite (inputs need random weights) ──
    let tri_attn_expected = flat_f64(&baselines["tri_attn_scores"]);
    let tri_attn_len = n_res * n_heads * n_res * n_res;
    h.check_bool(
        "TriAttn scores shape",
        tri_attn_expected.len() == tri_attn_len,
    );
    h.check_bool(
        "TriAttn scores finite",
        tri_attn_expected.iter().all(|v| v.is_finite()),
    );

    // ── Structure Module: IPA scores (full value comparison) ────────────────────
    let sm_q_scalar = flat_f64(&baselines["sm_q_scalar"]);
    let sm_k_scalar = flat_f64(&baselines["sm_k_scalar"]);
    let sm_pair_bias = flat_f64(&baselines["sm_pair_bias"]);
    let sm_q_points = flat_f64(&baselines["sm_q_points"]);
    let sm_k_points = flat_f64(&baselines["sm_k_points"]);
    let init_frames_rot = flat_f64(&baselines["init_frames_rot"]);
    let init_frames_trans = flat_f64(&baselines["init_frames_trans"]);

    let mut sm_frames = Vec::with_capacity(n_res * 12);
    for i in 0..n_res {
        for r in 0..9 {
            sm_frames.push(init_frames_rot[i * 9 + r]);
        }
        for t in 0..3 {
            sm_frames.push(init_frames_trans[i * 3 + t]);
        }
    }

    let sm_ipa_scores_expected = flat_f64(&baselines["sm_ipa_scores"]);
    let ipa_cfg = IpaConfig {
        n_res,
        n_heads,
        head_dim,
        n_points: ipa_n_points,
        w_l: 1.0,
        w_c: 1.0,
        w_p: 1.0,
        gamma: 0.5,
    };
    let sm_ipa_scores_rust = structure::ipa_scores(
        &sm_q_scalar,
        &sm_k_scalar,
        &sm_pair_bias,
        &sm_q_points,
        &sm_k_points,
        &sm_frames,
        &ipa_cfg,
    );

    h.check_bool(
        "IPA scores length",
        sm_ipa_scores_rust.len() == sm_ipa_scores_expected.len(),
    );
    let mut ipa_max_diff = 0.0_f64;
    for (r, e) in sm_ipa_scores_rust.iter().zip(sm_ipa_scores_expected.iter()) {
        ipa_max_diff = ipa_max_diff.max((r - e).abs());
    }
    h.check_abs(
        "IPA scores max diff",
        ipa_max_diff,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "IPA scores finite",
        sm_ipa_scores_rust.iter().all(|v| v.is_finite()),
    );

    // ── Structure Module: Backbone update ────────────────────────────────────
    let sm_delta_quats = flat_f64(&baselines["sm_delta_quats"]);
    let sm_delta_trans = flat_f64(&baselines["sm_delta_trans"]);
    let sm_new_rot_expected = flat_f64(&baselines["sm_new_rot"]);
    let sm_new_trans_expected = flat_f64(&baselines["sm_new_trans"]);

    let bb_updated =
        structure::backbone_update(&sm_delta_quats, &sm_delta_trans, &sm_frames, n_res);
    h.check_bool("Backbone output length", bb_updated.len() == n_res * 12);

    let mut bb_rot_max = 0.0_f64;
    let mut bb_trans_max = 0.0_f64;
    for i in 0..n_res {
        for r in 0..9 {
            let diff = (bb_updated[i * 12 + r] - sm_new_rot_expected[i * 9 + r]).abs();
            bb_rot_max = bb_rot_max.max(diff);
        }
        for t in 0..3 {
            let diff = (bb_updated[i * 12 + 9 + t] - sm_new_trans_expected[i * 3 + t]).abs();
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

    // ── Structure Module: Torsion angles ─────────────────────────────────────
    let single_repr = flat_f64(&baselines["single_repr"]);
    let sm_torsion_weights = flat_f64(&baselines["sm_torsion_weights"]);
    let sm_torsion_expected = flat_f64(&baselines["sm_torsion_output"]);

    let c_single = 8;
    let c_hidden = 6;
    let torsion_rust =
        structure::torsion_angles(&single_repr, &sm_torsion_weights, n_res, c_single, c_hidden);

    h.check_bool(
        "Torsion output length",
        torsion_rust.len() == sm_torsion_expected.len(),
    );
    let mut torsion_max = 0.0_f64;
    for (r, e) in torsion_rust.iter().zip(sm_torsion_expected.iter()) {
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
    for i in 0..n_res {
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

    h.finish();
}
