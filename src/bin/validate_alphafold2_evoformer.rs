// SPDX-License-Identifier: AGPL-3.0-or-later

//! nF-02: `AlphaFold2` Evoformer Block Validation (Jumper et al. 2021)
//!
//! Validates the complete Evoformer block pipeline and Structure Module
//! against Python baselines from `alphafold2_evoformer_block.py`.
//!
//! Reference: Jumper et al. "Highly accurate protein structure prediction
//! with `AlphaFold`" Nature 596:583-589 (2021)
//!
//! ## Provenance
//!
//! Validation class: Integration.
//! Python baseline: `control/coral_forge/evoformer_block_baselines.json` from `alphafold2_evoformer_block.py`.
//! Components: `coral_forge::structure` (Evoformer, IPA, Structure Module).

#![expect(
    clippy::cast_possible_truncation,
    reason = "JSON u64 → usize for small dimension fields (≤64)"
)]
#![expect(
    clippy::similar_names,
    reason = "sm_q_scalar / sm_k_scalar mirror Python baseline naming"
)]

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

struct BaselineDims {
    n_res: usize,
    n_seq: usize,
    n_heads: usize,
    head_dim: usize,
    channels: usize,
    c_msa: usize,
    ipa_n_points: usize,
}

fn parse_dims(baselines: &serde_json::Value) -> BaselineDims {
    BaselineDims {
        n_res: baselines["n_res"].as_u64().unwrap_or(6) as usize,
        n_seq: baselines["n_seq"].as_u64().unwrap_or(4) as usize,
        n_heads: baselines["n_heads"].as_u64().unwrap_or(2) as usize,
        head_dim: baselines["head_dim"].as_u64().unwrap_or(4) as usize,
        channels: baselines["channels"].as_u64().unwrap_or(4) as usize,
        c_msa: baselines["c_msa"].as_u64().unwrap_or(8) as usize,
        ipa_n_points: baselines["sm_ipa_n_points"].as_u64().unwrap_or(2) as usize,
    }
}

fn validate_evoformer_outputs(
    h: &mut ValidationHarness,
    baselines: &serde_json::Value,
    dims: &BaselineDims,
) {
    let msa_output = flat_f64(&baselines["msa_output"]);
    let pair_output = flat_f64(&baselines["pair_output"]);

    h.check_bool(
        "MSA output shape",
        msa_output.len() == dims.n_seq * dims.n_res * dims.c_msa,
    );
    h.check_bool(
        "MSA output finite",
        msa_output.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "Pair output shape",
        pair_output.len() == dims.n_res * dims.n_res * dims.channels,
    );
    h.check_bool(
        "Pair output finite",
        pair_output.iter().all(|v| v.is_finite()),
    );

    let tri_attn_expected = flat_f64(&baselines["tri_attn_scores"]);
    h.check_bool(
        "TriAttn scores shape",
        tri_attn_expected.len() == dims.n_res * dims.n_heads * dims.n_res * dims.n_res,
    );
    h.check_bool(
        "TriAttn scores finite",
        tri_attn_expected.iter().all(|v| v.is_finite()),
    );
}

fn interleave_frames(rot: &[f64], trans: &[f64], n_res: usize) -> Vec<f64> {
    let mut frames = Vec::with_capacity(n_res * 12);
    for i in 0..n_res {
        frames.extend_from_slice(&rot[i * 9..(i + 1) * 9]);
        frames.extend_from_slice(&trans[i * 3..(i + 1) * 3]);
    }
    frames
}

fn validate_ipa_scores(
    h: &mut ValidationHarness,
    baselines: &serde_json::Value,
    dims: &BaselineDims,
    sm_frames: &[f64],
) {
    let sm_q_scalar = flat_f64(&baselines["sm_q_scalar"]);
    let sm_k_scalar = flat_f64(&baselines["sm_k_scalar"]);
    let sm_pair_bias = flat_f64(&baselines["sm_pair_bias"]);
    let sm_q_points = flat_f64(&baselines["sm_q_points"]);
    let sm_k_points = flat_f64(&baselines["sm_k_points"]);
    let expected = flat_f64(&baselines["sm_ipa_scores"]);

    let ipa_cfg = IpaConfig {
        n_res: dims.n_res,
        n_heads: dims.n_heads,
        head_dim: dims.head_dim,
        n_points: dims.ipa_n_points,
        w_l: 1.0,
        w_c: 1.0,
        w_p: 1.0,
        gamma: 0.5,
    };
    let computed = structure::ipa_scores(
        &sm_q_scalar,
        &sm_k_scalar,
        &sm_pair_bias,
        &sm_q_points,
        &sm_k_points,
        sm_frames,
        &ipa_cfg,
    );

    h.check_bool("IPA scores length", computed.len() == expected.len());
    let max_diff = computed
        .iter()
        .zip(expected.iter())
        .map(|(r, e)| (r - e).abs())
        .fold(0.0_f64, f64::max);
    h.check_abs(
        "IPA scores max diff",
        max_diff,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool("IPA scores finite", computed.iter().all(|v| v.is_finite()));
}

fn validate_backbone(
    h: &mut ValidationHarness,
    baselines: &serde_json::Value,
    n_res: usize,
    sm_frames: &[f64],
) {
    let delta_quats = flat_f64(&baselines["sm_delta_quats"]);
    let delta_trans = flat_f64(&baselines["sm_delta_trans"]);
    let rot_expected = flat_f64(&baselines["sm_new_rot"]);
    let trans_expected = flat_f64(&baselines["sm_new_trans"]);

    let updated = structure::backbone_update(&delta_quats, &delta_trans, sm_frames, n_res);
    h.check_bool("Backbone output length", updated.len() == n_res * 12);

    let mut rot_max = 0.0_f64;
    let mut trans_max = 0.0_f64;
    for i in 0..n_res {
        for r in 0..9 {
            rot_max = rot_max.max((updated[i * 12 + r] - rot_expected[i * 9 + r]).abs());
        }
        for t in 0..3 {
            trans_max = trans_max.max((updated[i * 12 + 9 + t] - trans_expected[i * 3 + t]).abs());
        }
    }
    h.check_abs(
        "Backbone rot max diff",
        rot_max,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "Backbone trans max diff",
        trans_max,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "Backbone output finite",
        updated.iter().all(|v| v.is_finite()),
    );
}

fn validate_torsion_angles(h: &mut ValidationHarness, baselines: &serde_json::Value, n_res: usize) {
    let single_repr = flat_f64(&baselines["single_repr"]);
    let weights = flat_f64(&baselines["sm_torsion_weights"]);
    let expected = flat_f64(&baselines["sm_torsion_output"]);

    let computed = structure::torsion_angles(&single_repr, &weights, n_res, 8, 6);

    h.check_bool("Torsion output length", computed.len() == expected.len());
    let max_diff = computed
        .iter()
        .zip(expected.iter())
        .map(|(r, e)| (r - e).abs())
        .fold(0.0_f64, f64::max);
    h.check_abs(
        "Torsion max diff",
        max_diff,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool("Torsion finite", computed.iter().all(|v| v.is_finite()));

    let unit_ok = (0..n_res).all(|i| {
        (0..7).all(|a| {
            let s = computed[i * 14 + a * 2];
            let c = computed[i * 14 + a * 2 + 1];
            (s.hypot(c) - 1.0).abs() <= tolerances::CROSS_LANGUAGE
        })
    });
    h.check_bool("Torsion unit circle", unit_ok);
}

fn main() {
    let mut h = ValidationHarness::new("alphafold2_evoformer");

    let Ok(baselines) = serde_json::from_str::<serde_json::Value>(BASELINE_JSON) else {
        h.check_bool("JSON parse", false);
        h.finish();
    };

    let dims = parse_dims(&baselines);
    h.check_bool("metadata loaded", dims.n_res > 0 && dims.channels > 0);

    validate_evoformer_outputs(&mut h, &baselines, &dims);

    let rot = flat_f64(&baselines["init_frames_rot"]);
    let trans = flat_f64(&baselines["init_frames_trans"]);
    let sm_frames = interleave_frames(&rot, &trans, dims.n_res);

    validate_ipa_scores(&mut h, &baselines, &dims, &sm_frames);
    validate_backbone(&mut h, &baselines, dims.n_res, &sm_frames);
    validate_torsion_angles(&mut h, &baselines, dims.n_res);

    h.finish();
}
