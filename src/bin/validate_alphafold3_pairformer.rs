// SPDX-License-Identifier: AGPL-3.0-or-later

//! nF-03 Phase B: AlphaFold3 Pairformer block validation.
//!
//! Loads Python-generated baselines from `pairformer_baselines.json` and
//! validates that the Rust Pairformer block reproduces them.
//!
//! ## Provenance
//!
//! Python baseline: `control/sovereign_folding/alphafold3_pairformer.py`
//! Reuses: ~90% of Evoformer primitives from nF-02
//! Reference: Abramson et al. Nature 630:493-500 (2024)
//!
//! ## Experiments
//!
//! | Check | Primitive | What it validates |
//! |-------|-----------|-------------------|
//! | nF-PF01 | Timestep embedding | Sinusoidal positional encoding |
//! | nF-PF02 | Timestep conditioning | Pair repr + broadcast conditioning |
//! | nF-PF03 | Pairformer block (no cond) | Full block: TriMul + TriAttn + FFN |
//! | nF-PF04 | Pairformer block (with cond) | Block + timestep conditioning |
//! | nF-PF05 | Multi-block iteration | 3 blocks with decreasing timestep |

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_lines
)]

use neural_spring::sovereign_folding::pairformer;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

const BASELINE_JSON: &str =
    include_str!("../../control/sovereign_folding/pairformer_baselines.json");

fn flat_f64(val: &serde_json::Value) -> Vec<f64> {
    match val {
        serde_json::Value::Array(arr) => arr.iter().flat_map(flat_f64).collect(),
        serde_json::Value::Number(n) => vec![n.as_f64().unwrap_or(0.0)],
        _ => vec![],
    }
}

const N_RES: usize = 8;
const D_PAIR: usize = 4;
const N_HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const D_HIDDEN: usize = 16;

fn main() {
    let mut h = ValidationHarness::new("alphafold3_pairformer");

    let Ok(baselines) = serde_json::from_str::<serde_json::Value>(BASELINE_JSON) else {
        eprintln!("[ERROR] Failed to parse pairformer_baselines.json");
        std::process::exit(1);
    };

    // ─── nF-PF01: Timestep embedding ──────────────────────────────
    {
        let py_emb_0 = flat_f64(&baselines["t_emb_0"]);
        let py_emb_25 = flat_f64(&baselines["t_emb_25"]);
        let py_emb_49 = flat_f64(&baselines["t_emb_49"]);

        let rs_emb_0 = pairformer::sinusoidal_embedding(0.0, D_PAIR);
        let rs_emb_25 = pairformer::sinusoidal_embedding(25.0, D_PAIR);
        let rs_emb_49 = pairformer::sinusoidal_embedding(49.0, D_PAIR);

        let max_diff_0 = rs_emb_0.iter().zip(py_emb_0.iter())
            .map(|(r, p)| (r - p).abs()).fold(0.0_f64, f64::max);
        h.check_abs("nF-PF01a t_emb(0) vs Python", max_diff_0, 0.0, tolerances::CROSS_LANGUAGE);

        let max_diff_25 = rs_emb_25.iter().zip(py_emb_25.iter())
            .map(|(r, p)| (r - p).abs()).fold(0.0_f64, f64::max);
        h.check_abs("nF-PF01b t_emb(25) vs Python", max_diff_25, 0.0, tolerances::CROSS_LANGUAGE);

        let max_diff_49 = rs_emb_49.iter().zip(py_emb_49.iter())
            .map(|(r, p)| (r - p).abs()).fold(0.0_f64, f64::max);
        h.check_abs("nF-PF01c t_emb(49) vs Python", max_diff_49, 0.0, tolerances::CROSS_LANGUAGE);
    }

    // ─── nF-PF02: Timestep conditioning ───────────────────────────
    {
        let pair_repr = flat_f64(&baselines["pair_repr"]);
        let w_cond = flat_f64(&baselines["w_cond"]);
        let b_cond = flat_f64(&baselines["b_cond"]);
        let py_conditioned = flat_f64(&baselines["conditioned"]);

        let t_emb_25 = pairformer::sinusoidal_embedding(25.0, D_PAIR);
        let rs_conditioned = pairformer::condition_pair_with_timestep(
            &pair_repr, N_RES, D_PAIR, &t_emb_25, &w_cond, &b_cond,
        );

        let max_diff = rs_conditioned.iter().zip(py_conditioned.iter())
            .map(|(r, p)| (r - p).abs()).fold(0.0_f64, f64::max);
        h.check_abs("nF-PF02a conditioning vs Python", max_diff, 0.0, tolerances::CROSS_LANGUAGE);

        // All pairs get same shift (broadcast invariant)
        let shift_00: Vec<f64> = (0..D_PAIR)
            .map(|d| rs_conditioned[d] - pair_repr[d])
            .collect();
        let ij = 3 * N_RES + 5; // pair (3,5)
        let shift_35: Vec<f64> = (0..D_PAIR)
            .map(|d| rs_conditioned[ij * D_PAIR + d] - pair_repr[ij * D_PAIR + d])
            .collect();
        let shift_diff = shift_00.iter().zip(shift_35.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        h.check_abs("nF-PF02b broadcast invariance", shift_diff, 0.0, tolerances::CROSS_LANGUAGE);
    }

    // ─── nF-PF03: Pairformer block (no conditioning) ──────────────
    {
        let pair_input = flat_f64(&baselines["pair_input"]);
        let py_out = flat_f64(&baselines["pf_out_no_cond"]);

        let weights = build_weights(&baselines);
        let rs_out = pairformer::pairformer_block(&pair_input, N_RES, D_PAIR, &weights, None);

        let max_diff = rs_out.iter().zip(py_out.iter())
            .map(|(r, p)| (r - p).abs()).fold(0.0_f64, f64::max);
        h.check_abs("nF-PF03a block (no cond) vs Python", max_diff, 0.0, tolerances::CROSS_LANGUAGE);

        h.check_bool("nF-PF03b output finite", rs_out.iter().all(|v| v.is_finite()));
        h.check_bool(
            "nF-PF03c output differs from input",
            rs_out.iter().zip(pair_input.iter()).any(|(r, p)| (r - p).abs() > 1e-6),
        );
    }

    // ─── nF-PF04: Pairformer block (with conditioning) ────────────
    {
        let pair_input = flat_f64(&baselines["pair_input"]);
        let py_out = flat_f64(&baselines["pf_out_with_cond"]);
        let py_out_no_cond = flat_f64(&baselines["pf_out_no_cond"]);

        let weights = build_weights(&baselines);
        let t_emb_25 = pairformer::sinusoidal_embedding(25.0, D_PAIR);
        let rs_out = pairformer::pairformer_block(
            &pair_input, N_RES, D_PAIR, &weights, Some(&t_emb_25),
        );

        let max_diff = rs_out.iter().zip(py_out.iter())
            .map(|(r, p)| (r - p).abs()).fold(0.0_f64, f64::max);
        h.check_abs("nF-PF04a block (with cond) vs Python", max_diff, 0.0, tolerances::CROSS_LANGUAGE);

        // Conditioning should change the output
        h.check_bool(
            "nF-PF04b conditioning changes output",
            rs_out.iter().zip(py_out_no_cond.iter()).any(|(r, p)| (r - p).abs() > 1e-6),
        );
    }

    // ─── nF-PF05: Multi-block iteration ───────────────────────────
    {
        let pair_input = flat_f64(&baselines["pair_input"]);
        let py_multi = flat_f64(&baselines["multi_block_out"]);

        let weights = build_weights(&baselines);
        let mut pair_evolving = pair_input.clone();

        for block_idx in 0..3_usize {
            let t = 49.0 - (block_idx as f64) * 20.0;
            let t_emb = pairformer::sinusoidal_embedding(t, D_PAIR);
            pair_evolving = pairformer::pairformer_block(
                &pair_evolving, N_RES, D_PAIR, &weights, Some(&t_emb),
            );
        }

        let max_diff = pair_evolving.iter().zip(py_multi.iter())
            .map(|(r, p)| (r - p).abs()).fold(0.0_f64, f64::max);
        h.check_abs("nF-PF05a multi-block vs Python", max_diff, 0.0, tolerances::CROSS_LANGUAGE);

        h.check_bool(
            "nF-PF05b multi-block finite",
            pair_evolving.iter().all(|v| v.is_finite()),
        );

        let norm: f64 = pair_evolving.iter().map(|v| v * v).sum::<f64>().sqrt();
        h.check_bool("nF-PF05c representation norm bounded", norm < 1000.0);
    }

    h.finish();
}

fn build_weights(baselines: &serde_json::Value) -> pairformer::PairformerWeights<'static> {
    fn leak(val: &serde_json::Value) -> &'static [f64] {
        fn collect_f64(val: &serde_json::Value) -> Vec<f64> {
            match val {
                serde_json::Value::Array(arr) => arr.iter().flat_map(collect_f64).collect(),
                serde_json::Value::Number(n) => vec![n.as_f64().unwrap_or(0.0)],
                _ => vec![],
            }
        }
        Box::leak(collect_f64(val).into_boxed_slice())
    }

    pairformer::PairformerWeights {
        ln_gamma: leak(&baselines["w_ln1_gamma"]),
        ln_beta: leak(&baselines["w_ln1_beta"]),
        tri_out_wa: leak(&baselines["w_tri_out_wa"]),
        tri_out_wb: leak(&baselines["w_tri_out_wb"]),
        tri_out_wg: leak(&baselines["w_tri_out_wg"]),
        tri_in_wa: leak(&baselines["w_tri_in_wa"]),
        tri_in_wb: leak(&baselines["w_tri_in_wb"]),
        tri_in_wg: leak(&baselines["w_tri_in_wg"]),
        n_heads: N_HEADS,
        head_dim: HEAD_DIM,
        tri_attn_wq: leak(&baselines["w_tri_attn_wq"]),
        tri_attn_wk: leak(&baselines["w_tri_attn_wk"]),
        tri_attn_wv: leak(&baselines["w_tri_attn_wv"]),
        ffn_w1: leak(&baselines["w_ffn_w1"]),
        ffn_b1: leak(&baselines["w_ffn_b1"]),
        d_hidden: D_HIDDEN,
        ffn_w2: leak(&baselines["w_ffn_w2"]),
        ffn_b2: leak(&baselines["w_ffn_b2"]),
        cond_w: leak(&baselines["w_cond_w"]),
        cond_b: leak(&baselines["w_cond_b"]),
    }
}
