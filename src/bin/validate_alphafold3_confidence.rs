// SPDX-License-Identifier: AGPL-3.0-or-later

//! nF-03 Phase C: `AlphaFold3` confidence head validation.
//!
//! Loads Python-generated baselines from `confidence_baselines.json` and
//! validates that Rust CPU implementations of pLDDT, PAE, pDE, and ranking
//! score match within cross-language tolerance (1e-10).
//!
//! ## Provenance
//!
//! Python baseline: `control/coral_forge/alphafold3_confidence.py`
//! Reference: Abramson et al. Nature 630:493-500 (2024), §5.9
//!
//! ## Experiments
//!
//! | Check | Primitive | What it validates |
//! |-------|-----------|-------------------|
//! | nF-C01 | pLDDT head | Linear → sigmoid per-residue confidence |
//! | nF-C02 | PAE head | Pair → softmax → expected alignment error |
//! | nF-C03 | pDE head | Pair → softmax → predicted distance error |
//! | nF-C04 | Ranking score | Weighted combination of metrics |
//! | nF-C05 | Cross-head | Consistency checks across all heads |

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "validation binary"
)]

use neural_spring::coral_forge::confidence;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use serde_json::Value;
use std::io::BufReader;

fn flat_f64(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(arr) => arr.iter().flat_map(flat_f64).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap_or(0.0)],
        _ => vec![],
    }
}

fn json_usize(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize)
}

fn main() {
    let mut h = ValidationHarness::new("validate_alphafold3_confidence");

    let json_path =
        neural_spring::validation::baseline_path("control/coral_forge/confidence_baselines.json");
    let file = match std::fs::File::open(&json_path) {
        Ok(f) => f,
        Err(e) => {
            println!("Run alphafold3_confidence.py first: {e}");
            std::process::exit(1);
        }
    };
    let baselines: Value = match serde_json::from_reader(BufReader::new(file)) {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("parse confidence_baselines.json: {e}"), false);
            h.finish();
        }
    };

    let Some(n_res) = json_usize(&baselines, "n_res") else {
        h.check_bool("baseline missing n_res", false);
        h.finish();
    };
    let Some(d_pair) = json_usize(&baselines, "d_pair") else {
        h.check_bool("baseline missing d_pair", false);
        h.finish();
    };
    let Some(n_bins_pae) = json_usize(&baselines, "n_bins_pae") else {
        h.check_bool("baseline missing n_bins_pae", false);
        h.finish();
    };
    let Some(n_bins_pde) = json_usize(&baselines, "n_bins_pde") else {
        h.check_bool("baseline missing n_bins_pde", false);
        h.finish();
    };
    let Some(max_pde) = baselines["max_pde"].as_f64() else {
        h.check_bool("baseline missing max_pde", false);
        h.finish();
    };

    // ─── nF-C01: pLDDT head ────────────────────────────────────
    {
        let single_repr = flat_f64(&baselines["plddt_single_repr"]);
        let w = flat_f64(&baselines["plddt_w"]);
        let b = baselines["plddt_b"].as_f64().unwrap_or(0.0);
        let py_plddt = flat_f64(&baselines["plddt_values"]);

        let rs_plddt = confidence::plddt_head(&single_repr, n_res, d_pair, &w, b);

        let max_diff = rs_plddt
            .iter()
            .zip(py_plddt.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-C01a pLDDT values vs Python",
            max_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        let all_in_range = rs_plddt.iter().all(|&v| (0.0..=1.0).contains(&v));
        h.check_bool("nF-C01b pLDDT in [0,1]", all_in_range);

        let spread = rs_plddt.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - rs_plddt.iter().copied().fold(f64::INFINITY, f64::min);
        h.check_bool(
            "nF-C01c pLDDT not degenerate",
            spread > tolerances::PLDDT_DEGENERACY_THRESHOLD,
        );
    }

    // ─── nF-C02: PAE head ──────────────────────────────────────
    {
        let pair_repr = flat_f64(&baselines["pae_pair_repr"]);
        let w = flat_f64(&baselines["pae_w"]);
        let b = flat_f64(&baselines["pae_b"]);
        let py_expected = flat_f64(&baselines["pae_expected"]);
        let py_probs = flat_f64(&baselines["pae_probs"]);

        let (rs_expected, rs_probs) =
            confidence::pae_head(&pair_repr, n_res, d_pair, &w, &b, n_bins_pae);

        let max_exp_diff = rs_expected
            .iter()
            .zip(py_expected.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-C02a PAE expected vs Python",
            max_exp_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        let max_prob_diff = rs_probs
            .iter()
            .zip(py_probs.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-C02b PAE probs vs Python",
            max_prob_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        let probs_sum_ok = rs_probs.chunks_exact(n_bins_pae).all(|row| {
            let s: f64 = row.iter().sum();
            (s - 1.0).abs() < tolerances::CROSS_LANGUAGE
        });
        h.check_bool("nF-C02c PAE probs sum to 1", probs_sum_ok);
    }

    // ─── nF-C03: pDE head ──────────────────────────────────────
    {
        let pair_repr = flat_f64(&baselines["pae_pair_repr"]);
        let w = flat_f64(&baselines["pde_w"]);
        let b = flat_f64(&baselines["pde_b"]);
        let py_expected = flat_f64(&baselines["pde_expected"]);
        let py_probs = flat_f64(&baselines["pde_probs"]);

        let (rs_expected, rs_probs) =
            confidence::pde_head(&pair_repr, n_res, d_pair, &w, &b, n_bins_pde, max_pde);

        let max_exp_diff = rs_expected
            .iter()
            .zip(py_expected.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-C03a pDE expected vs Python",
            max_exp_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        let max_prob_diff = rs_probs
            .iter()
            .zip(py_probs.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-C03b pDE probs vs Python",
            max_prob_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        let probs_sum_ok = rs_probs.chunks_exact(n_bins_pde).all(|row| {
            let s: f64 = row.iter().sum();
            (s - 1.0).abs() < tolerances::CROSS_LANGUAGE
        });
        h.check_bool("nF-C03c pDE probs sum to 1", probs_sum_ok);

        let all_non_neg = rs_expected.iter().all(|&v| v >= 0.0);
        h.check_bool("nF-C03d pDE expected non-negative", all_non_neg);
    }

    // ─── nF-C04: Ranking score ─────────────────────────────────
    {
        let py_score = baselines["ranking_score"].as_f64().unwrap_or(0.0);
        let py_perfect = baselines["ranking_perfect"].as_f64().unwrap_or(0.0);
        let py_worst = baselines["ranking_worst"].as_f64().unwrap_or(0.0);

        let plddt = flat_f64(&baselines["plddt_values"]);
        let pae_expected = flat_f64(&baselines["pae_expected"]);
        let pde_expected = flat_f64(&baselines["pde_expected"]);

        let rs_score = confidence::ranking_score(
            &plddt,
            &pae_expected,
            &pde_expected,
            &confidence::RankingWeights::default(),
        );
        h.check_abs(
            "nF-C04a Ranking score vs Python",
            (rs_score - py_score).abs(),
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        let rs_perfect = confidence::ranking_score(
            &vec![1.0; n_res],
            &vec![0.0; n_res * n_res],
            &vec![0.0; n_res * n_res],
            &confidence::RankingWeights::default(),
        );
        h.check_abs(
            "nF-C04b Perfect score vs Python",
            (rs_perfect - py_perfect).abs(),
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        let rs_worst = confidence::ranking_score(
            &vec![0.0; n_res],
            &vec![31.75; n_res * n_res],
            &vec![30.0; n_res * n_res],
            &confidence::RankingWeights::default(),
        );
        h.check_abs(
            "nF-C04c Worst score vs Python",
            (rs_worst - py_worst).abs(),
            0.0,
            tolerances::CROSS_LANGUAGE,
        );
    }

    // ─── nF-C05: Cross-head consistency ────────────────────────
    {
        let plddt = flat_f64(&baselines["plddt_values"]);
        let pae_expected = flat_f64(&baselines["pae_expected"]);
        let pde_expected = flat_f64(&baselines["pde_expected"]);

        let mean_plddt: f64 = plddt.iter().sum::<f64>() / plddt.len() as f64;
        h.check_bool(
            "nF-C05a pLDDT mean in (0,1)",
            mean_plddt > 0.0 && mean_plddt < 1.0,
        );

        let mean_pae: f64 = pae_expected.iter().sum::<f64>() / pae_expected.len() as f64;
        h.check_bool(
            "nF-C05b PAE mean in (0, max)",
            mean_pae > 0.0 && mean_pae < 31.75,
        );

        let mean_pde: f64 = pde_expected.iter().sum::<f64>() / pde_expected.len() as f64;
        h.check_bool(
            "nF-C05c pDE mean in (0, max)",
            mean_pde > 0.0 && mean_pde < 30.0,
        );
    }

    h.finish();
}
