// SPDX-License-Identifier: AGPL-3.0-or-later

//! nW-05: Rust-side validation of ESN WDM regime classifier.
//!
//! Loads ESN weights from `esn_regime_baseline.json`, runs Rust
//! inference on test conditions, and validates that predictions
//! match Python baselines and physical expectations.
//!
//! ## Provenance
//!
//! Python baseline: `control/wdm/esn_regime_classifier.py`
//! Reference: Jaeger (2001), Ichimaru (1994)

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::wdm_esn;

const BASELINE_JSON: &str = include_str!("../../control/wdm/esn_regime_baseline.json");

fn main() {
    let mut h = ValidationHarness::new("wdm_esn");

    let classifier = match wdm_esn::load_esn_from_json(BASELINE_JSON) {
        Ok(c) => c,
        Err(e) => {
            h.check_bool("JSON load", false);
            eprintln!("FATAL: failed to load ESN classifier: {e}");
            h.finish();
        }
    };

    h.check_bool("classifier loaded", classifier.reservoir_size > 0);
    h.check_bool("3 classes", classifier.n_classes == 3);
    h.check_bool("W_in non-empty", !classifier.w_in.is_empty());

    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(BASELINE_JSON) else {
        h.check_bool("baseline JSON parse", false);
        h.finish();
    };
    let Some(refs) = parsed["reference_predictions"].as_array() else {
        h.check_bool("reference_predictions must be array", false);
        h.finish();
    };

    for (idx, ref_pred) in refs.iter().enumerate() {
        let log_rho = ref_pred["log_rho"].as_f64().unwrap_or(0.0);
        let log_t = ref_pred["log_T"].as_f64().unwrap_or(0.0);
        let py_label = ref_pred["pred_label"].as_u64().unwrap_or(0) as usize;
        let py_scores: Vec<f64> = ref_pred["scores"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .collect();

        let (rs_label, rs_scores) = classifier.classify(log_rho, log_t);

        h.check_bool(
            &format!("ref[{idx}] label matches Python"),
            rs_label == py_label,
        );

        if !py_scores.is_empty() {
            for (c, (&py_s, &rs_s)) in py_scores.iter().zip(rs_scores.iter()).enumerate() {
                h.check_abs(
                    &format!("ref[{idx}] score[{c}] parity"),
                    rs_s,
                    py_s,
                    tolerances::CROSS_LANGUAGE,
                );
            }
        }
    }

    // Determinism
    let (l1, s1) = classifier.classify(0.5, 5.5);
    let (l2, s2) = classifier.classify(0.5, 5.5);
    h.check_bool("label deterministic", l1 == l2);
    for (a, b) in s1.iter().zip(s2.iter()) {
        h.check_bool("scores deterministic", (a - b).abs() < f64::EPSILON);
    }

    // Physics: extreme conditions should classify correctly
    // Very hot + low density → Classical (Γ << 1)
    let (label_hot, _) = classifier.classify(-0.5, 8.0);
    h.check_bool("hot + sparse → Classical", label_hot == 0);

    // Very cold + high density → Degenerate (Γ >> 10)
    let (label_cold, _) = classifier.classify(2.0, 3.5);
    h.check_bool("cold + dense → Degenerate", label_cold == 2);

    // All scores finite
    let test_points: &[(f64, f64)] = &[(-1.0, 8.0), (0.0, 6.0), (1.0, 5.0), (2.0, 4.0), (0.5, 5.5)];
    for &(lr, lt) in test_points {
        let (label, scores) = classifier.classify(lr, lt);
        h.check_bool(&format!("({lr},{lt}) label in [0,3)"), label < 3);
        h.check_bool(
            &format!("({lr},{lt}) scores finite"),
            scores.iter().all(|s| s.is_finite()),
        );
    }

    h.finish();
}
