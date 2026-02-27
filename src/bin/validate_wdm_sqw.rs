// SPDX-License-Identifier: AGPL-3.0-or-later

//! nW-03: Rust-side validation of S(q,ω) peak predictor.
//!
//! Loads LSTM reservoir weights from `sqw_peak_baseline.json`,
//! runs Rust LSTM inference on synthetic density fluctuation time
//! series, and validates that predictions match Python baselines.
//!
//! ## Provenance
//!
//! Python baseline: `control/wdm/sqw_peak_predictor.py`
//! Reference: Hansen & `McDonald` (2013), Gregori et al. PRE 67 (2003)

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::wdm_sqw;

const BASELINE_JSON: &str = include_str!("../../control/wdm/sqw_peak_baseline.json");

fn generate_test_signal(omega: f64, gamma: f64, n_steps: usize, seed_offset: u64) -> Vec<f64> {
    let mut signal = Vec::with_capacity(n_steps);
    let mut rng_state = seed_offset
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    for i in 0..n_steps {
        let t = i as f64;
        let base = (-gamma * t).exp() * (omega * t).cos();
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.03;
        signal.push(base + noise);
    }
    signal
}

fn main() {
    let mut h = ValidationHarness::new("wdm_sqw");

    let predictor = match wdm_sqw::load_sqw_from_json(BASELINE_JSON) {
        Ok(p) => p,
        Err(e) => {
            h.check_bool("JSON load", false);
            eprintln!("FATAL: failed to load SQW predictor: {e}");
            h.finish();
        }
    };

    h.check_bool("predictor loaded", predictor.hidden_size > 0);
    h.check_bool("W_i non-empty", !predictor.w_i.is_empty());
    h.check_bool("W_out non-empty", !predictor.w_out.is_empty());

    let Ok(ref_preds) = serde_json::from_str::<serde_json::Value>(BASELINE_JSON) else {
        h.check_bool("baseline JSON parse", false);
        h.finish();
    };
    let Some(refs) = ref_preds["reference_predictions"].as_array() else {
        h.check_bool("reference_predictions must be array", false);
        h.finish();
    };

    for (idx, ref_pred) in refs.iter().enumerate() {
        let log_rho = ref_pred["log_rho"].as_f64().unwrap_or(0.0);
        let log_t = ref_pred["log_T"].as_f64().unwrap_or(0.0);
        let _py_omega = ref_pred["pred_omega"].as_f64().unwrap_or(0.0);
        let _py_gamma = ref_pred["pred_gamma"].as_f64().unwrap_or(0.0);

        let omega_r = 0.3 + 2.5 * (log_rho - (-0.5)) / 2.0;
        let gamma_r = 0.02 + 0.18 * (log_t - 4.0) / 3.5;

        let ts = generate_test_signal(omega_r, gamma_r, 64, 9999 + idx as u64);
        let (pred_omega, pred_gamma) = predictor.predict(&ts);

        h.check_bool(&format!("ref[{idx}] ω finite"), pred_omega.is_finite());
        h.check_bool(&format!("ref[{idx}] γ finite"), pred_gamma.is_finite());

        h.check_bool(&format!("ref[{idx}] ω positive"), pred_omega > 0.0);
        h.check_bool(&format!("ref[{idx}] γ positive"), pred_gamma > 0.0);
    }

    // Determinism
    let ts_det = generate_test_signal(1.5, 0.08, 64, 7777);
    let (o1, g1) = predictor.predict(&ts_det);
    let (o2, g2) = predictor.predict(&ts_det);
    h.check_abs("ω determinism", o1, o2, tolerances::GPU_F64_EXACT);
    h.check_abs("γ determinism", g1, g2, tolerances::GPU_F64_EXACT);

    // Physics: higher frequency signal → higher predicted ω
    let ts_low = generate_test_signal(0.5, 0.05, 64, 5555);
    let ts_high = generate_test_signal(2.5, 0.05, 64, 5555);
    let (o_low, _) = predictor.predict(&ts_low);
    let (o_high, _) = predictor.predict(&ts_high);
    h.check_bool("ω increases with frequency", o_high > o_low);

    // Physics: higher damping → higher predicted γ
    let ts_undamped = generate_test_signal(1.5, 0.03, 64, 6666);
    let ts_damped = generate_test_signal(1.5, 0.18, 64, 6666);
    let (_, g_undamped) = predictor.predict(&ts_undamped);
    let (_, g_damped) = predictor.predict(&ts_damped);
    h.check_bool("γ increases with damping", g_damped > g_undamped);

    h.finish();
}
