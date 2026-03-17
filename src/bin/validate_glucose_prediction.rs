// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper 026: Rust-side validation of LSTM blood glucose prediction.
//!
//! Validates Chuna (2020) key findings:
//! - LSTM prediction accuracy degrades with forecast horizon
//! - Autocorrelation decay sets the fundamental prediction limit
//! - Short horizon (5 min) is trivially accurate
//! - Sweet spot at 30 min where LSTM outperforms persistence
//! - Long horizon (240 min) converges to predicting the mean
//!
//! Loads LSTM reservoir weights from Python baseline JSON and runs
//! independent Rust validation. Also runs the full Rust experiment
//! from scratch and compares to Python results.
//!
//! ## Provenance
//!
//! Python baseline: `control/glucose_prediction/glucose_prediction.py`
//! Reference: Chuna (2020), medRxiv 2020.08.04.20117812

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "validation binary with multi-horizon experiment orchestration"
)]

use neural_spring::glucose_prediction::{
    autocorrelation, estimate_tau, generate_synthetic_cgm, load_glucose_from_json, r2_score, rmse,
    run_glucose_experiment,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

const BASELINE_JSON: &str =
    include_str!("../../control/glucose_prediction/glucose_prediction_baseline.json");

fn main() {
    let mut h = ValidationHarness::new("glucose_prediction");

    println!("\n── Paper 026: LSTM Blood Glucose Prediction (Chuna 2020) ──");

    let baseline = match load_glucose_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("JSON load", false);
            println!("FATAL: failed to load baseline: {e}");
            h.finish();
        }
    };

    h.check_bool("baseline loaded", baseline.hidden_size > 0);
    h.check_bool("has horizons", !baseline.readouts.is_empty());
    h.check_bool("hidden_size = 24", baseline.hidden_size == 24);
    h.check_bool("seq_len = 12", baseline.seq_len == 12);

    // ── Synthetic CGM validation ──
    println!("\n── Synthetic CGM generation ──");

    let cgm = generate_synthetic_cgm(14, 42);
    h.check_bool("CGM length = 4032", cgm.len() == 14 * 288);
    h.check_bool(
        "CGM in [40, 400]",
        cgm.iter().all(|&g| (40.0..=400.0).contains(&g)),
    );

    let g_mean = cgm.iter().sum::<f64>() / cgm.len() as f64;
    let g_std = (cgm.iter().map(|&g| (g - g_mean).powi(2)).sum::<f64>() / cgm.len() as f64).sqrt();

    h.check_abs(
        "CGM mean ≈ Python baseline",
        g_mean,
        baseline.cgm_mean,
        tolerances::GLUCOSE_CGM_STAT_TOL,
    );
    h.check_abs(
        "CGM std ≈ Python baseline",
        g_std,
        baseline.cgm_std,
        tolerances::GLUCOSE_CGM_STAT_TOL,
    );

    // ── Autocorrelation validation ──
    println!("\n── Autocorrelation analysis ──");

    let acor = autocorrelation(&cgm, 144);
    let tau_steps = estimate_tau(&acor);
    let tau_hours = tau_steps as f64 * 5.0 / 60.0;

    h.check_bool(
        "τ in [1.0, 5.0] hrs (Chuna: ~3 hrs)",
        (1.0..=5.0).contains(&tau_hours),
    );
    h.check_bool(
        "acor[0] ≈ 1.0",
        (acor[0] - 1.0).abs() < tolerances::ZERO_DETECTION,
    );

    // ── Python baseline parity (CGM stats) ──
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(BASELINE_JSON) else {
        h.check_bool("JSON reparse", false);
        h.finish();
    };

    let py_tau = parsed["autocorrelation"]["tau_hours"]
        .as_f64()
        .unwrap_or(0.0);
    h.check_abs(
        "τ matches Python",
        tau_hours,
        py_tau,
        tolerances::GLUCOSE_TAU_TOL,
    );

    // ── Full Rust experiment ──
    println!("\n── Full Rust glucose experiment (5 horizons) ──");

    let horizons = [1, 6, 12, 24, 48];
    let (results, _predictor) = run_glucose_experiment(14, 24, 12, &horizons, 42);

    for r in &results {
        println!(
            "  Horizon {:>3} min: R²={:.4}, RMSE={:.2} mg/dL, improvement={:.1}%",
            r.horizon_minutes, r.r2_lstm, r.rmse_lstm, r.lstm_improvement_pct
        );
    }

    // ── Horizon degradation checks ──
    println!("\n── Horizon degradation analysis ──");

    h.check_bool(
        "R²(5min) > 0.85 (short horizon accurate)",
        results[0].r2_lstm > 0.85,
    );
    h.check_bool(
        "R²(30min) > 0.30 (sweet spot useful)",
        results[1].r2_lstm > 0.30,
    );
    h.check_bool(
        "R²(240min) < R²(30min) (degrades with horizon)",
        results[4].r2_lstm < results[1].r2_lstm,
    );

    let monotonic = results
        .windows(2)
        .all(|w| w[0].r2_lstm >= w[1].r2_lstm - 0.05);
    h.check_bool("R² approximately monotonically decreasing", monotonic);

    h.check_bool("RMSE(5min) < 15 mg/dL", results[0].rmse_lstm < 15.0);
    h.check_bool(
        "RMSE(240min) > RMSE(5min)",
        results[4].rmse_lstm > results[0].rmse_lstm,
    );

    // ── LSTM vs persistence ──
    h.check_bool(
        "LSTM beats persistence at 30min",
        results[1].lstm_improvement_pct > 0.0,
    );

    // ── All predictions finite ──
    h.check_bool(
        "all predictions finite",
        results
            .iter()
            .all(|r| r.r2_lstm.is_finite() && r.rmse_lstm.is_finite()),
    );

    // ── Determinism ──
    println!("\n── Determinism ──");

    let (results2, _) = run_glucose_experiment(14, 24, 12, &horizons, 42);
    for (r1, r2) in results.iter().zip(results2.iter()) {
        h.check_abs(
            &format!("determinism R²({}min)", r1.horizon_minutes),
            r1.r2_lstm,
            r2.r2_lstm,
            tolerances::ZERO_DETECTION,
        );
    }

    // ── Cross-validation with Python horizon results ──
    println!("\n── Python ↔ Rust R² comparison ──");

    if let Some(py_horizons) = parsed["horizons"].as_array() {
        for py_h in py_horizons {
            let steps = py_h["horizon_steps"].as_u64().unwrap_or(0) as usize;
            let py_r2 = py_h["r2_lstm"].as_f64().unwrap_or(0.0);

            if let Some(rs_result) = results.iter().find(|r| r.horizon_steps == steps) {
                println!(
                    "  {}min: Python R²={:.4}, Rust R²={:.4}",
                    steps * 5,
                    py_r2,
                    rs_result.r2_lstm
                );
            }
        }
    }

    // ── R² and RMSE sanity on known data ──
    println!("\n── Metric sanity checks ──");

    let actual = vec![100.0, 120.0, 140.0, 160.0, 180.0];
    let r2_perfect = r2_score(&actual, &actual);
    h.check_abs(
        "R²(perfect) = 1.0",
        r2_perfect,
        1.0,
        tolerances::ZERO_DETECTION,
    );

    let rmse_zero = rmse(&actual, &actual);
    h.check_abs(
        "RMSE(perfect) = 0.0",
        rmse_zero,
        0.0,
        tolerances::ZERO_DETECTION,
    );

    h.finish();
}
