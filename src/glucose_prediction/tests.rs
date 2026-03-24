// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unit tests for the [`super`] module (CGM synthesis, autocorrelation, experiment, JSON load).

#![expect(clippy::expect_used, reason = "test assertions")]
#![expect(
    clippy::cast_precision_loss,
    reason = "test data lengths safely fit in f64 mantissa"
)]

use super::*;
use crate::tolerances;

#[test]
fn synthetic_cgm_range() {
    let cgm = generate_synthetic_cgm(14, 42);
    assert_eq!(cgm.len(), 14 * 288);
    assert!(cgm.iter().all(|&g| (40.0..=400.0).contains(&g)));
}

#[test]
fn synthetic_cgm_statistics() {
    let cgm = generate_synthetic_cgm(14, 42);
    let mean = cgm.iter().sum::<f64>() / cgm.len() as f64;
    assert!(
        mean > 100.0 && mean < 200.0,
        "mean glucose should be physiological"
    );
    let std = (cgm.iter().map(|&g| (g - mean).powi(2)).sum::<f64>() / cgm.len() as f64).sqrt();
    assert!(std > 5.0 && std < 50.0, "std should be reasonable");
}

#[test]
fn synthetic_cgm_deterministic() {
    let cgm1 = generate_synthetic_cgm(7, 42);
    let cgm2 = generate_synthetic_cgm(7, 42);
    assert_eq!(cgm1, cgm2, "CGM generation must be deterministic");
}

#[test]
fn autocorrelation_unit_lag() {
    let cgm = generate_synthetic_cgm(14, 42);
    let acor = autocorrelation(&cgm, 144);
    assert!((acor[0] - 1.0).abs() < tolerances::ZERO_DETECTION);
    assert!(
        acor[1] > 0.9,
        "adjacent CGM samples should be highly correlated"
    );
    assert!(acor[72] < acor[0], "correlation should decay over 6 hours");
}

#[test]
fn tau_estimate_reasonable() {
    let cgm = generate_synthetic_cgm(14, 42);
    let acor = autocorrelation(&cgm, 144);
    let tau = estimate_tau(&acor);
    let tau_hours = tau as f64 * DT_MINUTES / 60.0;
    assert!(
        (1.0..=6.0).contains(&tau_hours),
        "τ should be 1-6 hours, got {tau_hours}"
    );
}

#[test]
fn create_sequences_lengths() {
    let data: Vec<f64> = (0..100).map(f64::from).collect();
    let (inputs, targets) = create_sequences(&data, 12, 6);
    assert_eq!(inputs.len(), targets.len());
    assert!(!inputs.is_empty());
    assert_eq!(inputs[0].len(), 12);
}

#[test]
fn r2_score_perfect() {
    let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let r2 = r2_score(&actual, &actual);
    assert!((r2 - 1.0).abs() < tolerances::ZERO_DETECTION);
}

#[test]
fn rmse_zero_for_identical() {
    let actual = vec![1.0, 2.0, 3.0];
    let r = rmse(&actual, &actual);
    assert!(r.abs() < tolerances::ZERO_DETECTION);
}

#[test]
fn run_experiment_produces_results() {
    let horizons = [1, 6];
    let (results, predictor) = run_glucose_experiment(7, 8, 6, &horizons, 42);
    assert_eq!(results.len(), 2);
    assert!(results[0].r2_lstm.is_finite());
    assert!(results[1].r2_lstm.is_finite());
    assert_eq!(predictor.hidden_size, 8);
    assert_eq!(predictor.readouts.len(), 2);
}

#[test]
fn short_horizon_better_than_long() {
    let horizons = [1, 12];
    let (results, _) = run_glucose_experiment(14, 16, 12, &horizons, 42);
    assert!(
        results[0].r2_lstm > results[1].r2_lstm,
        "R²(1-step) should exceed R²(12-step)"
    );
}

#[test]
fn experiment_deterministic() {
    let horizons = [1, 6];
    let (r1, _) = run_glucose_experiment(7, 8, 6, &horizons, 42);
    let (r2, _) = run_glucose_experiment(7, 8, 6, &horizons, 42);
    for (a, b) in r1.iter().zip(r2.iter()) {
        assert!(
            (a.r2_lstm - b.r2_lstm).abs() < tolerances::ZERO_DETECTION,
            "experiment must be deterministic"
        );
    }
}

#[test]
fn extract_features_dimension() {
    let cgm = generate_synthetic_cgm(7, 42);
    let g_mean = cgm.iter().sum::<f64>() / cgm.len() as f64;
    let g_var = cgm.iter().map(|&g| (g - g_mean).powi(2)).sum::<f64>() / cgm.len() as f64;
    let g_std = g_var.sqrt().max(tolerances::VARIANCE_DIVISION_GUARD);
    let norm: Vec<f64> = cgm.iter().map(|&g| (g - g_mean) / g_std).collect();
    let window = &norm[..24];

    let hs = 8;
    let mut rng = crate::rng::Rng::new(42);
    let w_input: Vec<f64> = (0..4 * hs).map(|_| rng.normal() * 0.5).collect();
    let w_hidden: Vec<f64> = (0..4 * hs * hs).map(|_| rng.normal() * 0.1).collect();
    let mut b_input = vec![0.0; 4 * hs];
    let b_hidden = vec![0.0; 4 * hs];
    for b in &mut b_input[hs..2 * hs] {
        *b = 1.0;
    }
    let lstm_w = crate::sequence::LstmWeights {
        w_input: &w_input,
        w_hidden: &w_hidden,
        b_input: &b_input,
        b_hidden: &b_hidden,
        hidden_size: hs,
    };

    let features = extract_features(window, &lstm_w);
    assert_eq!(
        features.len(),
        3 * hs,
        "features should be [mean, std, last] × hidden_size"
    );
    assert!(
        features.iter().all(|v| v.is_finite()),
        "all features must be finite"
    );
}

#[test]
fn solve_symmetric_identity() {
    let a = vec![1.0, 0.0, 0.0, 1.0];
    let b = vec![3.0, 7.0];
    let x = solve_symmetric(&a, &b, 2);
    assert!(
        (x[0] - 3.0).abs() < tolerances::CROSS_LANGUAGE,
        "I·x = b → x = b"
    );
    assert!((x[1] - 7.0).abs() < tolerances::CROSS_LANGUAGE);
}

#[test]
fn solve_symmetric_known_system() {
    let a = vec![4.0, 2.0, 2.0, 3.0];
    let b = vec![8.0, 7.0];
    let x = solve_symmetric(&a, &b, 2);
    let residual_0 = (4.0f64.mul_add(x[0], 2.0 * x[1]) - 8.0).abs();
    let residual_1 = (2.0f64.mul_add(x[0], 3.0 * x[1]) - 7.0).abs();
    assert!(
        residual_0 < tolerances::ODE_ATOL,
        "residual[0] = {residual_0}"
    );
    assert!(
        residual_1 < tolerances::ODE_ATOL,
        "residual[1] = {residual_1}"
    );
}

#[test]
fn load_glucose_from_json_valid() {
    let json = r#"{
            "cgm_stats": {"mean": 120.0, "std": 15.0},
            "weights": {
                "hidden_size": 2,
                "W_i": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "W_h": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                         0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
                "b_i": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "b_h": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "lstm_config": {"seq_len": 12},
            "horizons": [
                {"horizon_steps": 1, "W_out": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "b_out": 0.0}
            ]
        }"#;
    let predictor = load_glucose_from_json(json).expect("valid JSON should parse");
    assert_eq!(predictor.hidden_size, 2);
    assert_eq!(predictor.seq_len, 12);
    assert!((predictor.cgm_mean - 120.0).abs() < tolerances::CROSS_LANGUAGE);
    assert_eq!(predictor.readouts.len(), 1);
    assert_eq!(predictor.readouts[0].0, 1);
}

#[test]
fn load_glucose_from_json_missing_field() {
    let json = r#"{"cgm_stats": {"mean": 120.0}}"#;
    assert!(load_glucose_from_json(json).is_err());
}

#[test]
fn load_glucose_from_json_invalid_json() {
    assert!(load_glucose_from_json("not json").is_err());
}
