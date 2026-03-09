// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU-path statistics, activations, and distribution tests.

use super::*;
use crate::tolerances;

fn cpu() -> Dispatcher {
    Dispatcher::cpu_only()
}

// ── Activations / distributions ─────────────────────────────

#[test]
fn cpu_softmax_sums_to_one() {
    let d = cpu();
    let result = d.softmax(&[1.0, 2.0, 3.0]);
    let total: f64 = result.iter().sum();
    assert!((total - 1.0).abs() < tolerances::EXACT_F64);
    assert!(result[2] > result[1] && result[1] > result[0]);
}

#[test]
fn cpu_boltzmann_sums_to_one() {
    let d = cpu();
    let result = d.boltzmann(&[1.0, 2.0, 3.0], 1.0);
    let total: f64 = result.iter().sum();
    assert!((total - 1.0).abs() < tolerances::EXACT_F64);
}

// ── Reductions / statistics ─────────────────────────────────

#[test]
fn cpu_l2_distance() {
    let d = cpu();
    let dist = d.l2_distance(&[0.0, 0.0], &[3.0, 4.0]);
    assert!((dist - 5.0).abs() < tolerances::EXACT_F64);
}

#[test]
fn cpu_shannon_entropy() {
    let d = cpu();
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let h = d.shannon_entropy(&p);
    let expected = 4.0_f64.ln();
    assert!((h - expected).abs() < tolerances::CROSS_LANGUAGE);
}

#[test]
fn cpu_mean() {
    let d = cpu();
    assert!((d.mean(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < tolerances::ZERO_DETECTION);
    assert!((d.mean(&[]) - 0.0).abs() < tolerances::ZERO_DETECTION);
}

#[test]
fn cpu_variance() {
    let d = cpu();
    let v = d.variance(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    assert!((v - 4.0).abs() < tolerances::EXACT_F64);
    assert!((d.variance(&[]) - 0.0).abs() < tolerances::ZERO_DETECTION);
}

#[test]
fn cpu_pearson_correlation() {
    let d = cpu();
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let r = d.pearson_correlation(&x, &y);
    assert!(
        (r - 1.0).abs() < tolerances::EXACT_F64,
        "perfect positive correlation"
    );
}

#[test]
fn cpu_pearson_short() {
    let d = cpu();
    assert!((d.pearson_correlation(&[1.0], &[2.0]) - 0.0).abs() < tolerances::ZERO_DETECTION);
}

#[test]
fn cpu_pearson_zero_variance() {
    let d = cpu();
    let r = d.pearson_correlation(&[3.0, 3.0, 3.0], &[1.0, 2.0, 3.0]);
    assert!((r - 0.0).abs() < tolerances::ZERO_DETECTION);
}

#[test]
fn cpu_chi_squared() {
    let d = cpu();
    let obs = vec![10.0, 20.0, 30.0];
    let exp = vec![20.0, 20.0, 20.0];
    let chi2 = d.chi_squared(&obs, &exp);
    assert!((chi2 - 10.0).abs() < tolerances::CROSS_LANGUAGE);
}

#[test]
fn cpu_chi_squared_zero_expected() {
    let d = cpu();
    let chi2 = d.chi_squared(&[5.0], &[0.0]);
    assert!(
        (chi2 - 0.0).abs() < tolerances::ZERO_DETECTION,
        "zero expected → 0 contribution"
    );
}

// ── Dispatch ops (GELU, softmax row-wise, boltzmann) ────────

#[test]
fn cpu_gelu_basic() {
    let d = cpu();
    let result = d.gelu(&[0.0, 1.0, -1.0]);
    assert_eq!(result.len(), 3);
    assert!(
        (result[0] - 0.0).abs() < tolerances::GELU_LARGE_INPUT,
        "gelu(0)≈0"
    );
    assert!(result[1] > 0.5, "gelu(1)>0.5");
    assert!(result[2] < 0.0, "gelu(-1)<0");
}

#[test]
fn cpu_gelu_matches_transformer_gelu() {
    let d = cpu();
    let xs = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let dispatched = d.gelu(&xs);
    for (i, &x) in xs.iter().enumerate() {
        let expected = crate::transformer::gelu(x);
        assert!(
            (dispatched[i] - expected).abs() < tolerances::EXACT_F64,
            "gelu({x}): dispatch={}, direct={expected}",
            dispatched[i]
        );
    }
}

#[test]
fn cpu_softmax_row_wise_basic() {
    let d = cpu();
    #[rustfmt::skip]
    let matrix = vec![
        1.0, 2.0, 3.0,
        3.0, 2.0, 1.0,
    ];
    let result = d.softmax_row_wise(&matrix, 2, 3);
    assert_eq!(result.len(), 6);
    let row0_sum: f64 = result[0..3].iter().sum();
    let row1_sum: f64 = result[3..6].iter().sum();
    assert!(
        (row0_sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
        "row 0 sum = {row0_sum}"
    );
    assert!(
        (row1_sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
        "row 1 sum = {row1_sum}"
    );
}

#[test]
fn cpu_softmax_row_wise_single_row() {
    let d = cpu();
    let row = vec![0.0, 0.0, 0.0];
    let result = d.softmax_row_wise(&row, 1, 3);
    assert_eq!(result.len(), 3);
    for &v in &result {
        assert!(
            (v - 1.0 / 3.0).abs() < tolerances::CROSS_LANGUAGE,
            "uniform softmax expected 1/3, got {v}"
        );
    }
}

#[test]
fn cpu_softmax_row_wise_sums_to_one() {
    let d = cpu();
    let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = d.softmax_row_wise(&matrix, 2, 3);
    let sum_r0: f64 = result[..3].iter().sum();
    let sum_r1: f64 = result[3..].iter().sum();
    assert!((sum_r0 - 1.0).abs() < tolerances::CROSS_LANGUAGE);
    assert!((sum_r1 - 1.0).abs() < tolerances::CROSS_LANGUAGE);
}

#[test]
fn cpu_boltzmann_normalizes() {
    let d = cpu();
    let fitnesses = vec![1.0, 2.0, 3.0, 4.0];
    let probs = d.boltzmann(&fitnesses, 1.0);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < tolerances::CROSS_LANGUAGE);
    assert!(probs[3] > probs[0], "higher fitness → higher probability");
}

#[test]
fn cpu_shannon_entropy_uniform() {
    let d = cpu();
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let h = d.shannon_entropy(&p);
    assert!(
        (h - 4.0_f64.ln()).abs() < tolerances::CROSS_LANGUAGE,
        "Shannon entropy mismatch: {h}"
    );
}

#[test]
fn cpu_l2_distance_known() {
    let d = cpu();
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let dist = d.l2_distance(&a, &b);
    assert!(
        (dist - std::f64::consts::SQRT_2).abs() < tolerances::CROSS_LANGUAGE,
        "L2 distance mismatch: {dist}"
    );
}

#[test]
fn cpu_kl_divergence_identical() {
    let d = cpu();
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let kl = d.kl_divergence(&p, &p);
    assert!(kl.abs() < tolerances::EXACT_F64);
}
