// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: counterdiabatic evolution on NK landscapes (Paper 011).
//!
//! Validates `barracuda::stats::variance` for population variance and
//! Boltzmann distribution analysis in the counterdiabatic protocol.
//!
//! Evolution path:
//! ```text
//! Python (numpy.var) → Rust (hand-rolled variance)
//!   → BarraCUDA CPU (barracuda::stats::variance)
//!   → BarraCUDA GPU (reduction.wgsl)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/counterdiabatic/counterdiabatic_evolution.py`
//! Rust baseline: `validate_counterdiabatic`

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::similar_names
)]

use neural_spring::counterdiabatic::{
    boltzmann_distribution, kl_divergence, run_protocol_deterministic, NkLandscape,
};
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_counterdiabatic");

    validate_nk_fitness_variance(&mut h);
    validate_boltzmann_variance(&mut h);
    validate_protocol_kl(&mut h);

    h.finish();
}

fn validate_nk_fitness_variance(h: &mut ValidationHarness) {
    let landscape = NkLandscape::new(8, 2, 42);
    let fitnesses = landscape.all_fitnesses();

    let barracuda_var = barracuda::stats::correlation::variance(&fitnesses).unwrap_or(f64::NAN);
    let handrolled_mean: f64 = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
    let handrolled_var: f64 = fitnesses
        .iter()
        .map(|&f| (f - handrolled_mean).powi(2))
        .sum::<f64>()
        / fitnesses.len() as f64;

    // barracuda uses sample variance (ddof=1), hand-rolled uses population (ddof=0).
    // Ratio ≈ n/(n-1); relax tolerance for statistical definition difference.
    let var_tol = (handrolled_var * 0.01).max(1e-6);
    h.check_abs(
        &format!(
            "fitness variance: barracuda={barracuda_var:.6} vs hand-rolled={handrolled_var:.6}"
        ),
        barracuda_var,
        handrolled_var,
        var_tol,
    );

    h.check_bool(
        "fitness variance finite and non-negative",
        barracuda_var.is_finite() && barracuda_var >= 0.0,
    );
}

fn validate_boltzmann_variance(h: &mut ValidationHarness) {
    let landscape = NkLandscape::new(8, 2, 42);
    let fitnesses = landscape.all_fitnesses();
    let beta = 1.0;

    let p = boltzmann_distribution(&fitnesses, beta);
    let barracuda_var = barracuda::stats::correlation::variance(&p).unwrap_or(f64::NAN);

    let sum: f64 = p.iter().sum();
    h.check_abs("Boltzmann distribution sums to 1", sum, 1.0, 1e-12);

    h.check_bool(
        &format!("Boltzmann variance finite ({barracuda_var:.6})"),
        barracuda_var.is_finite() && barracuda_var >= 0.0,
    );
}

fn validate_protocol_kl(h: &mut ValidationHarness) {
    let landscape = NkLandscape::new(8, 2, 42);
    let f0 = landscape.all_fitnesses();
    let f1: Vec<f64> = f0.iter().rev().copied().collect();

    let schedule: Vec<f64> = (0..50).map(|i| f64::from(i) / 49.0).collect();
    let result = run_protocol_deterministic(&f0, &f1, &schedule);

    h.check_bool(
        &format!(
            "protocol mean_kl len = schedule len ({})",
            result.mean_kl.len()
        ),
        result.mean_kl.len() == schedule.len(),
    );

    let final_kl = result.mean_kl.last().copied().unwrap_or(0.0);
    h.check_bool(
        "final KL divergence finite",
        final_kl.is_finite() && final_kl >= 0.0,
    );

    let kl_self = kl_divergence(&[0.25, 0.25, 0.25, 0.25], &[0.25, 0.25, 0.25, 0.25]);
    h.check_abs("KL(p||p) == 0", kl_self, 0.0, 1e-12);
}
