// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: ecological dynamics EA (Paper 013).
//!
//! Validates `barracuda::stats::variance` for fitness distribution analysis
//! in the multi-niche evolutionary algorithm.
//!
//! Evolution path:
//! ```text
//! Python (numpy.var) → Rust (hand-rolled variance)
//!   → BarraCUDA CPU (barracuda::stats::variance)
//!   → BarraCUDA GPU (stats reduction)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/eco_dynamics/eco_dynamics.py`
//! Rust baseline: `validate_eco_dynamics`

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use neural_spring::eco_dynamics::{run_ea, MultiNicheLandscape};
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_eco");

    validate_fitness_variance(&mut h);
    validate_diversity_metrics(&mut h);
    validate_mean_fitness(&mut h);

    h.finish();
}

fn validate_fitness_variance(h: &mut ValidationHarness) {
    let landscape = MultiNicheLandscape::new(16, 2, 0.15, 42);
    let result = run_ea(&landscape, 100, 50, 0.01, false, 5, 42);

    let late_fitness: Vec<f64> = result.mean_fitness[40..].to_vec();
    let barracuda_var = barracuda::stats::correlation::variance(&late_fitness).unwrap_or(f64::NAN);

    h.check_bool(
        &format!("late fitness variance finite ({barracuda_var:.6})"),
        barracuda_var.is_finite() && barracuda_var >= 0.0,
    );
}

fn validate_diversity_metrics(h: &mut ValidationHarness) {
    let landscape = MultiNicheLandscape::new(16, 2, 0.15, 42);
    let result = run_ea(&landscape, 100, 50, 0.01, false, 5, 42);

    let diversity_var =
        barracuda::stats::correlation::variance(&result.diversity).unwrap_or(f64::NAN);
    let richness_mean: f64 =
        result.richness.iter().map(|&r| r as f64).sum::<f64>() / result.richness.len() as f64;

    h.check_bool(
        "diversity trace in [0,1]",
        result
            .diversity
            .iter()
            .all(|&d| (0.0..=1.0 + 1e-10).contains(&d)),
    );
    h.check_bool(
        &format!("diversity variance finite ({diversity_var:.6})"),
        diversity_var.is_finite(),
    );
    h.check_lower("richness mean positive", richness_mean, 1.0);
}

fn validate_mean_fitness(h: &mut ValidationHarness) {
    let landscape = MultiNicheLandscape::new(16, 2, 0.15, 42);
    let result = run_ea(&landscape, 100, 50, 0.01, false, 5, 42);

    let mean_final = result.mean_fitness.last().copied().unwrap_or(0.0);

    h.check_bool(
        "mean fitness trace finite",
        result.mean_fitness.iter().all(|&f| f.is_finite()),
    );
    h.check_lower(
        &format!("final mean fitness positive ({mean_final:.4})"),
        mean_final,
        0.0,
    );
}
