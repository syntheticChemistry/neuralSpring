// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: directed evolution multi-objective (Paper 014).
//!
//! Validates `barracuda::stats::variance` for multi-objective fitness
//! and Pareto front analysis.
//!
//! Evolution path:
//! ```text
//! Python (numpy.var) → Rust (hand-rolled)
//!   → BarraCUDA CPU (barracuda::stats::variance)
//!   → BarraCUDA GPU (stats reduction)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/directed_evolution/directed_evolution.py`
//! Rust baseline: `validate_directed_evolution`

use neural_spring::directed_evolution::{
    lexicase_selection, multi_objective_fitness, pareto_front_count, run_selection_experiment,
};
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_directed");

    validate_multi_objective_variance(&mut h);
    validate_pareto_front(&mut h);
    validate_experiment_results(&mut h);

    h.finish();
}

fn validate_multi_objective_variance(h: &mut ValidationHarness) {
    let genotype: Vec<f64> = (0..40).map(|i| f64::from(i) / 40.0).collect();
    let fitnesses = multi_objective_fitness(&genotype, 4);

    let barracuda_var = barracuda::stats::correlation::variance(&fitnesses).unwrap_or(f64::NAN);

    h.check_bool(
        "multi_objective_fitness returns 4 values",
        fitnesses.len() == 4,
    );
    h.check_bool(
        &format!("fitness variance finite ({barracuda_var:.6})"),
        barracuda_var.is_finite() && barracuda_var >= 0.0,
    );
}

fn validate_pareto_front(h: &mut ValidationHarness) {
    // Flat row-major: 3 individuals × 2 objectives
    let fits: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
    let n = 3;
    let n_obj = 2;
    let count = pareto_front_count(&fits, n, n_obj);

    let var_per_obj: Vec<f64> = (0..n_obj)
        .map(|j| {
            let col: Vec<f64> = (0..n).map(|i| fits[i * n_obj + j]).collect();
            barracuda::stats::correlation::variance(&col).unwrap_or(f64::NAN)
        })
        .collect();

    h.check_bool(
        &format!("Pareto front count 1-3 ({count})"),
        (1..=3).contains(&count),
    );
    h.check_bool(
        "objectives have finite variance",
        var_per_obj.iter().all(|&v: &f64| v.is_finite()),
    );
}

fn validate_experiment_results(h: &mut ValidationHarness) {
    let result = run_selection_experiment(lexicase_selection, 40, 4, 100, 30, 0.03, 42);

    let mean_var =
        barracuda::stats::correlation::variance(&result.mean_fitness).unwrap_or(f64::NAN);

    h.check_bool(
        "mean_fitness trace length = n_gen",
        result.mean_fitness.len() == 30,
    );
    h.check_bool(
        &format!("mean fitness variance finite ({mean_var:.6})"),
        mean_var.is_finite(),
    );
    h.check_bool(
        "Pareto front count positive",
        !result.pareto_front.is_empty(),
    );
}
