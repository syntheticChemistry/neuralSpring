// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: swarm robotics controllers (Paper 015).
//!
//! Validates `barracuda::linalg::solve_f64` for neural net weight validation
//! and controller fitness / type diversity.
//!
//! Evolution path:
//! ```text
//! Python (numpy.linalg.solve) → Rust (hand-rolled)
//!   → BarraCUDA CPU (barracuda::linalg::solve_f64)
//!   → BarraCUDA GPU (linsolve.wgsl)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/swarm_robotics/swarm_robotics.py`
//! Rust baseline: `validate_swarm_robotics`

#![allow(clippy::cast_precision_loss, clippy::similar_names)]

use neural_spring::swarm_robotics::{
    create_controller, run_evolution_heterogeneous, run_evolution_homogeneous, shannon_diversity,
    ControllerType,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_swarm");

    validate_controller_weight_solve(&mut h);
    validate_evolution_fitness(&mut h);
    validate_type_diversity(&mut h);

    h.finish();
}

/// Validate that a trivial linear system (from controller param extraction)
/// is solved correctly by barracuda.
fn validate_controller_weight_solve(h: &mut ValidationHarness) {
    let mut rng = neural_spring::rng::Rng::new(42);
    let ctrl = create_controller(ControllerType::NeuralNet, &mut rng);
    let b = &ctrl.params[4..8];

    let a = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let b_vec: Vec<f64> = b.to_vec();

    match barracuda::linalg::solve_f64(&a, &b_vec, 4) {
        Ok(x) => {
            for (i, (&xi, &bi)) in x.iter().zip(b.iter()).enumerate() {
                h.check_abs(
                    &format!("solve(I,b)[{i}] == b[{i}]"),
                    xi,
                    bi,
                    tolerances::CROSS_LANGUAGE,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("solve_f64 identity [ERROR: {e}]"), false);
        }
    }
}

fn validate_evolution_fitness(h: &mut ValidationHarness) {
    let homo = run_evolution_homogeneous(42);
    let hetero = run_evolution_heterogeneous(42);

    let homo_var = barracuda::stats::correlation::variance(&homo.mean_fitness).unwrap_or(f64::NAN);
    let hetero_var =
        barracuda::stats::correlation::variance(&hetero.mean_fitness).unwrap_or(f64::NAN);

    h.check_bool(
        "homogeneous mean_fitness finite",
        homo.mean_fitness.iter().all(|&f| f.is_finite()),
    );
    h.check_bool(
        "heterogeneous mean_fitness finite",
        hetero.mean_fitness.iter().all(|&f| f.is_finite()),
    );
    h.check_bool(
        &format!("homogeneous fitness variance finite ({homo_var:.6})"),
        homo_var.is_finite(),
    );
    h.check_bool(
        &format!("heterogeneous fitness variance finite ({hetero_var:.6})"),
        hetero_var.is_finite(),
    );
}

fn validate_type_diversity(h: &mut ValidationHarness) {
    let types_homo = vec![ControllerType::NeuralNet; 10];
    let types_hetero = vec![
        ControllerType::NeuralNet,
        ControllerType::BehaviorTree,
        ControllerType::RuleBased,
        ControllerType::NeuralNet,
        ControllerType::BehaviorTree,
        ControllerType::RuleBased,
        ControllerType::NeuralNet,
        ControllerType::BehaviorTree,
        ControllerType::RuleBased,
        ControllerType::NeuralNet,
    ];

    let div_homo = shannon_diversity(&types_homo);
    let div_hetero = shannon_diversity(&types_hetero);

    h.check_abs(
        "homogeneous diversity = 0",
        div_homo,
        0.0,
        tolerances::EXACT_F64,
    );
    h.check_lower("heterogeneous diversity > 0", div_hetero, 0.0);
}
