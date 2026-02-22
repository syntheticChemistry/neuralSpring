// SPDX-License-Identifier: AGPL-3.0-or-later

//! Determinism tests: verify that stochastic algorithms with fixed seeds
//! produce bitwise-identical results across runs.
//!
//! Complements `tests/test_determinism.py` (Python side) with Rust-native
//! rerun-identical checks for all seeded algorithms.

use crate::counterdiabatic::{compute_cd_schedule, NkLandscape};
use crate::directed_evolution::{lexicase_selection, run_selection_experiment};
use crate::eco_dynamics::{self, MultiNicheLandscape};
use crate::hmm;
use crate::rng::Rng;
use crate::swarm_robotics;

#[test]
fn rng_deterministic_across_runs() {
    let seq1: Vec<f64> = {
        let mut rng = Rng::new(42);
        (0..1000).map(|_| rng.uniform()).collect()
    };
    let seq2: Vec<f64> = {
        let mut rng = Rng::new(42);
        (0..1000).map(|_| rng.uniform()).collect()
    };
    assert_eq!(seq1, seq2, "RNG sequences must be bitwise identical");
}

#[test]
fn nk_landscape_deterministic() {
    let f1 = NkLandscape::new(6, 2, 42).all_fitnesses();
    let f2 = NkLandscape::new(6, 2, 42).all_fitnesses();
    assert_eq!(f1, f2, "NK landscape fitnesses must be bitwise identical");
}

#[test]
fn cd_schedule_deterministic() {
    let l1 = NkLandscape::new(4, 2, 42);
    let l2 = NkLandscape::new(4, 2, 99);
    let f0 = l1.all_fitnesses();
    let f1 = l2.all_fitnesses();
    let s1 = compute_cd_schedule(&f0, &f1, 50, 1.0);
    let s2 = compute_cd_schedule(&f0, &f1, 50, 1.0);
    assert_eq!(s1, s2, "CD schedule must be bitwise identical");
}

#[test]
fn directed_evolution_deterministic() {
    let r1 = run_selection_experiment(lexicase_selection, 20, 3, 50, 10, 0.03, 42);
    let r2 = run_selection_experiment(lexicase_selection, 20, 3, 50, 10, 0.03, 42);
    assert_eq!(r1.mean_fitness, r2.mean_fitness);
    assert_eq!(r1.pareto_front, r2.pareto_front);
    assert_eq!(r1.diversity, r2.diversity);
}

#[test]
fn hmm_forward_deterministic() {
    let model = hmm::Hmm::new(
        vec![vec![0.7, 0.3], vec![0.4, 0.6]],
        vec![vec![0.5, 0.5], vec![0.1, 0.9]],
        vec![0.6, 0.4],
    );
    let obs = &[0, 1, 0, 1, 0];
    let (_, ll1) = model.forward(obs);
    let (_, ll2) = model.forward(obs);
    assert_eq!(
        ll1.to_bits(),
        ll2.to_bits(),
        "HMM log-likelihood must be bitwise identical"
    );
}

#[test]
fn swarm_evolution_deterministic() {
    let r1 = swarm_robotics::run_evolution_heterogeneous(42);
    let r2 = swarm_robotics::run_evolution_heterogeneous(42);
    assert_eq!(r1.mean_fitness, r2.mean_fitness);
    assert_eq!(r1.diversity, r2.diversity);
}

#[test]
fn eco_dynamics_deterministic() {
    let landscape = MultiNicheLandscape::new(8, 3, 2.0, 42);
    let r1 = eco_dynamics::run_ea(&landscape, 50, 10, 0.05, true, 5, 42);
    let r2 = eco_dynamics::run_ea(&landscape, 50, 10, 0.05, true, 5, 42);
    assert_eq!(r1.mean_fitness, r2.mean_fitness);
    assert_eq!(r1.dominance, r2.dominance);
}
