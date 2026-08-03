// SPDX-License-Identifier: AGPL-3.0-or-later

//! Determinism tests: verify that stochastic algorithms with fixed seeds
//! produce bitwise-identical results across runs.
//!
//! Complements `tests/test_determinism.py` (Python side) with Rust-native
//! rerun-identical checks for all seeded algorithms.

#[cfg(feature = "barracuda")]
use crate::anderson_localization;
#[cfg(not(feature = "barracuda"))]
use crate::counterdiabatic::NkLandscape;
#[cfg(feature = "barracuda")]
use crate::counterdiabatic::{NkLandscape, compute_cd_schedule};
use crate::directed_evolution::{lexicase_selection, run_selection_experiment};
use crate::eco_dynamics::{self, MultiNicheLandscape};
use crate::game_theory::{QsConfig, qs_cooperation_model};
use crate::hmm;
use crate::introgression;
use crate::meta_population;
use crate::pangenome_selection;

#[cfg(feature = "barracuda")]
use crate::regulatory_network::{GrnParams, integrate_grn};
use crate::rng::Rng;
use crate::sate_alignment;
#[cfg(feature = "barracuda")]
use crate::signal_integration::{OdeParams, OdeState, integrate_ode};
use crate::spectral_commutativity;
#[cfg(feature = "barracuda")]
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

#[cfg(feature = "barracuda")]
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

#[cfg(feature = "barracuda")]
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

#[test]
fn introgression_deterministic() {
    let hmm = introgression::phylonet_hmm();
    let (states1, obs1) = {
        let mut rng = Rng::new(42);
        introgression::generate_synthetic_loci(200, &hmm, &mut rng)
    };
    let (states2, obs2) = {
        let mut rng = Rng::new(42);
        introgression::generate_synthetic_loci(200, &hmm, &mut rng)
    };
    assert_eq!(
        states1, states2,
        "introgression synthetic states must be bitwise identical"
    );
    assert_eq!(
        obs1, obs2,
        "introgression synthetic observations must be bitwise identical"
    );
}

#[cfg(feature = "barracuda")]
#[test]
#[expect(
    clippy::float_cmp,
    reason = "determinism test: bitwise-identical outputs from identical inputs"
)]
fn regulatory_network_deterministic() {
    let p = GrnParams::default();
    let x0 = [0.5, 0.1, 0.5, 0.1];
    let r1 = integrate_grn(&x0, 0.5, &p, 2000, 0.02);
    let r2 = integrate_grn(&x0, 0.5, &p, 2000, 0.02);
    assert_eq!(r1, r2, "GRN integration must be bitwise identical");
}

#[test]
fn pangenome_selection_deterministic() {
    let env = vec![0, 0, 1, 1];
    let pa1 = {
        let mut rng = Rng::new(42);
        pangenome_selection::generate_pa_matrix(4, 20, 0.3, 0.1, &mut rng, &env)
    };
    let pa2 = {
        let mut rng = Rng::new(42);
        pangenome_selection::generate_pa_matrix(4, 20, 0.3, 0.1, &mut rng, &env)
    };
    assert_eq!(pa1, pa2, "pangenome PA matrix must be bitwise identical");
}

#[test]
fn meta_population_deterministic() {
    let run = || {
        let mut rng = Rng::new(42);
        let anc: Vec<f64> = (0..10).map(|_| rng.beta(2.0, 2.0)).collect();
        let pop =
            meta_population::generate_population(5, 10, &anc, 0.15, 70.0, 65.0, 90.0, 2, &mut rng);
        (anc, pop)
    };
    let (a1, p1) = run();
    let (a2, p2) = run();
    assert_eq!(
        a1, a2,
        "meta_population ancestral frequencies must be bitwise identical"
    );
    assert_eq!(
        p1, p2,
        "meta_population genotypes must be bitwise identical"
    );
}

#[test]
fn sate_alignment_deterministic() {
    let (seqs1, n1, len1) = {
        let mut rng = Rng::new(42);
        sate_alignment::generate_tree_guided_sequences(5, 50, 0.05, &mut rng)
    };
    let (seqs2, n2, len2) = {
        let mut rng = Rng::new(42);
        sate_alignment::generate_tree_guided_sequences(5, 50, 0.05, &mut rng)
    };
    assert_eq!(n1, n2);
    assert_eq!(len1, len2);
    assert_eq!(
        seqs1, seqs2,
        "SATE tree-guided sequences must be bitwise identical"
    );
}

#[cfg(feature = "barracuda")]
#[test]
fn signal_integration_deterministic() {
    let y0 = OdeState {
        cdg: 0.1,
        ai: 0.1,
        vps_t: 0.0,
        biofilm: 0.0,
    };
    let params = OdeParams {
        seed: 42,
        ..OdeParams::default()
    };
    let trace1 = integrate_ode(1.0, 0.1, &y0, &params);
    let trace2 = integrate_ode(1.0, 0.1, &y0, &params);
    assert_eq!(trace1.len(), trace2.len());
    for (a, b) in trace1.iter().zip(trace2.iter()) {
        assert_eq!(a.cdg.to_bits(), b.cdg.to_bits());
        assert_eq!(a.ai.to_bits(), b.ai.to_bits());
        assert_eq!(a.vps_t.to_bits(), b.vps_t.to_bits());
        assert_eq!(a.biofilm.to_bits(), b.biofilm.to_bits());
    }
}

#[test]
fn game_theory_deterministic() {
    let config = QsConfig {
        pop_size: 100,
        n_gen: 50,
        qs_threshold: 0.3,
        cooperation_cost: 0.1,
        cooperation_benefit: 0.3,
        dispersal_bonus: 0.5,
        mutation_rate: 0.02,
        seed: 42,
    };
    let r1 = qs_cooperation_model(&config);
    let r2 = qs_cooperation_model(&config);
    assert_eq!(r1.coop_freq, r2.coop_freq);
    assert_eq!(r1.mean_fitness, r2.mean_fitness);
}

#[test]
fn spectral_commutativity_deterministic() {
    let m1 = {
        let mut rng = Rng::new(42);
        spectral_commutativity::random_matrix(8, &mut rng)
    };
    let m2 = {
        let mut rng = Rng::new(42);
        spectral_commutativity::random_matrix(8, &mut rng)
    };
    assert_eq!(
        m1, m2,
        "spectral_commutativity random_matrix must be bitwise identical"
    );
}

#[cfg(feature = "barracuda")]
#[test]
fn anderson_localization_deterministic() {
    let w_vals = [1.0, 2.0, 3.0];
    let sweep1 = {
        let mut rng = Rng::new(42);
        anderson_localization::disorder_sweep(16, 1.0, &w_vals, &mut rng)
    };
    let sweep2 = {
        let mut rng = Rng::new(42);
        anderson_localization::disorder_sweep(16, 1.0, &w_vals, &mut rng)
    };
    assert_eq!(sweep1.len(), sweep2.len());
    for (a, b) in sweep1.iter().zip(sweep2.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "Anderson disorder sweep IPR must be bitwise identical"
        );
    }
}
