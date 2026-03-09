// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU-path bio dispatch tests: Hill gate, multi-objective fitness, swarm NN.

use super::*;
use crate::tolerances;

fn cpu() -> Dispatcher {
    Dispatcher::cpu_only()
}

// ── Game theory ─────────────────────────────────────────────

#[test]
fn cpu_replicator_step_preserves_simplex() {
    let d = cpu();
    let freq = [0.6, 0.4];
    let payoff = [[3.0, 0.0], [5.0, 1.0]];
    let next = d.replicator_step(&freq, &payoff, 0.01);
    let sum: f64 = next.iter().sum();
    assert!(
        (sum - 1.0).abs() < tolerances::EXACT_F64,
        "frequencies sum to 1"
    );
    assert!(next[0] >= 0.0 && next[1] >= 0.0, "non-negative");
}

// ── Regulatory ──────────────────────────────────────────────

#[test]
fn cpu_hill_activation_batch() {
    let d = cpu();
    let result = d.hill_activation_batch(&[0.0, 1.0, 10.0], 1.0, 1.0, 2.0);
    assert_eq!(result.len(), 3);
    assert!(
        (result[0] - 0.0).abs() < tolerances::CROSS_LANGUAGE,
        "hill(0)≈0"
    );
    assert!(
        (result[1] - 0.5).abs() < tolerances::NORM_PPF_TAIL,
        "hill(k)≈Vmax/2"
    );
    assert!(result[2] > 0.9, "hill(10k)≈Vmax");
}

#[test]
fn cpu_hill_activation_batch_single() {
    let d = cpu();
    let x = vec![1.0];
    let result = d.hill_activation_batch(&x, 1.0, 0.5, 2.0);
    assert_eq!(result.len(), 1);
    assert!(result[0] > 0.0 && result[0] <= 1.0);
}

// ── Hill gate ───────────────────────────────────────────────

#[test]
fn cpu_hill_gate_basic() {
    let d = cpu();
    let cfg = crate::gpu_ops::HillGateConfig {
        vmax: 1.0,
        k_a: 0.5,
        k_b: 0.5,
        n_a: 2.0,
        n_b: 2.0,
    };
    let result = d.hill_gate(&[0.0, 0.5, 1.0], &[1.0], &cfg);
    assert_eq!(result.len(), 3, "3 inputs × 1 input = 3 outputs");
    assert!(
        result[0].abs() < tolerances::ZERO_DETECTION,
        "hill(0, _) ≈ 0"
    );
    assert!(result[1] > 0.0, "hill(0.5, 1.0) > 0");
    assert!(result[2] > result[1], "hill(1.0, _) > hill(0.5, _)");
}

#[test]
fn cpu_hill_gate_symmetric_inputs() {
    let d = cpu();
    let cfg = crate::gpu_ops::HillGateConfig {
        vmax: 2.0,
        k_a: 1.0,
        k_b: 1.0,
        n_a: 2.0,
        n_b: 2.0,
    };
    let r1 = d.hill_gate(&[1.0], &[1.0], &cfg);
    assert_eq!(r1.len(), 1);
    assert!(
        (r1[0] - 0.5).abs() < 0.1,
        "at K with n=2, Hill ≈ 0.5, got {}",
        r1[0]
    );
}

#[test]
fn cpu_hill_gate_empty_inputs() {
    let d = cpu();
    let cfg = crate::gpu_ops::HillGateConfig {
        vmax: 1.0,
        k_a: 0.5,
        k_b: 0.5,
        n_a: 2.0,
        n_b: 2.0,
    };
    let result = d.hill_gate(&[], &[1.0], &cfg);
    assert!(result.is_empty(), "empty input_a → empty output");
}

// ── Multi-objective fitness ─────────────────────────────────

#[test]
fn cpu_multi_obj_fitness_basic() {
    let d = cpu();
    let genome_len = 4;
    let n_objectives = 2;
    let genotypes = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let result = d.multi_obj_fitness(&genotypes, 2, genome_len, n_objectives);
    assert_eq!(result.len(), 2 * n_objectives, "2 genotypes × 2 objectives");
    assert!(
        result.iter().all(|v| v.is_finite()),
        "all fitness values finite"
    );
}

#[test]
fn cpu_multi_obj_fitness_single() {
    let d = cpu();
    let genotypes = vec![0.5; 8];
    let result = d.multi_obj_fitness(&genotypes, 1, 8, 3);
    assert_eq!(result.len(), 3, "1 genotype × 3 objectives");
}

// ── Swarm NN ────────────────────────────────────────────────

#[test]
fn cpu_swarm_nn_forward_basic() {
    let d = cpu();
    let dims = crate::gpu_ops::SwarmNnDims {
        n_controllers: 2,
        n_evals: 1,
        input_dim: 1,
        hidden_dim: 4,
        output_dim: 5,
    };
    let weights_per = dims.input_dim * dims.hidden_dim
        + dims.hidden_dim
        + dims.hidden_dim * dims.output_dim
        + dims.output_dim;
    assert_eq!(weights_per, 33, "swarm NN expects 33 params per controller");
    let weights = vec![0.5_f64; dims.n_controllers * weights_per];
    let inputs = vec![0.5_f64; dims.n_controllers * dims.n_evals * dims.input_dim];
    let result = d.swarm_nn_forward(&weights, &inputs, &dims);
    assert_eq!(
        result.len(),
        dims.n_controllers * dims.n_evals,
        "one action per controller per eval"
    );
}
