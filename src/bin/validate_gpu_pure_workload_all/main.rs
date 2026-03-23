// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU workload validation: all Phase 0++ paper domains (011–026).
//!
//! Extends `validate_gpu_pure_workload` (fitness-only) to cover every
//! computational domain. Each domain dispatches its typed `BarraCUDA` GPU
//! op, reads back a scalar summary, and compares against the CPU reference.
//!
//! ## Evolution proof
//!
//! ```text
//! Python baseline (control/)
//!   ↓  cross-language validation (1e-10)
//! Rust CPU (neuralSpring lib)
//!   ↓  BarraCUDA CPU ports (pure Rust math)
//! BarraCUDA CPU (barracuda crate)
//!   ↓  GPU Tensor / WGSL shader dispatch
//! BarraCUDA GPU (this validator) — scalar-only readback
//!   ↓
//! Pure GPU sovereign pipeline (`ToadStool` streaming)
//! ```
//!
//! ## Domains validated
//!
//! 13 domains (Papers 011–026): `BatchFitnessGpu`, `MultiObjFitnessGpu`,
//! `SwarmNnGpu`, `HmmBatchForwardF64`, `SpatialPayoffGpu`, `Rk45AdaptiveGpu`,
//! `HillGateGpu`, `BatchIprGpu`, `PairwiseHammingGpu`, `PairwiseL2Gpu`,
//! `PairwiseJaccardGpu`, `LocusVarianceGpu`, `Tensor::matmul`.
//!
//! ## Provenance
//!
//! Session 74. Cross-spring: hotSpring validation patterns, wetSpring
//! bio-domain ops, all dispatched through typed `BarraCUDA` GPU wrappers.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::suboptimal_flops,
    reason = "validation binary — GPU buffer plumbing with numeric casts across 12 bio-compute domains"
)]

mod determinism;
mod fitness_multiobj;
mod hmm_spatial;
mod locus_lstm;
mod spectral_pairwise;
mod swarm_ode_signal;

use neural_spring::validation::ValidationHarness;
use std::time::Instant;

#[tokio::main]
async fn main() {
    let gpu = neural_spring::validation::gpu_or_exit().await;
    println!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend
    );

    let mut h = ValidationHarness::new("gpu_pure_workload_all");
    let t0 = Instant::now();

    fitness_multiobj::validate_fitness(&mut h, &gpu);
    fitness_multiobj::validate_multi_obj(&mut h, &gpu);
    swarm_ode_signal::validate_swarm_nn(&mut h, &gpu);
    hmm_spatial::validate_hmm(&mut h, &gpu);
    hmm_spatial::validate_spatial_payoff(&mut h, &gpu);
    swarm_ode_signal::validate_rk45_regulatory(&mut h, &gpu);
    swarm_ode_signal::validate_hill_gate_signal(&mut h, &gpu);
    spectral_pairwise::validate_batch_ipr(&mut h, &gpu);
    spectral_pairwise::validate_hamming(&mut h, &gpu);
    spectral_pairwise::validate_l2(&mut h, &gpu);
    spectral_pairwise::validate_jaccard(&mut h, &gpu);
    locus_lstm::validate_locus_variance(&mut h, &gpu);
    locus_lstm::validate_lstm_glucose(&mut h, &gpu);
    determinism::validate_determinism(&mut h, &gpu);

    let elapsed = t0.elapsed();
    println!(
        "\n  total GPU pure-workload time: {:.1}ms (13 domains + determinism)",
        elapsed.as_secs_f64() * 1000.0,
    );

    h.finish();
}
