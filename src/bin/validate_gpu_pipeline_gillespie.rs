// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU pipeline validation: `GillespieGpu` (`BarraCUDA`) + CPU mean.
//!
//! Replaces raw `mean_reduce` shader with `GillespieGpu` typed op + CPU mean.
//! Stage 1: GillespieGpu.simulate → `final_states` (f64, per trajectory).
//! Stage 2: CPU extracts relevant values and computes mean.
//!
//! ## Pipeline
//!
//! ```text
//! GillespieGpu.simulate → final_states[n_traj × n_species]
//!   ↓
//! CPU mean (totals or species counts) → scalar
//! ```
//!
//! This validates the Gillespie SSA GPU output; reduction to scalar
//! is done on CPU (no `mean_reduce` shader).
//!
//! ## Papers validated
//!
//! - Paper 013: Ecological Dynamics (Dolson & Ofria, 2018)
//! - Paper 020: Regulatory Network (Mhatre et al., 2020)
//!
//! ## Provenance
//!
//! Typed op: `barracuda::ops::bio::gillespie::GillespieGpu`.
//! Reduction: CPU mean over `final_states`.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "validation binary"
)]

use barracuda::ops::bio::gillespie::{GillespieConfig, GillespieGpu};
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn make_seeds(n_trajectories: usize) -> Vec<u32> {
    let mut seeds = Vec::with_capacity(n_trajectories * 4);
    for t in 0..n_trajectories {
        let mut sm = 42u32.wrapping_add(t as u32 * 1_000_003);
        for _ in 0..4 {
            sm = sm.wrapping_add(0x9e37_79b9);
            let mut z = sm;
            z = (z ^ (z >> 15)).wrapping_mul(0x85eb_ca6b);
            z = (z ^ (z >> 13)).wrapping_mul(0xc2b2_ae35);
            seeds.push(z ^ (z >> 16));
        }
    }
    seeds
}

fn run_gillespie(
    gpu: &Gpu,
    n_traj: usize,
) -> Option<barracuda::ops::bio::gillespie::GillespieResult> {
    let rate_k = vec![1.0_f64];
    let stoich_react = vec![1u32, 0];
    let stoich_net = vec![-1i32, 1];
    let initial_states: Vec<f64> = (0..n_traj).flat_map(|_| [100.0_f64, 0.0]).collect();
    let seeds = make_seeds(n_traj);
    let config = GillespieConfig {
        t_max: 2.0,
        max_steps: 10_000,
    };

    let dev = gpu.wgpu_device();
    let ssa = GillespieGpu::new(dev);

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ssa.simulate(
            &barracuda::ops::bio::gillespie::GillespieModel {
                rate_k: &rate_k,
                stoich_react: &stoich_react,
                stoich_net: &stoich_net,
            },
            &initial_states,
            &seeds,
            n_traj,
            &config,
        )
    })) {
        Ok(Ok(r)) => Some(r),
        _ => None,
    }
}

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            println!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let mut h = ValidationHarness::new("gpu_pipeline_gillespie");

    validate_conservation_reduce(&mut h, &gpu);
    validate_mean_species_a(&mut h, &gpu);
    validate_reduce_determinism(&mut h, &gpu);
    validate_multi_trajectory_reduce(&mut h, &gpu);

    h.finish();
}

/// SSA A → B (conservation): mean of A+B across trajectories == 100.
fn validate_conservation_reduce(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_traj = 4_usize;
    let n_species = 2_usize;

    let Some(result) = run_gillespie(gpu, n_traj) else {
        h.check_bool("conservation reduce: SSA failed (driver skip)", false);
        return;
    };

    let totals: Vec<f64> = (0..n_traj)
        .map(|t| result.states[t * n_species] + result.states[t * n_species + 1])
        .collect();
    let cpu_mean = totals.iter().sum::<f64>() / totals.len() as f64;

    h.check_abs(
        &format!("conservation reduce: mean total={cpu_mean:.2} ≈ 100"),
        cpu_mean,
        100.0,
        tolerances::TENSOR_EXACT_F32,
    );
}

/// Mean of final species-A counts (CPU reduction).
fn validate_mean_species_a(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_traj = 8_usize;
    let n_species = 2_usize;

    let Some(result) = run_gillespie(gpu, n_traj) else {
        h.check_bool("mean species A: SSA failed", false);
        return;
    };

    let final_a: Vec<f64> = (0..n_traj).map(|t| result.states[t * n_species]).collect();
    let cpu_mean = final_a.iter().sum::<f64>() / final_a.len() as f64;

    h.check_bool(
        &format!("mean species A: {cpu_mean:.1} in plausible range [0, 100]"),
        (0.0..=100.0).contains(&cpu_mean),
    );
}

/// Same seed → identical Gillespie output (determinism).
fn validate_reduce_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_traj = 4_usize;

    let r1 = run_gillespie(gpu, n_traj);
    let r2 = run_gillespie(gpu, n_traj);

    match (r1, r2) {
        (Some(a), Some(b)) => {
            let mean_a: f64 = a.states.iter().sum::<f64>() / a.states.len() as f64;
            let mean_b: f64 = b.states.iter().sum::<f64>() / b.states.len() as f64;
            h.check_bool(
                &format!("reduce determinism: run1 mean={mean_a:.6} == run2 mean={mean_b:.6}"),
                (mean_a - mean_b).abs() < f64::EPSILON,
            );
        }
        _ => {
            h.check_bool("reduce determinism: SSA failed", false);
        }
    }
}

/// 16 trajectories; CPU mean of species A.
fn validate_multi_trajectory_reduce(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_traj = 16_usize;
    let n_species = 2_usize;

    let Some(result) = run_gillespie(gpu, n_traj) else {
        h.check_bool("multi-trajectory reduce: SSA failed", false);
        return;
    };

    let final_a: Vec<f64> = (0..n_traj).map(|t| result.states[t * n_species]).collect();
    let cpu_mean = final_a.iter().sum::<f64>() / final_a.len() as f64;

    h.check_bool(
        &format!("multi-trajectory: mean species A={cpu_mean:.2} (16 traj)"),
        cpu_mean.is_finite() && (0.0..=100.0).contains(&cpu_mean),
    );
}
