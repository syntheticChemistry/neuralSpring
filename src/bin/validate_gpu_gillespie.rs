// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: parallel Gillespie SSA via `barracuda::ops::bio::GillespieGpu`.
//!
//! Validates the upstream GPU Gillespie stochastic simulation algorithm
//! against known analytical properties of simple reaction systems.
//!
//! ## Papers validated
//!
//! - Paper 013: Ecological Dynamics (Dolson & Ofria, 2018)
//! - Paper 020: Regulatory Network (Mhatre et al., 2020)
//!
//! ## Provenance
//!
//! Upstream: `barracuda::ops::bio::gillespie::GillespieGpu`
//! Algorithm: Gillespie (1977) direct method with mass-action propensities.

#![expect(clippy::cast_possible_truncation, reason = "validation binary")]

use barracuda::ops::bio::gillespie::{GillespieConfig, GillespieGpu, GillespieModel};
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

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

    let mut h = ValidationHarness::new("gpu_gillespie");

    validate_simple_decay(&mut h, &gpu);
    validate_conservation(&mut h, &gpu);
    validate_multiple_trajectories(&mut h, &gpu);

    h.finish();
}

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

fn validate_simple_decay(h: &mut ValidationHarness, gpu: &Gpu) {
    // System: A → B with rate k=1.0 (1 reaction, 2 species)
    let rate_k = vec![1.0_f64];
    let stoich_react = vec![1u32, 0]; // reaction consumes 1 A, 0 B
    let stoich_net = vec![-1i32, 1]; // net: -1 A, +1 B
    let n_traj = 4_usize;
    let n_species = 2_usize;
    let initial_states: Vec<f64> = (0..n_traj).flat_map(|_| [100.0_f64, 0.0]).collect();
    let seeds = make_seeds(n_traj);
    let config = GillespieConfig {
        t_max: 10.0,
        max_steps: 10_000,
    };

    let dev = gpu.wgpu_device();
    let ssa = GillespieGpu::new(dev);

    let model = GillespieModel {
        rate_k: &rate_k,
        stoich_react: &stoich_react,
        stoich_net: &stoich_net,
    };
    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ssa.simulate(&model, &initial_states, &seeds, n_traj, &config)
    })) {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            h.check_bool(
                &format!("GillespieGpu::simulate error: {e} (driver skip)"),
                false,
            );
            return;
        }
        Err(_) => {
            h.check_bool(
                "GillespieGpu panicked (f64 shader compilation failure)",
                false,
            );
            return;
        }
    };

    // Check: for each trajectory, A + B == 100 (conservation)
    for t in 0..n_traj {
        let a = result.states[t * n_species];
        let b = result.states[t * n_species + 1];
        let total = a + b;
        h.check_abs(
            &format!("simple_decay trajectory {t}: A+B conservation"),
            total,
            100.0,
            tolerances::TENSOR_EXACT_F32,
        );
    }

    // Check: A < 100 (some decay happened)
    let any_decayed = (0..n_traj).any(|t| result.states[t * n_species] < 100.0);
    h.check_bool("simple_decay: at least one trajectory decayed", any_decayed);

    // Check: times > 0 and <= t_max
    for t in 0..n_traj {
        let tm = result.times[t];
        h.check_bool(&format!("simple_decay trajectory {t}: time > 0"), tm > 0.0);
        h.check_upper(
            &format!("simple_decay trajectory {t}: time <= t_max"),
            tm,
            10.0 + tolerances::TENSOR_EXACT_F32,
        );
    }
}

fn validate_conservation(h: &mut ValidationHarness, gpu: &Gpu) {
    let rate_k = vec![1.0_f64];
    let stoich_react = vec![1u32, 0];
    let stoich_net = vec![-1i32, 1];
    let n_traj = 4_usize;
    let n_species = 2_usize;
    let initial_states: Vec<f64> = (0..n_traj).flat_map(|_| [100.0_f64, 0.0]).collect();
    let seeds = make_seeds(n_traj);
    let config = GillespieConfig {
        t_max: 10.0,
        max_steps: 10_000,
    };

    let dev = gpu.wgpu_device();
    let ssa = GillespieGpu::new(dev);
    let model = GillespieModel {
        rate_k: &rate_k,
        stoich_react: &stoich_react,
        stoich_net: &stoich_net,
    };

    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ssa.simulate(&model, &initial_states, &seeds, n_traj, &config)
    })) {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            h.check_bool(&format!("conservation: simulate failed — {e}"), false);
            return;
        }
        Err(_) => {
            h.check_bool("conservation: simulate panicked", false);
            return;
        }
    };

    for t in 0..n_traj {
        let a = result.states[t * n_species];
        let b = result.states[t * n_species + 1];
        h.check_abs(
            &format!("conservation trajectory {t}: A[t]+B[t] == 100"),
            a + b,
            100.0,
            tolerances::GPU_F64_STATS,
        );
    }
}

fn validate_multiple_trajectories(h: &mut ValidationHarness, gpu: &Gpu) {
    // Shorter t_max so trajectories don't all converge to A≈0 (preserves stochastic variation)
    let rate_k = vec![1.0_f64];
    let stoich_react = vec![1u32, 0];
    let stoich_net = vec![-1i32, 1];
    let n_traj = 16_usize;
    let n_species = 2_usize;
    let initial_states: Vec<f64> = (0..n_traj).flat_map(|_| [100.0_f64, 0.0]).collect();
    let seeds = make_seeds(n_traj);
    let config = GillespieConfig {
        t_max: 2.0, // Short run: E[A(2)] ≈ 13.5, so wide spread of final A
        max_steps: 10_000,
    };

    let dev = gpu.wgpu_device();
    let ssa = GillespieGpu::new(dev);
    let model = GillespieModel {
        rate_k: &rate_k,
        stoich_react: &stoich_react,
        stoich_net: &stoich_net,
    };

    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ssa.simulate(&model, &initial_states, &seeds, n_traj, &config)
    })) {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            h.check_bool(
                &format!("multiple_trajectories: simulate failed — {e}"),
                false,
            );
            return;
        }
        Err(_) => {
            h.check_bool("multiple_trajectories: simulate panicked", false);
            return;
        }
    };

    let final_a: Vec<f64> = (0..n_traj).map(|t| result.states[t * n_species]).collect();
    let final_b: Vec<f64> = (0..n_traj)
        .map(|t| result.states[t * n_species + 1])
        .collect();

    // Stochastic should produce variation: not all final A identical
    let first_a_val = final_a[0];
    let all_same = final_a
        .iter()
        .all(|&a| (a - first_a_val).abs() < tolerances::GPU_F64_STATS);
    h.check_bool("multiple_trajectories: variation in final A", !all_same);

    // All final A >= 0, all final B >= 0
    let all_nonneg_a = final_a.iter().all(|&x| x >= 0.0);
    let all_nonneg_b = final_b.iter().all(|&x| x >= 0.0);
    h.check_bool("multiple_trajectories: all final A >= 0", all_nonneg_a);
    h.check_bool("multiple_trajectories: all final B >= 0", all_nonneg_b);
}
