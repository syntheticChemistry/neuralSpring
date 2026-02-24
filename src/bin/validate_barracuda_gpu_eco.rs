// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: `eco_dynamics` (Paper 013).
//!
//! Validates that `BarraCUDA` `Tensor` matmul on GPU correctly computes
//! the population–niche distance matrix for multi-niche Gaussian fitness.
//!
//! Evolution path:
//! ```text
//! Python (numpy) → Rust (eco_dynamics::batch_fitness)
//!   → BarraCUDA CPU (barracuda::stats)
//!   → BarraCUDA GPU (Tensor matmul: pop × optima^T)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/eco_dynamics/eco_dynamics.py`
//! Rust baseline: `validate_eco_dynamics`, `validate_barracuda_eco`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_tensor;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{gpu_readback, max_abs_diff_gpu_vs_cpu, ValidationHarness};
use std::sync::Arc;

fn cpu_pop_optima_dot(pop: &[Vec<f64>], optima: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_pop = pop.len();
    let n_niches = optima.len();
    let dim = pop[0].len();

    let mut result = vec![vec![0.0_f64; n_niches]; n_pop];
    for i in 0..n_pop {
        for j in 0..n_niches {
            let mut dot = 0.0;
            for k in 0..dim {
                dot += pop[i][k] * optima[j][k];
            }
            result[i][j] = dot;
        }
    }
    result
}

fn cpu_gram_matrix(pop: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = pop.len();
    let dim = pop[0].len();
    let mut gram = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..dim {
                gram[i][j] += pop[i][k] * pop[j][k];
            }
        }
    }
    gram
}

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };
    let device = gpu.wgpu_device().clone();

    let mut h = ValidationHarness::new("barracuda_gpu_eco");

    validate_pop_optima_matmul(&mut h, &device);
    validate_self_similarity(&mut h, &device);
    validate_ones_ones_t(&mut h, &device);
    validate_non_negative_norms(&mut h, &device);
    validate_symmetry(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

fn validate_pop_optima_matmul(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(42);
    let n_pop = 20_usize;
    let dim = 10_usize;
    let n_niches = 3_usize;

    let pop: Vec<Vec<f64>> = (0..n_pop)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();
    let optima: Vec<Vec<f64>> = (0..n_niches)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_dots = cpu_pop_optima_dot(&pop, &optima);

    let pop_flat: Vec<f32> = pop
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let optima_flat: Vec<f32> = optima
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let pop_t = gpu_tensor!(h, &pop_flat, &[n_pop, dim], device);
    let optima_t = gpu_tensor!(h, &optima_flat, &[n_niches, dim], device);

    let optima_t_t = match optima_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("optima transpose: {e}"), false);
            return;
        }
    };

    let dots_t = match pop_t.matmul(&optima_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("pop × optima^T: {e}"), false);
            return;
        }
    };

    let Some(dots_gpu) = gpu_readback(h, &dots_t) else {
        return;
    };
    let cpu_flat: Vec<f64> = cpu_dots.iter().flat_map(|r| r.iter().copied()).collect();
    let max_diff = max_abs_diff_gpu_vs_cpu(&dots_gpu, &cpu_flat);

    h.check_upper(
        &format!("pop × optima^T: max diff GPU vs CPU ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_self_similarity(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(42);
    let n_pop = 8_usize;
    let dim = 4_usize;

    let pop: Vec<Vec<f64>> = (0..n_pop)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_gram = cpu_gram_matrix(&pop);

    let pop_flat: Vec<f32> = pop
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let pop_t1 = gpu_tensor!(h, &pop_flat, &[n_pop, dim], device);
    let pop_t2 = gpu_tensor!(h, &pop_flat, &[n_pop, dim], device);
    let pop_t2_t = match pop_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("pop transpose: {e}"), false);
            return;
        }
    };

    let gram_t = match pop_t1.matmul(&pop_t2_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("pop × pop^T: {e}"), false);
            return;
        }
    };

    let Some(gram) = gpu_readback(h, &gram_t) else {
        return;
    };
    let mut max_diff = 0.0_f64;
    for i in 0..n_pop {
        for j in 0..n_pop {
            let gpu_val = f64::from(gram[i * n_pop + j]);
            let cpu_val = cpu_gram[i][j];
            max_diff = max_diff.max((gpu_val - cpu_val).abs());
        }
    }

    h.check_upper(
        &format!("pop × pop^T ≈ CPU Gram (n=8, max diff {max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_ones_ones_t(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let n = 20_usize;
    let dim = 10_usize;

    let ones_row: Vec<f32> = vec![1.0; n * dim];
    let ones_col: Vec<f32> = vec![1.0; dim * n];

    let a_t = gpu_tensor!(h, &ones_row, &[n, dim], device);
    let b_t = gpu_tensor!(h, &ones_col, &[dim, n], device);

    let out_t = match a_t.matmul(&b_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("ones × ones^T: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };
    let expected = dim as f32;
    let max_diff: f64 = out
        .iter()
        .map(|&x| (f64::from(x) - f64::from(expected)).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("ones × ones^T = {dim} (max diff {max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_non_negative_norms(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(42);
    let n_pop = 8_usize;
    let dim = 4_usize;

    let pop_flat: Vec<f32> = (0..n_pop * dim).map(|_| rng.uniform() as f32).collect();

    let pop_t1 = gpu_tensor!(h, &pop_flat, &[n_pop, dim], device);
    let pop_t2 = gpu_tensor!(h, &pop_flat, &[n_pop, dim], device);
    let pop_t2_t = match pop_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("pop transpose: {e}"), false);
            return;
        }
    };

    let gram_t = match pop_t1.matmul(&pop_t2_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("pop × pop^T: {e}"), false);
            return;
        }
    };

    let Some(gram) = gpu_readback(h, &gram_t) else {
        return;
    };
    let min_diag = (0..n_pop)
        .map(|i| f64::from(gram[i * n_pop + i]))
        .fold(f64::INFINITY, f64::min);

    h.check_lower(
        &format!("pop × pop^T diagonal (squared norm) non-negative ({min_diag:.2e})"),
        min_diag,
        -1e-6,
    );
}

fn validate_symmetry(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let n_pop = 8_usize;
    let dim = 4_usize;

    let pop: Vec<Vec<f64>> = (0..n_pop)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();
    let pop_flat: Vec<f32> = pop
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let pop_t1 = gpu_tensor!(h, &pop_flat, &[n_pop, dim], device);
    let pop_t2 = gpu_tensor!(h, &pop_flat, &[n_pop, dim], device);
    let pop_t2_t = match pop_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("pop transpose: {e}"), false);
            return;
        }
    };

    let gram_t = match pop_t1.matmul(&pop_t2_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("pop × pop^T: {e}"), false);
            return;
        }
    };

    let Some(gram) = gpu_readback(h, &gram_t) else {
        return;
    };
    let mut max_asym = 0.0_f64;
    for i in 0..n_pop {
        for j in 0..n_pop {
            let g_ij = f64::from(gram[i * n_pop + j]);
            let g_ji = f64::from(gram[j * n_pop + i]);
            max_asym = max_asym.max((g_ij - g_ji).abs());
        }
    }
    h.check_upper(
        &format!("pop × pop^T symmetric (|G_ij - G_ji| ≤ {max_asym:.2e})"),
        max_asym,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let n_pop = 20_usize;
    let dim = 10_usize;
    let n_niches = 3_usize;

    let pop_flat: Vec<f32> = (0..n_pop * dim).map(|_| rng.uniform() as f32).collect();
    let optima_flat: Vec<f32> = (0..n_niches * dim).map(|_| rng.uniform() as f32).collect();

    let pop_t1 = gpu_tensor!(h, &pop_flat, &[n_pop, dim], device);
    let optima_t1 = gpu_tensor!(h, &optima_flat, &[n_niches, dim], device);
    let opt_t1 = match optima_t1.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };
    let out1_t = match pop_t1.matmul(&opt_t1) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul run1: {e}"), false);
            return;
        }
    };
    let Some(out1) = gpu_readback(h, &out1_t) else {
        return;
    };

    let pop_t2 = gpu_tensor!(h, &pop_flat, &[n_pop, dim], device);
    let optima_t2 = gpu_tensor!(h, &optima_flat, &[n_niches, dim], device);
    let opt_t2 = match optima_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };
    let out2_t = match pop_t2.matmul(&opt_t2) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul run2: {e}"), false);
            return;
        }
    };
    let Some(out2) = gpu_readback(h, &out2_t) else {
        return;
    };

    let bit_identical = out1
        .iter()
        .zip(out2.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());

    h.check_bool("determinism: two GPU runs identical", bit_identical);
}
