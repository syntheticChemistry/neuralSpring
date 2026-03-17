// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: directed evolution multi-objective (Paper 014).
//!
//! Validates that `BarraCUDA` `Tensor` matmul on GPU correctly computes
//! multi-objective fitness: fitness = genotype × `objective_weights^T`.
//! Domain: 5 selection algorithms evaluated on multi-objective fitness.
//!
//! ## S-14 workaround
//!
//! All matmul operations use A × B^T (transpose second operand).
//!
//! ## S-15 workaround
//!
//! All data uses `rng.uniform()` ([0, 1)) to avoid matmul hang.
//!
//! ## Provenance
//!
//! CPU baseline: `validate_barracuda_directed`

#![expect(
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    reason = "validation binary"
)]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_tensor;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, gpu_readback, max_abs_diff_gpu_vs_cpu};
use std::sync::Arc;

fn cpu_a_bt(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b.len();
    let depth = a[0].len();
    let mut out = vec![vec![0.0_f64; cols]; rows];
    for row_idx in 0..rows {
        for col_idx in 0..cols {
            for inner_idx in 0..depth {
                out[row_idx][col_idx] += a[row_idx][inner_idx] * b[col_idx][inner_idx];
            }
        }
    }
    out
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
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_gpu_directed");

    validate_multi_objective_fitness(&mut h, &device);
    validate_pareto_rank_ordering(&mut h, &device);
    validate_fitness_finite(&mut h, &device);
    validate_per_objective_agreement(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

fn validate_multi_objective_fitness(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(42);
    let pop = 25_usize;
    let genome_len = 12_usize;
    let n_objectives = 4_usize;

    let genotype: Vec<Vec<f64>> = (0..pop)
        .map(|_| (0..genome_len).map(|_| rng.uniform()).collect())
        .collect();
    let objective_weights: Vec<Vec<f64>> = (0..n_objectives)
        .map(|_| (0..genome_len).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_fitness = cpu_a_bt(&genotype, &objective_weights);

    let gen_flat: Vec<f32> = genotype
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let obj_flat: Vec<f32> = objective_weights
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let gen_t = gpu_tensor!(h, &gen_flat, &[pop, genome_len], device);
    let obj_t = gpu_tensor!(h, &obj_flat, &[n_objectives, genome_len], device);
    let obj_t_t = match obj_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let fit_t = match gen_t.matmul(&obj_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("genotype × objective_weights^T: {e}"), false);
            return;
        }
    };

    let Some(fit_gpu) = gpu_readback(h, &fit_t) else {
        return;
    };

    let cpu_flat: Vec<f64> = cpu_fitness.iter().flat_map(|r| r.iter().copied()).collect();
    let diff = max_abs_diff_gpu_vs_cpu(&fit_gpu, &cpu_flat);

    h.check_upper(
        &format!("multi-objective fitness via matmul: max diff ({diff:.2e})"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_pareto_rank_ordering(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(123);
    let pop = 25_usize;
    let genome_len = 12_usize;
    let n_objectives = 4_usize;

    let genotype: Vec<Vec<f64>> = (0..pop)
        .map(|_| (0..genome_len).map(|_| rng.uniform()).collect())
        .collect();
    let objective_weights: Vec<Vec<f64>> = (0..n_objectives)
        .map(|_| (0..genome_len).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_fitness = cpu_a_bt(&genotype, &objective_weights);
    let cpu_combined: Vec<(usize, f64)> = cpu_fitness
        .iter()
        .enumerate()
        .map(|(i, row)| (i, row.iter().sum()))
        .collect();
    let mut cpu_sorted = cpu_combined;
    cpu_sorted.sort_by(|a, b| f64::total_cmp(&b.1, &a.1));

    let gen_flat: Vec<f32> = genotype
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let obj_flat: Vec<f32> = objective_weights
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let gen_t = gpu_tensor!(h, &gen_flat, &[pop, genome_len], device);
    let obj_t = gpu_tensor!(h, &obj_flat, &[n_objectives, genome_len], device);
    let obj_t_t = match obj_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let fit_t = match gen_t.matmul(&obj_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };

    let Some(fit_gpu) = gpu_readback(h, &fit_t) else {
        return;
    };

    let gpu_combined: Vec<(usize, f64)> = fit_gpu
        .chunks(n_objectives)
        .enumerate()
        .map(|(i, chunk)| (i, chunk.iter().map(|&x| f64::from(x)).sum()))
        .collect();
    let mut gpu_sorted = gpu_combined;
    gpu_sorted.sort_by(|a, b| f64::total_cmp(&b.1, &a.1));

    let order_preserved = cpu_sorted
        .iter()
        .zip(gpu_sorted.iter())
        .all(|(c, g)| c.0 == g.0);

    h.check_bool(
        "Pareto rank ordering: GPU combined-fitness order matches CPU",
        order_preserved,
    );
}

fn validate_fitness_finite(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(99);
    let pop = 25_usize;
    let genome_len = 12_usize;
    let n_objectives = 4_usize;

    let gen_flat: Vec<f32> = (0..pop * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let obj_flat: Vec<f32> = (0..n_objectives * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();

    let gen_t = gpu_tensor!(h, &gen_flat, &[pop, genome_len], device);
    let obj_t = gpu_tensor!(h, &obj_flat, &[n_objectives, genome_len], device);
    let obj_t_t = match obj_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let fit_t = match gen_t.matmul(&obj_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };

    let Some(fit) = gpu_readback(h, &fit_t) else {
        return;
    };

    h.check_bool(
        "fitness finite: all GPU fitness values finite",
        fit.iter().all(|x| x.is_finite()),
    );
}

fn validate_per_objective_agreement(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(77);
    let pop = 25_usize;
    let genome_len = 12_usize;
    let n_objectives = 4_usize;

    let genotype: Vec<Vec<f64>> = (0..pop)
        .map(|_| (0..genome_len).map(|_| rng.uniform()).collect())
        .collect();
    let objective_weights: Vec<Vec<f64>> = (0..n_objectives)
        .map(|_| (0..genome_len).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_fitness = cpu_a_bt(&genotype, &objective_weights);

    let gen_flat: Vec<f32> = genotype
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let obj_flat: Vec<f32> = objective_weights
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let gen_t = gpu_tensor!(h, &gen_flat, &[pop, genome_len], device);
    let obj_t = gpu_tensor!(h, &obj_flat, &[n_objectives, genome_len], device);
    let obj_t_t = match obj_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let fit_t = match gen_t.matmul(&obj_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };

    let Some(fit_gpu) = gpu_readback(h, &fit_t) else {
        return;
    };

    let mut max_per_obj_diff = 0.0_f64;
    for obj in 0..n_objectives {
        for ind in 0..pop {
            let idx = ind * n_objectives + obj;
            let diff = (f64::from(fit_gpu[idx]) - cpu_fitness[ind][obj]).abs();
            max_per_obj_diff = max_per_obj_diff.max(diff);
        }
    }

    h.check_upper(
        &format!("per-objective agreement: max diff ({max_per_obj_diff:.2e})"),
        max_per_obj_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let pop = 25_usize;
    let genome_len = 12_usize;
    let n_objectives = 4_usize;

    let gen_flat: Vec<f32> = (0..pop * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();
    let obj_flat: Vec<f32> = (0..n_objectives * genome_len)
        .map(|_| rng.uniform() as f32)
        .collect();

    let run = |_: u32| -> Option<Vec<f32>> {
        let g = Tensor::from_data(&gen_flat, vec![pop, genome_len], device.clone()).ok()?;
        let o =
            Tensor::from_data(&obj_flat, vec![n_objectives, genome_len], device.clone()).ok()?;
        let ot = o.transpose().ok()?;
        let out = g.matmul(&ot).ok()?;
        out.to_vec().ok()
    };

    let Some(r1) = run(1) else {
        h.check_bool("determinism run1 failed", false);
        return;
    };
    let Some(r2) = run(2) else {
        h.check_bool("determinism run2 failed", false);
        return;
    };

    let identical = r1
        .iter()
        .zip(r2.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    h.check_bool("determinism: two GPU runs bit-identical", identical);
}
