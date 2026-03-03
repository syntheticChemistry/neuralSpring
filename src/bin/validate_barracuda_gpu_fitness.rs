// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: batch fitness evaluation (Papers 011–014).
//!
//! Validates GPU `Tensor::matmul` for evolutionary computation workloads:
//! genotype × weight^T → fitness scores.  Covers counterdiabatic (011),
//! MODES (012), ecological dynamics (013), and directed evolution (014).
//!
//! ## S-14 workaround
//!
//! All matmul operations use a transposed operand (A × B^T) to avoid
//! the Naive matmul hang on RTX 4070 Vulkan.  Data is stored in
//! transposed form and `Tensor::transpose()` restores the mathematical
//! layout before matmul.
//!
//! ## Provenance
//!
//! CPU baselines: `validate_barracuda_counterdiabatic` (7),
//! `validate_barracuda_modes` (7), `validate_barracuda_directed` (7).

#![expect(clippy::cast_possible_truncation, reason = "validation binary")]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_tensor;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{gpu_readback, max_abs_diff_gpu_vs_cpu, ValidationHarness};
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
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_gpu_fitness");

    validate_batch_fitness(&mut h, &device);
    validate_multi_objective(&mut h, &device);
    validate_population_ranking(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

fn validate_batch_fitness(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let pop = 20_usize;
    let dim = 10_usize;
    let n_traits = 3_usize;

    let gen: Vec<Vec<f64>> = (0..pop)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();
    let traits: Vec<Vec<f64>> = (0..n_traits)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_dots = cpu_a_bt(&gen, &traits);

    let gen_flat: Vec<f32> = gen
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let traits_flat: Vec<f32> = traits
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let gen_t = gpu_tensor!(h, &gen_flat, &[pop, dim], device);
    let tr_t = gpu_tensor!(h, &traits_flat, &[n_traits, dim], device);
    let tr_t_t = match tr_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let out_t = match gen_t.matmul(&tr_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };
    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };

    let cpu_flat: Vec<f64> = cpu_dots.iter().flat_map(|r| r.iter().copied()).collect();
    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_flat);
    h.check_upper(
        &format!("batch fitness: max diff ({diff:.2e})"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );

    h.check_bool(
        "all GPU fitness values finite",
        out.iter().all(|x| x.is_finite()),
    );
}

fn validate_multi_objective(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(123);
    let pop = 32_usize;
    let dim = 8_usize;
    let n_obj = 4_usize;

    let gen: Vec<Vec<f64>> = (0..pop)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();
    let obj: Vec<Vec<f64>> = (0..n_obj)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_dots = cpu_a_bt(&gen, &obj);

    let gen_flat: Vec<f32> = gen
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let obj_flat: Vec<f32> = obj
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let gen_t = gpu_tensor!(h, &gen_flat, &[pop, dim], device);
    let obj_t = gpu_tensor!(h, &obj_flat, &[n_obj, dim], device);
    let obj_t_t = match obj_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let out_t = match gen_t.matmul(&obj_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };
    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };

    let cpu_flat: Vec<f64> = cpu_dots.iter().flat_map(|r| r.iter().copied()).collect();
    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_flat);
    h.check_upper(
        &format!("multi-objective: max diff ({diff:.2e})"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );

    h.check_bool(
        &format!("output shape pop×K ({} elements)", out.len()),
        out.len() == pop * n_obj,
    );
}

fn validate_population_ranking(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(99);
    let n_gen = 20_usize;
    let dim = 5_usize;
    let n_rank = 3_usize;

    let met: Vec<Vec<f64>> = (0..n_gen)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();
    let rw: Vec<Vec<f64>> = (0..n_rank)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_dots = cpu_a_bt(&met, &rw);

    let met_flat: Vec<f32> = met
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let rw_flat: Vec<f32> = rw
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let met_t = gpu_tensor!(h, &met_flat, &[n_gen, dim], device);
    let rw_t = gpu_tensor!(h, &rw_flat, &[n_rank, dim], device);
    let rw_t_t = match rw_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let out_t = match met_t.matmul(&rw_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };
    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };

    let cpu_flat: Vec<f64> = cpu_dots.iter().flat_map(|r| r.iter().copied()).collect();
    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_flat);
    h.check_upper(
        &format!("population ranking: max diff ({diff:.2e})"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );

    h.check_bool(
        "all ranking scores finite",
        out.iter().all(|x| x.is_finite()),
    );
}

fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let pop = 20_usize;
    let dim = 10_usize;
    let n_traits = 3_usize;

    let gen_f32: Vec<f32> = (0..pop * dim).map(|_| rng.uniform() as f32).collect();
    let tr_f32: Vec<f32> = (0..n_traits * dim).map(|_| rng.uniform() as f32).collect();

    let run = |_: u32| -> Option<Vec<f32>> {
        let g = Tensor::from_data(&gen_f32, vec![pop, dim], device.clone()).ok()?;
        let t = Tensor::from_data(&tr_f32, vec![n_traits, dim], device.clone()).ok()?;
        let tt = t.transpose().ok()?;
        let out = g.matmul(&tt).ok()?;
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
