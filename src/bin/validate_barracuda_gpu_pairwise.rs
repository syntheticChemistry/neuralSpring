// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: pairwise distance matrices (Papers 017, 019, 024–025).
//!
//! Validates GPU `Tensor::matmul` for Gram matrix and cross-distance
//! computations used in `SATé` alignment (Paper 017), game theory (019),
//! pangenome selection (024), and meta-population dynamics (025).
//!
//! ## S-14 workaround
//!
//! All matmul operations use X × X^T via explicit transpose, following
//! the `validate_barracuda_gpu_eco` pattern.
//!
//! ## S-15 workaround
//!
//! `Tensor::matmul` hangs on RTX 4070 Vulkan when input buffers
//! contain negative f32 values.  All test data uses `rng.uniform()`
//! ([0, 1) range) to avoid the hang.  Mathematical correctness of
//! matmul with negative inputs is validated on CPU via the
//! `BarraCUDA` CPU-port validators.
//!
//! ## Provenance
//!
//! CPU baselines: `validate_barracuda_sate` (6), `validate_barracuda_game` (5),
//! `validate_barracuda_pangenome` (8), `validate_barracuda_meta_pop` (8).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_tensor;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{gpu_readback, max_abs_diff_gpu_vs_cpu, ValidationHarness};
use std::sync::Arc;

fn cpu_gram(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let size = data.len();
    let depth = data[0].len();
    let mut gram = vec![vec![0.0_f64; size]; size];
    for row_idx in 0..size {
        for col_idx in 0..size {
            for inner_idx in 0..depth {
                gram[row_idx][col_idx] += data[row_idx][inner_idx] * data[col_idx][inner_idx];
            }
        }
    }
    gram
}

fn cpu_cross(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows_a = a.len();
    let cols_b = b.len();
    let depth = a[0].len();
    let mut out = vec![vec![0.0_f64; cols_b]; rows_a];
    for row_idx in 0..rows_a {
        for col_idx in 0..cols_b {
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
        Err(e) => {
            eprintln!("  SKIP: {e}");
            std::process::exit(0);
        }
    };
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_gpu_pairwise");

    validate_gram_matrix(&mut h, &device);
    validate_cross_distance(&mut h, &device);
    validate_symmetry(&mut h, &device);
    validate_diagonal_norms(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

fn validate_gram_matrix(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let n = 20_usize;
    let d = 8_usize;

    let data: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..d).map(|_| rng.uniform()).collect())
        .collect();
    let cpu = cpu_gram(&data);

    let flat: Vec<f32> = data
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let x_t1 = gpu_tensor!(h, &flat, &[n, d], device);
    let x_t2 = gpu_tensor!(h, &flat, &[n, d], device);
    let x_t2_t = match x_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };
    let gram_t = match x_t1.matmul(&x_t2_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul X×X^T: {e}"), false);
            return;
        }
    };
    let Some(gram) = gpu_readback(h, &gram_t) else {
        return;
    };

    let cpu_flat: Vec<f64> = cpu.iter().flat_map(|r| r.iter().copied()).collect();
    let diff = max_abs_diff_gpu_vs_cpu(&gram, &cpu_flat);
    h.check_upper(
        &format!("Gram X×X^T: max diff ({diff:.2e})"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_cross_distance(
    harness: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(123);
    let n1 = 15_usize;
    let n2 = 10_usize;
    let dim = 6_usize;

    let mat_a: Vec<Vec<f64>> = (0..n1)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();
    let mat_b: Vec<Vec<f64>> = (0..n2)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();

    let cpu = cpu_cross(&mat_a, &mat_b);

    let a_flat: Vec<f32> = mat_a
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let b_flat: Vec<f32> = mat_b
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let a_t = gpu_tensor!(harness, &a_flat, &[n1, dim], device);
    let b_t = gpu_tensor!(harness, &b_flat, &[n2, dim], device);
    let b_t_t = match b_t.transpose() {
        Ok(transposed) => transposed,
        Err(err) => {
            harness.check_bool(&format!("transpose B: {err}"), false);
            return;
        }
    };
    let cross_t = match a_t.matmul(&b_t_t) {
        Ok(result) => result,
        Err(err) => {
            harness.check_bool(&format!("matmul A×B^T: {err}"), false);
            return;
        }
    };
    let Some(cross) = gpu_readback(harness, &cross_t) else {
        return;
    };

    let cpu_flat: Vec<f64> = cpu.iter().flat_map(|r| r.iter().copied()).collect();
    let diff = max_abs_diff_gpu_vs_cpu(&cross, &cpu_flat);
    harness.check_upper(
        &format!("cross-distance A×B^T: max diff ({diff:.2e})"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_symmetry(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(99);
    let n = 12_usize;
    let d = 5_usize;

    let data: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..d).map(|_| rng.uniform()).collect())
        .collect();
    let flat: Vec<f32> = data
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let x_t1 = gpu_tensor!(h, &flat, &[n, d], device);
    let x_t2 = gpu_tensor!(h, &flat, &[n, d], device);
    let x_t2_t = match x_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };
    let gram_t = match x_t1.matmul(&x_t2_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };
    let Some(gram) = gpu_readback(h, &gram_t) else {
        return;
    };

    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let g_ij = f64::from(gram[i * n + j]);
            let g_ji = f64::from(gram[j * n + i]);
            max_asym = max_asym.max((g_ij - g_ji).abs());
        }
    }

    h.check_upper(
        &format!("Gram symmetric: |G[i,j]-G[j,i]| ≤ {max_asym:.2e}"),
        max_asym,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_diagonal_norms(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(77);
    let n = 20_usize;
    let d = 8_usize;

    let data: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..d).map(|_| rng.uniform()).collect())
        .collect();
    let flat: Vec<f32> = data
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let x_t1 = gpu_tensor!(h, &flat, &[n, d], device);
    let x_t2 = gpu_tensor!(h, &flat, &[n, d], device);
    let x_t2_t = match x_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };
    let gram_t = match x_t1.matmul(&x_t2_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };
    let Some(gram) = gpu_readback(h, &gram_t) else {
        return;
    };

    let min_diag = (0..n)
        .map(|i| f64::from(gram[i * n + i]))
        .fold(f64::INFINITY, f64::min);

    h.check_lower(
        &format!("Gram diag non-negative ({min_diag:.2e})"),
        min_diag,
        -tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let n = 20_usize;
    let d = 8_usize;
    let x: Vec<f32> = (0..n * d).map(|_| rng.uniform() as f32).collect();

    let run = || -> Option<Vec<f32>> {
        let t1 = Tensor::from_data(&x, vec![n, d], device.clone()).ok()?;
        let t2 = Tensor::from_data(&x, vec![n, d], device.clone()).ok()?;
        let t2t = t2.transpose().ok()?;
        let out = t1.matmul(&t2t).ok()?;
        out.to_vec().ok()
    };

    let Some(r1) = run() else {
        h.check_bool("determinism run1 failed", false);
        return;
    };
    let Some(r2) = run() else {
        h.check_bool("determinism run2 failed", false);
        return;
    };

    let identical = r1
        .iter()
        .zip(r2.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    h.check_bool("determinism: two GPU runs bit-identical", identical);
}
