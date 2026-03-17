// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: MODES L2 distance (Paper 012).
//!
//! Validates that `BarraCUDA` `Tensor` matmul on GPU correctly computes
//! pairwise L2 distances between agent feature vectors for the MODES
//! novelty metric. Uses d²(a,b) = ||a||² + ||b||² - 2·a·b^T with matmul
//! for the cross-term.
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
//! Python baseline: `control/modes/modes.py`
//! Rust baseline: `validate_barracuda_modes`

#![expect(
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "validation binary"
)]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_tensor;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, gpu_readback};
use std::sync::Arc;

/// CPU Gram matrix: A × A^T for A [n×dim].
fn cpu_gram(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let dim = a[0].len();
    let mut gram = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..dim {
                gram[i][j] += a[i][k] * a[j][k];
            }
        }
    }
    gram
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
    let mut h = ValidationHarness::new("barracuda_gpu_modes");

    validate_pairwise_l2(&mut h, &device);
    validate_self_distance(&mut h, &device);
    validate_triangle_inequality(&mut h, &device);
    validate_symmetry(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

fn validate_pairwise_l2(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let n_agents = 15_usize;
    let dim = 8_usize;

    let feat: Vec<Vec<f64>> = (0..n_agents)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_gram = cpu_gram(&feat);

    let feat_flat: Vec<f32> = feat
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let feat_t = gpu_tensor!(h, &feat_flat, &[n_agents, dim], device);
    let feat_t2 = gpu_tensor!(h, &feat_flat, &[n_agents, dim], device);
    let feat_t2_t = match feat_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let gram_t = match feat_t.matmul(&feat_t2_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("feat × feat^T: {e}"), false);
            return;
        }
    };

    let Some(gram_gpu) = gpu_readback(h, &gram_t) else {
        return;
    };

    let mut max_diff = 0.0_f64;
    for i in 0..n_agents {
        for j in 0..n_agents {
            let gpu_val = f64::from(gram_gpu[i * n_agents + j]);
            let cpu_val = cpu_gram[i][j];
            max_diff = max_diff.max((gpu_val - cpu_val).abs());
        }
    }

    h.check_upper(
        &format!("pairwise L2 via matmul: Gram max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_self_distance(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(123);
    let n_agents = 15_usize;
    let dim = 8_usize;

    let feat: Vec<Vec<f64>> = (0..n_agents)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();

    let feat_flat: Vec<f32> = feat
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let feat_t = gpu_tensor!(h, &feat_flat, &[n_agents, dim], device);
    let feat_t2 = gpu_tensor!(h, &feat_flat, &[n_agents, dim], device);
    let feat_t2_t = match feat_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let gram_t = match feat_t.matmul(&feat_t2_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("feat × feat^T: {e}"), false);
            return;
        }
    };

    let Some(gram_gpu) = gpu_readback(h, &gram_t) else {
        return;
    };

    let mut max_diag_d2 = 0.0_f64;
    for i in 0..n_agents {
        let g_ii = f64::from(gram_gpu[i * n_agents + i]);
        let d2_ii = (-2.0_f64).mul_add(g_ii, g_ii + g_ii);
        max_diag_d2 = max_diag_d2.max(d2_ii.abs());
    }

    h.check_upper(
        &format!("self-distance (diagonal of d²): max |d²(i,i)| ({max_diag_d2:.2e})"),
        max_diag_d2,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_triangle_inequality(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(99);
    let n_agents = 15_usize;
    let dim = 8_usize;

    let feat: Vec<Vec<f64>> = (0..n_agents)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();

    let feat_flat: Vec<f32> = feat
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let feat_t = gpu_tensor!(h, &feat_flat, &[n_agents, dim], device);
    let feat_t2 = gpu_tensor!(h, &feat_flat, &[n_agents, dim], device);
    let feat_t2_t = match feat_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let gram_t = match feat_t.matmul(&feat_t2_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("feat × feat^T: {e}"), false);
            return;
        }
    };

    let Some(gram_gpu) = gpu_readback(h, &gram_t) else {
        return;
    };

    let mut tri_ok = true;
    for i in 0..n_agents {
        for j in 0..n_agents {
            for k in 0..n_agents {
                let g_ii = f64::from(gram_gpu[i * n_agents + i]);
                let g_jj = f64::from(gram_gpu[j * n_agents + j]);
                let g_kk = f64::from(gram_gpu[k * n_agents + k]);
                let g_ij = f64::from(gram_gpu[i * n_agents + j]);
                let g_ik = f64::from(gram_gpu[i * n_agents + k]);
                let g_jk = f64::from(gram_gpu[j * n_agents + k]);
                let d_ij_sq = (-2.0_f64).mul_add(g_ij, g_ii + g_jj);
                let d_ik_sq = (-2.0_f64).mul_add(g_ik, g_ii + g_kk);
                let d_jk_sq = (-2.0_f64).mul_add(g_jk, g_jj + g_kk);
                let d_ij = d_ij_sq.max(0.0).sqrt();
                let d_ik = d_ik_sq.max(0.0).sqrt();
                let d_jk = d_jk_sq.max(0.0).sqrt();
                if d_ik > d_ij + d_jk + tolerances::GPU_BOUNDS_SLACK_F32 {
                    tri_ok = false;
                }
            }
        }
    }

    h.check_bool("triangle inequality: d(i,k) ≤ d(i,j) + d(j,k)", tri_ok);
}

fn validate_symmetry(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(77);
    let n_agents = 15_usize;
    let dim = 8_usize;

    let feat: Vec<Vec<f64>> = (0..n_agents)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect();

    let feat_flat: Vec<f32> = feat
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let feat_t = gpu_tensor!(h, &feat_flat, &[n_agents, dim], device);
    let feat_t2 = gpu_tensor!(h, &feat_flat, &[n_agents, dim], device);
    let feat_t2_t = match feat_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let gram_t = match feat_t.matmul(&feat_t2_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("feat × feat^T: {e}"), false);
            return;
        }
    };

    let Some(gram) = gpu_readback(h, &gram_t) else {
        return;
    };

    let mut max_asym = 0.0_f64;
    for i in 0..n_agents {
        for j in 0..n_agents {
            let g_ij = f64::from(gram[i * n_agents + j]);
            let g_ji = f64::from(gram[j * n_agents + i]);
            max_asym = max_asym.max((g_ij - g_ji).abs());
        }
    }

    h.check_upper(
        &format!("symmetry: |G_ij - G_ji| ≤ {max_asym:.2e}"),
        max_asym,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let n_agents = 15_usize;
    let dim = 8_usize;

    let feat_flat: Vec<f32> = (0..n_agents * dim).map(|_| rng.uniform() as f32).collect();

    let run = |_: u32| -> Option<Vec<f32>> {
        let f = Tensor::from_data(&feat_flat, vec![n_agents, dim], device.clone()).ok()?;
        let f2 = Tensor::from_data(&feat_flat, vec![n_agents, dim], device.clone()).ok()?;
        let f2t = f2.transpose().ok()?;
        let g = f.matmul(&f2t).ok()?;
        g.to_vec().ok()
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
