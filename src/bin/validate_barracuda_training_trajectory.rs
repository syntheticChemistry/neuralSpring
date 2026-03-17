// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: Exp-050 — Training Trajectory Spectral Analysis.
//!
//! Proves GPU parity for the spectral analysis pipeline:
//! CPU `eigh_householder_qr` → GPU `eigh_gpu`, then IPR/variance on eigenvalues.
//!
//! Papers: Sub-thesis 01 (Weight Hamiltonians), Paper A (ICML 2027).
//!
//! ## Provenance
//!
//! CPU reference: neuralSpring lib (`weight_spectral`, `eigh_householder_qr` Rust CPU math).
//! GPU path: `BarraCUDA` `eigh_gpu` via wgpu.
//! Evolution: Python baseline → Rust CPU → `BarraCUDA` CPU → `BarraCUDA` GPU.

#![expect(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "validation binary"
)]

use neural_spring::anderson_localization::mean_ipr;
use neural_spring::eigh::eigh_householder_qr;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_ops;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{exit_no_gpu, ValidationHarness};
use neural_spring::weight_spectral;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("barracuda_training_trajectory_gpu");

    let gpu = match Gpu::new().await {
        Ok(g) => {
            println!(
                "GPU: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => exit_no_gpu(),
    };
    let dev = Arc::clone(gpu.wgpu_device());
    let mut rng = Rng::new(42);

    // ═══════════════════════════════════════════════════════════════════
    // 1. GPU eigensolve: random Hamiltonian (GOE-like)
    // ═══════════════════════════════════════════════════════════════════

    let m = 32;
    let ham = weight_spectral::weight_to_hamiltonian(
        &(0..16 * 16).map(|_| rng.normal()).collect::<Vec<f64>>(),
        16,
        16,
    );
    let dim = m;

    let cpu_decomp = eigh_householder_qr(&ham, dim);
    let (gpu_evals, gpu_evecs) = gpu_ops::eigh_gpu(&ham, dim, &dev).expect("eigh_gpu");

    let mut cpu_sorted = cpu_decomp.eigenvalues.clone();
    cpu_sorted.sort_by(f64::total_cmp);
    let mut gpu_sorted = gpu_evals;
    gpu_sorted.sort_by(f64::total_cmp);

    let max_eval_diff = cpu_sorted
        .iter()
        .zip(gpu_sorted.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "GPU eigenvalues match CPU (max diff < eigh tol)",
        max_eval_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 2. IPR from GPU eigenvectors vs CPU eigenvectors
    // ═══════════════════════════════════════════════════════════════════

    let cpu_ipr = mean_ipr(&cpu_decomp.eigenvectors, dim);
    let gpu_ipr = mean_ipr(&gpu_evecs, dim);

    h.check_abs(
        "GPU IPR matches CPU IPR (spectral fingerprint)",
        gpu_ipr,
        cpu_ipr,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    h.check_bool(
        "GPU IPR in delocalized range [0.01, 0.15]",
        (0.01..=0.15).contains(&gpu_ipr),
    );

    // ═══════════════════════════════════════════════════════════════════
    // 3. GPU variance on eigenvalue spectrum
    // ═══════════════════════════════════════════════════════════════════

    let cpu_var = {
        let mean = cpu_sorted.iter().sum::<f64>() / cpu_sorted.len() as f64;
        cpu_sorted.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / cpu_sorted.len() as f64
    };
    let gpu_var = gpu_ops::variance_gpu(&gpu_sorted, &dev).expect("variance_gpu");

    h.check_abs(
        "GPU eigenvalue variance matches CPU",
        gpu_var,
        cpu_var,
        tolerances::GPU_VARIANCE_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 4. Training trajectory: random → structured on GPU
    // ═══════════════════════════════════════════════════════════════════

    let mut rng2 = Rng::new(42);
    let mut random_mat = vec![0.0; m * m];
    for i in 0..m {
        for j in i..m {
            let v = rng2.normal();
            random_mat[i * m + j] = v;
            random_mat[j * m + i] = v;
        }
    }

    let rank = 5;
    let mut low_rank = vec![0.0; m * m];
    let mut rng3 = Rng::new(42);
    for _ in 0..rank {
        let v: Vec<f64> = (0..m).map(|_| rng3.normal()).collect();
        for i in 0..m {
            for j in 0..m {
                low_rank[i * m + j] += v[i] * v[j];
            }
        }
    }

    let (_, evecs_rand) = gpu_ops::eigh_gpu(&random_mat, m, &dev).expect("eigh_gpu random");
    let (_, evecs_struct) = gpu_ops::eigh_gpu(&low_rank, m, &dev).expect("eigh_gpu structured");

    let ipr_rand = mean_ipr(&evecs_rand, m);
    let ipr_struct = mean_ipr(&evecs_struct, m);

    h.check_bool(
        "GPU: structured matrix has higher IPR than random",
        ipr_struct > ipr_rand,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 5. GPU matmul for Hamiltonian construction parity
    // ═══════════════════════════════════════════════════════════════════

    let small_n = 8;
    let a: Vec<f64> = (0..small_n * small_n).map(|_| rng.normal()).collect();
    let b: Vec<f64> = (0..small_n * small_n).map(|_| rng.normal()).collect();

    let cpu_c = {
        let mut c = vec![0.0; small_n * small_n];
        for i in 0..small_n {
            for k in 0..small_n {
                let a_ik = a[i * small_n + k];
                for j in 0..small_n {
                    c[i * small_n + j] += a_ik * b[k * small_n + j];
                }
            }
        }
        c
    };
    let gpu_c = gpu_ops::mat_mul_gpu(&a, &b, small_n, &dev).expect("mat_mul_gpu");

    let matmul_max_diff = cpu_c
        .iter()
        .zip(gpu_c.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "GPU matmul matches CPU (Hamiltonian construction)",
        matmul_max_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 6. GPU mean/entropy for spectral summary statistics
    // ═══════════════════════════════════════════════════════════════════

    let cpu_mean = cpu_sorted.iter().sum::<f64>() / cpu_sorted.len() as f64;
    let gpu_mean = gpu_ops::mean_gpu(&gpu_sorted, &dev).expect("mean_gpu");

    h.check_abs(
        "GPU eigenvalue mean matches CPU",
        gpu_mean,
        cpu_mean,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );

    let probs: Vec<f64> = {
        let evals_abs: Vec<f64> = cpu_sorted
            .iter()
            .map(|v| v.abs().max(tolerances::LOG_ZERO_GUARD))
            .collect();
        let sum: f64 = evals_abs.iter().sum();
        evals_abs.iter().map(|v| v / sum).collect()
    };
    let cpu_entropy = neural_spring::primitives::shannon_entropy(&probs);
    let gpu_entropy = gpu_ops::shannon_entropy_gpu(&probs, &dev).expect("shannon_entropy_gpu");

    h.check_abs(
        "GPU spectral entropy matches CPU",
        gpu_entropy,
        cpu_entropy,
        tolerances::GPU_ENTROPY_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 7. Determinism: same eigensolve twice on GPU
    // ═══════════════════════════════════════════════════════════════════

    let (evals_a, _) = gpu_ops::eigh_gpu(&ham, dim, &dev).expect("eigh_gpu determinism a");
    let (evals_b, _) = gpu_ops::eigh_gpu(&ham, dim, &dev).expect("eigh_gpu determinism b");

    let det_diff = evals_a
        .iter()
        .zip(evals_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "GPU eigensolve is deterministic",
        det_diff < tolerances::NUMERICAL_DISTINCTNESS,
    );

    h.finish();
}
