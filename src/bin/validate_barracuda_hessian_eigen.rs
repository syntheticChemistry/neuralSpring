// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: Exp-052 — Hessian Eigenanalysis.
//!
//! Proves GPU parity for loss landscape spectral diagnostics:
//! CPU `eigh_householder_qr` → GPU `eigh_gpu` for Hessian eigensolve.
//!
//! Papers: Sub-thesis 03 (Loss Landscapes), Paper D (Digital Discovery 2027).
//!
//! ## Provenance
//!
//! CPU reference: neuralSpring lib (`eigh_householder_qr` Rust CPU math).
//! GPU path: `BarraCUDA` `eigh_gpu` via wgpu.
//! Evolution: Python baseline → Rust CPU → `BarraCUDA` CPU → `BarraCUDA` GPU.

#![expect(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::too_many_lines,
    reason = "validation binary"
)]

use neural_spring::eigh::eigh_householder_qr;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_ops;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, exit_no_gpu};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("barracuda_hessian_eigen_gpu");

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

    // ═══════════════════════════════════════════════════════════════════
    // 1. Diagonal Hessian: known eigenvalues [1, 2, ..., 20]
    // ═══════════════════════════════════════════════════════════════════

    let n = 20;
    let mut hessian = vec![0.0; n * n];
    let expected: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    for (i, &ev) in expected.iter().enumerate() {
        hessian[i * n + i] = ev;
    }

    let cpu_decomp = eigh_householder_qr(&hessian, n);
    let (gpu_evals, _) = gpu_ops::eigh_gpu(&hessian, n, &dev).expect("eigh_gpu diagonal");

    let mut cpu_sorted = cpu_decomp.eigenvalues;
    cpu_sorted.sort_by(f64::total_cmp);
    let mut gpu_sorted = gpu_evals;
    gpu_sorted.sort_by(f64::total_cmp);

    let max_diff = cpu_sorted
        .iter()
        .zip(gpu_sorted.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "GPU Hessian eigenvalues match CPU (diagonal)",
        max_diff,
        0.0,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    let gpu_vs_exact = gpu_sorted
        .iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "GPU eigenvalues match analytical [1..20]",
        gpu_vs_exact,
        0.0,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 2. Flat vs sharp minimum: GPU correctly discriminates
    // ═══════════════════════════════════════════════════════════════════

    let mut flat = vec![0.0; n * n];
    let mut sharp = vec![0.0; n * n];
    for i in 0..n {
        flat[i * n + i] = 0.01;
        sharp[i * n + i] = 100.0;
    }

    let (flat_evals, _) = gpu_ops::eigh_gpu(&flat, n, &dev).expect("eigh_gpu flat");
    let (sharp_evals, _) = gpu_ops::eigh_gpu(&sharp, n, &dev).expect("eigh_gpu sharp");

    let flat_max = flat_evals.iter().copied().fold(0.0_f64, f64::max);
    let sharp_max = sharp_evals.iter().copied().fold(0.0_f64, f64::max);

    h.check_bool(
        "GPU: sharp max eigenvalue >> flat max eigenvalue",
        sharp_max > flat_max * 100.0,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 3. Trace parity: GPU trace matches CPU trace
    // ═══════════════════════════════════════════════════════════════════

    let cpu_trace: f64 = cpu_sorted.iter().sum();
    let gpu_trace: f64 = gpu_sorted.iter().sum();

    h.check_abs(
        "GPU trace matches CPU trace",
        gpu_trace,
        cpu_trace,
        tolerances::GPU_EIGH_DISPATCH_F64 * n as f64,
    );

    let gpu_sum = gpu_ops::sum_gpu(&gpu_sorted, &dev).expect("sum_gpu eigenvalues");

    h.check_abs(
        "GPU sum_gpu matches eigenvalue trace",
        gpu_sum,
        cpu_trace,
        tolerances::GPU_SUM_DISPATCH_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 4. Mixed Hessian: random diagonal, GPU vs CPU
    // ═══════════════════════════════════════════════════════════════════

    let mut rng = Rng::new(42);
    let mut mixed = vec![0.0; n * n];
    for i in 0..n {
        mixed[i * n + i] = rng.uniform().mul_add(9.0, 1.0);
    }

    let cpu_mixed = eigh_householder_qr(&mixed, n);
    let (gpu_mixed_evals, _) = gpu_ops::eigh_gpu(&mixed, n, &dev).expect("eigh_gpu mixed");

    let mut cpu_mixed_sorted = cpu_mixed.eigenvalues;
    cpu_mixed_sorted.sort_by(f64::total_cmp);
    let mut gpu_mixed_sorted = gpu_mixed_evals;
    gpu_mixed_sorted.sort_by(f64::total_cmp);

    let mixed_max_diff = cpu_mixed_sorted
        .iter()
        .zip(gpu_mixed_sorted.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "GPU mixed Hessian eigenvalues match CPU",
        mixed_max_diff,
        0.0,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    h.check_bool(
        "GPU mixed Hessian: all eigenvalues positive",
        gpu_mixed_sorted.iter().all(|&v| v > 0.0),
    );

    // ═══════════════════════════════════════════════════════════════════
    // 5. GPU variance of eigenvalue spectrum (landscape width)
    // ═══════════════════════════════════════════════════════════════════

    let cpu_var = {
        let mean = cpu_sorted.iter().sum::<f64>() / n as f64;
        cpu_sorted.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64
    };
    let gpu_var = gpu_ops::variance_gpu(&gpu_sorted, &dev).expect("variance_gpu");

    h.check_abs(
        "GPU eigenvalue variance matches CPU (landscape width)",
        gpu_var,
        cpu_var,
        tolerances::GPU_VARIANCE_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 6. Frobenius norm of Hessian (GPU vs CPU)
    // ═══════════════════════════════════════════════════════════════════

    let cpu_frob = hessian.iter().map(|v| v * v).sum::<f64>().sqrt();
    let gpu_frob = gpu_ops::frobenius_norm_gpu(&hessian, &dev).expect("frobenius_norm_gpu");

    h.check_abs(
        "GPU Frobenius norm matches CPU",
        gpu_frob,
        cpu_frob,
        tolerances::GPU_MATMUL_RANDOM_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 7. Determinism
    // ═══════════════════════════════════════════════════════════════════

    let (evals_a, _) = gpu_ops::eigh_gpu(&hessian, n, &dev).expect("determinism a");
    let (evals_b, _) = gpu_ops::eigh_gpu(&hessian, n, &dev).expect("determinism b");

    let det_diff = evals_a
        .iter()
        .zip(evals_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "GPU Hessian eigensolve is deterministic",
        det_diff < tolerances::NUMERICAL_DISTINCTNESS,
    );

    h.finish();
}
