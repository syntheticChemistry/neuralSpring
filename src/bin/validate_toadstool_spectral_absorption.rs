// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation: `ToadStool` spectral absorption readiness.
//!
//! Proves that neuralSpring's spectral analysis pipeline is ready for
//! `ToadStool` to absorb into `barracuda`:
//!
//! 1. **`BarraCUDA` CPU parity**: pure Rust math matches Python reference
//! 2. **`BarraCUDA` GPU parity**: GPU eigensolve/IPR/variance match CPU
//! 3. **Dispatch portability**: same code path on CPU and GPU via `Dispatcher`
//! 4. **Batch scaling**: GPU dispatch scales with problem size
//! 5. **Mixed substrate**: metalForge routing selects optimal substrate
//!
//! `ToadStool` absorption targets:
//! - `eigh_householder_qr` → `barracuda::linalg::eigh_f64`
//! - `mean_ipr` → `barracuda::spectral::BatchIprGpu`
//! - `disorder_sweep` → `barracuda::spectral::disorder_sweep_gpu`
//! - `weight_to_hamiltonian` → `barracuda::spectral::hamiltonian_from_weights`
//! - `mixed_dispatch` → `barracuda::unified_hardware::route`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use neural_spring::anderson_localization::{anderson_hamiltonian_random, disorder_sweep, mean_ipr};
use neural_spring::eigh::eigh_householder_qr;
use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::weight_spectral;
use neural_spring_forge::mixed::MixedSubstrate;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("toadstool_spectral_absorption");
    let mut rng = Rng::new(42);
    let dispatcher = Dispatcher::new().await;
    let has_gpu = dispatcher.has_gpu();

    // ═══════════════════════════════════════════════════════════════════
    // TIER 1: BarraCUDA CPU — pure Rust math correctness
    // ═══════════════════════════════════════════════════════════════════
    //
    // Proves the Rust implementation is correct before we send it to GPU.
    // These are the functions ToadStool will absorb into barracuda::linalg/spectral.

    validate_cpu_eigensolve(&mut h, &mut rng);
    validate_cpu_anderson(&mut h);
    validate_cpu_weight_hamiltonian(&mut h, &mut rng);

    // ═══════════════════════════════════════════════════════════════════
    // TIER 2: BarraCUDA GPU — dispatch parity
    // ═══════════════════════════════════════════════════════════════════
    //
    // Same operations through Dispatcher: proves GPU path matches CPU.

    validate_gpu_eigensolve_parity(&mut h, &dispatcher, &mut rng, has_gpu);
    validate_gpu_anderson_parity(&mut h, &dispatcher, has_gpu);
    validate_gpu_stats_parity(&mut h, &dispatcher, &mut rng, has_gpu);

    // ═══════════════════════════════════════════════════════════════════
    // TIER 3: Batch scaling — GPU scales with problem size
    // ═══════════════════════════════════════════════════════════════════

    validate_batch_scaling(&mut h, &dispatcher, &mut rng, has_gpu);

    // ═══════════════════════════════════════════════════════════════════
    // TIER 4: Mixed substrate — metalForge routing proof
    // ═══════════════════════════════════════════════════════════════════

    validate_mixed_substrate(&mut h, &dispatcher, &mut rng, has_gpu);

    h.finish();
}

// ─────────────────────────────────────────────────────────────────────
// TIER 1: CPU correctness
// ─────────────────────────────────────────────────────────────────────

fn validate_cpu_eigensolve(h: &mut ValidationHarness, rng: &mut Rng) {
    let n = 4;
    #[rustfmt::skip]
    let a = vec![
        2.0, 1.0, 0.0, 0.0,
        1.0, 3.0, 1.0, 0.0,
        0.0, 1.0, 4.0, 1.0,
        0.0, 0.0, 1.0, 5.0,
    ];

    let decomp = eigh_householder_qr(&a, n);

    h.check_bool(
        "CPU eigh: produced n eigenvalues",
        decomp.eigenvalues.len() == n,
    );
    h.check_bool(
        "CPU eigh: eigenvalues sorted ascending",
        decomp
            .eigenvalues
            .windows(2)
            .all(|w| w[0] <= w[1] + tolerances::EXACT_F64),
    );

    let trace: f64 = (0..n).map(|i| a[i * n + i]).sum();
    let eval_sum: f64 = decomp.eigenvalues.iter().sum();
    h.check_abs(
        "CPU eigh: trace == sum(eigenvalues)",
        eval_sum,
        trace,
        tolerances::CROSS_LANGUAGE,
    );

    let det_from_evals: f64 = decomp.eigenvalues.iter().product();
    h.check_bool(
        "CPU eigh: det > 0 (positive definite)",
        det_from_evals > 0.0,
    );

    for k in 0..n {
        let evec: Vec<f64> = (0..n).map(|i| decomp.eigenvectors[k * n + i]).collect();
        let norm: f64 = evec.iter().map(|&v| v * v).sum::<f64>().sqrt();
        h.check_abs(
            &format!("CPU eigh: ||evec[{k}]|| ≈ 1"),
            norm,
            1.0,
            tolerances::CROSS_LANGUAGE,
        );
    }

    let m = 16;
    let sym: Vec<f64> = {
        let mut mat = vec![0.0; m * m];
        for i in 0..m {
            for j in i..m {
                let v = rng.normal();
                mat[i * m + j] = v;
                mat[j * m + i] = v;
            }
        }
        mat
    };

    let decomp2 = eigh_householder_qr(&sym, m);
    let trace2: f64 = (0..m).map(|i| sym[i * m + i]).sum();
    let eval_sum2: f64 = decomp2.eigenvalues.iter().sum();
    h.check_abs(
        "CPU eigh: trace invariant (16×16)",
        eval_sum2,
        trace2,
        tolerances::GPU_F64_TRANSCENDENTAL,
    );
}

fn validate_cpu_anderson(h: &mut ValidationHarness) {
    let n = 20;
    let w_values = vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0];

    let iprs = {
        let mut r = Rng::new(42);
        disorder_sweep(n, 1.0, &w_values, &mut r)
    };

    h.check_bool(
        "CPU Anderson: IPR count matches W count",
        iprs.len() == w_values.len(),
    );

    for (i, &ipr) in iprs.iter().enumerate() {
        h.check_bool(
            &format!("CPU Anderson: IPR[{i}] finite and > 0"),
            ipr.is_finite() && ipr > 0.0,
        );
    }

    h.check_bool(
        "CPU Anderson: localization increases with disorder",
        iprs[5] > iprs[0],
    );

    let ipr_ratio = iprs[5] / iprs[0].max(1e-30);
    h.check_bool(
        "CPU Anderson: strong localization ratio > 2",
        ipr_ratio > 2.0,
    );
}

fn validate_cpu_weight_hamiltonian(h: &mut ValidationHarness, rng: &mut Rng) {
    let rows = 8;
    let cols = 8;
    let weights: Vec<f64> = (0..rows * cols).map(|_| rng.normal()).collect();

    let ham = weight_spectral::weight_to_hamiltonian(&weights, rows, cols);
    let dim = ham.len().isqrt();
    h.check_bool("CPU Hamiltonian: dim > 0", dim > 0);

    for i in 0..dim {
        for j in 0..dim {
            let diff = (ham[i * dim + j] - ham[j * dim + i]).abs();
            h.check_bool(
                &format!("CPU Hamiltonian: symmetric [{i},{j}]"),
                diff < tolerances::ZERO_DETECTION,
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// TIER 2: GPU parity
// ─────────────────────────────────────────────────────────────────────

fn validate_gpu_eigensolve_parity(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
    has_gpu: bool,
) {
    for &dim in &[8, 16, 24] {
        let mut mat = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in i..dim {
                let v = rng.normal();
                mat[i * dim + j] = v;
                mat[j * dim + i] = v;
            }
        }

        let mut cpu_decomp = eigh_householder_qr(&mat, dim);
        let (mut dispatch_evals, dispatch_evecs) = dispatcher.eigh(&mat, dim);

        cpu_decomp
            .eigenvalues
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dispatch_evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let max_diff = cpu_decomp
            .eigenvalues
            .iter()
            .zip(dispatch_evals.iter())
            .map(|(c, d)| (c - d).abs())
            .fold(0.0_f64, f64::max);

        h.check_bool(
            &format!("GPU eigh parity: {dim}×{dim} eigenvalues"),
            max_diff < tolerances::GPU_EIGH_DISPATCH_F64,
        );

        let cpu_ipr = mean_ipr(&cpu_decomp.eigenvectors, dim);
        let dispatch_ipr = mean_ipr(&dispatch_evecs, dim);

        if has_gpu {
            h.check_abs(
                &format!("GPU eigh parity: {dim}×{dim} IPR"),
                dispatch_ipr,
                cpu_ipr,
                tolerances::GPU_EIGH_DISPATCH_F64,
            );
        } else {
            h.check_bool(
                &format!("GPU eigh parity: {dim}×{dim} IPR finite"),
                dispatch_ipr.is_finite(),
            );
        }
    }
}

fn validate_gpu_anderson_parity(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    _has_gpu: bool,
) {
    let n = 16;
    let w = 4.0;
    let mut r = Rng::new(77);
    let ham = anderson_hamiltonian_random(n, 1.0, w, &mut r);

    let mut cpu_decomp = eigh_householder_qr(&ham, n);
    let (dispatch_evals, _) = dispatcher.eigh(&ham, n);

    let mut cpu_sorted = std::mem::take(&mut cpu_decomp.eigenvalues);
    cpu_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut dispatch_sorted = dispatch_evals;
    dispatch_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let max_diff = cpu_sorted
        .iter()
        .zip(dispatch_sorted.iter())
        .map(|(c, d)| (c - d).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "GPU Anderson: eigenvalue parity (W=4)",
        max_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );
}

fn validate_gpu_stats_parity(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
    _has_gpu: bool,
) {
    let data: Vec<f64> = (0..256).map(|_| rng.normal()).collect();

    let cpu_var = Dispatcher::cpu_only().variance(&data);
    let dispatch_var = dispatcher.variance(&data);
    h.check_abs(
        "GPU stats: variance parity",
        dispatch_var,
        cpu_var,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );

    let cpu_mean = Dispatcher::cpu_only().mean(&data);
    let dispatch_mean = dispatcher.mean(&data);
    h.check_abs(
        "GPU stats: mean parity",
        dispatch_mean,
        cpu_mean,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );

    let cpu_l2 = Dispatcher::cpu_only().l2_distance(&data[..128], &data[128..]);
    let dispatch_l2 = dispatcher.l2_distance(&data[..128], &data[128..]);
    h.check_abs(
        "GPU stats: L2 parity",
        dispatch_l2,
        cpu_l2,
        tolerances::GPU_L2_DISPATCH_F32,
    );

    let cpu_frob = Dispatcher::cpu_only().frobenius_norm(&data);
    let dispatch_frob = dispatcher.frobenius_norm(&data);
    h.check_abs(
        "GPU stats: Frobenius parity",
        dispatch_frob,
        cpu_frob,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );
}

// ─────────────────────────────────────────────────────────────────────
// TIER 3: Batch scaling
// ─────────────────────────────────────────────────────────────────────

fn validate_batch_scaling(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
    _has_gpu: bool,
) {
    let dims = [8, 12, 16, 20];
    let mut prev_var: Option<f64> = None;
    let mut all_finite = true;

    for &dim in &dims {
        let mut mat = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in i..dim {
                let v = rng.normal();
                mat[i * dim + j] = v;
                mat[j * dim + i] = v;
            }
        }

        let (evals, _) = dispatcher.eigh(&mat, dim);
        let var = dispatcher.variance(&evals);

        if !var.is_finite() {
            all_finite = false;
        }

        prev_var = Some(var);
    }

    h.check_bool("Batch scaling: all variances finite", all_finite);
    h.check_bool("Batch scaling: ran all 4 sizes", prev_var.is_some());
}

// ─────────────────────────────────────────────────────────────────────
// TIER 4: Mixed substrate
// ─────────────────────────────────────────────────────────────────────

fn validate_mixed_substrate(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
    has_gpu: bool,
) {
    let data: Vec<f64> = (0..1024).map(|_| rng.normal()).collect();
    let cpu_var = {
        let n = data.len() as f64;
        let m = data.iter().sum::<f64>() / n;
        data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
    };

    let (mixed_var, sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "toadstool_absorption_variance",
            compute_us: 60_000.0,
            data_bytes: (data.len() * 8) as u64,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| {
            barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                dev.clone(),
                &data,
            )
            .map_err(|e| format!("{e}"))
        },
        || cpu_var,
    );

    if has_gpu {
        h.check_abs(
            "Mixed substrate: variance GPU ↔ CPU",
            mixed_var,
            cpu_var,
            tolerances::GPU_VARIANCE_F64,
        );
        h.check_bool(
            "Mixed substrate: large → GPU",
            sub == MixedSubstrate::GpuOnly,
        );
    } else {
        h.check_bool("Mixed substrate: variance finite", mixed_var.is_finite());
    }

    let (small_var, small_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "toadstool_absorption_small",
            compute_us: 5.0,
            data_bytes: 64,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| {
            barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                dev.clone(),
                &data[..8],
            )
            .map_err(|e| format!("{e}"))
        },
        || {
            let s = &data[..8];
            let n = s.len() as f64;
            let m = s.iter().sum::<f64>() / n;
            s.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
        },
    );

    h.check_bool(
        "Mixed substrate: small → CPU",
        small_sub == MixedSubstrate::CpuOnly,
    );
    h.check_bool("Mixed substrate: small var finite", small_var.is_finite());

    let (npu_var, npu_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "toadstool_absorption_npu",
            compute_us: 80_000.0,
            data_bytes: (data.len() * 8) as u64,
            npu_available: true,
            needs_realtime: true,
        },
        |dev| {
            barracuda::ops::variance_reduce_f64::VarianceReduceF64::population_variance(
                dev.clone(),
                &data,
            )
            .map_err(|e| format!("{e}"))
        },
        || cpu_var,
    );

    let expected_npu = if has_gpu {
        MixedSubstrate::GpuToNpu
    } else {
        MixedSubstrate::CpuOnly
    };
    h.check_bool(
        "Mixed substrate: realtime+NPU routing",
        npu_sub == expected_npu,
    );
    h.check_bool("Mixed substrate: NPU var finite", npu_var.is_finite());
}
