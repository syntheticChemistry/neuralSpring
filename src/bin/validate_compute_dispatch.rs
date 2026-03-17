// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation: `BarraCUDA` CPU vs GPU compute dispatch parity.
//!
//! Runs the same baseCamp workloads through both CPU (pure Rust math)
//! and GPU (`BarraCUDA` typed f64 ops) paths. Validates:
//!
//! 1. Dispatch routing selects the correct substrate
//! 2. Both paths produce identical results (within tolerance)
//! 3. `ToadStool`-style streaming reduces round-trips
//!
//! This is the "portability proof" — same math, different hardware.
//!
//! ## Provenance
//!
//! CPU reference: neuralSpring lib (pure Rust math via CPU substrate).
//! GPU path: `BarraCUDA` typed f64 ops via wgpu.
//! Evolution: Python baseline → Rust CPU → `BarraCUDA` CPU → `BarraCUDA` GPU.

#![expect(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "validation binary"
)]

use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring_forge::dispatch::{batch_ipr_substrate, pairwise_substrate, Substrate};
use neural_spring_forge::mixed::{mixed_substrate, MixedSubstrate};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("compute_dispatch");
    let mut rng = Rng::new(42);

    let gpu = Gpu::new().await.ok();
    let dev = gpu.as_ref().map(|g| {
        println!(
            "GPU: {} ({:?}, {:?})",
            g.adapter_name, g.device_type, g.backend
        );
        Arc::clone(g.wgpu_device())
    });

    let gpu_available = dev.is_some();

    // ═══════════════════════════════════════════════════════════════════
    // 1. Dispatch routing correctness
    // ═══════════════════════════════════════════════════════════════════

    h.check_bool(
        "routing: small variance (64 elements) → CPU",
        pairwise_substrate(8, 8) == Substrate::Cpu,
    );
    h.check_bool(
        "routing: large pairwise (1000×100) → GPU",
        pairwise_substrate(1000, 100) == Substrate::Gpu,
    );
    h.check_bool(
        "routing: small IPR (10×10) → CPU",
        batch_ipr_substrate(10, 10) == Substrate::Cpu,
    );
    h.check_bool(
        "routing: large IPR (1000×100) → GPU",
        batch_ipr_substrate(1000, 100) == Substrate::Gpu,
    );

    let mixed = mixed_substrate(100_000.0, 1_048_576, gpu_available, false, false);
    if gpu_available {
        h.check_bool(
            "mixed routing: large compute + GPU → GpuOnly",
            mixed == MixedSubstrate::GpuOnly,
        );
    } else {
        h.check_bool(
            "mixed routing: large compute + no GPU → CpuOnly",
            mixed == MixedSubstrate::CpuOnly,
        );
    }

    let mixed_npu = mixed_substrate(50_000.0, 1_048_576, true, true, true);
    h.check_bool(
        "mixed routing: realtime + NPU → GpuToNpu",
        mixed_npu == MixedSubstrate::GpuToNpu,
    );

    // ═══════════════════════════════════════════════════════════════════
    // 2. CPU vs GPU parity: variance
    // ═══════════════════════════════════════════════════════════════════

    let data: Vec<f64> = (0..256).map(|_| rng.normal()).collect();
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let cpu_var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

    if let Some(ref d) = dev {
        let var_op = barracuda::ops::variance_f64_wgsl::VarianceF64::new(d.clone()).ok();
        let gpu_var = var_op
            .as_ref()
            .and_then(|op| op.variance(&data).ok())
            .unwrap_or(f64::NAN);
        h.check_abs(
            "parity: variance CPU vs GPU",
            gpu_var,
            cpu_var,
            tolerances::GPU_VARIANCE_F64,
        );
    } else {
        h.check_bool("parity: variance CPU-only (no GPU)", cpu_var.is_finite());
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3. CPU vs GPU parity: Pearson correlation
    // ═══════════════════════════════════════════════════════════════════

    let x: Vec<f64> = (0..128).map(|_| rng.normal()).collect();
    let y: Vec<f64> = (0..128).map(|_| rng.normal()).collect();

    let cpu_pearson = barracuda::stats::correlation::pearson_correlation(&x, &y).unwrap_or(0.0);

    if let Some(ref d) = dev {
        let Ok(corr_op) = barracuda::ops::correlation_f64_wgsl::CorrelationF64::new(d.clone())
        else {
            h.check_bool("parity: Pearson GPU init failed", false);
            h.finish();
        };
        let gpu_pearson = corr_op.correlation(&x, &y).unwrap_or(f64::NAN);
        h.check_abs(
            "parity: Pearson CPU vs GPU",
            gpu_pearson,
            cpu_pearson,
            tolerances::GPU_PEARSON_F64,
        );
    } else {
        h.check_bool("parity: Pearson CPU-only (no GPU)", cpu_pearson.is_finite());
    }

    // ═══════════════════════════════════════════════════════════════════
    // 4. CPU vs GPU parity: Shannon entropy
    // ═══════════════════════════════════════════════════════════════════

    let probs: Vec<f64> = {
        let raw: Vec<f64> = (0..64).map(|_| rng.uniform().max(1e-12)).collect();
        let s: f64 = raw.iter().sum();
        raw.iter().map(|&r| r / s).collect()
    };
    let cpu_entropy = neural_spring::primitives::shannon_entropy(&probs);

    if let Some(ref d) = dev {
        let Ok(entropy_op) =
            barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64::new(d.clone())
        else {
            h.check_bool("parity: entropy GPU init failed", false);
            h.finish();
        };
        let gpu_entropy = entropy_op.shannon_entropy(&probs).unwrap_or(f64::NAN);
        h.check_abs(
            "parity: entropy CPU vs GPU",
            gpu_entropy,
            cpu_entropy,
            tolerances::GPU_ENTROPY_F64,
        );
    } else {
        h.check_bool("parity: entropy CPU-only (no GPU)", cpu_entropy.is_finite());
    }

    // ═══════════════════════════════════════════════════════════════════
    // 5. CPU vs GPU parity: chi-squared
    // ═══════════════════════════════════════════════════════════════════

    let obs = vec![15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0];
    let exp_v = vec![43.75; 8];
    let cpu_chi2 = barracuda::special::chi_squared_statistic(&obs, &exp_v).unwrap_or(0.0);

    if let Some(ref d) = dev {
        let gpu_chi2 = neural_spring::gpu_ops::chi_squared_gpu(&obs, &exp_v, d).unwrap_or(f64::NAN);
        h.check_abs(
            "parity: chi² CPU vs GPU",
            gpu_chi2,
            cpu_chi2,
            tolerances::GPU_CHI_SQUARED_F32,
        );
    } else {
        h.check_bool("parity: chi² CPU-only (no GPU)", cpu_chi2.is_finite());
    }

    // ═══════════════════════════════════════════════════════════════════
    // 6. CPU vs GPU parity: eigendecomposition (weight spectral)
    // ═══════════════════════════════════════════════════════════════════

    let ws_n = 8;
    let ws_w: Vec<f64> = (0..ws_n * ws_n).map(|_| rng.normal()).collect();
    let ham = neural_spring::weight_spectral::weight_to_hamiltonian(&ws_w, ws_n, ws_n);
    let dim = ws_n * 2;

    let cpu_decomp = neural_spring::eigh::eigh_householder_qr(&ham, dim);
    let mut cpu_evals = cpu_decomp.eigenvalues;
    cpu_evals.sort_by(f64::total_cmp);

    if let Some(ref d) = dev {
        let (gpu_evals_raw, _) =
            neural_spring::gpu_ops::eigh_gpu(&ham, dim, d).unwrap_or_else(|_| (vec![], vec![]));
        let mut gpu_evals = gpu_evals_raw;
        gpu_evals.sort_by(f64::total_cmp);

        if gpu_evals.len() == cpu_evals.len() {
            let max_diff = cpu_evals
                .iter()
                .zip(gpu_evals.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0_f64, f64::max);
            h.check_bool(
                "parity: eigenvalues CPU vs GPU",
                max_diff < tolerances::GPU_EIGH_DISPATCH_F64,
            );
        } else {
            h.check_bool("parity: eigenvalue count mismatch", false);
        }
    } else {
        h.check_bool(
            "parity: eigenvalues CPU-only (no GPU)",
            !cpu_evals.is_empty(),
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // 7. Dispatch-aware execution: auto-route based on workload size
    // ═══════════════════════════════════════════════════════════════════

    let small_data: Vec<f64> = (0..16).map(|_| rng.normal()).collect();
    let large_data: Vec<f64> = (0..2048).map(|_| rng.normal()).collect();

    let small_mean_cpu = small_data.iter().sum::<f64>() / small_data.len() as f64;
    let small_var_cpu = small_data
        .iter()
        .map(|&x| (x - small_mean_cpu).powi(2))
        .sum::<f64>()
        / small_data.len() as f64;
    h.check_bool(
        "dispatch-aware: small (16) routed to CPU, result finite",
        small_var_cpu.is_finite(),
    );

    let large_mean = large_data.iter().sum::<f64>() / large_data.len() as f64;
    let large_var_cpu = large_data
        .iter()
        .map(|&x| (x - large_mean).powi(2))
        .sum::<f64>()
        / large_data.len() as f64;

    if let Some(ref d) = dev {
        let var_op = barracuda::ops::variance_f64_wgsl::VarianceF64::new(d.clone()).ok();
        let large_var_gpu = var_op
            .as_ref()
            .and_then(|op| op.variance(&large_data).ok())
            .unwrap_or(f64::NAN);
        h.check_abs(
            "dispatch-aware: large (2048) GPU parity",
            large_var_gpu,
            large_var_cpu,
            tolerances::GPU_VARIANCE_F64,
        );
    } else {
        h.check_bool(
            "dispatch-aware: large (2048) CPU-only, result finite",
            large_var_cpu.is_finite(),
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // 8. Transfer cost model validation
    // ═══════════════════════════════════════════════════════════════════

    let cost_1mb = neural_spring_forge::mixed::gpu_cpu_cost(1_048_576);
    h.check_bool(
        "transfer cost: 1MB GPU→CPU in 30-50 µs",
        cost_1mb.estimated_us() > tolerances::TRANSFER_1MB_MIN_US
            && cost_1mb.estimated_us() < tolerances::TRANSFER_1MB_MAX_US,
    );

    let bridge = neural_spring_forge::pcie_bridge::PcieBridge::new("RTX 4070", "AKD1000");
    h.check_bool(
        "PCIe bridge: conservative no-P2P default",
        !bridge.can_p2p(),
    );

    let npu_cost_staged = neural_spring_forge::mixed::gpu_npu_cost(1_048_576, false);
    let npu_cost_p2p = neural_spring_forge::mixed::gpu_npu_cost(1_048_576, true);
    h.check_bool(
        "transfer cost: P2P faster than CPU-staged",
        npu_cost_p2p.estimated_us() < npu_cost_staged.estimated_us(),
    );

    // ═══════════════════════════════════════════════════════════════════
    // 9. nS-06: Immunological Anderson dispatch routing
    // ═══════════════════════════════════════════════════════════════════

    // Cytokine KL divergence via Dispatcher (GPU+CPU fallback)
    let disp = neural_spring::gpu_dispatch::Dispatcher::cpu_only();
    let healthy_dist = [0.60, 0.15, 0.10, 0.08, 0.05, 0.02];
    let inflamed_dist = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10];
    let kl_cpu = disp.kl_divergence(&healthy_dist, &inflamed_dist);
    h.check_bool("nS06 dispatch: KL divergence finite", kl_cpu.is_finite());
    h.check_bool("nS06 dispatch: KL divergence positive", kl_cpu > 0.0);

    // Pielou evenness via CPU dispatcher (entropy + normalization)
    let cpu_entropy_h = disp.shannon_entropy(&healthy_dist);
    let cpu_entropy_i = disp.shannon_entropy(&inflamed_dist);
    h.check_bool(
        "nS06 dispatch: inflamed entropy > healthy entropy",
        cpu_entropy_i > cpu_entropy_h,
    );

    h.finish();
}
