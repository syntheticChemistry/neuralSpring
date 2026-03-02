// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU workload validation for baseCamp modules.
//!
//! Validates that baseCamp science (all 6 sub-theses) produces correct
//! results when dispatched entirely through `BarraCUDA` GPU typed ops.
//! This is the "final mile" — proving the math is hardware-portable.
//!
//! ## Workload stream
//!
//! Each operation runs CPU (reference) → GPU (test) → compare (parity).
//! GPU ops use `BarraCUDA`'s unidirectional streaming dispatch, reducing
//! round-trips vs the old Tensor API.
//!
//! ```text
//! cargo run --release --bin validate_basecamp_gpu
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::similar_names,
    clippy::too_many_lines
)]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_ops;
use neural_spring::primitives::PROBABILITY_FLOOR;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{exit_no_gpu, ValidationHarness};
use neural_spring::weight_spectral;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("basecamp_gpu");

    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
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
    // Sub-01: Weight Spectral → GPU eigensolve + IPR
    // ═══════════════════════════════════════════════════════════════════

    let ws_n = 8;
    let ws_w: Vec<f64> = (0..ws_n * ws_n).map(|_| rng.normal()).collect();
    let ham = weight_spectral::weight_to_hamiltonian(&ws_w, ws_n, ws_n);
    let dim = ws_n * 2;

    let cpu_result = weight_spectral::weight_spectral_analysis(&ws_w, ws_n, ws_n);

    let (gpu_evals, _gpu_evecs) = gpu_ops::eigh_gpu(&ham, dim, &dev).expect("eigh_gpu dispatch");

    let mut sorted_gpu_evals = gpu_evals;
    sorted_gpu_evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    h.check_bool(
        "Sub-01: GPU eigenvalue count matches CPU",
        sorted_gpu_evals.len() == cpu_result.eigenvalues.len(),
    );

    let eval_diffs: Vec<f64> = sorted_gpu_evals
        .iter()
        .zip(cpu_result.eigenvalues.iter())
        .map(|(g, c)| (g - c).abs())
        .collect();
    let max_eval_diff = eval_diffs.iter().copied().fold(0.0_f64, f64::max);
    h.check_bool(
        "Sub-01: GPU eigenvalues match CPU",
        max_eval_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    // Disorder sweep: batch eigensolve
    let n_disorders = 4;
    let mut batch_hams = Vec::with_capacity(n_disorders * dim * dim);
    for d_idx in 0..n_disorders {
        let scale = f64::from(d_idx as u16).mul_add(0.3, 0.5);
        let disorder_w: Vec<f64> = (0..ws_n * ws_n)
            .map(|j| {
                let mut prng = Rng::new(42 + 100 * d_idx as u64 + j as u64);
                prng.normal() * scale
            })
            .collect();
        let disorder_ham = weight_spectral::weight_to_hamiltonian(&disorder_w, ws_n, ws_n);
        batch_hams.extend_from_slice(&disorder_ham);
    }
    let gpu_iprs = gpu_ops::disorder_sweep_gpu(&batch_hams, dim, n_disorders, &dev)
        .expect("disorder_sweep_gpu dispatch");
    h.check_bool(
        "Sub-01: batch disorder sweep returns correct count",
        gpu_iprs.len() == n_disorders,
    );
    h.check_bool(
        "Sub-01: all batch IPRs finite and positive",
        gpu_iprs.iter().all(|&ipr| ipr.is_finite() && ipr > 0.0),
    );

    // ═══════════════════════════════════════════════════════════════════
    // Sub-02: Information Flow → GPU variance + correlation
    // ═══════════════════════════════════════════════════════════════════

    let signal: Vec<f64> = (0..64).map(|_| rng.normal()).collect();

    let cpu_var = {
        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
        signal.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / signal.len() as f64
    };
    let gpu_var = gpu_ops::variance_gpu(&signal, &dev).expect("variance_gpu dispatch");
    h.check_abs(
        "Sub-02: GPU variance matches CPU (signal propagation)",
        gpu_var,
        cpu_var,
        tolerances::GPU_VARIANCE_F64,
    );

    // Attention correlation via GPU Pearson
    let attn_row1: Vec<f64> = (0..64).map(|_| rng.normal()).collect();
    let attn_row2: Vec<f64> = (0..64).map(|_| rng.normal()).collect();
    let cpu_pearson =
        barracuda::stats::correlation::pearson_correlation(&attn_row1, &attn_row2).unwrap_or(0.0);
    let gpu_pearson = gpu_ops::pearson_correlation_gpu(&attn_row1, &attn_row2, &dev)
        .expect("pearson_gpu dispatch");
    h.check_abs(
        "Sub-02: GPU Pearson matches CPU (attention analysis)",
        gpu_pearson,
        cpu_pearson,
        tolerances::GPU_PEARSON_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Sub-03: Loss Landscape → GPU entropy (spectral entropy proxy)
    // ═══════════════════════════════════════════════════════════════════

    let probs: Vec<f64> = {
        let raw: Vec<f64> = (0..100).map(|_| rng.uniform().max(1e-12)).collect();
        let sum: f64 = raw.iter().sum();
        raw.iter().map(|&r| r / sum).collect()
    };
    let cpu_entropy = neural_spring::primitives::shannon_entropy(&probs);
    let gpu_entropy = gpu_ops::shannon_entropy_gpu(&probs, &dev).expect("entropy_gpu dispatch");
    h.check_abs(
        "Sub-03: GPU entropy matches CPU (landscape spectral entropy)",
        gpu_entropy,
        cpu_entropy,
        tolerances::GPU_ENTROPY_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Sub-04: Neural PGM → GPU matmul for belief propagation
    // ═══════════════════════════════════════════════════════════════════

    let mat_a: Vec<f64> = (0..8 * 8).map(|_| rng.normal()).collect();
    let mat_b: Vec<f64> = (0..8 * 8).map(|_| rng.normal()).collect();

    let cpu_c = {
        let mut c = vec![0.0; 64];
        for i in 0..8 {
            for k in 0..8 {
                let a_ik = mat_a[i * 8 + k];
                for j in 0..8 {
                    c[i * 8 + j] += a_ik * mat_b[k * 8 + j];
                }
            }
        }
        c
    };
    let gpu_c = gpu_ops::mat_mul_gpu(&mat_a, &mat_b, 8, &dev).expect("mat_mul_gpu dispatch");

    let matmul_max_diff = cpu_c
        .iter()
        .zip(gpu_c.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "Sub-04: GPU matmul matches CPU (f32 path)",
        matmul_max_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Sub-05: Agent Coordination → GPU L2 + chi-squared
    // ═══════════════════════════════════════════════════════════════════

    let obs = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let exp_v = vec![37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5];
    let cpu_chi2 = barracuda::special::chi_squared_statistic(&obs, &exp_v).unwrap_or(0.0);
    let gpu_chi2 = gpu_ops::chi_squared_gpu(&obs, &exp_v, &dev).expect("chi_squared_gpu dispatch");
    h.check_abs(
        "Sub-05: GPU chi² matches CPU (agent distribution)",
        gpu_chi2,
        cpu_chi2,
        tolerances::GPU_KL_DISPATCH_F32,
    );

    // L2 distance for agent position comparisons
    let pos_a: Vec<f64> = (0..32).map(|_| rng.normal()).collect();
    let pos_b: Vec<f64> = (0..32).map(|_| rng.normal()).collect();
    let cpu_l2 = {
        let s: f64 = pos_a
            .iter()
            .zip(pos_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        s.sqrt()
    };
    let gpu_l2 = gpu_ops::l2_distance_gpu(&pos_a, &pos_b, &dev).expect("l2_distance_gpu dispatch");
    h.check_abs(
        "Sub-05: GPU L2 distance matches CPU (agent positions)",
        gpu_l2,
        cpu_l2,
        tolerances::GPU_L2_DISPATCH_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Cross-module: GPU mean/sum/max reductions
    // ═══════════════════════════════════════════════════════════════════

    let data: Vec<f64> = (0..256).map(|_| rng.normal()).collect();
    let cpu_mean = data.iter().sum::<f64>() / data.len() as f64;
    let gpu_mean = gpu_ops::mean_gpu(&data, &dev).expect("mean_gpu dispatch");
    h.check_abs(
        "Cross: GPU mean matches CPU",
        gpu_mean,
        cpu_mean,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );

    let cpu_sum: f64 = data.iter().sum();
    let gpu_sum = gpu_ops::sum_gpu(&data, &dev).expect("sum_gpu dispatch");
    h.check_abs(
        "Cross: GPU sum matches CPU",
        gpu_sum,
        cpu_sum,
        tolerances::GPU_SUM_DISPATCH_F32,
    );

    let cpu_max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let gpu_max = gpu_ops::max_gpu(&data, &dev).expect("max_gpu dispatch");
    h.check_abs(
        "Cross: GPU max matches CPU",
        gpu_max,
        cpu_max,
        tolerances::GPU_MAX_DISPATCH_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // KL divergence (PGM comparison metric)
    // ═══════════════════════════════════════════════════════════════════

    let p: Vec<f64> = {
        let raw: Vec<f64> = (0..32)
            .map(|_| rng.uniform().max(PROBABILITY_FLOOR))
            .collect();
        let s: f64 = raw.iter().sum();
        raw.iter().map(|&v| v / s).collect()
    };
    let q: Vec<f64> = {
        let raw: Vec<f64> = (0..32)
            .map(|_| rng.uniform().max(PROBABILITY_FLOOR))
            .collect();
        let s: f64 = raw.iter().sum();
        raw.iter().map(|&v| v / s).collect()
    };

    let cpu_kl: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| if pi > 1e-30 { pi * (pi / qi).ln() } else { 0.0 })
        .sum();
    let gpu_kl = gpu_ops::kl_divergence_gpu(&p, &q, &dev).expect("kl_divergence_gpu dispatch");
    h.check_abs(
        "Cross: GPU KL divergence matches CPU",
        gpu_kl,
        cpu_kl,
        tolerances::GPU_KL_DISPATCH_F32,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Sub-thesis 06: Immunological Anderson — GPU-accelerated primitives
    // ═══════════════════════════════════════════════════════════════════

    // Cytokine distribution KL divergence: healthy vs inflamed dermal
    // cell populations. GPU dispatches the same KL used in Dispatcher.
    let healthy_dermal: Vec<f64> = {
        let raw = [0.60, 0.15, 0.10, 0.08, 0.05, 0.02];
        let s: f64 = raw.iter().sum();
        raw.iter()
            .map(|&v| (v / s).max(PROBABILITY_FLOOR))
            .collect()
    };
    let inflamed_dermal: Vec<f64> = {
        let raw = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10];
        let s: f64 = raw.iter().sum();
        raw.iter()
            .map(|&v| (v / s).max(PROBABILITY_FLOOR))
            .collect()
    };

    let cpu_kl_immuno: f64 = healthy_dermal
        .iter()
        .zip(inflamed_dermal.iter())
        .map(|(&pi, &qi)| if pi > 1e-30 { pi * (pi / qi).ln() } else { 0.0 })
        .sum();
    let gpu_kl_immuno = gpu_ops::kl_divergence_gpu(&healthy_dermal, &inflamed_dermal, &dev)
        .expect("kl_divergence_gpu immuno");
    h.check_abs(
        "nS06: GPU KL cytokine distribution shift",
        gpu_kl_immuno,
        cpu_kl_immuno,
        tolerances::GPU_KL_DISPATCH_F32,
    );

    // Shannon entropy of cell populations for Pielou evenness numerator
    let cpu_h_healthy: f64 = healthy_dermal
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| -v * v.ln())
        .sum();
    let gpu_h_healthy =
        gpu_ops::shannon_entropy_gpu(&healthy_dermal, &dev).expect("shannon_entropy_gpu healthy");
    h.check_abs(
        "nS06: GPU Shannon entropy healthy dermis",
        gpu_h_healthy,
        cpu_h_healthy,
        tolerances::GPU_ENTROPY_F64,
    );

    let cpu_h_inflamed: f64 = inflamed_dermal
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| -v * v.ln())
        .sum();
    let gpu_h_inflamed =
        gpu_ops::shannon_entropy_gpu(&inflamed_dermal, &dev).expect("shannon_entropy_gpu inflamed");
    h.check_abs(
        "nS06: GPU Shannon entropy inflamed dermis",
        gpu_h_inflamed,
        cpu_h_inflamed,
        tolerances::GPU_ENTROPY_F64,
    );

    h.check_bool(
        "nS06: Inflamed H' > healthy H' (more disorder)",
        gpu_h_inflamed > gpu_h_healthy,
    );

    h.finish();
}
