// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-spring shader evolution benchmark — exercises fused `BarraCUDA` v0.3.5
//! ops that originated across all five Springs, tracking provenance and timing.
//!
//! ## Cross-Spring Evolution Map
//!
//! ```text
//! hotSpring  (precision/physics) → DF64 core, Welford variance, logsumexp, eigh
//! wetSpring  (bioinformatics)    → Shannon, diversity fusion, Bray-Curtis, HMM
//! neuralSpring (ML/evolution)    → chi-squared, KL divergence, pairwise L2, MHA
//! airSpring  (atmospheric)       → sensor correlation, RMSE, moving window
//! groundSpring (hydrology)       → matrix correlation, multinomial, jackknife
//! ```
//!
//! All absorbed into `BarraCUDA` v0.3.5 (845+ WGSL shaders, wgpu 28).
//!
//! ## What This Benchmarks
//!
//! | Op | Primary Spring | Dispatch |
//! |----|---------------|----------|
//! | Fused mean+variance (Welford) | hotSpring | `VarianceF64::mean_variance()` |
//! | Fused correlation (full) | wetSpring+hotSpring | `CorrelationF64::correlation_full()` |
//! | Correlation matrix (p×p) | airSpring+groundSpring | `stats_f64::matrix_correlation()` |
//! | Shannon entropy (fused map-reduce) | wetSpring | `FusedMapReduceF64::shannon_entropy()` |
//! | Chi-squared (fused) | neuralSpring | `FusedChiSquaredGpu::execute()` |
//! | KL divergence (fused) | neuralSpring | `FusedKlDivergenceGpu::execute()` |
//! | `LogSumExp` | hotSpring | `LogSumExp::compute()` |
//! | Diversity fusion (Shannon+Simpson) | wetSpring | `DiversityFusionGpu::compute()` |
//! | Pairwise L2 matrix | neuralSpring | `PairwiseL2Gpu::compute()` |
//! | Batched eigensolve | hotSpring | `BatchedEighGpu::eigh_batch()` |
//!
//! ```text
//! cargo run --release --bin bench_cross_spring_evolution
//! ```
//!
//! # Panics
//!
//! Panics if the tokio runtime cannot be created.

#![expect(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::suboptimal_flops,
    reason = "benchmark binary — mul+add patterns generate test data, not production math"
)]

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::DiversityFusionGpu;
use barracuda::ops::linalg::BatchedEighGpu;
use barracuda::ops::logsumexp::LogSumExp;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_ops;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 20;

fn bench<F: FnMut()>(label: &str, mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    let us_per = start.elapsed().as_micros() as f64 / ITERS as f64;
    eprintln!("    {label}: {us_per:.1}µs/iter");
    us_per
}

fn bench_hotspring_ops(h: &mut ValidationHarness, dev: &Arc<WgpuDevice>, rng: &mut Rng) {
    eprintln!("\n┌──────────────────────────────────────────────────────────┐");
    eprintln!("│  hotSpring: precision physics → BarraCUDA               │");
    eprintln!("│  DF64 core, Welford variance, logsumexp, eigensolve     │");
    eprintln!("└──────────────────────────────────────────────────────────┘");

    // Fused mean+variance (Welford) — 50k f64 elements
    let data: Vec<f64> = (0..50_000)
        .map(|_| rng.next_f64() * 200.0 - 100.0)
        .collect();
    let mv_us = bench("mean_variance_gpu (50k f64, Welford fused)", || {
        let _ = gpu_ops::mean_variance_gpu(&data, dev);
    });
    h.check_bool("bench_mean_variance", mv_us < 50_000.0);

    // Variance alone — same data
    let var_us = bench("variance_gpu (50k f64, Welford)", || {
        let _ = gpu_ops::variance_gpu(&data, dev);
    });
    h.check_bool("bench_variance", var_us < 50_000.0);

    // LogSumExp — 10k f64
    let lse_n = 10_000;
    let lse_data: Vec<f64> = (0..lse_n).map(|_| rng.next_f64() * 100.0 - 50.0).collect();
    let lse_us = bench("LogSumExp (10k f64)", || {
        let t =
            Tensor::from_data_pod(&lse_data, vec![lse_n], dev.clone()).expect("LogSumExp tensor");
        let _ = std::hint::black_box(LogSumExp::new(t).execute().expect("LogSumExp"));
    });
    h.check_bool("bench_logsumexp", lse_us < 50_000.0);

    // Batched eigensolve — 20×16 symmetric matrices
    let n_mat = 20;
    let dim = 16;
    let mut eigh_data = vec![0.0_f64; n_mat * dim * dim];
    for m in 0..n_mat {
        for i in 0..dim {
            for j in i..dim {
                let v = rng.next_f64() * 2.0 - 1.0;
                eigh_data[m * dim * dim + i * dim + j] = v;
                eigh_data[m * dim * dim + j * dim + i] = v;
            }
        }
    }
    let eigh_us = bench("BatchedEighGpu (20×16×16)", || {
        let _ = BatchedEighGpu::execute_single_dispatch(
            dev.clone(),
            &eigh_data,
            dim,
            n_mat,
            100,
            tolerances::JACOBI_GPU_CONVERGENCE,
        );
    });
    h.check_bool("bench_batched_eigh", eigh_us < 200_000.0);
}

fn bench_wetspring_ops(h: &mut ValidationHarness, dev: &Arc<WgpuDevice>, rng: &mut Rng) {
    eprintln!("\n┌──────────────────────────────────────────────────────────┐");
    eprintln!("│  wetSpring: bioinformatics → BarraCUDA                  │");
    eprintln!("│  Shannon entropy, diversity fusion, fused correlation    │");
    eprintln!("└──────────────────────────────────────────────────────────┘");

    // Shannon entropy — 10k probabilities
    let raw: Vec<f64> = (0..10_000)
        .map(|_| rng.next_f64() * 0.999 + 0.001)
        .collect();
    let sum: f64 = raw.iter().sum();
    let probs: Vec<f64> = raw.iter().map(|&x| x / sum).collect();
    let ent_us = bench("shannon_entropy_gpu (10k probs)", || {
        let _ = gpu_ops::shannon_entropy_gpu(&probs, dev);
    });
    h.check_bool("bench_shannon", ent_us < 50_000.0);

    // Fused correlation (full) — 50k pairs
    let x: Vec<f64> = (0..50_000).map(|_| rng.next_f64() * 20.0 - 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&v| v * 2.3 + rng.next_f64() * 0.2 - 0.1)
        .collect();
    let corr_us = bench("correlation_full_gpu (50k pairs)", || {
        let _ = gpu_ops::correlation_full_gpu(&x, &y, dev);
    });
    h.check_bool("bench_correlation_full", corr_us < 50_000.0);

    // Pearson correlation (scalar) — same data
    let pearson_us = bench("pearson_correlation_gpu (50k pairs)", || {
        let _ = gpu_ops::pearson_correlation_gpu(&x, &y, dev);
    });
    h.check_bool("bench_pearson", pearson_us < 50_000.0);

    // Diversity fusion — 32 samples × 200 taxa
    let n_samples = 32;
    let n_taxa = 200;
    let mut div_data = vec![0.0_f64; n_samples * n_taxa];
    for v in &mut div_data {
        *v = rng.next_f64() * 100.0;
    }
    let div_op = DiversityFusionGpu::new(dev.clone()).expect("DiversityFusionGpu::new");
    let div_us = bench("DiversityFusionGpu (32×200 taxa)", || {
        let _ = div_op.compute(&div_data, n_samples, n_taxa);
    });
    h.check_bool("bench_diversity_fusion", div_us < 100_000.0);
}

fn bench_neuralspring_ops(h: &mut ValidationHarness, dev: &Arc<WgpuDevice>, rng: &mut Rng) {
    eprintln!("\n┌──────────────────────────────────────────────────────────┐");
    eprintln!("│  neuralSpring: ML/neuroevolution → BarraCUDA            │");
    eprintln!("│  chi-squared, KL divergence, pairwise L2                │");
    eprintln!("└──────────────────────────────────────────────────────────┘");

    // Chi-squared — 1k bins
    let observed: Vec<f64> = (0..1000).map(|_| rng.next_f64() * 45.0 + 5.0).collect();
    let expected: Vec<f64> = (0..1000).map(|_| rng.next_f64() * 45.0 + 5.0).collect();
    let chi2_us = bench("chi_squared_gpu (1k bins)", || {
        let _ = gpu_ops::chi_squared_gpu(&observed, &expected, dev);
    });
    h.check_bool("bench_chi_squared", chi2_us < 50_000.0);

    // KL divergence — 1k bins
    let kl_us = bench("kl_divergence_gpu (1k bins)", || {
        let _ = gpu_ops::kl_divergence_gpu(&observed, &expected, dev);
    });
    h.check_bool("bench_kl_divergence", kl_us < 50_000.0);

    // Pairwise L2 via neuralSpring wrapper — 100 vectors × 32 dims
    let n_vecs = 100;
    let dims = 32;
    let vecs: Vec<f64> = (0..n_vecs * dims)
        .map(|_| rng.next_f64() * 10.0 - 5.0)
        .collect();
    let pw_us = bench("pairwise_l2_matrix_gpu (100×32)", || {
        let _ = gpu_ops::pairwise_l2_matrix_gpu(&vecs, n_vecs, dims, dev);
    });
    h.check_bool("bench_pairwise_l2", pw_us < 100_000.0);
}

fn bench_cross_spring_ops(h: &mut ValidationHarness, dev: &Arc<WgpuDevice>, rng: &mut Rng) {
    eprintln!("\n┌──────────────────────────────────────────────────────────┐");
    eprintln!("│  airSpring + groundSpring: sensor/hydrology → BarraCUDA │");
    eprintln!("│  correlation matrix (multi-spring convergence)           │");
    eprintln!("└──────────────────────────────────────────────────────────┘");

    // Correlation matrix — 200 samples × 10 features
    let n = 200_u32;
    let p = 10_u32;
    let data: Vec<f64> = (0..(n * p) as usize)
        .map(|_| rng.next_f64() * 20.0 - 10.0)
        .collect();
    let cm_us = bench("correlation_matrix_gpu (200×10 → 10×10)", || {
        let _ = gpu_ops::correlation_matrix_gpu(&data, n, p, dev);
    });
    h.check_bool("bench_correlation_matrix", cm_us < 100_000.0);
}

fn main() {
    eprintln!("╔════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Cross-Spring Shader Evolution Benchmark                                   ║");
    eprintln!("║  BarraCUDA v0.3.5 (wgpu 28) · ToadStool S94b · 845+ f64-canonical WGSL    ║");
    eprintln!("╚════════════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Five Springs → one math engine:");
    eprintln!("    hotSpring:    DF64 core, Welford variance, logsumexp, eigensolve");
    eprintln!("    wetSpring:    Shannon entropy, diversity fusion, fused correlation");
    eprintln!("    neuralSpring: chi-squared, KL divergence, pairwise L2, MHA");
    eprintln!("    airSpring:    sensor correlation, statistical metrics");
    eprintln!("    groundSpring: matrix correlation, multinomial, jackknife");
    eprintln!();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt
        .block_on(async { Gpu::new().await })
        .expect("GPU required for benchmark");

    eprintln!(
        "  GPU: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend
    );
    eprintln!();

    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("cross_spring_evolution");
    let mut rng = Rng::new(42);

    bench_hotspring_ops(&mut h, &device, &mut rng);
    bench_wetspring_ops(&mut h, &device, &mut rng);
    bench_neuralspring_ops(&mut h, &device, &mut rng);
    bench_cross_spring_ops(&mut h, &device, &mut rng);

    eprintln!("\n────────────────────────────────────────────────────────────");
    eprintln!("  Cross-Spring Provenance Summary:");
    eprintln!("    hotSpring     → 4 ops (Welford, logsumexp, eigensolve, variance)");
    eprintln!("    wetSpring     → 4 ops (Shannon, diversity, correlation_full, pearson)");
    eprintln!("    neuralSpring  → 3 ops (chi-squared, KL divergence, pairwise L2)");
    eprintln!("    airSpring     → 1 op  (correlation matrix)");
    eprintln!("    groundSpring  → 1 op  (correlation matrix, shared with airSpring)");
    eprintln!("  Total: 13 benchmarked ops from 5 Springs → BarraCUDA v0.3.5");
    eprintln!("────────────────────────────────────────────────────────────");

    h.finish();
}
