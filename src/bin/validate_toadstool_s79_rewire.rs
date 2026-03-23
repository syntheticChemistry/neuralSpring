// SPDX-License-Identifier: AGPL-3.0-or-later

//! `ToadStool` S79 rewire validation and cross-spring benchmark.
//!
//! Validates functions rewired during the S79 sync and benchmarks the modern
//! upstream paths against the prior implementations.
//!
//! ## Provenance
//!
//! Cross-spring origin: hotSpring, wetSpring, neuralSpring → `BarraCUDA`/`ToadStool` S76/S79 → neuralSpring.
//! Absorption: `FusedChiSquaredGpu`, `FusedKlDivergenceGpu`, `spectral_bandwidth`, `spectral_condition_number`, `shannon_entropy`.
//! Validation: S79 rewired functions vs prior implementations, spectral/chi-squared/KL/shannon correctness.
//!
//! ```text
//! cargo run --release --bin validate_toadstool_s79_rewire
//! ```

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use barracuda::device::WgpuDevice;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, bench_once};
use neural_spring::weight_spectral;
use std::sync::Arc;

fn validate_spectral(h: &mut ValidationHarness) {
    println!("── spectral_bandwidth (upstream delegate) ──");
    let eigenvalues = vec![-3.0, -1.5, 0.0, 1.0, 4.5];
    let bw = weight_spectral::spectral_bandwidth(&eigenvalues);
    h.check_abs("spectral_bandwidth sorted", bw, 7.5, tolerances::EXACT_F64);

    let unsorted = vec![4.5, -3.0, 0.0, 1.0, -1.5];
    let bw_unsorted = weight_spectral::spectral_bandwidth(&unsorted);
    h.check_abs(
        "spectral_bandwidth unsorted (upstream handles)",
        bw_unsorted,
        7.5,
        tolerances::EXACT_F64,
    );

    let bw_empty = weight_spectral::spectral_bandwidth(&[]);
    h.check_abs(
        "spectral_bandwidth empty",
        bw_empty,
        0.0,
        tolerances::EXACT_F64,
    );

    println!("\n── spectral_condition_number (upstream delegate) ──");
    let evals_cond = vec![1.0, 2.0, 3.0, 4.0];
    let cond = weight_spectral::spectral_condition_number(&evals_cond);
    h.check_abs(
        "condition_number [1,2,3,4]",
        cond,
        4.0,
        tolerances::EXACT_F64,
    );

    let singular = vec![0.0, 0.0, 1.0];
    let cond_singular = weight_spectral::spectral_condition_number(&singular);
    h.check_bool(
        "condition_number singular → inf",
        cond_singular.is_infinite(),
    );

    let cond_empty = weight_spectral::spectral_condition_number(&[]);
    h.check_bool("condition_number empty → inf", cond_empty.is_infinite());
}

fn validate_gpu_chi_squared(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    dev: &Arc<WgpuDevice>,
) {
    println!("\n── chi_squared_gpu (FusedChiSquaredGpu, neuralSpring→ToadStool→back) ──");

    let observed: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];
    let expected: Vec<f64> = vec![12.5, 17.5, 27.5, 42.5];

    let cpu_chi2: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(&o, &e)| (o - e).powi(2) / e)
        .sum();

    let (gpu_chi2, chi2_us) = bench_once("chi2_gpu", || {
        neural_spring::gpu_ops::chi_squared_gpu(&observed, &expected, dev)
    });
    if let Some(v) = h.require("chi_squared_gpu", gpu_chi2) {
        h.check_abs(
            "chi_squared GPU vs CPU",
            v,
            cpu_chi2,
            tolerances::GPU_F64_STATS,
        );
    }

    let large_obs: Vec<f64> = (0..1000).map(|i| (f64::from(i) + 1.0) * 2.0).collect();
    let large_exp: Vec<f64> = (0..1000)
        .map(|i| (f64::from(i) + 1.0).mul_add(1.9, 0.5))
        .collect();

    let (_, chi2_large_us) = bench_once("chi2_gpu 1K", || {
        neural_spring::gpu_ops::chi_squared_gpu(&large_obs, &large_exp, dev)
    });
    println!("  chi2 benchmark: small={chi2_us:.0}µs, large={chi2_large_us:.0}µs");

    let d_chi2 = dispatcher.chi_squared(&observed, &expected);
    h.check_abs(
        "dispatcher chi_squared GPU",
        d_chi2,
        cpu_chi2,
        tolerances::GPU_F64_STATS,
    );
}

fn validate_gpu_kl_divergence(h: &mut ValidationHarness, dev: &Arc<WgpuDevice>) {
    println!("\n── kl_divergence_gpu (FusedKlDivergenceGpu, neuralSpring→ToadStool→back) ──");

    let p: Vec<f64> = vec![0.4, 0.3, 0.2, 0.1];
    let q: Vec<f64> = vec![0.25, 0.25, 0.25, 0.25];

    let cpu_kl: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi > 1e-300 && qi > 1e-300 {
                pi * (pi / qi).ln()
            } else {
                0.0
            }
        })
        .sum();

    let (gpu_kl, kl_us) = bench_once("kl_gpu", || {
        neural_spring::gpu_ops::kl_divergence_gpu(&p, &q, dev)
    });
    if let Some(v) = h.require("kl_divergence_gpu", gpu_kl) {
        h.check_abs(
            "kl_divergence GPU vs CPU",
            v,
            cpu_kl,
            tolerances::GPU_F64_STATS,
        );
    }

    let large_p: Vec<f64> = (1..=1000).map(f64::from).collect();
    let large_q: Vec<f64> = (1..=1000).map(|i| f64::from(i) + 0.5).collect();

    let (_, kl_large_us) = bench_once("kl_gpu 1K", || {
        neural_spring::gpu_ops::kl_divergence_gpu(&large_p, &large_q, dev)
    });
    println!("  kl benchmark: small={kl_us:.0}µs, large={kl_large_us:.0}µs");
}

fn validate_gpu_entropy_variance_pearson(h: &mut ValidationHarness, dev: &Arc<WgpuDevice>) {
    println!("\n── shannon_entropy_gpu (wetSpring bio → hotSpring precision → ToadStool) ──");

    let probs: Vec<f64> = vec![0.25, 0.25, 0.25, 0.25];
    let expected_entropy = 4.0_f64.ln();

    let (gpu_ent, ent_us) = bench_once("entropy_gpu", || {
        neural_spring::gpu_ops::shannon_entropy_gpu(&probs, dev)
    });
    if let Some(v) = h.require("shannon_entropy_gpu", gpu_ent) {
        h.check_abs(
            "shannon entropy uniform",
            v,
            expected_entropy,
            tolerances::GPU_ENTROPY_F64,
        );
    }
    println!("  entropy benchmark: {ent_us:.0}µs");

    println!("\n── variance_gpu (hotSpring Welford → ToadStool VarianceF64) ──");

    let var_data: Vec<f64> = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let cpu_mean = var_data.iter().sum::<f64>() / var_data.len() as f64;
    let cpu_var = var_data
        .iter()
        .map(|&x| (x - cpu_mean).powi(2))
        .sum::<f64>()
        / var_data.len() as f64;

    let (gpu_var, var_us) = bench_once("variance_gpu", || {
        neural_spring::gpu_ops::variance_gpu(&var_data, dev)
    });
    if let Some(v) = h.require("variance_gpu", gpu_var) {
        h.check_abs(
            "variance GPU vs CPU",
            v,
            cpu_var,
            tolerances::GPU_VARIANCE_F64,
        );
    }
    println!("  variance benchmark: {var_us:.0}µs");

    println!("\n── pearson_gpu (wetSpring stats → hotSpring f64 → ToadStool CorrelationF64) ──");

    let x_corr: Vec<f64> = (0..100).map(f64::from).collect();
    let y_corr: Vec<f64> = (0..100).map(|i| f64::from(i).mul_add(2.0, 1.0)).collect();

    let (gpu_r, pearson_us) = bench_once("pearson_gpu", || {
        neural_spring::gpu_ops::pearson_correlation_gpu(&x_corr, &y_corr, dev)
    });
    if let Some(v) = h.require("pearson_correlation_gpu", gpu_r) {
        h.check_abs(
            "pearson r=1.0 for perfect linear",
            v,
            1.0,
            tolerances::GPU_PEARSON_F64,
        );
    }
    println!("  pearson benchmark: {pearson_us:.0}µs");
}

fn validate_fp64_strategy(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    println!("\n── Fp64Strategy (hotSpring hardware detection → ToadStool driver profile) ──");

    let strategy = dispatcher.fp64_strategy();
    println!("  Fp64Strategy: {strategy:?}");
    h.check_bool(
        "fp64 strategy valid (Native|Hybrid|Concurrent)",
        matches!(
            strategy,
            barracuda::device::capabilities::Fp64Strategy::Native
                | barracuda::device::capabilities::Fp64Strategy::Hybrid
                | barracuda::device::capabilities::Fp64Strategy::Concurrent
        ),
    );
}

fn validate_weight_spectral(h: &mut ValidationHarness) {
    println!("\n── Weight spectral analysis (all rewired paths compose) ──");

    let mut rng = neural_spring::rng::Rng::new(42);
    let weights: Vec<f64> = (0..64).map(|_| rng.normal()).collect();

    let (result, spectral_us) = bench_once("weight_spectral 8×8", || {
        weight_spectral::weight_spectral_analysis(&weights, 8, 8)
    });

    h.check_bool("bandwidth > 0", result.bandwidth > 0.0);
    h.check_bool("condition_number > 1", result.condition_number > 1.0);
    h.check_bool("eigenvalues count = 16", result.eigenvalues.len() == 16);
    h.check_bool("mean_ipr finite", result.mean_ipr.is_finite());
    h.check_bool("spectral_entropy > 0", result.spectral_entropy > 0.0);
    h.check_bool(
        "phase is Extended or Critical",
        matches!(
            result.phase,
            weight_spectral::SpectralPhase::Extended | weight_spectral::SpectralPhase::Critical
        ),
    );

    println!("  spectral analysis: {spectral_us:.0}µs");
    println!(
        "  phase={}, bandwidth={:.3}, cond={:.1}, ipr={:.4}, lsr={:.4}",
        result.phase,
        result.bandwidth,
        result.condition_number,
        result.mean_ipr,
        result.level_spacing_ratio
    );
}

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  ToadStool S79 Rewire Validation + Benchmark    ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    let mut h = ValidationHarness::new("toadstool_s79_rewire");
    let dispatcher = Dispatcher::new().await;

    validate_spectral(&mut h);

    if let Some(dev) = dispatcher.wgpu_device() {
        validate_gpu_chi_squared(&mut h, &dispatcher, dev);
        validate_gpu_kl_divergence(&mut h, dev);
        validate_gpu_entropy_variance_pearson(&mut h, dev);
        validate_fp64_strategy(&mut h, &dispatcher);
    } else {
        println!("\n[skip] No GPU available — skipping GPU rewire validation");
        h.check_bool("chi_squared (no GPU, skip)", true);
        h.check_bool("kl_divergence (no GPU, skip)", true);
        h.check_bool("shannon_entropy (no GPU, skip)", true);
        h.check_bool("variance (no GPU, skip)", true);
        h.check_bool("pearson (no GPU, skip)", true);
        h.check_bool("fp64_strategy (no GPU, skip)", true);
    }

    validate_weight_spectral(&mut h);

    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  Cross-Spring Provenance Map                     ║");
    println!("╠══════════════════════════════════════════════════╣");
    println!("║  hotSpring  → f64 pipeline, VarianceF64        ║");
    println!("║  wetSpring  → FusedMapReduceF64, CorrelationF64 ║");
    println!("║  neuralSpring → chi², KL, spectral → ToadStool  ║");
    println!("║  ToadStool  → FusedChiSquared, FusedKL (back)   ║");
    println!("╚══════════════════════════════════════════════════╝");

    h.finish();
}
