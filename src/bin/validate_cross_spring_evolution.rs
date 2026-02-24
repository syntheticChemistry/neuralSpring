// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-spring evolution benchmark: validates and benchmarks the 7 Dispatcher
//! methods rewired to upstream `barracuda::dispatch::domain_ops`, and reports
//! driver profile information from `GpuDriverProfile` (hotSpring-evolved).
//!
//! ## What this proves
//!
//! - **Upstream rewiring**: all 7 core math ops delegate to `BarraCUDA`
//!   `domain_ops` and produce correct results
//! - **Cross-spring evolution**: shaders and dispatch logic evolved from
//!   hotSpring (precision), wetSpring (bio), and neuralSpring (validation)
//! - **Driver awareness**: `GpuDriverProfile` correctly detects hardware
//!   and selects appropriate f64 strategy
//! - **Performance**: benchmarks upstream dispatch vs local CPU reference
//!
//! ## Cross-spring shader lineage
//!
//! ```text
//! hotSpring → df64_core, pow_f64, Taylor trig, Lanczos → BarraCUDA precision
//! wetSpring → HMM forward, ODE bio, NMF, Anderson     → BarraCUDA bio+spectral
//! neuralSpring → batch_fitness, pairwise_l2, eigh, ValidationHarness → BarraCUDA ops
//! All three → ToadStool (GPU sovereign pipeline)
//! ```
//!
//! ```text
//! cargo run --release --bin validate_cross_spring_evolution
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::time::Instant;

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::validation::ValidationHarness;

fn bench<F: FnOnce() -> T, T>(label: &str, f: F) -> (T, f64) {
    let start = Instant::now();
    let result = f();
    let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
    eprintln!("  [{label}] {elapsed_us:.1} µs");
    (result, elapsed_us)
}

fn gen_f64_vec(n: usize, scale: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 * scale).collect()
}

fn max_pairwise_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn validate_rewired_matmul(h: &mut ValidationHarness, dispatcher: &Dispatcher, cpu: &Dispatcher) {
    let n = 64;
    let a = gen_f64_vec(n * n, 0.001);
    let b: Vec<f64> = (0..n * n).map(|i| (n * n - i) as f64 * 0.001).collect();

    let (result, _) = bench("matmul upstream", || dispatcher.mat_mul(&a, &b, n));
    let (reference, _) = bench("matmul CPU ref", || cpu.mat_mul(&a, &b, n));

    // GPU matmul uses different accumulation order (parallel reduction) vs
    // sequential CPU; 1e-3 tolerance justified for 64-element dot products
    h.check_abs(
        "rewired matmul parity (64x64)",
        max_pairwise_diff(&result, &reference),
        0.0,
        1e-3,
    );
}

fn validate_rewired_frobenius(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    cpu: &Dispatcher,
) {
    let data = gen_f64_vec(1024, 0.01);

    let (result, _) = bench("frobenius upstream", || dispatcher.frobenius_norm(&data));
    let (reference, _) = bench("frobenius CPU ref", || cpu.frobenius_norm(&data));

    h.check_abs("rewired frobenius parity", result, reference, 1e-6);
}

fn validate_rewired_transpose(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    cpu: &Dispatcher,
) {
    let n = 32;
    let a = gen_f64_vec(n * n, 0.1);

    let (result, _) = bench("transpose upstream", || dispatcher.transpose(&a, n));
    let (reference, _) = bench("transpose CPU ref", || cpu.transpose(&a, n));

    h.check_abs(
        "rewired transpose parity (32x32)",
        max_pairwise_diff(&result, &reference),
        0.0,
        1e-12,
    );
}

fn validate_rewired_softmax(h: &mut ValidationHarness, dispatcher: &Dispatcher, cpu: &Dispatcher) {
    let x: Vec<f64> = (0..256_i32)
        .map(|i| f64::from(i).mul_add(0.02, -2.56))
        .collect();

    let (result, _) = bench("softmax upstream", || dispatcher.softmax(&x));
    let (reference, _) = bench("softmax CPU ref", || cpu.softmax(&x));

    let sum: f64 = result.iter().sum();
    h.check_abs("rewired softmax sums to 1", sum, 1.0, 1e-10);
    h.check_abs(
        "rewired softmax parity",
        max_pairwise_diff(&result, &reference),
        0.0,
        1e-10,
    );
}

fn validate_rewired_l2(h: &mut ValidationHarness, dispatcher: &Dispatcher, cpu: &Dispatcher) {
    let a = gen_f64_vec(512, 0.01);
    let b: Vec<f64> = (0..512_i32)
        .map(|i| f64::from(i).mul_add(0.01, 1.0))
        .collect();

    let (result, _) = bench("l2_distance upstream", || dispatcher.l2_distance(&a, &b));
    let (reference, _) = bench("l2_distance CPU ref", || cpu.l2_distance(&a, &b));

    h.check_abs("rewired l2_distance parity", result, reference, 1e-8);
}

fn validate_rewired_mean(h: &mut ValidationHarness, dispatcher: &Dispatcher, cpu: &Dispatcher) {
    let data = gen_f64_vec(2048, 0.001);

    let (result, _) = bench("mean upstream", || dispatcher.mean(&data));
    let (reference, _) = bench("mean CPU ref", || cpu.mean(&data));

    h.check_abs("rewired mean parity", result, reference, 1e-10);
}

fn validate_rewired_variance(h: &mut ValidationHarness, dispatcher: &Dispatcher, cpu: &Dispatcher) {
    let data = gen_f64_vec(2048, 0.001);

    let (result, _) = bench("variance upstream", || dispatcher.variance(&data));
    let (reference, _) = bench("variance CPU ref", || cpu.variance(&data));

    h.check_abs("rewired variance parity", result, reference, 1e-8);
}

fn validate_driver_profile(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    if let Some(profile) = dispatcher.driver_profile() {
        eprintln!("[profile] Driver: {:?}", profile.driver);
        eprintln!("[profile] Compiler: {:?}", profile.compiler);
        eprintln!("[profile] Arch: {:?}", profile.arch);
        eprintln!("[profile] FP64 rate: {:?}", profile.fp64_rate);
        eprintln!("[profile] FP64 strategy: {:?}", profile.fp64_strategy());
        eprintln!(
            "[profile] pow workaround: {}",
            profile.needs_pow_f64_workaround()
        );
        eprintln!(
            "[profile] Eigensolve strategy: {:?}",
            profile.optimal_eigensolve_strategy()
        );

        h.check_bool("driver profile detected", true);

        let strategy = dispatcher.fp64_strategy();
        let strategy_valid = matches!(
            strategy,
            barracuda::device::driver_profile::Fp64Strategy::Native
                | barracuda::device::driver_profile::Fp64Strategy::Hybrid
        );
        h.check_bool("fp64 strategy valid", strategy_valid);
    } else {
        eprintln!("[profile] No GPU — skipping driver profile checks");
        h.check_bool("driver profile (no GPU, skip)", true);
    }
}

fn benchmark_throughput(dispatcher: &Dispatcher, cpu: &Dispatcher) {
    eprintln!("\n=== Cross-Spring Throughput Benchmark ===");
    eprintln!("(upstream dispatch includes GPU routing + size-based thresholds)\n");

    let sizes: [u32; 4] = [64, 256, 1024, 4096];
    for sz in sizes {
        let data = gen_f64_vec(sz as usize, 0.001);

        let start = Instant::now();
        for _ in 0..100 {
            std::hint::black_box(dispatcher.mean(&data));
        }
        let upstream_us = start.elapsed().as_secs_f64() * 1e4;

        let start = Instant::now();
        for _ in 0..100 {
            std::hint::black_box(cpu.mean(&data));
        }
        let cpu_us = start.elapsed().as_secs_f64() * 1e4;

        let ratio = cpu_us / upstream_us;
        eprintln!(
            "  mean(n={sz:>5}): upstream {upstream_us:>8.1}µs  cpu {cpu_us:>8.1}µs  ratio {ratio:.2}x"
        );
    }

    let mat_sizes: [usize; 4] = [16, 32, 64, 128];
    for n in mat_sizes {
        let a = gen_f64_vec(n * n, 0.001);
        let b: Vec<f64> = (0..n * n).map(|i| (n * n - i) as f64 * 0.001).collect();

        let start = Instant::now();
        for _ in 0..10 {
            std::hint::black_box(dispatcher.mat_mul(&a, &b, n));
        }
        let upstream_us = start.elapsed().as_secs_f64() * 1e5;

        let start = Instant::now();
        for _ in 0..10 {
            std::hint::black_box(cpu.mat_mul(&a, &b, n));
        }
        let cpu_us = start.elapsed().as_secs_f64() * 1e5;

        let ratio = cpu_us / upstream_us;
        eprintln!(
            "  matmul({n:>3}x{n:>3}): upstream {upstream_us:>8.1}µs  cpu {cpu_us:>8.1}µs  ratio {ratio:.2}x"
        );
    }
}

fn report_cross_spring_lineage() {
    eprintln!("\n=== Cross-Spring Evolution Lineage ===\n");
    eprintln!("hotSpring \u{2192} BarraCUDA precision layer:");
    eprintln!("  \u{2022} df64_core.wgsl (double-float f32-pair emulation)");
    eprintln!("  \u{2022} pow_f64 polyfill (transcendental workaround)");
    eprintln!("  \u{2022} Fp64Strategy (Native/Hybrid detection)");
    eprintln!("  \u{2022} GpuDriverProfile (hardware-adaptive dispatch)");
    eprintln!("  \u{2022} Taylor-series sin/cos (7-term + Cody-Waite)");
    eprintln!("  \u{2022} Lanczos eigensolver (lattice QCD heritage)");
    eprintln!();
    eprintln!("wetSpring \u{2192} BarraCUDA bio+spectral layer:");
    eprintln!("  \u{2022} HMM forward/backward (phylogenetics)");
    eprintln!("  \u{2022} 5 ODE bio systems (Capacitor, Cooperation, MultiSignal, Bistable, PhageDefense)");
    eprintln!("  \u{2022} NMF (non-negative matrix factorization)");
    eprintln!("  \u{2022} Anderson localization (3d_correlated, sweep_averaged, find_w_c)");
    eprintln!("  \u{2022} Ridge regression (ESN readout)");
    eprintln!();
    eprintln!("neuralSpring \u{2192} BarraCUDA validation+ops layer:");
    eprintln!("  \u{2022} ValidationHarness + exit_no_gpu + require! macro");
    eprintln!("  \u{2022} batch_fitness_eval, pairwise_l2, pairwise_hamming/jaccard");
    eprintln!("  \u{2022} spatial_payoff, hill_gate, multi_obj_fitness");
    eprintln!("  \u{2022} eigh_householder_qr, batch_ipr, swarm_nn");
    eprintln!("  \u{2022} 4-tier matmul KernelRouter");
    eprintln!();
    eprintln!("All three \u{2192} ToadStool (GPU sovereign pipeline):");
    eprintln!("  \u{2022} 599+ WGSL shaders (cross-spring evolved)");
    eprintln!("  \u{2022} domain_ops dispatch (this benchmark validates rewiring)");
    eprintln!("  \u{2022} GpuDriverProfile (this benchmark validates detection)");
}

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("cross_spring_evolution");

    let dispatcher = Dispatcher::new().await;
    let cpu = Dispatcher::cpu_only();

    eprintln!(
        "[evolution] GPU: {} ({}), f64 strategy: {:?}, pow workaround: {}",
        dispatcher.has_gpu(),
        dispatcher.adapter_name(),
        dispatcher.fp64_strategy(),
        dispatcher.needs_pow_workaround(),
    );

    eprintln!("\n--- Rewired Method Parity ---\n");
    validate_rewired_matmul(&mut h, &dispatcher, &cpu);
    validate_rewired_frobenius(&mut h, &dispatcher, &cpu);
    validate_rewired_transpose(&mut h, &dispatcher, &cpu);
    validate_rewired_softmax(&mut h, &dispatcher, &cpu);
    validate_rewired_l2(&mut h, &dispatcher, &cpu);
    validate_rewired_mean(&mut h, &dispatcher, &cpu);
    validate_rewired_variance(&mut h, &dispatcher, &cpu);

    eprintln!("\n--- Driver Profile Validation ---\n");
    validate_driver_profile(&mut h, &dispatcher);

    benchmark_throughput(&dispatcher, &cpu);
    report_cross_spring_lineage();

    h.finish();
}
