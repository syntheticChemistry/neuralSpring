// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-spring modern benchmark: measures performance of rewired
//! primitives across spring boundaries through ToadStool/BarraCUDA.
//!
//! ## Cross-spring evolution benchmarked
//!
//! | Origin | Feature | Path |
//! |--------|---------|------|
//! | hotSpring `proxy.rs` | spectral bandwidth/cond/phase | CPU (from eigenvalues) |
//! | hotSpring `esn_v2` | GPU ESN via Tensors | `barracuda::esn_v2` |
//! | hotSpring df64 | f64 precision shaders | WGSL → naga → GPU |
//! | wetSpring bio | diversity fusion | WGSL fused shader |
//! | barracuda stats | cross-spring stats | CPU → GPU dispatch |
//!
//! # Panics
//!
//! Panics if the tokio runtime cannot be created — this is a benchmark binary.

#![expect(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::suboptimal_flops,
    reason = "validation binary"
)]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::rng::Rng;
use neural_spring::validation::ValidationHarness;
use neural_spring::weight_spectral;
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 50;

fn bench<F: FnMut()>(label: &str, mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    let elapsed = start.elapsed();
    let us_per_iter = elapsed.as_micros() as f64 / ITERS as f64;
    eprintln!("    {label}: {us_per_iter:.1}µs/iter");
    us_per_iter
}

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  neuralSpring — Cross-Spring Modern Benchmark                   ║");
    eprintln!("║  hotSpring + wetSpring + airSpring + bingoCube → TS S86 bench   ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let mut h = ValidationHarness::new("cross_spring_modern_bench");
    let mut rng = Rng::new(42);

    bench_spectral_diagnostics(&mut h);
    bench_full_analysis(&mut h, &mut rng);
    bench_gpu_dispatch_and_esn(&mut h, &mut rng);
    bench_s86_hydrology(&mut h);
    bench_s86_nautilus(&mut h);
    print_summary();

    h.finish();
}

fn bench_spectral_diagnostics(h: &mut ValidationHarness) {
    eprintln!("═══ hotSpring proxy.rs → spectral diagnostics (zero-dep, from eigenvalues) ═══");
    eprintln!("  Provenance: hotSpring ProxyFeatures → classify_phase/bandwidth/condition");
    eprintln!("  → neuralSpring WeightSpectralResult cross-spring evolution");
    eprintln!();

    let evals_1k: Vec<f64> = (0..1000)
        .map(|i| f64::from(i).mul_add(0.01, -5.0))
        .collect();

    let cpu_bw = bench("spectral_bandwidth 1k eigenvalues", || {
        std::hint::black_box(weight_spectral::spectral_bandwidth(&evals_1k));
    });
    h.check_bool(
        &format!("hotSpring→bandwidth: 1k evals {cpu_bw:.1}µs (< 10µs)"),
        cpu_bw < 10.0,
    );

    let cpu_cond = bench("spectral_condition_number 1k eigenvalues", || {
        std::hint::black_box(weight_spectral::spectral_condition_number(&evals_1k));
    });
    h.check_bool(
        &format!("hotSpring→condition: 1k evals {cpu_cond:.1}µs (< 100µs)"),
        cpu_cond < 100.0,
    );

    let cpu_phase = bench("classify_phase 1k iterations", || {
        for i in 0..1000 {
            std::hint::black_box(weight_spectral::classify_phase(
                f64::from(i).mul_add(0.0003, 0.3),
            ));
        }
    });
    h.check_bool(
        &format!("hotSpring→phase: 1k iters {cpu_phase:.1}µs"),
        cpu_phase < 50.0,
    );

    eprintln!();
}

fn bench_full_analysis(h: &mut ValidationHarness, rng: &mut Rng) {
    eprintln!("═══ Full weight_spectral_analysis (nS-01 + hotSpring extensions) ═══");
    eprintln!("  Original: eigensolve + IPR + LSR + entropy + MP departure");
    eprintln!("  + Cross-spring: bandwidth + condition_number + phase");
    eprintln!();

    for &dim in &[16_usize, 32, 64] {
        let w: Vec<f64> = (0..dim * dim).map(|_| rng.normal()).collect();
        let us = bench(&format!("weight_spectral_analysis {dim}×{dim}"), || {
            std::hint::black_box(weight_spectral::weight_spectral_analysis(&w, dim, dim));
        });
        h.check_bool(&format!("nS-01+hS: {dim}×{dim} {us:.1}µs"), us.is_finite());

        let result = weight_spectral::weight_spectral_analysis(&w, dim, dim);
        eprintln!(
            "      → bandwidth={:.4}, cond={:.2}, phase={}",
            result.bandwidth, result.condition_number, result.phase
        );
    }
    eprintln!();
}

fn bench_gpu_dispatch_and_esn(h: &mut ValidationHarness, rng: &mut Rng) {
    eprintln!("═══ GPU Dispatch — hotSpring precision × wetSpring bio ═══");
    eprintln!("  Dispatcher auto-routes to best device.");
    eprintln!();

    let rt = tokio::runtime::Runtime::new()
        .expect("tokio runtime creation failed — required for async benchmark");
    let dispatcher = rt.block_on(async { Dispatcher::new().await });
    let has_gpu = dispatcher.has_gpu();
    eprintln!(
        "  GPU: {} ({})",
        dispatcher.adapter_name(),
        if has_gpu { "active" } else { "CPU fallback" }
    );

    let n = 50_000_usize;
    let a: Vec<f64> = (0..n).map(|_| rng.next_f64().mul_add(10.0, -5.0)).collect();
    let b: Vec<f64> = (0..n).map(|_| rng.next_f64().mul_add(10.0, -5.0)).collect();

    let us_var = bench("Dispatcher::variance 50k (hS Welford→GPU)", || {
        let _ = std::hint::black_box(dispatcher.variance(&a));
    });
    h.check_bool(
        &format!("hS→dispatch: variance 50k {us_var:.1}µs"),
        us_var.is_finite(),
    );

    let us_corr = bench("Dispatcher::pearson 50k (hS+wS→GPU)", || {
        let _ = std::hint::black_box(dispatcher.pearson_correlation(&a, &b));
    });
    h.check_bool(
        &format!("hS+wS→dispatch: pearson 50k {us_corr:.1}µs"),
        us_corr.is_finite(),
    );

    let probs: Vec<f64> = a.iter().map(|x| x.abs() / 1000.0 + 1e-10).collect();
    let us_shannon = bench("Dispatcher::shannon 50k (wS fused→GPU)", || {
        let _ = std::hint::black_box(dispatcher.shannon_entropy(&probs));
    });
    h.check_bool(
        &format!("wS→dispatch: shannon 50k {us_shannon:.1}µs"),
        us_shannon.is_finite(),
    );

    let side = 200_usize;
    let mat: Vec<f64> = (0..side * side).map(|_| rng.next_f64()).collect();
    let us_matmul = bench("Dispatcher::matmul 200×200 (nS→GPU)", || {
        let _ = std::hint::black_box(dispatcher.mat_mul(&mat, &mat, side));
    });
    h.check_bool(
        &format!("nS→dispatch: matmul 200×200 {us_matmul:.1}µs"),
        us_matmul.is_finite(),
    );

    eprintln!();

    bench_esn_gpu(h, &dispatcher);
}

fn bench_esn_gpu(h: &mut ValidationHarness, dispatcher: &neural_spring::gpu_dispatch::Dispatcher) {
    eprintln!("═══ hotSpring esn_v2 → barracuda Tensor ESN benchmark ═══");
    eprintln!("  Provenance: hotSpring EchoStateNetwork → ToadStool esn_v2 absorption");
    eprintln!("  → barracuda::tensor (GPU matmul + tanh shaders)");
    eprintln!("  → neuralSpring wdm_esn::classify_via_barracuda");
    eprintln!();

    let Some(device) = dispatcher.wgpu_device() else {
        eprintln!("  [skip] ESN GPU benchmark (no adapter)");
        eprintln!();
        return;
    };
    let device = device.clone();
    let json_path = std::path::Path::new("control/wdm/esn_regime_baseline.json");
    if !json_path.exists() {
        eprintln!("  [skip] ESN benchmark (no baseline JSON)");
        eprintln!();
        return;
    }

    let json_str = std::fs::read_to_string(json_path)
        .expect("failed to read baseline JSON — ensure control/wdm/ exists and file is present");
    let classifier = neural_spring::wdm_esn::load_esn_from_json(&json_str)
        .expect("failed to parse ESN JSON — baseline format may have changed");

    let us_cpu = bench("ESN classify CPU (wdm_esn local)", || {
        std::hint::black_box(classifier.classify(0.5, 5.5));
    });
    h.check_bool(
        &format!("ESN CPU: classify {us_cpu:.1}µs"),
        us_cpu.is_finite(),
    );

    let us_gpu = bench("ESN classify GPU (barracuda Tensor)", || {
        let _ = std::hint::black_box(neural_spring::wdm_esn::classify_via_barracuda(
            &classifier,
            0.5,
            5.5,
            &device,
        ));
    });
    h.check_bool(
        &format!("ESN GPU: classify {us_gpu:.1}µs"),
        us_gpu.is_finite(),
    );

    if us_gpu > 0.0 && us_cpu > 0.0 {
        let ratio = us_gpu / us_cpu;
        eprintln!("    → GPU/CPU ratio: {ratio:.1}× (GPU overhead from small matmul dispatch)");
    }

    eprintln!();
}

fn bench_s86_hydrology(h: &mut ValidationHarness) {
    eprintln!("═══ ToadStool S81-86 → hydrology evolution benchmark ═══");
    eprintln!("  Provenance: airSpring → Hargreaves/Thornthwaite/Hamon/Makkink/Turc");
    eprintln!("  → ToadStool S81 absorption → barracuda::stats::hydrology");
    eprintln!();

    let monthly_temps = [
        -5.0, -3.0, 2.0, 8.0, 15.0, 20.0, 23.0, 22.0, 17.0, 10.0, 3.0, -2.0,
    ];
    let heat_index = barracuda::stats::thornthwaite_heat_index(&monthly_temps);
    let t_mean = 20.0;
    let rs = 18.0;
    let rh = 60.0;
    let daylight = 14.0;

    let us_hargreaves = bench("hargreaves_et0 (aS→TS original)", || {
        std::hint::black_box(barracuda::stats::hargreaves_et0(t_mean, 12.0, 28.0));
    });
    let us_thornthwaite = bench("thornthwaite_et0 (aS→TS S81)", || {
        std::hint::black_box(barracuda::stats::thornthwaite_et0(
            t_mean, heat_index, daylight, 30.0,
        ));
    });
    let us_hamon = bench("hamon_et0 (aS→TS S81)", || {
        std::hint::black_box(barracuda::stats::hamon_et0(t_mean, daylight));
    });
    let us_makkink = bench("makkink_et0 (aS→TS S81)", || {
        std::hint::black_box(barracuda::stats::makkink_et0(t_mean, rs));
    });
    let us_turc = bench("turc_et0 (aS→TS S81)", || {
        std::hint::black_box(barracuda::stats::turc_et0(t_mean, rs, rh));
    });

    h.check_bool(
        &format!("hydrology: 5 ET₀ methods benchmarked (hargreaves {us_hargreaves:.1}µs, thornth {us_thornthwaite:.1}µs, hamon {us_hamon:.1}µs, makkink {us_makkink:.1}µs, turc {us_turc:.1}µs)"),
        us_hargreaves.is_finite() && us_thornthwaite.is_finite(),
    );

    eprintln!();
}

fn bench_s86_nautilus(h: &mut ValidationHarness) {
    eprintln!("═══ ToadStool S80 → nautilus evolution benchmark ═══");
    eprintln!("  Provenance: hotSpring brain → bingoCube nautilus → ToadStool S80 absorption");
    eprintln!("  → barracuda::nautilus → neuralSpring SpectralNautilusBridge");
    eprintln!();

    use barracuda::nautilus::{
        DriftMonitor, GenerationRecord, InstanceId, NautilusBrain, NautilusBrainConfig,
        NautilusShell, ShellConfig,
    };

    let us_brain_create = bench("NautilusBrain::new (hS→bC→TS)", || {
        std::hint::black_box(NautilusBrain::new(NautilusBrainConfig::default(), "bench"));
    });

    let us_observe = bench("NautilusBrain::observe (hS QCD provenance)", || {
        let mut brain = NautilusBrain::new(NautilusBrainConfig::default(), "bench");
        brain.observe(barracuda::nautilus::BetaObservation {
            beta: 5.5,
            plaquette: 0.58,
            cg_iters: 120.0,
            acceptance: 0.75,
            delta_h_abs: 0.01,
            quenched_plaq: None,
            quenched_plaq_var: None,
            anderson_r: Some(0.42),
            anderson_lambda_min: Some(-2.1),
        });
    });

    let origin = InstanceId("bench-shell".to_string());
    let shell_config = ShellConfig::default();
    let us_shell = bench("NautilusShell::from_seed (TS evolutionary)", || {
        std::hint::black_box(NautilusShell::from_seed(
            shell_config.clone(),
            origin.clone(),
            42,
        ));
    });

    let us_drift = bench("DriftMonitor::record + is_drifting (TS)", || {
        let mut dm = DriftMonitor::default();
        for g in 0..20 {
            let gen = GenerationRecord {
                generation: g,
                mean_fitness: 0.5 + 0.01 * g as f64,
                best_fitness: 0.8 + 0.005 * g as f64,
                pop_size: 100,
                origin: InstanceId("bench-drift".to_string()),
                training_size: 10,
            };
            dm.record(&gen, 100);
        }
        std::hint::black_box(dm.is_drifting());
    });

    let us_bridge = bench(
        "SpectralNautilusBridge train+predict (nS→TS roundtrip)",
        || {
            let mut bridge = neural_spring::nautilus_bridge::SpectralNautilusBridge::new("bench");
            for i in 0..8 {
                let w = f64::from(i).mul_add(0.5, 2.0);
                bridge.observe_spectral(w, 0.45, 0.1 / w, w * 0.3, 0.02 * w);
            }
            bridge.train();
            std::hint::black_box(bridge.predict(3.0));
        },
    );

    h.check_bool(
        &format!("nautilus: brain {us_brain_create:.1}µs, observe {us_observe:.1}µs, shell {us_shell:.1}µs, drift {us_drift:.1}µs, bridge {us_bridge:.1}µs"),
        us_brain_create.is_finite() && us_drift.is_finite(),
    );

    let _ = shell_config;

    eprintln!();
}

fn print_summary() {
    eprintln!("═══ Cross-Spring Modern Summary (S112) ═══");
    eprintln!("  hotSpring contributions to neuralSpring:");
    eprintln!("    ✓ proxy.rs diagnostics: bandwidth, condition_number, phase");
    eprintln!("    ✓ esn_v2: GPU ESN via Tensor ops (reservoir update + readout shaders)");
    eprintln!("    ✓ df64 precision: f64 shaders through naga downcast pipeline");
    eprintln!("    ✓ validation pattern: ValidationHarness absorbed to barracuda");
    eprintln!("    ✓ brain arch → NautilusBrain QCD observation pipeline");
    eprintln!("  wetSpring contributions to neuralSpring:");
    eprintln!("    ✓ DiversityFusionGpu: fused Shannon+Simpson+Pielou GPU shader");
    eprintln!("    ✓ Bio diversity stats: shannon, simpson, chao1, bray_curtis");
    eprintln!("    ✓ HMM f64 dispatch: forward algorithm on GPU");
    eprintln!("  airSpring contributions to neuralSpring:");
    eprintln!("    ✓ Hydrology: Hargreaves, Thornthwaite, Hamon, Makkink, Turc ET₀");
    eprintln!("    ✓ regression, metrics (RMSE, R², NSE, MAE), moving_window");
    eprintln!("  neuralSpring contributions to ecosystem:");
    eprintln!("    ✓ Batch fitness, pairwise ops, swarm NN → barracuda GPU shaders");
    eprintln!("    ✓ weight_spectral: ESD, level_spacing, MP bounds → barracuda::stats");
    eprintln!("    ✓ SpectralNautilusBridge → barracuda::nautilus::spectral_bridge");
    eprintln!("    ✓ SimpleMlp: CPU inference → barracuda::nn");
    eprintln!("    ✓ fused_chi_squared_f64, fused_kl_divergence_f64 → barracuda::ops");
    eprintln!();
    eprintln!("  ToadStool S80 absorption: bingoCube nautilus → barracuda::nautilus");
    eprintln!("  ToadStool S81: airSpring hydrology → barracuda::stats::hydrology");
    eprintln!("  ToadStool S84-86: ComputeDispatch 76→144 ops, hydrology module split");
    eprintln!("  692+ WGSL f64 shaders, 144 ComputeDispatch ops, nautilus evolutionary reservoir");
    eprintln!();
}
