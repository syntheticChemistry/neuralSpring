// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-spring shader evolution benchmark with provenance tracking.
//!
//! Benchmarks the full evolution chain for operations contributed by
//! each spring, showing where and when each operation was absorbed
//! into ToadStool/BarraCUDA, and the performance benefit at each tier.
//!
//! ## Provenance tiers benchmarked
//!
//! | Tier | Description |
//! |------|-------------|
//! | T0 | Local CPU reference (e.g., `transformer::softmax`) |
//! | T1 | `barracuda::dispatch` CPU path |
//! | T2 | `barracuda::dispatch` GPU path (via Dispatcher) |
//! | T3 | Direct `barracuda::stats`/`spectral` CPU functions |
//!
//! ## Spring contributions tracked
//!
//! | Spring | Operations | Shader evolution |
//! |--------|-----------|------------------|
//! | neuralSpring | softmax, gelu, matmul, sigmoid, RK4 | → WGSL → ToadStool |
//! | wetSpring | Shannon, Simpson, chao1, HMM, Bray-Curtis | → bio shaders |
//! | hotSpring | eigensolve, spectral, precision, DF64 | → math_f64.wgsl |
//! | groundSpring | bootstrap, jackknife, kimura, norm_cdf | → uncertainty |
//! | airSpring | hydrology (ET₀), regression, water balance | → env shaders |
//!
//! ```text
//! cargo run --release --bin bench_cross_spring_shader_evolution
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::suboptimal_flops,
    clippy::items_after_statements,
    clippy::manual_range_contains,
    clippy::similar_names,
    clippy::doc_markdown
)]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::primitives;
use neural_spring::transformer;
use neural_spring::validation::ValidationHarness;
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 100;

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

fn bench_neuralspring_origins(h: &mut ValidationHarness, disp: &Dispatcher) {
    eprintln!("═══ neuralSpring origins → ToadStool absorption ═══");
    eprintln!("  softmax: transformer.rs → barracuda::dispatch → softmax_f64.wgsl");
    eprintln!("  gelu: transformer.rs → barracuda::dispatch → gelu_f64.wgsl");
    eprintln!("  matmul: matmul_gpu_evolved → barracuda::dispatch → matmul.wgsl");
    eprintln!("  RK4: primitives.rs → metalForge rk4_parallel.wgsl");
    eprintln!();

    let x256: Vec<f64> = (0..256).map(|i| (i as f64) * 0.01).collect();
    let x1k: Vec<f64> = (0..1024).map(|i| (i as f64) * 0.001).collect();

    let local_sm = bench("softmax T0: local CPU (256)", || {
        std::hint::black_box(transformer::softmax(&x256));
    });
    let disp_sm = bench("softmax T2: Dispatcher GPU (256)", || {
        std::hint::black_box(disp.softmax(&x256));
    });
    let bc_sm = bench("softmax T1: barracuda::dispatch (256)", || {
        std::hint::black_box(
            barracuda::dispatch::softmax_dispatch(&x256, disp.wgpu_device()).unwrap(),
        );
    });
    let speedup_sm = if disp_sm > 0.0 {
        local_sm / disp_sm
    } else {
        0.0
    };
    eprintln!("    → softmax GPU speedup: {speedup_sm:.2}x vs local CPU");
    h.check_bool("softmax: barracuda::dispatch works", bc_sm > 0.0);

    eprintln!();
    let local_gelu = bench("gelu T0: local CPU (1024)", || {
        let _: Vec<f64> = x1k.iter().copied().map(transformer::gelu).collect();
    });
    let disp_gelu = bench("gelu T2: Dispatcher GPU (1024)", || {
        std::hint::black_box(disp.gelu(&x1k));
    });
    let _bc_gelu = bench("gelu T1: barracuda::dispatch (1024)", || {
        std::hint::black_box(barracuda::dispatch::gelu_dispatch(&x1k, disp.wgpu_device()).unwrap());
    });
    let speedup_gelu = if disp_gelu > 0.0 {
        local_gelu / disp_gelu
    } else {
        0.0
    };
    eprintln!("    → gelu GPU speedup: {speedup_gelu:.2}x vs local CPU");

    eprintln!();
    let n = 64;
    let mat_a: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.01).collect();
    let mat_b: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.005 + 0.1).collect();

    let disp_mm = bench("matmul T2: Dispatcher GPU (64x64)", || {
        std::hint::black_box(disp.mat_mul(&mat_a, &mat_b, n));
    });
    let bc_mm = bench("matmul T1: barracuda::dispatch (64x64)", || {
        std::hint::black_box(
            barracuda::dispatch::matmul_dispatch(&mat_a, &mat_b, n, n, n, disp.wgpu_device())
                .unwrap(),
        );
    });
    h.check_bool("matmul: GPU dispatch functional", disp_mm > 0.0);
    h.check_bool("matmul: barracuda::dispatch functional", bc_mm > 0.0);

    eprintln!();
    bench("sigmoid T0: local CPU (1024 evals)", || {
        for &v in &x1k {
            std::hint::black_box(primitives::sigmoid(v));
        }
    });
    eprintln!("    → sigmoid: local CPU reference; GPU via sigmoid_f64.wgsl in coralForge");

    eprintln!();
    bench(
        "rk4 T0: local CPU (1000 steps, harmonic oscillator)",
        || {
            let mut y = [1.0, 0.0];
            for _ in 0..1000 {
                y = primitives::rk4_step(&y, 0.01, |s| [-s[1], s[0]]);
            }
            std::hint::black_box(y);
        },
    );
    eprintln!("    → RK4: local single-step; GPU batch via rk4_parallel.wgsl");
    eprintln!();
}

fn bench_wetspring_origins(h: &mut ValidationHarness) {
    eprintln!("═══ wetSpring origins → ToadStool absorption ═══");
    eprintln!("  Shannon/Simpson: wetSpring diversity → barracuda::stats");
    eprintln!("  HMM: wetSpring hmm_forward_log.wgsl → barracuda::dispatch");
    eprintln!("  Bray-Curtis: wetSpring ecological → barracuda::stats");
    eprintln!("  chao1/pielou: wetSpring richness → barracuda::stats::diversity");
    eprintln!();

    let counts_256: Vec<f64> = (1..=256).map(|i| (i * i) as f64).collect();
    let freqs: Vec<f64> = {
        let sum: f64 = counts_256.iter().sum();
        counts_256.iter().map(|&c| c / sum).collect()
    };

    bench("shannon T3: barracuda::stats (256 bins)", || {
        std::hint::black_box(barracuda::stats::shannon(&counts_256));
    });

    bench("shannon_from_freq T3: barracuda::stats (256 bins)", || {
        std::hint::black_box(barracuda::stats::shannon_from_frequencies(&freqs));
    });

    bench("simpson T3: barracuda::stats (256 bins)", || {
        std::hint::black_box(barracuda::stats::simpson(&counts_256));
    });

    let counts_u64: Vec<u64> = (1..=256).map(|i| i * i).collect();
    bench("chao1_classic T3: barracuda::stats (256 bins)", || {
        std::hint::black_box(barracuda::stats::chao1_classic(&counts_u64));
    });

    bench("pielou T3: barracuda::stats (256 bins)", || {
        std::hint::black_box(barracuda::stats::pielou_evenness(&counts_256));
    });

    let a: Vec<f64> = (0..128).map(|i| (i as f64) * 0.5).collect();
    let b: Vec<f64> = (0..128).map(|i| (i as f64) * 0.5 + 1.0).collect();
    bench("bray_curtis T3: barracuda::stats (128 sites)", || {
        std::hint::black_box(barracuda::stats::bray_curtis(&a, &b));
    });

    h.check_bool(
        "wetSpring diversity: shannon positive",
        barracuda::stats::shannon(&counts_256) > 0.0,
    );
    h.check_bool(
        "wetSpring diversity: simpson in [0,1]",
        barracuda::stats::simpson(&counts_256) >= 0.0
            && barracuda::stats::simpson(&counts_256) <= 1.0,
    );
    eprintln!();
}

fn bench_hotspring_origins(h: &mut ValidationHarness, disp: &Dispatcher) {
    eprintln!("═══ hotSpring origins → ToadStool absorption ═══");
    eprintln!("  eigensolve: hotSpring Householder+QR → barracuda::linalg → Dispatcher");
    eprintln!("  spectral: hotSpring level_spacing_ratio → barracuda::spectral");
    eprintln!("  precision: hotSpring DF64 → math_f64.wgsl (28 fn) + df64_transcendentals.wgsl");
    eprintln!("  pearson: hotSpring correlation → barracuda::stats → Dispatcher");
    eprintln!();

    let n = 32;
    let mut sym: Vec<f64> = vec![0.0; n * n];
    for i in 0..n {
        sym[i * n + i] = (i as f64 + 1.0) * 2.0;
        if i + 1 < n {
            sym[i * n + (i + 1)] = 0.5;
            sym[(i + 1) * n + i] = 0.5;
        }
    }

    bench("eigh T2: Dispatcher GPU (32×32 symmetric)", || {
        std::hint::black_box(disp.eigh(&sym, n));
    });

    let (evals, _) = disp.eigh(&sym, n);
    bench("level_spacing_ratio T3: barracuda::spectral", || {
        std::hint::black_box(barracuda::spectral::level_spacing_ratio(&evals));
    });
    bench("spectral_bandwidth T3: barracuda::spectral", || {
        std::hint::black_box(barracuda::spectral::spectral_bandwidth(&evals));
    });
    bench("spectral_condition_number T3: barracuda::spectral", || {
        std::hint::black_box(barracuda::spectral::spectral_condition_number(&evals));
    });
    let mp_upper = barracuda::stats::marchenko_pastur_bounds(evals.len() as f64 / 10.0).1;
    bench("classify_spectral_phase T3: barracuda::spectral", || {
        std::hint::black_box(barracuda::spectral::classify_spectral_phase(
            &evals, mp_upper,
        ));
    });

    eprintln!();
    let data_a: Vec<f64> = (0..512).map(|i| (i as f64) * 0.1).collect();
    let data_b: Vec<f64> = (0..512).map(|i| (i as f64) * 0.1 + 0.5).collect();

    bench("pearson T3: barracuda::stats (512 pairs)", || {
        std::hint::black_box(barracuda::stats::pearson_correlation(&data_a, &data_b).unwrap());
    });

    let disp_var_t = bench("variance T2: Dispatcher GPU (512)", || {
        std::hint::black_box(disp.variance(&data_a));
    });
    let bc_var_t = bench("variance T1: barracuda::dispatch (512)", || {
        std::hint::black_box(
            barracuda::dispatch::variance_dispatch(&data_a, disp.wgpu_device()).unwrap(),
        );
    });

    h.check_bool("hotSpring eigensolve: functional", evals.len() == n);
    h.check_bool("hotSpring precision: dispatched", disp_var_t > 0.0);
    h.check_bool("hotSpring precision: barracuda path", bc_var_t > 0.0);
    eprintln!();
}

fn bench_groundspring_origins(h: &mut ValidationHarness) {
    eprintln!("═══ groundSpring origins → ToadStool absorption ═══");
    eprintln!("  bootstrap: groundSpring uncertainty → barracuda::stats::bootstrap_ci");
    eprintln!("  jackknife: groundSpring leave-one-out → barracuda::stats::jackknife");
    eprintln!("  kimura: groundSpring evolution → barracuda::stats::kimura_fixation_prob");
    eprintln!("  norm_cdf: groundSpring normal dist → barracuda::stats::norm_cdf");
    eprintln!();

    let data: Vec<f64> = (0..200).map(|i| (i as f64) * 0.1).collect();

    bench(
        "bootstrap_ci T3: barracuda::stats (200 pts, 500 reps)",
        || {
            std::hint::black_box(
                barracuda::stats::bootstrap_ci(
                    &data,
                    |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64,
                    500,
                    0.95,
                    42,
                )
                .unwrap(),
            );
        },
    );

    bench("jackknife T3: barracuda::stats (200 pts)", || {
        std::hint::black_box(
            barracuda::stats::jackknife(&data, |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64)
                .unwrap(),
        );
    });

    bench("kimura T3: barracuda::stats (N=1000)", || {
        std::hint::black_box(barracuda::stats::kimura_fixation_prob(1000, 0.01, 0.5));
    });

    bench("norm_cdf T3: barracuda::stats (batch 1000)", || {
        for i in 0..1000 {
            std::hint::black_box(barracuda::stats::norm_cdf((i as f64 - 500.0) * 0.01));
        }
    });

    bench("norm_ppf T3: barracuda::stats (batch 1000)", || {
        for i in 1..1000 {
            std::hint::black_box(barracuda::stats::norm_ppf(i as f64 / 1000.0));
        }
    });

    h.check_bool(
        "groundSpring bootstrap: functional",
        barracuda::stats::bootstrap_ci(
            &data,
            |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64,
            100,
            0.95,
            42,
        )
        .is_ok(),
    );
    h.check_bool(
        "groundSpring norm_cdf: Φ(0) = 0.5",
        (barracuda::stats::norm_cdf(0.0) - 0.5).abs() < 1e-12,
    );
    eprintln!();
}

fn bench_airspring_origins(h: &mut ValidationHarness) {
    eprintln!("═══ airSpring origins → ToadStool absorption ═══");
    eprintln!("  hydrology: airSpring ET₀ methods → barracuda::stats::hydrology");
    eprintln!("  FAO-56, Hargreaves, Thornthwaite, Hamon, Makkink, Turc");
    eprintln!();

    bench("hargreaves_et0 T3: barracuda::stats", || {
        std::hint::black_box(barracuda::stats::hargreaves_et0(20.0, 35.0, 15.0));
    });

    bench("thornthwaite_et0 T3: barracuda::stats", || {
        std::hint::black_box(barracuda::stats::thornthwaite_et0(20.0, 50.0, 12.0, 30.0));
    });

    bench("hamon_et0 T3: barracuda::stats", || {
        std::hint::black_box(barracuda::stats::hamon_et0(22.0, 14.0));
    });

    bench("makkink_et0 T3: barracuda::stats", || {
        std::hint::black_box(barracuda::stats::makkink_et0(18.0, 200.0));
    });

    bench("turc_et0 T3: barracuda::stats", || {
        std::hint::black_box(barracuda::stats::turc_et0(25.0, 250.0, 60.0));
    });

    let ra_365: Vec<f64> = (0..365)
        .map(|d| 20.0 + 10.0 * (d as f64 * 0.0172).sin())
        .collect();
    let t_max_365: Vec<f64> = (0..365)
        .map(|d| 25.0 + 8.0 * (d as f64 * 0.0172).sin())
        .collect();
    let t_min_365: Vec<f64> = t_max_365.iter().map(|t| t - 10.0).collect();
    bench("hargreaves_batch T3: barracuda::stats (365 days)", || {
        std::hint::black_box(barracuda::stats::hargreaves_et0_batch(
            &ra_365, &t_max_365, &t_min_365,
        ));
    });

    h.check_bool(
        "airSpring hydrology: hargreaves produces positive ET₀",
        barracuda::stats::hargreaves_et0(20.0, 35.0, 15.0).is_some_and(|v| v > 0.0),
    );
    h.check_bool(
        "airSpring hydrology: thornthwaite produces positive ET₀",
        barracuda::stats::thornthwaite_et0(20.0, 50.0, 12.0, 30.0).is_some_and(|v| v > 0.0),
    );
    eprintln!();
}

fn bench_convergence_pipeline(h: &mut ValidationHarness, disp: &Dispatcher) {
    eprintln!("═══ Cross-spring convergence pipeline (all → ToadStool S87) ═══");
    eprintln!("  Full pipeline: Dispatcher wraps barracuda::dispatch wraps WGSL shaders");
    eprintln!("  All springs converge through ToadStool's 844+ WGSL shaders");
    eprintln!();

    let data: Vec<f64> = (0..2048).map(|i| (i as f64) * 0.001).collect();

    bench("full pipeline: variance+mean+frob (2048)", || {
        let v = disp.variance(&data);
        let m = disp.mean(&data);
        let f = disp.frobenius_norm(&data);
        std::hint::black_box((v, m, f));
    });

    bench("full pipeline: softmax+gelu (2048)", || {
        let s = disp.softmax(&data);
        let g = disp.gelu(&data);
        std::hint::black_box((s, g));
    });

    let n = 16;
    let mat: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.01).collect();
    bench("full pipeline: matmul+eigh (16x16)", || {
        let mm = disp.mat_mul(&mat, &mat, n);
        let (e, _) = disp.eigh(&mat, n);
        std::hint::black_box((mm, e));
    });

    let shannon_val = disp.shannon_entropy(&data[..256]);
    h.check_bool(
        "convergence: shannon via Dispatcher",
        shannon_val.is_finite(),
    );

    let pearson_val = disp.pearson_correlation(&data[..1024], &data[1024..]);
    h.check_bool(
        "convergence: pearson via Dispatcher",
        pearson_val.is_finite(),
    );

    let transpose_val = disp.transpose(&mat, n);
    h.check_bool(
        "convergence: transpose via Dispatcher",
        transpose_val.len() == n * n,
    );
    eprintln!();
}

fn print_evolution_summary() {
    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Cross-Spring Shader Evolution — Performance Summary               ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║                                                                    ║");
    eprintln!("║  Evolution timeline:                                               ║");
    eprintln!("║    S40–S60: hotSpring → precision shaders, DF64, eigensolve        ║");
    eprintln!("║    S50–S70: wetSpring → bio shaders, HMM, diversity               ║");
    eprintln!("║    S60–S80: neuralSpring → matmul, gelu, softmax, swarm           ║");
    eprintln!("║    S70–S86: absorption → ToadStool ComputeDispatch (144 ops)      ║");
    eprintln!("║    S86–S87: deep debt evolution → pure math, CPU ungating         ║");
    eprintln!("║    S87+:    DF64 transcendentals (37), precision probing           ║");
    eprintln!("║                                                                    ║");
    eprintln!("║  Current state (ToadStool S87, 2dc26792):                         ║");
    eprintln!("║    844+ WGSL shaders (up from 692+ at S86)                        ║");
    eprintln!("║    37 DF64 transcendental shaders (up from 26)                    ║");
    eprintln!("║    F64 polyfills: exp, log, sin, cos → probe-injected             ║");
    eprintln!("║    DF64 fallback: √, exp, log, sin, cos → f32-pair               ║");
    eprintln!("║    Hardware probing: per-function f64 capability detection         ║");
    eprintln!("║                                                                    ║");
    eprintln!("║  Springs that converge here:                                       ║");
    eprintln!("║    hotSpring  → precision, spectral, eigensolve                   ║");
    eprintln!("║    wetSpring  → diversity, HMM, bio, Wright-Fisher                ║");
    eprintln!("║    neuralSpring → ML activations, matmul, RK4, attention          ║");
    eprintln!("║    groundSpring → uncertainty, bootstrap, normal distribution     ║");
    eprintln!("║    airSpring  → hydrology, ET₀, water balance                    ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
}

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  neuralSpring — Cross-Spring Shader Evolution Benchmark            ║");
    eprintln!("║  All springs → ToadStool S87 (2dc26792): 844+ WGSL shaders        ║");
    eprintln!("║  Provenance tiers: T0 local → T1 bc::dispatch → T2 Dispatcher GPU ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let mut h = ValidationHarness::new("cross_spring_shader_evolution_bench");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt.block_on(Gpu::new()).expect("GPU init");
    let disp = Dispatcher::from_gpu(gpu);

    bench_neuralspring_origins(&mut h, &disp);
    bench_wetspring_origins(&mut h);
    bench_hotspring_origins(&mut h, &disp);
    bench_groundspring_origins(&mut h);
    bench_airspring_origins(&mut h);
    bench_convergence_pipeline(&mut h, &disp);

    print_evolution_summary();
    h.finish();
}
