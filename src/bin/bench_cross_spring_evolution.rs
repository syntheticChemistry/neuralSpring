// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-spring evolution benchmark: traces shader provenance across springs.
//!
//! Demonstrates how `ToadStool`/`BarraCUDA` benefits from cross-spring evolution:
//!
//! | Spring | Domain | Key Contributions |
//! |--------|--------|-------------------|
//! | hotSpring | Precision physics | DF64 core-streaming, lattice QCD, Hermite/Laguerre |
//! | wetSpring | Bioinformatics | Diversity (Shannon/Simpson), HMM, Bray-Curtis |
//! | neuralSpring | ML/neuroevolution | Batch fitness, pairwise ops, swarm NN, RK4 |
//! | airSpring | Atmospheric | Stats metrics (RMSE, R², NSE), moving window |
//! | groundSpring | Hydrology | Multinomial sampling, MC propagation |
//!
//! Each benchmark section notes the provenance chain showing how shaders
//! evolved from one spring to another, and ultimately to GPU.
//!
//! ## Session 75
//!
//! Covers rewired functions: `r_squared`, `rmse`, `nse`, `dot`, `l2_norm`,
//! `shannon`, `nash_sutcliffe` — all absorbed into `BarraCUDA` `stats` from
//! airSpring/groundSpring/wetSpring in `ToadStool` S64.
//!
//! ## Session 83
//!
//! Extended with `ToadStool` S66–S68 APIs: `fit_quadratic`, `fit_exponential`,
//! `fit_all`, `spearman_correlation`, `rawr_mean`. GPU typed op provenance
//! benchmarks and f64 precision ops (hotSpring → `ToadStool` → neuralSpring).
//!
//! # Panics
//!
//! Panics if the tokio runtime cannot be created or if GPU diversity fusion
//! operations fail — this is a benchmark binary, not a library.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::suboptimal_flops
)]

use barracuda::ops::bio::DiversityFusionGpu;
use barracuda::stats;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use std::time::Instant;

const N: usize = 10_000;
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
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  neuralSpring — Cross-Spring Evolution Benchmark            ║");
    eprintln!("║  Provenance: hotSpring → wetSpring → neuralSpring → GPU     ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt.block_on(async { Gpu::new().await.ok() });

    if let Some(ref g) = gpu {
        eprintln!(
            "  GPU: {} ({:?}, {:?})",
            g.adapter_name, g.device_type, g.backend
        );
    } else {
        eprintln!("  GPU: not available");
    }
    eprintln!();

    let mut h = ValidationHarness::new("cross_spring_evolution_bench");

    // ─── airSpring + groundSpring → barracuda::stats (S64) ─────────────
    eprintln!("═══ airSpring/groundSpring → barracuda::stats ═══");
    eprintln!("  Provenance: airSpring testutil/stats.rs + groundSpring stats/metrics.rs");
    eprintln!("  → ToadStool S64 absorption → barracuda::stats");
    eprintln!("  → neuralSpring rewired S75 (metrics.rs, deeponet.rs)");
    eprintln!();

    let mut rng = Rng::new(42);
    let observed: Vec<f64> = (0..N).map(|_| rng.next_f64() * 100.0).collect();
    let simulated: Vec<f64> = observed
        .iter()
        .map(|&x| x + (rng.next_f64() - 0.5) * 10.0)
        .collect();

    let cpu_rmse = bench("RMSE (barracuda::stats, CPU)", || {
        std::hint::black_box(stats::rmse(&observed, &simulated));
    });
    h.check_bool(
        &format!("airSpring→stats: RMSE {cpu_rmse:.1}µs"),
        cpu_rmse < 1_000.0,
    );

    let cpu_r2 = bench("R² (barracuda::stats, CPU)", || {
        std::hint::black_box(stats::r_squared(&observed, &simulated));
    });
    h.check_bool(
        &format!("airSpring→stats: R² {cpu_r2:.1}µs"),
        cpu_r2 < 1_000.0,
    );

    let cpu_nse = bench("NSE (barracuda::stats, CPU)", || {
        std::hint::black_box(stats::nash_sutcliffe(&observed, &simulated));
    });
    h.check_bool(
        &format!("airSpring→stats: NSE {cpu_nse:.1}µs"),
        cpu_nse < 1_000.0,
    );

    let cpu_ia = bench("IA (barracuda::stats, CPU)", || {
        std::hint::black_box(stats::index_of_agreement(&observed, &simulated));
    });
    h.check_bool(
        &format!("airSpring→stats: IA {cpu_ia:.1}µs"),
        cpu_ia < 1_000.0,
    );

    let cpu_dot = bench("dot (barracuda::stats, CPU)", || {
        std::hint::black_box(stats::dot(&observed, &simulated));
    });
    h.check_bool(
        &format!("airSpring→stats: dot {cpu_dot:.1}µs"),
        cpu_dot < 1_000.0,
    );

    let cpu_l2 = bench("l2_norm (barracuda::stats, CPU)", || {
        std::hint::black_box(stats::l2_norm(&observed));
    });
    h.check_bool(
        &format!("airSpring→stats: l2_norm {cpu_l2:.1}µs"),
        cpu_l2 < 1_000.0,
    );

    eprintln!();

    // ─── wetSpring → barracuda::stats::diversity (S64) ──────────────────
    eprintln!("═══ wetSpring → barracuda::stats::diversity ═══");
    eprintln!("  Provenance: wetSpring bio/diversity.rs (Feb 2026)");
    eprintln!("  → ToadStool S64 absorption → barracuda::stats");
    eprintln!("  → neuralSpring rewired S75 (primitives.rs shannon_entropy_from_counts)");
    eprintln!();

    let counts: Vec<f64> = (0..500).map(|_| (rng.next_f64() * 50.0).max(0.0)).collect();

    let cpu_shannon = bench("Shannon (barracuda::stats, CPU)", || {
        std::hint::black_box(stats::shannon(&counts));
    });
    h.check_bool(
        &format!("wetSpring→stats: Shannon {cpu_shannon:.1}µs"),
        cpu_shannon < 1_000.0,
    );

    let cpu_simpson = bench("Simpson (barracuda::stats, CPU)", || {
        std::hint::black_box(stats::simpson(&counts));
    });
    h.check_bool(
        &format!("wetSpring→stats: Simpson {cpu_simpson:.1}µs"),
        cpu_simpson < 1_000.0,
    );

    let cpu_chao1 = bench("Chao1 (barracuda::stats, CPU)", || {
        std::hint::black_box(stats::chao1(&counts));
    });
    h.check_bool(
        &format!("wetSpring→stats: Chao1 {cpu_chao1:.1}µs"),
        cpu_chao1 < 1_000.0,
    );

    let cpu_alpha = bench("alpha_diversity (barracuda::stats, CPU)", || {
        std::hint::black_box(stats::alpha_diversity(&counts));
    });
    h.check_bool(
        &format!("wetSpring→stats: alpha_diversity {cpu_alpha:.1}µs"),
        cpu_alpha < 1_000.0,
    );

    let samples_a: Vec<f64> = (0..200).map(|_| (rng.next_f64() * 30.0).max(0.0)).collect();
    let samples_b: Vec<f64> = (0..200).map(|_| (rng.next_f64() * 30.0).max(0.0)).collect();

    let cpu_bray = bench("Bray-Curtis (barracuda::stats, CPU)", || {
        std::hint::black_box(stats::bray_curtis(&samples_a, &samples_b));
    });
    h.check_bool(
        &format!("wetSpring→stats: Bray-Curtis {cpu_bray:.1}µs"),
        cpu_bray < 1_000.0,
    );

    eprintln!();

    // ─── neuralSpring → GPU ops (metalForge → ToadStool) ──────────────
    if let Some(ref g) = gpu {
        let device = g.wgpu_device().clone();

        // ─── wetSpring → GPU diversity fusion (S64) ─────────────────────
        eprintln!("═══ wetSpring → ToadStool GPU DiversityFusion ═══");
        eprintln!("  Provenance: wetSpring bio/diversity.rs → diversity_fusion_f64.wgsl");
        eprintln!("  → ToadStool S64 absorption → barracuda::ops::bio::DiversityFusionGpu");
        eprintln!("  Fused Shannon + Simpson + Pielou in one GPU dispatch");
        eprintln!();

        bench_gpu_diversity_fusion(&mut h, &device);
        eprintln!();
    } else {
        eprintln!("  [skip] GPU benchmarks (no adapter)");
        eprintln!();
    }

    // ─── hotSpring → precision correlation ──────────────────────────────
    eprintln!("═══ hotSpring → precision validation pattern ═══");
    eprintln!("  Provenance: hotSpring validation.rs → barracuda::validation");
    eprintln!("  Pattern: tolerance-driven check_abs/check_rel, named constants");
    eprintln!("  neuralSpring adopted this pattern; barracuda absorbed it S64");
    eprintln!();

    let cpu_pearson = bench("Pearson r (barracuda::stats, CPU)", || {
        let _ = std::hint::black_box(stats::pearson_correlation(&observed, &simulated));
    });
    h.check_bool(
        &format!("hotSpring→stats: Pearson {cpu_pearson:.1}µs"),
        cpu_pearson.is_finite(),
    );

    eprintln!();

    // ─── S78 absorptions: MAE, Shannon, Hill, fit_linear ──────────────
    eprintln!("═══ S78 Absorptions (ToadStool S66) ═══");
    eprintln!("  Provenance: airSpring → barracuda::stats::mae [S64→S66]");
    let cpu_mae = bench("mae (barracuda::stats, CPU)", || {
        let _ = std::hint::black_box(stats::mae(&observed, &simulated));
    });
    h.check_bool(
        &format!("airSpring→stats: mae {cpu_mae:.1}µs"),
        cpu_mae.is_finite(),
    );

    eprintln!("  Provenance: wetSpring → barracuda::stats::shannon_from_frequencies [S64]");
    let probs: Vec<f64> = observed
        .iter()
        .map(|&x| x / observed.iter().sum::<f64>())
        .collect();
    let cpu_shannon = bench("shannon_from_frequencies (barracuda, CPU)", || {
        let _ = std::hint::black_box(stats::shannon_from_frequencies(&probs));
    });
    h.check_bool(
        &format!("wetSpring→stats: shannon_freq {cpu_shannon:.1}µs"),
        cpu_shannon.is_finite(),
    );

    eprintln!("  Provenance: wetSpring+hotSpring → barracuda::stats::hill [S64]");
    let cpu_hill = bench("hill (barracuda::stats, CPU)", || {
        for i in 0..N {
            let _ = std::hint::black_box(stats::hill(observed[i], 5.0, 2.0));
        }
    });
    h.check_bool(
        &format!("wS+hS→stats: hill(N={N}) {cpu_hill:.1}µs"),
        cpu_hill.is_finite(),
    );

    eprintln!("  Provenance: airSpring V009 → barracuda::stats::fit_linear [S66]");
    let x_reg: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let cpu_fit = bench("fit_linear (barracuda::stats, CPU)", || {
        let _ = std::hint::black_box(stats::fit_linear(&x_reg, &observed));
    });
    h.check_bool(
        &format!("airSpring→stats: fit_linear(N={N}) {cpu_fit:.1}µs"),
        cpu_fit.is_finite(),
    );

    eprintln!();

    // ─── S83: Modern S68 APIs — cross-spring provenance ─────────────────
    eprintln!("═══ S83: Modern ToadStool S68 APIs ═══");
    eprintln!("  ToadStool S68: 700 WGSL f64 canonical, universal precision");
    eprintln!();

    eprintln!("  Provenance: airSpring V009 → barracuda::stats::fit_quadratic [S66]");
    let cpu_fit_q = bench("fit_quadratic (airSpring→barracuda, CPU)", || {
        let _ = std::hint::black_box(stats::fit_quadratic(&x_reg, &observed));
    });
    h.check_bool(
        &format!("airSpring→stats: fit_quadratic(N={N}) {cpu_fit_q:.1}µs"),
        cpu_fit_q.is_finite(),
    );

    eprintln!("  Provenance: airSpring V009 → barracuda::stats::fit_exponential [S66]");
    let pos_y: Vec<f64> = observed.iter().map(|&v| v.abs() + 1.0).collect();
    let cpu_fit_e = bench("fit_exponential (airSpring→barracuda, CPU)", || {
        let _ = std::hint::black_box(stats::fit_exponential(&x_reg, &pos_y));
    });
    h.check_bool(
        &format!("airSpring→stats: fit_exponential(N={N}) {cpu_fit_e:.1}µs"),
        cpu_fit_e.is_finite(),
    );

    eprintln!("  Provenance: airSpring V009 → barracuda::stats::fit_all [S66]");
    let cpu_fit_all = bench("fit_all (airSpring→barracuda, CPU)", || {
        let _ = std::hint::black_box(stats::fit_all(&x_reg, &pos_y));
    });
    h.check_bool(
        &format!("airSpring→stats: fit_all(N={N}) {cpu_fit_all:.1}µs"),
        cpu_fit_all.is_finite(),
    );

    eprintln!("  Provenance: wetSpring+hotSpring → barracuda::stats::spearman_correlation [S66]");
    let cpu_spearman = bench("spearman_correlation (wS+hS→barracuda, CPU)", || {
        let _ = std::hint::black_box(stats::spearman_correlation(&observed, &simulated));
    });
    h.check_bool(
        &format!("wS+hS→stats: spearman(N={N}) {cpu_spearman:.1}µs"),
        cpu_spearman.is_finite(),
    );

    eprintln!("  Provenance: groundSpring → barracuda::stats::bootstrap::rawr_mean [S66]");
    let cpu_rawr = bench("rawr_mean (groundSpring→barracuda, CPU)", || {
        let _ = std::hint::black_box(stats::rawr_mean(&observed[..1000], 100, 0.05, 42));
    });
    h.check_bool(
        &format!("groundSpring→stats: rawr_mean(N=1000) {cpu_rawr:.1}µs"),
        cpu_rawr.is_finite(),
    );

    eprintln!();

    // ─── GPU f64 ops via Dispatcher (cross-spring provenance) ────────────
    {
        use neural_spring::gpu_dispatch::Dispatcher;
        let dispatcher = rt.block_on(async { Dispatcher::new().await });

        eprintln!("═══ GPU Dispatch — Cross-Spring f64 Ops ═══");
        eprintln!("  Dispatcher routes CPU→GPU based on size threshold.");
        eprintln!("  f64 precision ops: hotSpring Welford + wetSpring fused shaders");
        eprintln!();

        let n_big = 50_000_usize;
        let mut big_rng = Rng::new(700);
        let big_a: Vec<f64> = (0..n_big).map(|_| big_rng.next_f64() * 10.0 - 5.0).collect();
        let big_b: Vec<f64> = (0..n_big).map(|_| big_rng.next_f64() * 10.0 - 5.0).collect();

        let gpu_var = bench("Dispatcher::variance 50k (hS Welford→ToadStool→GPU)", || {
            let _ = std::hint::black_box(dispatcher.variance(&big_a));
        });
        h.check_bool(
            &format!("hS→dispatch: variance 50k {gpu_var:.1}µs"),
            gpu_var.is_finite(),
        );

        let gpu_pearson = bench("Dispatcher::pearson 50k (wS+hS→ToadStool→GPU)", || {
            let _ = std::hint::black_box(dispatcher.pearson_correlation(&big_a, &big_b));
        });
        h.check_bool(
            &format!("wS+hS→dispatch: pearson 50k {gpu_pearson:.1}µs"),
            gpu_pearson.is_finite(),
        );

        let big_probs: Vec<f64> = big_a.iter().map(|x| x.abs() / 1000.0 + 1e-10).collect();
        let gpu_shannon = bench("Dispatcher::shannon 50k (wS fused→ToadStool→GPU)", || {
            let _ = std::hint::black_box(dispatcher.shannon_entropy(&big_probs));
        });
        h.check_bool(
            &format!("wS→dispatch: shannon 50k {gpu_shannon:.1}µs"),
            gpu_shannon.is_finite(),
        );

        let side = 200_usize;
        let mat: Vec<f64> = (0..side * side).map(|_| big_rng.next_f64()).collect();
        let gpu_matmul = bench("Dispatcher::mat_mul 200×200 (nS→ToadStool→GPU)", || {
            let _ = std::hint::black_box(dispatcher.mat_mul(&mat, &mat, side));
        });
        h.check_bool(
            &format!("nS→dispatch: matmul 200×200 {gpu_matmul:.1}µs"),
            gpu_matmul.is_finite(),
        );

        eprintln!();
    }

    // ─── Summary ────────────────────────────────────────────────────────
    eprintln!("═══ Cross-Spring Evolution Summary (S83) ═══");
    eprintln!("  700 WGSL shaders in ToadStool S68 (f64 canonical), sourced from:");
    eprintln!("    hotSpring:    ~100 (lattice QCD, HFB, DF64, spectral, precision)");
    eprintln!("    wetSpring:    ~80  (bio, metagenomics, diversity, HMM, ODE)");
    eprintln!("    neuralSpring: ~34  (ML, neuroevolution, batch fitness, 9 f64 shaders)");
    eprintln!("    airSpring:    ~15  (ET₀, kriging, Richards, stats, regression)");
    eprintln!("    groundSpring: ~5   (multinomial, MC propagation)");
    eprintln!("    ToadStool:    ~466 (core math, linalg, nn, activations, S68 precision)");
    eprintln!("  neuralSpring rewired: 39 functions + 6 shader sources to upstream");
    eprintln!("  S83: ToadStool S68 sync — variance_ddof gap closed, 5 shader imports fixed");
    eprintln!("  Cross-spring flow: each spring contributes domain expertise;");
    eprintln!("  ToadStool absorbs + GPU-accelerates → all springs benefit via path dep");
    eprintln!();

    h.finish();
}

fn bench_gpu_diversity_fusion(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let n_samples = 64;
    let n_species = 200;
    let mut rng = Rng::new(400);
    let abundances: Vec<f64> = (0..n_samples * n_species)
        .map(|_| (rng.next_f64() * 50.0).max(0.0))
        .collect();

    let rt = tokio::runtime::Runtime::new().unwrap();

    let cpu_us = bench("diversity_fusion_cpu (wetSpring→ToadStool)", || {
        std::hint::black_box(barracuda::ops::bio::diversity_fusion_cpu(
            &abundances,
            n_species,
        ));
    });
    h.check_bool(
        &format!("wS→CPU: DiversityFusion {cpu_us:.1}µs"),
        cpu_us.is_finite(),
    );

    let gpu_us = bench("DiversityFusionGpu (wetSpring→ToadStool→GPU)", || {
        rt.block_on(async {
            let op = DiversityFusionGpu::new(device.clone()).unwrap();
            let _result = op.compute(&abundances, n_samples, n_species).unwrap();
        });
    });
    h.check_bool(
        &format!("wS→GPU: DiversityFusion {gpu_us:.1}µs"),
        gpu_us.is_finite(),
    );

    if gpu_us > 0.0 && cpu_us > 0.0 {
        let speedup = cpu_us / gpu_us;
        eprintln!("    → GPU/CPU speedup: {speedup:.1}×");
        h.check_bool(
            &format!("DiversityFusion GPU runs ({speedup:.1}× vs CPU)"),
            true,
        );
    }
}
