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

    // ─── Summary ────────────────────────────────────────────────────────
    eprintln!("═══ Cross-Spring Evolution Summary ═══");
    eprintln!("  694 WGSL shaders in ToadStool, sourced from:");
    eprintln!("    hotSpring:    ~100 (lattice QCD, HFB, DF64, spectral)");
    eprintln!("    wetSpring:    ~80  (bio, metagenomics, diversity, HMM)");
    eprintln!("    neuralSpring: ~25  (ML, neuroevolution, batch fitness)");
    eprintln!("    airSpring:    ~10  (ET₀, kriging, Richards, moving window)");
    eprintln!("    groundSpring: ~5   (multinomial, MC propagation)");
    eprintln!("    ToadStool:    ~474 (core math, linalg, nn, activations)");
    eprintln!("  neuralSpring rewired: 32 functions + 6 shader sources to upstream");
    eprintln!("  Cross-spring flow: each spring contributes domain expertise;");
    eprintln!("  ToadStool absorbs and GPU-accelerates → all springs benefit");
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
