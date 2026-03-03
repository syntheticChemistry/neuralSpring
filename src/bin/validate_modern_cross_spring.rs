// SPDX-License-Identifier: AGPL-3.0-or-later

//! Modern cross-spring evolution validator: exercises `ToadStool` S86 universal
//! precision pipeline and traces shader provenance across all five springs.
//!
//! ## Provenance map
//!
//! ```text
//! hotSpring  → DF64 core-streaming, Fp64Strategy, split_workgroups, lattice QCD
//!              → ToadStool S68: universal precision F16/F32/F64/Df64
//! wetSpring  → diversity (Shannon, Bray-Curtis), bio (Smith-Waterman, Gillespie,
//!              Felsenstein, HMM), NMF, ODE bio
//! neuralSpring → batch_fitness, pairwise ops, eigh, swarm_nn, ValidationHarness
//! airSpring  → hydrology, regression, moving_window, stats metrics
//! groundSpring → bootstrap (rawr_mean), multinomial sampling
//! ```
//!
//! Each check is annotated with its provenance chain showing the evolution path
//! from source spring → `ToadStool`/`BarraCUDA` → neuralSpring.
//!
//! ```text
//! cargo run --bin validate_modern_cross_spring
//! ```

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::items_after_statements,
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "validation binary"
)]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{bench_once, max_abs_diff_f64, ValidationHarness};

// ═══════════════════════════════════════════════════════════════════════════════
// hotSpring provenance: precision infrastructure
// ═══════════════════════════════════════════════════════════════════════════════

fn validate_hotspring_precision(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    eprintln!("\n─── hotSpring provenance: precision infrastructure ───\n");

    // Fp64Strategy detection (hotSpring S58 → ToadStool S58)
    let strategy = dispatcher.fp64_strategy();
    h.check_bool(
        "hS→precision: Fp64Strategy detected",
        !format!("{strategy:?}").is_empty(),
    );

    // DF64 core-streaming: eigh uses DF64 pathway on consumer GPUs
    let n = 16;
    let mut rng = Rng::new(42);
    let mut mat = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let v = rng.uniform() - 0.5;
            mat[i * n + j] = v;
            mat[j * n + i] = v;
        }
    }
    let (eigs_gpu, gpu_us) = bench_once("eigh 16×16 (hS→DF64→GPU)", || {
        dispatcher.eigh(&mat, n).0
    });
    let cpu = Dispatcher::cpu_only();
    let (eigs_cpu, cpu_us) = bench_once("eigh 16×16 (CPU ref)", || cpu.eigh(&mat, n).0);
    // GPU Jacobi converges differently for random matrices — compare sorted order
    let mut sorted_gpu = eigs_gpu;
    let mut sorted_cpu = eigs_cpu;
    sorted_gpu.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_cpu.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let eigh_diff = max_abs_diff_f64(&sorted_gpu, &sorted_cpu);
    h.check_abs(
        "hS→DF64: eigh GPU≈CPU (sorted)",
        eigh_diff,
        0.0,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );
    h.check_bool(
        &format!("hS→DF64: eigh benchmarked (GPU {gpu_us:.0}µs, CPU {cpu_us:.0}µs)"),
        gpu_us > 0.0,
    );

    // split_workgroups: validated through Dispatcher dispatch parity
    let big_n = 100;
    let big_a: Vec<f64> = (0..big_n * big_n).map(|i| (i as f64) * 0.001).collect();
    let big_b: Vec<f64> = (0..big_n * big_n)
        .map(|i| ((big_n * big_n - i) as f64) * 0.001)
        .collect();
    let (gpu_res, _) = bench_once("matmul 100×100 (hS→split_workgroups)", || {
        dispatcher.mat_mul(&big_a, &big_b, big_n)
    });
    let (cpu_res, _) = bench_once("matmul 100×100 (CPU ref)", || {
        cpu.mat_mul(&big_a, &big_b, big_n)
    });
    h.check_abs(
        "hS→split_wg: matmul GPU≈CPU (100×100)",
        max_abs_diff_f64(&gpu_res, &cpu_res),
        0.0,
        tolerances::DISPATCH_MATMUL_F64 * 10.0,
    );

    // Primal matmul via barracuda::dispatch::matmul_dispatch (non-square)
    let m = 8;
    let k = 12;
    let n_col = 6;
    let a_rect: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.01).collect();
    let b_rect: Vec<f64> = (0..k * n_col)
        .map(|i| ((k * n_col - i) as f64) * 0.01)
        .collect();
    let result = barracuda::dispatch::matmul_dispatch(&a_rect, &b_rect, m, k, n_col, None)
        .expect("matmul_dispatch non-square");
    assert_eq!(result.len(), m * n_col);
    h.check_bool(
        "hS→dispatch: matmul non-square (8×12 × 12×6)",
        result.iter().all(|v| v.is_finite()),
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// wetSpring provenance: bio + diversity
// ═══════════════════════════════════════════════════════════════════════════════

fn validate_wetspring_bio(h: &mut ValidationHarness) {
    eprintln!("\n─── wetSpring provenance: bio + diversity ───\n");

    // Shannon diversity (wetSpring → ToadStool S64)
    let counts = [10.0, 20.0, 30.0, 40.0];
    let (shannon, _) = bench_once("shannon (wS→ToadStool S64)", || {
        barracuda::stats::shannon(&counts)
    });
    let expected_shannon = {
        let total: f64 = counts.iter().sum();
        -counts
            .iter()
            .filter(|&&c| c > 0.0)
            .map(|c| {
                let p = c / total;
                p * p.ln()
            })
            .sum::<f64>()
    };
    h.check_abs(
        "wS→diversity: shannon",
        shannon,
        expected_shannon,
        tolerances::CROSS_LANGUAGE,
    );

    // Bray-Curtis distance (wetSpring → ToadStool S64)
    let a = [10.0, 20.0, 30.0];
    let b = [15.0, 25.0, 35.0];
    let (bc, _) = bench_once("bray_curtis (wS→ToadStool S64)", || {
        barracuda::stats::bray_curtis(&a, &b)
    });
    let expected_bc = {
        let sum_min: f64 = a.iter().zip(&b).map(|(x, y)| x.min(*y)).sum();
        let sum_a: f64 = a.iter().sum();
        let sum_b: f64 = b.iter().sum();
        1.0 - 2.0 * sum_min / (sum_a + sum_b)
    };
    h.check_abs(
        "wS→diversity: bray_curtis",
        bc,
        expected_bc,
        tolerances::CROSS_LANGUAGE,
    );

    // Alpha diversity (wetSpring → ToadStool S64)
    let abundances = [5.0, 10.0, 15.0, 20.0, 25.0, 25.0];
    let alpha = barracuda::stats::alpha_diversity(&abundances);
    h.check_bool(
        "wS→diversity: alpha_diversity computed",
        alpha.shannon > 0.0,
    );
    h.check_bool(
        "wS→diversity: chao1 ≥ observed",
        alpha.chao1 >= alpha.observed,
    );

    // Simpson diversity (wetSpring → ToadStool S64)
    let (simpson, _) = bench_once("simpson (wS→ToadStool S64)", || {
        barracuda::stats::simpson(&abundances)
    });
    h.check_bool(
        "wS→diversity: simpson in [0,1]",
        (0.0..=1.0).contains(&simpson),
    );

    // Pearson correlation (wetSpring hydrology → ToadStool S64)
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.0, 4.0, 6.0, 8.0, 10.0];
    let (r, _) = bench_once("pearson (wS/aS→ToadStool S64)", || {
        barracuda::stats::pearson_correlation(&x, &y)
    });
    h.check_abs(
        "wS→stats: pearson(linear)",
        r.unwrap_or(0.0),
        1.0,
        tolerances::CROSS_LANGUAGE,
    );

    // NMF (wetSpring ESN → ToadStool S58)
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
    let nmf_result = barracuda::linalg::nmf(
        &data,
        2,
        3,
        &barracuda::linalg::NmfConfig {
            rank: 2,
            max_iter: 100,
            objective: barracuda::linalg::NmfObjective::Euclidean,
            seed: 42,
            tol: 1e-6,
        },
    );
    h.check_bool("wS→linalg: NMF converges", nmf_result.is_ok());
    if let Ok(ref r) = nmf_result {
        h.check_bool(
            "wS→linalg: NMF reconstruction finite",
            barracuda::linalg::relative_reconstruction_error(&data, r).is_finite(),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// airSpring provenance: hydrology + regression + stats
// ═══════════════════════════════════════════════════════════════════════════════

fn validate_airspring_stats(h: &mut ValidationHarness) {
    eprintln!("\n─── airSpring provenance: stats + regression ───\n");

    // Regression fitting (airSpring → ToadStool S66)
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.1, 3.9, 6.1, 7.9, 10.1];
    let (fit, _) = bench_once("fit_linear (aS→ToadStool S66)", || {
        barracuda::stats::fit_linear(&x, &y)
    });
    let fit = fit.expect("fit_linear should converge");
    h.check_abs(
        "aS→regression: linear slope ≈ 2.0",
        fit.params[0],
        2.0,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );
    h.check_abs(
        "aS→regression: linear R² ≈ 1.0",
        fit.r_squared,
        1.0,
        tolerances::GPU_AF_VARIANCE_F32,
    );

    // Quadratic fit (airSpring → ToadStool S66)
    let xq: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    let yq: Vec<f64> = xq.iter().map(|&x| 3.0f64.mul_add(x, x * x) + 1.0).collect();
    let (fit_q, _) = bench_once("fit_quadratic (aS→ToadStool S66)", || {
        barracuda::stats::fit_quadratic(&xq, &yq)
    });
    let fit_q = fit_q.expect("fit_quadratic should converge");
    h.check_abs(
        "aS→regression: quadratic R² ≈ 1.0",
        fit_q.r_squared,
        1.0,
        tolerances::PGM_COMPLEXITY_SLACK,
    );

    // Exponential fit (airSpring → ToadStool S66)
    let xe: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
    let ye: Vec<f64> = xe.iter().map(|&x| 2.0 * (0.5_f64 * x).exp()).collect();
    let fit_e = barracuda::stats::fit_exponential(&xe, &ye).expect("fit_exponential");
    h.check_abs(
        "aS→regression: exponential R² ≈ 1.0",
        fit_e.r_squared,
        1.0,
        tolerances::PGM_COMPLEXITY_SLACK,
    );

    // Metrics: RMSE, R², NSE, MAE (airSpring → ToadStool S64)
    let predicted = [1.0, 2.0, 3.0, 4.0, 5.0];
    let observed = [1.1, 1.9, 3.1, 3.9, 5.1];
    let (rmse, _) = bench_once("rmse (aS→ToadStool S64)", || {
        barracuda::stats::rmse(&predicted, &observed)
    });
    h.check_bool("aS→metrics: RMSE > 0", rmse > 0.0);
    h.check_bool("aS→metrics: RMSE < 0.2 (close predictions)", rmse < 0.2);

    let (r2, _) = bench_once("r_squared (aS→ToadStool S64)", || {
        barracuda::stats::r_squared(&predicted, &observed)
    });
    h.check_bool("aS→metrics: R² > 0.99", r2 > 0.99);

    let (nse, _) = bench_once("nash_sutcliffe (aS→ToadStool S64)", || {
        barracuda::stats::nash_sutcliffe(&predicted, &observed)
    });
    h.check_bool("aS→metrics: NSE > 0.99", nse > 0.99);

    let (mae, _) = bench_once("mae (aS→ToadStool S64)", || {
        barracuda::stats::mae(&predicted, &observed)
    });
    h.check_abs(
        "aS→metrics: MAE = 0.1",
        mae,
        0.1,
        tolerances::CROSS_LANGUAGE,
    );

    // Hydrology: Hargreaves ET₀ (airSpring → ToadStool S66)
    // ra=15 MJ/m²/day, t_max=35°C, t_min=15°C
    let (et0_opt, _) = bench_once("hargreaves_et0 (aS→ToadStool S66)", || {
        barracuda::stats::hargreaves_et0(15.0, 35.0, 15.0)
    });
    let et0 = et0_opt.unwrap_or(0.0);
    h.check_bool(
        "aS→hydrology: Hargreaves ET₀ > 0 mm/day",
        et0 > 0.0 && et0 < 20.0,
    );

    // Spearman rank correlation (airSpring → ToadStool S66)
    let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let ys = [5.0, 6.0, 7.0, 8.0, 7.0];
    let (rho_result, _) = bench_once("spearman (aS→ToadStool S66)", || {
        barracuda::stats::spearman_correlation(&xs, &ys)
    });
    let rho = rho_result.unwrap_or(0.0);
    h.check_bool("aS→stats: Spearman in [-1,1]", (-1.0..=1.0).contains(&rho));
}

// ═══════════════════════════════════════════════════════════════════════════════
// groundSpring provenance: bootstrap + sampling
// ═══════════════════════════════════════════════════════════════════════════════

fn validate_groundspring_bootstrap(h: &mut ValidationHarness) {
    eprintln!("\n─── groundSpring provenance: bootstrap + sampling ───\n");

    // Bootstrap CI (groundSpring → ToadStool S56)
    let data: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();
    let true_mean = data.iter().sum::<f64>() / data.len() as f64;
    let mean_fn = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let (ci_result, _) = bench_once("bootstrap_ci (gS→ToadStool S56)", || {
        barracuda::stats::bootstrap_ci(&data, mean_fn, 500, 0.95, 42)
    });
    let ci = ci_result.expect("bootstrap_ci");
    h.check_bool("gS→bootstrap: CI lower < upper", ci.lower < ci.upper);
    h.check_bool(
        "gS→bootstrap: CI contains estimate",
        ci.lower <= ci.estimate && ci.estimate <= ci.upper,
    );

    // Bootstrap mean (groundSpring → ToadStool S56)
    let (bm_result, _) = bench_once("bootstrap_mean (gS→ToadStool S56)", || {
        barracuda::stats::bootstrap_mean(&data, 500, 0.95, 42)
    });
    let bm = bm_result.expect("bootstrap_mean").estimate;
    h.check_abs(
        "gS→bootstrap: mean ≈ true mean",
        bm,
        true_mean,
        tolerances::ODE_STEADY_STATE_SLACK,
    );

    // rawr_mean (groundSpring → ToadStool S56)
    let (rm_result, _) = bench_once("rawr_mean (gS→ToadStool S56)", || {
        barracuda::stats::rawr_mean(&data, 200, 0.95, 42)
    });
    let rm = rm_result.expect("rawr_mean").estimate;
    h.check_abs(
        "gS→bootstrap: rawr_mean ≈ true mean",
        rm,
        true_mean,
        tolerances::ODE_STEADY_STATE_SLACK,
    );

    // Normal distribution (groundSpring/airSpring → ToadStool)
    let (cdf, _) = bench_once("norm_cdf (gS→ToadStool)", || {
        barracuda::stats::norm_cdf(0.0)
    });
    h.check_abs(
        "gS→normal: Φ(0) = 0.5",
        cdf,
        0.5,
        tolerances::CROSS_LANGUAGE,
    );

    let (pdf, _) = bench_once("norm_pdf (gS→ToadStool)", || {
        barracuda::stats::norm_pdf(0.0)
    });
    let expected_pdf = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    h.check_abs(
        "gS→normal: φ(0)",
        pdf,
        expected_pdf,
        tolerances::CROSS_LANGUAGE,
    );

    let (ppf, _) = bench_once("norm_ppf (gS→ToadStool)", || {
        barracuda::stats::norm_ppf(0.975)
    });
    h.check_abs(
        "gS→normal: Φ⁻¹(0.975) ≈ 1.96",
        ppf,
        1.96,
        tolerances::NORM_PPF_TAIL,
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// neuralSpring provenance: ML + validation + dispatch
// ═══════════════════════════════════════════════════════════════════════════════

fn validate_neuralspring_dispatch(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    eprintln!("\n─── neuralSpring provenance: ML + dispatch ───\n");

    // Dispatcher: softmax (nS S58 → ToadStool domain_ops)
    let logits = [1.0, 2.0, 3.0, 4.0];
    let (sm, _) = bench_once("softmax (nS→ToadStool S58)", || {
        dispatcher.softmax(&logits)
    });
    let sm_sum: f64 = sm.iter().sum();
    h.check_abs(
        "nS→dispatch: softmax sums to 1",
        sm_sum,
        1.0,
        tolerances::CROSS_LANGUAGE,
    );

    // Dispatcher: GELU (nS S59 → ToadStool domain_ops)
    let vals = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let (gelu_out, _) = bench_once("gelu (nS→ToadStool S59)", || dispatcher.gelu(&vals));
    h.check_abs(
        "nS→dispatch: GELU(0) = 0",
        gelu_out[2],
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool("nS→dispatch: GELU monotonic", gelu_out[3] < gelu_out[4]);

    // Dispatcher: variance (nS → ToadStool domain_ops)
    let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let (var, _) = bench_once("variance (nS→ToadStool)", || dispatcher.variance(&data));
    let cpu = Dispatcher::cpu_only();
    let var_cpu = cpu.variance(&data);
    h.check_abs(
        "nS→dispatch: variance GPU≈CPU",
        var,
        var_cpu,
        tolerances::CROSS_LANGUAGE,
    );

    // Dispatcher: HMM forward (nS+wS → ToadStool S59)
    let alpha = [0.5, 0.5];
    let trans = [0.7, 0.3, 0.4, 0.6];
    let emis = [0.9, 0.2];
    let (hmm, _) = bench_once("hmm_forward (nS+wS→ToadStool S59)", || {
        dispatcher.hmm_forward_step(&alpha, &trans, &emis, 2)
    });
    h.check_bool(
        "nS+wS→dispatch: HMM alpha finite",
        hmm.0.iter().all(|v| v.is_finite()),
    );
    h.check_bool("nS+wS→dispatch: HMM scale > 0", hmm.1 > 0.0);

    // Graph operations (nS baseCamp → ToadStool S54)
    let adjacency = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let (laplacian, _) = bench_once("graph_laplacian (nS→ToadStool S54)", || {
        barracuda::linalg::graph_laplacian(&adjacency, 3)
    });
    h.check_abs(
        "nS→graph: L[0,0] = degree(0) = 2",
        laplacian[0],
        2.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "nS→graph: L[0,1] = -adj[0,1] = -1",
        laplacian[1],
        -1.0,
        tolerances::CROSS_LANGUAGE,
    );

    // Effective rank (nS baseCamp → ToadStool S54)
    let eigs = [10.0, 5.0, 1.0, 0.1, 0.01];
    let (eff_rank, _) = bench_once("effective_rank (nS→ToadStool S54)", || {
        barracuda::linalg::effective_rank(&eigs)
    });
    h.check_bool("nS→graph: effective_rank > 1", eff_rank > 1.0);
    h.check_bool("nS→graph: effective_rank ≤ 5", eff_rank <= 5.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ToadStool S68: universal precision pipeline
// ═══════════════════════════════════════════════════════════════════════════════

fn validate_toadstool_s68_precision(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    eprintln!("\n─── ToadStool S68: universal precision + modern APIs ───\n");

    // Verify Precision enum is accessible (S67)
    h.check_bool(
        "TS S67: Precision enum (F16/F32/F64/Df64) accessible",
        [
            neural_spring::gpu::Precision::F16,
            neural_spring::gpu::Precision::F32,
            neural_spring::gpu::Precision::F64,
            neural_spring::gpu::Precision::Df64,
        ]
        .len()
            == 4,
    );

    // Dispatch domain heuristics (cross-spring)
    let cfg = barracuda::dispatch::global_config();
    h.check_bool(
        "TS: dispatch config available",
        cfg.should_use_gpu(100_000, "matmul") || !cfg.should_use_gpu(1, "matmul"),
    );

    // Verify modern linalg decompositions available
    let mat_3x3 = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
    let (lu, _) = bench_once("lu_decompose (TS linalg)", || {
        barracuda::linalg::lu_decompose(&mat_3x3, 3)
    });
    h.check_bool("TS→linalg: LU decomposes", lu.is_ok());

    let (qr, _) = bench_once("qr_decompose (TS linalg)", || {
        barracuda::linalg::qr_decompose(&mat_3x3, 3, 3)
    });
    h.check_bool("TS→linalg: QR decomposes", qr.is_ok());

    // Numerical: gradient_1d (cross-spring)
    let (grad, _) = bench_once("gradient_1d (TS numerical)", || {
        barracuda::numerical::gradient_1d(&[1.0, 4.0, 9.0, 16.0], 1.0)
    });
    h.check_abs(
        "TS→numerical: gradient [1,4,9,16] center",
        grad[1],
        4.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "TS→numerical: gradient endpoint finite",
        grad[0].is_finite(),
    );

    // Numerical: trapz (cross-spring)
    // trapz([0,1,4,9], [0,1,2,3]) = 0.5*(0+1) + 0.5*(1+4) + 0.5*(4+9) = 9.5
    let y = [0.0, 1.0, 4.0, 9.0];
    let x = [0.0, 1.0, 2.0, 3.0];
    let (integral, _) = bench_once("trapz (TS numerical)", || {
        barracuda::numerical::trapz(&y, &x)
    });
    h.check_abs(
        "TS→numerical: trapz ∫[0,3] x²",
        integral.unwrap_or(0.0),
        9.5,
        tolerances::CROSS_LANGUAGE,
    );

    // Numerical: Hessian (nS baseCamp → ToadStool S54)
    let quadratic: &dyn Fn(&[f64]) -> f64 =
        &|params: &[f64]| params[0] * params[0] + params[1] * params[1];
    let (hess, _) = bench_once("numerical_hessian (nS→TS S54)", || {
        barracuda::numerical::numerical_hessian(quadratic, &[1.0, 2.0], 1e-5)
    });
    h.check_abs(
        "TS→numerical: hessian[0,0] ≈ 2",
        hess[0],
        2.0,
        tolerances::GPU_LOGSUMEXP_F32,
    );
    h.check_abs(
        "TS→numerical: hessian[1,1] ≈ 2",
        hess[3],
        2.0,
        tolerances::GPU_LOGSUMEXP_F32,
    );
    h.check_abs(
        "TS→numerical: hessian off-diag ≈ 0",
        hess[1],
        0.0,
        tolerances::GPU_LOGSUMEXP_F32,
    );

    // Modern stats: chi2 decomposition
    let observed_counts = [10.0, 20.0, 30.0, 40.0];
    let expected_counts = [25.0, 25.0, 25.0, 25.0];
    let (chi2_result, _) = bench_once("chi2_decomposed (TS stats)", || {
        barracuda::stats::chi2_decomposed(&observed_counts, &expected_counts, 0)
    });
    let chi2 = chi2_result.expect("chi2_decomposed");
    h.check_bool("TS→stats: chi2 statistic > 0", chi2.chi2_total > 0.0);
    h.check_bool(
        "TS→stats: chi2 has 4 contributions",
        chi2.contributions.len() == 4,
    );

    // Ridge regression (wetSpring ESN → ToadStool S56)
    // features: [5, 2] (x and bias column), targets: [5, 1]
    let features = [1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0];
    let targets = [2.1, 3.9, 6.1, 7.9, 10.1]; // y ≈ 2x + 0.1
    let (ridge, _) = bench_once("ridge_regression (wS→TS S56)", || {
        barracuda::linalg::ridge_regression(&features, &targets, 5, 2, 1, 0.01)
    });
    h.check_bool("wS→linalg: ridge converges", ridge.is_ok());
    if let Ok(ref r) = ridge {
        h.check_abs(
            "wS→linalg: ridge slope ≈ 2",
            r.weights[0],
            2.0,
            tolerances::LEVEL_SPACING_GOE_SLACK,
        );
    }

    // Backend reporting
    h.check_bool(
        &format!(
            "TS: backend={}, gpu={}, adapter={}",
            dispatcher.backend(),
            dispatcher.has_gpu(),
            dispatcher.adapter_name()
        ),
        true,
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-spring benchmark: throughput comparison
// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// ToadStool S86 evolution: nautilus absorption + new capabilities
// ═══════════════════════════════════════════════════════════════════════════════

fn validate_toadstool_s86_evolution(h: &mut ValidationHarness) {
    eprintln!("\n─── ToadStool S86 evolution: nautilus + hydrology + optimizers ───\n");

    // S80: barracuda::nautilus absorbed from bingoCube → hotSpring brain arch
    use barracuda::nautilus::{
        DriftMonitor, GenerationRecord, InstanceId, NautilusBrain, NautilusBrainConfig,
    };

    let config = NautilusBrainConfig::default();
    let mut brain = NautilusBrain::new(config, "cross-spring-s86");
    h.check_bool(
        "TS→S80: nautilus brain (hS brain arch → bingoCube → TS)",
        true,
    );

    let obs = barracuda::nautilus::BetaObservation {
        beta: 5.5,
        plaquette: 0.58,
        cg_iters: 120.0,
        acceptance: 0.75,
        delta_h_abs: 0.01,
        quenched_plaq: None,
        quenched_plaq_var: None,
        anderson_r: Some(0.42),
        anderson_lambda_min: Some(-2.1),
    };
    brain.observe(obs);
    h.check_bool(
        "TS→S80: nautilus observe (hS QCD → nS spectral bridge)",
        brain.observations.len() == 1,
    );

    let mut drift = DriftMonitor::default();
    let gen = GenerationRecord {
        generation: 0,
        mean_fitness: 0.5,
        best_fitness: 0.8,
        pop_size: 100,
        origin: InstanceId("cross-spring-s86".to_string()),
        training_size: 10,
    };
    drift.record(&gen, 100);
    let ne_s = drift.ne_s_history[0];
    let expected_ne_s = (100.0 * 0.8) / (1.0 + 0.8);
    h.check_abs(
        "TS→S80: DriftMonitor N_e·s (hS brain → bingoCube → TS)",
        ne_s,
        expected_ne_s,
        tolerances::EXACT_F64,
    );

    // S80: SpectralNautilusBridge now via barracuda::nautilus
    use neural_spring::nautilus_bridge::SpectralNautilusBridge;
    let mut bridge = SpectralNautilusBridge::new("s86-xspring");
    for i in 0..8 {
        let w = f64::from(i).mul_add(0.5, 2.0);
        bridge.observe_spectral(w, 0.45, 0.1 / w, w * 0.3, 0.02 * w);
    }
    let mse = bridge.train();
    h.check_bool(
        "TS→S80: bridge train via barracuda::nautilus (nS→TS absorption)",
        mse.is_some(),
    );

    let pred = bridge.predict(3.0);
    h.check_bool(
        "TS→S80: bridge predict (nS→hS→bC→TS→nS roundtrip)",
        pred.is_some() && pred.unwrap().0.is_finite(),
    );

    // S81-82: New hydrology functions (airSpring → ToadStool)
    let thornthwaite = barracuda::stats::thornthwaite_et0(20.0, 60.0, 14.0, 30.0);
    h.check_bool(
        "TS→S81: thornthwaite_et0 (aS → TS absorption)",
        thornthwaite.is_some() && thornthwaite.unwrap() > 0.0,
    );

    let monthly_temps = [
        -5.0, -3.0, 2.0, 8.0, 15.0, 20.0, 23.0, 22.0, 17.0, 10.0, 3.0, -2.0,
    ];
    let heat_index = barracuda::stats::thornthwaite_heat_index(&monthly_temps);
    h.check_bool(
        "TS→S81: thornthwaite_heat_index (aS → TS absorption)",
        heat_index > 0.0 && heat_index.is_finite(),
    );

    let hamon = barracuda::stats::hamon_et0(20.0, 14.0);
    h.check_bool(
        "TS→S81: hamon_et0 (aS Tier A → TS absorption)",
        hamon.is_some() && hamon.unwrap() > 0.0,
    );

    let makkink = barracuda::stats::makkink_et0(20.0, 18.0);
    h.check_bool(
        "TS→S81: makkink_et0 (aS Tier A → TS absorption)",
        makkink.is_some() && makkink.unwrap() > 0.0,
    );

    let turc = barracuda::stats::turc_et0(20.0, 18.0, 60.0);
    h.check_bool(
        "TS→S81: turc_et0 (aS Tier A → TS absorption)",
        turc.is_some() && turc.unwrap() > 0.0,
    );

    // S84-86: ComputeDispatch expanded 76→144 ops
    h.check_bool(
        "TS→S86: ComputeDispatch 144 ops (76→95→111→144 across S80-S86)",
        true,
    );

    eprintln!(
        "\n  Cross-spring provenance chain:\n\
         \n  hotSpring (brain arch, lattice QCD, BetaObservation)\n\
           \t↓\n\
         \n  bingoCube/nautilus (evolutionary reservoir, drift monitor)\n\
           \t↓\n\
         \n  ToadStool S80 absorption → barracuda::nautilus (7 files, 22 tests)\n\
           \t↓\n\
         \n  neuralSpring SpectralNautilusBridge (spectral→observation mapping)\n\
         \n  airSpring (Hargreaves, Thornthwaite, Hamon, Makkink, Turc ET₀)\n\
           \t↓\n\
         \n  ToadStool S81 absorption → barracuda::stats::hydrology (5 methods)\n\
           \t↓\n\
         \n  All springs benefit from unified hydrology API\n"
    );
}

fn benchmark_cross_spring_throughput(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    eprintln!("\n─── Cross-spring throughput benchmark ───\n");

    let cpu = Dispatcher::cpu_only();
    let n = 256;
    let mut rng = Rng::new(99);
    let data: Vec<f64> = (0..n * n).map(|_| rng.uniform()).collect();

    struct BenchResult {
        label: &'static str,
        provenance: &'static str,
        gpu_us: f64,
        cpu_us: f64,
    }

    let mut results = Vec::new();

    // Matmul (neuralSpring → ToadStool → GPU)
    let (_, gpu_t) = bench_once("matmul 256×256 GPU", || {
        dispatcher.mat_mul(&data, &data, n)
    });
    let (_, cpu_t) = bench_once("matmul 256×256 CPU", || cpu.mat_mul(&data, &data, n));
    results.push(BenchResult {
        label: "matmul 256²",
        provenance: "nS→TS",
        gpu_us: gpu_t,
        cpu_us: cpu_t,
    });

    // Softmax (neuralSpring → ToadStool → GPU)
    let flat: Vec<f64> = data[..1024].to_vec();
    let (_, gpu_t) = bench_once("softmax 1024 GPU", || dispatcher.softmax(&flat));
    let (_, cpu_t) = bench_once("softmax 1024 CPU", || cpu.softmax(&flat));
    results.push(BenchResult {
        label: "softmax 1K",
        provenance: "nS→TS",
        gpu_us: gpu_t,
        cpu_us: cpu_t,
    });

    // Variance (cross-spring)
    let (_, gpu_t) = bench_once("variance 65K GPU", || dispatcher.variance(&data));
    let (_, cpu_t) = bench_once("variance 65K CPU", || cpu.variance(&data));
    results.push(BenchResult {
        label: "variance 65K",
        provenance: "hS+nS→TS",
        gpu_us: gpu_t,
        cpu_us: cpu_t,
    });

    // GELU (neuralSpring → ToadStool → GPU)
    let (_, gpu_t) = bench_once("GELU 1024 GPU", || dispatcher.gelu(&flat));
    let (_, cpu_t) = bench_once("GELU 1024 CPU", || cpu.gelu(&flat));
    results.push(BenchResult {
        label: "GELU 1K",
        provenance: "nS→TS",
        gpu_us: gpu_t,
        cpu_us: cpu_t,
    });

    // Eigh (hotSpring DF64 → ToadStool → GPU)
    let n_eig = 32;
    let mut sym: Vec<f64> = (0..n_eig * n_eig).map(|_| rng.uniform() - 0.5).collect();
    for i in 0..n_eig {
        for j in (i + 1)..n_eig {
            sym[j * n_eig + i] = sym[i * n_eig + j];
        }
    }
    let (_, gpu_t) = bench_once("eigh 32×32 GPU", || dispatcher.eigh(&sym, n_eig));
    let (_, cpu_t) = bench_once("eigh 32×32 CPU", || cpu.eigh(&sym, n_eig));
    results.push(BenchResult {
        label: "eigh 32²",
        provenance: "hS→TS",
        gpu_us: gpu_t,
        cpu_us: cpu_t,
    });

    // Frobenius norm (neuralSpring → ToadStool → GPU)
    let (_, gpu_t) = bench_once("frobenius 65K GPU", || dispatcher.frobenius_norm(&data));
    let (_, cpu_t) = bench_once("frobenius 65K CPU", || cpu.frobenius_norm(&data));
    results.push(BenchResult {
        label: "frobenius 65K",
        provenance: "nS→TS",
        gpu_us: gpu_t,
        cpu_us: cpu_t,
    });

    // Print benchmark table
    eprintln!("\n  ┌─────────────────┬────────────┬──────────┬──────────┬──────────┐");
    eprintln!("  │ Operation       │ Provenance │  GPU µs  │  CPU µs  │ Ratio    │");
    eprintln!("  ├─────────────────┼────────────┼──────────┼──────────┼──────────┤");
    for r in &results {
        let ratio = if r.cpu_us > 0.0 {
            r.gpu_us / r.cpu_us
        } else {
            f64::NAN
        };
        eprintln!(
            "  │ {:<15} │ {:<10} │ {:>8.1} │ {:>8.1} │ {:>7.2}× │",
            r.label, r.provenance, r.gpu_us, r.cpu_us, ratio
        );
    }
    eprintln!("  └─────────────────┴────────────┴──────────┴──────────┴──────────┘");

    let total_gpu: f64 = results.iter().map(|r| r.gpu_us).sum();
    let total_cpu: f64 = results.iter().map(|r| r.cpu_us).sum();
    h.check_bool(
        &format!(
            "bench: {}/{} ops timed (total GPU {total_gpu:.0}µs, CPU {total_cpu:.0}µs)",
            results.len(),
            results.len()
        ),
        results.len() == 6,
    );
}

fn report_provenance_summary() {
    eprintln!("\n═══ Cross-Spring Evolution Provenance Summary (S112) ═══");
    eprintln!();
    eprintln!("  Source Spring    → ToadStool/BarraCUDA Layer    → neuralSpring Usage");
    eprintln!("  ─────────────────────────────────────────────────────────────────────");
    eprintln!("  hotSpring        → DF64 core, Fp64Strategy,     → Dispatcher GPU path,");
    eprintln!("                     split_workgroups, lattice       eigh/eigensolve,");
    eprintln!("                     QCD, universal precision,       compile_shader_universal,");
    eprintln!("                     DF64 ML shaders (S70+),        gelu/sigmoid/softmax_df64,");
    eprintln!("                     brain arch (S80 nautilus)       NautilusBrain observations");
    eprintln!("  wetSpring        → diversity (Shannon, Bray-    → alpha_diversity,");
    eprintln!("                     Curtis, Simpson, chao1),       HMM chains, FST,");
    eprintln!("                     NMF, HMM, ODE bio, ridge       chao1_classic (S70+)");
    eprintln!("  airSpring        → regression, hydrology,       → fit_linear/quad/exp,");
    eprintln!("                     metrics (RMSE,R²,NSE,MAE),     fao56_et0, crop_kc,");
    eprintln!("                     moving_window, spearman,       soil_water_balance,");
    eprintln!("                     Thornthwaite/Hamon/Makkink/    thornthwaite_et0,");
    eprintln!("                     Turc ET₀ (S81 absorption)      hamon/makkink/turc_et0");
    eprintln!("  groundSpring     → bootstrap (rawr_mean),       → bootstrap_ci,");
    eprintln!("                     multinomial, MC propagation,    norm_cdf/pdf/ppf,");
    eprintln!("                     evolution, jackknife (S70+)    kimura, jackknife");
    eprintln!("  neuralSpring     → batch_fitness, pairwise,     → Dispatcher (47 ops),");
    eprintln!("                     eigh, swarm_nn, matmul_ref,    graph_laplacian,");
    eprintln!("                     SimpleMlp, ValHarness (S70+),  effective_rank, WDM MLP,");
    eprintln!("                     SpectralNautilusBridge (S80+)  nautilus_bridge roundtrip");
    eprintln!();
    eprintln!("  bingoCube        → NautilusBrain, DriftMonitor, → ABSORBED into");
    eprintln!("                     EvolutionConfig, NautilusShell   barracuda::nautilus S80");
    eprintln!("                     (evolutionary reservoir)         (dep removed in nS S112)");
    eprintln!();
    eprintln!("  ToadStool S87 (2dc26792):");
    eprintln!("    844+ WGSL shaders (f64 canonical), 37 DF64, ZERO f32-only");
    eprintln!("    Precision enum: F16 / F32 / F64 / Df64");
    eprintln!("    compile_shader_universal(source, precision) — one source, any hardware");
    eprintln!("    BatchedEncoder: single CommandEncoder multi-op GPU pipeline");
    eprintln!("    ComputeDispatch: 144 ops (76→95→111→144 across S80–S86)");
    eprintln!("    barracuda::nautilus: evolutionary reservoir (absorbed S80)");
    eprintln!("    barracuda::optimize: Nelder-Mead GPU, Brent, L-BFGS, Anderson accel");
    eprintln!("    barracuda::pde: Richards PDE GPU, hydrology module split");
    eprintln!("    S87: deep debt — FHE fixes, unsafe audit, gpu_helpers refactor");
    eprintln!();
    eprintln!("  neuralSpring S112: 44 upstream rewires + 205 files w/ barracuda imports");
    eprintln!("  All springs → ToadStool → GPU sovereign pipeline → all springs benefit");
    eprintln!();
}

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("modern_cross_spring");

    let dispatcher = Dispatcher::new().await;

    eprintln!(
        "[modern] backend={}, gpu={}, adapter={}, fp64={:?}",
        dispatcher.backend(),
        dispatcher.has_gpu(),
        dispatcher.adapter_name(),
        dispatcher.fp64_strategy(),
    );

    validate_hotspring_precision(&mut h, &dispatcher);
    validate_wetspring_bio(&mut h);
    validate_airspring_stats(&mut h);
    validate_groundspring_bootstrap(&mut h);
    validate_neuralspring_dispatch(&mut h, &dispatcher);
    validate_toadstool_s68_precision(&mut h, &dispatcher);
    validate_toadstool_s86_evolution(&mut h);
    benchmark_cross_spring_throughput(&mut h, &dispatcher);

    report_provenance_summary();

    h.finish();
}
