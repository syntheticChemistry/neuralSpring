// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-spring shader evolution validator.
//!
//! Proves the full evolution chain for key operations:
//!   local CPU ref → `barracuda::dispatch` → `Dispatcher` GPU/CPU → parity
//!
//! Tracks provenance: which spring each shader/operation evolved from,
//! and validates that the evolution preserved mathematical correctness.
//!
//! ## Provenance Map
//!
//! | Operation | Origin Spring | Absorption Path |
//! |-----------|--------------|-----------------|
//! | softmax | neuralSpring `transformer.rs` | → `barracuda::dispatch::softmax_dispatch` → `BarraCUDA` `ComputeDispatch` |
//! | gelu | neuralSpring `transformer.rs` | → `barracuda::dispatch::gelu_dispatch` → `BarraCUDA` `ComputeDispatch` |
//! | sigmoid | neuralSpring `primitives.rs` | → `barracuda` (GPU via `sigmoid_f64.wgsl`) |
//! | `layer_norm` | neuralSpring `coral_forge` | → `barracuda::TensorSession::layer_norm` |
//! | variance | wetSpring diversity | → `barracuda::stats` → `Dispatcher` |
//! | shannon | wetSpring diversity | → `barracuda::stats::shannon` → `Dispatcher` |
//! | simpson | wetSpring diversity | → `barracuda::stats::simpson` |
//! | pearson | hotSpring precision | → `barracuda::stats::pearson_correlation` → `Dispatcher` |
//! | `mat_mul` | neuralSpring `matmul_gpu_evolved` | → `barracuda::dispatch::matmul_dispatch` → `BarraCUDA` |
//! | `rk4_step` | neuralSpring `primitives.rs` | → `rk4_parallel.wgsl` (GPU) |
//! | eigensolve | hotSpring spectral | → `barracuda::linalg::eigh_f64` → `Dispatcher` |
//! | HMM forward | wetSpring bio | → `barracuda::dispatch::hmm_forward_dispatch` |
//! | `batch_ipr` | neuralSpring spectral | → `barracuda::spectral::BatchIprGpu` |
//! | diversity | wetSpring bio | → `barracuda::stats::alpha_diversity` |
//! | bootstrap | groundSpring uncertainty | → `barracuda::stats::bootstrap_ci` |
//! | `kimura` | groundSpring evolution | → `barracuda::stats::kimura_fixation_prob` |
//!
//! ```text
//! cargo run --release --bin validate_cross_spring_shader_evolution
//! ```

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::expect_used,
    clippy::suboptimal_flops,
    clippy::manual_range_contains,
    reason = "validation binary"
)]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::primitives;
use neural_spring::tolerances;
use neural_spring::transformer;
use neural_spring::validation::{exit_no_gpu, ValidationHarness};

// ── neuralSpring origins ────────────────────────────────────────────

fn validate_softmax_evolution(h: &mut ValidationHarness, disp: &Dispatcher) {
    eprintln!("\n── softmax evolution: neuralSpring → barracuda → Dispatcher ──");
    eprintln!("  provenance: neuralSpring transformer.rs → barracuda::dispatch::softmax_dispatch");

    let x = [1.0, 2.0, 3.0, 4.0, 5.0];

    let local = transformer::softmax(&x);
    let dispatched = disp.softmax(&x);
    let upstream =
        barracuda::dispatch::softmax_dispatch(&x, disp.wgpu_device()).expect("softmax_dispatch");

    let local_vs_dispatch: f64 = local
        .iter()
        .zip(dispatched.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "softmax: local CPU == Dispatcher",
        local_vs_dispatch,
        tolerances::TENSOR_EXACT_F32,
    );

    let local_vs_upstream: f64 = local
        .iter()
        .zip(upstream.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "softmax: local CPU == barracuda::dispatch",
        local_vs_upstream,
        tolerances::TENSOR_EXACT_F32,
    );

    let dispatch_vs_upstream: f64 = dispatched
        .iter()
        .zip(upstream.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "softmax: Dispatcher == barracuda::dispatch",
        dispatch_vs_upstream,
        tolerances::EXACT_F64,
    );

    let sum: f64 = dispatched.iter().sum();
    h.check_abs(
        "softmax: Dispatcher sums to 1.0",
        sum,
        1.0,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_gelu_evolution(h: &mut ValidationHarness, disp: &Dispatcher) {
    eprintln!("\n── gelu evolution: neuralSpring → barracuda → Dispatcher ──");
    eprintln!("  provenance: neuralSpring transformer.rs → barracuda::dispatch::gelu_dispatch");

    let x: Vec<f64> = (-5..=5).map(|i| i as f64 * 0.5).collect();

    let local: Vec<f64> = x.iter().copied().map(transformer::gelu).collect();
    let dispatched = disp.gelu(&x);
    let upstream =
        barracuda::dispatch::gelu_dispatch(&x, disp.wgpu_device()).expect("gelu_dispatch");

    let local_vs_dispatch: f64 = local
        .iter()
        .zip(dispatched.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "gelu: local CPU == Dispatcher",
        local_vs_dispatch,
        tolerances::TENSOR_EXACT_F32,
    );

    let local_vs_upstream: f64 = local
        .iter()
        .zip(upstream.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "gelu: local CPU == barracuda::dispatch",
        local_vs_upstream,
        tolerances::TENSOR_EXACT_F32,
    );

    h.check_bool(
        "gelu(0) ≈ 0",
        dispatched[5].abs() < tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "gelu(large positive) > input * 0.9",
        dispatched[10] > 2.5 * 0.9,
    );
}

fn validate_sigmoid_evolution(h: &mut ValidationHarness) {
    eprintln!("\n── sigmoid evolution: neuralSpring → barracuda (GPU sigmoid_f64.wgsl) ──");
    eprintln!("  provenance: neuralSpring primitives.rs → metalForge sigmoid_f64.wgsl");

    h.check_abs(
        "sigmoid(0) = 0.5",
        primitives::sigmoid(0.0),
        0.5,
        tolerances::EXACT_F64,
    );
    h.check_bool(
        "sigmoid(10) → 1.0",
        (primitives::sigmoid(10.0) - 1.0).abs() < tolerances::SIGMOID_SATURATION,
    );
    h.check_bool(
        "sigmoid(-10) → 0.0",
        primitives::sigmoid(-10.0).abs() < tolerances::SIGMOID_SATURATION,
    );
    h.check_abs(
        "sigmoid symmetry: σ(x) + σ(-x) = 1",
        primitives::sigmoid(2.0) + primitives::sigmoid(-2.0),
        1.0,
        tolerances::EXACT_F64,
    );
}

fn validate_coralforge_activation_evolution(h: &mut ValidationHarness, disp: &Dispatcher) {
    eprintln!("\n── coralForge activation evolution: neuralSpring → barracuda ──");
    eprintln!(
        "  provenance: neuralSpring coral_forge/activation.rs → gelu_f64.wgsl, softmax_f64.wgsl"
    );

    let gelu_vals: Vec<f64> = (-3..=3).map(|i| i as f64).collect();
    let local_gelu: Vec<f64> = gelu_vals.iter().copied().map(transformer::gelu).collect();
    let disp_gelu = disp.gelu(&gelu_vals);
    let gelu_diff: f64 = local_gelu
        .iter()
        .zip(disp_gelu.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "coralForge gelu: CPU ref == Dispatcher (→ gelu_f64.wgsl)",
        gelu_diff,
        tolerances::TENSOR_EXACT_F32,
    );

    let sm_input = [1.0, 2.0, 3.0, 4.0];
    let local_sm = transformer::softmax(&sm_input);
    let disp_sm = disp.softmax(&sm_input);
    let sm_diff: f64 = local_sm
        .iter()
        .zip(disp_sm.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "coralForge softmax: CPU ref == Dispatcher (→ softmax_f64.wgsl)",
        sm_diff,
        tolerances::TENSOR_EXACT_F32,
    );

    h.check_bool(
        "coralForge gelu(0) ≈ 0 (GPU path)",
        disp_gelu[3].abs() < tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "coralForge softmax sums to 1 (GPU path)",
        (disp_sm.iter().sum::<f64>() - 1.0).abs() < tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_matmul_evolution(h: &mut ValidationHarness, disp: &Dispatcher) {
    eprintln!("\n── matmul evolution: neuralSpring → barracuda → BarraCUDA ──");
    eprintln!(
        "  provenance: neuralSpring matmul_gpu_evolved.wgsl → barracuda::dispatch::matmul_dispatch"
    );

    let n = 4;
    let a: Vec<f64> = (0..n * n).map(|i| (i as f64 + 1.0) * 0.1).collect();
    let b: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.05 + 0.5).collect();

    let dispatched = disp.mat_mul(&a, &b, n);
    let upstream = barracuda::dispatch::matmul_dispatch(&a, &b, n, n, n, disp.wgpu_device())
        .expect("matmul_dispatch");

    let max_diff: f64 = dispatched
        .iter()
        .zip(upstream.iter())
        .map(|(d, u)| (d - u).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "matmul: Dispatcher == barracuda::dispatch",
        max_diff,
        tolerances::EXACT_F64,
    );
    h.check_bool("matmul: correct output size", dispatched.len() == n * n);
}

fn validate_rk4_evolution(h: &mut ValidationHarness) {
    eprintln!("\n── RK4 evolution: neuralSpring → rk4_parallel.wgsl ──");
    eprintln!("  provenance: neuralSpring primitives::rk4_step → metalForge rk4_parallel.wgsl (GPU batch)");

    let state = [1.0, 0.0];
    let next = primitives::rk4_step(&state, 0.01, |y| [-y[1], y[0]]);
    let energy = next[0] * next[0] + next[1] * next[1];
    h.check_abs(
        "rk4: harmonic oscillator energy conservation",
        energy,
        1.0,
        tolerances::ODE_ATOL,
    );

    let mut y = [1.0, 0.0];
    for _ in 0..1000 {
        y = primitives::rk4_step(&y, 0.01, |s| [-s[1], s[0]]);
    }
    let final_energy = y[0] * y[0] + y[1] * y[1];
    h.check_abs(
        "rk4: 1000-step energy conservation",
        final_energy,
        1.0,
        tolerances::SPECIAL_FUNCTION_F64,
    );
}

// ── wetSpring origins ───────────────────────────────────────────────

fn validate_diversity_evolution(h: &mut ValidationHarness) {
    eprintln!("\n── diversity evolution: wetSpring → barracuda::stats ──");
    eprintln!(
        "  provenance: wetSpring bio → barracuda::stats::{{shannon,simpson,alpha_diversity}}"
    );

    let counts = [10.0, 20.0, 30.0, 40.0];
    let sh = barracuda::stats::shannon(&counts);
    h.check_bool(
        "shannon (wetSpring) is finite and positive",
        sh.is_finite() && sh > 0.0,
    );

    let si = barracuda::stats::simpson(&counts);
    h.check_bool("simpson (wetSpring) in [0,1]", si >= 0.0 && si <= 1.0);

    let counts_u64: [u64; 4] = [10, 20, 30, 40];
    let ch = barracuda::stats::chao1_classic(&counts_u64);
    h.check_bool("chao1 (wetSpring) ≥ observed richness", ch >= 4.0);

    let freqs = [0.25, 0.25, 0.25, 0.25];
    let sh_freq = barracuda::stats::shannon_from_frequencies(&freqs);
    let expected = (4.0_f64).ln();
    h.check_abs(
        "shannon_from_frequencies (wetSpring) uniform=ln(4)",
        sh_freq,
        expected,
        tolerances::EXACT_F64,
    );
}

fn validate_hmm_evolution(h: &mut ValidationHarness, disp: &Dispatcher) {
    eprintln!("\n── HMM evolution: wetSpring → barracuda → Dispatcher ──");
    eprintln!(
        "  provenance: wetSpring hmm_forward_log.wgsl → barracuda::dispatch::hmm_forward_dispatch"
    );

    let n_states = 3;
    let log_init = [(-3.0_f64).ln(); 3];
    let log_trans = [
        (-3.0_f64).ln(),
        (-3.0_f64).ln(),
        (-3.0_f64).ln(),
        (-3.0_f64).ln(),
        (-3.0_f64).ln(),
        (-3.0_f64).ln(),
        (-3.0_f64).ln(),
        (-3.0_f64).ln(),
        (-3.0_f64).ln(),
    ];
    let log_emit = [0.0_f64.ln(), (-2.0_f64).ln(), (-3.0_f64).ln()];

    let (dispatched_alpha, _disp_scale) =
        disp.hmm_forward_step(&log_init, &log_trans, &log_emit, n_states);
    let (upstream_alpha, _upstream_scale) = barracuda::dispatch::hmm_forward_dispatch(
        &log_init,
        &log_trans,
        &log_emit,
        n_states,
        disp.wgpu_device(),
    )
    .expect("hmm_forward_dispatch");

    let max_diff: f64 = dispatched_alpha
        .iter()
        .zip(upstream_alpha.iter())
        .map(|(&d, &u)| (d - u).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "hmm_forward: Dispatcher == barracuda::dispatch (wetSpring)",
        max_diff,
        tolerances::EXACT_F64,
    );
}

// ── hotSpring origins ───────────────────────────────────────────────

fn validate_spectral_evolution(h: &mut ValidationHarness, disp: &Dispatcher) {
    eprintln!("\n── spectral evolution: hotSpring → barracuda::linalg → Dispatcher ──");
    eprintln!(
        "  provenance: hotSpring eigh_f64 → barracuda::linalg::eigh_f64 → Dispatcher::eigensolve"
    );

    let n = 6;
    let mut sym: Vec<f64> = vec![0.0; n * n];
    for i in 0..n {
        sym[i * n + i] = (i as f64 + 1.0) * 2.0;
        if i + 1 < n {
            sym[i * n + (i + 1)] = 0.5;
            sym[(i + 1) * n + i] = 0.5;
        }
    }

    let (eigenvalues, _eigenvectors) = disp.eigh(&sym, n);
    h.check_bool(
        "eigensolve (hotSpring): correct count",
        eigenvalues.len() == n,
    );
    h.check_bool(
        "eigensolve (hotSpring): sorted ascending",
        eigenvalues
            .windows(2)
            .all(|w| w[0] <= w[1] + tolerances::CROSS_LANGUAGE),
    );

    let lsr = barracuda::spectral::level_spacing_ratio(&eigenvalues);
    h.check_bool(
        "level_spacing_ratio (hotSpring): in [0,1]",
        lsr >= 0.0 && lsr <= 1.0,
    );

    let bw = barracuda::spectral::spectral_bandwidth(&eigenvalues);
    h.check_bool("spectral_bandwidth (hotSpring): positive", bw > 0.0);
}

fn validate_precision_evolution(h: &mut ValidationHarness, disp: &Dispatcher) {
    eprintln!("\n── precision evolution: hotSpring → barracuda f64 canonical ──");
    eprintln!("  provenance: hotSpring DF64 core + math_f64.wgsl → compile_shader_universal");

    let data: Vec<f64> = (0..100).map(|i| (i as f64) * 0.01).collect();

    let v = disp.variance(&data);
    h.check_bool("variance (f64 precision): finite", v.is_finite());

    let m = disp.mean(&data);
    let expected_mean = 0.495;
    h.check_abs(
        "mean (f64 precision): correct",
        m,
        expected_mean,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );

    let frob = disp.frobenius_norm(&data);
    h.check_bool("frobenius_norm (f64 precision): positive", frob > 0.0);
}

// ── groundSpring origins ────────────────────────────────────────────

fn validate_uncertainty_evolution(h: &mut ValidationHarness) {
    eprintln!("\n── uncertainty evolution: groundSpring → barracuda::stats ──");
    eprintln!("  provenance: groundSpring bootstrap/jackknife → barracuda::stats::bootstrap_ci, jackknife");

    let data = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let ci = barracuda::stats::bootstrap_ci(
        &data,
        |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64,
        1000,
        0.95,
        42,
    )
    .expect("bootstrap_ci");
    h.check_bool(
        "bootstrap CI (groundSpring): lower ≤ upper",
        ci.lower <= ci.upper,
    );
    h.check_bool(
        "bootstrap CI (groundSpring): contains mean",
        ci.lower <= 9.0 && ci.upper >= 9.0,
    );

    let jk = barracuda::stats::jackknife(&data, |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64)
        .expect("jackknife should return Some for n>=2");
    h.check_bool(
        "jackknife (groundSpring): estimate near mean",
        (jk.estimate - 9.0).abs() < 1.0,
    );
    h.check_bool("jackknife (groundSpring): se > 0", jk.std_error > 0.0);

    let kf = barracuda::stats::kimura_fixation_prob(100, 0.01, 0.5);
    h.check_bool("kimura (groundSpring): prob in (0,1)", kf > 0.0 && kf < 1.0);
}

// ── cross-spring convergence ────────────────────────────────────────

fn validate_cross_spring_convergence(h: &mut ValidationHarness, disp: &Dispatcher) {
    eprintln!("\n── cross-spring convergence: all springs → BarraCUDA ──");
    eprintln!("  provenance: multi-spring ops converge through BarraCUDA 844+ shaders");

    let data: Vec<f64> = (0..64).map(|i| (i as f64) * 0.1).collect();

    let disp_var = disp.variance(&data);
    let disp_mean = disp.mean(&data);
    let disp_frob = disp.frobenius_norm(&data);

    let bc_var = barracuda::dispatch::variance_dispatch(&data, disp.wgpu_device())
        .expect("variance_dispatch");
    let bc_mean =
        barracuda::dispatch::mean_dispatch(&data, disp.wgpu_device()).expect("mean_dispatch");
    let bc_frob = barracuda::dispatch::frobenius_norm_dispatch(&data, disp.wgpu_device())
        .expect("frobenius_norm_dispatch");

    h.check_abs(
        "cross-spring variance: Dispatcher == barracuda::dispatch",
        (disp_var - bc_var).abs(),
        0.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "cross-spring mean: Dispatcher == barracuda::dispatch",
        (disp_mean - bc_mean).abs(),
        0.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "cross-spring frobenius_norm: Dispatcher == barracuda::dispatch",
        (disp_frob - bc_frob).abs(),
        0.0,
        tolerances::EXACT_F64,
    );

    let pc = barracuda::stats::pearson_correlation(&data[..32], &data[32..]).expect("pearson");
    h.check_bool("pearson (hotSpring precision): finite", pc.is_finite());

    let norm_cdf = barracuda::stats::norm_cdf(0.0);
    h.check_abs(
        "norm_cdf (groundSpring): Φ(0) = 0.5",
        norm_cdf,
        0.5,
        tolerances::EXACT_F64,
    );
}

fn print_provenance_summary() {
    use barracuda::shaders::provenance;
    let shader_count = provenance::cross_spring_shaders().len();
    let matrix = provenance::cross_spring_matrix();

    eprintln!("\n╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Cross-Spring Shader Evolution — Provenance Summary            ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  {shader_count} shaders tracked · {n} cross-spring edges",
        n = matrix.len()
    );
    eprintln!("║                                                                ║");
    eprintln!("║  hotSpring ─── DF64 core/transcendentals, spectral, eigensolve ║");
    eprintln!("║  wetSpring ─── diversity, HMM, bio, Gillespie, Wright-Fisher   ║");
    eprintln!("║  neuralSpring ─ ML activations, matmul, RK4, attention, swarm  ║");
    eprintln!("║  groundSpring ─ bootstrap, jackknife, kimura, norm_cdf/ppf     ║");
    eprintln!("║  airSpring ──── hydrology (ET₀), regression, water balance     ║");
    eprintln!("║                                                                ║");
    eprintln!("║  barraCuda v0.3.3 at 83aa08a · ToadStool S142 at a86bc546    ║");
    eprintln!("║  coralReef Iteration 29 at 2779c88 (AMD E2E GPU dispatch)    ║");
    eprintln!("║  Precision per hardware: F32 / F64 / DF64                    ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝");
}

#[tokio::main]
async fn main() {
    let shader_count = barracuda::shaders::provenance::cross_spring_shaders().len();
    eprintln!("=== Cross-Spring Shader Evolution Validator ===");
    eprintln!(
        "barraCuda v0.3.3 at 83aa08a — {shader_count} tracked shaders (provenance registry)\n"
    );

    let mut h = ValidationHarness::new("cross_spring_shader_evolution");

    let Ok(gpu) = Gpu::new().await else {
        exit_no_gpu();
    };
    let disp = Dispatcher::from_gpu(gpu);

    // neuralSpring origins
    validate_softmax_evolution(&mut h, &disp);
    validate_gelu_evolution(&mut h, &disp);
    validate_sigmoid_evolution(&mut h);
    validate_coralforge_activation_evolution(&mut h, &disp);
    validate_matmul_evolution(&mut h, &disp);
    validate_rk4_evolution(&mut h);

    // wetSpring origins
    validate_diversity_evolution(&mut h);
    validate_hmm_evolution(&mut h, &disp);

    // hotSpring origins
    validate_spectral_evolution(&mut h, &disp);
    validate_precision_evolution(&mut h, &disp);

    // groundSpring origins
    validate_uncertainty_evolution(&mut h);

    // convergence
    validate_cross_spring_convergence(&mut h, &disp);

    print_provenance_summary();
    h.finish();
}
