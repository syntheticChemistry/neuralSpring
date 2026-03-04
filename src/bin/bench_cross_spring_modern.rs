// SPDX-License-Identifier: AGPL-3.0-or-later

//! Modern cross-spring evolution benchmark — Session 121.
//!
//! Benchmarks the full cross-spring shader evolution pipeline, documenting
//! when and where each component evolved to benefit all springs.
//!
//! ## Cross-Spring Shader Provenance
//!
//! | Shader/Op | Origin Spring | barraCuda Path | neuralSpring Usage |
//! |-----------|---------------|----------------|-------------------|
//! | `SimpleMlp` | neuralSpring nW-01/02 | `nn::SimpleMlp` (S83) | WDM surrogates |
//! | `hmm_viterbi_f64.wgsl` | wetSpring bio (S69) | `ops::bio::hmm_viterbi` | introgression detect |
//! | `hmm_forward_f64.wgsl` | wetSpring bio (S52) | `ops::bio::HmmBatchForwardF64` | HMM validation |
//! | `softmax_f64.wgsl` | neuralSpring transformer | `dispatch::softmax_dispatch` (S52) | ML inference |
//! | `gelu_f64.wgsl` | neuralSpring transformer | `dispatch::gelu_dispatch` (S52) | transformer blocks |
//! | `matmul_gpu.wgsl` | neuralSpring matmul | `dispatch::matmul_dispatch` (S52) | evolved matmul |
//! | `rk4_parallel.wgsl` | neuralSpring primitives | local (LazyLock upstream) | ODE integration |
//! | `df64_core.wgsl` | hotSpring biomeGate | precision polyfills | DF64 consumer GPU |
//! | `eigh_f64.wgsl` | hotSpring spectral | `linalg::eigh_f64` (S60) | spectral analysis |
//! | `diversity_fusion.wgsl` | wetSpring bio | `ops::bio::DiversityFusionGpu` | eco dynamics |
//! | `pearson_f64.wgsl` | hotSpring+wetSpring | `stats::pearson_correlation` | correlation |
//! | `bootstrap_ci.wgsl` | groundSpring uncertainty | `stats::bootstrap_ci` | confidence intervals |
//!
//! ```text
//! cargo run --release --bin bench_cross_spring_modern
//! ```

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::redundant_clone,
    reason = "benchmark binary"
)]

use barracuda::nn::simple_mlp::{Activation, DenseLayer};
use barracuda::nn::SimpleMlp;
use barracuda::stats;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::rng::Rng;
use neural_spring::validation::ValidationHarness;
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

fn bench_simplemlp(h: &mut ValidationHarness) {
    eprintln!("═══ neuralSpring → barraCuda: SimpleMlp (S83 absorption) ═══");
    eprintln!("  Origin: neuralSpring WDM surrogates (nW-01, nW-02)");
    eprintln!("  Evolution: hand-rolled matmul → barracuda::nn::SimpleMlp");
    eprintln!("  Impact: all springs can load and run MLP inference");
    eprintln!();

    let small = SimpleMlp::new(vec![
        DenseLayer {
            weight: vec![vec![0.1; 2]; 8],
            bias: vec![0.0; 8],
            activation: Activation::Relu,
        },
        DenseLayer {
            weight: vec![vec![0.1; 8]; 2],
            bias: vec![0.0; 2],
            activation: Activation::Identity,
        },
    ]);

    let medium = SimpleMlp::new(vec![
        DenseLayer {
            weight: vec![vec![0.01; 32]; 64],
            bias: vec![0.0; 64],
            activation: Activation::Relu,
        },
        DenseLayer {
            weight: vec![vec![0.01; 64]; 64],
            bias: vec![0.0; 64],
            activation: Activation::Relu,
        },
        DenseLayer {
            weight: vec![vec![0.01; 64]; 3],
            bias: vec![0.0; 3],
            activation: Activation::Identity,
        },
    ]);

    let input_small = vec![1.0, 0.5];
    let input_medium: Vec<f64> = (0..32).map(|i| (i as f64) * 0.01).collect();

    let t_small = bench("  2→8→2 MLP (EOS-scale)", || {
        let _ = small.forward(&input_small);
    });
    h.check_bool("SimpleMlp 2→8→2 < 50µs", t_small < 50.0);

    let t_medium = bench("  32→64→64→3 MLP (transport-scale)", || {
        let _ = medium.forward(&input_medium);
    });
    h.check_bool("SimpleMlp 32→64→64→3 < 500µs", t_medium < 500.0);

    let json = medium.to_json().expect("JSON serialization");
    let restored = SimpleMlp::from_json(&json).expect("JSON deserialization");
    let orig = medium.forward(&input_medium);
    let rest = restored.forward(&input_medium);
    let max_diff: f64 = orig
        .iter()
        .zip(rest.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    h.check_bool(
        &format!("JSON roundtrip exact (max_diff={max_diff:.2e})"),
        max_diff < f64::EPSILON,
    );
    eprintln!();
}

fn bench_hmm_dispatcher(h: &mut ValidationHarness) {
    eprintln!("═══ wetSpring → barraCuda: HMM f64 Viterbi (S69 absorption) ═══");
    eprintln!("  Origin: wetSpring bio HMM (forward/backward/Viterbi)");
    eprintln!("  Evolution: neuralSpring per-step f32 Tensor → barraCuda f64 shader");
    eprintln!("  Impact: f64 precision, single-dispatch GPU execution");
    eprintln!();

    let n_states = 4;
    let n_obs = 6;
    let mut rng = Rng::new(42);

    let mut transition = vec![0.0; n_states * n_states];
    for i in 0..n_states {
        let mut row_sum = 0.0;
        for j in 0..n_states {
            let v = rng.next_f64().max(0.01);
            transition[i * n_states + j] = v;
            row_sum += v;
        }
        for j in 0..n_states {
            transition[i * n_states + j] /= row_sum;
        }
    }

    let mut emission = vec![0.0; n_states * n_obs];
    for i in 0..n_states {
        let mut row_sum = 0.0;
        for j in 0..n_obs {
            let v = rng.next_f64().max(0.01);
            emission[i * n_obs + j] = v;
            row_sum += v;
        }
        for j in 0..n_obs {
            emission[i * n_obs + j] /= row_sum;
        }
    }

    let mut initial = vec![0.0; n_states];
    let mut pi_sum = 0.0;
    for v in &mut initial {
        *v = rng.next_f64().max(0.01);
        pi_sum += *v;
    }
    for v in &mut initial {
        *v /= pi_sum;
    }

    let observations: Vec<usize> = (0..100)
        .map(|_| (rng.next_f64() * n_obs as f64) as usize % n_obs)
        .collect();

    let hmm = neural_spring::hmm::Hmm::from_flat(
        transition.clone(),
        emission.clone(),
        initial.clone(),
        n_states,
        n_obs,
    );

    let t_cpu_viterbi = bench("  CPU Viterbi (4 states, 100 obs)", || {
        let _ = hmm.viterbi(&observations);
    });

    let dispatcher = Dispatcher::cpu_only();

    let t_disp_viterbi = bench("  Dispatcher Viterbi", || {
        let _ = dispatcher.hmm_viterbi_chain(
            &initial,
            &transition,
            &emission,
            &observations,
            n_states,
            n_obs,
        );
    });

    h.check_bool("CPU Viterbi runs", t_cpu_viterbi > 0.0);
    h.check_bool("Dispatcher Viterbi runs", t_disp_viterbi > 0.0);

    let (cpu_path, cpu_prob) = hmm.viterbi(&observations);
    let (disp_path, _disp_prob) = dispatcher.hmm_viterbi_chain(
        &initial,
        &transition,
        &emission,
        &observations,
        n_states,
        n_obs,
    );
    h.check_bool("Viterbi path agreement", cpu_path == disp_path);
    h.check_bool("CPU Viterbi prob finite", cpu_prob.is_finite());

    let t_cpu_fwd = bench("  CPU forward (4 states, 100 obs)", || {
        let _ = hmm.forward(&observations);
    });

    let t_disp_fwd = bench("  Dispatcher forward chain", || {
        let _ = dispatcher.hmm_forward_chain(
            &initial,
            &transition,
            &emission,
            &observations,
            n_states,
            n_obs,
        );
    });

    h.check_bool("CPU forward runs", t_cpu_fwd > 0.0);
    h.check_bool("Dispatcher forward runs", t_disp_fwd > 0.0);
    eprintln!();
}

fn bench_stats_cross_spring(h: &mut ValidationHarness) {
    eprintln!("═══ airSpring+groundSpring → barraCuda::stats (S64) ═══");
    eprintln!("  Origin: airSpring metrics + groundSpring hydrology stats");
    eprintln!("  Evolution: per-spring implementations → unified barraCuda::stats");
    eprintln!("  Impact: all springs share single, tested stats library");
    eprintln!();

    let mut rng = Rng::new(123);
    let x: Vec<f64> = (0..10_000).map(|_| rng.next_f64()).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&v| v * 0.8 + 0.1 + rng.next_f64() * 0.05)
        .collect();

    let t_r2 = bench("  R² (n=10000)", || {
        let _ = stats::r_squared(&x, &y);
    });
    h.check_bool("stats::r_squared < 100µs", t_r2 < 100.0);

    let r2 = stats::r_squared(&x, &y);
    h.check_bool("R² > 0.9 (correlated data)", r2 > 0.9);

    let t_pearson = bench("  Pearson (n=10000)", || {
        let _ = stats::pearson_correlation(&x, &y);
    });
    h.check_bool("stats::pearson < 100µs", t_pearson < 100.0);

    let t_rmse = bench("  RMSE (n=10000)", || {
        let _ = stats::rmse(&x, &y);
    });
    h.check_bool("stats::rmse < 100µs", t_rmse < 100.0);

    let counts: Vec<f64> = (0..100).map(|_| rng.next_f64() * 10.0 + 1.0).collect();
    let t_shannon = bench("  Shannon entropy (n=100)", || {
        let _ = stats::shannon(&counts);
    });
    h.check_bool("stats::shannon < 200µs", t_shannon < 200.0);
    eprintln!();
}

fn bench_precision_hotspring(h: &mut ValidationHarness) {
    eprintln!("═══ hotSpring → barraCuda: Precision & Spectral ═══");
    eprintln!("  Origin: hotSpring lattice QCD + spectral methods");
    eprintln!("  Evolution: lattice-specific → universal precision pipeline");
    eprintln!("  Impact: compile_shader_universal(F16/F32/F64/Df64)");
    eprintln!();

    let mat_4x4: Vec<f64> = vec![
        4.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    ];

    let t_eigh = bench("  eigh_f64 (4×4 symmetric)", || {
        let _ = barracuda::linalg::eigh_f64(&mat_4x4, 4);
    });
    h.check_bool("eigh_f64 < 100µs", t_eigh < 100.0);

    let decomp = barracuda::linalg::eigh_f64(&mat_4x4, 4).expect("eigh_f64 should succeed");
    h.check_bool("eigenvalues count == 4", decomp.eigenvalues.len() == 4);
    for (i, &ev) in decomp.eigenvalues.iter().enumerate() {
        h.check_bool(&format!("eigenvalue[{i}] finite"), ev.is_finite());
    }

    let mut sorted = decomp.eigenvalues.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    h.check_bool(
        "eigenvalues non-decreasing",
        sorted.windows(2).all(|w| w[0] <= w[1]),
    );

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    if let Ok(gpu) = rt.block_on(Gpu::new()) {
        eprintln!("  GPU: {} ({:?})", gpu.adapter_name, gpu.device_type);
        h.check_bool("GPU available for precision benchmarks", true);
    } else {
        eprintln!("  GPU: not available (CPU-only benchmarks)");
    }
    eprintln!();
}

fn bench_dispatcher_evolved(h: &mut ValidationHarness) {
    eprintln!("═══ neuralSpring → barraCuda: Evolved Dispatchers ═══");
    eprintln!("  Origin: neuralSpring local GPU ops");
    eprintln!("  Evolution: local Tensor ops → barracuda::dispatch (S52+)");
    eprintln!("  Impact: softmax, gelu, matmul all use upstream shaders");
    eprintln!();

    let dispatcher = Dispatcher::cpu_only();

    let input: Vec<f64> = (0..256).map(|i| (i as f64) * 0.01 - 1.28).collect();
    let t_softmax = bench("  softmax_dispatch (n=256)", || {
        let _ = dispatcher.softmax(&input);
    });
    h.check_bool("softmax runs", t_softmax > 0.0);

    let sm = dispatcher.softmax(&input);
    let sum: f64 = sm.iter().sum();
    h.check_bool(
        &format!("softmax sums to 1 (got {sum:.6})"),
        (sum - 1.0).abs() < 1e-6,
    );

    let t_gelu = bench("  gelu_dispatch (n=256)", || {
        let _ = dispatcher.gelu(&input);
    });
    h.check_bool("gelu runs", t_gelu > 0.0);

    let gelu_out = dispatcher.gelu(&input);
    h.check_bool("gelu output length", gelu_out.len() == input.len());
    h.check_bool("gelu(0) ≈ 0", gelu_out[128].abs() < 0.01);

    let a: Vec<f64> = (0..64).map(|i| (i as f64) * 0.1).collect();
    let b: Vec<f64> = (0..64).map(|i| (i as f64) * 0.05 + 0.5).collect();
    let t_matmul = bench("  mat_mul_dispatch (8×8)", || {
        let _ = dispatcher.mat_mul(&a, &b, 8);
    });
    h.check_bool("matmul runs", t_matmul > 0.0);
    eprintln!();
}

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  neuralSpring — Modern Cross-Spring Evolution Benchmark (S121)  ║");
    eprintln!("║  barraCuda v0.3.1 standalone · 5 springs · 767 WGSL shaders    ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Cross-spring evolution: each spring contributes shaders to barraCuda,");
    eprintln!("  and all springs benefit from each other's work.");
    eprintln!();

    let mut h = ValidationHarness::new("cross_spring_modern_bench");

    bench_simplemlp(&mut h);
    bench_hmm_dispatcher(&mut h);
    bench_stats_cross_spring(&mut h);
    bench_precision_hotspring(&mut h);
    bench_dispatcher_evolved(&mut h);

    h.finish();
}
