// SPDX-License-Identifier: AGPL-3.0-or-later

//! `ToadStool` S70+++ cross-spring evolution validator: exercises newly absorbed
//! APIs and traces shader provenance across all five springs.
//!
//! ## S70+++ Absorption Provenance
//!
//! ```text
//! hotSpring  → DF64 activation shaders (gelu_df64, sigmoid_df64, softmax_df64,
//!              layer_norm_df64, sdpa_df64) → ToadStool S70+ → neuralSpring ML
//! wetSpring  → bio diversity (chao1_classic), ODE bio → ToadStool S70+ → stats
//! airSpring  → batched_elementwise (HargreavesEt0, SensorCal, KcClimate, DualKc),
//!              seasonal_pipeline.wgsl, fao56_et0 → ToadStool S70+ → hydrology
//! groundSpring → evolution stats (kimura, error_threshold, detection_power),
//!                jackknife resampling → ToadStool S70+ → stats
//! neuralSpring → matmul_ref (non-consuming for ESN/LSTM), SimpleMlp (JSON serde)
//!                → ToadStool S70+ absorption → used by all springs
//! ```
//!
//! Each check validates correctness and notes the cross-spring evolution chain.
//!
//! ```text
//! cargo run --release --bin validate_toadstool_s70_evolution
//! ```

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::expect_used,
    reason = "validation binary"
)]

use neural_spring::rng::Rng;
use neural_spring::validation::{bench_once, ValidationHarness};

struct BenchResult {
    label: &'static str,
    provenance: &'static str,
    us: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// groundSpring provenance: evolution + jackknife (S70+)
// ═══════════════════════════════════════════════════════════════════════════════

fn validate_groundspring_evolution(h: &mut ValidationHarness) {
    eprintln!("\n─── groundSpring → ToadStool S70+: evolution stats ───\n");

    // Kimura fixation probability (groundSpring → ToadStool S70+ → barracuda::stats::evolution)
    let (fix_neutral, _) = bench_once("kimura fixation (neutral)", || {
        barracuda::stats::evolution::kimura_fixation_prob(1000, 0.0, 0.001)
    });
    h.check_abs(
        "gS→evolution: kimura neutral ≈ initial freq (1/N drift)",
        fix_neutral,
        0.001,
        0.01,
    );

    let (fix_beneficial, _) = bench_once("kimura fixation (beneficial s=0.01)", || {
        barracuda::stats::evolution::kimura_fixation_prob(1000, 0.01, 0.001)
    });
    h.check_bool(
        "gS→evolution: kimura beneficial > neutral",
        fix_beneficial > fix_neutral,
    );

    // Error threshold (Eigen's quasispecies — groundSpring spectral theory)
    let (threshold, _) = bench_once("error threshold (L=100)", || {
        barracuda::stats::evolution::error_threshold(2.0, 100)
    });
    h.check_bool(
        "gS→evolution: error_threshold returns Some for L=100",
        threshold.is_some(),
    );
    if let Some(q_min) = threshold {
        h.check_bool(
            "gS→evolution: error_threshold q_min in (0,1)",
            (0.0..1.0).contains(&q_min),
        );
    }

    // Detection power/threshold (groundSpring rare biosphere)
    let (power, _) = bench_once("detection power (abundance=0.01, depth=1000)", || {
        barracuda::stats::evolution::detection_power(0.01, 1000)
    });
    h.check_bool(
        "gS→evolution: detection_power in [0,1]",
        (0.0..=1.0).contains(&power),
    );

    let (depth_needed, _) = bench_once("detection threshold (abundance=0.01, power=0.95)", || {
        barracuda::stats::evolution::detection_threshold(0.01, 0.95)
    });
    h.check_bool("gS→evolution: detection_threshold > 0", depth_needed > 0);

    // Jackknife resampling (groundSpring uncertainty → ToadStool S70+)
    let data: Vec<f64> = (0..50).map(|i| (i as f64).mul_add(0.5, 1.0)).collect();
    let (jk_result, _) = bench_once("jackknife mean/variance (n=50)", || {
        barracuda::stats::jackknife::jackknife_mean_variance(&data)
    });
    h.check_bool("gS→jackknife: result is Some for n=50", jk_result.is_some());
    if let Some(ref jk) = jk_result {
        h.check_abs(
            "gS→jackknife: mean ≈ analytical (13.25)",
            jk.estimate,
            13.25,
            0.5,
        );
        h.check_bool("gS→jackknife: std_error > 0", jk.std_error > 0.0);
    }

    // Generalized jackknife with custom statistic
    let (jk_custom, _) = bench_once("jackknife custom (variance)", || {
        barracuda::stats::jackknife::jackknife(&data, |slice| {
            let mean = slice.iter().sum::<f64>() / slice.len() as f64;
            slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (slice.len() - 1) as f64
        })
    });
    h.check_bool(
        "gS→jackknife: custom statistic produces result",
        jk_custom.is_some(),
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// airSpring provenance: hydrology (S70+)
// ═══════════════════════════════════════════════════════════════════════════════

fn validate_airspring_hydrology(h: &mut ValidationHarness) {
    eprintln!("\n─── airSpring → ToadStool S70+: hydrology ───\n");

    // FAO-56 Penman-Monteith ET₀ (airSpring → ToadStool S70+)
    // fao56_et0(t_max, t_min, rh_max, rh_min, wind_2m, rs, elevation, lat_deg, doy)
    let (et0, _) = bench_once("fao56_et0 (summer day)", || {
        barracuda::stats::hydrology::fao56_et0(
            30.0,  // t_max °C
            20.0,  // t_min °C
            80.0,  // rh_max %
            40.0,  // rh_min %
            2.0,   // wind_2m m/s
            20.0,  // rs MJ/m²/day
            100.0, // elevation m
            45.0,  // lat_deg
            180,   // doy (summer solstice)
        )
    });
    h.check_bool("aS→hydrology: fao56_et0 returns Some", et0.is_some());
    if let Some(val) = et0 {
        h.check_bool(
            "aS→hydrology: fao56_et0 in reasonable range (0-20 mm/day)",
            (0.0..20.0).contains(&val),
        );
    }

    // Hargreaves ET₀ (airSpring temperature-based method)
    // hargreaves_et0(ra, t_max, t_min) where ra = extraterrestrial radiation
    let (hg, _) = bench_once("hargreaves_et0", || {
        barracuda::stats::hydrology::hargreaves_et0(35.0, 30.0, 20.0)
    });
    h.check_bool("aS→hydrology: hargreaves returns Some", hg.is_some());
    if let Some(val) = hg {
        h.check_bool(
            "aS→hydrology: hargreaves in range (0-20 mm/day)",
            (0.0..20.0).contains(&val),
        );
    }

    // Crop coefficient interpolation (airSpring FAO-56)
    let (kc, _) = bench_once("crop_coefficient (mid-stage)", || {
        barracuda::stats::hydrology::crop_coefficient(0.3, 1.2, 15, 30)
    });
    h.check_bool("aS→hydrology: kc in [0.3, 1.2]", (0.3..=1.2).contains(&kc));

    // Soil water balance (airSpring coupled pipeline)
    // soil_water_balance(theta, precip, irrigation, et_c, field_capacity)
    let (swb, _) = bench_once("soil_water_balance", || {
        barracuda::stats::hydrology::soil_water_balance(
            100.0, // theta (current storage) mm
            5.0,   // precipitation mm
            0.0,   // irrigation mm
            4.0,   // et_c mm
            150.0, // field_capacity mm
        )
    });
    h.check_bool(
        "aS→hydrology: soil_water_balance in [0, field_capacity]",
        (0.0..=150.0).contains(&swb),
    );
    h.check_abs(
        "aS→hydrology: swb = theta + precip + irrig - et_c",
        swb,
        101.0, // 100 + 5 + 0 - 4
        0.01,
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// wetSpring provenance: chao1_classic (S70+)
// ═══════════════════════════════════════════════════════════════════════════════

fn validate_wetspring_diversity(h: &mut ValidationHarness) {
    eprintln!("\n─── wetSpring → ToadStool S70+: diversity ───\n");

    // chao1_classic with u64 counts (wetSpring Chao 1984 → ToadStool S70+)
    let counts_u64: Vec<u64> = vec![10, 5, 3, 1, 1, 1, 1, 0, 0, 0];
    let (chao1, _) = bench_once("chao1_classic (u64 counts)", || {
        barracuda::stats::diversity::chao1_classic(&counts_u64)
    });
    let observed = counts_u64.iter().filter(|&&c| c > 0).count() as f64;
    h.check_bool(
        "wS→diversity: chao1_classic ≥ observed species",
        chao1 >= observed,
    );

    // Compare with f64-based chao1
    let counts_f64: Vec<f64> = counts_u64.iter().map(|&c| c as f64).collect();
    let (chao1_f64, _) = bench_once("chao1 (f64 counts)", || {
        barracuda::stats::diversity::chao1(&counts_f64)
    });
    h.check_abs(
        "wS→diversity: chao1_classic ≈ chao1 (f64 path)",
        chao1,
        chao1_f64,
        1.0, // allow small difference due to integer vs float counting
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// neuralSpring provenance: matmul_ref + SimpleMlp (S70+)
// ═══════════════════════════════════════════════════════════════════════════════

fn validate_neuralspring_s70(
    h: &mut ValidationHarness,
    device: &std::sync::Arc<barracuda::device::WgpuDevice>,
) {
    eprintln!("\n─── neuralSpring → ToadStool S70+: matmul_ref + SimpleMlp ───\n");

    let m = 8;
    let k = 16;
    let n = 4;
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((k * n - i) as f32) * 0.01).collect();

    let a_tensor = barracuda::tensor::Tensor::from_data(&a_data, vec![m, k], device.clone())
        .expect("create A tensor");
    let b_tensor = barracuda::tensor::Tensor::from_data(&b_data, vec![k, n], device.clone())
        .expect("create B tensor");

    // First matmul_ref — non-consuming, a_tensor survives
    let (c1, t1) = bench_once("matmul_ref (8×16 × 16×4)", || {
        a_tensor.matmul_ref(&b_tensor).expect("matmul_ref call 1")
    });
    let c1_vec = c1.to_vec().expect("readback c1");
    h.check_bool(
        "nS→matmul_ref: output shape correct (m*n elements)",
        c1_vec.len() == m * n,
    );

    // Second matmul_ref on same tensor — proves non-consuming
    let (c2, t2) = bench_once("matmul_ref (reuse same tensor)", || {
        a_tensor.matmul_ref(&b_tensor).expect("matmul_ref call 2")
    });
    let c2_vec = c2.to_vec().expect("readback c2");
    h.check_abs(
        "nS→matmul_ref: repeated call bit-identical",
        c1_vec
            .iter()
            .zip(c2_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max) as f64,
        0.0,
        1e-12,
    );

    // Consuming matmul on same tensor (last use)
    let (c3, _) = bench_once("matmul (consuming, last use)", || {
        a_tensor.matmul(&b_tensor).expect("consuming matmul")
    });
    let c3_vec = c3.to_vec().expect("readback c3");
    h.check_abs(
        "nS→matmul_ref: ref vs consuming identical",
        c1_vec
            .iter()
            .zip(c3_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max) as f64,
        0.0,
        1e-12,
    );

    // SimpleMlp: feed-forward MLP with JSON weight serde
    let mlp = barracuda::nn::simple_mlp::SimpleMlp {
        layers: vec![
            barracuda::nn::simple_mlp::DenseLayer {
                weight: vec![vec![0.5, -0.3], vec![-0.2, 0.8], vec![0.1, 0.4]],
                bias: vec![0.1, -0.1, 0.05],
                activation: barracuda::nn::simple_mlp::Activation::Relu,
            },
            barracuda::nn::simple_mlp::DenseLayer {
                weight: vec![vec![0.6, -0.4, 0.2]],
                bias: vec![0.0],
                activation: barracuda::nn::simple_mlp::Activation::Identity,
            },
        ],
    };

    let input = vec![1.0_f64, 0.5];
    let (output, _) = bench_once("SimpleMlp forward (2→3→1)", || mlp.forward(&input));
    h.check_bool(
        "nS→SimpleMlp: output length = 1 (single output)",
        output.len() == 1,
    );
    h.check_bool("nS→SimpleMlp: output finite", output[0].is_finite());

    // Verify manually: layer 1: y = relu(W·x + b)
    // [0.5*1.0 + (-0.3)*0.5 + 0.1, -0.2*1.0 + 0.8*0.5 + (-0.1), 0.1*1.0 + 0.4*0.5 + 0.05]
    // = [0.45, 0.1, 0.35] after relu = [0.45, 0.1, 0.35] (all positive)
    // layer 2: y = 0.6*0.45 + (-0.4)*0.1 + 0.2*0.35 + 0.0 = 0.27 - 0.04 + 0.07 = 0.3
    h.check_abs(
        "nS→SimpleMlp: forward matches hand computation",
        output[0],
        0.3,
        1e-10,
    );

    // JSON round-trip
    let json = serde_json::to_string(&mlp).expect("serialize SimpleMlp");
    let (mlp2, _) = bench_once("SimpleMlp JSON round-trip", || {
        serde_json::from_str::<barracuda::nn::simple_mlp::SimpleMlp>(&json)
            .expect("deserialize SimpleMlp")
    });
    let output2 = mlp2.forward(&input);
    h.check_abs(
        "nS→SimpleMlp: JSON round-trip preserves output",
        (output[0] - output2[0]).abs(),
        0.0,
        1e-15,
    );

    eprintln!("\n  matmul_ref benchmark: call1={t1:.1}µs, call2={t2:.1}µs (no clone overhead)");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-spring throughput benchmark (S70+++ expanded)
// ═══════════════════════════════════════════════════════════════════════════════

fn benchmark_s70_throughput(
    h: &mut ValidationHarness,
    device: &std::sync::Arc<barracuda::device::WgpuDevice>,
) {
    eprintln!("\n─── S70+++ throughput benchmark (cross-spring provenance) ───\n");

    let mut results: Vec<BenchResult> = Vec::new();
    let mut rng = Rng::new(42);

    // Kimura fixation (groundSpring → ToadStool S70+)
    let ((), us) = bench_once("kimura 10K iterations", || {
        for pop in (100..10_100).step_by(1) {
            let _ = barracuda::stats::evolution::kimura_fixation_prob(pop, 0.01, 0.001);
        }
    });
    results.push(BenchResult {
        label: "kimura 10K",
        provenance: "gS→TS",
        us,
    });

    // Jackknife (groundSpring → ToadStool S70+)
    let big_data: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.1).collect();
    let (_, us) = bench_once("jackknife n=1000", || {
        barracuda::stats::jackknife::jackknife_mean_variance(&big_data)
    });
    results.push(BenchResult {
        label: "jackknife 1K",
        provenance: "gS→TS",
        us,
    });

    // chao1_classic (wetSpring → ToadStool S70+)
    let counts: Vec<u64> = (0..500).map(|i| if i < 200 { i + 1 } else { 0 }).collect();
    let (_, us) = bench_once("chao1_classic 500 taxa", || {
        barracuda::stats::diversity::chao1_classic(&counts)
    });
    results.push(BenchResult {
        label: "chao1 500",
        provenance: "wS→TS",
        us,
    });

    // fao56_et0 (airSpring → ToadStool S70+)
    let ((), us) = bench_once("fao56_et0 10K calls", || {
        for t in 0..10_000 {
            let temp = (t as f64).mul_add(0.001, 15.0);
            let _ = barracuda::stats::hydrology::fao56_et0(
                temp + 5.0,
                temp - 5.0,
                80.0,
                40.0,
                2.0,
                20.0,
                100.0,
                45.0,
                180,
            );
        }
    });
    results.push(BenchResult {
        label: "fao56 10K",
        provenance: "aS→TS",
        us,
    });

    // matmul_ref (neuralSpring → ToadStool S70+)
    let n = 64;
    let a_data: Vec<f32> = (0..n * n).map(|_| rng.uniform() as f32).collect();
    let b_data: Vec<f32> = (0..n * n).map(|_| rng.uniform() as f32).collect();
    let a_tensor = barracuda::tensor::Tensor::from_data(&a_data, vec![n, n], device.clone())
        .expect("create bench A tensor");
    let b_tensor = barracuda::tensor::Tensor::from_data(&b_data, vec![n, n], device.clone())
        .expect("create bench B tensor");
    let ((), us) = bench_once("matmul_ref 64×64", || {
        let _ = a_tensor.matmul_ref(&b_tensor).expect("bench matmul_ref");
    });
    results.push(BenchResult {
        label: "matmul_ref 64²",
        provenance: "nS→TS",
        us,
    });

    // SimpleMlp (neuralSpring → ToadStool S70+)
    let mlp = barracuda::nn::simple_mlp::SimpleMlp {
        layers: vec![
            barracuda::nn::simple_mlp::DenseLayer {
                weight: (0..64)
                    .map(|_| (0..32).map(|_| rng.uniform()).collect())
                    .collect(),
                bias: (0..64).map(|_| rng.uniform() * 0.1).collect(),
                activation: barracuda::nn::simple_mlp::Activation::Relu,
            },
            barracuda::nn::simple_mlp::DenseLayer {
                weight: (0..3)
                    .map(|_| (0..64).map(|_| rng.uniform()).collect())
                    .collect(),
                bias: (0..3).map(|_| rng.uniform() * 0.1).collect(),
                activation: barracuda::nn::simple_mlp::Activation::Identity,
            },
        ],
    };
    let test_input: Vec<f64> = (0..32).map(|_| rng.uniform()).collect();
    let ((), us) = bench_once("SimpleMlp 32→64→3 × 1000", || {
        for _ in 0..1000 {
            let _ = mlp.forward(&test_input);
        }
    });
    results.push(BenchResult {
        label: "MLP 1K fwd",
        provenance: "nS→TS",
        us,
    });

    // Print benchmark table
    eprintln!("\n  ┌─────────────────┬────────────┬──────────┐");
    eprintln!("  │ Operation       │ Provenance │    µs    │");
    eprintln!("  ├─────────────────┼────────────┼──────────┤");
    for r in &results {
        eprintln!(
            "  │ {:<15} │ {:<10} │ {:>8.1} │",
            r.label, r.provenance, r.us
        );
    }
    eprintln!("  └─────────────────┴────────────┴──────────┘");

    h.check_bool(
        &format!("bench: {}/6 S70+++ ops timed", results.len()),
        results.len() == 6,
    );
}

fn report_s70_provenance() {
    eprintln!("\n═══ S70+++ Cross-Spring Evolution Provenance ═══");
    eprintln!();
    eprintln!("  Source Spring    → ToadStool S70+ Absorption        → neuralSpring S97c");
    eprintln!("  ───────────────────────────────────────────────────────────────────────");
    eprintln!("  hotSpring        → gelu_df64, sigmoid_df64,         → DF64 ML precision");
    eprintln!("                     softmax_df64, layer_norm_df64,     path ready for");
    eprintln!("                     sdpa_df64 (DF64 ML shaders)        protein folding");
    eprintln!("  wetSpring        → chao1_classic (u64 counts),      → alpha diversity,");
    eprintln!("                     diversity extensions                metagenomics QC");
    eprintln!("  airSpring        → fao56_et0, hargreaves_et0,       → hydrology stats,");
    eprintln!("                     crop_coefficient, soil_water_      water balance");
    eprintln!("                     balance, seasonal_pipeline.wgsl    pipeline GPU");
    eprintln!("  groundSpring     → kimura_fixation_prob, error_     → evolution theory,");
    eprintln!("                     threshold, detection_power,        rare biosphere");
    eprintln!("                     jackknife (leave-one-out)          uncertainty");
    eprintln!("  neuralSpring     → matmul_ref (non-consuming),      → ESN/LSTM recurrence");
    eprintln!("                     SimpleMlp (JSON serde, 5 acts)     WDM surrogates S97d");
    eprintln!();
    eprintln!("  ToadStool S70+++ (1dd7e338):");
    eprintln!("    668 WGSL shaders, 26 DF64, 4700+ workspace tests");
    eprintln!("    ComputeDispatch: 34/250 migrated to fluent builder");
    eprintln!("    chrono eliminated, unsafe 45, 0 clippy warnings");
    eprintln!("    All springs contribute → ToadStool absorbs → all springs benefit");
    eprintln!();
}

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("toadstool_s70_evolution");

    validate_groundspring_evolution(&mut h);
    validate_airspring_hydrology(&mut h);
    validate_wetspring_diversity(&mut h);

    let Ok(gpu) = neural_spring::gpu::Gpu::new().await else {
        neural_spring::validation::exit_no_gpu();
    };
    let device = gpu.wgpu_device().clone();

    validate_neuralspring_s70(&mut h, &device);
    benchmark_s70_throughput(&mut h, &device);

    report_s70_provenance();

    h.finish();
}
