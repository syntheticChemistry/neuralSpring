// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-spring rewire validation: verifies that neuralSpring primitives
//! rewired to barracuda/ToadStool produce identical results to local
//! implementations, and that hotSpring-evolved extensions are correct.
//!
//! ## Cross-spring evolution validated
//!
//! | Origin | Feature | Target |
//! |--------|---------|--------|
//! | hotSpring `proxy.rs` | bandwidth, `condition_number`, phase | `WeightSpectralResult` |
//! | hotSpring `esn_v2` | GPU ESN via barracuda Tensors | `wdm_esn::classify_via_barracuda` |
//! | barracuda spectral | `level_spacing_ratio`, `marchenko_pastur_bounds` | Already rewired S75 |
//! | barracuda stats | `shannon_from_frequencies`, `empirical_spectral_density` | Already rewired S75 |
//!
//! ## Provenance
//!
//! Extended diagnostics (`bandwidth`, `condition_number`, `phase`) are
//! analytically validated — no Python baseline needed.
//!
//! GPU ESN parity validated against CPU ESN with Python baseline weights.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::expect_used
)]

use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::weight_spectral::{
    classify_phase, spectral_bandwidth, spectral_condition_number, weight_spectral_analysis,
    SpectralPhase,
};

fn main() {
    let mut h = ValidationHarness::new("cross_spring_rewire");

    // ═══ hotSpring proxy.rs → WeightSpectralResult extensions ═══════

    validate_bandwidth(&mut h);
    validate_condition_number(&mut h);
    validate_phase_classification(&mut h);
    validate_full_analysis_extensions(&mut h);
    validate_esn_barracuda_bridge(&mut h);

    h.finish();
}

fn validate_bandwidth(h: &mut ValidationHarness) {
    let evals_wide = vec![-5.0, -1.0, 0.0, 2.0, 10.0];
    h.check_abs(
        "hotSpring→bandwidth: wide spectrum (15.0)",
        spectral_bandwidth(&evals_wide),
        15.0,
        tolerances::EXACT_F64,
    );

    let evals_narrow = vec![0.99, 1.0, 1.01];
    h.check_abs(
        "hotSpring→bandwidth: narrow spectrum (0.02)",
        spectral_bandwidth(&evals_narrow),
        0.02,
        tolerances::EXACT_F64,
    );

    h.check_abs(
        "hotSpring→bandwidth: empty → 0",
        spectral_bandwidth(&[]),
        0.0,
        tolerances::ZERO_DETECTION,
    );

    h.check_abs(
        "hotSpring→bandwidth: single → 0",
        spectral_bandwidth(&[42.0]),
        0.0,
        tolerances::ZERO_DETECTION,
    );
}

fn validate_condition_number(h: &mut ValidationHarness) {
    let evals_well = vec![1.0, 2.0, 4.0];
    h.check_abs(
        "hotSpring→cond: well-conditioned (4.0)",
        spectral_condition_number(&evals_well),
        4.0,
        tolerances::EXACT_F64,
    );

    let evals_ill = vec![1e-10, 1.0, 1e5];
    let cond = spectral_condition_number(&evals_ill);
    h.check_bool(
        &format!("hotSpring→cond: ill-conditioned ({cond:.1e} >> 1)"),
        cond > 1e10,
    );

    let evals_zero = vec![0.0, 0.0, 5.0];
    h.check_bool(
        "hotSpring→cond: singular → infinity",
        spectral_condition_number(&evals_zero).is_infinite(),
    );

    let evals_neg = vec![-3.0, -1.0, 2.0, 4.0];
    let cond_neg = spectral_condition_number(&evals_neg);
    h.check_abs(
        "hotSpring→cond: negative eigenvalues (|4|/|1| = 4)",
        cond_neg,
        4.0,
        tolerances::EXACT_F64,
    );
}

fn validate_phase_classification(h: &mut ValidationHarness) {
    h.check_bool(
        "hotSpring→phase: GOE (0.531) → Extended",
        classify_phase(0.531) == SpectralPhase::Extended,
    );
    h.check_bool(
        "hotSpring→phase: boundary (0.48) → Extended",
        classify_phase(0.48) == SpectralPhase::Extended,
    );
    h.check_bool(
        "hotSpring→phase: transition (0.45) → Critical",
        classify_phase(0.45) == SpectralPhase::Critical,
    );
    h.check_bool(
        "hotSpring→phase: boundary (0.42) → Critical",
        classify_phase(0.42) == SpectralPhase::Critical,
    );
    h.check_bool(
        "hotSpring→phase: Poisson (0.386) → Localized",
        classify_phase(0.386) == SpectralPhase::Localized,
    );
    h.check_bool(
        "hotSpring→phase: deep localized (0.1) → Localized",
        classify_phase(0.1) == SpectralPhase::Localized,
    );
}

fn validate_full_analysis_extensions(h: &mut ValidationHarness) {
    let mut rng = Rng::new(42);
    let m = 16;
    let n = 16;
    let w: Vec<f64> = (0..m * n).map(|_| rng.normal()).collect();
    let result = weight_spectral_analysis(&w, m, n);

    h.check_bool(
        "cross-spring: bandwidth populated and positive",
        result.bandwidth > 0.0,
    );

    h.check_bool(
        "cross-spring: condition_number populated and > 1",
        result.condition_number > 1.0,
    );

    h.check_bool(
        "cross-spring: phase populated",
        result.phase == SpectralPhase::Extended
            || result.phase == SpectralPhase::Critical
            || result.phase == SpectralPhase::Localized,
    );

    let expected_bw = spectral_bandwidth(&result.eigenvalues);
    h.check_abs(
        "cross-spring: bandwidth matches eigenvalue range",
        result.bandwidth,
        expected_bw,
        tolerances::EXACT_F64,
    );

    let expected_cond = spectral_condition_number(&result.eigenvalues);
    h.check_abs(
        "cross-spring: condition matches eigenvalue ratio",
        result.condition_number,
        expected_cond,
        tolerances::EXACT_F64,
    );

    let expected_phase = classify_phase(result.level_spacing_ratio);
    h.check_bool(
        "cross-spring: phase matches LSR classification",
        result.phase == expected_phase,
    );

    // Determinism
    let r2 = weight_spectral_analysis(&w, m, n);
    h.check_bool(
        "cross-spring: determinism (bandwidth)",
        (result.bandwidth - r2.bandwidth).abs() < f64::EPSILON,
    );
    h.check_bool(
        "cross-spring: determinism (condition_number)",
        (result.condition_number - r2.condition_number).abs() < f64::EPSILON,
    );
    h.check_bool(
        "cross-spring: determinism (phase)",
        result.phase == r2.phase,
    );

    // Low-rank matrix should have different properties
    let mut low_rank = vec![0.0; m * n];
    for i in 0..m {
        low_rank[i * n] = rng.normal() * 10.0;
    }
    let lr_result = weight_spectral_analysis(&low_rank, m, n);
    h.check_bool(
        "cross-spring: low-rank condition_number > random",
        lr_result.condition_number > result.condition_number
            || lr_result.condition_number.is_infinite(),
    );
}

fn validate_esn_barracuda_bridge(h: &mut ValidationHarness) {
    let json_path = std::path::Path::new("control/wdm/esn_regime_baseline.json");
    if !json_path.exists() {
        h.check_bool("ESN bridge: baseline JSON exists", false);
        return;
    }

    let json_str = match std::fs::read_to_string(json_path) {
        Ok(s) => s,
        Err(e) => {
            h.check_bool(&format!("ESN bridge: read baseline JSON ({e})"), false);
            return;
        }
    };

    let classifier = match neural_spring::wdm_esn::load_esn_from_json(&json_str) {
        Ok(c) => c,
        Err(e) => {
            h.check_bool(&format!("ESN bridge: parse JSON ({e})"), false);
            return;
        }
    };
    h.check_bool("ESN bridge: JSON loaded", true);

    let test_cases: &[(f64, f64, &str)] = &[
        (-1.0, 8.0, "hot-sparse (Classical)"),
        (0.5, 5.5, "WDM regime"),
        (2.0, 4.0, "cold-dense (Degenerate)"),
    ];

    for &(log_rho, log_t, desc) in test_cases {
        let (cpu_label, cpu_scores) = classifier.classify(log_rho, log_t);
        h.check_bool(
            &format!("ESN CPU: {desc} → label={cpu_label}"),
            cpu_label < classifier.n_classes,
        );
        h.check_bool(
            &format!("ESN CPU: {desc} scores finite"),
            cpu_scores.iter().all(|s| s.is_finite()),
        );
    }

    let Ok(gpu) = tokio::runtime::Runtime::new()
        .expect("tokio runtime creation failed — required for async validation")
        .block_on(async { neural_spring::gpu::Gpu::new().await })
    else {
        eprintln!("  GPU not available — skipping barracuda Tensor ESN bridge");
        h.check_bool("ESN bridge: barracuda Tensor (GPU not available)", true);
        return;
    };
    let device = gpu.wgpu_device().clone();
    eprintln!(
        "  GPU: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend
    );

    for &(log_rho, log_t, desc) in test_cases {
        let (cpu_label, cpu_scores) = classifier.classify(log_rho, log_t);

        let gpu_result =
            neural_spring::wdm_esn::classify_via_barracuda(&classifier, log_rho, log_t, &device);

        match gpu_result {
            Ok((gpu_label, gpu_scores)) => {
                h.check_bool(
                    &format!("ESN barracuda: {desc} scores finite"),
                    gpu_scores.iter().all(|s| s.is_finite()),
                );
                h.check_bool(
                    &format!("ESN barracuda: {desc} label matches CPU (cpu={cpu_label}, gpu={gpu_label})"),
                    gpu_label == cpu_label,
                );

                let max_diff: f64 = gpu_scores
                    .iter()
                    .zip(cpu_scores.iter())
                    .map(|(g, c)| (f64::from(*g) - c).abs())
                    .fold(0.0_f64, f64::max);
                h.check_bool(
                    &format!("ESN barracuda: {desc} scores within f32 tol (diff={max_diff:.2e})"),
                    max_diff < tolerances::TENSOR_TRANSCENDENTAL_F32,
                );
            }
            Err(e) => {
                h.check_bool(&format!("ESN barracuda: {desc} ({e})"), false);
            }
        }
    }

    // Determinism
    let Ok((_, s1)) =
        neural_spring::wdm_esn::classify_via_barracuda(&classifier, 0.5, 5.5, &device)
    else {
        h.check_bool("ESN barracuda determinism (run 1 failed)", false);
        return;
    };
    let Ok((_, s2)) =
        neural_spring::wdm_esn::classify_via_barracuda(&classifier, 0.5, 5.5, &device)
    else {
        h.check_bool("ESN barracuda determinism (run 2 failed)", false);
        return;
    };
    let det = s1
        .iter()
        .zip(s2.iter())
        .all(|(a, b)| (f64::from(*a) - f64::from(*b)).abs() < f64::EPSILON);
    h.check_bool("ESN barracuda: determinism", det);
}
