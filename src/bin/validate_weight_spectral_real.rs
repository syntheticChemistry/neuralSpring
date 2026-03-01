// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: Weight spectral analysis on real pretrained models (nS-01 Paper A).
//!
//! First real-data experiment for baseCamp Sub-thesis 01: Weight Matrices
//! as Disordered Hamiltonians. Loads pretrained model weights from
//! safetensors files and runs Anderson spectral diagnostics (IPR, level
//! spacing ratio, Marchenko-Pastur departure, spectral entropy) on each
//! weight matrix layer.
//!
//! ## Data requirement
//!
//! Run `python scripts/download_pretrained.py` first to generate
//! `control/weight_spectral/pretrained/*.safetensors`.
//!
//! ## Novel questions
//!
//! - Do real weight matrices cluster near GOE (r ~ 0.531) or Poisson (r ~ 0.386)?
//! - Does IPR correlate with layer depth?
//! - Does Marchenko-Pastur departure predict layer importance?
//! - Are there spectral fingerprints that distinguish architectures?

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names
)]

use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::weight_loader;
use neural_spring::weight_spectral::{
    empirical_spectral_density, weight_spectral_analysis, GOE_LEVEL_SPACING, POISSON_LEVEL_SPACING,
};
use std::path::PathBuf;
use std::time::Instant;

fn pretrained_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("control")
        .join("weight_spectral")
        .join("pretrained")
}

fn main() {
    let mut h = ValidationHarness::new("weight_spectral_real");

    let dir = pretrained_dir();
    if !dir.exists() {
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║  No pretrained weights found. Run:                          ║");
        eprintln!("║    python scripts/download_pretrained.py                    ║");
        eprintln!("║  to download models into control/weight_spectral/pretrained ║");
        eprintln!("╚══════════════════════════════════════════════════════════════╝");
        eprintln!();
        eprintln!("[SKIP] No pretrained data — generating synthetic fallback");
        validate_synthetic_fallback(&mut h);
        h.finish();
    }

    let read_dir = match std::fs::read_dir(&dir) {
        Ok(rd) => rd,
        Err(e) => {
            eprintln!("[ERROR] Cannot read pretrained dir {}: {e}", dir.display());
            h.check_bool("pretrained dir readable", false);
            h.finish();
        }
    };
    let mut models: Vec<PathBuf> = read_dir
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    models.sort();

    if models.is_empty() {
        eprintln!("[SKIP] No .safetensors files found — using synthetic fallback");
        validate_synthetic_fallback(&mut h);
        h.finish();
    }

    eprintln!(
        "Found {} pretrained models in {}",
        models.len(),
        dir.display()
    );
    eprintln!();

    let mut total_layers = 0usize;
    let mut total_time_ms = 0.0f64;
    let mut all_lsr: Vec<f64> = Vec::new();
    let mut all_ipr: Vec<f64> = Vec::new();
    let mut all_mp_dep: Vec<f64> = Vec::new();

    for model_path in &models {
        let model_name = model_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        eprintln!("═══ {model_name} ═══");

        let weights = match weight_loader::load_all_weight_matrices(model_path) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("  ERROR loading {model_name}: {e}");
                continue;
            }
        };

        let max_dim = 512;
        let mut model_layers = 0;

        for tensor in &weights.tensors {
            if tensor.rows > max_dim || tensor.cols > max_dim {
                continue;
            }
            if tensor.rows < 4 || tensor.cols < 4 {
                continue;
            }

            let start = Instant::now();
            let result = weight_spectral_analysis(&tensor.data, tensor.rows, tensor.cols);
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            total_time_ms += elapsed_ms;

            let tag = format!("{model_name}/{}", tensor.name);

            h.check_bool(
                &format!("{tag}: all eigenvalues finite"),
                result.eigenvalues.iter().all(|v| v.is_finite()),
            );

            let (_, counts) = empirical_spectral_density(&result.eigenvalues, 10);
            let esd_sum: f64 = counts.iter().sum();
            h.check_abs(
                &format!("{tag}: ESD sums to 1.0"),
                esd_sum,
                1.0,
                tolerances::EXACT_F64,
            );

            h.check_bool(&format!("{tag}: IPR positive"), result.mean_ipr > 0.0);
            h.check_bool(
                &format!("{tag}: IPR bounded"),
                result.mean_ipr <= 1.0 + tolerances::EXACT_F64,
            );

            h.check_bool(
                &format!("{tag}: LSR in [0, 1]"),
                result.level_spacing_ratio >= 0.0 && result.level_spacing_ratio <= 1.0,
            );

            h.check_bool(
                &format!("{tag}: entropy non-negative"),
                result.spectral_entropy >= 0.0,
            );

            h.check_bool(
                &format!("{tag}: MP departure in [0, 1]"),
                result.mp_departure >= 0.0 && result.mp_departure <= 1.0,
            );

            all_lsr.push(result.level_spacing_ratio);
            all_ipr.push(result.mean_ipr);
            all_mp_dep.push(result.mp_departure);

            eprintln!(
                "  {}: {}×{} | IPR={:.4} LSR={:.4} entropy={:.4} MP_dep={:.4} | {:.1}ms",
                tensor.name,
                tensor.rows,
                tensor.cols,
                result.mean_ipr,
                result.level_spacing_ratio,
                result.spectral_entropy,
                result.mp_departure,
                elapsed_ms,
            );

            model_layers += 1;
            total_layers += 1;
        }

        eprintln!("  [{model_name}] {model_layers} layers analyzed");
        eprintln!();
    }

    // ── Aggregate spectral characterization ──────────────────────────

    if !all_lsr.is_empty() {
        let mean_lsr = all_lsr.iter().sum::<f64>() / all_lsr.len() as f64;
        let mean_ipr = all_ipr.iter().sum::<f64>() / all_ipr.len() as f64;
        let mean_mp = all_mp_dep.iter().sum::<f64>() / all_mp_dep.len() as f64;

        let goe_fraction = all_lsr
            .iter()
            .filter(|&&r| (r - GOE_LEVEL_SPACING).abs() < (r - POISSON_LEVEL_SPACING).abs())
            .count() as f64
            / all_lsr.len() as f64;

        eprintln!("═══ Aggregate Results ({total_layers} layers) ═══");
        eprintln!("  Mean LSR:          {mean_lsr:.4} (GOE={GOE_LEVEL_SPACING:.3}, Poisson={POISSON_LEVEL_SPACING:.3})");
        eprintln!("  Mean IPR:          {mean_ipr:.6}");
        eprintln!("  Mean MP departure: {mean_mp:.4}");
        eprintln!(
            "  GOE-like fraction: {goe_fraction:.2} ({}/{total_layers})",
            (goe_fraction * total_layers as f64).round() as usize
        );
        eprintln!(
            "  Total time:        {total_time_ms:.1}ms ({:.1}ms/layer)",
            total_time_ms / total_layers as f64
        );
        eprintln!();

        h.check_bool("Aggregate: mean LSR finite", mean_lsr.is_finite());
        h.check_bool("Aggregate: mean IPR finite", mean_ipr.is_finite());
        h.check_bool("Aggregate: GOE fraction > 0", goe_fraction > 0.0);
    }

    h.finish();
}

/// Synthetic fallback when no pretrained weights are available.
/// Uses random matrices to verify the pipeline works end-to-end.
fn validate_synthetic_fallback(h: &mut ValidationHarness) {
    use neural_spring::rng::Rng;

    let mut rng = Rng::new(42);

    for (m, n, label) in [(32, 64, "small"), (64, 64, "square"), (128, 32, "tall")] {
        let weights: Vec<f64> = (0..m * n).map(|_| rng.normal()).collect();

        let result = weight_spectral_analysis(&weights, m, n);

        h.check_bool(
            &format!("synthetic/{label}: eigenvalues finite"),
            result.eigenvalues.iter().all(|v| v.is_finite()),
        );

        let (_, counts) = empirical_spectral_density(&result.eigenvalues, 10);
        let esd_sum: f64 = counts.iter().sum();
        h.check_abs(
            &format!("synthetic/{label}: ESD sums to 1"),
            esd_sum,
            1.0,
            tolerances::EXACT_F64,
        );

        h.check_bool(
            &format!("synthetic/{label}: IPR positive"),
            result.mean_ipr > 0.0,
        );

        h.check_bool(
            &format!("synthetic/{label}: LSR in [0,1]"),
            result.level_spacing_ratio >= 0.0 && result.level_spacing_ratio <= 1.0,
        );

        eprintln!(
            "  synthetic/{label} ({}×{}): IPR={:.4} LSR={:.4} entropy={:.4}",
            m, n, result.mean_ipr, result.level_spacing_ratio, result.spectral_entropy
        );
    }
}
