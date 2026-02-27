// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: Exp-052 — Hessian Eigenanalysis at Trained Minima.
//!
//! Paper D: GPU-Accelerated Nudged Elastic Band for Neural Network
//! Loss Landscape Analysis (Digital Discovery, RSC).
//!
//! Validates that the numerical Hessian computation and eigenanalysis
//! primitives produce correct spectral diagnostics. Uses synthetic
//! quadratic loss surfaces with known analytical Hessians.
//!
//! ## Provenance
//!
//! - **Python baseline**: `control/hessian_eigenanalysis/hessian_eigenanalysis.py`
//! - **Baseline values**: `control/hessian_eigenanalysis/baseline_values.json`
//! - **Rust primitives**: `loss_landscape.rs`, `eigh.rs`, `numerical::hessian`
//! - **Commit**: `BASELINE_COMMIT` (`f9ad0268`)
//! - **Date**: 2026-02-26
//! - **Command**: `python3 control/hessian_eigenanalysis/hessian_eigenanalysis.py`
//! - **Environment**: Python 3.12, `PyTorch` 2.9.0+cu128, `NumPy`, seed=42
//! - **Hardware**: Eastgate (i9-12900K, RTX 4070 12GB, Pop!_OS 22.04)
//! - **Provenance record**: `provenance::HESSIAN_EIGENANALYSIS_PROVENANCE`

#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_lines
)]

use neural_spring::eigh::eigh_householder_qr;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::weight_spectral::spectral_entropy;

fn main() {
    let mut h = ValidationHarness::new("hessian_eigenanalysis (Exp-052)");

    // ── 1. Analytical Hessian of quadratic: f(x) = 0.5 * x^T H x ──
    //
    // For a diagonal quadratic, the Hessian is the diagonal matrix itself.
    // Eigenvalues of the Hessian are the diagonal entries.

    let n = 20;
    let mut hessian_exact = vec![0.0; n * n];
    let expected_eigenvalues: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    for (i, &ev) in expected_eigenvalues.iter().enumerate() {
        hessian_exact[i * n + i] = ev;
    }

    let decomp = eigh_householder_qr(&hessian_exact, n);
    let mut computed_evals = decomp.eigenvalues;
    computed_evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let max_diff: f64 = computed_evals
        .iter()
        .zip(expected_eigenvalues.iter())
        .map(|(c, e)| (c - e).abs())
        .fold(0.0f64, f64::max);

    h.check_abs(
        "Diagonal Hessian eigenvalues exact",
        max_diff,
        0.0,
        tolerances::RELATIVE_ERROR_FLOOR,
    );

    // ── 2. Near-zero fraction computation ───────────────────────────

    let threshold = 0.5;
    let near_zero_count = computed_evals
        .iter()
        .filter(|&&v| v.abs() < threshold)
        .count();

    h.check_bool(
        "Near-zero fraction: no eigenvalues < 0.5 in [1..20] spectrum",
        near_zero_count == 0,
    );

    // ── 3. Flat vs sharp minimum ────────────────────────────────────
    //
    // Flat: small eigenvalues → many directions with low curvature
    // Sharp: large eigenvalues → few flat directions

    let mut flat_hessian = vec![0.0; n * n];
    let mut sharp_hessian = vec![0.0; n * n];
    for i in 0..n {
        flat_hessian[i * n + i] = 0.01;
        sharp_hessian[i * n + i] = 100.0;
    }

    let flat_evals = eigh_householder_qr(&flat_hessian, n).eigenvalues;
    let sharp_evals = eigh_householder_qr(&sharp_hessian, n).eigenvalues;

    let flat_max = flat_evals.iter().copied().fold(0.0f64, f64::max);
    let sharp_max = sharp_evals.iter().copied().fold(0.0f64, f64::max);

    h.check_bool(
        "Sharp minimum has larger max eigenvalue than flat",
        sharp_max > flat_max * 100.0,
    );

    let flat_nz = flat_evals.iter().filter(|&&v| v.abs() < 0.1).count() as f64 / n as f64;
    let sharp_nz = sharp_evals.iter().filter(|&&v| v.abs() < 0.1).count() as f64 / n as f64;

    h.check_bool(
        "Flat minimum has higher near-zero fraction",
        flat_nz >= sharp_nz,
    );

    // ── 4. Positive-definite Hessian → positive trace ───────────────

    let trace: f64 = computed_evals.iter().sum();
    h.check_bool("Positive-definite Hessian has positive trace", trace > 0.0);

    let trace_expected: f64 = expected_eigenvalues.iter().sum();
    h.check_abs(
        "Trace matches sum of eigenvalues",
        trace,
        trace_expected,
        tolerances::RELATIVE_ERROR_FLOOR,
    );

    // ── 5. Hessian spectral entropy ─────────────────────────────────

    let se_flat = spectral_entropy(&flat_evals);
    let se_sharp = spectral_entropy(&sharp_evals);

    h.check_bool(
        "Spectral entropy is finite for flat minimum",
        se_flat.is_finite(),
    );

    // For identical eigenvalues, entropy should be maximal (uniform distribution)
    // Both flat and sharp have identical eigenvalues, so entropy should be equal
    h.check_abs(
        "Equal-eigenvalue Hessians have equal spectral entropy",
        se_flat,
        se_sharp,
        tolerances::RELATIVE_ERROR_FLOOR,
    );

    // ── 6. Mixed Hessian: varying curvature ─────────────────────────

    let mut mixed = vec![0.0; n * n];
    let mut rng = Rng::new(42);
    for i in 0..n {
        mixed[i * n + i] = (rng.uniform() * 9.0) + 1.0;
    }

    let mixed_evals = eigh_householder_qr(&mixed, n).eigenvalues;
    let se_mixed = spectral_entropy(&mixed_evals);

    h.check_bool(
        "Mixed Hessian: spectral entropy different from uniform",
        (se_mixed - se_flat).abs() > 0.01,
    );

    h.check_bool(
        "Mixed Hessian: all eigenvalues positive",
        mixed_evals.iter().all(|&v| v > 0.0),
    );

    // ── 7. Numerical Hessian via finite differences ─────────────────
    //
    // f(x) = 0.5 * (x[0]^2 + 4*x[1]^2) → H = [[1, 0], [0, 4]]

    let eps = tolerances::HESSIAN_FD_STEP;
    let dim = 2;
    let x0 = [0.0f64; 2];

    let f = |x: &[f64]| 0.5 * (x[0] * x[0] + 4.0 * x[1] * x[1]);

    let mut hess_fd = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let mut xpp = x0.to_vec();
            let mut xpm = x0.to_vec();
            let mut xmp = x0.to_vec();
            let mut xmm = x0.to_vec();

            xpp[i] += eps;
            xpp[j] += eps;
            xpm[i] += eps;
            xpm[j] -= eps;
            xmp[i] -= eps;
            xmp[j] += eps;
            xmm[i] -= eps;
            xmm[j] -= eps;

            let val = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * eps * eps);
            hess_fd[i * dim + j] = val;
            hess_fd[j * dim + i] = val;
        }
    }

    h.check_abs(
        "FD Hessian[0,0] = 1.0",
        hess_fd[0],
        1.0,
        tolerances::HESSIAN_FD_STEP,
    );
    h.check_abs(
        "FD Hessian[1,1] = 4.0",
        hess_fd[3],
        4.0,
        tolerances::HESSIAN_FD_STEP,
    );
    h.check_abs(
        "FD Hessian[0,1] = 0.0",
        hess_fd[1],
        0.0,
        tolerances::HESSIAN_FD_STEP,
    );

    // ── 8. Determinism ──────────────────────────────────────────────

    let decomp_a = eigh_householder_qr(&hessian_exact, n);
    let decomp_b = eigh_householder_qr(&hessian_exact, n);
    let eval_match = decomp_a
        .eigenvalues
        .iter()
        .zip(decomp_b.eigenvalues.iter())
        .all(|(a, b)| (a - b).abs() < tolerances::EXACT_F64);
    h.check_bool("eigh deterministic", eval_match);

    h.finish();
}
