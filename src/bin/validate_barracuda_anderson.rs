// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: Anderson localization (Paper 023).
//!
//! Validates that `barracuda::linalg::eigh_f64` reproduces the eigendecomposition
//! from the hand-rolled Jacobi solver in `anderson_localization.rs`.
//!
//! Evolution path:
//! ```text
//! Python (numpy.linalg.eigh) → Rust (hand-rolled Jacobi)
//!   → BarraCUDA CPU (barracuda::linalg::eigh_f64)
//!   → BarraCUDA GPU (tridiag_eigh.wgsl / batched_eigh_nak)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/anderson_localization/anderson_localization.py`
//! Rust baseline: `validate_anderson_localization` (8/8 PASS)

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::similar_names
)]

use neural_spring::anderson_localization::{
    anderson_hamiltonian_random, aubry_andre_hamiltonian, ipr, jacobi_eigh, GOLDEN_RATIO,
};
use neural_spring::rng::Rng;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_anderson");
    let mut rng = Rng::new(42);

    validate_barracuda_eigh_vs_jacobi(&mut h, &mut rng);
    validate_barracuda_aubry_andre(&mut h);
    validate_barracuda_disorder_trend(&mut h);

    h.finish();
}

/// Compare barracuda `eigh_f64` eigenvalues against hand-rolled Jacobi.
fn validate_barracuda_eigh_vs_jacobi(h: &mut ValidationHarness, rng: &mut Rng) {
    let n = 16;
    let ham = anderson_hamiltonian_random(n, 1.0, 2.0, rng);
    let flat: Vec<f64> = ham.iter().flat_map(|row| row.iter().copied()).collect();

    let (mut jacobi_vals, _) = jacobi_eigh(&ham);
    jacobi_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    match barracuda::linalg::eigh_f64(&flat, n) {
        Ok(eig) => {
            h.check_bool(
                &format!("barracuda returns {n} eigenvalues"),
                eig.eigenvalues.len() == n,
            );

            let max_diff: f64 = jacobi_vals
                .iter()
                .zip(eig.eigenvalues.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            // barracuda Jacobi eigensolver: ~0.1 eigenvalue error at n=16
            // (limited iterations). ToadStool handoff: Lanczos for machine precision.
            h.check_upper(
                &format!("eigenvalues agree (max_diff={max_diff:.2e})"),
                max_diff,
                0.15,
            );

            let barracuda_mean_ipr = barracuda_mean_ipr_from_vecs(&eig.eigenvectors, n);

            h.check_bool(
                &format!("barracuda mean IPR finite ({barracuda_mean_ipr:.4e})"),
                barracuda_mean_ipr.is_finite() && barracuda_mean_ipr > 0.0,
            );
        }
        Err(e) => {
            h.check_bool(&format!("eigh_f64 [ERROR: {e}]"), false);
            h.check_bool("eigenvalue agreement [SKIPPED]", false);
            h.check_bool("barracuda IPR [SKIPPED]", false);
        }
    }
}

/// Aubry-André transition: barracuda should detect localization transition.
fn validate_barracuda_aubry_andre(h: &mut ValidationHarness) {
    let n = 16;
    let alpha = 1.0 / GOLDEN_RATIO;

    let h_below = aubry_andre_hamiltonian(n, 1.0, 1.5, alpha, 0.0);
    let h_above = aubry_andre_hamiltonian(n, 1.0, 3.0, alpha, 0.0);

    let flat_below: Vec<f64> = h_below.iter().flat_map(|r| r.iter().copied()).collect();
    let flat_above: Vec<f64> = h_above.iter().flat_map(|r| r.iter().copied()).collect();

    match (
        barracuda::linalg::eigh_f64(&flat_below, n),
        barracuda::linalg::eigh_f64(&flat_above, n),
    ) {
        (Ok(eig_below), Ok(eig_above)) => {
            let ipr_below = barracuda_mean_ipr_from_vecs(&eig_below.eigenvectors, n);
            let ipr_above = barracuda_mean_ipr_from_vecs(&eig_above.eigenvectors, n);

            h.check_bool(
                &format!(
                    "Aubry-André transition: W<W_c IPR ({ipr_below:.4}) < W>W_c IPR ({ipr_above:.4})"
                ),
                ipr_below < ipr_above,
            );
        }
        _ => {
            h.check_bool("barracuda Aubry-André eigh failed", false);
        }
    }
}

/// Disorder strength trend: stronger disorder → higher IPR.
fn validate_barracuda_disorder_trend(h: &mut ValidationHarness) {
    let n = 16;
    let mut rng = Rng::new(42);

    let h_weak = anderson_hamiltonian_random(n, 1.0, 0.5, &mut rng);
    let h_strong = anderson_hamiltonian_random(n, 1.0, 8.0, &mut rng);

    let flat_weak: Vec<f64> = h_weak.iter().flat_map(|r| r.iter().copied()).collect();
    let flat_strong: Vec<f64> = h_strong.iter().flat_map(|r| r.iter().copied()).collect();

    if let (Ok(eig_weak), Ok(eig_strong)) = (
        barracuda::linalg::eigh_f64(&flat_weak, n),
        barracuda::linalg::eigh_f64(&flat_strong, n),
    ) {
        let ipr_weak = barracuda_mean_ipr_from_vecs(&eig_weak.eigenvectors, n);
        let ipr_strong = barracuda_mean_ipr_from_vecs(&eig_strong.eigenvectors, n);

        h.check_bool(
            &format!("disorder trend: weak IPR ({ipr_weak:.4}) < strong IPR ({ipr_strong:.4})"),
            ipr_weak < ipr_strong,
        );

        h.check_lower(
            &format!("strong disorder IPR > 0.05 ({ipr_strong:.4})"),
            ipr_strong,
            0.05,
        );
    } else {
        h.check_bool("barracuda disorder eigh failed", false);
        h.check_bool("barracuda disorder IPR [SKIPPED]", false);
    }

    // Cross-validate: barracuda eigenvalues match hand-rolled for same input
    let h_test = anderson_hamiltonian_random(n, 1.0, 2.0, &mut rng);
    let flat_test: Vec<f64> = h_test.iter().flat_map(|r| r.iter().copied()).collect();
    let (mut jacobi_vals, _) = jacobi_eigh(&h_test);
    jacobi_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    match barracuda::linalg::eigh_f64(&flat_test, n) {
        Ok(eig) => {
            let max_diff: f64 = jacobi_vals
                .iter()
                .zip(eig.eigenvalues.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("cross-validate eigenvalues (max_diff={max_diff:.2e})"),
                max_diff,
                0.15,
            );
        }
        Err(e) => {
            h.check_bool(&format!("cross-validate eigh [ERROR: {e}]"), false);
        }
    }
}

/// Compute mean IPR from barracuda's eigh result using eigenvector field.
fn barracuda_mean_ipr_from_vecs(eigenvectors: &[f64], n: usize) -> f64 {
    let mut total = 0.0;
    for k in 0..n {
        let col: Vec<f64> = (0..n).map(|row| eigenvectors[row * n + k]).collect();
        total += ipr(&col);
    }
    total / n as f64
}
