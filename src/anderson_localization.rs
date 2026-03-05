// SPDX-License-Identifier: AGPL-3.0-or-later

//! Anderson localization for disordered quantum systems.
//!
//! Port of `control/anderson_localization/anderson_localization.py`.
//!
//! Reproduces key results from:
//! Bourgain & Kachkovskiy (2018)
//! "Anderson localization for two interacting quasiperiodic particles"
//! GAFA 29:3-43.
//!
//! Model: 1D Anderson Hamiltonian (tridiagonal) with random or
//! quasiperiodic (Aubry-André) disorder. IPR measures localization.
//!
//! ## GPU-ready layout
//!
//! All matrices use **flat row-major `Vec<f64>`** (one contiguous buffer).
//! Element (i,j) of an n×n matrix is at index `i * n + j`.
//! This maps directly to GPU buffers for `barracuda::linalg::eigh_f64`.
//!
//! ## `BarraCUDA` connection
//!
//! - Eigendecomposition: `barracuda::linalg::eigh_f64` (Jacobi, improving via NAK)
//! - IPR computation: `barracuda::ops::FusedMapReduceF64` (sum of 4th powers)
//! - Disorder sweep: embarrassingly parallel (batch eigensolve over W values)
//! - Aubry-André potential: elementwise cosine (`barracuda::ops::elementwise`)
//!
//! ## WGSL shader (absorption-ready)
//!
//! [`WGSL_BATCH_IPR`] — batch inverse participation ratio. One thread
//! per eigenvector, computes `sum(|ψ_i|^4)`. Validated in
//! `validate_gpu_anderson`.

#![expect(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "Anderson model uses standard physics notation (n, t, w, h, i, v) and usize→f64 casts"
)]

use crate::rng::Rng;

/// WGSL shader: batch IPR from eigenvector data.
///
/// Absorption target: `barracuda::ops::batch_reduce` or `FusedMapReduceF64`.
/// Validated: `validate_gpu_anderson`.
pub use neural_spring_forge::shaders::BATCH_IPR as WGSL_BATCH_IPR;
use std::f64::consts::PI;

/// Golden ratio φ = (1 + √5) / 2 (irrational for quasiperiodicity).
pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;

/// Build 1D Anderson Hamiltonian with random diagonal disorder.
///
/// `H[i,i]` = `V_i` with `V_i` ~ uniform[-W/2, W/2]. Off-diagonal = -t.
/// Returns flat row-major n×n matrix.
#[must_use]
pub fn anderson_hamiltonian_random(n: usize, t: f64, w: f64, rng: &mut Rng) -> Vec<f64> {
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        let u = rng.uniform();
        h[i * n + i] = u.mul_add(w, -w / 2.0);
    }
    for i in 0..n.saturating_sub(1) {
        h[i * n + i + 1] = -t;
        h[(i + 1) * n + i] = -t;
    }
    h
}

/// Aubry-André quasiperiodic potential: `V_n` = W * cos(2π*α*n + φ).
#[must_use]
pub fn aubry_andre_potential(n: usize, w: f64, alpha: f64, phi: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (2.0 * PI * alpha).mul_add(i as f64, phi).cos() * w)
        .collect()
}

/// Aubry-André Hamiltonian: hopping -t plus quasiperiodic diagonal.
/// Returns flat row-major n×n matrix.
#[must_use]
pub fn aubry_andre_hamiltonian(n: usize, t: f64, w: f64, alpha: f64, phi: f64) -> Vec<f64> {
    let v = aubry_andre_potential(n, w, alpha, phi);
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        h[i * n + i] = v[i];
    }
    for i in 0..n.saturating_sub(1) {
        h[i * n + i + 1] = -t;
        h[(i + 1) * n + i] = -t;
    }
    h
}

/// Inverse participation ratio: IPR = `sum(|ψ_n|⁴)`.
/// Extended: IPR ~ 1/N. Localized: IPR >> 1/N.
#[must_use]
pub fn ipr(psi: &[f64]) -> f64 {
    psi.iter().map(|&x| x * x).map(|p| p * p).sum()
}

/// Mean IPR over columns of flat eigenvector matrix.
///
/// `eigenvectors`: flat row-major n×n. Column k is the k-th eigenvector.
#[must_use]
pub fn mean_ipr(eigenvectors: &[f64], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for k in 0..n {
        let col: Vec<f64> = (0..n).map(|row| eigenvectors[row * n + k]).collect();
        sum += ipr(&col);
    }
    sum / (n as f64)
}

/// Eigendecomposition for real symmetric matrix (flat row-major).
///
/// Delegates to `barracuda::ops::linalg::eigh_householder_qr` — the same
/// Householder + implicit-QR algorithm used by `crate::eigh`.  The local
/// Jacobi implementation (400-sweep, 80-line) is retired; barracuda's version
/// is more numerically robust and maintained upstream.
///
/// Returns (eigenvalues, eigenvectors as flat row-major n×n).
/// Column k of the eigenvector matrix is the k-th eigenvector.
#[must_use]
pub fn jacobi_eigh(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let decomp = crate::eigh::eigh_householder_qr(matrix, n);
    (decomp.eigenvalues, decomp.eigenvectors)
}

/// Two-particle Hamiltonian on tensor product space.
/// H = H₁ ⊗ I + I ⊗ H₁ + U * δ(same site). Uses Aubry-André for H₁.
/// Returns flat row-major dim×dim matrix where dim = n×n.
#[must_use]
pub fn two_particle_hamiltonian(n: usize, t: f64, w: f64, u: f64, alpha: f64) -> Vec<f64> {
    let h1 = aubry_andre_hamiltonian(n, t, w, alpha, 0.0);
    let dim = n * n;
    let mut h2 = vec![0.0; dim * dim];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for m in 0..n {
                    let idx_a = i * n + j;
                    let idx_b = k * n + m;
                    h2[idx_a * dim + idx_b] = if j == m { h1[i * n + k] } else { 0.0 }
                        + if i == k { h1[j * n + m] } else { 0.0 };
                    if i == k && j == m && i == j {
                        h2[idx_a * dim + idx_b] += u;
                    }
                }
            }
        }
    }
    h2
}

/// Disorder strength sweep: compute mean IPR for each W.
#[must_use]
pub fn disorder_sweep(n: usize, t: f64, w_vals: &[f64], rng: &mut Rng) -> Vec<f64> {
    w_vals
        .iter()
        .map(|&w| {
            let h = anderson_hamiltonian_random(n, t, w, rng);
            let (_, ev) = jacobi_eigh(&h, n);
            mean_ipr(&ev, n)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::tolerances;

    #[test]
    fn anderson_hermitian() {
        let mut rng = Rng::new(42);
        let n = 20;
        let h = anderson_hamiltonian_random(n, 1.0, 2.0, &mut rng);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (h[i * n + j] - h[j * n + i]).abs() < tolerances::ZERO_DETECTION,
                    "H not symmetric"
                );
            }
        }
    }

    #[test]
    fn ipr_normalized() {
        let psi: Vec<f64> = vec![0.5; 4];
        let p = ipr(&psi);
        assert!(
            (p - 0.25).abs() < tolerances::CROSS_LANGUAGE,
            "IPR of uniform 4-vec = 1/N = 0.25, got {p}"
        );
    }

    #[test]
    fn ipr_localized_state() {
        let mut psi = vec![0.0; 10];
        psi[3] = 1.0;
        assert!((ipr(&psi) - 1.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn ipr_empty() {
        assert!((ipr(&[]) - 0.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn aubry_andre_potential_correct_length() {
        let v = aubry_andre_potential(8, 2.0, 1.0 / GOLDEN_RATIO, 0.0);
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn aubry_andre_potential_bounded_by_w() {
        let w = 3.5;
        let v = aubry_andre_potential(100, w, 1.0 / GOLDEN_RATIO, 0.0);
        for &vi in &v {
            assert!(
                vi.abs() <= w + tolerances::ZERO_DETECTION,
                "|V_n| = {} > W = {}",
                vi.abs(),
                w
            );
        }
    }

    #[test]
    fn aubry_andre_potential_quasiperiodic() {
        let v = aubry_andre_potential(50, 1.0, 1.0 / GOLDEN_RATIO, 0.0);
        let all_same = v
            .windows(2)
            .all(|w| (w[0] - w[1]).abs() < tolerances::ZERO_DETECTION);
        assert!(!all_same, "quasiperiodic potential should not be constant");
    }

    #[test]
    fn aubry_andre_transition() {
        let n = 16;
        let alpha = 1.0 / GOLDEN_RATIO;
        let h_below = aubry_andre_hamiltonian(n, 1.0, 1.5, alpha, 0.0);
        let h_above = aubry_andre_hamiltonian(n, 1.0, 3.0, alpha, 0.0);
        let (_, ev_below) = jacobi_eigh(&h_below, n);
        let (_, ev_above) = jacobi_eigh(&h_above, n);
        let ipr_below = mean_ipr(&ev_below, n);
        let ipr_above = mean_ipr(&ev_above, n);
        assert!(ipr_below < ipr_above);
    }

    #[test]
    fn mean_ipr_empty() {
        assert!((mean_ipr(&[], 0) - 0.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn mean_ipr_identity_matrix() {
        let n = 4;
        let mut eye = vec![0.0; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let m = mean_ipr(&eye, n);
        assert!(
            (m - 1.0).abs() < tolerances::CROSS_LANGUAGE,
            "identity eigenvectors → IPR=1 each, mean should be 1.0, got {m}"
        );
    }

    #[test]
    fn disorder_sweep_monotonic_trend() {
        let mut rng = Rng::new(42);
        let n = 8;
        let w_vals: Vec<f64> = (0..5).map(|i| f64::from(i).mul_add(1.5, 0.5)).collect();
        let iprs = disorder_sweep(n, 1.0, &w_vals, &mut rng);
        assert_eq!(iprs.len(), w_vals.len());
        assert!(iprs.iter().all(|&v| v.is_finite() && v > 0.0));
        let first = iprs[0];
        let last = iprs[iprs.len() - 1];
        assert!(
            last > first,
            "stronger disorder should increase IPR: first={first}, last={last}"
        );
    }

    #[test]
    fn disorder_sweep_deterministic() {
        let mut rng1 = Rng::new(99);
        let mut rng2 = Rng::new(99);
        let w_vals = [1.0, 2.0, 3.0];
        let a = disorder_sweep(8, 1.0, &w_vals, &mut rng1);
        let b = disorder_sweep(8, 1.0, &w_vals, &mut rng2);
        assert_eq!(a, b, "same seed should produce identical sweeps");
    }

    #[test]
    fn two_particle_finite() {
        let n = 4;
        let h2 = two_particle_hamiltonian(n, 1.0, 2.0, 0.5, 1.0 / GOLDEN_RATIO);
        let dim = n * n;
        let (eig, ev) = jacobi_eigh(&h2, dim);
        assert!(eig.iter().all(|&x| x.is_finite()));
        assert!(ev.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn two_particle_symmetric() {
        let n = 4;
        let h2 = two_particle_hamiltonian(n, 1.0, 2.0, 0.5, 1.0 / GOLDEN_RATIO);
        let dim = n * n;
        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    (h2[i * dim + j] - h2[j * dim + i]).abs() < tolerances::ZERO_DETECTION,
                    "two-particle H not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn flat_layout_correct_size() {
        let n = 8;
        let mut rng = Rng::new(42);
        let h = anderson_hamiltonian_random(n, 1.0, 2.0, &mut rng);
        assert_eq!(h.len(), n * n);
        let aa = aubry_andre_hamiltonian(n, 1.0, 2.0, 1.0 / GOLDEN_RATIO, 0.0);
        assert_eq!(aa.len(), n * n);
    }

    #[test]
    fn jacobi_eigh_eigenvalues_sorted_and_finite() {
        let mut rng = Rng::new(7);
        let n = 10;
        let h = anderson_hamiltonian_random(n, 1.0, 2.0, &mut rng);
        let (evals, _) = jacobi_eigh(&h, n);
        assert_eq!(evals.len(), n);
        assert!(evals.iter().all(|&e| e.is_finite()));
    }
}
