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

#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::suboptimal_flops
)]

use crate::rng::Rng;

/// WGSL shader: batch IPR from eigenvector data.
///
/// Absorption target: `barracuda::ops::batch_reduce` or `FusedMapReduceF64`.
/// Validated: `validate_gpu_anderson`.
pub const WGSL_BATCH_IPR: &str = include_str!("../metalForge/shaders/batch_ipr.wgsl");
use std::f64::consts::PI;

/// Golden ratio φ = (1 + √5) / 2 (irrational for quasiperiodicity).
pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;

/// Build 1D Anderson Hamiltonian with random diagonal disorder.
///
/// `H[i,i]` = V_i with V_i ~ uniform[-W/2, W/2]. Off-diagonal = -t.
/// Returns flat row-major n×n matrix.
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

/// Aubry-André quasiperiodic potential: V_n = W * cos(2π*α*n + φ).
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

/// Inverse participation ratio: IPR = sum(|ψ_n|⁴).
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

/// Jacobi eigensolver for real symmetric matrix (flat row-major).
///
/// Returns (eigenvalues, eigenvectors as flat row-major n×n).
/// Column k of the eigenvector matrix is the k-th eigenvector.
/// Eigenvectors normalized to unit L2 norm.
#[must_use]
pub fn jacobi_eigh(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    const MAX_SWEEPS: usize = 400;
    const TOL: f64 = 1e-12;

    for _ in 0..MAX_SWEEPS {
        let mut max_off = 0.0f64;
        let mut p = 0;
        let mut q = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let off = a[i * n + j].abs();
                if off > max_off {
                    max_off = off;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < TOL {
            break;
        }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            -tau + (tau * tau + 1.0).sqrt()
        } else {
            -tau - (tau * tau + 1.0).sqrt()
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for i in 0..n {
            let aip = a[i * n + p];
            let aiq = a[i * n + q];
            a[i * n + p] = c * aip - s * aiq;
            a[i * n + q] = s * aip + c * aiq;
        }
        for i in 0..n {
            let api = a[p * n + i];
            let aqi = a[q * n + i];
            a[p * n + i] = c * api - s * aqi;
            a[q * n + i] = s * api + c * aqi;
        }
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip - s * viq;
            v[i * n + q] = s * vip + c * viq;
        }
    }

    let eigvals: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    for k in 0..n {
        let norm: f64 = (0..n)
            .map(|i| v[i * n + k] * v[i * n + k])
            .sum::<f64>()
            .sqrt();
        if norm > 1e-300 {
            for i in 0..n {
                v[i * n + k] /= norm;
            }
        }
    }
    (eigvals, v)
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

    #[test]
    fn anderson_hermitian() {
        let mut rng = Rng::new(42);
        let n = 20;
        let h = anderson_hamiltonian_random(n, 1.0, 2.0, &mut rng);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (h[i * n + j] - h[j * n + i]).abs() < 1e-14,
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
            (p - 0.25).abs() < 1e-10,
            "IPR of uniform 4-vec = 1/N = 0.25, got {p}"
        );
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
    fn two_particle_finite() {
        let n = 4;
        let h2 = two_particle_hamiltonian(n, 1.0, 2.0, 0.5, 1.0 / GOLDEN_RATIO);
        let dim = n * n;
        let (eig, ev) = jacobi_eigh(&h2, dim);
        assert!(eig.iter().all(|&x| x.is_finite()));
        assert!(ev.iter().all(|&x| x.is_finite()));
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
}
