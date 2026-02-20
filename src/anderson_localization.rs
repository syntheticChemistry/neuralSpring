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

#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::must_use_candidate,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::suboptimal_flops
)]

use crate::rng::Rng;
use std::f64::consts::PI;

/// Golden ratio φ = (1 + √5) / 2 (irrational for quasiperiodicity).
pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;

/// Build 1D Anderson Hamiltonian with random diagonal disorder.
///
/// `H\[i,i\]` = V_i with V_i ~ uniform\[-W/2, W/2\]. Off-diagonal = -t.
/// Returns full N×N symmetric matrix.
pub fn anderson_hamiltonian_random(n: usize, t: f64, w: f64, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut h = vec![vec![0.0; n]; n];
    for i in 0..n {
        let u = rng.uniform();
        h[i][i] = u.mul_add(w, -w / 2.0);
    }
    for i in 0..n.saturating_sub(1) {
        h[i][i + 1] = -t;
        h[i + 1][i] = -t;
    }
    h
}

/// Aubry-André quasiperiodic potential: V_n = W * cos(2π*α*n + φ).
pub fn aubry_andre_potential(n: usize, w: f64, alpha: f64, phi: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (2.0 * PI * alpha).mul_add(i as f64, phi).cos() * w)
        .collect()
}

/// Aubry-André Hamiltonian: hopping -t plus quasiperiodic diagonal.
pub fn aubry_andre_hamiltonian(n: usize, t: f64, w: f64, alpha: f64, phi: f64) -> Vec<Vec<f64>> {
    let v = aubry_andre_potential(n, w, alpha, phi);
    let mut h = vec![vec![0.0; n]; n];
    for i in 0..n {
        h[i][i] = v[i];
    }
    for i in 0..n.saturating_sub(1) {
        h[i][i + 1] = -t;
        h[i + 1][i] = -t;
    }
    h
}

/// Inverse participation ratio: IPR = sum(|ψ_n|⁴).
/// Extended: IPR ~ 1/N. Localized: IPR >> 1/N.
pub fn ipr(psi: &[f64]) -> f64 {
    psi.iter().map(|&x| x * x).map(|p| p * p).sum()
}

/// Mean IPR over columns of eigenvector matrix.
pub fn mean_ipr(eigenvectors: &[Vec<f64>]) -> f64 {
    if eigenvectors.is_empty() {
        return 0.0;
    }
    let n_vecs = eigenvectors[0].len();
    let mut sum = 0.0;
    for k in 0..n_vecs {
        let col: Vec<f64> = eigenvectors.iter().map(|row| row[k]).collect();
        sum += ipr(&col);
    }
    sum / (n_vecs as f64)
}

/// Jacobi eigensolver for real symmetric matrix.
/// Returns (eigenvalues, eigenvectors as columns).
/// Eigenvectors normalized to unit L2 norm.
pub fn jacobi_eigh(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    const MAX_SWEEPS: usize = 400;
    const TOL: f64 = 1e-12;

    for _ in 0..MAX_SWEEPS {
        let mut max_off = 0.0f64;
        let mut p = 0;
        let mut q = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let off = a[i][j].abs();
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

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            -tau + (tau * tau + 1.0).sqrt()
        } else {
            -tau - (tau * tau + 1.0).sqrt()
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for i in 0..n {
            let aip = a[i][p];
            let aiq = a[i][q];
            a[i][p] = c * aip - s * aiq;
            a[i][q] = s * aip + c * aiq;
        }
        for i in 0..n {
            let api = a[p][i];
            let aqi = a[q][i];
            a[p][i] = c * api - s * aqi;
            a[q][i] = s * api + c * aqi;
        }
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }

    let eigvals: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    for k in 0..n {
        let norm: f64 = v.iter().map(|row| row[k] * row[k]).sum::<f64>().sqrt();
        if norm > 1e-300 {
            for row in &mut v {
                row[k] /= norm;
            }
        }
    }
    (eigvals, v)
}

/// Two-particle Hamiltonian on tensor product space.
/// H = H₁ ⊗ I + I ⊗ H₁ + U * δ(same site). Uses Aubry-André for H₁.
pub fn two_particle_hamiltonian(n: usize, t: f64, w: f64, u: f64, alpha: f64) -> Vec<Vec<f64>> {
    let h1 = aubry_andre_hamiltonian(n, t, w, alpha, 0.0);
    let dim = n * n;
    let mut h2 = vec![vec![0.0; dim]; dim];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for m in 0..n {
                    let idx_a = i * n + j;
                    let idx_b = k * n + m;
                    h2[idx_a][idx_b] =
                        if j == m { h1[i][k] } else { 0.0 } + if i == k { h1[j][m] } else { 0.0 };
                    if i == k && j == m && i == j {
                        h2[idx_a][idx_b] += u;
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
            let (_, ev) = jacobi_eigh(&h);
            mean_ipr(&ev)
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
        let h = anderson_hamiltonian_random(20, 1.0, 2.0, &mut rng);
        for i in 0..20 {
            for j in 0..20 {
                assert!((h[i][j] - h[j][i]).abs() < 1e-14, "H not symmetric");
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
        let alpha = 1.0 / GOLDEN_RATIO;
        let h_below = aubry_andre_hamiltonian(16, 1.0, 1.5, alpha, 0.0);
        let h_above = aubry_andre_hamiltonian(16, 1.0, 3.0, alpha, 0.0);
        let (_, ev_below) = jacobi_eigh(&h_below);
        let (_, ev_above) = jacobi_eigh(&h_above);
        let ipr_below = mean_ipr(&ev_below);
        let ipr_above = mean_ipr(&ev_above);
        assert!(ipr_below < ipr_above);
    }

    #[test]
    fn two_particle_finite() {
        let h2 = two_particle_hamiltonian(4, 1.0, 2.0, 0.5, 1.0 / GOLDEN_RATIO);
        let (eig, ev) = jacobi_eigh(&h2);
        assert!(eig.iter().all(|&x| x.is_finite()));
        assert!(ev.iter().all(|row| row.iter().all(|&x| x.is_finite())));
    }
}
