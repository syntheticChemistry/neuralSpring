// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spectral commutativity and distance to normal.
//!
//! Port of `control/spectral_commutativity/spectral_commutativity.py`.
//!
//! Paper: Kachkovskiy & Safarov (2016)
//! "Distance to normal elements in C*-algebras of real rank zero"
//! JAMS 29:61–80.
//!
//! Core thesis: Quantifies how close an operator is to being normal.
//! For neural networks: skip connections and residual layers relate to
//! approximate commutativity.
//!
//! ## Layout
//!
//! All matrices use **flat row-major** `Vec<f64>` storage with explicit
//! dimension `n`. This is cache-friendly and directly uploadable to GPU.
//!
//! ## `BarraCUDA` connection
//!
//! - Matrix multiplication A×B: `barracuda::ops::matmul` (GEMM f64)
//! - Commutator \[A,B\] = AB − BA: two GEMM + elementwise subtract
//! - Frobenius norm: `barracuda::ops::NormReduceF64`
//! - Distance to normal: composed from commutator + Frobenius (GPU pipeline)

#![expect(
    clippy::cast_precision_loss,
    reason = "matrix dimension → f64 for norm computation"
)]

use crate::rng::Rng;

/// Frobenius norm: sqrt(sum of squares of entries).
///
/// CPU reference for GPU validation — `barracuda::dispatch::frobenius_norm_dispatch`
/// is the production equivalent.  Kept separate so GPU validators have an
/// independent reference to compare against.
#[must_use]
pub fn frobenius_norm(a: &[f64]) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Transpose an n×n matrix (flat row-major → flat row-major).
///
/// CPU reference for GPU validation — `barracuda::dispatch::transpose_dispatch`
/// is the production equivalent.
#[must_use]
pub fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

/// Matrix multiplication C = A × B for n×n square matrices (flat row-major).
///
/// CPU reference implementation for validation. The GPU equivalent is
/// `barracuda::ops::matmul` / `gemm_f64.wgsl`. This stays CPU-side so
/// GPU validation binaries have an independent reference to compare against.
#[must_use]
pub fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    c
}

/// Commutator \[A, B\] = AB - BA (flat row-major).
#[must_use]
pub fn commutator(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let ab = mat_mul(a, b, n);
    let ba = mat_mul(b, a, n);
    ab.iter().zip(ba.iter()).map(|(x, y)| x - y).collect()
}

/// Distance to normal: `||A*A - AA*||_F / (2||A||_F)`.
/// For real matrices, `A*` = Aᵀ. A is normal iff `A*A = AA*`.
#[must_use]
pub fn distance_to_normal(a: &[f64], n: usize) -> f64 {
    let norm = frobenius_norm(a);
    if norm < crate::primitives::LOG_GUARD {
        return 0.0;
    }
    let at = transpose(a, n);
    let ata = mat_mul(&at, a, n);
    let aat = mat_mul(a, &at, n);
    let diff: Vec<f64> = ata.iter().zip(aat.iter()).map(|(x, y)| x - y).collect();
    frobenius_norm(&diff) / (2.0 * norm)
}

/// Commutativity ratio: `||[A,B]||_F / (||A||_F * ||B||_F)`. Scale-invariant.
#[must_use]
pub fn commutativity_ratio(a: &[f64], b: &[f64], n: usize) -> f64 {
    let na = frobenius_norm(a);
    let nb = frobenius_norm(b);
    if na * nb < crate::primitives::LOG_GUARD {
        return 0.0;
    }
    let comm = commutator(a, b, n);
    frobenius_norm(&comm) / (na * nb)
}

/// Skip connection analysis: `(raw_ratio, skip_ratio)` for W1,W2 vs (I+W1),(I+W2).
#[must_use]
pub fn skip_commutativity(w1: &[f64], w2: &[f64], n: usize) -> (f64, f64) {
    let ident = identity_matrix(n);
    let i_plus_w1: Vec<f64> = ident.iter().zip(w1.iter()).map(|(a, b)| a + b).collect();
    let i_plus_w2: Vec<f64> = ident.iter().zip(w2.iter()).map(|(a, b)| a + b).collect();
    let raw = commutativity_ratio(w1, w2, n);
    let skip = commutativity_ratio(&i_plus_w1, &i_plus_w2, n);
    (raw, skip)
}

/// Identity matrix of size n (flat row-major).
#[must_use]
pub fn identity_matrix(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Generate random matrix (n × n) with standard normal entries / sqrt(n).
#[must_use]
pub fn random_matrix(n: usize, rng: &mut Rng) -> Vec<f64> {
    let scale = 1.0 / (n as f64).sqrt();
    (0..n * n)
        .map(|_| rng.normal_params(0.0, 1.0) * scale)
        .collect()
}

/// Symmetric matrix from random H: (H + Hᵀ) / 2. Normal (commutes with adjoint).
#[must_use]
pub fn random_symmetric(n: usize, rng: &mut Rng) -> Vec<f64> {
    let h = random_matrix(n, rng);
    let ht = transpose(&h, n);
    h.iter()
        .zip(ht.iter())
        .map(|(a, b)| (a + b) * 0.5)
        .collect()
}

/// Spectral gap: max|eig(`AᵀA`) - eig(`AAᵀ`)|. Zero for normal matrices.
///
/// Uses Frobenius norm of `AᵀA` - `AAᵀ` as proxy (zero for normal).
#[must_use]
pub fn spectral_gap_approx(a: &[f64], n: usize) -> f64 {
    let at = transpose(a, n);
    let ata = mat_mul(&at, a, n);
    let aat = mat_mul(a, &at, n);
    let diff: Vec<f64> = ata.iter().zip(aat.iter()).map(|(x, y)| x - y).collect();
    frobenius_norm(&diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn commutator_antisymmetric() {
        let mut rng = Rng::new(42);
        let a = random_matrix(8, &mut rng);
        let b = random_matrix(8, &mut rng);
        let ab = commutator(&a, &b, 8);
        let ba = commutator(&b, &a, 8);
        let sum: Vec<f64> = ab.iter().zip(ba.iter()).map(|(x, y)| x + y).collect();
        assert!(
            frobenius_norm(&sum) < tolerances::CROSS_LANGUAGE,
            " [A,B] = -[B,A]"
        );
    }

    #[test]
    fn distance_to_normal_symmetric_zero() {
        let mut rng = Rng::new(42);
        let sym = random_symmetric(16, &mut rng);
        let d = distance_to_normal(&sym, 16);
        assert!(
            d < tolerances::CROSS_LANGUAGE,
            "symmetric (normal) should have d≈0, got {d}"
        );
    }

    #[test]
    fn skip_reduces_commutativity() {
        let mut rng = Rng::new(42);
        let w1 = random_matrix(16, &mut rng);
        let w2 = random_matrix(16, &mut rng);
        let (raw, skip) = skip_commutativity(&w1, &w2, 16);
        assert!(skip < raw, "skip ({skip}) < raw ({raw})");
    }

    #[test]
    fn mat_mul_identity() {
        let n = 4;
        let ident = identity_matrix(n);
        let mut rng = Rng::new(42);
        let a = random_matrix(n, &mut rng);
        let result = mat_mul(&a, &ident, n);
        let diff = frobenius_norm(
            &a.iter()
                .zip(result.iter())
                .map(|(x, y)| x - y)
                .collect::<Vec<_>>(),
        );
        assert!(
            diff < tolerances::ZERO_DETECTION,
            "A*I should equal A, diff={diff}"
        );
    }

    #[test]
    fn transpose_involutory() {
        let n = 5;
        let mut rng = Rng::new(42);
        let a = random_matrix(n, &mut rng);
        let att = transpose(&transpose(&a, n), n);
        let diff = frobenius_norm(
            &a.iter()
                .zip(att.iter())
                .map(|(x, y)| x - y)
                .collect::<Vec<_>>(),
        );
        assert!(
            diff < tolerances::ZERO_DETECTION,
            "transpose(transpose(A)) should equal A"
        );
    }

    #[test]
    fn spectral_gap_approx_symmetric_zero() {
        let mut rng = Rng::new(42);
        let sym = random_symmetric(8, &mut rng);
        let gap = spectral_gap_approx(&sym, 8);
        assert!(
            gap < tolerances::CROSS_LANGUAGE,
            "symmetric matrix: AᵀA = AAᵀ, gap ≈ 0, got {gap}"
        );
    }

    #[test]
    fn spectral_gap_approx_identity() {
        let n = 4;
        let eye = identity_matrix(n);
        let gap = spectral_gap_approx(&eye, n);
        assert!(gap < tolerances::ZERO_DETECTION, "identity gap = 0");
    }

    #[test]
    fn commutativity_ratio_identity() {
        let n = 4;
        let eye = identity_matrix(n);
        let mut rng = Rng::new(42);
        let a = random_matrix(n, &mut rng);
        let ratio = commutativity_ratio(&eye, &a, n);
        assert!(
            ratio < tolerances::CROSS_LANGUAGE,
            "I commutes with everything, ratio ≈ 0"
        );
    }
}
