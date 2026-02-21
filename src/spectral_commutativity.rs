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
//! ## `BarraCUDA` connection
//!
//! - Matrix multiplication A×B: `barracuda::ops::matmul` (GEMM f64)
//! - Commutator [A,B] = AB − BA: two GEMM + elementwise subtract
//! - Frobenius norm: `barracuda::ops::NormReduceF64`
//! - Distance to normal: composed from commutator + Frobenius (GPU pipeline)

#![allow(clippy::cast_precision_loss)]

use crate::rng::Rng;

/// Frobenius norm: sqrt(sum of squares of entries).
#[must_use]
pub fn frobenius_norm(a: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0;
    for row in a {
        for &x in row {
            sum = x.mul_add(x, sum);
        }
    }
    sum.sqrt()
}

/// Matrix transpose.
#[must_use]
pub fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = if rows > 0 { a[0].len() } else { 0 };
    (0..cols)
        .map(|j| (0..rows).map(|i| a[i][j]).collect())
        .collect()
}

/// Matrix multiplication C = A @ B.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let inner = a[0].len();
    let cols = b[0].len();
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|k| (0..inner).map(|j| a[i][j] * b[j][k]).sum())
                .collect()
        })
        .collect()
}

/// Commutator [A, B] = AB - BA.
#[must_use]
pub fn commutator(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let ab = mat_mul(a, b);
    let ba = mat_mul(b, a);
    let n = ab.len();
    let m = ab[0].len();
    (0..n)
        .map(|i| (0..m).map(|j| ab[i][j] - ba[i][j]).collect())
        .collect()
}

/// Distance to normal: ||A*A - AA*||_F / (2||A||_F).
/// For real matrices, A* = A^T. A is normal iff A*A = AA*.
#[must_use]
pub fn distance_to_normal(a: &[Vec<f64>]) -> f64 {
    let n = frobenius_norm(a);
    if n < 1e-300 {
        return 0.0;
    }
    let at = transpose(a);
    let ata = mat_mul(&at, a);
    let aat = mat_mul(a, &at);
    let diff_norm = frobenius_norm(&sub_matrices(&ata, &aat));
    diff_norm / (2.0 * n)
}

fn sub_matrices(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..a.len())
        .map(|i| (0..a[i].len()).map(|j| a[i][j] - b[i][j]).collect())
        .collect()
}

/// Commutativity ratio: `||[A,B]||_F / (||A||_F * ||B||_F)`. Scale-invariant.
#[must_use]
pub fn commutativity_ratio(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let na = frobenius_norm(a);
    let nb = frobenius_norm(b);
    if na * nb < 1e-300 {
        return 0.0;
    }
    let comm = commutator(a, b);
    frobenius_norm(&comm) / (na * nb)
}

/// Skip connection analysis: `(raw_ratio, skip_ratio)` for W1,W2 vs (I+W1),(I+W2).
#[must_use]
pub fn skip_commutativity(w1: &[Vec<f64>], w2: &[Vec<f64>]) -> (f64, f64) {
    let n = w1.len();
    let i = identity_matrix(n);
    let i_plus_w1 = add_matrices(&i, w1);
    let i_plus_w2 = add_matrices(&i, w2);
    let raw = commutativity_ratio(w1, w2);
    let skip = commutativity_ratio(&i_plus_w1, &i_plus_w2);
    (raw, skip)
}

/// Identity matrix of size n.
#[must_use]
pub fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

fn add_matrices(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..a.len())
        .map(|i| (0..a[i].len()).map(|j| a[i][j] + b[i][j]).collect())
        .collect()
}

/// Generate random matrix (n x n) with standard normal entries / sqrt(n).
#[must_use]
pub fn random_matrix(n: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let scale = 1.0 / (n as f64).sqrt();
    (0..n)
        .map(|_| {
            (0..n)
                .map(|_| rng.normal_params(0.0, 1.0) * scale)
                .collect()
        })
        .collect()
}

/// Symmetric matrix from random H: (H + H^T) / 2. Normal (commutes with adjoint).
#[must_use]
pub fn random_symmetric(n: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let h = random_matrix(n, rng);
    let ht = transpose(&h);
    (0..n)
        .map(|i| (0..n).map(|j| (h[i][j] + ht[i][j]) * 0.5).collect())
        .collect()
}

/// Spectral gap: max|eig(A^T A) - eig(A A^T)|. Zero for normal matrices.
///
/// Uses Frobenius norm of A^T A - A A^T as proxy (zero for normal).
#[must_use]
pub fn spectral_gap_approx(a: &[Vec<f64>]) -> f64 {
    let at = transpose(a);
    let ata = mat_mul(&at, a);
    let aat = mat_mul(a, &at);
    let diff = sub_matrices(&ata, &aat);
    frobenius_norm(&diff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commutator_antisymmetric() {
        let mut rng = Rng::new(42);
        let a = random_matrix(8, &mut rng);
        let b = random_matrix(8, &mut rng);
        let ab = commutator(&a, &b);
        let ba = commutator(&b, &a);
        let sum: Vec<Vec<f64>> = (0..8)
            .map(|i| (0..8).map(|j| ab[i][j] + ba[i][j]).collect())
            .collect();
        assert!(frobenius_norm(&sum) < 1e-10, " [A,B] = -[B,A]");
    }

    #[test]
    fn distance_to_normal_symmetric_zero() {
        let mut rng = Rng::new(42);
        let sym = random_symmetric(16, &mut rng);
        let d = distance_to_normal(&sym);
        assert!(d < 1e-10, "symmetric (normal) should have d≈0, got {d}");
    }

    #[test]
    fn skip_reduces_commutativity() {
        let mut rng = Rng::new(42);
        let w1 = random_matrix(16, &mut rng);
        let w2 = random_matrix(16, &mut rng);
        let (raw, skip) = skip_commutativity(&w1, &w2);
        assert!(skip < raw, "skip ({skip}) < raw ({raw})");
    }
}
