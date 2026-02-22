// SPDX-License-Identifier: AGPL-3.0-or-later

//! Symmetric eigenvalue decomposition: Householder + implicit QR.
//!
//! ## Upstream absorption (Feb 22, 2026)
//!
//! This module now delegates to `barracuda::ops::linalg::eigh_householder_qr`,
//! which absorbed neuralSpring's original implementation verbatim. The local
//! fossil is preserved at `metalForge/fossils/evolved_s01_s11/eigh_local.rs`.
//!
//! ## Algorithm
//!
//! 1. **Householder tridiagonalization**: A = Q T Qᵀ in O(4n³/3) flops.
//! 2. **Implicit QR iteration** with Wilkinson shift on T.
//! 3. **Back-transform**: eigenvectors of T → eigenvectors of A via Q.

/// Result of symmetric eigenvalue decomposition A = V·D·Vᵀ.
///
/// Type alias for the upstream `barracuda` decomposition struct.
/// Fields: `eigenvalues`, `eigenvectors` (n×n row-major columns), `n`.
pub type EighResult = barracuda::ops::linalg::EighDecompositionF64;

/// Symmetric eigenvalue decomposition via Householder + implicit QR.
///
/// Delegates to `barracuda::ops::linalg::eigh_householder_qr`.
/// Input: `a` is an n×n symmetric matrix in row-major order.
/// Returns eigenvalues in ascending order with orthonormal eigenvectors.
///
/// # Panics
///
/// Panics if `a.len() != n * n` or `n == 0`.
#[must_use]
pub fn eigh_householder_qr(a: &[f64], n: usize) -> EighResult {
    barracuda::ops::linalg::eigh_householder_qr(a, n)
}

#[cfg(test)]
#[allow(clippy::suboptimal_flops)]
mod tests {
    use super::*;

    fn max_off_diag(m: &[f64], n: usize) -> f64 {
        let mut mx = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    mx = mx.max(m[i * n + j].abs());
                }
            }
        }
        mx
    }

    #[test]
    fn test_2x2_simple() {
        let a = [3.0, 1.0, 1.0, 3.0];
        let r = eigh_householder_qr(&a, 2);
        assert!((r.eigenvalues[0] - 2.0).abs() < 1e-12);
        assert!((r.eigenvalues[1] - 4.0).abs() < 1e-12);
        assert!(r.reconstruction_error(&a) < 1e-12);
    }

    #[test]
    fn test_3x3_tridiagonal() {
        let a = [2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let r = eigh_householder_qr(&a, 3);
        let expected = [
            2.0 - std::f64::consts::SQRT_2,
            2.0,
            2.0 + std::f64::consts::SQRT_2,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (r.eigenvalues[i] - exp).abs() < 1e-12,
                "eigenvalue[{i}]: got {}, expected {exp}",
                r.eigenvalues[i]
            );
        }
        assert!(r.reconstruction_error(&a) < 1e-12);
    }

    fn random_symmetric(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = crate::rng::Rng::new(seed);
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let v = rng.uniform() * 10.0 - 5.0;
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        a
    }

    #[test]
    fn test_5x5_random() {
        let a = random_symmetric(5, 77);
        let r = eigh_householder_qr(&a, 5);
        assert!(
            r.reconstruction_error(&a) < 1e-12,
            "n=5 reconstruction error: {}",
            r.reconstruction_error(&a)
        );
    }

    #[test]
    fn test_8x8_reconstruction() {
        let a = random_symmetric(8, 42);
        let r = eigh_householder_qr(&a, 8);
        assert!(
            r.reconstruction_error(&a) < 1e-10,
            "n=8 reconstruction error: {}",
            r.reconstruction_error(&a)
        );
    }

    #[test]
    fn test_16x16_accuracy() {
        let a = random_symmetric(16, 123);
        let r = eigh_householder_qr(&a, 16);
        assert!(
            r.reconstruction_error(&a) < 1e-8,
            "n=16 reconstruction error: {}",
            r.reconstruction_error(&a)
        );
    }

    #[test]
    fn test_32x32_accuracy() {
        let a = random_symmetric(32, 999);
        let r = eigh_householder_qr(&a, 32);
        assert!(
            r.reconstruction_error(&a) < 1e-6,
            "n=32 reconstruction error: {}",
            r.reconstruction_error(&a)
        );
    }

    #[test]
    fn test_orthogonality() {
        let n = 8;
        let a = random_symmetric(n, 42);
        let r = eigh_householder_qr(&a, n);

        let mut vtv = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    vtv[i * n + j] += r.eigenvectors[k * n + i] * r.eigenvectors[k * n + j];
                }
            }
        }

        let off_err = max_off_diag(&vtv, n);
        assert!(off_err < 1e-10, "VᵀV off-diagonal max: {off_err}");

        for i in 0..n {
            assert!(
                (vtv[i * n + i] - 1.0).abs() < 1e-10,
                "VᵀV diagonal[{i}]: {}",
                vtv[i * n + i]
            );
        }
    }
}
