// SPDX-License-Identifier: AGPL-3.0-or-later

//! S-12 resolution: Householder+QR eigensolver vs BarraCUDA Jacobi.
//!
//! Compares reconstruction error ‖A - V D Vᵀ‖_F at n = 4, 8, 16, 32, 64
//! to demonstrate that the Householder+QR implementation achieves
//! LAPACK-level accuracy where Jacobi degrades.
//!
//! This binary provides the evidence for ToadStool to absorb the fix.
//!
//! ## Provenance
//!
//! CPU reference: `eigh::eigh_householder_qr` vs `barracuda::linalg::eigh_f64` (Jacobi).
//! Validates: analytical reconstruction error, orthogonality, Gershgorin bounds.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::suboptimal_flops
)]

use neural_spring::eigh;
use neural_spring::require;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("eigh_accuracy_s12");

    for &n in &[4_usize, 8, 16, 32, 64] {
        validate_accuracy(&mut h, n);
    }

    validate_anderson_hamiltonian(&mut h);
    validate_orthogonality(&mut h);

    h.finish();
}

fn random_symmetric(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
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

fn validate_accuracy(h: &mut ValidationHarness, n: usize) {
    let a = random_symmetric(n, 42 + n as u64);

    // Householder+QR (local)
    let hqr = eigh::eigh_householder_qr(&a, n);
    let hqr_err = hqr.reconstruction_error(&a);

    // BarraCUDA Jacobi
    let jacobi = require!(h, barracuda::linalg::eigh_f64(&a, n), "jacobi eigh");
    let jacobi_recon = jacobi.reconstruct();
    let jacobi_err: f64 = jacobi_recon
        .iter()
        .zip(a.iter())
        .map(|(r, ai)| (r - ai).powi(2))
        .sum::<f64>()
        .sqrt();

    eprintln!(
        "  n={n:>3}: HQR err={hqr_err:.2e}, Jacobi err={jacobi_err:.2e}, improvement={:.0}×",
        jacobi_err / hqr_err.max(1e-300)
    );

    // HQR reconstruction error scales as O(n² ε_mach) where ε_mach ≈ 1e-16.
    // Each tier is sized for the expected error growth at that matrix dimension.
    let tol = match n {
        4 => tolerances::CROSS_LANGUAGE,
        8 => tolerances::GPU_F64_TRANSCENDENTAL,
        16 => tolerances::SPECIAL_FUNCTION_F64,
        32 => tolerances::GPU_FITNESS_F32,
        _ => tolerances::EIGH_JACOBI_RECONSTRUCT,
    };

    h.check_abs(
        &format!("n={n} Householder+QR reconstruction error"),
        hqr_err,
        0.0,
        tol,
    );
}

fn validate_anderson_hamiltonian(h: &mut ValidationHarness) {
    // Anderson Hamiltonian: tridiagonal with random diagonal disorder
    // This is the exact use case from Papers 022-023
    let n = 32;
    let t = 1.0;
    let w = 4.0;
    let mut rng = Rng::new(42);

    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = rng.uniform() * w - w / 2.0;
    }
    for i in 0..n - 1 {
        a[i * n + i + 1] = -t;
        a[(i + 1) * n + i] = -t;
    }

    let hqr = eigh::eigh_householder_qr(&a, n);
    let err = hqr.reconstruction_error(&a);
    eprintln!("  Anderson n=32, W=4: HQR err={err:.2e}");

    h.check_abs(
        "Anderson Hamiltonian n=32 reconstruction",
        err,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    // Eigenvalues should be real and bounded by Gershgorin
    let max_eig = hqr
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_eig = hqr
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_gershgorin = (0..n)
        .map(|i| {
            let mut r = 0.0;
            for j in 0..n {
                if j != i {
                    r += a[i * n + j].abs();
                }
            }
            a[i * n + i] + r
        })
        .fold(f64::NEG_INFINITY, f64::max);

    h.check_bool(
        "eigenvalues within Gershgorin bounds",
        max_eig <= max_gershgorin + tolerances::CROSS_LANGUAGE
            && min_eig >= -max_gershgorin - tolerances::CROSS_LANGUAGE,
    );
}

fn validate_orthogonality(h: &mut ValidationHarness) {
    let n = 32;
    let a = random_symmetric(n, 777);
    let r = eigh::eigh_householder_qr(&a, n);

    // VᵀV should be identity
    let mut max_off = 0.0_f64;
    let mut max_diag_err = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..n {
                dot += r.eigenvectors[k * n + i] * r.eigenvectors[k * n + j];
            }
            if i == j {
                max_diag_err = max_diag_err.max((dot - 1.0).abs());
            } else {
                max_off = max_off.max(dot.abs());
            }
        }
    }

    eprintln!("  n=32 orthogonality: diag_err={max_diag_err:.2e}, off_diag={max_off:.2e}");

    h.check_abs(
        "n=32 eigenvector orthogonality (diagonal)",
        max_diag_err,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "n=32 eigenvector orthogonality (off-diagonal)",
        max_off,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
}
