// SPDX-License-Identifier: AGPL-3.0-only

//! Validation binary: extended `barracuda::linalg` / `barracuda::ops::linalg` CPU primitives.
//!
//! Validates SVD (`svd_decompose`, `svd_values`, `svd_pinv`), LU inverse
//! (`lu_inverse`), and generalized eigendecomposition (`gen_eigh_f64`)
//! against analytically known values. Extends `validate_barracuda_linalg`.
//!
//! ## Provenance
//!
//! Expected values: analytical solutions for small systems.
//! SVD reference: A = U Σ V^T for known matrices.

use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_linalg_ext");

    validate_svd(&mut h);
    validate_lu_inverse(&mut h);
    validate_gen_eigh(&mut h);

    h.finish();
}

fn validate_svd(h: &mut ValidationHarness) {
    // 2×2 diagonal matrix: [[3,0],[0,2]]
    // Singular values: [3, 2]
    let a = vec![3.0, 0.0, 0.0, 2.0];

    match barracuda::ops::linalg::svd::svd_values(&a, 2, 2) {
        Ok(s) => {
            let s0 = s[0].max(s[1]);
            let s1 = s[0].min(s[1]);
            h.check_abs("svd_values diag σ₀=3", s0, 3.0, tolerances::CROSS_LANGUAGE);
            h.check_abs("svd_values diag σ₁=2", s1, 2.0, tolerances::CROSS_LANGUAGE);
        }
        Err(e) => h.check_bool(&format!("svd_values diagonal [ERROR: {e}]"), false),
    }

    // SVD of identity: all singular values = 1
    let eye = vec![1.0, 0.0, 0.0, 1.0];
    match barracuda::ops::linalg::svd::svd_values(&eye, 2, 2) {
        Ok(s) => {
            h.check_abs("svd_values(I) σ₀=1", s[0], 1.0, tolerances::CROSS_LANGUAGE);
            h.check_abs("svd_values(I) σ₁=1", s[1], 1.0, tolerances::CROSS_LANGUAGE);
        }
        Err(e) => h.check_bool(&format!("svd_values identity [ERROR: {e}]"), false),
    }

    // Pseudoinverse of diagonal: [[1/3,0],[0,1/2]]
    match barracuda::ops::linalg::svd::svd_pinv(&a, 2, 2, 1e-10) {
        Ok(pinv) => {
            h.check_abs(
                "svd_pinv diag [0,0]=1/3",
                pinv[0],
                1.0 / 3.0,
                tolerances::CROSS_LANGUAGE,
            );
            h.check_abs(
                "svd_pinv diag [1,1]=1/2",
                pinv[3],
                0.5,
                tolerances::CROSS_LANGUAGE,
            );
        }
        Err(e) => h.check_bool(&format!("svd_pinv diagonal [ERROR: {e}]"), false),
    }

    // Full SVD decomposition: verify A ≈ U Σ V^T
    match barracuda::ops::linalg::svd::svd_decompose(&a, 2, 2) {
        Ok(svd) => {
            let reconstructed = reconstruct_2x2(&svd.u, &svd.s, &svd.vt);
            let max_err = a
                .iter()
                .zip(reconstructed.iter())
                .map(|(a_i, r_i)| (a_i - r_i).abs())
                .fold(0.0_f64, f64::max);
            h.check_abs(
                "svd reconstruct max_err",
                max_err,
                0.0,
                tolerances::CROSS_LANGUAGE,
            );
        }
        Err(e) => h.check_bool(&format!("svd_decompose diagonal [ERROR: {e}]"), false),
    }
}

fn validate_lu_inverse(h: &mut ValidationHarness) {
    // Inverse of [[2,1],[1,3]]: det=5, inv = [[3/5, -1/5],[-1/5, 2/5]]
    let a = vec![2.0, 1.0, 1.0, 3.0];
    match barracuda::ops::linalg::lu::lu_inverse(&a, 2) {
        Ok(inv) => {
            h.check_abs("lu_inv[0,0]=3/5", inv[0], 0.6, tolerances::CROSS_LANGUAGE);
            h.check_abs("lu_inv[0,1]=-1/5", inv[1], -0.2, tolerances::CROSS_LANGUAGE);
            h.check_abs("lu_inv[1,0]=-1/5", inv[2], -0.2, tolerances::CROSS_LANGUAGE);
            h.check_abs("lu_inv[1,1]=2/5", inv[3], 0.4, tolerances::CROSS_LANGUAGE);
        }
        Err(e) => h.check_bool(&format!("lu_inverse [ERROR: {e}]"), false),
    }

    // Inverse of identity = identity
    let eye = vec![1.0, 0.0, 0.0, 1.0];
    match barracuda::ops::linalg::lu::lu_inverse(&eye, 2) {
        Ok(inv) => {
            h.check_abs("lu_inv(I)[0,0]=1", inv[0], 1.0, tolerances::EXACT_F64);
            h.check_abs("lu_inv(I)[0,1]=0", inv[1], 0.0, tolerances::EXACT_F64);
        }
        Err(e) => h.check_bool(&format!("lu_inverse identity [ERROR: {e}]"), false),
    }
}

fn validate_gen_eigh(h: &mut ValidationHarness) {
    // Generalized eigenvalue problem: A x = λ B x
    // When B = I, this reduces to standard eigenvalue problem A x = λ x
    // A = [[3,1],[1,3]], eigenvalues = 2 and 4
    let a = vec![3.0, 1.0, 1.0, 3.0];

    match barracuda::linalg::gen_eigh::gen_eigh_identity_b(&a, 2) {
        Ok(result) => {
            let l0 = result.eigenvalues[0].min(result.eigenvalues[1]);
            let l1 = result.eigenvalues[0].max(result.eigenvalues[1]);
            h.check_abs("gen_eigh(A,I) λ₀=2", l0, 2.0, tolerances::CROSS_LANGUAGE);
            h.check_abs("gen_eigh(A,I) λ₁=4", l1, 4.0, tolerances::CROSS_LANGUAGE);
        }
        Err(e) => h.check_bool(&format!("gen_eigh_identity_b [ERROR: {e}]"), false),
    }

    // Full generalized problem with non-trivial B
    // A = [[4,2],[2,4]], B = [[2,0],[0,2]] (scaled identity)
    // A x = λ B x  ⟹  (A/2) x = λ x  ⟹  eigenvalues of A/2 = {1, 3}
    let a2 = vec![4.0, 2.0, 2.0, 4.0];
    let b2 = vec![2.0, 0.0, 0.0, 2.0];

    match barracuda::linalg::gen_eigh::gen_eigh_f64(&a2, &b2, 2) {
        Ok(result) => {
            let l0 = result.eigenvalues[0].min(result.eigenvalues[1]);
            let l1 = result.eigenvalues[0].max(result.eigenvalues[1]);
            h.check_abs("gen_eigh(A,2I) λ₀=1", l0, 1.0, tolerances::CROSS_LANGUAGE);
            h.check_abs("gen_eigh(A,2I) λ₁=3", l1, 3.0, tolerances::CROSS_LANGUAGE);
        }
        Err(e) => h.check_bool(&format!("gen_eigh_f64 [ERROR: {e}]"), false),
    }
}

#[allow(clippy::cast_precision_loss)]
fn reconstruct_2x2(u: &[f64], s: &[f64], vt: &[f64]) -> Vec<f64> {
    let n = 2;
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..s.len().min(n) {
                result[i * n + j] += u[i * n + k] * s[k] * vt[k * n + j];
            }
        }
    }
    result
}
