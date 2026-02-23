// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `barracuda::linalg` CPU f64 primitives.
//!
//! Validates `solve_f64`, `lu_det`/`lu_solve`, `eigh_f64`,
//! `cholesky_f64`, and `tridiagonal_solve` against analytically known solutions.
//!
//! ## Provenance
//!
//! Expected values: analytical (textbook linear algebra).
//! Cross-validated against `NumPy` 1.26 / `SciPy` 1.15.3.

use barracuda::device::WgpuDevice;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let device = rt
        .block_on(async { WgpuDevice::new().await })
        .map(Arc::new)
        .expect("GPU device");

    let mut h = ValidationHarness::new("barracuda_linalg");

    validate_solve(&mut h, &device);
    validate_lu(&mut h);
    validate_eigh(&mut h);
    validate_cholesky(&mut h, &device);
    validate_tridiagonal(&mut h);

    h.finish();
}

fn validate_solve(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let a = vec![2.0, 1.0, 1.0, 3.0];
    let b = vec![5.0, 8.0];

    // A = [[2, 1], [1, 3]], b = [5, 8] => x = [7/5, 11/5] = [1.4, 2.2]
    match barracuda::linalg::solve_f64(device.clone(), &a, &b, 2) {
        Ok(x) => {
            h.check_abs("solve x[0] == 1.4", x[0], 1.4, tolerances::CROSS_LANGUAGE);
            h.check_abs("solve x[1] == 2.2", x[1], 2.2, tolerances::CROSS_LANGUAGE);
        }
        Err(e) => h.check_bool(&format!("solve_f64 [ERROR: {e}]"), false),
    }

    // 3x3 identity => x = b
    let eye3 = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let b3 = vec![3.0, 7.0, 11.0];
    match barracuda::linalg::solve_f64(device.clone(), &eye3, &b3, 3) {
        Ok(x) => {
            h.check_abs("solve(I,b)[0] == 3", x[0], 3.0, tolerances::EXACT_F64);
            h.check_abs("solve(I,b)[1] == 7", x[1], 7.0, tolerances::EXACT_F64);
            h.check_abs("solve(I,b)[2] == 11", x[2], 11.0, tolerances::EXACT_F64);
        }
        Err(e) => h.check_bool(&format!("solve(I,b) [ERROR: {e}]"), false),
    }
}

fn validate_lu(h: &mut ValidationHarness) {
    let a = vec![2.0, 1.0, 1.0, 3.0];
    let b = vec![5.0, 8.0];

    // det([[2, 1], [1, 3]]) = 5
    match barracuda::linalg::lu_det(&a, 2) {
        Ok(det) => h.check_abs("lu_det == 5", det, 5.0, tolerances::CROSS_LANGUAGE),
        Err(e) => h.check_bool(&format!("lu_det [ERROR: {e}]"), false),
    }

    match barracuda::linalg::lu_solve(&a, 2, &b) {
        Ok(x) => {
            h.check_abs("lu_solve[0] == 1.4", x[0], 1.4, tolerances::CROSS_LANGUAGE);
            h.check_abs("lu_solve[1] == 2.2", x[1], 2.2, tolerances::CROSS_LANGUAGE);
        }
        Err(e) => h.check_bool(&format!("lu_solve [ERROR: {e}]"), false),
    }
}

fn validate_eigh(h: &mut ValidationHarness) {
    // A = [[4, 1], [1, 3]] => eigenvalues: (7±√5)/2
    let sym = vec![4.0, 1.0, 1.0, 3.0];
    match barracuda::linalg::eigh_f64(&sym, 2) {
        Ok(eigh) => {
            let half_d = 5.0_f64.sqrt() / 2.0;
            h.check_abs(
                "eigh λ₀",
                eigh.eigenvalues[0],
                3.5 - half_d,
                tolerances::CROSS_LANGUAGE,
            );
            h.check_abs(
                "eigh λ₁",
                eigh.eigenvalues[1],
                3.5 + half_d,
                tolerances::CROSS_LANGUAGE,
            );

            let v0 = eigh.eigenvectors[0];
            let v1 = eigh.eigenvectors[2];
            let av = sym[0].mul_add(v0, sym[1] * v1);
            let lv = eigh.eigenvalues[0] * v0;
            h.check_abs("eigh Av₀≈λ₀v₀", av, lv, tolerances::CROSS_LANGUAGE);
        }
        Err(e) => h.check_bool(&format!("eigh [ERROR: {e}]"), false),
    }
}

fn validate_cholesky(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    // A = [[4, 2], [2, 5]] => L = [[2, 0], [1, 2]]
    let spd = vec![4.0, 2.0, 2.0, 5.0];
    match barracuda::linalg::cholesky_f64(device.clone(), &spd, 2) {
        Ok(chol) => {
            h.check_abs("chol L[0,0]==2", chol.l[0], 2.0, tolerances::CROSS_LANGUAGE);
            h.check_abs("chol L[1,0]==1", chol.l[2], 1.0, tolerances::CROSS_LANGUAGE);
            h.check_abs("chol L[1,1]==2", chol.l[3], 2.0, tolerances::CROSS_LANGUAGE);
        }
        Err(e) => h.check_bool(&format!("cholesky [ERROR: {e}]"), false),
    }
}

fn validate_tridiagonal(h: &mut ValidationHarness) {
    // [2,-1,0; -1,2,-1; 0,-1,2] x = [1, 0, 1] => x = [1, 1, 1]
    let lower = vec![-1.0, -1.0];
    let main_diag = vec![2.0, 2.0, 2.0];
    let upper = vec![-1.0, -1.0];
    let rhs = vec![1.0, 0.0, 1.0];
    match barracuda::linalg::tridiagonal_solve(&lower, &main_diag, &upper, &rhs) {
        Ok(sol) => {
            h.check_abs("tridiag[0]==1", sol[0], 1.0, tolerances::CROSS_LANGUAGE);
            h.check_abs("tridiag[1]==1", sol[1], 1.0, tolerances::CROSS_LANGUAGE);
            h.check_abs("tridiag[2]==1", sol[2], 1.0, tolerances::CROSS_LANGUAGE);
        }
        Err(err) => h.check_bool(&format!("tridiag [ERROR: {err}]"), false),
    }
}
