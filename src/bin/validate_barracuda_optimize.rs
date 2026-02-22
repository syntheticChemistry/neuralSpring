// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `barracuda::optimize` CPU algorithms.
//!
//! Validates `nelder_mead` on Rosenbrock and Rastrigin, and `bisect`/`brent`
//! on known root-finding problems, against analytical solutions.
//!
//! ## Provenance
//!
//! Expected values: analytical (Rosenbrock min at (1,1), Rastrigin at (0,0)).

use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_optimize");

    // --- Nelder-Mead on Rosenbrock ---
    let rosenbrock = |x: &[f64]| {
        let dx = 1.0 - x[0];
        let dy = x[0].mul_add(-x[0], x[1]);
        dx.mul_add(dx, 100.0 * dy * dy)
    };

    let x0 = vec![0.0, 0.0];
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

    match barracuda::optimize::nelder_mead(rosenbrock, &x0, &bounds, 10_000, 1e-10) {
        Ok((x_best, f_best, n_evals)) => {
            h.check_abs(
                "NM Rosenbrock x[0] ≈ 1",
                x_best[0],
                1.0,
                tolerances::OPTIMIZER_POSITION,
            );
            h.check_abs(
                "NM Rosenbrock x[1] ≈ 1",
                x_best[1],
                1.0,
                tolerances::OPTIMIZER_POSITION,
            );
            h.check_upper(
                "NM Rosenbrock f < OPTIMIZER_VALUE_AT_MIN",
                f_best,
                tolerances::OPTIMIZER_VALUE_AT_MIN,
            );
            h.check_bool("NM Rosenbrock converged", n_evals > 0);
        }
        Err(e) => h.check_bool(&format!("NM Rosenbrock [ERROR: {e}]"), false),
    }

    // --- Nelder-Mead on Rastrigin (multimodal) ---
    #[allow(clippy::cast_precision_loss)]
    let rastrigin = |x: &[f64]| {
        let n = x.len() as f64;
        let sum: f64 = x
            .iter()
            .map(|&xi| xi.mul_add(xi, -(10.0 * (2.0 * std::f64::consts::PI * xi).cos())))
            .sum();
        10.0f64.mul_add(n, sum)
    };

    let x0_rast = vec![0.1, 0.1];
    let bounds_rast = vec![(-1.0, 1.0), (-1.0, 1.0)];

    match barracuda::optimize::nelder_mead(rastrigin, &x0_rast, &bounds_rast, 5_000, 1e-10) {
        Ok((x_best, f_best, _)) => {
            h.check_abs(
                "NM Rastrigin x[0] ≈ 0",
                x_best[0],
                0.0,
                tolerances::OPTIMIZER_POSITION_MULTIMODAL,
            );
            h.check_abs(
                "NM Rastrigin x[1] ≈ 0",
                x_best[1],
                0.0,
                tolerances::OPTIMIZER_POSITION_MULTIMODAL,
            );
            h.check_upper(
                "NM Rastrigin f < OPTIMIZER_VALUE_MULTIMODAL",
                f_best,
                tolerances::OPTIMIZER_VALUE_MULTIMODAL,
            );
        }
        Err(e) => h.check_bool(&format!("NM Rastrigin [ERROR: {e}]"), false),
    }

    // --- Bisection: √2 ---
    let f_sqrt2 = |x: f64| x.mul_add(x, -2.0);
    match barracuda::optimize::bisect(f_sqrt2, 1.0, 2.0, 1e-12, 100) {
        Ok(root) => {
            h.check_abs(
                "bisect(x²-2) ≈ √2",
                root,
                std::f64::consts::SQRT_2,
                tolerances::CROSS_LANGUAGE,
            );
        }
        Err(e) => h.check_bool(&format!("bisect √2 [ERROR: {e}]"), false),
    }

    // --- Bisection: cube root of 1 ---
    let f_cbrt = |x: f64| (x * x).mul_add(x, -1.0);
    match barracuda::optimize::bisect(f_cbrt, 0.5, 2.0, 1e-12, 100) {
        Ok(root) => {
            h.check_abs("bisect(x³-1) ≈ 1.0", root, 1.0, tolerances::CROSS_LANGUAGE);
        }
        Err(e) => h.check_bool(&format!("bisect cbrt [ERROR: {e}]"), false),
    }

    // --- Brent: sin(x) = 0.5 => x = π/6 ---
    let f_sin = |x: f64| x.sin() - 0.5;
    match barracuda::optimize::brent(f_sin, 0.0, 1.0, 1e-12, 100) {
        Ok(result) => {
            h.check_abs(
                "brent(sin(x)-0.5) ≈ π/6",
                result.root,
                std::f64::consts::FRAC_PI_6,
                tolerances::CROSS_LANGUAGE,
            );
        }
        Err(e) => h.check_bool(&format!("brent sin [ERROR: {e}]"), false),
    }

    h.finish();
}
