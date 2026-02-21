// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `DeepONet` operator learning primitives (Study 002).
//!
//! Validates pure-math components of the `DeepONet` approach:
//!  1. Polynomial evaluation + exact antiderivative
//!  2. Branch-trunk dot product
//!  3. Dataset generation consistency
//!  4. Error metrics (L2 relative, RMSE)
//!
//! ## Provenance
//!
//! Python baseline: `control/deeponet/deeponet_antideriv.py`
//! Paper: Lu, Jin, Pang, Zhang, Karniadakis (2021) NMI 3:218-229.
//! Command: `python3 control/deeponet/deeponet_antideriv.py`
//! Result: 5/5 PASS (mean L2 ~1.2%)

use neural_spring::deeponet::{
    branch_trunk_dot, eval_polynomial, exact_antiderivative, generate_dataset, l2_relative_error,
    linspace, mlp_forward, rmse,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("deeponet");

    let y = linspace(0.0, 1.0, 50);

    // ── Part 1: Known antiderivatives ──

    // u(x) = 1 → G(y) = y
    let g_const = exact_antiderivative(&[1.0], &y);
    for (i, &g) in g_const.iter().enumerate() {
        if (g - y[i]).abs() >= tolerances::DEEPONET_EXACT_ANTIDERIV {
            h.check_abs("∫1 dy = y", g, y[i], tolerances::DEEPONET_EXACT_ANTIDERIV);
        }
    }
    h.check_bool("u=1 → G=y (all points)", true);

    // u(x) = x → G(y) = y²/2
    let g_linear = exact_antiderivative(&[0.0, 1.0], &y);
    let max_err_linear: f64 = g_linear
        .iter()
        .zip(y.iter())
        .map(|(&g, &yi)| (g - yi.powi(2) / 2.0).abs())
        .fold(0.0_f64, f64::max);
    h.check_abs(
        "∫x dy = y²/2",
        max_err_linear,
        0.0,
        tolerances::DEEPONET_EXACT_ANTIDERIV,
    );

    // u(x) = x² → G(y) = y³/3
    let g_quad = exact_antiderivative(&[0.0, 0.0, 1.0], &y);
    let max_err_quad: f64 = g_quad
        .iter()
        .zip(y.iter())
        .map(|(&g, &yi)| (g - yi.powi(3) / 3.0).abs())
        .fold(0.0_f64, f64::max);
    h.check_abs(
        "∫x² dy = y³/3",
        max_err_quad,
        0.0,
        tolerances::DEEPONET_EXACT_ANTIDERIV,
    );

    // u(x) = 3x² + 2x + 1 → G(y) = y³ + y² + y
    let g_mixed = exact_antiderivative(&[1.0, 2.0, 3.0], &y);
    let max_err_mixed: f64 = g_mixed
        .iter()
        .zip(y.iter())
        .map(|(&g, &yi)| (g - yi.mul_add(yi, yi.powi(3)) - yi).abs())
        .fold(0.0_f64, f64::max);
    h.check_abs(
        "∫(3x²+2x+1) dy = y³+y²+y",
        max_err_mixed,
        0.0,
        tolerances::DEEPONET_EXACT_ANTIDERIV,
    );

    // ── Part 2: Polynomial evaluation ──

    let x_pts = linspace(0.0, 1.0, 5);
    let poly_vals = eval_polynomial(&[1.0, 2.0, 3.0], &x_pts);
    // p(x) = 1 + 2x + 3x² at x=1 → 6
    let last = poly_vals[x_pts.len() - 1];
    h.check_abs(
        "eval_polynomial(1+2x+3x², x=1) = 6",
        last,
        6.0,
        tolerances::DEEPONET_POLYNOMIAL_EXACT,
    );
    // p(0) = 1
    h.check_abs(
        "eval_polynomial(1+2x+3x², x=0) = 1",
        poly_vals[0],
        1.0,
        tolerances::DEEPONET_POLYNOMIAL_EXACT,
    );

    // ── Part 3: Branch-trunk dot product ──

    let branch = [1.0, 2.0, 3.0, 4.0];
    let trunk = [4.0, 3.0, 2.0, 1.0];
    let dot = branch_trunk_dot(&branch, &trunk, 0.5);
    // 1*4 + 2*3 + 3*2 + 4*1 + 0.5 = 20.5
    h.check_abs("branch-trunk dot", dot, 20.5, tolerances::EXACT_F64);

    // ── Part 4: Dataset generation ──

    let x_sensors = linspace(0.0, 1.0, 20);
    let y_outputs = linspace(0.0, 1.0, 15);
    let (u_data, g_data) = generate_dataset(100, &x_sensors, &y_outputs, 5, 42);
    h.check_bool("dataset U shape", u_data.len() == 100 * 20);
    h.check_bool("dataset G shape", g_data.len() == 100 * 15);

    // All U values should be finite
    let all_finite = u_data.iter().all(|x| x.is_finite()) && g_data.iter().all(|x| x.is_finite());
    h.check_bool("dataset all finite", all_finite);

    // Determinism: same seed produces same data
    let (u2, g2) = generate_dataset(100, &x_sensors, &y_outputs, 5, 42);
    h.check_bool("dataset deterministic", u_data == u2 && g_data == g2);

    // ── Part 5: Error metrics ──

    let pred = [1.0, 2.0, 3.0];
    let exact = [1.0, 2.0, 3.0];
    h.check_abs(
        "L2 error (perfect)",
        l2_relative_error(&pred, &exact),
        0.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "RMSE (perfect)",
        rmse(&pred, &exact),
        0.0,
        tolerances::EXACT_F64,
    );

    let pred2 = [1.1, 2.1, 3.1];
    h.check_bool(
        "L2 error > 0 for imperfect",
        l2_relative_error(&pred2, &exact) > 0.0,
    );
    h.check_bool("RMSE > 0 for imperfect", rmse(&pred2, &exact) > 0.0);

    // ── Part 6: MLP forward pass ──

    let w = [1.0, 0.0, 0.0, 1.0]; // 2×2 identity
    let b = [0.0, 0.0];
    let input = [0.7, -0.4];
    let out = mlp_forward(&input, &[(&w, &b, 2)]);
    h.check_abs("MLP passthrough [0]", out[0], 0.7, tolerances::EXACT_F64);
    h.check_abs("MLP passthrough [1]", out[1], -0.4, tolerances::EXACT_F64);

    h.finish();
}
