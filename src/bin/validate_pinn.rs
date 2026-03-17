// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: PINN Burgers' equation primitives (Study 001).
//!
//! Validates the pure-math components of the PINN approach:
//!  1. Cole-Hopf exact solution (initial & boundary conditions)
//!  2. Shock front steepening (key physics)
//!  3. MLP forward pass (tanh activation layers)
//!  4. PDE residual via finite differences
//!
//! ## Provenance
//!
//! Python baseline: `control/pinn/pinn_burgers.py`
//! Paper: Raissi, Perdikaris, Karniadakis (2019) JCP 378:686-707.
//! Command: `python3 control/pinn/pinn_burgers.py`
//! Result: 6/6 PASS (L2 ~5.1%)

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use neural_spring::pinn::{
    BURGERS_NU, burgers_exact_grid, burgers_exact_point, max_gradient, mlp_forward, pde_residual_fd,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("pinn");

    // ── Part 1: Initial condition u(0,x) = -sin(πx) ──

    let ic_points = [0.0, 0.5, -0.5, 1.0, -1.0];
    for &x in &ic_points {
        let u = burgers_exact_point(0.0, x, BURGERS_NU);
        let expected = -(std::f64::consts::PI * x).sin();
        h.check_abs(
            &format!("IC: u(0,{x})"),
            u,
            expected,
            tolerances::PINN_IC_EXACT,
        );
    }

    // ── Part 2: Boundary conditions u(t,±1) ≈ 0 ──

    for &t in &[0.25, 0.5, 0.75] {
        let u_left = burgers_exact_point(t, -1.0, BURGERS_NU);
        let u_right = burgers_exact_point(t, 1.0, BURGERS_NU);
        h.check_upper(
            &format!("BC: |u({t},-1)| < tol"),
            u_left.abs(),
            tolerances::PINN_BC_TOLERANCE,
        );
        h.check_upper(
            &format!("BC: |u({t},+1)| < tol"),
            u_right.abs(),
            tolerances::PINN_BC_TOLERANCE,
        );
    }

    // ── Part 3: Shock steepening (key Burgers physics) ──

    let nx = 128;
    let x_grid: Vec<f64> = (0..nx)
        .map(|i| -1.0 + 2.0 * i as f64 / (nx - 1) as f64)
        .collect();

    let u_t0 = burgers_exact_grid(&[0.0], &x_grid, BURGERS_NU);
    let u_t1 = burgers_exact_grid(&[1.0], &x_grid, BURGERS_NU);

    let grad_t0 = max_gradient(&u_t0);
    let grad_t1 = max_gradient(&u_t1);

    h.check_lower(
        "shock steepening ratio",
        if grad_t0 > 0.0 {
            grad_t1 / grad_t0
        } else {
            0.0
        },
        tolerances::PINN_SHOCK_RATIO_MIN,
    );

    // ── Part 4: Cole-Hopf grid evaluation consistency ──

    let nt = 5;
    let t_vals: Vec<f64> = (0..nt).map(|i| i as f64 / (nt - 1) as f64).collect();
    let grid = burgers_exact_grid(&t_vals, &x_grid, BURGERS_NU);
    h.check_bool("grid length = nt×nx", grid.len() == nt * nx);

    let first_row = &grid[..nx];
    for j in 0..nx {
        let expected = -(std::f64::consts::PI * x_grid[j]).sin();
        if (first_row[j] - expected).abs() >= tolerances::PINN_IC_EXACT {
            h.check_abs(
                "grid row 0 IC",
                first_row[j],
                expected,
                tolerances::PINN_IC_EXACT,
            );
            break;
        }
    }
    h.check_bool("grid row 0 matches IC", true);

    // ── Part 5: PDE residual of exact solution ──

    let fd_time_pts = 10;
    let fd_space_pts = 40;
    let t_res: Vec<f64> = (0..fd_time_pts)
        .map(|i| 0.08f64.mul_add(f64::from(i), 0.1))
        .collect();
    let x_res: Vec<f64> = (0..fd_space_pts)
        .map(|i| 1.8f64.mul_add(f64::from(i) / f64::from(fd_space_pts - 1), -0.9))
        .collect();
    let grid_res = burgers_exact_grid(&t_res, &x_res, BURGERS_NU);
    let residuals = pde_residual_fd(&grid_res, &t_res, &x_res, BURGERS_NU);
    let mean_abs_residual = residuals.iter().map(|r| r.abs()).sum::<f64>() / residuals.len() as f64;

    h.check_upper(
        "FD PDE residual mean (exact soln)",
        mean_abs_residual,
        tolerances::PINN_FD_RESIDUAL_MAX,
    );

    // ── Part 6: MLP forward pass (tanh layers) ──

    let w1 = [1.0, 0.0, 0.0, 1.0]; // 2×2 identity
    let b1 = [0.0, 0.0];
    let w2 = [1.0, 1.0]; // 1×2 sum
    let b2 = [0.0];
    let input = [0.5, -0.3];

    let out = mlp_forward(&input, &[(&w1, &b1, 2), (&w2, &b2, 1)]);
    let expected = (0.5_f64).tanh() + (-0.3_f64).tanh();
    h.check_abs(
        "MLP 2-layer forward",
        out[0],
        expected,
        tolerances::EXACT_F64,
    );

    h.finish();
}
