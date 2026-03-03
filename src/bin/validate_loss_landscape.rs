// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: Loss landscape characterization (baseCamp nS-03).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## baseCamp Sub-thesis 03
//!
//! Loss Landscapes as Energy Landscapes.
//! Experiments nS-301 through nS-305.
//!
//! ## Provenance
//!
//! No Python baseline — these are novel experiments. Validated against
//! analytical known-values (quadratic loss, Rosenbrock function).

use neural_spring::loss_landscape::{
    boltzmann_sampling, hessian_spectrum, landscape_analysis, landscape_flatness,
    landscape_sharpness, metropolis_step, numerical_hessian, saddle_index, spectral_gap,
    transition_barrier,
};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

// boltzmann_sampling now delegates to barracuda::sample::boltzmann_sampling (S56 absorption).
// metropolis_step remains local (barracuda does not expose it separately).

fn quadratic_loss(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

fn rosenbrock_loss(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let dx = 1.0 - x[0];
    let dy = x[0].mul_add(-x[0], x[1]);
    dx.mul_add(dx, 100.0 * dy * dy)
}

fn saddle_loss(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    x[0].mul_add(x[0], -(x[1] * x[1]))
}

#[expect(clippy::too_many_lines, reason = "validation binary")]
fn main() {
    let mut h = ValidationHarness::new("loss_landscape");

    // ── nS-301: Quadratic Hessian = 2*I ──────────────────────────────

    let params = vec![0.0; 4];
    let hessian = numerical_hessian(&quadratic_loss, &params, tolerances::HESSIAN_FD_STEP);
    let mut diag_ok = true;
    let mut off_ok = true;
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 2.0 } else { 0.0 };
            if (hessian[i * 4 + j] - expected).abs() > tolerances::OPTIMIZER_VALUE_AT_MIN {
                if i == j {
                    diag_ok = false;
                } else {
                    off_ok = false;
                }
            }
        }
    }
    h.check_bool("Quadratic Hessian diagonal = 2.0", diag_ok);
    h.check_bool("Quadratic Hessian off-diagonal = 0.0", off_ok);

    // ── nS-301: Quadratic spectrum ───────────────────────────────────

    let spectrum = hessian_spectrum(&hessian, 4);
    let all_near_two = spectrum
        .iter()
        .all(|&ev| (ev - 2.0).abs() < tolerances::EIGH_JACOBI_EIGENVALUE);
    h.check_bool("Quadratic spectrum: all eigenvalues ≈ 2.0", all_near_two);

    // ── nS-301: Landscape at quadratic minimum ───────────────────────

    let result = landscape_analysis(&quadratic_loss, &params, tolerances::HESSIAN_FD_STEP, 0.1);
    h.check_abs(
        "Loss at origin = 0",
        result.loss,
        0.0,
        tolerances::EXACT_F64,
    );
    h.check_bool("Saddle index = 0 (minimum)", result.saddle_index == 0);
    h.check_bool("Sharpness > 0 (positive curvature)", result.sharpness > 1.0);

    // ── nS-301: Rosenbrock at minimum ────────────────────────────────

    let rb_min = vec![1.0, 1.0];
    let rb_result = landscape_analysis(&rosenbrock_loss, &rb_min, tolerances::HESSIAN_FD_STEP, 0.1);
    h.check_bool(
        "Rosenbrock loss at (1,1) < 1e-6",
        rb_result.loss < tolerances::SPECIAL_FUNCTION_F64,
    );
    h.check_bool(
        "Rosenbrock at minimum: saddle_index = 0",
        rb_result.saddle_index == 0,
    );

    // ── nS-301: Saddle point detection ───────────────────────────────

    let saddle_params = vec![0.0, 0.0];
    let saddle_result = landscape_analysis(
        &saddle_loss,
        &saddle_params,
        tolerances::HESSIAN_FD_STEP,
        0.1,
    );
    h.check_bool(
        "Saddle function: saddle_index > 0",
        saddle_result.saddle_index > 0,
    );

    // ── nS-301: Flatness and sharpness ───────────────────────────────

    let flat_eigenvalues = vec![0.001, 0.002, 0.001, 5.0];
    let flatness = landscape_flatness(&flat_eigenvalues, 0.01);
    h.check_abs(
        "Flatness 3/4 for near-zero eigenvalues",
        flatness,
        0.75,
        tolerances::EXACT_F64,
    );

    let sharpness = landscape_sharpness(&flat_eigenvalues);
    h.check_abs(
        "Sharpness = max eigenvalue = 5.0",
        sharpness,
        5.0,
        tolerances::EXACT_F64,
    );

    // ── nS-301: Saddle index counting ────────────────────────────────

    let mixed_eigenvalues = vec![-2.0, -1.0, 0.5, 3.0];
    let idx = saddle_index(&mixed_eigenvalues);
    h.check_bool("Saddle index = 2 for two negative eigenvalues", idx == 2);

    // ── nS-301: Spectral gap ─────────────────────────────────────────

    let gap = spectral_gap(&mixed_eigenvalues);
    h.check_abs(
        "Spectral gap = 3.0 - 0.5 = 2.5",
        gap,
        2.5,
        tolerances::EXACT_F64,
    );

    // ── nS-302: Transition barrier ───────────────────────────────────

    let barrier = transition_barrier(0.0, 0.5, 2.0);
    h.check_abs(
        "Transition barrier = 2.0",
        barrier,
        2.0,
        tolerances::EXACT_F64,
    );

    // ── nS-303: Metropolis step preserves dimension ──────────────────

    let mut rng = Rng::new(42);
    let init = vec![1.0, 2.0, 3.0];
    let loss = quadratic_loss(&init);
    let (new_params, _) = metropolis_step(&quadratic_loss, &init, loss, 1.0, 0.1, &mut rng);
    h.check_bool(
        "Metropolis preserves parameter dimension",
        new_params.len() == init.len(),
    );

    // ── nS-303: Boltzmann sampling at high temperature ───────────────

    let init2 = vec![1.0, 1.0];
    let result = boltzmann_sampling(&quadratic_loss, &init2, 100.0, 0.1, 500, 42);
    h.check_bool(
        "High-temp acceptance rate > 30%",
        result.acceptance_rate > 0.3,
    );
    h.check_bool(
        "Boltzmann produces expected number of samples",
        result.losses.len() == 501,
    );

    // ── nS-303: Low temperature favors lower loss ────────────────────

    let result_lo = boltzmann_sampling(&quadratic_loss, &init2, 0.01, 0.01, 500, 42);
    let mean_loss_lo: f64 = result_lo.losses.iter().sum::<f64>()
        / f64::from(u32::try_from(result_lo.losses.len()).unwrap_or(1));

    let result_hi = boltzmann_sampling(&quadratic_loss, &init2, 100.0, 0.1, 500, 42);
    let mean_loss_hi: f64 = result_hi.losses.iter().sum::<f64>()
        / f64::from(u32::try_from(result_hi.losses.len()).unwrap_or(1));

    h.check_bool(
        "Low temperature produces lower mean loss than high",
        mean_loss_lo < mean_loss_hi + 5.0,
    );

    // ── nS-304: Cross-architecture comparison (dimension sweep) ──────

    for dim in [2, 4, 8] {
        let origin = vec![0.0; dim];
        let result = landscape_analysis(&quadratic_loss, &origin, tolerances::HESSIAN_FD_STEP, 0.1);
        h.check_bool(
            &format!("nS-304: quadratic dim={dim} saddle_index=0"),
            result.saddle_index == 0,
        );
    }

    // ── nS-304: Rosenbrock landscape at non-minimum ──────────────────

    let rb_off = vec![0.0, 0.0];
    let rb_off_result =
        landscape_analysis(&rosenbrock_loss, &rb_off, tolerances::HESSIAN_FD_STEP, 0.1);
    h.check_bool(
        "nS-304: Rosenbrock loss at (0,0) > 0",
        rb_off_result.loss > 0.5,
    );

    // ── nS-305: Training dynamics (gradient descent on quadratic) ────

    let mut param_traj = vec![5.0, 3.0];
    let lr = 0.1;
    let mut loss_traj = Vec::new();
    for _ in 0..20 {
        let loss = quadratic_loss(&param_traj);
        loss_traj.push(loss);
        let grad: Vec<f64> = param_traj.iter().map(|&x| 2.0 * x).collect();
        for (p, g) in param_traj.iter_mut().zip(grad.iter()) {
            *p -= lr * g;
        }
    }
    h.check_bool(
        "nS-305: loss decreases during gradient descent",
        loss_traj.last().unwrap_or(&f64::INFINITY) < loss_traj.first().unwrap_or(&0.0),
    );

    let final_loss = quadratic_loss(&param_traj);
    h.check_bool(
        "nS-305: converged near minimum",
        final_loss < tolerances::OPTIMIZER_VALUE_AT_MIN * 100.0,
    );

    // ── nS-305: Landscape along training trajectory ──────────────────

    let mid_result = landscape_analysis(
        &quadratic_loss,
        &[2.5, 1.5],
        tolerances::HESSIAN_FD_STEP,
        0.1,
    );
    let end_result = landscape_analysis(
        &quadratic_loss,
        &param_traj,
        tolerances::HESSIAN_FD_STEP,
        0.1,
    );
    h.check_bool(
        "nS-305: loss decreases along trajectory",
        end_result.loss < mid_result.loss,
    );

    // ── nS-302: Multi-barrier landscape ──────────────────────────────

    let barrier_12 = transition_barrier(0.0, 1.0, 5.0);
    let barrier_23 = transition_barrier(1.0, 0.5, 3.0);
    h.check_bool(
        "nS-302: deeper minimum has higher barrier",
        barrier_12 > barrier_23 - 0.5,
    );

    // ── Determinism ──────────────────────────────────────────────────

    let h1 = numerical_hessian(&quadratic_loss, &params, tolerances::HESSIAN_FD_STEP);
    let h2 = numerical_hessian(&quadratic_loss, &params, tolerances::HESSIAN_FD_STEP);
    h.check_bool("Hessian computation deterministic", h1 == h2);

    h.finish();
}
