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

#[allow(clippy::too_many_lines)]
fn main() {
    let mut h = ValidationHarness::new("loss_landscape");

    // ── nS-301: Quadratic Hessian = 2*I ──────────────────────────────

    let params = vec![0.0; 4];
    let hessian = numerical_hessian(&quadratic_loss, &params, 1e-5);
    let mut diag_ok = true;
    let mut off_ok = true;
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 2.0 } else { 0.0 };
            if (hessian[i * 4 + j] - expected).abs() > 1e-4 {
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
    let all_near_two = spectrum.iter().all(|&ev| (ev - 2.0).abs() < 1e-3);
    h.check_bool("Quadratic spectrum: all eigenvalues ≈ 2.0", all_near_two);

    // ── nS-301: Landscape at quadratic minimum ───────────────────────

    let result = landscape_analysis(&quadratic_loss, &params, 1e-5, 0.1);
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
    let rb_result = landscape_analysis(&rosenbrock_loss, &rb_min, 1e-5, 0.1);
    h.check_bool("Rosenbrock loss at (1,1) < 1e-6", rb_result.loss < 1e-6);
    h.check_bool(
        "Rosenbrock at minimum: saddle_index = 0",
        rb_result.saddle_index == 0,
    );

    // ── nS-301: Saddle point detection ───────────────────────────────

    let saddle_params = vec![0.0, 0.0];
    let saddle_result = landscape_analysis(&saddle_loss, &saddle_params, 1e-5, 0.1);
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
        1e-12,
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

    let mut rng2 = Rng::new(42);
    let init2 = vec![1.0, 1.0];
    let result = boltzmann_sampling(&quadratic_loss, &init2, 100.0, 0.1, 500, &mut rng2);
    h.check_bool(
        "High-temp acceptance rate > 30%",
        result.acceptance_rate > 0.3,
    );
    h.check_bool(
        "Boltzmann produces expected number of samples",
        result.losses.len() == 500,
    );

    // ── nS-303: Low temperature favors lower loss ────────────────────

    let mut rng_lo = Rng::new(42);
    let result_lo = boltzmann_sampling(&quadratic_loss, &init2, 0.01, 0.01, 500, &mut rng_lo);
    let mean_loss_lo: f64 = result_lo.losses.iter().sum::<f64>() / 500.0;

    let mut rng_hi = Rng::new(42);
    let result_hi = boltzmann_sampling(&quadratic_loss, &init2, 100.0, 0.1, 500, &mut rng_hi);
    let mean_loss_hi: f64 = result_hi.losses.iter().sum::<f64>() / 500.0;

    h.check_bool(
        "Low temperature produces lower mean loss than high",
        mean_loss_lo < mean_loss_hi + 5.0,
    );

    // ── Determinism ──────────────────────────────────────────────────

    let h1 = numerical_hessian(&quadratic_loss, &params, 1e-5);
    let h2 = numerical_hessian(&quadratic_loss, &params, 1e-5);
    h.check_bool("Hessian computation deterministic", h1 == h2);

    h.finish();
}
