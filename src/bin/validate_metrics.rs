// SPDX-License-Identifier: AGPL-3.0-only

//! Validation binary: statistical metrics (R², RMSE, MAE, NSE).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Expected values: analytically derived (pure arithmetic, no iteration).
//! Verified via `python3 -c` one-liners against `NumPy` 2.2.6.
//! Reference: [`METRICS_REFS`](neural_spring::provenance::METRICS_REFS)

use neural_spring::metrics;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("metrics");

    let y = [1.0, 2.0, 3.0, 4.0, 5.0];

    // --- R² ---
    h.check_abs(
        "R² perfect prediction",
        metrics::r_squared(&y, &y),
        1.0,
        tolerances::METRIC_EXACT,
    );

    let y_mean = [3.0; 5];
    h.check_abs(
        "R² mean prediction == 0",
        metrics::r_squared(&y, &y_mean),
        0.0,
        tolerances::METRIC_EXACT,
    );

    let y_bad = [10.0, 20.0, 30.0, 40.0, 50.0];
    h.check_bool(
        "R² worse-than-mean < 0",
        metrics::r_squared(&y, &y_bad) < 0.0,
    );

    // SS_res = 0.75, SS_tot = 2.0 => R² = 0.625
    let yt3 = [1.0, 2.0, 3.0];
    let yp3 = [1.5, 2.5, 3.5];
    h.check_abs(
        "R² known value 0.625",
        metrics::r_squared(&yt3, &yp3),
        0.625,
        tolerances::METRIC_EXACT,
    );

    // --- RMSE ---
    h.check_abs(
        "RMSE zero error",
        metrics::rmse(&y, &y),
        0.0,
        tolerances::METRIC_EXACT,
    );

    let y_off = [1.1, 2.1, 3.1, 4.1, 5.1];
    h.check_abs(
        "RMSE constant offset 0.1",
        metrics::rmse(&y, &y_off),
        0.1,
        tolerances::METRIC_EXACT,
    );

    // MSE = (9+16)/2 = 12.5 => RMSE = sqrt(12.5)
    let a = [0.0, 0.0];
    let b = [3.0, 4.0];
    h.check_abs(
        "RMSE known sqrt(12.5)",
        metrics::rmse(&a, &b),
        12.5_f64.sqrt(),
        tolerances::METRIC_EXACT,
    );

    // --- MAE ---
    h.check_abs(
        "MAE zero error",
        metrics::mae(&y, &y),
        0.0,
        tolerances::METRIC_EXACT,
    );

    let y_off1 = [2.0, 3.0, 4.0, 5.0, 6.0];
    h.check_abs(
        "MAE constant offset 1.0",
        metrics::mae(&y, &y_off1),
        1.0,
        tolerances::METRIC_EXACT,
    );

    // --- NSE ---
    h.check_abs(
        "NSE == R² delegation",
        metrics::nse(&y, &y_mean),
        metrics::r_squared(&y, &y_mean),
        tolerances::METRIC_EXACT,
    );

    h.finish();
}
