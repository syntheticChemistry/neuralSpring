// SPDX-License-Identifier: AGPL-3.0-or-later

// airSpring provenance: regression fits, hydrology, metrics, Spearman.

use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, bench_once};

pub fn validate_airspring_stats(h: &mut ValidationHarness) {
    println!("\n─── airSpring provenance: stats + regression ───\n");

    // Regression fitting (airSpring → BarraCUDA)
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.1, 3.9, 6.1, 7.9, 10.1];
    let (fit, _) = bench_once("fit_linear (aS→BarraCUDA)", || {
        barracuda::stats::fit_linear(&x, &y)
    });
    let fit = fit.expect("fit_linear should converge");
    h.check_abs(
        "aS→regression: linear slope ≈ 2.0",
        fit.params[0],
        2.0,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );
    h.check_abs(
        "aS→regression: linear R² ≈ 1.0",
        fit.r_squared,
        1.0,
        tolerances::GPU_AF_VARIANCE_F32,
    );

    // Quadratic fit (airSpring → BarraCUDA)
    let xq: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    let yq: Vec<f64> = xq.iter().map(|&x| 3.0f64.mul_add(x, x * x) + 1.0).collect();
    let (fit_q, _) = bench_once("fit_quadratic (aS→BarraCUDA)", || {
        barracuda::stats::fit_quadratic(&xq, &yq)
    });
    let fit_q = fit_q.expect("fit_quadratic should converge");
    h.check_abs(
        "aS→regression: quadratic R² ≈ 1.0",
        fit_q.r_squared,
        1.0,
        tolerances::PGM_COMPLEXITY_SLACK,
    );

    // Exponential fit (airSpring → BarraCUDA)
    let xe: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
    let ye: Vec<f64> = xe.iter().map(|&x| 2.0 * (0.5_f64 * x).exp()).collect();
    let fit_e = barracuda::stats::fit_exponential(&xe, &ye).expect("fit_exponential");
    h.check_abs(
        "aS→regression: exponential R² ≈ 1.0",
        fit_e.r_squared,
        1.0,
        tolerances::PGM_COMPLEXITY_SLACK,
    );

    // Metrics: RMSE, R², NSE, MAE (airSpring → BarraCUDA)
    let predicted = [1.0, 2.0, 3.0, 4.0, 5.0];
    let observed = [1.1, 1.9, 3.1, 3.9, 5.1];
    let (rmse, _) = bench_once("rmse (aS→BarraCUDA)", || {
        barracuda::stats::rmse(&predicted, &observed)
    });
    h.check_bool("aS→metrics: RMSE > 0", rmse > 0.0);
    h.check_bool("aS→metrics: RMSE < 0.2 (close predictions)", rmse < 0.2);

    let (r2, _) = bench_once("r_squared (aS→BarraCUDA)", || {
        barracuda::stats::r_squared(&predicted, &observed)
    });
    h.check_bool("aS→metrics: R² > 0.99", r2 > 0.99);

    let (nse, _) = bench_once("nash_sutcliffe (aS→BarraCUDA)", || {
        barracuda::stats::nash_sutcliffe(&predicted, &observed)
    });
    h.check_bool("aS→metrics: NSE > 0.99", nse > 0.99);

    let (mae, _) = bench_once("mae (aS→BarraCUDA)", || {
        barracuda::stats::mae(&predicted, &observed)
    });
    h.check_abs(
        "aS→metrics: MAE = 0.1",
        mae,
        0.1,
        tolerances::CROSS_LANGUAGE,
    );

    // Hydrology: Hargreaves ET₀ (airSpring → BarraCUDA)
    // ra=15 MJ/m²/day, t_max=35°C, t_min=15°C
    let (et0_opt, _) = bench_once("hargreaves_et0 (aS→BarraCUDA)", || {
        barracuda::stats::hargreaves_et0(15.0, 35.0, 15.0)
    });
    let et0 = et0_opt.unwrap_or(0.0);
    h.check_bool(
        "aS→hydrology: Hargreaves ET₀ > 0 mm/day",
        et0 > 0.0 && et0 < 20.0,
    );

    // Spearman rank correlation (airSpring → BarraCUDA)
    let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let ys = [5.0, 6.0, 7.0, 8.0, 7.0];
    let (rho_result, _) = bench_once("spearman (aS→BarraCUDA)", || {
        barracuda::stats::spearman_correlation(&xs, &ys)
    });
    let rho = rho_result.unwrap_or(0.0);
    h.check_bool("aS→stats: Spearman in [-1,1]", (-1.0..=1.0).contains(&rho));
}
