// SPDX-License-Identifier: AGPL-3.0-or-later

// groundSpring provenance: bootstrap CIs and normal distribution helpers.

use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, bench_once};

pub fn validate_groundspring_bootstrap(h: &mut ValidationHarness) {
    println!("\n─── groundSpring provenance: bootstrap + sampling ───\n");

    // Bootstrap CI (groundSpring → BarraCUDA)
    let data: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();
    let true_mean = data.iter().sum::<f64>() / data.len() as f64;
    let mean_fn = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let (ci_result, _) = bench_once("bootstrap_ci (gS→BarraCUDA)", || {
        barracuda::stats::bootstrap_ci(&data, mean_fn, 500, 0.95, 42)
    });
    let ci = ci_result.expect("bootstrap_ci");
    h.check_bool("gS→bootstrap: CI lower < upper", ci.lower < ci.upper);
    h.check_bool(
        "gS→bootstrap: CI contains estimate",
        ci.lower <= ci.estimate && ci.estimate <= ci.upper,
    );

    // Bootstrap mean (groundSpring → BarraCUDA)
    let (bm_result, _) = bench_once("bootstrap_mean (gS→BarraCUDA)", || {
        barracuda::stats::bootstrap_mean(&data, 500, 0.95, 42)
    });
    let bm = bm_result.expect("bootstrap_mean").estimate;
    h.check_abs(
        "gS→bootstrap: mean ≈ true mean",
        bm,
        true_mean,
        tolerances::ODE_STEADY_STATE_SLACK,
    );

    // rawr_mean (groundSpring → BarraCUDA)
    let (rm_result, _) = bench_once("rawr_mean (gS→BarraCUDA)", || {
        barracuda::stats::rawr_mean(&data, 200, 0.95, 42)
    });
    let rm = rm_result.expect("rawr_mean").estimate;
    h.check_abs(
        "gS→bootstrap: rawr_mean ≈ true mean",
        rm,
        true_mean,
        tolerances::ODE_STEADY_STATE_SLACK,
    );

    // Normal distribution (groundSpring/airSpring → BarraCUDA)
    let (cdf, _) = bench_once("norm_cdf (gS→BarraCUDA)", || {
        barracuda::stats::norm_cdf(0.0)
    });
    h.check_abs(
        "gS→normal: Φ(0) = 0.5",
        cdf,
        0.5,
        tolerances::CROSS_LANGUAGE,
    );

    let (pdf, _) = bench_once("norm_pdf (gS→BarraCUDA)", || {
        barracuda::stats::norm_pdf(0.0)
    });
    let expected_pdf = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    h.check_abs(
        "gS→normal: φ(0)",
        pdf,
        expected_pdf,
        tolerances::CROSS_LANGUAGE,
    );

    let (ppf, _) = bench_once("norm_ppf (gS→BarraCUDA)", || {
        barracuda::stats::norm_ppf(0.975)
    });
    h.check_abs(
        "gS→normal: Φ⁻¹(0.975) ≈ 1.96",
        ppf,
        1.96,
        tolerances::NORM_PPF_TAIL,
    );
}
