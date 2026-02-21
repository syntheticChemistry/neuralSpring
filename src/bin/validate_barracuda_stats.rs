// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `barracuda::stats` CPU primitives.
//!
//! Validates variance, `std_dev`, `pearson_correlation`, covariance, Spearman
//! correlation, `norm_cdf`, and `norm_pdf` against analytically known values.
//!
//! ## Provenance
//!
//! Expected values: analytical (pure math, textbook definitions).
//! Cross-validated against `NumPy` 2.2.6 / `SciPy` 1.15.3.

use barracuda::stats::correlation;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_stats");

    validate_descriptive(&mut h);
    validate_distribution(&mut h);

    h.finish();
}

fn validate_descriptive(h: &mut ValidationHarness) {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // --- Variance ---
    // Var([1,2,3,4,5]) = 10/4 = 2.5 (sample variance, ddof=1)
    let var = correlation::variance(&x);
    check_result(
        h,
        "variance([1..5]) == 2.5",
        var,
        2.5,
        tolerances::METRIC_EXACT,
    );

    // Var([c,c,c]) = 0 for any constant
    let constant = vec![7.0, 7.0, 7.0, 7.0];
    let var_const = correlation::variance(&constant);
    check_result(
        h,
        "variance(constant) == 0",
        var_const,
        0.0,
        tolerances::METRIC_EXACT,
    );

    // --- Std Dev ---
    let sd = correlation::std_dev(&x);
    check_result(
        h,
        "std_dev([1..5]) == sqrt(2.5)",
        sd,
        2.5_f64.sqrt(),
        tolerances::METRIC_EXACT,
    );

    // --- Pearson Correlation ---
    let y_perfect = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let r_pos = barracuda::stats::pearson_correlation(&x, &y_perfect);
    check_result(
        h,
        "pearson(x, 2x) == 1.0",
        r_pos,
        1.0,
        tolerances::CROSS_LANGUAGE,
    );

    let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
    let r_neg = barracuda::stats::pearson_correlation(&x, &y_neg);
    check_result(
        h,
        "pearson(x, -x+c) == -1.0",
        r_neg,
        -1.0,
        tolerances::CROSS_LANGUAGE,
    );

    let x_orth = vec![1.0, -1.0, 1.0, -1.0];
    let y_orth = vec![1.0, 1.0, -1.0, -1.0];
    let r_zero = barracuda::stats::pearson_correlation(&x_orth, &y_orth);
    check_result(
        h,
        "pearson(orthogonal) == 0",
        r_zero,
        0.0,
        tolerances::CROSS_LANGUAGE,
    );

    // --- Covariance ---
    // Cov(X, 2X) = 2*Var(X) = 5.0
    let cov = barracuda::stats::covariance(&x, &y_perfect);
    check_result(h, "cov(x, 2x) == 5.0", cov, 5.0, tolerances::METRIC_EXACT);

    // --- Spearman ---
    let y_mono = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // x²
    let r_spearman = correlation::spearman_correlation(&x, &y_mono);
    check_result(
        h,
        "spearman(x, x²) == 1.0",
        r_spearman,
        1.0,
        tolerances::CROSS_LANGUAGE,
    );
}

fn validate_distribution(h: &mut ValidationHarness) {
    let cdf_zero = barracuda::stats::norm_cdf(0.0);
    h.check_abs(
        "norm_cdf(0) == 0.5",
        cdf_zero,
        0.5,
        tolerances::CROSS_LANGUAGE,
    );

    // norm_cdf(1.96) ≈ 0.9750021 (97.5 percentile)
    let cdf_196 = barracuda::stats::norm_cdf(1.96);
    h.check_abs(
        "norm_cdf(1.96) ≈ 0.975",
        cdf_196,
        0.975_002_104_859_278,
        tolerances::SPECIAL_FUNCTION_F64,
    );

    // norm_pdf(0) = 1/sqrt(2π)
    let pdf_zero = barracuda::stats::norm_pdf(0.0);
    let expected_pdf = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    h.check_abs(
        "norm_pdf(0) == 1/√(2π)",
        pdf_zero,
        expected_pdf,
        tolerances::CROSS_LANGUAGE,
    );

    // norm_ppf(0.5) = 0 (median of standard normal)
    let ppf_half = barracuda::stats::norm_ppf(0.5);
    h.check_abs(
        "norm_ppf(0.5) == 0",
        ppf_half,
        0.0,
        tolerances::SPECIAL_FUNCTION_F64,
    );

    // norm_ppf(0.975) ≈ 1.96
    let ppf_975 = barracuda::stats::norm_ppf(0.975);
    h.check_abs("norm_ppf(0.975) ≈ 1.96", ppf_975, 1.96, 0.01);
}

fn check_result(
    h: &mut ValidationHarness,
    label: &str,
    result: Result<f64, barracuda::error::BarracudaError>,
    expected: f64,
    tolerance: f64,
) {
    match result {
        Ok(val) => h.check_abs(label, val, expected, tolerance),
        Err(e) => {
            h.check_bool(&format!("{label} [ERROR: {e}]"), false);
        }
    }
}
