// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation: `barracuda::special::chi_squared` CPU functions.
//!
//! Validates chi-squared distribution functions against known analytical
//! values and `SciPy` reference outputs. Used in Paper 024 (pangenome
//! selection) for goodness-of-fit testing.
//!
//! ## Provenance
//!
//! Upstream: `barracuda::special::chi_squared::{chi_squared_pdf, chi_squared_cdf, ...}`
//! Reference: `SciPy` `scipy.stats.chi2`

#![allow(clippy::cast_precision_loss, clippy::expect_used)]

use barracuda::special::{
    chi_squared_cdf, chi_squared_mean, chi_squared_mode, chi_squared_pdf, chi_squared_statistic,
    chi_squared_test, chi_squared_variance,
};
use neural_spring::validation::ValidationHarness;

const TOL: f64 = 1e-4;

fn main() {
    let mut h = ValidationHarness::new("barracuda_chi_squared");

    validate_pdf(&mut h);
    validate_cdf(&mut h);
    validate_moments(&mut h);
    validate_test_statistic(&mut h);

    h.finish();
}

fn validate_pdf(h: &mut ValidationHarness) {
    // chi_squared_pdf(x=2.0, k=3.0) ≈ 0.2075537 (SciPy reference)
    let pdf = chi_squared_pdf(2.0, 3.0).expect("chi_squared_pdf(2,3)");
    h.check_abs("chi_squared_pdf(2.0, 3.0)", pdf, 0.207_553_7, TOL);

    // chi_squared_pdf(x=0.0, k=3.0) == 0.0 (x=0 with k>2)
    let pdf = chi_squared_pdf(0.0, 3.0).expect("chi_squared_pdf(0,3)");
    h.check_abs("chi_squared_pdf(0.0, 3.0)", pdf, 0.0, TOL);

    // chi_squared_pdf(x=5.0, k=1.0) ≈ 0.01464
    let pdf = chi_squared_pdf(5.0, 1.0).expect("chi_squared_pdf(5,1)");
    h.check_abs("chi_squared_pdf(5.0, 1.0)", pdf, 0.014_64, TOL);
}

fn validate_cdf(h: &mut ValidationHarness) {
    // chi_squared_cdf(x=3.84, k=1.0) ≈ 0.95 (standard chi-squared critical value)
    let cdf = chi_squared_cdf(3.84, 1.0).expect("chi_squared_cdf(3.84,1)");
    h.check_abs("chi_squared_cdf(3.84, 1.0)", cdf, 0.95, TOL);

    // chi_squared_cdf(x=5.99, k=2.0) ≈ 0.95
    let cdf = chi_squared_cdf(5.99, 2.0).expect("chi_squared_cdf(5.99,2)");
    h.check_abs("chi_squared_cdf(5.99, 2.0)", cdf, 0.95, TOL);

    // chi_squared_cdf(x=0.0, k=any) == 0.0
    let cdf = chi_squared_cdf(0.0, 5.0).expect("chi_squared_cdf(0,5)");
    h.check_abs("chi_squared_cdf(0.0, 5.0)", cdf, 0.0, TOL);
}

fn validate_moments(h: &mut ValidationHarness) {
    // chi_squared_mean(k=5.0) == 5.0
    h.check_abs(
        "chi_squared_mean(5.0)",
        chi_squared_mean(5.0),
        5.0,
        f64::EPSILON,
    );

    // chi_squared_variance(k=5.0) == 10.0
    h.check_abs(
        "chi_squared_variance(5.0)",
        chi_squared_variance(5.0),
        10.0,
        f64::EPSILON,
    );

    // chi_squared_mode(k=5.0) == 3.0
    h.check_abs(
        "chi_squared_mode(5.0)",
        chi_squared_mode(5.0),
        3.0,
        f64::EPSILON,
    );
}

fn validate_test_statistic(h: &mut ValidationHarness) {
    let observed = [20.0, 30.0, 50.0];
    let expected = [25.0, 25.0, 50.0];
    // chi2 = (20-25)²/25 + (30-25)²/25 + (50-50)²/50 = 1.0 + 1.0 + 0.0 = 2.0
    let chi2 = chi_squared_statistic(&observed, &expected).expect("chi_squared_statistic");
    h.check_abs("chi_squared_statistic", chi2, 2.0, TOL);

    let (stat, p_value, df) = chi_squared_test(&observed, &expected).expect("chi_squared_test");
    h.check_abs("chi_squared_test: stat", stat, 2.0, TOL);
    h.check_bool("chi_squared_test: p_value > 0.1", p_value > 0.1);
    h.check_bool("chi_squared_test: df == 2", df == 2);
}
