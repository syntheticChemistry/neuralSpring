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
//! Reference: `SciPy` 1.15.3 `scipy.stats.chi2` (Python 3.10.12, 2026-02-16)
//! Analytical: moments and statistic derived from textbook definitions.

use barracuda::special::{
    chi_squared_cdf, chi_squared_mean, chi_squared_mode, chi_squared_pdf, chi_squared_statistic,
    chi_squared_test, chi_squared_variance,
};
use neural_spring::require;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_chi_squared");

    validate_pdf(&mut h);
    validate_cdf(&mut h);
    validate_moments(&mut h);
    validate_test_statistic(&mut h);

    h.finish();
}

fn validate_pdf(h: &mut ValidationHarness) {
    let pdf = require!(h, chi_squared_pdf(2.0, 3.0), "chi_squared_pdf(2,3)");
    h.check_abs(
        "chi_squared_pdf(2.0, 3.0)",
        pdf,
        0.207_553_7,
        tolerances::SPECIAL_FUNCTION_F64,
    );

    let pdf = require!(h, chi_squared_pdf(0.0, 3.0), "chi_squared_pdf(0,3)");
    h.check_abs(
        "chi_squared_pdf(0.0, 3.0)",
        pdf,
        0.0,
        tolerances::SPECIAL_FUNCTION_F64,
    );

    let pdf = require!(h, chi_squared_pdf(5.0, 1.0), "chi_squared_pdf(5,1)");
    h.check_abs(
        "chi_squared_pdf(5.0, 1.0)",
        pdf,
        0.014_644_982_561_926,
        tolerances::SPECIAL_FUNCTION_F64,
    );
}

fn validate_cdf(h: &mut ValidationHarness) {
    let cdf = require!(h, chi_squared_cdf(3.84, 1.0), "chi_squared_cdf(3.84,1)");
    h.check_abs(
        "chi_squared_cdf(3.84, 1.0)",
        cdf,
        0.949_956_478_75,
        tolerances::SPECIAL_FUNCTION_F64,
    );

    let cdf = require!(h, chi_squared_cdf(5.99, 2.0), "chi_squared_cdf(5.99,2)");
    h.check_abs(
        "chi_squared_cdf(5.99, 2.0)",
        cdf,
        0.949_963_372_91,
        tolerances::SPECIAL_FUNCTION_F64,
    );

    let cdf = require!(h, chi_squared_cdf(0.0, 5.0), "chi_squared_cdf(0,5)");
    h.check_abs(
        "chi_squared_cdf(0.0, 5.0)",
        cdf,
        0.0,
        tolerances::SPECIAL_FUNCTION_F64,
    );
}

fn validate_moments(h: &mut ValidationHarness) {
    h.check_abs(
        "chi_squared_mean(5.0)",
        chi_squared_mean(5.0),
        5.0,
        f64::EPSILON,
    );

    h.check_abs(
        "chi_squared_variance(5.0)",
        chi_squared_variance(5.0),
        10.0,
        f64::EPSILON,
    );

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
    let chi2 = require!(
        h,
        chi_squared_statistic(&observed, &expected),
        "chi_squared_statistic"
    );
    h.check_abs(
        "chi_squared_statistic",
        chi2,
        2.0,
        tolerances::SPECIAL_FUNCTION_F64,
    );

    let (stat, p_value, df) = require!(
        h,
        chi_squared_test(&observed, &expected),
        "chi_squared_test"
    );
    h.check_abs(
        "chi_squared_test: stat",
        stat,
        2.0,
        tolerances::SPECIAL_FUNCTION_F64,
    );
    h.check_bool("chi_squared_test: p_value > 0.1", p_value > 0.1);
    h.check_bool("chi_squared_test: df == 2", df == 2);
}
