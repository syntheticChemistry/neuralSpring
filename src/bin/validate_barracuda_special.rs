// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `barracuda::special` CPU f64 functions.
//!
//! Validates gamma, factorial, erf/erfc, Bessel, Legendre, Hermite, and
//! Laguerre polynomials against analytical identities and NIST DLMF values.
//!
//! ## Provenance
//!
//! Expected values: exact mathematical identities (A&S, NIST DLMF).
//! Cross-validated against `SciPy` 1.15.3 `scipy.special`.

use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_special");

    validate_gamma(&mut h);
    validate_factorial(&mut h);
    validate_erf(&mut h);
    validate_bessel(&mut h);
    validate_polynomials(&mut h);

    h.finish();
}

fn validate_gamma(h: &mut ValidationHarness) {
    check_gamma(h, "Γ(1) == 1", 1.0, 1.0);
    check_gamma(h, "Γ(5) == 24", 5.0, 24.0);
    check_gamma(h, "Γ(0.5) == √π", 0.5, std::f64::consts::PI.sqrt());
    check_gamma(h, "Γ(1.5) == √π/2", 1.5, std::f64::consts::PI.sqrt() / 2.0);
}

fn validate_factorial(h: &mut ValidationHarness) {
    h.check_abs(
        "0! == 1",
        barracuda::special::factorial(0),
        1.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "5! == 120",
        barracuda::special::factorial(5),
        120.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "10! == 3628800",
        barracuda::special::factorial(10),
        3_628_800.0,
        tolerances::EXACT_F64,
    );
}

fn validate_erf(h: &mut ValidationHarness) {
    h.check_abs(
        "erf(0) == 0",
        barracuda::special::erf(0.0),
        0.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "erf(5) ≈ 1",
        barracuda::special::erf(5.0),
        1.0,
        tolerances::SPECIAL_FUNCTION_F64,
    );
    h.check_abs(
        "erf(1) ≈ 0.8427",
        barracuda::special::erf(1.0),
        0.842_700_792_949_715,
        tolerances::SPECIAL_FUNCTION_F64,
    );
    h.check_abs(
        "erf(-1) == -erf(1)",
        barracuda::special::erf(-1.0),
        -barracuda::special::erf(1.0),
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "erfc(0) == 1",
        barracuda::special::erfc(0.0),
        1.0,
        tolerances::EXACT_F64,
    );
    let x = 1.5;
    h.check_abs(
        "erf(1.5) + erfc(1.5) == 1",
        barracuda::special::erf(x) + barracuda::special::erfc(x),
        1.0,
        tolerances::EXACT_F64,
    );
}

fn validate_bessel(h: &mut ValidationHarness) {
    h.check_abs(
        "J₀(0) == 1",
        barracuda::special::bessel_j0(0.0),
        1.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "J₁(0) == 0",
        barracuda::special::bessel_j1(0.0),
        0.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "I₀(0) == 1",
        barracuda::special::bessel_i0(0.0),
        1.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "J₀(2.4048) ≈ 0",
        barracuda::special::bessel_j0(2.404_825_557_695_773),
        0.0,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_polynomials(h: &mut ValidationHarness) {
    // Legendre
    h.check_abs(
        "P₀(0.5) == 1",
        barracuda::special::legendre(0, 0.5),
        1.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "P₁(0.7) == 0.7",
        barracuda::special::legendre(1, 0.7),
        0.7,
        tolerances::EXACT_F64,
    );
    let x2 = 0.6_f64;
    let p2_expected = (3.0 * x2).mul_add(x2, -1.0) / 2.0;
    h.check_abs(
        "P₂(0.6) analytical",
        barracuda::special::legendre(2, x2),
        p2_expected,
        tolerances::EXACT_F64,
    );

    // Hermite
    h.check_abs(
        "H₀(1.5) == 1",
        barracuda::special::hermite(0, 1.5),
        1.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "H₁(3.0) == 6",
        barracuda::special::hermite(1, 3.0),
        6.0,
        tolerances::EXACT_F64,
    );
    let hx = 2.0;
    h.check_abs(
        "H₂(2) == 14",
        barracuda::special::hermite(2, hx),
        (4.0 * hx).mul_add(hx, -2.0),
        tolerances::EXACT_F64,
    );

    // Laguerre
    h.check_abs(
        "L₀(0,1) == 1",
        barracuda::special::laguerre(0, 0.0, 1.0),
        1.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "L₁(0,0.5) == 0.5",
        barracuda::special::laguerre(1, 0.0, 0.5),
        0.5,
        tolerances::EXACT_F64,
    );
    let lx = 1.0_f64;
    let l2_expected = f64::midpoint(lx.mul_add(lx, -4.0 * lx), 2.0);
    h.check_abs(
        "L₂(0,1) == -0.5",
        barracuda::special::laguerre(2, 0.0, lx),
        l2_expected,
        tolerances::EXACT_F64,
    );
}

fn check_gamma(h: &mut ValidationHarness, label: &str, x: f64, expected: f64) {
    match barracuda::special::gamma(x) {
        Ok(val) => h.check_abs(label, val, expected, tolerances::CROSS_LANGUAGE),
        Err(e) => h.check_bool(&format!("{label} [ERROR: {e}]"), false),
    }
}
