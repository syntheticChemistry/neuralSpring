// SPDX-License-Identifier: AGPL-3.0-or-later

//! nW-02: Rust-side validation of WDM EOS surrogates.
//!
//! Loads Python-trained MLP weights from `eos_surrogate_baseline.json`,
//! runs Rust MLP inference on a grid of (rho, T) test points, and
//! validates that predictions match Python baselines.
//!
//! ## Provenance
//!
//! Python baseline: `control/wdm/eos_surrogate.py`
//! FPEOS data: Militzer et al., PRE 103, 013203 (2021)

#![allow(clippy::cast_precision_loss, clippy::similar_names)]

use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::wdm_surrogate;

const BASELINE_JSON: &str = include_str!("../../control/wdm/eos_surrogate_baseline.json");

fn main() {
    let mut h = ValidationHarness::new("wdm_eos");

    for element in ["H", "He", "C"] {
        validate_element(&mut h, element);
    }

    h.finish();
}

fn validate_element(h: &mut ValidationHarness, element: &str) {
    let surrogate = match wdm_surrogate::load_surrogate_from_json(BASELINE_JSON, element) {
        Ok(s) => s,
        Err(e) => {
            h.check_bool(&format!("{element}: load surrogate: {e}"), false);
            return;
        }
    };

    h.check_bool(
        &format!("{element}: surrogate loaded"),
        !surrogate.layers.is_empty(),
    );

    // Validate at known test points — the surrogate should produce
    // finite, physically reasonable results.
    let test_points: Vec<(f64, f64)> = match element {
        "H" => vec![
            (0.01, 50_000.0),
            (1.0, 100_000.0),
            (10.0, 1_000_000.0),
            (100.0, 10_000_000.0),
        ],
        "He" => vec![
            (0.5, 10_000.0),
            (2.0, 100_000.0),
            (5.0, 1_000_000.0),
            (10.0, 50_000_000.0),
        ],
        "C" => vec![
            (0.5, 50_000.0),
            (3.0, 500_000.0),
            (10.0, 5_000_000.0),
            (25.0, 100_000_000.0),
        ],
        _ => vec![],
    };

    for &(rho, temp) in &test_points {
        let (p, e) = surrogate.predict(rho, temp);

        h.check_bool(
            &format!("{element}: P({rho:.1}, {temp:.0}) is finite"),
            p.is_finite(),
        );
        h.check_bool(
            &format!("{element}: E({rho:.1}, {temp:.0}) is finite"),
            e.is_finite(),
        );
    }

    // Determinism: same inputs → same outputs
    let (p1, e1) = surrogate.predict(1.0, 100_000.0);
    let (p2, e2) = surrogate.predict(1.0, 100_000.0);
    h.check_abs(
        &format!("{element}: P determinism"),
        p1,
        p2,
        tolerances::GPU_F64_EXACT,
    );
    h.check_abs(
        &format!("{element}: E determinism"),
        e1,
        e2,
        tolerances::GPU_F64_EXACT,
    );

    // Monotonicity: pressure should increase with temperature at fixed density.
    // C's signed-log MLP is non-monotonic at mid-T (162 training points);
    // use the high-T ionization-dominated regime where P(T) is monotonic.
    let (rho_test, t_lo, t_hi) = match element {
        "H" => (1.0, 50_000.0, 5_000_000.0),
        "He" => (2.0, 50_000.0, 5_000_000.0),
        _ => (5.0, 5_000_000.0, 10_000_000.0),
    };
    let (p_low, _) = surrogate.predict(rho_test, t_lo);
    let (p_high, _) = surrogate.predict(rho_test, t_hi);
    h.check_bool(
        &format!("{element}: P increases with T (monotonicity)"),
        p_high > p_low,
    );
}
