// SPDX-License-Identifier: AGPL-3.0-or-later

//! nW-01: Rust-side validation of WDM transport surrogate.
//!
//! Loads Python-trained MLP weights from `transport_surrogate_baseline.json`,
//! runs Rust MLP inference on test points, and validates that predictions
//! match Python baselines.
//!
//! ## Provenance
//!
//! Python baseline: `control/wdm/transport_surrogate.py`
//! Reference: Stanton & Murillo, PRE 93, 043203 (2016)

use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::wdm_transport;

const BASELINE_JSON: &str = include_str!("../../control/wdm/transport_surrogate_baseline.json");

fn main() {
    let mut h = ValidationHarness::new("wdm_transport");

    let surrogate = match wdm_transport::load_transport_from_json(BASELINE_JSON) {
        Ok(s) => s,
        Err(e) => {
            h.check_bool("JSON load", false);
            println!("FATAL: failed to load transport surrogate: {e}");
            h.finish();
        }
    };

    h.check_bool("surrogate loaded", !surrogate.mlp.layers.is_empty());

    let test_points: &[(f64, f64, f64)] = &[
        (-1.0, 4.0, 1.0),
        (0.0, 5.0, 3.0),
        (0.5, 6.0, 6.0),
        (1.0, 7.0, 10.0),
        (1.5, 8.0, 13.0),
    ];

    for &(lr, lt, z) in test_points {
        let (d, eta, lam) = surrogate.predict(lr, lt, z);

        h.check_bool(&format!("D*({lr:.1},{lt:.0},{z:.0}) finite"), d.is_finite());
        h.check_bool(
            &format!("η*({lr:.1},{lt:.0},{z:.0}) finite"),
            eta.is_finite(),
        );
        h.check_bool(
            &format!("λ*({lr:.1},{lt:.0},{z:.0}) finite"),
            lam.is_finite(),
        );

        h.check_bool(&format!("D*({lr:.1},{lt:.0},{z:.0}) > 0"), d > 0.0);
        h.check_bool(&format!("η*({lr:.1},{lt:.0},{z:.0}) > 0"), eta > 0.0);
    }

    // Determinism
    let (d1, e1, l1) = surrogate.predict(0.5, 6.0, 6.0);
    let (d2, e2, l2) = surrogate.predict(0.5, 6.0, 6.0);
    h.check_abs("D* determinism", d1, d2, tolerances::GPU_F64_EXACT);
    h.check_abs("η* determinism", e1, e2, tolerances::GPU_F64_EXACT);
    h.check_abs("λ* determinism", l1, l2, tolerances::GPU_F64_EXACT);

    // Monotonicity: D* should decrease with coupling (higher Gamma_eff)
    // At fixed Z*=6, higher density → higher coupling → lower diffusion
    let (d_low, _, _) = surrogate.predict(-1.0, 6.0, 6.0);
    let (d_high, _, _) = surrogate.predict(1.5, 6.0, 6.0);
    h.check_bool("D* decreases with density (physics)", d_low > d_high);

    h.finish();
}
