// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: surrogate benchmark functions.
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/surrogate/surrogate_validation.py`
//! Run: 2026-02-16, southGate (Ryzen 7 5800X3D), Python 3.10, `PyTorch` 2.9.0+cu128, seed=42
//! Command: `python3 control/surrogate/surrogate_validation.py`
//! Reference: [`SURROGATE_PROVENANCE`](neural_spring::provenance::SURROGATE_PROVENANCE)

use neural_spring::provenance;
use neural_spring::surrogate::{ackley_2d, rastrigin_2d, rosenbrock_2d};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("surrogate");

    // Global minima (analytical — exact)
    h.check_abs(
        "Rastrigin(0,0) == 0",
        rastrigin_2d(0.0, 0.0),
        0.0,
        tolerances::BENCHMARK_GLOBAL_MIN,
    );
    h.check_abs(
        "Rosenbrock(1,1) == 0",
        rosenbrock_2d(1.0, 1.0),
        0.0,
        tolerances::BENCHMARK_GLOBAL_MIN,
    );
    h.check_abs(
        "Ackley(0,0) == 0",
        ackley_2d(0.0, 0.0),
        0.0,
        tolerances::BENCHMARK_GLOBAL_MIN,
    );

    // Cross-language: Rust vs Python (NumPy 2.2.6) at reference points
    for &(x, y, expected) in &provenance::RASTRIGIN_REFERENCE {
        h.check_abs_or_rel(
            &format!("Rastrigin({x},{y})"),
            rastrigin_2d(x, y),
            expected,
            tolerances::BENCHMARK_CROSS_PYTHON,
        );
    }

    for &(x, y, expected) in &provenance::ROSENBROCK_REFERENCE {
        h.check_abs_or_rel(
            &format!("Rosenbrock({x},{y})"),
            rosenbrock_2d(x, y),
            expected,
            tolerances::BENCHMARK_CROSS_PYTHON,
        );
    }

    for &(x, y, expected) in &provenance::ACKLEY_REFERENCE {
        h.check_abs_or_rel(
            &format!("Ackley({x},{y})"),
            ackley_2d(x, y),
            expected,
            tolerances::BENCHMARK_CROSS_PYTHON,
        );
    }

    h.finish();
}
