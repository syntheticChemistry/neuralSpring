// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: transformer primitives (softmax, GELU).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/transformer/transformer_inference.py`
//! Run: 2026-02-16, southGate (Ryzen 7 5800X3D), Python 3.10, `NumPy` 2.2.6, seed=42
//! Command: `python3 control/transformer/transformer_inference.py`
//! Reference: [`TRANSFORMER_PROVENANCE`](neural_spring::provenance::TRANSFORMER_PROVENANCE)

use neural_spring::provenance;
use neural_spring::tolerances;
use neural_spring::transformer::{gelu, softmax};
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("transformer");

    // --- Softmax properties ---

    let uniform = softmax(&[1.0, 1.0, 1.0, 1.0]);
    h.check_abs(
        "softmax uniform [1,1,1,1][0] == 0.25",
        uniform[0],
        0.25,
        tolerances::SOFTMAX_SUM,
    );

    let s = softmax(&[1.0, 2.0, 3.0]);
    let sum: f64 = s.iter().sum();
    h.check_abs("softmax sums to 1", sum, 1.0, tolerances::SOFTMAX_SUM);

    h.check_bool("softmax preserves ordering", s[0] < s[1] && s[1] < s[2]);

    // --- Softmax cross-language: Rust vs Python ---

    let s5 = softmax(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    for (i, &expected) in provenance::SOFTMAX_1_TO_5.iter().enumerate() {
        h.check_abs(
            &format!("softmax([1..5])[{i}]"),
            s5[i],
            expected,
            tolerances::SOFTMAX_CROSS_PYTHON,
        );
    }

    // Numerical stability with large inputs
    let s_big = softmax(&[1e10, 1e10 + 1.0, 1e10 + 2.0]);
    let sum_big: f64 = s_big.iter().sum();
    h.check_abs(
        "softmax stable (large inputs)",
        sum_big,
        1.0,
        tolerances::SOFTMAX_SUM,
    );

    // --- GELU properties ---

    h.check_abs(
        "GELU(0) == 0",
        gelu(0.0),
        0.0,
        tolerances::GELU_CROSS_PYTHON,
    );

    h.check_abs(
        "GELU(10) ≈ 10",
        gelu(10.0),
        10.0,
        tolerances::GELU_LARGE_INPUT,
    );

    h.check_bool(
        "GELU positive for x > 0",
        gelu(0.5) > 0.0 && gelu(1.0) > 0.0 && gelu(5.0) > 0.0,
    );

    // --- GELU cross-language: Rust vs Python ---

    for &(x, expected) in &provenance::GELU_REFERENCE {
        h.check_abs_or_rel(
            &format!("GELU({x})"),
            gelu(x),
            expected,
            tolerances::GELU_CROSS_PYTHON,
        );
    }

    h.finish();
}
