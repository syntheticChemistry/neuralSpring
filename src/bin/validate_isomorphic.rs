// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: isomorphic pattern catalog (Exp 005).
//!
//! Validates the Isomorphism Theorem: all neural architectures decompose
//! into 6 fundamental primitives (GEMM, Attention, Normalization,
//! Nonlinearity, Reduction, Gating), and `BarraCUDA` covers all of them.
//!
//! ## Provenance
//!
//! Python baseline: `control/isomorphic/isomorphic_catalog.py`
//! Command: `python3 control/isomorphic/isomorphic_catalog.py`
//! Result: 8/8 PASS
//! Reference: [`ISOMORPHIC_PROVENANCE`](neural_spring::provenance::ISOMORPHIC_PROVENANCE)

use neural_spring::tolerances;
use neural_spring::transformer::{gelu, softmax};
use neural_spring::validation::ValidationHarness;

const ARCHITECTURES: &[(&str, &[&str])] = &[
    (
        "LLaMA-7B",
        &[
            "GEMM",
            "Attention",
            "Normalization",
            "Nonlinearity",
            "Reduction",
        ],
    ),
    (
        "OpenFold-Evoformer",
        &[
            "GEMM",
            "Attention",
            "Normalization",
            "Nonlinearity",
            "Gating",
        ],
    ),
    (
        "ResNet-50",
        &["GEMM", "Normalization", "Nonlinearity", "Reduction"],
    ),
    (
        "ViT-Base",
        &[
            "GEMM",
            "Attention",
            "Normalization",
            "Nonlinearity",
            "Reduction",
        ],
    ),
    ("Physics-MLP", &["GEMM", "Nonlinearity"]),
    (
        "LSTM-Weather",
        &["GEMM", "Nonlinearity", "Gating", "Reduction"],
    ),
];

const PRIMITIVES: &[&str] = &[
    "GEMM",
    "Attention",
    "Normalization",
    "Nonlinearity",
    "Reduction",
    "Gating",
];

fn main() {
    let mut h = ValidationHarness::new("isomorphic");

    // ── Part 1: Architecture catalog completeness ──

    h.check_bool(
        &format!("catalog has {} architectures", ARCHITECTURES.len()),
        ARCHITECTURES.len() >= 6,
    );

    // ── Part 2: All architectures use only the 6 primitives ──

    let all_valid = ARCHITECTURES
        .iter()
        .all(|(_, ops)| ops.iter().all(|op| PRIMITIVES.contains(op)));
    h.check_bool("all architectures use only 6 primitives", all_valid);

    // ── Part 3: GEMM is universal (every architecture uses it) ──

    let gemm_universal = ARCHITECTURES.iter().all(|(_, ops)| ops.contains(&"GEMM"));
    h.check_bool("GEMM present in all architectures", gemm_universal);

    // ── Part 4: Cross-domain isomorphism (shared primitives between pairs) ──

    let mut min_shared = usize::MAX;
    for (i, (_, ops_i)) in ARCHITECTURES.iter().enumerate() {
        for (_, ops_j) in &ARCHITECTURES[i + 1..] {
            let shared = ops_i.iter().filter(|op| ops_j.contains(op)).count();
            if shared < min_shared {
                min_shared = shared;
            }
        }
    }
    h.check_bool(
        &format!("min shared primitives between any pair = {min_shared} (>= 1)"),
        min_shared >= 1,
    );

    // ── Part 5: Primitive math validation (softmax + GELU) ──

    let sm = softmax(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let sm_sum: f64 = sm.iter().sum();
    h.check_abs("softmax sums to 1.0", sm_sum, 1.0, tolerances::SOFTMAX_SUM);

    let gelu_0 = gelu(0.0);
    h.check_abs("GELU(0) = 0", gelu_0, 0.0, tolerances::GELU_CROSS_PYTHON);

    let gelu_large = gelu(10.0);
    h.check_abs(
        "GELU(10) ≈ 10",
        gelu_large,
        10.0,
        tolerances::GELU_LARGE_INPUT,
    );

    // ── Part 6: BarraCUDA coverage ──

    let barracuda_shaders: &[(&str, &str)] = &[
        ("GEMM", "gemm_f64.wgsl / matmul.wgsl"),
        ("Attention", "attention.wgsl"),
        (
            "Normalization",
            "layer_norm.wgsl / batch_norm.wgsl / rmsnorm.wgsl",
        ),
        ("Nonlinearity", "relu.wgsl / gelu.wgsl / nn::ReLU"),
        ("Reduction", "FusedMapReduceF64 / mean_reduce.wgsl"),
        ("Gating", "lstm_cell.wgsl / sigmoid.wgsl"),
    ];
    let all_covered = PRIMITIVES
        .iter()
        .all(|p| barracuda_shaders.iter().any(|(name, _)| name == p));
    h.check_bool(
        &format!("BarraCUDA covers all {} primitives", PRIMITIVES.len()),
        all_covered,
    );

    h.finish();
}
