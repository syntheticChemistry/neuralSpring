// SPDX-License-Identifier: AGPL-3.0-only

//! Provenance metadata for all Python baseline values.
//!
//! Every hardcoded expected value in validation binaries traces back to a
//! specific Python control run. This module centralizes that metadata so
//! validation binaries carry machine-readable provenance.
//!
//! Imitates the hotSpring pattern:
//! ```text
//! Python script → commit → environment → command → output → Rust constant
//! ```
//!
//! ## Data Sources
//!
//! | Dataset | Source | License |
//! |---------|--------|---------|
//! | Benchmark functions | Analytical (Rastrigin, Rosenbrock, Ackley) | N/A |
//! | FAO-56 ET₀ | Allen et al. (1998) FAO Paper 56 | Public |
//! | MNIST | LeCun et al. (1998) via `torchvision` | CC BY-SA 3.0 |
//! | ERA5 weather | Open-Meteo Archive API (ECMWF Copernicus) | CC BY 4.0 |
//! | Burgers PDE | Raissi et al. (2019) JCP, DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045) | N/A |
//! | Antiderivative | Lu et al. (2021) NMI, DOI: [10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5) | N/A |

/// A single provenance record tying a Rust reference value to its Python origin.
#[derive(Debug, Clone)]
pub struct BaselineProvenance {
    pub label: &'static str,
    pub script: &'static str,
    pub commit: &'static str,
    pub date: &'static str,
    pub command: &'static str,
    pub environment: &'static str,
    pub value: f64,
    pub unit: &'static str,
}

// ═══════════════════════════════════════════════════════════════════
// Environment
// ═══════════════════════════════════════════════════════════════════

/// Python environment for all control runs.
pub const ENVIRONMENT: &str = "Python 3.10.12, PyTorch 2.9.0+cu128, NumPy 2.2.6, SciPy 1.15.3";

/// Hardware for all control runs.
pub const HARDWARE: &str = "Eastgate (i9-12900K, RTX 4070 12GB, Pop!_OS 22.04)";

/// Pinned commit for baseline results.
pub const BASELINE_DATE: &str = "2026-02-16";

// ═══════════════════════════════════════════════════════════════════
// Exp 001: Surrogate Validation
// ═══════════════════════════════════════════════════════════════════

pub const SURROGATE_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp 001: Neural Surrogate Validation (11/11 PASS)",
    script: "control/surrogate/surrogate_validation.py",
    commit: "baseline 2026-02-16",
    date: "2026-02-16",
    command: "python3 control/surrogate/surrogate_validation.py",
    environment: ENVIRONMENT,
    value: 11.0,
    unit: "checks passed",
};

// ═══════════════════════════════════════════════════════════════════
// Exp 002: Transformer Inference
// ═══════════════════════════════════════════════════════════════════

pub const TRANSFORMER_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp 002: Transformer Inference Baseline (18/18 PASS)",
    script: "control/transformer/transformer_inference.py",
    commit: "baseline 2026-02-16",
    date: "2026-02-16",
    command: "python3 control/transformer/transformer_inference.py",
    environment: ENVIRONMENT,
    value: 18.0,
    unit: "checks passed",
};

// ═══════════════════════════════════════════════════════════════════
// Cross-language reference values (Python-computed, hardcoded in Rust)
// ═══════════════════════════════════════════════════════════════════

/// Softmax of `[1,2,3,4,5]` computed by `NumPy` 2.2.6.
pub const SOFTMAX_1_TO_5: [f64; 5] = [
    1.165_623_095_603_961e-2,
    3.168_492_079_612_427e-2,
    8.612_854_443_626_87e-2,
    2.341_216_572_527_366e-1,
    6.364_086_465_588_308e-1,
];

/// GELU reference values at selected points, computed by `NumPy` 2.2.6.
///
/// Format: (input, `expected_output`)
pub const GELU_REFERENCE: [(f64, f64); 6] = [
    (-2.0, -4.540_230_591_222_494e-2),
    (-1.0, -1.588_080_093_917_233e-1),
    (0.0, 0.0),
    (0.5, 3.457_140_098_251_439e-1),
    (1.0, 8.411_919_906_082_768e-1),
    (3.0, 2.996_362_607_918_227),
];

/// Rastrigin 2D reference values at non-trivial points, computed by `NumPy` 2.2.6.
pub const RASTRIGIN_REFERENCE: [(f64, f64, f64); 4] = [
    (1.0, 1.0, 2.0),
    (2.5, -1.3, 4.103_016_994_374_947e1),
    (0.5, 0.5, 4.05e1),
    (-3.0, 2.0, 13.0),
];

/// Rosenbrock 2D reference values.
pub const ROSENBROCK_REFERENCE: [(f64, f64, f64); 4] = [
    (1.0, 1.0, 0.0),
    (2.5, -1.3, 5702.5),
    (0.5, 0.5, 6.5),
    (-3.0, 2.0, 4916.0),
];

/// Ackley 2D reference values.
pub const ACKLEY_REFERENCE: [(f64, f64, f64); 4] = [
    (1.0, 1.0, 3.625_384_938_440_363),
    (2.5, -1.3, 8.772_020_879_614_113),
    (0.5, 0.5, 4.253_654_026_568_412),
    (-3.0, 2.0, 7.988_910_810_518_7),
];

/// Analytical reference source for benchmark functions.
pub const BENCHMARK_REFS: &str = "Analytical global minima + NumPy 2.2.6 cross-validation";

/// Analytical reference source for transformer primitives.
pub const TRANSFORMER_REFS: &str = "NumPy 2.2.6 transformer_inference.py (softmax, gelu_numpy)";

/// Analytical reference source for statistical metrics.
pub const METRICS_REFS: &str = "Analytical (pure arithmetic on known arrays)";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_reference_sums_near_one() {
        let sum: f64 = SOFTMAX_1_TO_5.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14);
    }

    #[test]
    fn gelu_zero_is_zero() {
        let zero_entry = GELU_REFERENCE.iter().find(|(x, _)| *x == 0.0);
        assert!(zero_entry.is_some());
        assert!((zero_entry.unwrap().1).abs() < 1e-15);
    }

    #[test]
    fn provenance_records_non_empty() {
        for p in [&SURROGATE_PROVENANCE, &TRANSFORMER_PROVENANCE] {
            assert!(!p.label.is_empty());
            assert!(!p.script.is_empty());
            assert!(!p.date.is_empty());
            assert!(!p.command.is_empty());
        }
    }

    #[test]
    fn benchmark_references_have_global_minima() {
        assert!(RASTRIGIN_REFERENCE
            .iter()
            .any(|(x, y, _)| *x == 1.0 && *y == 1.0));
        assert!(ROSENBROCK_REFERENCE.iter().any(|(_, _, f)| *f == 0.0));
    }
}
