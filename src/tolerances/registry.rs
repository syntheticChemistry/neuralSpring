// SPDX-License-Identifier: AGPL-3.0-or-later

//! Runtime introspection for tolerance values (primal self-knowledge).
//!
//! Allows primals to discover available tolerances, their categories,
//! and their values at runtime — no hardcoded knowledge of the
//! tolerance namespace required.

#[allow(clippy::wildcard_imports)]
use super::*;

/// Named tolerance for runtime introspection.
///
/// Allows primals to discover and describe available tolerances
/// without hardcoded assumptions about what's defined.
#[derive(Debug, Clone, Copy)]
pub struct NamedTolerance {
    pub name: &'static str,
    pub value: f64,
    pub category: &'static str,
}

/// All tolerances in the system, queryable at runtime.
///
/// Each primal can discover what tolerances exist, what categories
/// they belong to, and what values they have — no hardcoded knowledge
/// of the tolerance namespace required.
///
/// Complete: every public constant in `tolerances::` is registered.
#[must_use]
#[allow(clippy::too_many_lines)]
pub const fn all_tolerances() -> &'static [NamedTolerance] {
    &[
        // ── Machine precision ──────────────────────────────────────────
        NamedTolerance {
            name: "EXACT_F64",
            value: EXACT_F64,
            category: "machine",
        },
        NamedTolerance {
            name: "CROSS_LANGUAGE",
            value: CROSS_LANGUAGE,
            category: "machine",
        },
        NamedTolerance {
            name: "ZERO_DETECTION",
            value: ZERO_DETECTION,
            category: "machine",
        },
        NamedTolerance {
            name: "NORM_PPF_TAIL",
            value: NORM_PPF_TAIL,
            category: "machine",
        },
        // ── Benchmark functions ────────────────────────────────────────
        NamedTolerance {
            name: "BENCHMARK_GLOBAL_MIN",
            value: BENCHMARK_GLOBAL_MIN,
            category: "benchmark",
        },
        NamedTolerance {
            name: "BENCHMARK_CROSS_PYTHON",
            value: BENCHMARK_CROSS_PYTHON,
            category: "benchmark",
        },
        NamedTolerance {
            name: "OPTIMIZER_POSITION",
            value: OPTIMIZER_POSITION,
            category: "benchmark",
        },
        NamedTolerance {
            name: "OPTIMIZER_POSITION_MULTIMODAL",
            value: OPTIMIZER_POSITION_MULTIMODAL,
            category: "benchmark",
        },
        NamedTolerance {
            name: "OPTIMIZER_VALUE_AT_MIN",
            value: OPTIMIZER_VALUE_AT_MIN,
            category: "benchmark",
        },
        NamedTolerance {
            name: "OPTIMIZER_VALUE_MULTIMODAL",
            value: OPTIMIZER_VALUE_MULTIMODAL,
            category: "benchmark",
        },
        // ── Transformer primitives ─────────────────────────────────────
        NamedTolerance {
            name: "SOFTMAX_SUM",
            value: SOFTMAX_SUM,
            category: "transformer",
        },
        NamedTolerance {
            name: "SOFTMAX_CROSS_PYTHON",
            value: SOFTMAX_CROSS_PYTHON,
            category: "transformer",
        },
        NamedTolerance {
            name: "GELU_CROSS_PYTHON",
            value: GELU_CROSS_PYTHON,
            category: "transformer",
        },
        NamedTolerance {
            name: "GELU_LARGE_INPUT",
            value: GELU_LARGE_INPUT,
            category: "transformer",
        },
        NamedTolerance {
            name: "SPECIAL_FUNCTION_F64",
            value: SPECIAL_FUNCTION_F64,
            category: "transformer",
        },
        // ── Metrics ────────────────────────────────────────────────────
        NamedTolerance {
            name: "METRIC_EXACT",
            value: METRIC_EXACT,
            category: "metric",
        },
        // ── Training / model (Python baselines) ────────────────────────
        NamedTolerance {
            name: "SURROGATE_R2_MIN",
            value: SURROGATE_R2_MIN,
            category: "training",
        },
        NamedTolerance {
            name: "TRANSFORMER_NUMPY_VS_PYTORCH",
            value: TRANSFORMER_NUMPY_VS_PYTORCH,
            category: "training",
        },
        NamedTolerance {
            name: "CAUSAL_MASK_LEAK",
            value: CAUSAL_MASK_LEAK,
            category: "training",
        },
        NamedTolerance {
            name: "SEQUENCE_R2_MIN",
            value: SEQUENCE_R2_MIN,
            category: "training",
        },
        NamedTolerance {
            name: "PINN_L2_ERROR_MAX",
            value: PINN_L2_ERROR_MAX,
            category: "training",
        },
        NamedTolerance {
            name: "PINN_IC_EXACT",
            value: PINN_IC_EXACT,
            category: "training",
        },
        NamedTolerance {
            name: "PINN_BC_TOLERANCE",
            value: PINN_BC_TOLERANCE,
            category: "training",
        },
        NamedTolerance {
            name: "PINN_SHOCK_RATIO_MIN",
            value: PINN_SHOCK_RATIO_MIN,
            category: "training",
        },
        NamedTolerance {
            name: "DEEPONET_EXACT_ANTIDERIV",
            value: DEEPONET_EXACT_ANTIDERIV,
            category: "training",
        },
        NamedTolerance {
            name: "DEEPONET_POLYNOMIAL_EXACT",
            value: DEEPONET_POLYNOMIAL_EXACT,
            category: "training",
        },
        NamedTolerance {
            name: "QUANT_INT8_DEGRADATION",
            value: QUANT_INT8_DEGRADATION,
            category: "training",
        },
        NamedTolerance {
            name: "QUANT_INT4_DEGRADATION",
            value: QUANT_INT4_DEGRADATION,
            category: "training",
        },
        NamedTolerance {
            name: "QUANT_Q8_ELEMENT_ERROR",
            value: QUANT_Q8_ELEMENT_ERROR,
            category: "training",
        },
        NamedTolerance {
            name: "QUANT_Q4_ELEMENT_ERROR",
            value: QUANT_Q4_ELEMENT_ERROR,
            category: "training",
        },
        // ── Evolutionary / stochastic algorithms ───────────────────────
        NamedTolerance {
            name: "CD_COMPARABLE_DIST",
            value: CD_COMPARABLE_DIST,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "ADIABATIC_KL_GAP",
            value: ADIABATIC_KL_GAP,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "HMM_POSTERIOR_SUM",
            value: HMM_POSTERIOR_SUM,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "QS_VARIANCE_MAX",
            value: QS_VARIANCE_MAX,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "HMM_DECODE_ACCURACY_MIN",
            value: HMM_DECODE_ACCURACY_MIN,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "INTROGRESSION_FRACTION_MIN",
            value: INTROGRESSION_FRACTION_MIN,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "INTROGRESSION_FRACTION_ABS",
            value: INTROGRESSION_FRACTION_ABS,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "INTROGRESSION_FPR_MAX",
            value: INTROGRESSION_FPR_MAX,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "GENE_TREE_CONCORDANT_MIN",
            value: GENE_TREE_CONCORDANT_MIN,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "GAME_COOPERATION_MIN",
            value: GAME_COOPERATION_MIN,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "REGULATORY_RESPONSE_MIN",
            value: REGULATORY_RESPONSE_MIN,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "ECO_FITNESS_IMPROVEMENT_MIN",
            value: ECO_FITNESS_IMPROVEMENT_MIN,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "PANGENOME_SELECTION_P_MIN",
            value: PANGENOME_SELECTION_P_MIN,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "META_POP_FST_MIN",
            value: META_POP_FST_MIN,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "META_POP_AF_VARIANCE_MIN",
            value: META_POP_AF_VARIANCE_MIN,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "HMM_PHYLO_DECODE_MARGIN",
            value: HMM_PHYLO_DECODE_MARGIN,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "SIGNAL_DYNAMIC_RANGE_MIN",
            value: SIGNAL_DYNAMIC_RANGE_MIN,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "SWARM_FITNESS_COMPARISON",
            value: SWARM_FITNESS_COMPARISON,
            category: "evolutionary",
        },
        // ── Stochastic model ───────────────────────────────────────────
        NamedTolerance {
            name: "BARRACUDA_GPU_ECO_F32",
            value: BARRACUDA_GPU_ECO_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "SPECTRAL_COMMUTATIVITY_EPS",
            value: SPECTRAL_COMMUTATIVITY_EPS,
            category: "physics",
        },
        // ── Tensor / WGSL shader (f32 compute) ────────────────────────
        NamedTolerance {
            name: "TENSOR_EXACT_F32",
            value: TENSOR_EXACT_F32,
            category: "tensor",
        },
        NamedTolerance {
            name: "TENSOR_TRANSCENDENTAL_F32",
            value: TENSOR_TRANSCENDENTAL_F32,
            category: "tensor",
        },
        NamedTolerance {
            name: "TENSOR_MATMUL_F32",
            value: TENSOR_MATMUL_F32,
            category: "tensor",
        },
        NamedTolerance {
            name: "TENSOR_NORM_F32",
            value: TENSOR_NORM_F32,
            category: "tensor",
        },
        // ── GPU f64 shader ─────────────────────────────────────────────
        NamedTolerance {
            name: "GPU_F64_EXACT",
            value: GPU_F64_EXACT,
            category: "gpu_f64",
        },
        NamedTolerance {
            name: "GPU_F64_TRANSCENDENTAL",
            value: GPU_F64_TRANSCENDENTAL,
            category: "gpu_f64",
        },
        NamedTolerance {
            name: "GPU_F64_STATS",
            value: GPU_F64_STATS,
            category: "gpu_f64",
        },
        // ── FFT ────────────────────────────────────────────────────────
        NamedTolerance {
            name: "FFT_INVERSE_F32",
            value: FFT_INVERSE_F32,
            category: "fft",
        },
        NamedTolerance {
            name: "FFT_INVERSE_F64",
            value: FFT_INVERSE_F64,
            category: "fft",
        },
        NamedTolerance {
            name: "FFT_PARSEVAL_F32",
            value: FFT_PARSEVAL_F32,
            category: "fft",
        },
        NamedTolerance {
            name: "FFT_PARSEVAL_F64",
            value: FFT_PARSEVAL_F64,
            category: "fft",
        },
        NamedTolerance {
            name: "FFT_KNOWN_PAIR_F32",
            value: FFT_KNOWN_PAIR_F32,
            category: "fft",
        },
        NamedTolerance {
            name: "FFT_KNOWN_PAIR_F64",
            value: FFT_KNOWN_PAIR_F64,
            category: "fft",
        },
        NamedTolerance {
            name: "FFT_SPECTRAL_LEAKAGE_F32",
            value: FFT_SPECTRAL_LEAKAGE_F32,
            category: "fft",
        },
        NamedTolerance {
            name: "FFT_SPECTRAL_LEAKAGE_F64",
            value: FFT_SPECTRAL_LEAKAGE_F64,
            category: "fft",
        },
        NamedTolerance {
            name: "RFFT_DC_COMPONENT_F32",
            value: RFFT_DC_COMPONENT_F32,
            category: "fft",
        },
        // ── GPU shader (metalForge Phase 3c) ───────────────────────────
        NamedTolerance {
            name: "GPU_HMM_LOG_LIKELIHOOD_F32",
            value: GPU_HMM_LOG_LIKELIHOOD_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_HMM_ALPHA_F32",
            value: GPU_HMM_ALPHA_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_FITNESS_F32",
            value: GPU_FITNESS_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_RK4_F32",
            value: GPU_RK4_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_JACCARD_F32",
            value: GPU_JACCARD_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_LOCUS_VARIANCE_F32",
            value: GPU_LOCUS_VARIANCE_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_SPATIAL_PAYOFF_F32",
            value: GPU_SPATIAL_PAYOFF_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_BATCH_IPR_F32",
            value: GPU_BATCH_IPR_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_HAMMING_F32",
            value: GPU_HAMMING_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_MULTI_OBJ_FITNESS_F32",
            value: GPU_MULTI_OBJ_FITNESS_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_UPSTREAM_MULTI_OBJ_PARITY_F32",
            value: GPU_UPSTREAM_MULTI_OBJ_PARITY_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_MODES_L2_F32",
            value: GPU_MODES_L2_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_HILL_F32",
            value: GPU_HILL_F32,
            category: "gpu_shader",
        },
        // ── ML inference pipeline ──────────────────────────────────────
        NamedTolerance {
            name: "ML_MLP_F32",
            value: ML_MLP_F32,
            category: "ml_pipeline",
        },
        NamedTolerance {
            name: "ML_TRANSFORMER_F32",
            value: ML_TRANSFORMER_F32,
            category: "ml_pipeline",
        },
        // ── Eigenvalue decomposition ───────────────────────────────────
        NamedTolerance {
            name: "EIGH_JACOBI_RECONSTRUCT",
            value: EIGH_JACOBI_RECONSTRUCT,
            category: "linalg",
        },
        NamedTolerance {
            name: "EIGH_JACOBI_EIGENVALUE",
            value: EIGH_JACOBI_EIGENVALUE,
            category: "linalg",
        },
        // ── ODE integrator agreement ───────────────────────────────────
        NamedTolerance {
            name: "ODE_INTEGRATOR_AGREEMENT",
            value: ODE_INTEGRATOR_AGREEMENT,
            category: "numerical",
        },
        // ── Statistical critical values ────────────────────────────────
        NamedTolerance {
            name: "CHI2_CRITICAL_DF9_P05",
            value: CHI2_CRITICAL_DF9_P05,
            category: "statistical",
        },
        NamedTolerance {
            name: "CHI2_CRITICAL_DF1_P05",
            value: CHI2_CRITICAL_DF1_P05,
            category: "statistical",
        },
        NamedTolerance {
            name: "PANGENOME_MIN_ASSOCIATED_GENES",
            value: PANGENOME_MIN_ASSOCIATED_GENES,
            category: "statistical",
        },
        // ── Physics ────────────────────────────────────────────────────
        NamedTolerance {
            name: "IPR_LOCALIZATION_MIN",
            value: IPR_LOCALIZATION_MIN,
            category: "physics",
        },
        NamedTolerance {
            name: "SPECTRAL_EIGENSOLVER_CROSS",
            value: SPECTRAL_EIGENSOLVER_CROSS,
            category: "spectral",
        },
        NamedTolerance {
            name: "KAPPUS_WEGNER_REL",
            value: KAPPUS_WEGNER_REL,
            category: "spectral",
        },
        NamedTolerance {
            name: "LEVEL_SPACING_POISSON_TOL",
            value: LEVEL_SPACING_POISSON_TOL,
            category: "spectral",
        },
        // ── GPU pipeline (Phase 5c) ───────────────────────────────────
        NamedTolerance {
            name: "GPU_REDUCE_F64",
            value: GPU_REDUCE_F64,
            category: "gpu_pipeline",
        },
        // ── Miscellaneous validation ───────────────────────────────────
        NamedTolerance {
            name: "PINN_FD_RESIDUAL_MAX",
            value: PINN_FD_RESIDUAL_MAX,
            category: "numerical",
        },
        NamedTolerance {
            name: "SEASONAL_ANNUAL_MEAN",
            value: SEASONAL_ANNUAL_MEAN,
            category: "numerical",
        },
        NamedTolerance {
            name: "SEASONAL_ANNUAL_MEAN_TOL",
            value: SEASONAL_ANNUAL_MEAN_TOL,
            category: "numerical",
        },
        NamedTolerance {
            name: "ECO_DOMINANCE_COMPARISON",
            value: ECO_DOMINANCE_COMPARISON,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "ML_PIPELINE_NORM_REL",
            value: ML_PIPELINE_NORM_REL,
            category: "ml_pipeline",
        },
        NamedTolerance {
            name: "DIVERSITY_EPSILON",
            value: DIVERSITY_EPSILON,
            category: "numerical",
        },
        NamedTolerance {
            name: "VARIANCE_FLOOR",
            value: VARIANCE_FLOOR,
            category: "numerical",
        },
        NamedTolerance {
            name: "GPU_LOGSUMEXP_F32",
            value: GPU_LOGSUMEXP_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_RK45_F32",
            value: GPU_RK45_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_BOUNDS_SLACK_F32",
            value: GPU_BOUNDS_SLACK_F32,
            category: "gpu_shader",
        },
    ]
}

/// Look up a tolerance by name at runtime. Returns `None` if not found.
#[must_use]
pub fn tolerance_by_name(name: &str) -> Option<f64> {
    all_tolerances()
        .iter()
        .find(|t| t.name == name)
        .map(|t| t.value)
}

/// List all tolerance categories available in the system.
#[must_use]
pub fn categories() -> Vec<&'static str> {
    let mut cats: Vec<&str> = all_tolerances().iter().map(|t| t.category).collect();
    cats.sort_unstable();
    cats.dedup();
    cats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn tolerance_ordering() {
        assert!(
            EXACT_F64 < CROSS_LANGUAGE,
            "exact should be tighter than cross-language"
        );
        assert!(
            SOFTMAX_CROSS_PYTHON < CROSS_LANGUAGE,
            "softmax should be tighter than cross-language"
        );
    }

    #[test]
    fn introspection_works() {
        let all = all_tolerances();
        assert!(!all.is_empty());
        assert!(tolerance_by_name("EXACT_F64").is_some());
        assert!(tolerance_by_name("NONEXISTENT").is_none());
        let cats = categories();
        for expected in [
            "machine",
            "benchmark",
            "transformer",
            "metric",
            "training",
            "evolutionary",
            "gpu_shader",
            "gpu_f64",
            "gpu_pipeline",
            "tensor",
            "fft",
            "physics",
            "spectral",
            "numerical",
            "statistical",
            "linalg",
            "ml_pipeline",
        ] {
            assert!(cats.contains(&expected), "missing category: {expected}");
        }
    }

    #[test]
    fn registry_complete() {
        let all = all_tolerances();
        assert!(
            all.len() >= 90,
            "registry should contain all tolerances, got {}",
            all.len()
        );
    }

    #[test]
    fn all_finite_and_documented() {
        let known_negative = ["VARIANCE_FLOOR"];
        for t in all_tolerances() {
            assert!(t.value.is_finite(), "{} must be finite", t.name);
            assert!(!t.name.is_empty(), "tolerance name must not be empty");
            assert!(!t.category.is_empty(), "category must not be empty");
            if !known_negative.contains(&t.name) {
                assert!(
                    t.value > 0.0,
                    "{} must be positive, got {}",
                    t.name,
                    t.value
                );
            }
        }
    }
}
