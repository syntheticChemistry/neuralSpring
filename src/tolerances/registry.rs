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
#[must_use]
#[allow(clippy::too_many_lines)]
pub const fn all_tolerances() -> &'static [NamedTolerance] {
    &[
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
            name: "GPU_FITNESS_F32",
            value: GPU_FITNESS_F32,
            category: "gpu_shader",
        },
        NamedTolerance {
            name: "GPU_HAMMING_F32",
            value: GPU_HAMMING_F32,
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
            name: "GPU_REDUCE_F64",
            value: GPU_REDUCE_F64,
            category: "gpu_pipeline",
        },
        NamedTolerance {
            name: "SPECTRAL_EIGENSOLVER_CROSS",
            value: SPECTRAL_EIGENSOLVER_CROSS,
            category: "spectral",
        },
        NamedTolerance {
            name: "TENSOR_EXACT_F32",
            value: TENSOR_EXACT_F32,
            category: "tensor",
        },
        NamedTolerance {
            name: "TENSOR_MATMUL_F32",
            value: TENSOR_MATMUL_F32,
            category: "tensor",
        },
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
            name: "SWARM_FITNESS_COMPARISON",
            value: SWARM_FITNESS_COMPARISON,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "IPR_LOCALIZATION_MIN",
            value: IPR_LOCALIZATION_MIN,
            category: "physics",
        },
        NamedTolerance {
            name: "SPECTRAL_COMMUTATIVITY_EPS",
            value: SPECTRAL_COMMUTATIVITY_EPS,
            category: "physics",
        },
        NamedTolerance {
            name: "SIGNAL_DYNAMIC_RANGE_MIN",
            value: SIGNAL_DYNAMIC_RANGE_MIN,
            category: "evolutionary",
        },
        NamedTolerance {
            name: "PINN_FD_RESIDUAL_MAX",
            value: PINN_FD_RESIDUAL_MAX,
            category: "numerical",
        },
        NamedTolerance {
            name: "BARRACUDA_GPU_ECO_F32",
            value: BARRACUDA_GPU_ECO_F32,
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
        assert!(cats.contains(&"machine"));
        assert!(cats.contains(&"gpu_shader"));
    }

    #[test]
    fn all_positive() {
        let tols = [
            EXACT_F64,
            CROSS_LANGUAGE,
            ZERO_DETECTION,
            NORM_PPF_TAIL,
            BENCHMARK_GLOBAL_MIN,
            BENCHMARK_CROSS_PYTHON,
            OPTIMIZER_POSITION,
            OPTIMIZER_POSITION_MULTIMODAL,
            OPTIMIZER_VALUE_AT_MIN,
            OPTIMIZER_VALUE_MULTIMODAL,
            SOFTMAX_SUM,
            SOFTMAX_CROSS_PYTHON,
            GELU_CROSS_PYTHON,
            GELU_LARGE_INPUT,
            SPECIAL_FUNCTION_F64,
            METRIC_EXACT,
            SURROGATE_R2_MIN,
            TRANSFORMER_NUMPY_VS_PYTORCH,
            CAUSAL_MASK_LEAK,
            SEQUENCE_R2_MIN,
            PINN_L2_ERROR_MAX,
            PINN_IC_EXACT,
            PINN_BC_TOLERANCE,
            PINN_SHOCK_RATIO_MIN,
            DEEPONET_EXACT_ANTIDERIV,
            DEEPONET_POLYNOMIAL_EXACT,
            QUANT_INT8_DEGRADATION,
            QUANT_INT4_DEGRADATION,
            QUANT_Q8_ELEMENT_ERROR,
            QUANT_Q4_ELEMENT_ERROR,
            TENSOR_EXACT_F32,
            TENSOR_TRANSCENDENTAL_F32,
            TENSOR_MATMUL_F32,
            TENSOR_NORM_F32,
            CD_COMPARABLE_DIST,
            ADIABATIC_KL_GAP,
            HMM_POSTERIOR_SUM,
            QS_VARIANCE_MAX,
            GPU_F64_EXACT,
            GPU_F64_TRANSCENDENTAL,
            GPU_F64_STATS,
            ML_MLP_F32,
            ML_TRANSFORMER_F32,
            FFT_INVERSE_F32,
            FFT_INVERSE_F64,
            FFT_PARSEVAL_F32,
            FFT_PARSEVAL_F64,
            FFT_KNOWN_PAIR_F32,
            FFT_KNOWN_PAIR_F64,
            FFT_SPECTRAL_LEAKAGE_F32,
            FFT_SPECTRAL_LEAKAGE_F64,
            RFFT_DC_COMPONENT_F32,
            GPU_HMM_LOG_LIKELIHOOD_F32,
            GPU_HMM_ALPHA_F32,
            GPU_FITNESS_F32,
            GPU_RK4_F32,
            GPU_JACCARD_F32,
            GPU_LOCUS_VARIANCE_F32,
            GPU_SPATIAL_PAYOFF_F32,
            GPU_BATCH_IPR_F32,
            GPU_HAMMING_F32,
            GPU_MULTI_OBJ_FITNESS_F32,
            GPU_MODES_L2_F32,
            GPU_HILL_F32,
            EIGH_JACOBI_RECONSTRUCT,
            EIGH_JACOBI_EIGENVALUE,
            ODE_INTEGRATOR_AGREEMENT,
            IPR_LOCALIZATION_MIN,
            HMM_DECODE_ACCURACY_MIN,
            INTROGRESSION_FRACTION_MIN,
            INTROGRESSION_FRACTION_ABS,
            INTROGRESSION_FPR_MAX,
            GENE_TREE_CONCORDANT_MIN,
            GAME_COOPERATION_MIN,
            REGULATORY_RESPONSE_MIN,
            ECO_FITNESS_IMPROVEMENT_MIN,
            PANGENOME_SELECTION_P_MIN,
            META_POP_FST_MIN,
            META_POP_AF_VARIANCE_MIN,
            HMM_PHYLO_DECODE_MARGIN,
            SIGNAL_DYNAMIC_RANGE_MIN,
            BARRACUDA_GPU_ECO_F32,
            SPECTRAL_COMMUTATIVITY_EPS,
            CHI2_CRITICAL_DF9_P05,
            CHI2_CRITICAL_DF1_P05,
            PANGENOME_MIN_ASSOCIATED_GENES,
            SWARM_FITNESS_COMPARISON,
            PINN_FD_RESIDUAL_MAX,
            SEASONAL_ANNUAL_MEAN,
            SEASONAL_ANNUAL_MEAN_TOL,
            ECO_DOMINANCE_COMPARISON,
            ML_PIPELINE_NORM_REL,
        ];
        for (i, &t) in tols.iter().enumerate() {
            assert!(t > 0.0, "tolerance index {i} must be positive, got {t}");
        }
    }
}
