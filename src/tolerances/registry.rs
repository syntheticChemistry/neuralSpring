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

/// Declarative registry: maps each constant to its category with zero
/// boilerplate. `stringify!` guarantees name strings stay in sync with
/// the actual constant identifiers — no manual typos possible.
macro_rules! tolerance_registry {
    ($( $cat:literal : [ $($name:ident),+ $(,)? ] ),+ $(,)?) => {
        &[
            $($(
                NamedTolerance {
                    name: stringify!($name),
                    value: $name,
                    category: $cat,
                },
            )+)+
        ]
    };
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
    tolerance_registry![
        "machine": [
            EXACT_F64, CROSS_LANGUAGE, ZERO_DETECTION, NORM_PPF_TAIL,
        ],
        "benchmark": [
            BENCHMARK_GLOBAL_MIN, BENCHMARK_CROSS_PYTHON,
            OPTIMIZER_POSITION, OPTIMIZER_POSITION_MULTIMODAL,
            OPTIMIZER_VALUE_AT_MIN, OPTIMIZER_VALUE_MULTIMODAL,
        ],
        "transformer": [
            SOFTMAX_SUM, SOFTMAX_CROSS_PYTHON,
            GELU_CROSS_PYTHON, GELU_LARGE_INPUT, SPECIAL_FUNCTION_F64,
        ],
        "metric": [METRIC_EXACT],
        "training": [
            SURROGATE_R2_MIN, TRANSFORMER_NUMPY_VS_PYTORCH,
            CAUSAL_MASK_LEAK, SEQUENCE_R2_MIN,
            PINN_L2_ERROR_MAX, PINN_IC_EXACT,
            PINN_BC_TOLERANCE, PINN_SHOCK_RATIO_MIN,
            DEEPONET_EXACT_ANTIDERIV, DEEPONET_POLYNOMIAL_EXACT,
            QUANT_INT8_DEGRADATION, QUANT_INT4_DEGRADATION,
            QUANT_Q8_ELEMENT_ERROR, QUANT_Q4_ELEMENT_ERROR,
        ],
        "evolutionary": [
            CD_COMPARABLE_DIST, ADIABATIC_KL_GAP,
            HMM_POSTERIOR_SUM, QS_VARIANCE_MAX,
            HMM_DECODE_ACCURACY_MIN,
            INTROGRESSION_FRACTION_MIN, INTROGRESSION_FRACTION_ABS,
            INTROGRESSION_FPR_MAX, GENE_TREE_CONCORDANT_MIN,
            GAME_COOPERATION_MIN, REPLICATOR_DYNAMICS,
            REGULATORY_RESPONSE_MIN, ECO_FITNESS_IMPROVEMENT_MIN,
            PANGENOME_SELECTION_P_MIN,
            META_POP_FST_MIN, META_POP_AF_VARIANCE_MIN,
            HMM_PHYLO_DECODE_MARGIN, SIGNAL_DYNAMIC_RANGE_MIN,
            SPECTRAL_SELF_SIMILARITY, PGM_COMPLEXITY_SLACK,
            SWARM_FITNESS_COMPARISON, ECO_DOMINANCE_COMPARISON,
            GAME_DEFECTION_UPPER, GAME_QS_COOPERATION_MIN,
            GAME_QS_VARIANCE_MAX,
            FST_IDENTICAL_POP_TOL, FST_ESTIMATOR_AGREEMENT,
        ],
        "physics": [
            SPECTRAL_COMMUTATIVITY_EPS, IPR_LOCALIZATION_MIN,
        ],
        "tensor": [
            TENSOR_EXACT_F32, TENSOR_TRANSCENDENTAL_F32,
            TENSOR_MATMUL_F32, TENSOR_NORM_F32,
            LAYER_NORM_EPS,
        ],
        "gpu_f64": [
            GPU_F64_EXACT, GPU_F64_TRANSCENDENTAL, GPU_F64_STATS,
        ],
        "fft": [
            FFT_INVERSE_F32, FFT_INVERSE_F64,
            FFT_PARSEVAL_F32, FFT_PARSEVAL_F64,
            FFT_KNOWN_PAIR_F32, FFT_KNOWN_PAIR_F64,
            FFT_SPECTRAL_LEAKAGE_F32, FFT_SPECTRAL_LEAKAGE_F64,
            RFFT_DC_COMPONENT_F32,
        ],
        "gpu_shader": [
            BARRACUDA_GPU_ECO_F32,
            GPU_HMM_LOG_LIKELIHOOD_F32, GPU_HMM_ALPHA_F32,
            GPU_FITNESS_F32, GPU_RK4_F32,
            GPU_JACCARD_F32, GPU_LOCUS_VARIANCE_F32,
            GPU_SPATIAL_PAYOFF_F32, GPU_BATCH_IPR_F32,
            GPU_HAMMING_F32, GPU_MULTI_OBJ_FITNESS_F32,
            GPU_UPSTREAM_MULTI_OBJ_PARITY_F32,
            GPU_MODES_L2_F32, GPU_HILL_F32,
            GPU_LOGSUMEXP_F32, GPU_RK45_F32, GPU_BOUNDS_SLACK_F32,
        ],
        "ml_pipeline": [
            ML_MLP_F32, ML_TRANSFORMER_F32, ML_PIPELINE_NORM_REL,
        ],
        "linalg": [
            EIGH_JACOBI_RECONSTRUCT, EIGH_JACOBI_EIGENVALUE,
        ],
        "numerical": [
            ODE_INTEGRATOR_AGREEMENT, ODE_ATOL, ODE_RTOL,
            LOG_ZERO_GUARD, HESSIAN_FD_STEP, HESSIAN_FD_ABS, SADDLE_EIGENVALUE_THRESHOLD,
            PINN_FD_RESIDUAL_MAX,
            SEASONAL_ANNUAL_MEAN, SEASONAL_ANNUAL_MEAN_TOL,
            DIVERSITY_EPSILON, VARIANCE_FLOOR,
            RELATIVE_ERROR_FLOOR, ODE_STEADY_STATE_SLACK,
        ],
        "statistical": [
            CHI2_CRITICAL_DF9_P05, CHI2_CRITICAL_DF1_P05,
            PANGENOME_MIN_ASSOCIATED_GENES,
        ],
        "spectral": [
            SPECTRAL_EIGENSOLVER_CROSS, KAPPUS_WEGNER_REL,
            LEVEL_SPACING_POISSON_TOL,
            LEVEL_SPACING_GOE_SLACK, SPECTRAL_IPR_COMPARISON_SLACK,
            NUMERICAL_DISTINCTNESS,
            GATE_DISORDER_COMPARISON, SPECTRAL_RADIUS_SWEEP_SLACK,
            GOE_LSR_TOLERANCE, IPR_RATIO_SPREAD_MAX,
        ],
        "folding": [
            FOLDING_EPS, DIFFUSION_ALPHA_BAR_FLOOR, DIFFUSION_BETA_FLOOR,
        ],
        "domain_guards": [
            FISHER_EPS, BURGERS_IC_GUARD, DP_EQUALITY_EPS,
            SINGLETON_FREQ_EPS, PHENOTYPE_TIE_EPS,
        ],
        "gpu_pipeline": [GPU_REDUCE_F64],
        "gpu_dispatch": [
            GPU_MATMUL_IDENTITY_F32, GPU_MATMUL_RANDOM_F32,
            GPU_TRANSPOSE_F32, GPU_FROBENIUS_F32, GPU_COMMUTATOR_F32,
            GPU_NORMAL_DISTANCE_SYMMETRIC_F32,
            CPU_NORMAL_DISTANCE_SYMMETRIC_F64,
            GPU_SOFTMAX_DISPATCH_F32, GPU_SOFTMAX_SUM_F32,
            GPU_BOLTZMANN_F32, GPU_L2_DISPATCH_F32,
            GPU_MEAN_DISPATCH_F32, GPU_VARIANCE_DISPATCH_F32,
            GPU_ENTROPY_F32, GPU_PEARSON_F32,
            GPU_CHI_SQUARED_F32, GPU_GELU_F32,
            GPU_HMM_STEP_F32, GPU_SUM_DISPATCH_F32,
            GPU_MAX_DISPATCH_F32, GPU_KL_DISPATCH_F32,
            GPU_MULTI_OBJ_FITNESS_F64, GPU_AF_VARIANCE_F32,
            GPU_HMM_VITERBI_LOGPROB_F64,
            GPU_VITERBI_PATH_AGREEMENT_MIN,
            GPU_FST_PAIRWISE_F32,
            GPU_VARIANCE_F64, GPU_PEARSON_F64, GPU_ENTROPY_F64,
            GPU_EIGH_DISPATCH_F64, PGM_NORMALIZATION_SUM,
            GPU_COMMUTATOR_NEAR_ZERO_F64, GPU_COMMUTATOR_RESIDUAL_F64,
        ],
        "training_quantized": [
            QUANT_Q8_GEMV_ERROR, QUANT_Q4_GEMV_ERROR, QUANT_SIGN_AGREEMENT,
        ],
        "hardware": [
            BRIDGE_COST_MIN_US, BRIDGE_COST_MAX_US,
            BRIDGE_CHAIN_OVERHEAD_MAX, BRIDGE_PROBE_MIN_US,
            TRANSFER_1MB_MIN_US, TRANSFER_1MB_MAX_US,
            DISPATCH_COST_RATIO_MIN, DISPATCH_COST_RATIO_MAX,
        ],
        "cross_dispatch": [
            DISPATCH_MATMUL_F64, DISPATCH_FROBENIUS_F64,
            DISPATCH_TRANSPOSE_F64, DISPATCH_ELEMENTWISE_F64,
            DISPATCH_TWOPASS_F64, DISPATCH_NEAR_ZERO_F64,
            DISPATCH_F32_ROUNDTRIP, DISPATCH_VITERBI_F32,
        ],
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
            "training_quantized",
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
            "cross_dispatch",
            "gpu_dispatch",
            "hardware",
        ] {
            assert!(cats.contains(&expected), "missing category: {expected}");
        }
    }

    #[test]
    fn registry_complete() {
        let all = all_tolerances();
        assert!(
            all.len() >= 139,
            "registry should contain all tolerances, got {}",
            all.len()
        );
    }

    #[test]
    fn all_finite_and_documented() {
        let known_negative = [
            "VARIANCE_FLOOR",
            "SADDLE_EIGENVALUE_THRESHOLD",
            "SPECTRAL_IPR_COMPARISON_SLACK",
        ];
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
