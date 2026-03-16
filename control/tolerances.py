# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared tolerance vocabulary for neuralSpring Python baseline scripts.

Mirrors the Rust tolerance constants in ``src/tolerances/mod.rs`` so that
Python baselines and Rust validators agree on acceptance thresholds.

Usage in a baseline script::

    from tolerances import CROSS_LANGUAGE, SOFTMAX_SUM
    assert abs(rust_softmax_sum - 1.0) < SOFTMAX_SUM
    assert abs(rust_gelu - python_gelu) < CROSS_LANGUAGE
"""

# ═══════════════════════════════════════════════════════════════════
# Machine-precision tolerances (IEEE 754 f64)
# ═══════════════════════════════════════════════════════════════════

EXACT_F64 = 1e-12
CROSS_LANGUAGE = 1e-10
ZERO_DETECTION = 1e-14
NORM_PPF_TAIL = 0.01

# ═══════════════════════════════════════════════════════════════════
# Benchmark function tolerances
# ═══════════════════════════════════════════════════════════════════

BENCHMARK_GLOBAL_MIN = EXACT_F64
BENCHMARK_CROSS_PYTHON = CROSS_LANGUAGE
OPTIMIZER_POSITION = 1e-3
OPTIMIZER_POSITION_MULTIMODAL = 0.1
OPTIMIZER_VALUE_AT_MIN = 1e-4
OPTIMIZER_VALUE_MULTIMODAL = 1.0

# ═══════════════════════════════════════════════════════════════════
# Transformer primitive tolerances
# ═══════════════════════════════════════════════════════════════════

SOFTMAX_SUM = EXACT_F64
SOFTMAX_CROSS_PYTHON = 1e-14
GELU_CROSS_PYTHON = EXACT_F64
GELU_LARGE_INPUT = 1e-6
SIGMOID_SATURATION = 1e-4
SPECIAL_FUNCTION_F64 = 1e-6

# ═══════════════════════════════════════════════════════════════════
# Metric tolerances
# ═══════════════════════════════════════════════════════════════════

METRIC_EXACT = 1e-14

# ═══════════════════════════════════════════════════════════════════
# Training / model tolerances
# ═══════════════════════════════════════════════════════════════════

SURROGATE_R2_MIN = 0.40
TRANSFORMER_NUMPY_VS_PYTORCH = 1e-10
CAUSAL_MASK_LEAK = 1e-6
SEQUENCE_R2_MIN = 0.80
PINN_L2_ERROR_MAX = 0.15
PINN_IC_EXACT = EXACT_F64
PINN_BC_TOLERANCE = 0.01
PINN_SHOCK_RATIO_MIN = 1.5
DEEPONET_EXACT_ANTIDERIV = EXACT_F64
DEEPONET_POLYNOMIAL_EXACT = EXACT_F64

# ═══════════════════════════════════════════════════════════════════
# Quantization tolerances
# ═══════════════════════════════════════════════════════════════════

QUANT_INT8_DEGRADATION = 0.01
QUANT_INT4_DEGRADATION = 0.05
QUANT_Q8_ELEMENT_ERROR = 0.5
QUANT_Q4_ELEMENT_ERROR = 1.0

# ═══════════════════════════════════════════════════════════════════
# ODE integrator configuration
# ═══════════════════════════════════════════════════════════════════

ODE_ATOL = 1e-8
ODE_RTOL = 1e-6

# ═══════════════════════════════════════════════════════════════════
# Numerical stability guards
# ═══════════════════════════════════════════════════════════════════

LOG_ZERO_GUARD = 1e-30
FITNESS_FLOOR = 1e-10
LEXICASE_EPSILON = 1e-8
LAYER_NORM_EPS = 1e-5
HESSIAN_FD_STEP = 1e-5
HESSIAN_FD_ABS = 1.0
SADDLE_EIGENVALUE_THRESHOLD = -1e-10

# ═══════════════════════════════════════════════════════════════════
# Eigenvalue decomposition
# ═══════════════════════════════════════════════════════════════════

EIGH_JACOBI_RECONSTRUCT = 1e-2
EIGH_JACOBI_EIGENVALUE = 1e-3
JACOBI_GPU_CONVERGENCE = 1e-12
ODE_INTEGRATOR_AGREEMENT = 1e-2

# ═══════════════════════════════════════════════════════════════════
# Statistical critical values
# ═══════════════════════════════════════════════════════════════════

CHI2_CRITICAL_DF9_P05 = 16.92
CHI2_CRITICAL_DF1_P05 = 3.84
PANGENOME_MIN_ASSOCIATED_GENES = 5.0

# ═══════════════════════════════════════════════════════════════════
# Spectral analysis tolerances
# ═══════════════════════════════════════════════════════════════════

SPECTRAL_EIGENSOLVER_CROSS = 0.05
IPR_CROSS_PYTHON = 0.005
KAPPUS_WEGNER_REL = 0.5
LEVEL_SPACING_POISSON_TOL = 0.05
LEVEL_SPACING_GOE_SLACK = 0.2
SPECTRAL_IPR_COMPARISON_SLACK = -0.5
NUMERICAL_DISTINCTNESS = 1e-15

# ═══════════════════════════════════════════════════════════════════
# GPU-specific tolerances
# ═══════════════════════════════════════════════════════════════════

GPU_EIGENVALUE_AGREEMENT = 1e-6
VARIANCE_PARITY_FLOOR = 1e-10
PAIRFORMER_PARITY = 1e-6
GPU_SOFTMAX_SUM_F32 = 5e-3
GPU_LAYERNORM_F32 = 1e-3
