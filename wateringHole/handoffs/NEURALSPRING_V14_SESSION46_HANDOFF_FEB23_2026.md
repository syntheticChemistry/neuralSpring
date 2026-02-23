# neuralSpring V14 — Sessions 45+46 Handoff

**Date**: February 23, 2026
**From**: neuralSpring → ToadStool / BarraCUDA
**Sessions**: 45–46
**ToadStool HEAD**: `6ee71f07` + 2 local fixes pending absorption (mean_reduce, chi²)
**Previous**: V13 (Session 44 — multi-GPU portability, benchmarks, bug fixes)
**License**: AGPL-3.0-or-later

---

## Executive Summary

Sessions 45–46 achieved **pure GPU promotion** for ~90% of neuralSpring's
production math. 38 previously CPU-bound operations now dispatch to GPU via
a new capability-based `Dispatcher` layer built on the BarraCUDA `Tensor` API.
Validated on both RTX 4070 (proprietary Vulkan) and TITAN V (NVK open-source).

**Key numbers**: 133/133 PASS. 38 CPU→GPU promotions (27 Phase A + 11 Phase B).
47/47 GPU promotion checks. ~90% of production math on GPU. ~10% remains
CPU-only (ODE loops, FST, introgression chain, argmax).

---

## Part 1: Architecture — gpu_ops + gpu_dispatch

### Design

```
                          ┌─────────────────┐
                          │   Dispatcher     │
                          │  (gpu_dispatch)  │
                          └────┬────────┬────┘
                               │        │
                    GPU avail?  │        │  No GPU
                               ▼        ▼
                          ┌─────────┐ ┌──────┐
                          │ gpu_ops │ │ CPU  │
                          │ (Tensor)│ │ math │
                          └─────────┘ └──────┘
```

- `gpu_dispatch::Dispatcher` — runtime capability detection, zero configuration
- `gpu_ops` — 38 GPU-accelerated functions using `barracuda::tensor::Tensor`
- CPU fallbacks in `gpu_dispatch` for all operations
- All operations validated against CPU references within f32→f64 tolerance

### Tensor API Usage Patterns

| Pattern | Operations | Tensor Methods |
|---------|-----------|----------------|
| GEMV/GEMM | hmm_forward, hmm_backward, viterbi, replicator, neural_forward | `matmul` (consumes self) |
| Elementwise | pearson, diversity, nucleotide_diversity, hill_activation | `mul`, `sub`, `add` (borrow) |
| Reductions | variance, mean, allele_freq, log_likelihood | `sum`, `mean`, `sum_dim`, `mean_dim` |
| Transcendental | hill_activation (x^n), softmax, logsumexp | `log_wgsl`, `exp_wgsl`, `sqrt_wgsl` (consume) |
| Broadcast | viterbi score matrix | `broadcast` (consumes self) |
| Dimension reduction | allele_freq column-sum, viterbi max | `sum_dim(0)`, `max_dim(0)` |

---

## Part 2: Phase A — 27 Operations (Session 45)

All standard Tensor-compatible operations promoted to GPU dispatch:

| Category | Operations | Count |
|----------|-----------|-------|
| Linear algebra | matmul, transpose, frobenius_norm | 3 |
| Statistics | pearson_correlation, variance, mean, chi_squared | 4 |
| Distance | l2_distance, hamming_distance, jaccard_similarity, pairwise_distances | 4 |
| ML inference | neural_forward, softmax, pca_project | 3 |
| HMM | hmm_forward_step | 1 |
| ODE | rk4_step | 1 |
| Evolution | fitness_evaluation, batch_fitness, diversity_metrics | 3 |
| Genomics | tree_distance, geographic_distances | 2 |
| Reductions | logsumexp, log_likelihood | 2 |
| Bio ops | locus_variance, spatial_payoff | 2 |
| Other | hill_activation_batch, allele_frequencies | 2 |

**Validator**: `validate_gpu_promotion` — 27/27 PASS (both GPUs)

---

## Part 3: Phase B — 11 Operations (Session 46)

The harder cases requiring multi-step GPU pipelines or hybrid GPU/CPU approaches:

### HMM Backward Step
- GPU GEMV: `β_{t+1} ⊙ emit → weighted @ A^T / scale`
- Uses: `Tensor::mul`, `transpose`, `matmul`
- Fully GPU except for the scalar division by scale factor

### HMM Viterbi Step
- GPU score matrix: `δ_{t-1}` broadcast → add `log A` → `max_dim(0)`
- CPU argmax (BarraCUDA `max_dim` returns values, not indices)
- **ToadStool opportunity**: `argmax_dim()` would make this fully GPU

### Meta-Population Statistics (6 ops)
- `allele_frequencies_gpu`: column `sum_dim(0)` + `div_scalar`
- `nucleotide_diversity_gpu`: allele freq → `mul` → `sub` → `mul_scalar` → `mean`
- `matrix_correlation_gpu`: upper-triangle extract → `pearson_correlation_gpu`
- `geographic_distance_matrix_gpu`: pairwise Euclidean via `l2_distance_gpu`
- `thermal_diversity_correlation_gpu`: direct `pearson_correlation_gpu`
- `inter_population_af_variance_gpu`: per-pop allele freq → `variance_gpu` → `mean_gpu`

### Replicator Dynamics
- 2×2 payoff GEMV via `Tensor::matmul([1,2] × [2,2])`
- Nonlinear update `x + dt*x*(f - f̄)` on CPU (custom WGSL needed for full GPU)

### Hill Activation (Refactored)
- Previously: CPU compute, upload to GPU (pseudo-GPU)
- Now: genuine GPU pipeline: `log_wgsl → mul_scalar → exp_wgsl → add → div → mul_scalar`
- Guard: `x.max(1e-30)` prevents `ln(0)`

**Validator**: `validate_gpu_phase_b` — 20/20 PASS (both GPUs)

---

## Part 4: Tensor API Findings for ToadStool

### New API Requests (Priority Order)

| # | Request | Motivation | Workaround |
|---|---------|-----------|------------|
| 1 | `Tensor::argmax_dim(axis)` | HMM Viterbi requires argmax, not just max | CPU argmax after `max_dim` readback |
| 2 | `Tensor::pow_scalar(n)` | Hill activation `x^n` | `exp(n * ln(x))` pipeline |
| 3 | `Tensor::softmax_dim(axis)` | Row-wise attention softmax | `ScaledDotProductAttention` or manual per-row |
| 4 | `Tensor::div(&other)` | Element-wise division | `mul` by reciprocal or manual |

### Ownership Clarification Needed in Docs

Methods that **consume** `self` (must clone to reuse):
`matmul`, `softmax`, `sigmoid`, `gelu_wgsl`, `log_wgsl`, `exp_wgsl`, `sqrt_wgsl`, `broadcast`

Methods that **borrow** `&self` (can reuse):
`transpose`, `add`, `sub`, `mul`, `sum`, `mean`, `max`, `norm`, `mul_scalar`,
`add_scalar`, `div_scalar`, `sum_dim`, `mean_dim`, `max_dim`, `min_dim`, `reshape`, `to_vec`

### Numerical Stability Notes

| Computation | Guard Applied | Reason |
|-------------|--------------|--------|
| `ln(x)` in Hill activation | `x.max(1e-30)` | Prevents `-inf` |
| HMM scale factor | `scale.abs() < LOG_GUARD → LOG_GUARD` | Prevents division by zero |
| Log-domain HMM | All log probabilities | Underflow prevention |

---

## Part 5: Remaining Work (~10% CPU-Only)

| Operation | GPU Blocker | Suggested Solution |
|-----------|------------|-------------------|
| Full ODE loops (integrate_ode, integrate_grn) | Sequential time-stepping with state dependency | `StatefulPipeline` + batched encoder, GPU PRNG for stochastic terms |
| FST variance decomposition | Multi-step between/within variance | Custom `fst_decompose.wgsl` shader |
| Introgression HMM chain | Full forward → backward → Viterbi sequence | Compose existing `hmm_*_step_gpu` functions |
| Viterbi argmax | `max_dim` returns values only | `argmax_dim()` in Tensor API |

---

## Part 6: Cross-Spring Relevance

### For hotSpring
- `gpu_dispatch::Dispatcher` pattern applicable to physics workloads
- Tensor-based GEMV for HMM backward/Viterbi is directly reusable for transfer matrix methods
- Hill activation GPU pipeline (`exp(n*ln(x))`) useful for power-law physics

### For wetSpring
- HMM backward/Viterbi GPU steps directly applicable to metagenomics HMM chains
- Allele frequency column-sum and nucleotide diversity — population genetics on GPU
- `inter_population_af_variance_gpu` ready for ecological metagenomic comparisons

### For ToadStool
- 38 new usage patterns for the Tensor API documented in this handoff
- 4 new API requests (argmax_dim, pow_scalar, softmax_dim, div)
- Ownership documentation gap identified and catalogued
- NVK compatibility re-confirmed for all new operations

---

## Part 7: Validation Summary

| Validator | Checks | RTX 4070 | TITAN V (NVK) |
|-----------|--------|----------|---------------|
| `validate_gpu_promotion` | 27 | **27/27 PASS** | **27/27 PASS** |
| `validate_gpu_phase_b` | 20 | **20/20 PASS** | **20/20 PASS** |
| `validate_all` | 133 | **133/133 PASS** | **133/133 PASS** |

**Grand total**: 1800+ checks, ALL GREEN, ~90% production math on GPU.

---

## Related Handoffs

- **V13** (Session 44): Multi-GPU portability, benchmarks, 2 upstream bug fixes
- **V12** (Session 43): Upstream expansion, mixed-hardware dispatch
- **V11** (Session 42): ToadStool sync, deep audit
- `specs/TOADSTOOL_HANDOFF.md`: Full shortcoming tracker (S-01 through S-16)
- `specs/PURE_GPU_ROADMAP.md`: Phase A complete, Phase B partial, Phase C pending
