# neuralSpring → ToadStool/BarraCUDA Comprehensive Evolution Handoff V59

**Date**: February 27, 2026
**From**: neuralSpring (ML/neuroevolution validation)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Full neuralSpring evolution surface — what to absorb, what evolved, what helped

---

## Executive Summary

- neuralSpring exercises **16 barracuda submodules** across **177 files** with **124 import sites**
- **177 binaries**, **177/177 validate_all**, **668 lib tests**, **3111+ checks**
- BarraCUDA CPU is **83.6× faster** than Python/NumPy (geomean, 11 domains)
- CPU→GPU portability **proven** (9/9 parity checks, 7 domains)
- **42 upstream rewires** — neuralSpring local code replaced by barracuda APIs
- **42 metalForge WGSL shaders** (15 sovereign folding df64, 17 domain-specific)
- Cross-spring shader evolution tracked: hotSpring, wetSpring, neuralSpring all contribute

---

## Part 1: How neuralSpring Uses BarraCUDA

### Submodules Exercised (16 total)

| Submodule | What neuralSpring Uses It For | Key APIs |
|-----------|------------------------------|----------|
| `barracuda::ops::bio` | GPU dispatch for 15+ domain-specific operations | `PairwiseL2Gpu`, `BatchFitnessGpu`, `HmmBatchForwardF64`, `SpatialPayoffGpu`, `PairwiseHammingGpu`, `PairwiseJaccardGpu`, `DiversityFusionGpu`, `WrightFisherGpu`, `GillespieGpu`, `SwarmNnGpu`, `StencilCooperationGpu`, `LocusVarianceGpu`, `MultiObjFitnessGpu` |
| `barracuda::ops::linalg` | GPU eigensolve | `BatchedEighGpu` |
| `barracuda::ops::logsumexp` | Numerically stable log-sum-exp | `LogSumExp` |
| `barracuda::ops::pairwise_distance` | General pairwise distance | `PairwiseDistance` (L1/L2/Lp) |
| `barracuda::ops::fft` | Frequency-domain analysis | FFT ops |
| `barracuda::ops::mha` | Multi-head attention | `MultiHeadAttention` |
| `barracuda::spectral` | Anderson localization, IPR | `BatchIprGpu`, `disorder_sweep_gpu` |
| `barracuda::tensor` | GPU-resident N-d arrays | `Tensor`, matmul, transpose, sigmoid, tanh, softmax |
| `barracuda::device` | GPU device management | `WgpuDevice`, `Fp64Strategy`, `GpuDriverProfile` |
| `barracuda::stats` | CPU statistics | variance, pearson, shannon, hill, mae, fit_linear, spearman, bootstrap |
| `barracuda::special` | Special math functions | chi_squared, gamma, erf, bessel, Legendre, Hermite |
| `barracuda::linalg` | CPU linear algebra | `eigh_f64` |
| `barracuda::numerical` | ODE solvers | `rk45_solve`, `Rk45Config` |
| `barracuda::dispatch` | CPU↔GPU routing | `matmul_dispatch`, `transpose_dispatch`, `DispatchTarget` |
| `barracuda::staging` | Stateful GPU pipelines | `StatefulPipeline`, `KernelDispatch` |
| `barracuda::unified_hardware` | Cross-substrate routing | `BandwidthTier`, `route`, `discovery` |
| `barracuda::pipeline` | Reduce pipelines | `ReduceScalarPipeline` |

### Functions Rewired to Upstream (42)

neuralSpring evolved local implementations that were subsequently absorbed into
barracuda. Each rewire replaces local code with the upstream API:

**Core math (9)**: matmul, transpose, frobenius_norm, softmax, l2_distance,
mean, variance, pearson_correlation, shannon_entropy

**Domain-specific (15)**: graph_laplacian, disordered_laplacian,
belief_propagation_chain, numerical_hessian, fst_single_locus, pairwise_fst_full,
softmax_row_wise, Viterbi argmax, mae, hill_activation, hill_repression,
complexity_metric (fit_linear), spectral_entropy

**GPU ops (3 modern S88+)**: pairwise_l2_matrix_gpu → PairwiseL2Gpu,
geographic_distance_matrix_gpu → PairwiseL2Gpu, disorder_sweep_gpu IPR → BatchIprGpu

**Shader sources (6)**: batch_fitness_eval, pairwise_l2, spatial_payoff, batch_ipr,
hmm_forward_log, mean_reduce — all now use upstream `barracuda::shaders::*` constants

---

## Part 2: Performance Proof — Why BarraCUDA Matters

### CPU: Pure Rust vs Python/NumPy (83.6× geomean)

| Domain | Papers | Python µs | Rust µs | Speedup |
|--------|--------|-----------|---------|---------|
| Multi-Obj Fitness | 014 | 3,020 | 3 | **1,104×** |
| NK Fitness | 011 | 14,682 | 18 | **821×** |
| Pairwise L2 | 012 | 119 | 0.4 | **315×** |
| Swarm NN | 015 | 11,239 | 39 | **290×** |
| Replicator | 019 | 36,659 | 151 | **243×** |
| Hill Gate | 021 | 527 | 3 | **212×** |
| HMM Forward | 016-018 | 13,138 | 84 | **157×** |
| RK4 GRN | 020 | 25,567 | 375 | **68×** |
| Jaccard | 024 | 2,110 | 142 | **15×** |
| Hamming | 017 | 430 | 35 | **13×** |
| Commutator | 022 | 24 | 84 | 0.3×† |

†Commutator: NumPy delegates 64×64 matmul to optimized BLAS. Pure Rust matmul
is intentionally naive for portability. **toadStool action**: consider BLAS-backed
small-matrix fast-path for `matmul_dispatch` — this would close the one domain
where interpreted language outperforms.

### GPU: Portability Proven (9/9)

Same math produces identical results at CPU and GPU tiers:
- HMM Forward: GPU-CPU diff 1.6e-7
- Batch Fitness: diff 0.0e0 (bit-identical f64)
- Pairwise L2: rel diff 7.9e-9
- Pairwise Hamming: rel diff 2.4e-8
- Dispatcher variance: diff 9.8e-5 (f32 GPU Welford)

---

## Part 3: Cross-Spring Shader Evolution

neuralSpring's validated primitives flow into ToadStool alongside contributions
from all Five Springs:

```
hotSpring (precision physics)
├── df64 core streaming (f64 I/O → df64 compute on f32 cores)
├── Welford variance (precision accumulation)
├── LogSumExp (log-domain HMM stability)
├── Jacobi eigensolve (BatchedEighGpu)
└── rk45_adaptive.wgsl (Dormand-Prince)

wetSpring (bioinformatics)
├── Shannon + Simpson + Pielou (DiversityFusionGpu)
├── HMM forward/backward/Viterbi (hmm_forward_log.wgsl)
├── Bray-Curtis distance
├── 16S taxonomy classification
└── UniFrac tree propagation

neuralSpring (ML/neuroevolution)          ← THIS HANDOFF
├── Pairwise L2 (PairwiseL2Gpu, from MODES novelty search)
├── Batch IPR (BatchIprGpu, from Anderson localization)
├── Batch fitness evaluation (NK landscapes)
├── Swarm NN forward (heterogeneous controllers)
├── Multi-head attention (Evoformer AlphaFold2)
├── 15 sovereign folding df64 shaders (GELU, LayerNorm, SDPA, IPA, etc.)
└── ESN reservoir computing patterns

airSpring (atmospheric)
├── RMSE, R², NSE, MAE (accuracy metrics)
├── fit_linear / moving_window
└── Ensemble statistics

groundSpring (hydrology)
├── Multinomial sampling
├── Monte Carlo propagation
└── Spectral reconstruction
```

**toadStool action**: The cross-spring lineage document at
`whitePaper/CROSS_SPRING_SHADER_LINEAGE.md` tracks which Spring originated
each shader. This provenance is valuable for the ToadStool shader catalog.

---

## Part 4: What neuralSpring Evolved That ToadStool Should Absorb

### Priority 1: Sovereign Folding df64 Shaders (15 shaders)

These are production-quality df64 shaders for protein structure prediction
(AlphaFold2 Evoformer + Structure Module). All validated GPU vs CPU.

| Shader | Algorithm | Max GPU-CPU diff |
|--------|-----------|------------------|
| `gelu_f64.wgsl` | GELU activation | 3.41e-4 |
| `triangle_mul_outgoing_f64.wgsl` | Algorithm 11 | 3.10e-7 |
| `triangle_mul_incoming_f64.wgsl` | Algorithm 12 | 4.66e-7 |
| `sdpa_scores_f64.wgsl` | QKᵀ/√d attention | 6.76e-8 |
| `softmax_f64.wgsl` | Row-wise softmax | 2.92e-4 |
| `attention_apply_f64.wgsl` | Weighted value sum | 6.89e-8 |
| `layer_norm_f64.wgsl` | Layer normalization | 5.58e-7 |
| `sigmoid_f64.wgsl` | Sigmoid gate | CPU validated |
| `outer_product_mean_f64.wgsl` | MSA→pair bridge | 6.43e-8 |
| `msa_row_attention_scores_f64.wgsl` | Row attention + bias | 1.06e-7 |
| `msa_col_attention_scores_f64.wgsl` | Column attention | 9.57e-8 |
| `ipa_scores_f64.wgsl` | SE(3)-equivariant IPA | 3.40e-7 |
| `backbone_update_f64.wgsl` | Frame composition | 3.59e-8 |
| `torsion_angles_f64.wgsl` | Fused ResNet + normalize | 1.10e-7 |
| `triangle_attention_f64.wgsl` | Algorithms 13-14 | 1.54e-7 |

**toadStool action**: These shaders use `df64_core.wgsl` + `df64_transcendentals.wgsl`
(already in ToadStool). Absorbing them gives ToadStool a complete AlphaFold2
primitive set for sovereign protein structure prediction.

### Priority 2: Phase 4 WGSL Shaders (4 shaders)

| Shader | Validated | What It Does |
|--------|-----------|-------------|
| `hmm_backward_log.wgsl` | 22/22 | HMM backward pass (log-domain) |
| `hmm_viterbi.wgsl` | 22/22 | Viterbi decoding |
| `matrix_correlation.wgsl` | 22/22 | Matrix correlation coefficient |
| `linear_regression.wgsl` | 22/22 | Linear regression (slope/intercept) |

### Priority 3: Streaming Pipeline Pattern

neuralSpring validated ToadStool's unidirectional streaming pattern:
- Batch eigensolve → IPR → statistics (8 Hamiltonians, 28/28 PASS)
- Anderson disorder sweep (6 W values, IPR 0.09→0.79)
- Dispatcher CPU↔GPU parity (diff 1.6e-14)

**toadStool action**: The streaming pattern reduces round-trips and is proven
correct for scientific workloads. Consider documenting this as a first-class
ToadStool usage pattern.

---

## Part 5: Dispatcher Evolution

neuralSpring's `gpu_dispatch::Dispatcher` routes 44 operations to the optimal
substrate (GPU when available, CPU fallback). Key patterns:

1. **Capability-based routing**: `GpuDriverProfile` + `Fp64Strategy` determine
   whether to use native f64, df64 emulation, or CPU fallback
2. **Transparent overhead**: ≤1.04× for 9/10 ops via `Dispatcher::cpu_only()`
3. **Domain heuristics**: `MixedSubstrate` routes by workload size and PCIe cost

**toadStool action**: neuralSpring's Dispatcher demonstrates the ideal consumer
of barracuda's dispatch infrastructure. The capability detection and domain
heuristic patterns are worth studying for the dispatch API design.

---

## Part 6: Validation Infrastructure Worth Studying

| Pattern | What It Does | Where |
|---------|-------------|-------|
| `ValidationHarness` | Structured pass/fail with exit codes | `src/validation.rs` |
| `tolerance_registry!` macro | Centralized named tolerances (129+) | `src/tolerances.rs` |
| `baseline_path()` | Workspace-relative baseline resolution | `src/validation.rs` |
| `check_drift.sh` | Python baseline regression detection (34 baselines) | `control/` |
| `require!` macro | Graceful GPU error handling in validators | `src/validation.rs` |
| Cross-language parity | JSON-based Python→Rust verification | `control/generate_cpu_references.py` |

---

## Part 7: Recommendations

1. **BLAS small-matrix fast-path**: Close the commutator gap (0.3× vs NumPy)
2. **Sovereign folding absorption**: 15 df64 shaders ready for upstream
3. **Streaming pattern documentation**: First-class ToadStool usage pattern
4. **Cross-spring lineage**: Maintain shader provenance as catalog grows
5. **CPU benchmark suite**: neuralSpring's 11-domain benchmark pattern is
   portable to any Spring — consider standardizing as a ToadStool benchmark

---

## Validation Matrix

| Metric | Count |
|--------|-------|
| Total binaries | 175 |
| validate_all | 174/175 PASS |
| Library tests | 668 |
| Total checks | 3034+ |
| Barracuda import sites | 124 |
| Files using barracuda | 177 |
| Barracuda submodules | 16 |
| Upstream rewires | 42 |
| metalForge WGSL shaders | 42 |
| Named tolerances | 129+ |
| Python baselines | 263 checks (34 drift baselines) |
| CPU vs Python speedup | 83.6× geomean |
| GPU portability | 9/9 |
| Multi-GPU (RTX 4070 + Titan V) | Bit-identical |

---

*AGPL-3.0-or-later*
