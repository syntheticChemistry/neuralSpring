# neuralSpring v9 → ToadStool / BarraCUDA Team Handoff

**Date:** February 22, 2026 (Session 40 — capability-based dispatch + cross-eigensolver)
**From:** neuralSpring (ML validation & evolutionary computation biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Supersedes:** `archive/NEURALSPRING_V8_TOADSTOOL_BARRACUDA_HANDOFF_FEB22_2026.md`
**ToadStool HEAD:** `d45fdfb3` (Session 39)

---

## Executive Summary

neuralSpring has completed its full validation stack: **25 papers, 1607+ checks,
119 binaries, 258 lib tests, 17 WGSL shaders (13 upstream, 4 local)**. This handoff
documents the full BarraCUDA integration surface, what ToadStool should absorb next,
and architectural insights for BarraCUDA's evolution.

Session 40 introduced **capability-based GPU dispatch** (`Gpu::dispatch_1d`) across
12 validators + the evolved HMM module, and **cross-eigensolver validation** proving
dense Householder+QR agrees with tridiag Sturm bisection to machine epsilon
(2.89e-15 at n=64). Spectral theory validator now at 17/17 PASS.

| Category | Status |
|----------|--------|
| Python controls | 25/25, 206 checks | ALL PASS |
| Rust CPU | 258 lib tests + 119 binaries | ALL PASS |
| BarraCUDA CPU | 24/25 papers (96%) | ALL GREEN |
| BarraCUDA GPU Tensor | 23/25 papers (92%) | ALL GREEN |
| metalForge WGSL | 15/25 papers, 17 shaders | ALL PASS |
| GPU Pipeline | 15/25 papers | ALL PASS |
| Cross-dispatch | 15/15 Phase 0++ papers | ALL GREEN |
| Upstream parity | 6/6 dual-path, 0.00e0 diff | Bit-identical |
| Spectral theory | 17/17 checks | ALL PASS |
| Capability dispatch | 12 validators + HMM evolved | Runtime-validated |

---

## Part 1: BarraCUDA Integration Surface (48 files, 20+ API categories)

neuralSpring imports from barracuda in 48 source files. This is the full surface:

### 1.1 Core Infrastructure

| API | neuralSpring Usage | Files |
|-----|-------------------|-------|
| `device::WgpuDevice` | GPU context (`Gpu::new`), adapter selection, shader compilation | `gpu.rs`, 20+ validators |
| `tensor::Tensor` | GPU tensor ops (matmul, transpose, tanh, sigmoid, dot, etc.) | 28 validators |
| `error::BarracudaError` | Error propagation in evolved MHA | `evolved/mha.rs` |

### 1.2 Bio Operations (GPU Wrappers)

| API | Papers | Validated | Parity |
|-----|--------|-----------|--------|
| `ops::bio::BatchFitnessGpu` | 011-015 | 12/12 PASS | 0.00e0 (bit-identical) |
| `ops::bio::PairwiseHammingGpu` | 017 | 6/6 PASS | 0.00e0 |
| `ops::bio::PairwiseJaccardGpu` | 024 | 7/7 PASS | 0.00e0 |
| `ops::bio::LocusVarianceGpu` | 025 | 8/8 PASS | 0.00e0 |
| `ops::bio::SpatialPayoffGpu` | 019 | 6/6 PASS | 0.00e0 |
| `ops::bio::HmmBatchForwardF64` | 016-018 | 11/11 PASS | 2.47e-10 diff (f64) |

### 1.3 Spectral Theory (hotSpring lineage)

| API | Usage | Validated |
|-----|-------|-----------|
| `spectral::BatchIprGpu` | Anderson localization IPR | 7/7 PASS, 0.00e0 parity |
| `spectral::find_all_eigenvalues` | Sturm bisection eigensolver | 17/17 PASS |
| `spectral::lanczos` / `lanczos_eigenvalues` | Sparse Lanczos eigensolver | 2D/3D validated |
| `spectral::anderson_hamiltonian` | Anderson disorder model | Bandwidth, eigenvalue count |
| `spectral::almost_mathieu_hamiltonian` | Aubry-André quasiperiodic model | Cross-validated vs Jacobi |
| `spectral::hofstadter_butterfly` | Hofstadter fractal spectrum | 21 α, 2100 eigenvalues |
| `spectral::lyapunov_exponent` / `lyapunov_averaged` | Localization measure | Kappus-Wegner anomaly verified |
| `spectral::level_spacing_ratio` | GOE vs Poisson statistics | Extended/localized phases |
| `spectral::detect_bands` | Gap detection in spectra | Gapped spectrum test |

### 1.4 Pipeline Infrastructure

| API | Usage | Validated |
|-----|-------|-----------|
| `pipeline::ReduceScalarPipeline` | f64 GPU scalar reduction (mean IPR) | 5.55e-17 diff |
| `staging::StatefulPipeline` | Iterative GPU compute chains | 10/10 PASS |
| `dispatch::dispatch_for` | CPU↔GPU routing | 49 cross-dispatch checks |

### 1.5 CPU Math Primitives

| API | Papers | Key Finding |
|-----|--------|-------------|
| `stats::variance` / `pearson_correlation` | All 15 Phase 0++ | Machine-precision vs hand-rolled |
| `linalg::eigh_f64` (Householder+QR) | 022-023 | 1.75e-14 at n=32 |
| `linalg::solve_f64` | 016, 015 | Machine precision |
| `numerical::rk45_solve` | 019-021 | Machine precision vs RK4 |
| `special::chi_squared_sf/cdf` | 018 | Correct LRT p-values |
| `optimize::nelder_mead/bisect/brent` | Validation | 10/10 PASS |
| `ops::fft::Fft1D/Fft1DF64` | Spectral validation | 24/24 PASS |
| `ops::logsumexp::LogSumExp` | HMM numerics | 5/5 PASS |

### 1.6 f64 GPU Operations

| API | Usage | Validated |
|-----|-------|-----------|
| `ops::variance_reduce_f64::VarianceReduceF64` | GPU statistics | mean, var, std, pop_var, pop_std |
| `ops::sum_reduce_f64::SumReduceF64` | GPU summation | 35 total f64 checks |
| `ops::norm_reduce_f64::NormReduceF64` | GPU norms | |
| `ops::cosine_similarity_f64::CosineSimilarityF64` | GPU cosine sim | |
| `ops::fused_map_reduce_f64::FusedMapReduceF64` | Fused GPU map+reduce | |
| `ops::max_abs_diff_f64::MaxAbsDiffF64` | GPU max abs diff | |
| `ops::weighted_dot_f64::WeightedDotF64` | GPU weighted dot | |

### 1.7 WGSL Shader Constants

neuralSpring imports shader WGSL source from barracuda for validation:

| Shader Constant | Source Module |
|-----------------|--------------|
| `ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32` | HMM forward pass |
| `ops::bio::batch_fitness::WGSL_BATCH_FITNESS_EVAL` | Fitness evaluation |
| `ops::rk_stage::WGSL_RK4_PARALLEL` | RK4 ODE integration |
| `ops::bio::pairwise_jaccard::WGSL_PAIRWISE_JACCARD` | Jaccard distance |
| `ops::bio::pairwise_hamming::WGSL_PAIRWISE_HAMMING` | Hamming distance |
| `ops::bio::locus_variance::WGSL_LOCUS_VARIANCE` | Locus variance |
| `ops::bio::spatial_payoff::WGSL_SPATIAL_PAYOFF` | Spatial payoff |
| `spectral::batch_ipr::WGSL_BATCH_IPR` | Inverse participation ratio |

---

## Part 2: What ToadStool Should Absorb

### 2.1 Capability-Based Dispatch Pattern (Priority: High)

neuralSpring introduced `GpuCapabilities` for runtime hardware discovery and
`Gpu::dispatch_1d()` for validated dispatch. The pattern:

```rust
// Validates shader workgroup_size against hardware limits, clamps dispatch
let wg_count = gpu.dispatch_1d(n_items, 256); // panics if hw limit < 256
pass.dispatch_workgroups(wg_count, 1, 1);
```

**Recommendation**: BarraCUDA's bio-op wrappers (`BatchFitnessGpu`, etc.) should
query `device.limits()` and validate that the shader's `@workgroup_size` is
supported, rather than assuming 256 is always valid. This matters for:
- WebGPU targets (mobile, browser) with smaller workgroup limits
- CPU adapters (llvmpipe) with constrained dispatch limits
- Future NPU targets

### 2.2 Remaining Local Shaders (4 shaders, Priority: Medium)

| Shader | Domain | Blocker | Suggested Fix |
|--------|--------|---------|---------------|
| `head_split.wgsl` | MHA | S-03b projection hang | Replace fused projection dispatch with matmul + head_split |
| `head_concat.wgsl` | MHA | S-03b projection hang | Replace fused concat dispatch with head_concat + matmul |
| `xoshiro128ss.wgsl` | GPU PRNG | No upstream module | New `barracuda::ops::prng` module |
| `swarm_nn_scores.wgsl` | Swarm (015) | No upstream equivalent | New shader or generalize `batch_fitness` |

### 2.3 S-03b MHA Projection Fix (Priority: High)

The native `multi_head_attention` in BarraCUDA has a dispatch bug: the z-dimension
dispatch divides by 16 instead of 1 for both `project_with_head_split` and
`concat_and_project`. neuralSpring's workaround decomposes MHA into:
- `matmul` for Q/K/V/O projections (correct)
- `attention` for SDPA (correct)
- CPU-side head split/concat (slow but correct)

**Fix in barracuda**: In `barracuda/src/ops/mha/projections.rs`, change the
z-dimension from `div_ceil(16)` to `div_ceil(1)` for both projection shaders.
Then neuralSpring can retire `evolved::mha` and the `head_split/head_concat` shaders.

### 2.4 S-15 Matmul Hang (Priority: Critical)

Elements with magnitude ≤ 0.1 trigger a WGPU/Vulkan driver hang on RTX 4070.
This affects ALL matmul tiers. Root-caused to a driver-level interaction with
IEEE 754 bit patterns, not a WGSL logic error.

**Workaround**: All validators use data ≥ 0.5.
**Investigation path**: Test on AMD/Intel to isolate NVIDIA-specific behavior.
File an upstream `wgpu` or NVIDIA driver bug report with minimal repro.

### 2.5 Cross-Eigensolver Validation Data

Session 40 proved dense Householder+QR and tridiag Sturm bisection agree:
- n=64 W=3: max eigval diff **2.89e-15** (machine epsilon)
- n=200 W=6: max eigval diff **1.42e-14**

This validates that `barracuda::ops::linalg::eigh_householder_qr` and
`barracuda::spectral::find_all_eigenvalues` can be used interchangeably on
tridiagonal matrices. The Sturm method is O(n) per eigenvalue (vs O(n³) for QR),
making it the better choice for large sparse systems.

**Recommendation**: Consider adding a `barracuda::spectral::eigh_tridiag` alias
that dispatches to the fastest method based on problem size and structure.

### 2.6 Data Layout Gotchas (Document in BarraCUDA)

| API | Layout Requirement | Discovered By |
|-----|-------------------|---------------|
| `PairwiseJaccardGpu` | **Column-major** PA: `pa[gene * n_genomes + genome]` | neuralSpring Exp 008 |
| `BatchIprGpu` | Returns raw `Σ|ψ_i|⁴` (NOT reciprocal `1/Σ|ψ_i|⁴`) | neuralSpring Exp 008 |
| `HmmBatchForwardF64` | Batch dimension first: `[batch, states]` | neuralSpring Exp 008 |

These should be documented in the BarraCUDA API docs to prevent other consumers
from hitting the same discovery cost.

---

## Part 3: What We Learned for BarraCUDA Evolution

### 3.1 The Validation Progression Proves Math Portability

Every paper passes through 7 tiers. Each tier proves a correctness property:

| Tier | What It Proves |
|------|---------------|
| Py (Python) | The science is correct and reproducible |
| Rs (Rust CPU) | The math translates to type-safe Rust |
| bC (BarraCUDA CPU) | Pure Rust primitives reproduce hand-rolled math |
| gT (GPU Tensor) | Math is portable CPU → GPU via Tensor API |
| mF (metalForge WGSL) | Domain-specific GPU kernels produce correct results |
| gP (GPU Pipeline) | Multi-kernel chains compose correctly |
| xD (Cross-dispatch) | CPU ↔ GPU routing preserves correctness |

This progression is the **strongest possible argument** for BarraCUDA's correctness:
206 independent Python baselines, reproduced in Rust, then validated at every
GPU abstraction layer, across 25 papers from 5 scientific disciplines.

### 3.2 Cross-Spring Evolution Is Real

| Flow | What Happened | Precision Gain |
|------|---------------|---------------|
| nS f32 HMM → TS → wS f64 batch → TS → nS | neuralSpring evolved f32, wetSpring evolved f64 batch, neuralSpring validates both | 10⁹× (f32 → f64) |
| hS spectral → TS → nS validates | hotSpring contributed spectral theory, neuralSpring validates against analytical results | 17/17 PASS |
| nS Householder+QR → TS → nS+hS use | neuralSpring contributed dense eigensolver, both Springs benefit | 1.75e-14 at n=32 |

### 3.3 Upstream Wrapper Overhead Is Negligible

Benchmarked 6 bio ops: local `include_str!` dispatch vs upstream barracuda wrapper.
Median overhead < 5%. One op (Jaccard) is 8% faster via upstream.

| Bio Op | Local→Upstream Ratio |
|--------|---------------------|
| BatchFitness | 1.16× |
| PairwiseHamming | 1.03× |
| PairwiseJaccard | 0.92× (faster!) |
| LocusVariance | 1.12× |
| SpatialPayoff | 0.96× |
| BatchIpr | 1.03× |

### 3.4 GPU Dispatch Has a 1.5ms Crossover

Below 1.5ms of CPU work, CPU is faster than GPU due to dispatch overhead.
This is codified in `barracuda::dispatch::dispatch_for()`. Key implication:
BarraCUDA should NOT automatically promote small workloads to GPU.

### 3.5 The Six Isomorphic Primitives

All 25 papers decompose into six fundamental ops. BarraCUDA already has
high-quality implementations of all six:

| Primitive | BarraCUDA Module | Status |
|-----------|-----------------|--------|
| GEMM | `Tensor::matmul`, 4-tier KernelRouter | Validated (S-15 workaround) |
| Attention | `Tensor::attention` | Validated (S-03b workaround for MHA) |
| Normalization | `Tensor::layer_norm_wgsl`, `log_softmax_wgsl` | Validated |
| Nonlinearity | `Tensor::relu/gelu/silu/tanh/sigmoid` | Validated (90/90 PASS) |
| Reduction | `ReduceScalarPipeline`, `VarianceReduceF64` | Validated (5.55e-17 diff) |
| Gating | Hill function, sigmoid gating | Validated via `hill_gate.wgsl` |

---

## Part 4: Shader Absorption Status

### 4.1 Absorbed (Identical — `77f70b2e`)

| Shader | Upstream API | Rust Wrapper |
|--------|-------------|--------------|
| `hmm_forward_log.wgsl` | `WGSL_HMM_FORWARD_LOG_F32` | `HmmBatchForwardF64` |
| `batch_fitness_eval.wgsl` | `WGSL_BATCH_FITNESS_EVAL` | `BatchFitnessGpu` |
| `rk4_parallel.wgsl` | `WGSL_RK4_PARALLEL` | — |
| `pairwise_jaccard.wgsl` | `WGSL_PAIRWISE_JACCARD` | `PairwiseJaccardGpu` |
| `pairwise_hamming.wgsl` | `WGSL_PAIRWISE_HAMMING` | `PairwiseHammingGpu` |
| `locus_variance.wgsl` | `WGSL_LOCUS_VARIANCE` | `LocusVarianceGpu` |
| `spatial_payoff.wgsl` | `WGSL_SPATIAL_PAYOFF` | `SpatialPayoffGpu` |
| `batch_ipr.wgsl` | `WGSL_BATCH_IPR` | `BatchIprGpu` |

### 4.2 Absorbed (Generalized — `d45fdfb3`)

| Shader | Upstream Improvement |
|--------|---------------------|
| `pairwise_l2.wgsl` | O(1) pair decode |
| `multi_obj_fitness.wgsl` | Bessel correction |
| `hill_gate.wgsl` | Mode generalization |
| `swarm_nn_forward.wgsl` | Generic MLP, clamped sigmoid |
| `mean_reduce.wgsl` | Effectively identical |

### 4.3 Still Local (4 shaders)

| Shader | Blocker | Absorption Target |
|--------|---------|-------------------|
| `head_split.wgsl` | S-03b | `barracuda::ops::mha` |
| `head_concat.wgsl` | S-03b | `barracuda::ops::mha` |
| `xoshiro128ss.wgsl` | No module | `barracuda::ops::prng` |
| `swarm_nn_scores.wgsl` | No equivalent | `barracuda::ops::bio::swarm` |

---

## Part 5: Capability-Based Dispatch (Session 40)

### What Changed

All 12 core GPU validators and the evolved `hmm_forward_gpu` module now use
`Gpu::dispatch_1d()` instead of hardcoded `.div_ceil(256)`. At startup,
validators log discovered capabilities:

```
capabilities: wg_x=256, dispatch_max=65535, buffers=12, f64=true, f16=true
```

### GpuCapabilities Struct

```rust
pub struct GpuCapabilities {
    pub max_buffer_size: u64,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub supports_f64: bool,
    pub supports_f16: bool,
    pub supports_timestamp_query: bool,
}
```

Queried from `device.limits()` and `device.features()` at initialization.
No hardcoded assumptions about hardware capabilities.

### Validators Updated

| Validator | Shader WG | Previous | Current |
|-----------|-----------|----------|---------|
| `validate_gpu_batch_fitness` | 256 | `.div_ceil(256)` | `gpu.dispatch_1d(n, 256)` |
| `validate_gpu_anderson` | 256 | `.div_ceil(256)` | `gpu.dispatch_1d(n, 256)` |
| `validate_gpu_game_theory` | 256 | `.div_ceil(256)` | `gpu.dispatch_1d(n, 256)` |
| `validate_gpu_sate` | 256 | `.div_ceil(256)` | `gpu.dispatch_1d(n, 256)` |
| `validate_gpu_pangenome` | 256 | `.div_ceil(256)` | `gpu.dispatch_1d(n, 256)` |
| `validate_gpu_meta_pop` | 256 | `.div_ceil(256)` | `gpu.dispatch_1d(n, 256)` |
| `validate_gpu_modes` | 256 | `.div_ceil(256)` | `gpu.dispatch_1d(n, 256)` |
| `validate_gpu_directed` | 256 | `.div_ceil(256)` | `gpu.dispatch_1d(n, 256)` |
| `validate_gpu_swarm` | 256 | `.div_ceil(256)` | `gpu.dispatch_1d(n, 256)` |
| `validate_gpu_signal` | 256 | `.div_ceil(256)` | `gpu.dispatch_1d(n, 256)` |
| `validate_gpu_rk4` | 64 | `.div_ceil(64)` | `gpu.dispatch_1d(n, 64)` |
| `evolved::hmm_forward_gpu` | 256 | `.div_ceil(256)` | `gpu.dispatch_1d(n, 256)` |

---

## Part 6: Remaining Gaps and Next Steps

### For ToadStool

| Priority | Item | Impact |
|----------|------|--------|
| Critical | Fix S-15 matmul hang (magnitude ≤ 0.1) | Unblocks real-world data with small values |
| High | Fix S-03b MHA projection dispatch | Retires `evolved::mha`, `head_split/head_concat` |
| High | Wire Conv2D/Pool to GpuExecutor | Enables full LeNet-5 GPU validation |
| Medium | Add `barracuda::ops::prng` | Absorbs `xoshiro128ss.wgsl` |
| Medium | Add capability validation to bio-op wrappers | Prevents silent failures on limited hardware |
| Low | Document data layout requirements for bio ops | Prevents discovery cost for new consumers |

### For neuralSpring

| Priority | Item | Impact |
|----------|------|--------|
| Medium | Wire remaining validators to `dispatch_1d` | Pipeline, cross-dispatch, benchmark validators |
| Medium | Evolve `hmm_forward_gpu` callers to `HmmBatchForwardF64` | Retires local f32 dispatch |
| Low | Wire `cpu_conv_pool` for LeNet-5 bC validation | Closes last bC gap (currently 24/25) |
| Low | GPU PRNG → Wright-Fisher/Gillespie stochastic pipelines | Next phase of GPU promotion |

---

## Appendix A: Cross-Spring Shader Lineage

See `whitePaper/CROSS_SPRING_SHADER_LINEAGE.md` for the full map.

| Spring | Contribution Count | Key Shader Domains |
|--------|-------------------|-------------------|
| hotSpring | 20+ | Lattice QCD, HFB nuclear, spectral theory, precision |
| wetSpring | 12+ | Smith-Waterman, Gillespie, Felsenstein, HMM f64, SNP |
| neuralSpring | 15 | Bio ops, fitness, pairwise, IPR, matmul tiers, eigh |

## Appendix B: Codebase Health

| Metric | Value |
|--------|-------|
| Library tests | 258 PASS, 1 ignored |
| Doc-tests | 9 PASS |
| Validation binaries | 119 |
| Line coverage | 94.9% |
| Clippy | 0 warnings (pedantic + nursery) |
| Unsafe code | Forbidden (`unsafe_code = "forbid"`) |
| Centralized tolerances | 20+ named constants in `tolerances.rs` |
| Runtime introspection | `GpuCapabilities`, `RuntimeEnvironment`, `NamedTolerance` |

## Appendix C: Paper Control Matrix

All 25 papers validated at Py + Rs. 24/25 at bC (96%). 23/25 at gT (92%).
15/15 Phase 0++ at mF + gP + xD (100%).

See `specs/PAPER_REVIEW_QUEUE.md` for the full 7-tier matrix.

---

*neuralSpring v9 — 25 papers, 5 disciplines, 4 faculty. 1607+ total checks.
258 lib tests, 119 binaries, 17 WGSL shaders (13 upstream, 4 local).
Capability-based dispatch. Cross-eigensolver validation. ALL GREEN.*
