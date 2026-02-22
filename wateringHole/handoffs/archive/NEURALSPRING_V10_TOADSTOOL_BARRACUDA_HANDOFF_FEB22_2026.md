# neuralSpring v10 → ToadStool / BarraCUDA Team Handoff

**Date:** February 22, 2026 (Session 42 — deep audit + code quality evolution)
**From:** neuralSpring (ML validation & evolutionary computation biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Supersedes:** `archive/NEURALSPRING_V9_TOADSTOOL_BARRACUDA_HANDOFF_FEB22_2026.md`
**ToadStool HEAD:** `5437c170` (Session 42)

---

## Executive Summary

neuralSpring has completed its full validation stack **and** a comprehensive code
quality audit: **25 papers, 1600+ checks, 119 binaries, 264 lib + 9 integration
tests, 17 WGSL shaders (13 upstream, 4 local)**. Session 42 focused on deep debt
resolution — the codebase now passes `cargo fmt`, `cargo clippy` (pedantic +
nursery + `unwrap_used` + `expect_used`), and `cargo doc` with zero warnings.
GPU validation helper code was deduplicated across 23 binaries. The tolerance
system gained runtime introspection. The entire dependency tree is pure Rust.

| Category | Status |
|----------|--------|
| Python controls | 25/25, 206 checks | ALL PASS |
| Rust CPU | 264 lib + 9 integration tests + 119 binaries | ALL PASS |
| BarraCUDA CPU | 24/25 papers (96%) | ALL GREEN |
| BarraCUDA GPU Tensor | 23/25 papers (92%) | ALL GREEN |
| metalForge WGSL | 15/25 papers, 17 shaders | ALL PASS |
| GPU Pipeline | 15/25 papers | ALL PASS |
| Cross-dispatch | 15/15 Phase 0++ papers | ALL GREEN |
| Upstream parity | 6/6 dual-path, 0.00e0 diff | Bit-identical |
| Code quality | fmt + clippy (pedantic) + doc | Zero warnings |
| Unsafe code | `#![forbid(unsafe_code)]` | None |
| Dependencies | Pure Rust (zero C/C++ FFI) | Verified |
| Line coverage | 94.9% (`llvm-cov`) | Above 90% target |

---

## Part 1: What Changed in Session 42 (Deep Audit)

### 1.1 Code Quality Gates — All Clean

| Gate | Before | After |
|------|--------|-------|
| `cargo fmt --check` | 33 violations | 0 |
| `cargo clippy` (pedantic+nursery+unwrap_used+expect_used) | 123 warnings | 0 |
| `cargo doc --no-deps` | 1 warning | 0 |
| Library tests | 255 pass | 264 pass |
| Integration tests | 0 | 9 pass |
| Unsafe code | 0 blocks | 0 blocks (`forbid`) |

### 1.2 GPU Validation Helper Deduplication

23 GPU validation binaries had local copies of `readback()`, `tensor!` macro,
and `max_abs_diff_flat()`. These are now shared from `validation.rs`:

| Shared Helper | Purpose | Consumers |
|--------------|---------|-----------|
| `gpu_readback()` | GPU buffer → `Vec<f32>` | 23 binaries |
| `max_abs_diff_f32()` | Max absolute difference (f32 arrays) | 23 binaries |
| `max_abs_diff_gpu_vs_cpu()` | f32 GPU vs f64 CPU comparison | 23 binaries |
| `gpu_tensor()` | `Tensor` creation wrapper | 23 binaries |
| `gpu_tensor!` macro | Ergonomic tensor construction | 23 binaries |

~400 lines of duplicated code removed.

### 1.3 Tolerance System Evolution

The tolerance module was split to stay under the 1000-line wateringHole limit:

| File | Lines | Content |
|------|-------|---------|
| `tolerances/mod.rs` | 696 | All named constants (20+) |
| `tolerances/registry.rs` | 341 | `NamedTolerance`, `all_tolerances()`, `tolerance_by_name()`, `categories()` |

18 previously unregistered tolerances are now in the runtime registry. 3 inline
magic numbers in validation binaries were replaced with named constants.

**For BarraCUDA**: This pattern (named constants + runtime registry + categories)
is a good model for BarraCUDA's own tolerance/precision system. Every tolerance
is discoverable, categorized, and documented.

### 1.4 Provenance Enhancement

Exact reproduction commands added for key validation targets:

```
SOFTMAX_1_TO_5:   python3 -c "import numpy as np; x=np.array([1,2,3,4,5],dtype=np.float64); ..."
GELU_REFERENCE:   python3 -c "import numpy as np; x=np.array([-2,-1,0,1,2],dtype=np.float64); ..."
RASTRIGIN_REF:    python3 -c "import numpy as np; A=10; ..."
```

NumPy/SciPy versions and environment details are recorded for each.

### 1.5 Determinism Tests

9 new determinism tests verify bitwise-identical results for seeded stochastic
algorithms: introgression, regulatory_network, pangenome_selection,
meta_population, sate_alignment, signal_integration, game_theory,
spectral_commutativity, anderson_localization.

### 1.6 Python Baseline Drift Detection

New: `control/check_drift.sh` — re-runs all 25 Python baselines and verifies
no numeric drift. Ready for CI integration.

### 1.7 Dependency Analysis

Confirmed the full stack is pure Rust:

| Crate | Role | C/C++ deps |
|-------|------|------------|
| `barracuda` | Unified math | None |
| `wgpu` | WebGPU impl | None (pure Rust Vulkan) |
| `naga` | WGSL compiler | None |
| `rand`/`rand_xoshiro` | PRNG | None |
| `approx` | Float comparison | None |

No C FFI, no `cc` build script, no system library linkage.

---

## Part 2: Full BarraCUDA Integration Surface (48 files, 20+ API categories)

### 2.1 Core Infrastructure

| API | neuralSpring Usage | Files |
|-----|-------------------|-------|
| `device::WgpuDevice` | GPU context, adapter selection, shader compilation | `gpu.rs`, 20+ validators |
| `tensor::Tensor` | GPU tensor ops (matmul, transpose, tanh, sigmoid, dot, etc.) | 28 validators |
| `error::BarracudaError` | Error propagation in evolved MHA | `evolved/mha.rs` |

### 2.2 Bio Operations (GPU Wrappers) — All Bit-Identical

| API | Papers | Validated | Parity |
|-----|--------|-----------|--------|
| `ops::bio::BatchFitnessGpu` | 011-015 | 12/12 PASS | 0.00e0 |
| `ops::bio::PairwiseHammingGpu` | 017 | 6/6 PASS | 0.00e0 |
| `ops::bio::PairwiseJaccardGpu` | 024 | 7/7 PASS | 0.00e0 |
| `ops::bio::LocusVarianceGpu` | 025 | 8/8 PASS | 0.00e0 |
| `ops::bio::SpatialPayoffGpu` | 019 | 6/6 PASS | 0.00e0 |
| `ops::bio::HmmBatchForwardF64` | 016-018 | 11/11 PASS | 2.47e-10 (f64) |

### 2.3 Spectral Theory (hotSpring lineage)

| API | Usage | Validated |
|-----|-------|-----------|
| `spectral::BatchIprGpu` | Anderson localization IPR | 7/7 PASS, 0.00e0 |
| `spectral::find_all_eigenvalues` | Sturm bisection eigensolver | 17/17 PASS |
| `spectral::lanczos` / `lanczos_eigenvalues` | Sparse Lanczos | 2D/3D validated |
| `spectral::anderson_hamiltonian` | Anderson disorder | Bandwidth, eigenvalue count |
| `spectral::almost_mathieu_hamiltonian` | Aubry-André model | Cross-validated vs Jacobi |
| `spectral::hofstadter_butterfly` | Hofstadter fractal | 21 α, 2100 eigenvalues |
| `spectral::lyapunov_exponent` | Localization measure | Kappus-Wegner anomaly |
| `spectral::level_spacing_ratio` | GOE vs Poisson | Extended/localized phases |

### 2.4 CPU Math Primitives

| API | Papers | Precision |
|-----|--------|-----------|
| `stats::variance` / `pearson_correlation` | All 15 Phase 0++ | Machine-precision |
| `linalg::eigh_f64` (Householder+QR) | 022-023 | 1.75e-14 at n=32 |
| `linalg::solve_f64` | 016, 015 | Machine precision |
| `numerical::rk45_solve` | 019-021 | Machine precision |
| `special::chi_squared_sf/cdf` | 018 | Correct LRT p-values |
| `optimize::nelder_mead/bisect/brent` | Validation | 10/10 PASS |
| `ops::fft::Fft1D/Fft1DF64` | Spectral | 24/24 PASS |
| `ops::logsumexp::LogSumExp` | HMM numerics | 5/5 PASS |

### 2.5 Pipeline Infrastructure

| API | Usage | Validated |
|-----|-------|-----------|
| `pipeline::ReduceScalarPipeline` | f64 GPU scalar reduction | 5.55e-17 diff |
| `staging::StatefulPipeline` | Iterative GPU compute | 10/10 PASS |
| `dispatch::dispatch_for` | CPU↔GPU routing | 49 cross-dispatch checks |

---

## Part 3: What ToadStool Should Absorb

### 3.1 Critical: S-15 Matmul Hang (magnitude ≤ 0.1)

Elements with magnitude ≤ 0.1 trigger a WGPU/Vulkan driver hang on RTX 4070.
Affects ALL matmul tiers. Root-caused to a driver-level interaction with IEEE 754
bit patterns — not a WGSL logic error.

**Workaround**: All validators use data ≥ 0.5.
**Investigation path**: Test on AMD/Intel to isolate NVIDIA-specific behavior.
File an upstream `wgpu` or NVIDIA driver bug report with minimal repro.

### 3.2 High: S-03b MHA Projection Fix

Native `multi_head_attention` dispatch bug: z-dimension divides by 16 instead
of 1 for projection shaders. Fix in `barracuda/src/ops/mha/projections.rs`:
change `div_ceil(16)` to `div_ceil(1)`. Retires `evolved::mha` + 2 local
shaders (`head_split.wgsl`, `head_concat.wgsl`).

### 3.3 High: Capability-Based Dispatch Pattern

neuralSpring introduced `GpuCapabilities` and `Gpu::dispatch_1d()` for runtime
hardware validation. **Recommendation**: BarraCUDA's bio-op wrappers should
query `device.limits()` and validate workgroup size — matters for WebGPU
targets (mobile, browser) and CPU adapters (llvmpipe).

### 3.4 Medium: Remaining Local Shaders (4)

| Shader | Blocker | Absorption Target |
|--------|---------|-------------------|
| `head_split.wgsl` | S-03b projection hang | `barracuda::ops::mha` |
| `head_concat.wgsl` | S-03b projection hang | `barracuda::ops::mha` |
| `xoshiro128ss.wgsl` | No upstream module | `barracuda::ops::prng` |
| `swarm_nn_scores.wgsl` | No equivalent | `barracuda::ops::bio::swarm` |

### 3.5 Medium: Wire Conv2D/Pool to GpuExecutor

`conv2d.wgsl`, `maxpool2d.wgsl`, `avgpool2d.wgsl` exist but aren't wired to
the executor. This blocks full LeNet-5 GPU validation (the last bC gap: 24→25/25).

### 3.6 Low: Document Data Layout Requirements

| API | Layout Requirement | Discovered By |
|-----|-------------------|---------------|
| `PairwiseJaccardGpu` | **Column-major** PA: `pa[gene * n_genomes + genome]` | neuralSpring Exp 008 |
| `BatchIprGpu` | Returns raw `Σ|ψ_i|⁴` (NOT reciprocal) | neuralSpring Exp 008 |
| `HmmBatchForwardF64` | Batch dimension first: `[batch, states]` | neuralSpring Exp 008 |

---

## Part 4: Lessons for BarraCUDA Evolution

### 4.1 The 7-Tier Validation Progression

Every paper passes through 7 tiers, each proving a correctness property:

| Tier | What It Proves | neuralSpring Coverage |
|------|---------------|----------------------|
| Py (Python) | Science is correct | 25/25 (100%) |
| Rs (Rust CPU) | Math translates to type-safe Rust | 25/25 (100%) |
| bC (BarraCUDA CPU) | Pure Rust primitives match hand-rolled | 24/25 (96%) |
| gT (GPU Tensor) | Math portable CPU → GPU | 23/25 (92%) |
| mF (metalForge WGSL) | Domain-specific GPU kernels correct | 15/25 (100%†) |
| gP (GPU Pipeline) | Multi-kernel chains compose | 15/25 (100%†) |
| xD (Cross-dispatch) | CPU ↔ GPU parity | 15/15 (100%) |

`†` 100% of applicable papers. Phase 0/0+ studies use PyTorch training.

This is the **strongest possible correctness argument** for BarraCUDA: 206
independent Python baselines, reproduced in Rust, validated at every GPU
abstraction layer, across 25 papers from 5 scientific disciplines.

### 4.2 Cross-Spring Evolution Is Working

| Flow | What Happened | Precision Gain |
|------|---------------|---------------|
| nS f32 HMM → TS → wS f64 batch → TS → nS | neuralSpring evolved f32, wetSpring evolved f64, nS validates both | 10⁹× |
| hS spectral → TS → nS validates | hotSpring contributed spectral theory, nS validates analytical | 17/17 PASS |
| nS Householder+QR → TS → nS+hS | Dense eigensolver benefits both Springs | 1.75e-14 at n=32 |

### 4.3 Upstream Wrapper Overhead Is Negligible

| Bio Op | Local→Upstream Ratio |
|--------|---------------------|
| BatchFitness | 1.16× |
| PairwiseHamming | 1.03× |
| PairwiseJaccard | 0.92× (faster!) |
| LocusVariance | 1.12× |
| SpatialPayoff | 0.96× |
| BatchIpr | 1.03× |

Median overhead < 5%. Springs should always prefer upstream wrappers.

### 4.4 GPU Dispatch Has a 1.5ms Crossover

Below 1.5ms of CPU work, CPU is faster than GPU due to dispatch overhead.
Codified in `barracuda::dispatch::dispatch_for()`. BarraCUDA should NOT
automatically promote small workloads to GPU.

### 4.5 The Six Isomorphic Primitives — All Validated

| Primitive | BarraCUDA Module | Status |
|-----------|-----------------|--------|
| GEMM | `Tensor::matmul`, 4-tier KernelRouter | Validated (S-15 workaround) |
| Attention | `Tensor::attention` | Validated (S-03b workaround) |
| Normalization | `Tensor::layer_norm_wgsl`, `log_softmax_wgsl` | Validated |
| Nonlinearity | `Tensor::relu/gelu/silu/tanh/sigmoid` | Validated (90/90) |
| Reduction | `ReduceScalarPipeline`, `VarianceReduceF64` | Validated (5.55e-17) |
| Gating | Hill function, sigmoid gating | Validated via `hill_gate.wgsl` |

### 4.6 Tolerance System Pattern

neuralSpring's tolerance system could serve as a model for BarraCUDA:

```rust
pub struct NamedTolerance {
    pub name: &'static str,
    pub value: f64,
    pub category: &'static str,
    pub justification: &'static str,
}

pub fn all_tolerances() -> Vec<NamedTolerance> { ... }
pub fn tolerance_by_name(name: &str) -> Option<NamedTolerance> { ... }
pub fn categories() -> Vec<&'static str> { ... }
```

Every threshold is named, categorized, justified, and runtime-discoverable.
This eliminates hidden magic numbers and makes precision auditing trivial.

### 4.7 Pure Rust Dependency Chain

The complete stack — neuralSpring + BarraCUDA + wgpu + naga — is pure Rust.
No C FFI, no system libraries, no `cc` build scripts. This means:
- Cross-compilation works without a C toolchain
- `cargo audit` covers the entire dependency tree
- No hidden ABI compatibility issues
- Reproducible builds across all platforms

---

## Part 5: Shader Absorption Status

### 5.1 Absorbed (Identical — `77f70b2e`)

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

### 5.2 Absorbed (Generalized — `5437c170`)

| Shader | Upstream Improvement |
|--------|---------------------|
| `pairwise_l2.wgsl` | O(1) pair decode |
| `multi_obj_fitness.wgsl` | Bessel correction |
| `hill_gate.wgsl` | Mode generalization |
| `swarm_nn_forward.wgsl` | Generic MLP, clamped sigmoid |
| `mean_reduce.wgsl` | Effectively identical |

### 5.3 Still Local (4 shaders)

| Shader | Blocker | Target |
|--------|---------|--------|
| `head_split.wgsl` | S-03b | `barracuda::ops::mha` |
| `head_concat.wgsl` | S-03b | `barracuda::ops::mha` |
| `xoshiro128ss.wgsl` | No module | `barracuda::ops::prng` |
| `swarm_nn_scores.wgsl` | No equivalent | `barracuda::ops::bio::swarm` |

---

## Part 6: Primitives neuralSpring Can Contribute Back

### 6.1 Ready for Absorption

| neuralSpring Module | BarraCUDA Target | Tests | Notes |
|--------------------|-----------------|-------|-------|
| `primitives::shannon_entropy` | `stats::entropy` | 8 unit tests | Well-tested, flat-layout ready |
| `primitives::hill_activation` | `numerical::hill` | Used by Papers 020-021 | Cross-spring: hotSpring could use |
| `gpu_readback()` | `testing::gpu_readback` | 23 consumer binaries | Shared GPU buffer readback pattern |
| `max_abs_diff_gpu_vs_cpu()` | `testing::precision` | 23 consumer binaries | f32 GPU vs f64 CPU comparison |
| `GpuCapabilities` struct | `device::capabilities` | 12 validators | Runtime hardware discovery |
| `dispatch_1d()` | `device::dispatch` | 12 validators | Validated workgroup dispatch |
| `NamedTolerance` system | `testing::tolerances` | 20+ named entries | Runtime-discoverable thresholds |

### 6.2 Candidate for Future Absorption

| Pattern | Papers | Current | Absorption Path |
|---------|--------|---------|-----------------|
| Determinism testing | 16 tests | `determinism_tests.rs` | `barracuda::testing::determinism` |
| Drift detection | 25 baselines | `check_drift.sh` | CI pattern for any consumer |
| `require!` macro | All validators | `validation.rs` | `barracuda::testing::require` |

---

## Part 7: Remaining Gaps and Roadmap

### For ToadStool (Priority Order)

| Priority | Item | Impact |
|----------|------|--------|
| Critical | Fix S-15 matmul hang (magnitude ≤ 0.1) | Unblocks real-world data |
| High | Fix S-03b MHA projection dispatch | Retires `evolved::mha` + 2 shaders |
| High | Wire Conv2D/Pool to GpuExecutor | Enables full LeNet-5 GPU (24→25/25 bC) |
| Medium | Add `barracuda::ops::prng` | Absorbs `xoshiro128ss.wgsl` |
| Medium | Capability validation in bio-op wrappers | Prevents silent failures |
| Medium | `barracuda::testing` module (require!, tolerances) | Cross-spring reuse |
| Low | Document data layout requirements | Prevents discovery cost |

### For neuralSpring (Priority Order)

| Priority | Item | Impact |
|----------|------|--------|
| Medium | Wire remaining validators to `dispatch_1d` | Pipeline/bench validators |
| Medium | Evolve `hmm_forward_gpu` to `HmmBatchForwardF64` | Retires local f32 |
| Low | Wire `cpu_conv_pool` for LeNet-5 bC | Closes last bC gap |
| Low | GPU PRNG → Wright-Fisher/Gillespie | Next-phase GPU promotion |

---

## Appendix A: Codebase Health After Deep Audit

| Metric | Before (Session 40) | After (Session 42) |
|--------|---------------------|---------------------|
| Library tests | 255 | 264 |
| Integration tests | 0 | 9 |
| `cargo fmt` violations | 33 | 0 |
| `cargo clippy` warnings | 123 | 0 |
| `cargo doc` warnings | 1 | 0 |
| Unsafe code blocks | 0 | 0 (forbidden) |
| Inline magic numbers | 3 | 0 |
| Duplicated GPU helpers | ~400 LOC in 23 files | Shared in `validation.rs` |
| Tolerance registry | Partial | 20+ entries, runtime-discoverable |
| Files over 1000 LOC | 1 (`tolerances.rs`) | 0 |
| C/C++ dependencies | 0 | 0 (verified) |
| Line coverage | 94.9% | 94.9% |

## Appendix B: 3-Way Performance Benchmark

| Scale | Python (1t) | BarraCUDA CPU | BarraCUDA GPU | CPU/Py | GPU/Py |
|-------|-------------|---------------|---------------|--------|--------|
| MLP large (3.1M) | 3.0 ms | **2.7 ms** | **178 µs** | 1.1× | 16.8× |
| TF medium (103M) | 59 ms | **15.1 ms** | **566 µs** | 3.9× | 104× |
| TF xlarge (6.6B) | 232 ms | 1.42 s | **17.8 ms** | — | 13.1× |

Pure Rust math (Phase 0++ kernels): 71.8× faster than NumPy overall.

## Appendix C: Paper Control Matrix

All 25 papers validated at Py + Rs. 24/25 at bC (96%). 23/25 at gT (92%).
15/15 Phase 0++ at mF + gP + xD (100%). See `specs/PAPER_REVIEW_QUEUE.md`.

---

*neuralSpring v10 — 25 papers, 5 disciplines, 4 faculty. 1600+ total checks.
264 lib + 9 integration tests, 119 binaries, 17 WGSL shaders (13 upstream, 4 local).
Deep audit: fmt/clippy/doc clean, GPU helpers deduplicated, tolerances split + registry,
provenance enhanced, 16 determinism tests, drift detection, pure Rust verified.
Capability-based dispatch. Cross-eigensolver validation. ALL GREEN.*
