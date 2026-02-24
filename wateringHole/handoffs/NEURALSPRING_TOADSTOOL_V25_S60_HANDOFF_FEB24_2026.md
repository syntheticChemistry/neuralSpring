# neuralSpring → ToadStool Handoff V25: Cross-Spring Evolution & Absorption Targets

**Date**: February 24, 2026
**From**: neuralSpring (ML validation & evolutionary computation)
**To**: ToadStool / BarraCUDA team
**Phase**: Session 60 — Cross-spring benchmark validation complete
**License**: AGPL-3.0-or-later

---

## Executive Summary

neuralSpring has completed the S54–S60 absorption cycle: **16 functions** now
delegate to upstream BarraCUDA APIs. Cross-spring evolution benchmarks confirm
that shaders and primitives from all three Springs (hotSpring precision, wetSpring
bio, neuralSpring ML) work together through ToadStool with measurable performance
benefits. This handoff documents what we've absorbed, what remains, and concrete
absorption targets for the ToadStool team.

## Current State

| Metric | Value |
|--------|-------|
| Python controls | 206/206 PASS (25 papers + 5 studies) |
| Rust lib tests | 482 PASS |
| Rust integration | 9 PASS |
| validate_all | 145/146 PASS (1 pre-existing upstream logsumexp) |
| Cross-spring evolution | 22/22 PASS |
| Functions rewired to upstream | 16 |
| WGSL shaders absorbed | 19/21 (2 local: head_split, head_concat) |
| GPU dispatch ops | 38 (9 upstream dispatch, 29 local gpu_ops) |
| Hardware validated | RTX 4070 (Ada) + TITAN V (NVK) — bit-identical |

---

## What neuralSpring Now Leans On (from upstream)

### Dispatcher Methods (9 rewired to `domain_ops`)

| Method | Upstream API | Session | Cross-Spring Origin |
|--------|-------------|---------|---------------------|
| `mat_mul` | `matmul_dispatch` | S58 | hotSpring (tile kernels, KernelRouter) |
| `frobenius_norm` | `frobenius_norm_dispatch` | S58 | hotSpring (reduction) |
| `transpose` | `transpose_dispatch` | S58 | neuralSpring (spectral) |
| `softmax` | `softmax_dispatch` | S58 | hotSpring (f64 numerics) |
| `l2_distance` | `l2_distance_dispatch` | S58 | neuralSpring (MODES) |
| `mean` | `mean_dispatch` | S58 | hotSpring (reduction) |
| `variance` | `variance_dispatch` | S58 | hotSpring (Welford) |
| `gelu` | `gelu_dispatch` | S59 | neuralSpring ML → absorbed S52 |
| `hmm_forward_step` | `hmm_forward_dispatch` | S59 | wetSpring bio (phylo) → absorbed S52 |

### Library Functions (3 rewired to stats/linalg)

| Function | Upstream API | Session | Origin |
|----------|-------------|---------|--------|
| `empirical_spectral_density` | `barracuda::stats::empirical_spectral_density` | S59 | neuralSpring → absorbed S54 |
| `marchenko_pastur_bounds` | `barracuda::stats::marchenko_pastur_bounds` | S59 | neuralSpring → absorbed S54 |
| `effective_rank` | `barracuda::linalg::effective_rank` | S59 | neuralSpring → absorbed S54 |

### baseCamp Functions (4 rewired S56)

| Function | Upstream API | Module |
|----------|-------------|--------|
| `graph_laplacian` | `barracuda::linalg::graph_laplacian` | agent_coordination |
| `disordered_laplacian` | `barracuda::linalg::disordered_laplacian` | agent_coordination |
| `belief_propagation_chain` | `barracuda::linalg::belief_propagation_chain` | neural_pgm |
| `numerical_hessian` | `barracuda::numerical::numerical_hessian` | loss_landscape |

### GPU Typed Ops (from all three Springs)

| Op | Origin Spring | neuralSpring Uses |
|----|---------------|-------------------|
| `VarianceReduceF64` | hotSpring (Welford) | Production variance (2.46× faster) |
| `CorrelationF64` | wetSpring + hotSpring | Production Pearson (1.11× faster) |
| `FusedMapReduceF64` | wetSpring | Production entropy (2.59× faster) |
| `BatchedEighGpu` | hotSpring (nuclear) | Anderson localization, baseCamp spectral |
| `HmmBatchForwardF64` | wetSpring (phylo) | Available for batch HMM |
| `BatchIprGpu` | neuralSpring (Anderson) | Spectral localization |

---

## Cross-Spring Evolution Benchmarks (RTX 4070, `--release`)

### f32 Tensor → f64 Typed Ops (10k elements)

| Op | Old (µs) | New (µs) | Speedup | Origin |
|----|----------|----------|---------|--------|
| Variance | 5,773 | 2,350 | **2.46×** | hotSpring Welford |
| Pearson | 3,254 | 2,938 | **1.11×** | wetSpring + hotSpring |
| Entropy | 3,191 | 1,232 | **2.59×** | wetSpring fused |

### GPU Typed Ops (cross-spring)

| Op | Size | Median (µs) | Origin |
|----|------|-------------|--------|
| `BatchFitnessGpu` | 1024×64 | 1,274 | neuralSpring |
| `PairwiseL2Gpu` | 128×16 | 1,846 | neuralSpring |
| `BatchIprGpu` | 32×64 | 2,541 | neuralSpring |
| `SpatialPayoffGpu` | 32×32 | 1,518 | neuralSpring |
| `PairwiseHammingGpu` | 64×100 | 1,430 | neuralSpring |
| `HmmBatchForwardF64` | 4s×50t×32b | 1,743 | wetSpring |
| `BatchedEighGpu` | 12×12×40 | 5,355 | hotSpring |

---

## Absorption Targets for ToadStool Team

### Priority 1: MHA Projection Shaders (S-03b)

neuralSpring still maintains local `head_split.wgsl` / `head_concat.wgsl`
because upstream `barracuda::ops::mha::MultiHeadAttention` projection shaders
hang on RTX 4070 / Vulkan at production sizes (B=4, S=128, H=8, d=512).

**Action**: Validate upstream MHA at production sizes on RTX 4070, then
neuralSpring can retire `evolved/mha.rs` and the 2 local WGSL shaders.

### Priority 2: LogSumExp Buffer Size

`validate_barracuda_logsumexp` fails with WGPU validation error: "Buffer is
bound with size 4 where the shader expects 8". This is the only failing
validator (1/146).

**Action**: Fix buffer size mismatch in `barracuda::ops::logsumexp`.

### Priority 3: Remaining Local GPU Ops (29 ops)

These use raw Tensor API because no upstream typed dispatch exists:

| Category | Ops | Upstream Candidate |
|----------|-----|-------------------|
| Commutator / distance-to-normal | 2 | Composed from existing matmul dispatch |
| Boltzmann distribution | 1 | `softmax_dispatch` with beta scaling |
| Hill activation batch | 1 | Specialized transcendental — keep local |
| Shannon entropy (GPU) | 1 | Already uses `FusedMapReduceF64` |
| Pearson correlation (GPU) | 1 | Already uses `CorrelationF64` |
| Chi-squared (GPU) | 1 | Could use `chi_squared_statistic` dispatch |
| HMM backward/Viterbi | 2 | Could extend `hmm_forward_dispatch` pattern |
| Population genetics | 6 | Domain-specific compositions |
| Game theory | 1 | Small 2×2 — CPU optimal |
| Eigensolve (GPU) | 2 | Already uses `BatchedEighGpu` |
| Pangenome | 2 | Could use `chi_squared_statistic` dispatch |

**Action**: Consider adding `hmm_backward_dispatch` and `hmm_viterbi_dispatch`
to `domain_ops` — these follow the same GEMV pattern as forward.

### Priority 4: Tridiagonal Eigensolver

Papers 022–023 would benefit from a dedicated `tridiag_eigh.wgsl` shader.
Currently falls back to dense Householder+QR via `eigh_f64`.

**Action**: NAK-optimized tridiagonal eigensolver (Sturm bisection on GPU).

---

## Lessons Learned (for ToadStool evolution)

### 1. Cross-Spring Dispatch Design Works

The `domain_ops` dispatch pattern — try upstream GPU, fall back to CPU — is
exactly right. For validation-scale workloads (n ≤ 4096), dispatch correctly
routes to CPU with zero overhead. GPU benefits appear at production scales.

### 2. f64 Typed Ops Are Worth It

The move from f32 Tensor paths to f64 typed ops (VarianceReduceF64, etc.)
delivers 2–3× speedups AND better precision. The Welford algorithm for
variance is a particular win.

### 3. GpuDriverProfile Is Essential

RTX 4070 (Ada Lovelace) needs the Hybrid f64 strategy (df64 for bulk,
native for reductions) and the `pow_f64` polyfill. Without `GpuDriverProfile`,
shaders crash on NVVM. This detection should be the default for all Springs.

### 4. Cross-Spring Shader Provenance Matters

Tracking where each shader originated (hotSpring precision, wetSpring bio,
neuralSpring ML) helps debug issues and prioritize absorption. The lineage
is: each Spring evolves what it needs → ToadStool absorbs → all Springs benefit.

### 5. Bit-Identical Multi-GPU Is Achievable

RTX 4070 (proprietary Vulkan) and TITAN V (NVK open-source) produce
bit-identical results for all 145 passing validators. The WGSL abstraction
delivers genuine hardware portability.

---

## Modified Files (this session)

| File | Change |
|------|--------|
| `src/bin/validate_cross_spring_evolution.rs` | +5 S59 validators (22 total checks) |
| `src/bin/bench_cross_spring_evolution.rs` | +rewired dispatcher throughput section |
| `specs/CROSS_SPRING_EVOLUTION.md` | S60 benchmark results + evolution narrative |
| `specs/TOADSTOOL_HANDOFF.md` | S60 sync entry |
| `specs/BARRACUDA_USAGE.md` | Session range update |
| `specs/EVOLUTION_MAPPING.md` | Session range update |

## Remaining Work

1. Retire `evolved/mha.rs` after upstream MHA validated at production sizes
2. Fix upstream logsumexp buffer size
3. Consider `hmm_backward_dispatch` / `hmm_viterbi_dispatch` in `domain_ops`
4. NAK tridiagonal eigensolver for Papers 022–023
5. Exercise baseCamp GPU promotion targets (weight_to_hamiltonian, numerical_hessian)
