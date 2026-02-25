# neuralSpring → ToadStool/BarraCUDA Handoff V30

**Sessions 66–67b — Phase C GPU Completion, CPU↔Python Parity, Dispatch Tier Characterization**
**Date**: February 25, 2026
**From**: neuralSpring
**To**: ToadStool / BarraCUDA core team
**License**: AGPL-3.0-or-later
**ToadStool HEAD**: `02207c4a`
**Supersedes**: V29 (Session 64 — forge v0.2.0, substrate discovery, workload tracking)

---

## Executive Summary

Sessions 66–67b close the neuralSpring math validation loop:

1. **Session 66 (Phase C GPU)**: Composed step-level GPU ops into chain-level ops
   — HMM forward/Viterbi chains, pairwise/global FST, inter-pop AF variance.
   GPU coverage: ~90% → **~97%** of production math. 18/18 PASS.

2. **Session 67 (CPU parity)**: Cross-language validation proving Rust CPU =
   Python/NumPy for 9 primitives + 9 paper kernels + 6 Dispatcher methods.
   **39/39 PASS** at 1e-10 tolerance.

3. **Session 67b (dispatch tiers)**: Three-tier benchmark quantifying dispatch
   overhead. Library direct → Dispatcher::cpu_only() → Dispatcher::new() GPU.
   **9/10 ops ≤1.04× CPU overhead**. Per-call GPU is driver-bound for small
   workloads — **motivates StatefulPipeline/UnidirectionalPipeline batching**.

---

## Current State

| Metric | Value |
|--------|-------|
| `cargo test --lib` | **505 PASS** |
| `cargo test -p neural-spring-forge` | **43 PASS** |
| `cargo clippy --all-targets` (pedantic + nursery) | **0 warnings** |
| `validate_all` | **147/148 PASS** (1 pre-existing logsumexp driver) |
| Python baselines | **25/25 PASS** (zero drift) |
| CPU↔Python parity | **39/39 PASS** (1e-10 cross-language) |
| Dispatch overhead (CPU) | **≤1.04× for 9/10 ops** |
| GPU dispatch coverage | **44 CPU→GPU ops** (~97% of production math) |
| Rust vs Python speedup | **201.7×** (11 kernels) |
| Validation/bench binaries | **159** |
| Total validation checks | **2120+** |

---

## Part 1: What neuralSpring Contributed (Sessions 66–67b)

### Session 66 — Phase C GPU Chain Composition

| New GPU Op | File | Composition Pattern |
|------------|------|---------------------|
| `hmm_forward_chain_gpu` | `src/gpu_ops/bio.rs` | Loop T observations → `hmm_forward_step_gpu` per step |
| `hmm_viterbi_chain_gpu` | `src/gpu_ops/bio.rs` | Loop T observations → `hmm_viterbi_step_gpu` per step |
| `pairwise_fst_gpu` | `src/gpu_ops/population.rs` | `allele_frequencies_gpu` + Weir-Cockerham per-locus |
| `global_fst_gpu` | `src/gpu_ops/population.rs` | Per-pop `allele_frequencies_gpu` + global decomposition |

| New Dispatcher Method | GPU Path | CPU Fallback |
|-----------------------|----------|--------------|
| `hmm_forward_chain` | `hmm_forward_chain_gpu` | `Hmm::from_flat` → `.forward()` |
| `hmm_viterbi_chain` | `hmm_viterbi_chain_gpu` | `Hmm::from_flat` → `.viterbi()` |
| `pairwise_fst` | `pairwise_fst_gpu` | `meta_population::pairwise_fst` |
| `global_fst` | `global_fst_gpu` | `meta_population::global_fst` |
| `inter_population_af_variance` | Existing `gpu_op` → dispatch | `meta_population::inter_population_af_variance` |

### Session 67 — CPU Math Parity Infrastructure

| New File | Purpose |
|----------|---------|
| `control/generate_cpu_references.py` | Deterministic Python → JSON (9 primitives + 9 kernels) |
| `control/cpu_parity_references.json` | Cross-language reference data |
| `src/bin/validate_cpu_math_parity.rs` | 39/39 PASS (library + Dispatcher::cpu_only()) |

### Session 67b — Dispatch Tier Benchmark

| New File | Purpose |
|----------|---------|
| `src/bin/bench_dispatch_tiers.rs` | Three-tier: library → cpu_only → gpu (10 kernels) |

---

## Part 2: What neuralSpring Consumes from BarraCUDA

### Upstream Dispatch Calls (10 — via `barracuda::dispatch`)

| Operation | Upstream Function |
|-----------|-------------------|
| matmul | `matmul_dispatch` |
| frobenius_norm | `frobenius_norm_dispatch` |
| transpose | `transpose_dispatch` |
| softmax | `softmax_dispatch` |
| gelu | `gelu_dispatch` |
| l2_distance | `l2_distance_dispatch` |
| mean | `mean_dispatch` |
| variance | `variance_dispatch` |
| hmm_forward_step | `hmm_forward_dispatch` |

### Upstream Typed Ops (via `barracuda::ops`, `barracuda::spectral`)

| Op | Module | Origin |
|----|--------|--------|
| BatchFitnessGpu | `ops::bio` | neuralSpring → ToadStool |
| PairwiseHammingGpu | `ops::bio` | neuralSpring → ToadStool |
| PairwiseJaccardGpu | `ops::bio` | neuralSpring → ToadStool |
| LocusVarianceGpu | `ops::bio` | neuralSpring → ToadStool |
| SpatialPayoffGpu | `ops::bio` | neuralSpring → ToadStool |
| BatchIprGpu | `spectral` | neuralSpring → ToadStool |
| HillGateGpu | `ops::bio` | neuralSpring → ToadStool |
| MultiObjFitnessGpu | `ops::bio` | neuralSpring → ToadStool |
| PairwiseL2Gpu | `ops::bio` | neuralSpring → ToadStool |
| SwarmNnGpu | `ops::bio` | neuralSpring → ToadStool |
| MultiHeadAttention | `ops::mha` | neuralSpring S-03b → ToadStool |
| HmmBatchForwardF64 | `ops::bio::hmm` | wetSpring → ToadStool |
| BatchedEighGpu | `spectral` | hotSpring → ToadStool |

### Upstream CPU Primitives

| Module | Functions Used |
|--------|---------------|
| `stats::correlation` | `variance`, `pearson_correlation` |
| `special` | `chi_squared_statistic`, `gamma`, `erf`, `bessel_j0` |
| `linalg` | `solve_f64`, `eigh_f64`, `cholesky_f64`, `lu_det`, `lu_solve`, `tridiagonal_solve`, `effective_rank` |
| `linalg::graph` | `belief_propagation_chain`, `graph_laplacian`, `disordered_laplacian` |
| `numerical` | `rk45_solve`, `numerical_hessian` |
| `device` | `WgpuDevice`, `GpuDriverProfile`, `Fp64Strategy` |

---

## Part 3: Lessons Learned for ToadStool Evolution

### 3.1 Per-Call GPU Dispatch Overhead

**Finding**: Per-call GPU dispatch incurs ~1.5ms fixed overhead (driver + encoder +
submit). For small workloads (< 1.5ms CPU), CPU is faster. This is measured and
documented in `bench_dispatch_tiers.rs`.

**Implication for ToadStool**: `StatefulPipeline` and `UnidirectionalPipeline` are
critical for real GPU speedups. Per-call dispatch should be reserved for large
workloads (50k+ elements). The dispatch heuristic in `Dispatcher` already routes
small workloads to CPU — this is correct and should be preserved upstream.

### 3.2 Chain Composition Pattern

**Finding**: Composing step-level GPU ops into chain-level ops (e.g., T × forward_step
= forward_chain) works but incurs T × dispatch overhead. For HMM with 500 observations,
this means 500 GPU dispatches per chain call.

**Recommendation for ToadStool**: Add `hmm_forward_chain_dispatch` and
`hmm_viterbi_chain_dispatch` to `barracuda::dispatch::domain_ops` that use a single
`CommandEncoder` for the entire chain. The `StatefulPipeline` API already supports
this pattern — the chain just needs to be pre-compiled.

### 3.3 f32 Precision Accumulation

**Finding**: f32 GPU ops over long chains (200+ HMM steps) diverge from f64 CPU.
Viterbi path agreement drops to ~90% at 200 steps. FST values diverge by ~0.04.

**Recommendation for ToadStool**: The `df64` infrastructure (double-float emulation)
is essential for long-chain scientific operations. neuralSpring can be an early
consumer of `df64` dispatch variants when available.

### 3.4 Dispatcher Transparency

**Finding**: `Dispatcher::cpu_only()` adds ≤1.04× overhead for 9/10 ops (the one
outlier is Hill batch at 19.17× due to batch allocation). The dispatch layer is
effectively transparent for CPU paths.

**Recommendation**: The `gpu_or_cpu` fallback pattern is sound. ToadStool can adopt
this pattern for `barracuda::dispatch` — users get GPU acceleration transparently
with zero-overhead CPU fallback.

### 3.5 Cross-Language Parity Methodology

**Finding**: Generating deterministic Python reference data (inputs + expected outputs)
as JSON, then loading in Rust for comparison, is a robust cross-language validation
pattern. No RNG dependency, no network calls, pure math comparison.

**Recommendation**: ToadStool could adopt `control/generate_*_references.py` →
JSON → `validate_*_parity.rs` as a standard pattern for upstream validation.

---

## Part 4: Absorption Recommendations

### Tier 1 — Ready Now (from V29 + V30)

| Item | Files | Priority |
|------|-------|----------|
| `chi_squared_f64.wgsl` | `metalForge/forge/src/shaders/` | P1 — fused single-dispatch |
| `kl_divergence_f64.wgsl` | `metalForge/forge/src/shaders/` | P1 — fused single-dispatch |
| HMM chain dispatch | `src/gpu_dispatch/dispatch_ops.rs` | P1 — compose forward/Viterbi steps |
| FST composed ops | `src/gpu_ops/population.rs` | P2 — pairwise/global from allele_freq |
| CPU parity methodology | `control/generate_cpu_references.py` | P2 — pattern for upstream |
| Dispatch tier benchmark | `src/bin/bench_dispatch_tiers.rs` | P3 — characterization tool |

### Tier 2 — Needs Upstream Evolution

| Item | Dependency | Priority |
|------|-----------|----------|
| HMM chain single-encoder | `StatefulPipeline` chain API | P1 — eliminates T×dispatch overhead |
| df64 HMM forward | `df64` chain support | P2 — precision for long sequences |
| Tridiagonal eigensolver | NAK eigensolve upstream | P3 — blocked since Session 44 |

### Tier 3 — Bug Reports (Existing)

| # | Issue | Status |
|---|-------|--------|
| S-14 | Naive matmul hang (small square matrices) | Workaround: A×B^T |
| S-15 | Matmul hang when elements ≤ 0.1 magnitude | Root-caused: driver bug |
| logsumexp | Buffer-size mismatch in logsumexp driver | Known upstream |

---

## Part 5: Full Metrics

### Validation Stack

| Tier | Coverage | Checks | Status |
|------|----------|--------|--------|
| Python baselines (Py) | 25/25 (100%) | 206 | **ALL PASS** |
| Rust CPU (Rs) | 25/25 (100%) | 505+ lib | **ALL PASS** |
| BarraCUDA CPU (bC) | 24/25 (96%) | 203 | **ALL PASS** |
| CPU↔Python parity | 18/18 operations | 39 | **ALL PASS** |
| GPU Tensor (gT) | 23/25 (92%) | 98+ | **ALL PASS** |
| metalForge WGSL (mF) | 15/25 | 108 | **ALL PASS** |
| GPU Pipeline (gP) | 15/25 | 94 | **ALL PASS** |
| Cross-dispatch (xD) | 15/15 (100%) | 49 | **ALL PASS** |
| GPU Promotion (Phase C) | 44 ops (~97%) | 18 | **ALL PASS** |
| Dispatch tiers | 10 kernels × 3 tiers | — | **Characterized** |
| Mixed-hardware (mH) | 14/14 | 14 | **ALL PASS** |
| Multi-GPU (mG) | RTX 4070 + TITAN V | bit-identical | **ALL PASS** |

### Performance

| Benchmark | Result |
|-----------|--------|
| Rust vs Python (11 kernels) | **201.7× faster** |
| CPU dispatch overhead | **≤1.04× (9/10 ops)** |
| GPU fused pipeline | **46–78× over per-op dispatch** |
| GPU vs Python (TF medium) | **104× faster** |

### Pipeline Status

```
Session 44:   Rust CPU 178.5× faster than Python/NumPy (7 kernels)
Session 66:   Rust CPU 201.7× faster (11 kernels) + ~97% GPU coverage
Session 67:   CPU = Python mathematically (39/39 PASS, 1e-10)
Session 67b:  Dispatch layer is transparent (≤1.04× overhead 9/10 ops)
Next:         ToadStool pipeline batching → GPU-resident acceleration
              → UnidirectionalPipeline streaming → pure GPU workloads
```

---

## Supersedes

- V29: Session 64 — forge v0.2.0, substrate discovery, workload tracking
  (`wateringHole/handoffs/archive/NEURALSPRING_TOADSTOOL_V29_S64_HANDOFF_FEB25_2026.md`)

---

*neuralSpring → ToadStool handoff V30 — AGPL-3.0-or-later*
