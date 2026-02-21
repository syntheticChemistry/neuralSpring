# neuralSpring → ToadStool: Shader Evolution Handoff — 12 WGSL Shaders, Phase 4d Complete

**Date:** 2026-02-21
**From:** neuralSpring (ML / isomorphic learning / scholarly reproduction Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**ToadStool reviewed:** commit `dc540afd` (Session 25, Feb 20, 2026)
**Supersedes:** Feb 20 consolidated handoff (for shader inventory only)

---

## Executive Summary

neuralSpring has completed **25 experiments** across 5 scientific disciplines
with **966 total checks** (206 Python + 760 Rust+GPU). All Phase 0++
modules (15 papers) now have **full GPU coverage**: Python baseline → Rust
native → BarraCUDA CPU → GPU WGSL shader → cross-dispatch → pure GPU pipeline.

**Key updates (Phase 4a–4d):**
- **Performance benchmarks**: Rust 71.8× faster than single-thread NumPy (7 kernels)
- **Pure GPU end-to-end pipelines**: 4 multi-kernel chains, 20/20 PASS, zero CPU round-trips
- **GPU dispatch crossover mapping**: ~1.5ms Vulkan dispatch cost confirmed empirically
- **GPU PRNG**: Xoshiro128** shader (5/5 PASS), enables stochastic GPU algorithms
- **S-12 RESOLVED**: Householder+QR eigensolver in `src/eigh.rs` (9/9 PASS, machine-epsilon accuracy)
- **S-03b PARTIAL FIX**: GPU `head_split.wgsl` + `head_concat.wgsl` for MHA (10/10 PASS)
- **12 validated WGSL shaders** (9 paper kernels + PRNG + head\_split + head\_concat), all exported as `pub const WGSL_*`

---

## Part 1: WGSL Shader Inventory (12 shaders, 93/93 PASS)

All shaders are in `metalForge/shaders/` and exported via `include_str!`.

| Shader | Rust Export | Domain | Checks | Absorption Target |
|--------|-----------|--------|--------|-------------------|
| `hmm_forward_log.wgsl` | `hmm::WGSL_HMM_FORWARD_LOG` | Phylogenetics (016–018) | 13/13 | `barracuda::ops::hmm` |
| `batch_fitness_eval.wgsl` | `evolved::WGSL_BATCH_FITNESS_EVAL` | Evolution (011–015) | 20/20 | `barracuda::ops::batch_gemm` |
| `rk4_parallel.wgsl` | `evolved::WGSL_RK4_PARALLEL` | Regulatory/Signal (020–021) | 8/8 | `barracuda::ops::ode` |
| `mean_reduce.wgsl` | `evolved::WGSL_MEAN_REDUCE` | Fitness aggregation | 7/7 | `barracuda::pipeline::ReduceScalarPipeline` |
| `pairwise_jaccard.wgsl` | `pangenome_selection::WGSL_PAIRWISE_JACCARD` | Pangenome (024) | 6/6 | `barracuda::ops::pairwise_distance` |
| `locus_variance.wgsl` | `meta_population::WGSL_LOCUS_VARIANCE` | Meta-population (025) | 7/7 | `barracuda::ops::VarianceReduceF64` |
| `spatial_payoff.wgsl` | `game_theory::WGSL_SPATIAL_PAYOFF` | Game theory (019) | 5/5 | `barracuda::ops::stencil` |
| `batch_ipr.wgsl` | `anderson_localization::WGSL_BATCH_IPR` | Spectral (022–023) | 5/5 | `barracuda::ops::batch_reduce` |
| `pairwise_hamming.wgsl` | `sate_alignment::WGSL_PAIRWISE_HAMMING` | Alignment (017) | 5/5 | `barracuda::ops::pairwise_distance` |
| `xoshiro128ss.wgsl` | `rng::WGSL_XOSHIRO128SS` | All stochastic algorithms | 5/5 | `barracuda::ops::prng` |
| `head_split.wgsl` | `evolved::WGSL_HEAD_SPLIT` | MHA (S-03b fix) | 5/5 | `barracuda::ops::mha` |
| `head_concat.wgsl` | `evolved::WGSL_HEAD_CONCAT` | MHA (S-03b fix) | 5/5 | `barracuda::ops::mha` |

### New in this handoff (3 shaders)

**`spatial_payoff.wgsl`** — Spatial prisoner's dilemma payoff stencil.
One thread per cell, 8 Moore neighbors with periodic boundary. Uses
integer×1000 encoding for payoff parameters to avoid f32 uniform precision
issues. Validated at max GPU–CPU diff 1.91e-6.

**`batch_ipr.wgsl`** — Batch inverse participation ratio for eigenvectors.
One thread per eigenvector, computes sum(|ψ_i|^4). Validated against CPU
`jacobi_eigh` + `ipr()` at max diff 1.26e-7. Demonstrates the extended →
localized transition at the Aubry-André critical point.

**`pairwise_hamming.wgsl`** — Pairwise Hamming distance for sequence comparison.
One thread per sequence pair, counts differing sites and divides by length.
Validated at max GPU–CPU diff 4.77e-8 (near f32 epsilon for the division).

---

## Part 2: Cross-Dispatch Validation (45/45 PASS)

| Binary | BarraCUDA API | Checks | Status |
|--------|--------------|--------|--------|
| `validate_gpu_stateful_pipeline` | `StatefulPipeline` | 10 | **PASS** |
| `validate_gpu_pure_workload` | Multi-kernel chain | 7 | **PASS** |
| `validate_cross_dispatch` | `DispatchConfig` | 8 | **PASS** |
| `validate_cross_dispatch_genomics` | Jaccard + variance | 8 | **PASS** |
| `validate_cross_dispatch_extended` | Payoff + IPR + Hamming | 12 | **PASS** |

The extended cross-dispatch validator proves GPU ↔ CPU parity for all 3 new
shader domains plus correct dispatch routing (small workloads → CPU, large → GPU).

---

## Part 3: BarraCUDA Module Documentation (hotSpring Pattern)

All 13 Phase 0++ Rust library modules now have standardized doc sections:

```
//! ## `BarraCUDA` connection
//!
//! - [primitive]: `barracuda::ops::*` or `barracuda::stats::*`
//! - ...
//!
//! ## WGSL shader (absorption-ready)
//!
//! [`WGSL_*`] — description. Validated in `validate_gpu_*`.
```

This follows the hotSpring pattern where each module documents:
1. Which BarraCUDA primitives its math maps to
2. Which WGSL shader(s) implement the GPU path
3. Validation binary and check count

### Modules with BarraCUDA connection sections (13)

| Module | Key BarraCUDA Primitives | WGSL Export |
|--------|--------------------------|-------------|
| `hmm` | `StatefulPipeline`, `logsumexp`, `matmul` | `WGSL_HMM_FORWARD_LOG` |
| `pangenome_selection` | `pairwise_distance` | `WGSL_PAIRWISE_JACCARD` |
| `meta_population` | `VarianceReduceF64`, `pearson_correlation` | `WGSL_LOCUS_VARIANCE` |
| `game_theory` | `batch_gemm`, `stencil` | `WGSL_SPATIAL_PAYOFF` |
| `anderson_localization` | `eigh_f64`, `FusedMapReduceF64` | `WGSL_BATCH_IPR` |
| `sate_alignment` | `pairwise_distance` | `WGSL_PAIRWISE_HAMMING` |
| `counterdiabatic` | `batch_gemm`, `softmax`, `FusedMapReduceF64` | — |
| `directed_evolution` | `stats::variance`, `batch_gemm` | — |
| `eco_dynamics` | `pairwise_distance`, `batch_gemm` | — |
| `modes` | `SumReduceF64`, `pairwise_distance` | — |
| `spectral_commutativity` | `matmul`, `NormReduceF64` | — |
| `regulatory_network` | `numerical::rk45_solve`, `elementwise` | — |
| `signal_integration` | `numerical::rk45_solve`, `elementwise` | — |

Additional modules: `swarm_robotics`, `introgression`, `transformer`, `metrics`, `sequence`.

---

## Part 4: Absorption Recommendations

### Immediate (drop-in WGSL copy)

These 12 shaders can be absorbed by ToadStool with minimal changes:
1. Copy WGSL source from `metalForge/shaders/`
2. Create `barracuda::ops::*` wrapper with matching binding layout
3. neuralSpring rewires to upstream `barracuda::*` import

### Medium-term (new primitives)

| Primitive | Papers | Why |
|-----------|--------|-----|
| `barracuda::ops::stencil` | 019 | 2D neighborhood convolution, reusable for spatial models |
| `barracuda::ops::batch_reduce` | 022–023 | Batch sum-of-powers reduction, generalizes IPR |
| `barracuda::ops::cdist` | 017, 024 | Pairwise distance with pluggable metric (Hamming, Jaccard) |
| `barracuda::ops::prng` | All stochastic | GPU-parallel PRNG, enables Wright-Fisher/Gillespie/EA gen loops |

### Outstanding (S-12) — RESOLVED

`eigh_f64` accuracy gap **locally resolved** via Householder tridiagonalization
+ QL implicit shift (`src/eigh.rs`). Machine-epsilon accuracy at all sizes
(n=4–64), validated in `validate_eigh_accuracy` (9/9 PASS). Ready for
absorption: replace Jacobi iteration in `barracuda::linalg::eigh_f64` with
Householder+QR approach.

### Outstanding (S-03b) — PARTIAL FIX

MHA projection shader hang resolved by decomposition: use `Tensor::matmul`
(already validated) + new `head_split.wgsl` / `head_concat.wgsl` for
GPU-resident data reindexing. Validated in `validate_mha_gpu` (10/10 PASS).
Absorption: replace fused `mha_projection.wgsl` with `matmul` + `head_split` +
`head_concat` pipeline.

---

## Part 5: GPU Dispatch Crossover (Phase 4c)

`bench_gpu_kernels` empirically confirms the crossover point for dispatch routing:

| Kernel | Scale | GPU µs | Rust CPU µs | Winner |
|--------|-------|--------|-------------|--------|
| Hamming | Small (20×500) | 1,589 | 34 | CPU 46× |
| Hamming | Large (200×1000) | 1,675 | 7,089 | **GPU 4.2×** |
| Jaccard | Small (30×500) | 1,659 | 142 | CPU 12× |
| Jaccard | Large (100×2000) | 1,464 | 8,246 | **GPU 5.6×** |

**Crossover**: ~1.5ms Vulkan dispatch overhead. GPU wins when CPU work > 1.5ms.
This validates `barracuda::dispatch` routing and `StatefulPipeline` design.

---

## Part 6: Check Summary

| Layer | Checks | Status |
|-------|--------|--------|
| Python baselines | 206/206 | **PASS** |
| Rust native validation | 183/183 | **PASS** |
| BarraCUDA CPU primitives | 272/272 | **PASS** |
| BarraCUDA CPU ports | 147/147 | **PASS** |
| GPU shader validation | 69/69 | **PASS** |
| GPU pipeline/cross-dispatch | 65/65 | **PASS** |
| GPU PRNG validation | 5/5 | **PASS** |
| Phase 4d (S-12 + S-03b) | 19/19 | **PASS** |
| **Total** | **966/966** | **ALL PASS** |

Hardware: NVIDIA GeForce RTX 4070 (12 GB, Vulkan, proprietary driver).

---

*neuralSpring shader evolution handoff — following the hotSpring Write → Absorb → Lean pattern.*
