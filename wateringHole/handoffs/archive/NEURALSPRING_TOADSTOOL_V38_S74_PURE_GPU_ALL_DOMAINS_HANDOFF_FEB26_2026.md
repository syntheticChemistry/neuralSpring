# neuralSpring → ToadStool/BarraCUDA Handoff V38 — Pure GPU All-Domains

**Session 74 | February 26, 2026**
**Previous**: V37 (Session 73 — Comprehensive evolution, cross-spring rewiring)

---

## Part 1: Executive Summary

neuralSpring is a **validation Spring** — it proves that Python baselines from 25
published papers + 5 novel sub-theses can be faithfully ported to BarraCUDA (Rust)
and eventually promoted to ToadStool (GPU sovereign pipeline). The evolution path:

```
Python baseline → Rust validation → GPU acceleration → sovereign pipeline
```

### What's New in V38 (Session 74)

Session 74 closes the "pure GPU all-domains" milestone: every Phase 0++ paper domain
now runs through typed BarraCUDA GPU ops with scalar-only readback. This proves the
math is truly portable to GPU across all 15 paper domains.

| New Binary | Result | What It Proves |
|------------|--------|----------------|
| `validate_gpu_pure_workload_all` | **10/10 PASS** | 9 typed GPU ops across all Phase 0++ domains |
| `validate_cross_system_dispatch` | **46/46 PASS** | Full metalForge stack: discovery → heuristics → parity → NPU |
| `bench_evolution_tiers` | 8 domains | CPU→GPU portability, dispatch overhead characterization |

### Current State

| Metric | Value |
|--------|-------|
| Papers reproduced | 25 + 5 baseCamp sub-theses |
| Python baselines | 206/206 PASS |
| Rust+GPU checks | 1970+ PASS |
| Total validation | **2180+** checks |
| Library tests | **580/580** PASS |
| Integration tests | 9/9 PASS |
| Validation binaries | **163** |
| Coverage | **94.53%** (llvm-cov) |
| Named tolerances | **107+** (zero ad-hoc in test assertions) |
| Shortcomings | **17/17 RESOLVED** upstream |
| Upstream rewires | **21 functions + 6 shader sources** |
| Cross-spring validator | **39/39 PASS** |
| GPU promotion | ~97% of production math |
| Pure GPU all-domains | **10/10 PASS** (9 ops + determinism) |
| Cross-system dispatch | **46/46 PASS** (discovery + heuristics + parity + NPU) |
| CPU↔Python parity | 39/39 PASS (1e-10) |
| Dispatch overhead | ≤1.04× (9/10 ops) |
| Clippy warnings | 0 |
| Doc warnings | 0 |
| SPDX compliance | 100% (AGPL-3.0-or-later) |
| Files ≤1000 lines | 100% |
| Dependencies | All Pure Rust (ecoBin compliant) |

---

## Part 2: Pure GPU All-Domains Validation (NEW — S74)

### 9 Typed BarraCUDA GPU Ops + Determinism Check

`validate_gpu_pure_workload_all` dispatches to 9 typed GPU ops, one per paper domain,
with scalar-only readback (no full buffer round-trips). Each result is compared against
a CPU reference computed inline.

| # | Domain | GPU Op | Papers | Precision | Tolerance | Status |
|---|--------|--------|--------|-----------|-----------|--------|
| 1 | NK Fitness | `BatchFitnessGpu` | 011–013 | f64 | 1e-10 | **PASS** |
| 2 | Multi-obj Fitness | `MultiObjFitnessGpu` | 014 | f64 | 1e-10 | **PASS** |
| 3 | HMM Forward | `HmmBatchForwardF64` | 016–018 | f64 | 1e-10 | **PASS** |
| 4 | Spatial Payoff | `SpatialPayoffGpu` | 019 | f32 | 1e-6 | **PASS** |
| 5 | Batch IPR | `BatchIprGpu` | 022–023 | f32 | 1e-5 | **PASS** |
| 6 | Pairwise Hamming | `PairwiseHammingGpu` | 017 | f32 | 1e-6 | **PASS** |
| 7 | Pairwise L2 | `PairwiseL2Gpu` | 012 | f32 | 1e-5 | **PASS** |
| 8 | Pairwise Jaccard | `PairwiseJaccardGpu` | 024 | f32 | 1e-5 | **PASS** |
| 9 | Locus Variance | `LocusVarianceGpu` | 025 | f64 | 1e-10 | **PASS** |
| 10 | Determinism | Re-run BatchFitness | — | exact | 0.0 | **PASS** |

### f32 vs f64 Precision Boundary

The precision boundary is systematic and deliberate:

| Precision | Domains | Rationale |
|-----------|---------|-----------|
| f64 | Fitness, HMM, Locus Variance | Numerical stability required (log-space, accumulation) |
| f32 | IPR, L2, Hamming, Jaccard, Spatial Payoff | Domain-specific ops, GPU shader precision sufficient |

ToadStool implications: when absorbing these ops, maintain the f32/f64 boundary.
The f32 ops use `gpu.read_buffer_f32()` and `size: N * 4`; f64 ops use
`gpu.read_buffer_f64()` and `size: N * 8`.

### Data Preparation Lessons

| Op | Lesson |
|----|--------|
| `BatchIprGpu` | Input vectors **must be pre-normalized** (unit norm). GPU shader expects normalized eigenvectors. |
| `PairwiseJaccardGpu` | Input is **f32 presence/absence** (0.0/1.0). Output is **upper triangle** of distance matrix. |
| `SpatialPayoffGpu` | Input strategies are **u32**, benefit/cost are **f32**. Mixed-type dispatch. |
| `PairwiseHammingGpu` | Input sequences are **u32** tokens. Output distances are **f32**. |
| `HmmBatchForwardF64` | f64 path for numerical stability in log-space accumulation. |

---

## Part 3: Evolution Tier Benchmarks (NEW — S74)

### CPU → GPU Portability at Validation Scale

`bench_evolution_tiers` measures the same math on Rust CPU vs BarraCUDA GPU
at the validation problem sizes used by our test suite.

| Kernel | Scale | CPU µs | GPU µs | Winner | Notes |
|--------|-------|--------|--------|--------|-------|
| HMM forward | 3×5000 | 149 | 188 | CPU | GPU wins at 64+ states |
| NK fitness | 1000×10 | 0.3 | 183 | CPU | Dispatch overhead dominates |
| Pairwise Hamming | 20×500 | 49 | 186 | CPU | GPU crossover at 200×1000 |
| Pairwise L2 | 10×8 | 0.3 | 185 | CPU | GPU wins at 100×64 |
| Pairwise Jaccard | 30×500 | 316 | 186 | **GPU** | GPU already competitive |
| Spatial payoff | 6×6 | 0.5 | 184 | CPU | GPU wins at 128×128+ |
| Hill gate | 50×50 | 3.1 | 184 | CPU | GPU wins at 200×200+ |
| Commutator | 64×64 | 183 | — | — | CPU-only benchmark |

### Dispatch Overhead Characterization

GPU dispatch overhead is **~186µs** per `queue.submit()`. This is structural
(wgpu + Vulkan driver initialization). Implications for ToadStool:

1. **Batching is essential**: `StatefulPipeline` and `UnidirectionalPipeline`
   amortize this across multi-step chains (HMM, ODE, iterative solvers).
2. **Size-based routing works**: `Dispatcher` correctly routes small workloads
   to CPU. The ~1.5ms crossover is documented in Sessions 44 and 67b.
3. **Production scale is different**: At 50000×64 fitness evals or 200×1000
   Hamming comparisons, GPU is 4–84× faster than CPU (Session 44 data).

---

## Part 4: What neuralSpring Contributed to ToadStool

### Primitives Absorbed Upstream

These originated in neuralSpring and are now part of BarraCUDA:

| Primitive | Domain | Absorbed At | Cross-Spring Value |
|-----------|--------|------------|-------------------|
| `ValidationHarness` | Testing | S52 | Used by all Springs |
| `exit_no_gpu` | Testing | S52 | Graceful GPU-absent handling |
| `require!` macro | Testing | S52 | Error-to-fail-and-continue |
| `BatchFitnessGpu` | Evolution | S25 (`77f70b2e`) | Fitness landscape evaluation |
| `PairwiseL2Gpu` | Distance | S42 (`5437c170`) | MODES diversity metrics |
| `PairwiseHammingGpu` | Genomics | S25 | SATé alignment scoring |
| `PairwiseJaccardGpu` | Genomics | S25 | Pangenome comparison |
| `SpatialPayoffGpu` | Game theory | S25 | QS spatial game matrix |
| `HillGateGpu` | Regulatory | S25 | Hill function activation |
| `MultiObjFitnessGpu` | Evolution | S25 | Pareto frontier scoring |
| `BatchedEighGpu` | Eigensolver | S25 | Anderson localization |
| `BatchIprGpu` | Spectral | S25 | Inverse participation ratio |
| `SwarmNnGpu` | Robotics | S25 | Neural controller evaluation |
| 4-tier KernelRouter | Matmul | S39 | Size-based matmul dispatch |
| `empirical_spectral_density` | Statistics | S54 | Random matrix analysis |
| `marchenko_pastur_bounds` | Statistics | S54 | MP law bounds |
| `effective_rank` | Statistics | S54 | Eigenvalue entropy rank |
| `gelu_dispatch` | Activations | S52 | ML activation function |
| `hmm_forward_dispatch` | Bio | S52 | HMM phylogenetics |
| S-17 `pow(f64)` fix | Precision | S58 | Transcendental polyfill |

### 21 WGSL Shaders Absorbed

All 21 neuralSpring WGSL shaders have been absorbed into ToadStool's shader
inventory. The `evolved/` directory contains fossil records only.

### 21 Functions Rewired to Upstream

| # | Function | Upstream API | Session | Lineage |
|---|----------|-------------|---------|---------|
| 1 | `mat_mul` | `domain_ops::matmul_dispatch` | S58 | hotSpring precision (df64, KernelRouter) |
| 2 | `frobenius_norm` | `domain_ops::frobenius_norm_dispatch` | S58 | hotSpring precision |
| 3 | `transpose` | `domain_ops::transpose_dispatch` | S58 | hotSpring precision (S-16 fix) |
| 4 | `softmax` | `domain_ops::softmax_dispatch` | S58 | hotSpring f64 numerics |
| 5 | `l2_distance` | `domain_ops::l2_distance_dispatch` | S58 | hotSpring/wetSpring |
| 6 | `mean` | `domain_ops::mean_dispatch` | S58 | hotSpring Welford |
| 7 | `variance` | `domain_ops::variance_dispatch` | S58 | hotSpring Welford |
| 8 | `gelu` | `domain_ops::gelu_dispatch` | S59 | neuralSpring ML → absorbed S52 |
| 9 | `hmm_forward_step` | `domain_ops::hmm_forward_dispatch` | S59 | wetSpring bio → absorbed S52 |
| 10 | `graph_laplacian` | `barracuda::linalg::graph` | S56 | baseCamp sub-05 |
| 11 | `disordered_laplacian` | `barracuda::linalg::graph` | S56 | baseCamp sub-05 |
| 12 | `belief_propagation_chain` | `barracuda::linalg::graph` | S56 | baseCamp sub-04 |
| 13 | `numerical_hessian` | `barracuda::numerical` | S56 | baseCamp sub-03 |
| 14 | `empirical_spectral_density` | `barracuda::stats` | S59 | neuralSpring spectral → absorbed S54 |
| 15 | `marchenko_pastur_bounds` | `barracuda::linalg` | S59 | neuralSpring spectral → absorbed S54 |
| 16 | `effective_rank` | `barracuda::linalg` | S59 | neuralSpring PGM → absorbed S54 |
| 17 | `boltzmann_sampling` | `barracuda::sample` | S68 | neuralSpring counterdiabatic |
| 18 | `softmax_row_wise` | `Tensor::softmax_dim(1)` | S73 | neuralSpring V20 → ToadStool S60 |
| 19 | `fst_single_locus` | `barracuda::ops::bio::fst_variance_decomposition` | S73 | wetSpring S53 → BarraCUDA bio |
| 20 | `pairwise_fst_full` | upstream per-locus decomposition | S73 | wetSpring S53 → BarraCUDA bio |
| 21 | Viterbi argmax | `Tensor::argmax_dim(0)` | S73 | neuralSpring V20 → ToadStool S60 |

---

## Part 5: Cross-Spring Evolution Provenance

```
hotSpring → BarraCUDA precision layer:
  • df64_core.wgsl (double-float f32-pair emulation)
  • pow_f64 polyfill → S-17 RESOLVED (patch_transcendentals covers pow)
  • Fp64Strategy (Native/Hybrid detection)
  • GpuDriverProfile (hardware-adaptive dispatch)
  • Taylor-series sin/cos (7-term + Cody-Waite)
  • Lanczos eigensolver (lattice QCD heritage)
  • Welford variance, thermodynamic reductions
  • BatchIprGpu from spectral primitives (S74 pure GPU validation)

wetSpring → BarraCUDA bio+spectral layer:
  • HMM forward/backward (phylogenetics)
  • 5 ODE bio systems (Capacitor, Cooperation, MultiSignal, Bistable, PhageDefense)
  • NMF, Anderson localization, ridge regression
  • fst_variance_decomposition (F-statistics: θ, f_is, f_it)
  • SpatialPayoffGpu game theory stencil (S74 pure GPU validation)

neuralSpring → BarraCUDA validation+ops layer:
  • ValidationHarness, exit_no_gpu, require! macro
  • 13 GPU ops (batch fitness, pairwise L2/hamming/jaccard, spatial payoff, etc.)
  • eigh, batch IPR, swarm NN, KernelRouter
  • ESD, marchenko_pastur, effective_rank
  • gelu_dispatch, hmm_forward_dispatch
  • S73: softmax_row_wise, fst_single_locus, pairwise_fst_full, Viterbi argmax_dim
  • S74: 9-domain pure GPU validation (all typed ops proven correct)

All three → ToadStool GPU sovereign pipeline:
  599+ WGSL shaders, unified dispatch, multi-substrate
```

---

## Part 6: Shortcomings — ALL 17 RESOLVED

| # | Description | Resolution |
|---|-------------|-----------|
| S-01 | Missing batch fitness GPU | RESOLVED (S25, `77f70b2e`) |
| S-02 | Missing pairwise L2 GPU | RESOLVED (S42, `5437c170`) |
| S-03 | Missing eigh GPU | RESOLVED (S25, `77f70b2e`) |
| S-03b | MHA decomposition | RESOLVED (S60, `0c998992`) |
| S-04–S-13 | Various GPU ops | RESOLVED (S25–S42) |
| S-14 | Naive matmul hang | RESOLVED (S39, `a4996b34`: Naive tier removed) |
| S-15 | Matmul hang ≤0.1 magnitude | RESOLVED (S39, `a4996b34`) |
| S-16 | Transpose workgroup dispatch | RESOLVED (S39, `a4996b34`: TILE=16) |
| S-17 | `pow(f64,f64)` NVVM crash | RESOLVED (S58, `c82c23d1`: `patch_transcendentals`) |

---

## Part 7: Recommendations for ToadStool/BarraCUDA

### New from S74

1. **f32/f64 type boundary**: Document which GPU ops expect f32 vs f64 input/output.
   The boundary is systematic (see Part 2 table). Callers must match types.

2. **Pre-normalization requirements**: `BatchIprGpu` requires pre-normalized
   eigenvectors (unit norm). Consider adding a `normalize` step to the op or
   documenting the precondition in the API.

3. **Upper-triangle output**: `PairwiseJaccardGpu` outputs a flat upper-triangle
   of the distance matrix (not a full symmetric matrix). Callers need to account
   for this when interpreting results.

4. **Dispatch overhead floor**: ~186µs per `queue.submit()` is structural. The
   `UnidirectionalPipeline` streaming model will amortize this for production
   workloads. Validate that streaming dispatch reduces overhead as expected.

### Carried from V37

5. **Tolerance registry pattern** — `tolerance_registry!` macro with compile-time
   validation and runtime introspection. 107+ constants organized by category.

6. **Cross-spring evolution validator** — `validate_cross_spring_evolution` (39 checks)
   validates every upstream rewire and benchmarks cross-spring lineage.

7. **metalForge mixed-hardware dispatch** — `Dispatcher::mixed_dispatch()` routes
   through `metalForge::mixed::mixed_substrate()` cost model.

### Tensor API Evolution Requests

| Request | Priority | Status |
|---------|----------|--------|
| `argmax_dim(axis)` | P0 | **AVAILABLE** (S60, used in S73) |
| `softmax_dim(axis)` | P0 | **AVAILABLE** (S60, used in S73) |
| `fst_variance_decomposition` | P1 | **AVAILABLE** (S53, used in S73) |
| GPU `argmax_dim` (WGSL) | P2 | Available via `argmax_dim_keepdim` |
| GPU `softmax_dim` (WGSL) | P3 | Not yet — current impl is CPU-only |
| `Tensor::viterbi` (full chain) | P3 | Not implemented — would fuse Viterbi loop |
| `Tensor::normalize(dim)` | P3 | Would simplify IPR input prep (NEW, S74) |

---

## Part 8: Evolution Path — Proven End-to-End

```
Python/NumPy (baseline, 206/206 PASS)
  ↓ 201.7× faster
Rust CPU (pure math, BarraCUDA CPU, 39/39 parity vs Python)
  ↓ transparent dispatch (≤1.04× overhead)
BarraCUDA GPU (same WGSL math, typed ops)
  ↓ 4–84× faster at production scale
Pure GPU pipeline (scalar-only readback, 10/10 PASS — S74)
  ↓ 46/46 PASS (S74)
metalForge cross-system (GPU → NPU → CPU, 46/46 PASS — S74)
  ↓ next milestone
ToadStool sovereign pipeline (UnidirectionalPipeline streaming)
```

The **entire evolution path from Python to pure GPU is now validated**.
The next frontier is metalForge cross-system dispatch and ToadStool streaming.

---

## Part 9: Quality Gates

| Gate | Result |
|------|--------|
| `cargo fmt --check` | **PASS** |
| `cargo clippy --lib` | **0 warnings** |
| `cargo test --lib` | **580/580 PASS** |
| `cargo test --test integration` | **9/9 PASS** |
| `validate_cross_spring_evolution` | **39/39 PASS** |
| `validate_gpu_pure_workload_all` | **10/10 PASS** |
| `validate_cross_system_dispatch` | **46/46 PASS** |
| `validate_all` | **149/150 PASS** (1 known logsumexp) |
| `cargo doc --no-deps` | **0 warnings** |
| Python baselines | **206/206 PASS** |
| CPU↔Python parity | **39/39 PASS** (1e-10) |
| Coverage | **94.53%** |
| SPDX compliance | **100%** |

---

## Part 10: Document Index

| Document | Location | Purpose |
|----------|----------|---------|
| This handoff | `wateringHole/handoffs/` | V38 pure GPU all-domains handoff |
| BARRACUDA_USAGE | `specs/BARRACUDA_USAGE.md` | Module-level usage inventory |
| CROSS_SPRING_EVOLUTION | `specs/CROSS_SPRING_EVOLUTION.md` | Shader/primitive provenance |
| TOADSTOOL_HANDOFF | `specs/TOADSTOOL_HANDOFF.md` | Shortcoming tracking (all resolved) |
| EVOLUTION_READINESS | `EVOLUTION_READINESS.md` | Module → WGSL → pipeline mapping |
| BENCHMARK_ANALYSIS | `specs/BENCHMARK_ANALYSIS.md` | 3-way scaling + evolution tier benchmarks |
| Experiment 042 | `experiments/README.md` | S74 pure GPU all-domains journal |
| V37 (archived) | `wateringHole/handoffs/archive/` | S73 comprehensive evolution handoff |

---

*AGPL-3.0-or-later | neuralSpring → ToadStool V38 | Session 74 | February 26, 2026*
