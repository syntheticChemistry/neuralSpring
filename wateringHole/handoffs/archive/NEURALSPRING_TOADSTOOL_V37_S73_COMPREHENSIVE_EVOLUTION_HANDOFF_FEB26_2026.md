# neuralSpring → ToadStool/BarraCUDA Handoff V37 — Comprehensive Evolution

**Session 73 | February 26, 2026**
**Previous**: V36 (Session 73 — Cross-spring rewiring, 4 upstream Tensor API rewires)

---

## Part 1: Executive Summary

neuralSpring is a **validation Spring** — it proves that Python baselines from 25
published papers + 5 novel sub-theses can be faithfully ported to BarraCUDA (Rust)
and eventually promoted to ToadStool (GPU sovereign pipeline). The evolution path:

```
Python baseline → Rust validation → GPU acceleration → sovereign pipeline
```

### Current State

| Metric | Value |
|--------|-------|
| Papers reproduced | 25 + 5 baseCamp sub-theses |
| Python baselines | 206/206 PASS |
| Rust+GPU checks | 1910+ PASS |
| Total validation | **2120+** checks |
| Library tests | **580/580** PASS |
| Integration tests | 9/9 PASS |
| Coverage | **94.53%** (llvm-cov) |
| Named tolerances | **107+** (zero ad-hoc in test assertions) |
| Shortcomings | **17/17 RESOLVED** upstream |
| Upstream rewires | **21 functions + 6 shader sources** |
| Cross-spring validator | **39/39 PASS** |
| GPU promotion | ~97% of production math |
| CPU↔Python parity | 39/39 PASS (1e-10) |
| Dispatch overhead | ≤1.04× (9/10 ops) |
| Clippy warnings | 0 |
| Doc warnings | 0 |
| SPDX compliance | 100% (AGPL-3.0-or-later) |
| Files ≤1000 lines | 100% |
| Dependencies | All Pure Rust (ecoBin compliant) |

---

## Part 2: What neuralSpring Contributed to ToadStool

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

---

## Part 3: What neuralSpring Consumes from ToadStool

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
| 18 | `softmax_row_wise` | `Tensor::softmax_dim(1)` | **S73** | neuralSpring V20 → ToadStool S60 |
| 19 | `fst_single_locus` | `barracuda::ops::bio::fst_variance_decomposition` | **S73** | wetSpring S53 → BarraCUDA bio |
| 20 | `pairwise_fst_full` | upstream per-locus decomposition | **S73** | wetSpring S53 → BarraCUDA bio |
| 21 | Viterbi argmax | `Tensor::argmax_dim(0)` | **S73** | neuralSpring V20 → ToadStool S60 |

### 6 Validator Shader Sources Rewired

RK4, RK45, batch fitness, logsumexp, swarm NN, multi-obj fitness shader
`include_str!` paths → upstream `barracuda::*_WGSL` constants (S69).

### Additional BarraCUDA Surface

- **20+ submodules** consumed (device, tensor, stats, linalg, dispatch, ops::bio, etc.)
- **90+ import sites** across library and validation code
- **117+ upstream APIs** exercised
- `GpuDriverProfile` wired in for f64 strategy detection (Hybrid on RTX 4070)
- `BandwidthTier` wired in for PCIe transfer cost modelling
- `NVK guard` for TITAN V allocation safety

---

## Part 4: Cross-Spring Evolution Provenance

```
hotSpring → BarraCUDA precision layer:
  • df64_core.wgsl (double-float f32-pair emulation)
  • pow_f64 polyfill → S-17 RESOLVED (patch_transcendentals covers pow)
  • Fp64Strategy (Native/Hybrid detection)
  • GpuDriverProfile (hardware-adaptive dispatch)
  • Taylor-series sin/cos (7-term + Cody-Waite)
  • Lanczos eigensolver (lattice QCD heritage)
  • Welford variance, thermodynamic reductions

wetSpring → BarraCUDA bio+spectral layer:
  • HMM forward/backward (phylogenetics)
  • 5 ODE bio systems (Capacitor, Cooperation, MultiSignal, Bistable, PhageDefense)
  • NMF, Anderson localization, ridge regression
  • fst_variance_decomposition (F-statistics: θ, f_is, f_it)

neuralSpring → BarraCUDA validation+ops layer:
  • ValidationHarness, exit_no_gpu, require! macro
  • 13 GPU ops (batch fitness, pairwise L2/hamming/jaccard, spatial payoff, etc.)
  • eigh, batch IPR, swarm NN, KernelRouter
  • ESD, marchenko_pastur, effective_rank
  • gelu_dispatch, hmm_forward_dispatch
  • S73: softmax_row_wise, fst_single_locus, pairwise_fst_full, Viterbi argmax_dim

All three → ToadStool GPU sovereign pipeline:
  599+ WGSL shaders, unified dispatch, multi-substrate
```

---

## Part 5: Shortcomings — ALL 17 RESOLVED

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

**Validator workarounds retained** (defense-in-depth): positive-only test data
for matmul (S-15), A×B^T pattern (S-14). These produce correct results regardless
and would require extensive retesting to remove.

---

## Part 6: Recommendations for ToadStool/BarraCUDA

### Absorption Gap

ToadStool references neuralSpring handoffs V16/V18 (absorbed at S39). Handoffs
V19–V37 have not been consumed. This handoff (V37) provides the complete delta.

Key items to absorb:
1. **Tolerance registry pattern** — `tolerance_registry!` macro with compile-time
   validation and runtime introspection. 107+ constants organized by category.
2. **Smart refactoring pattern** — `gpu_dispatch` split: `mod.rs` (304 lines),
   `dispatch_ops.rs` (domain methods), `tests_cpu.rs`, `tests_gpu.rs`, `basecamp.rs`
3. **Cross-spring evolution validator** — `validate_cross_spring_evolution` (39 checks)
   validates every upstream rewire and benchmarks cross-spring lineage
4. **metalForge mixed-hardware dispatch** — `Dispatcher::mixed_dispatch()` routes
   through `metalForge::mixed::mixed_substrate()` cost model

### Tensor API Evolution Requests

| Request | Priority | Status |
|---------|----------|--------|
| `argmax_dim(axis)` | P0 | **AVAILABLE** (S60, used in S73) |
| `softmax_dim(axis)` | P0 | **AVAILABLE** (S60, used in S73) |
| `fst_variance_decomposition` | P1 | **AVAILABLE** (S53, used in S73) |
| GPU `argmax_dim` (WGSL) | P2 | Available via `argmax_dim_keepdim` |
| GPU `softmax_dim` (WGSL) | P3 | Not yet — current impl is CPU-only |
| `Tensor::viterbi` (full chain) | P3 | Not implemented — would fuse Viterbi loop |

### Performance Observations

For neuralSpring's validation-scale workloads (N<64 states, <256 matrix elements),
GPU dispatch overhead exceeds compute. The `Dispatcher` size-based thresholds
correctly route these to CPU. Benefits manifest at production scale.

| Operation | GPU (µs) | CPU (µs) | Notes |
|-----------|----------|----------|-------|
| `softmax_row_wise(4×64)` | ~4000 | ~1 | Device init dominates |
| `softmax_row_wise(64×256)` | ~4000 | ~65 | Batching needed |
| `viterbi(s=32, T=10)` | ~83000 | ~10 | Per-step round-trips |
| `matmul(128×128)` | ~2000 | ~300 | GPU wins at larger N |

### What We Learned (Relevant to ToadStool Evolution)

1. **f32 round-trip precision**: `softmax_dim` and `argmax_dim` operate in f32.
   For downstream f64 validation, expect ~1e-7 agreement. Tolerance
   `DISPATCH_F32_ROUNDTRIP` (1e-6) covers this.

2. **Mean-of-ratios vs ratio-of-sums**: `pairwise_fst_full` (using upstream
   per-locus `fst_variance_decomposition`) gives a different FST estimator
   than `pairwise_fst` (ratio-of-sums). Both are valid Weir-Cockerham; differ
   by ~1% on typical data.

3. **Viterbi on GPU**: The per-timestep dispatch overhead dominates for small
   HMM state spaces. A fused `Tensor::viterbi` that runs the entire chain
   on GPU would avoid T round-trips. For N>256 states, GPU wins at step level.

4. **NVK (nouveau) allocation guard**: TITAN V NVK PTE-faults at ~1.4 GB
   combined allocation. `GpuDriverProfile::check_allocation_safe()` prevents
   this. Essential for multi-GPU deployment.

5. **Defense-in-depth workarounds**: Even after upstream fixes, retain validator
   patterns that avoid historically-buggy paths (positive-only data, A×B^T).
   Cost is zero; removing requires retesting all validators.

---

## Part 7: Quality Gates

| Gate | Result |
|------|--------|
| `cargo fmt --check` | **PASS** |
| `cargo clippy --lib` | **0 warnings** |
| `cargo test --lib` | **580/580 PASS** |
| `cargo test --test integration` | **9/9 PASS** |
| `validate_cross_spring_evolution` | **39/39 PASS** |
| `cargo doc --no-deps` | **0 warnings** |
| Python baselines | **206/206 PASS** |
| CPU↔Python parity | **39/39 PASS** (1e-10) |
| Coverage | **94.53%** |
| SPDX compliance | **100%** |

---

## Part 8: Document Index

| Document | Location | Purpose |
|----------|----------|---------|
| This handoff | `wateringHole/handoffs/` | Comprehensive evolution handoff |
| BARRACUDA_USAGE | `specs/BARRACUDA_USAGE.md` | Module-level usage inventory |
| CROSS_SPRING_EVOLUTION | `specs/CROSS_SPRING_EVOLUTION.md` | Shader/primitive provenance |
| TOADSTOOL_HANDOFF | `specs/TOADSTOOL_HANDOFF.md` | Shortcoming tracking (all resolved) |
| EVOLUTION_READINESS | `EVOLUTION_READINESS.md` | Module → WGSL → pipeline mapping |
| BARRACUDA_REQUIREMENTS | `specs/BARRACUDA_REQUIREMENTS.md` | Primitive requirements |
| Experiment 041 | `experiments/README.md` | S73 cross-spring rewiring journal |
| V36 (archived) | `wateringHole/handoffs/archive/` | S73 initial rewiring handoff |

---

*AGPL-3.0-or-later | neuralSpring → ToadStool V37 | Session 73 | February 26, 2026*
