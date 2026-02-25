# neuralSpring → ToadStool/BarraCUDA Handoff V32

**Session 69 — Validator Shader Rewiring, Cross-Spring Evolution Benchmarks, Provenance Map**
**Date**: February 25, 2026
**From**: neuralSpring
**To**: ToadStool / BarraCUDA core team
**License**: AGPL-3.0-or-later
**ToadStool HEAD**: `02207c4a`
**Supersedes**: V31 (Session 68 — Deep Debt Audit, Tolerance Centralization)

---

## Executive Summary

Session 69 completes the shader-source lean phase and delivers a comprehensive
cross-spring evolution benchmark with full provenance mapping:

1. **6 validator shader sources rewired** from local `include_str!` to upstream
   barracuda constants — validators now pull WGSL from the single source of truth.

2. **Upstream-vs-local benchmark**: 10 neuralSpring-origin shaders compared across
   local manual dispatch vs barracuda wrapper APIs. Result: negligible overhead
   (8/10 ≈, 2/10 ~, zero ⚠). The absorption architecture works.

3. **Cross-spring provenance**: Complete map of how hotSpring precision, wetSpring
   bio, and neuralSpring ML contributions flow through ToadStool/BarraCUDA.
   645+ WGSL shaders traced to origin Springs with benchmark validation.

4. **Quality gates green**: 505 lib + 9 integration tests, 147/148 validate_all,
   22/22 cross-spring evolution, 0 clippy warnings.

---

## Part 1: What Changed (Session 69)

### Validator Shader Source Rewiring

| Validator | Shader | Old Source | New Source |
|-----------|--------|-----------|-----------|
| `validate_gpu_rk4` | `rk4_parallel.wgsl` | `include_str!` local | `barracuda::ops::rk_stage::WGSL_RK4_PARALLEL` |
| `validate_gpu_rk45` | `rk45_adaptive.wgsl` | `include_str!` local | `barracuda::ops::rk45_adaptive::WGSL_RK45_ADAPTIVE` |
| `validate_gpu_stateful_pipeline` | `rk4_parallel.wgsl` | `include_str!` local | `barracuda::ops::rk_stage::WGSL_RK4_PARALLEL` |
| `validate_gpu_pure_workload` | `batch_fitness_eval.wgsl` | `include_str!` local | `barracuda::ops::bio::batch_fitness::WGSL_BATCH_FITNESS_EVAL` |
| `validate_gpu_logsumexp` | `logsumexp_reduce.wgsl` | `include_str!` local | `barracuda::ops::logsumexp::LogSumExp::WGSL_LOGSUMEXP_REDUCE` |
| `validate_gpu_pipeline_swarm` | `swarm_nn_scores.wgsl` | `include_str!` local | `barracuda::ops::bio::swarm_nn::WGSL_SWARM_NN_SCORES` |

### Remaining Local `include_str!` (Intentional / Blocked)

| File | Shader | Reason |
|------|--------|--------|
| `validate_gpu_pure_workload.rs` | `mean_reduce.wgsl` | barracuda uses internally but no public `WGSL_MEAN_REDUCE` constant |
| `validate_mha_gpu.rs` | `head_split.wgsl`, `head_concat.wgsl` | No upstream shader equivalent |
| `bench_upstream_vs_local.rs` | 10 shaders | Intentional — benchmark compares local vs upstream dispatch paths |

**Recommendation for ToadStool**: Expose `WGSL_MEAN_REDUCE` as a public constant in
`barracuda::shaders::reduce` or `barracuda::ops::mean` so consumers can reference
the shader source directly (same pattern as `WGSL_BATCH_FITNESS_EVAL`, etc.).

---

## Part 2: Cross-Spring Evolution — The Complete Provenance Map

### 2.1 hotSpring Contributions (Precision Physics & Lattice QCD)

hotSpring established BarraCUDA's f64 math foundation. neuralSpring uses these
in every f64 validation.

| Contribution | BarraCUDA Location | neuralSpring Impact |
|-------------|-------------------|---------------------|
| `df64_core.wgsl` (f32-pair emulation) | `shaders/math/` | All f64 GPU validation |
| `pow_f64` polyfill (S-17) | `shaders/math/math_f64.wgsl` | HillGate f64 on all drivers |
| `SubstrateCapability` enum | `device::substrate` | Cross-dispatch routing |
| `SHADER_F64` adapter detection | `device::wgpu_device` | Dual CPU/GPU tensor validation |
| `GpuDriverProfile` | `device::driver_profile` | Hardware-adaptive f64 strategy |
| `VarianceReduceF64` (Welford) | `ops::variance_reduce_f64` | **3.49× faster** than f32 Tensor variance |
| Lanczos eigensolver | `spectral::lanczos` | Anderson localization diagnostics |
| `BatchedEighGpu` | `ops::linalg::eigh_f64` | GPU eigendecomposition |
| Hermite/Laguerre polynomials | `special::*` | Special function validation |
| `weighted_dot_f64` | `ops::weighted_dot_f64` | f64 precision inner product |
| `target` WGSL keyword fix | All shaders | Driver correctness |
| Spectral theory module | `spectral::*` | BatchIprGpu pipeline |
| 7-term Taylor sin/cos + Cody-Waite | `shaders/math/trig_f64.wgsl` | f64 transcendental accuracy |

### 2.2 wetSpring Contributions (Bioinformatics & Genomics)

wetSpring established BarraCUDA's bio-compute layer and critical f64 precision fixes.

| Contribution | BarraCUDA Location | neuralSpring Impact |
|-------------|-------------------|---------------------|
| `log_f64` coefficient fix | `shaders/math/math_f64.wgsl` | All f64 shader math (critical!) |
| Ada Lovelace NVVM f64 workaround | `device::*` | RTX 4070 GPU support |
| `FusedMapReduceF64` (Shannon, Simpson) | `ops::fused_map_reduce_f64` | **2.56× faster** entropy |
| `CorrelationF64` | `ops::correlation_f64_wgsl` | **1.33× faster** Pearson |
| `HmmBatchForwardF64` | `ops::bio::hmm` | f64 batch HMM validation |
| Smith-Waterman banded f64 | `ops::bio::smith_waterman` | Available |
| Gillespie SSA f64 | `ops::bio::gillespie` | GPU stochastic simulation |
| Felsenstein f64 | `ops::bio::felsenstein` | Phylogenetic likelihood |
| `TaxonomyFcGpu`, `KmerHistogramGpu`, `UniFracPropagateGpu` | `ops::bio::*` | Validated from neuralSpring |
| `chi_squared_statistic` | `special::*` | CPU fallback chi² |
| `pearson_correlation` | `stats::*` | CPU fallback Pearson |
| `cosine_similarity_f64` | `ops::cosine_similarity_f64` | f64 tensor validation |
| `Bray-Curtis f64` | `ops::batch_pair_reduce_f64` | Diversity distance |

### 2.3 neuralSpring Contributions (ML Validation & Evolutionary Computation)

neuralSpring established BarraCUDA's ML and evolutionary computation layer.

| Contribution | BarraCUDA Location | Cross-Spring Impact |
|-------------|-------------------|---------------------|
| `eigh_householder_qr` | `ops::linalg::eigh_f64` | **Trillion-fold accuracy** vs Jacobi at n≥8 (all Springs benefit) |
| `TensorSession` ML ops | `session::*` | matmul, relu, gelu, softmax, layer_norm |
| 4-tier `KernelRouter` | `ops::matmul` | Auto-tuning for all Springs |
| `empirical_spectral_density` | `stats::empirical_spectral_density` | Eigenvalue analysis (all Springs) |
| `marchenko_pastur_bounds` | `stats::marchenko_pastur_bounds` | Random matrix theory |
| `effective_rank` | `linalg::effective_rank` | Entropy-based dimensionality |
| `boltzmann_sampling` | `sample::boltzmann_sampling` | MCMC loss landscape exploration |
| `ValidationHarness` | Adapted upstream | Structured validation pattern |
| `batch_fitness_eval.wgsl` | `ops::bio::batch_fitness` | EA fitness evaluation |
| `pairwise_hamming.wgsl` | `ops::bio::pairwise_hamming` | Alignment distance |
| `pairwise_jaccard.wgsl` | `ops::bio::pairwise_jaccard` | Pangenome distance |
| `pairwise_l2.wgsl` | `ops::bio::pairwise_l2` | MODES novelty (closed-form pair decode) |
| `locus_variance.wgsl` | `ops::bio::locus_variance` | FST / allele freq variance |
| `spatial_payoff.wgsl` | `ops::bio::spatial_payoff` | Game theory stencil |
| `batch_ipr.wgsl` | `spectral::batch_ipr` | Spectral localization |
| `hill_gate.wgsl` | `ops::bio::hill_gate` | Signal AND gate (mode 0/1 generalization) |
| `multi_obj_fitness.wgsl` | `ops::bio::multi_obj_fitness` | Directed evolution (Bessel correction) |
| `swarm_nn_forward.wgsl` | `ops::bio::swarm_nn` | Swarm NN inference (generic MLP) |
| `rk4_parallel.wgsl` | `ops::rk_stage` | Parallel ODE integration |
| `rk45_adaptive.wgsl` | `ops::rk45_adaptive` | Adaptive Dormand-Prince |
| `hmm_forward_log.wgsl` | `ops::bio::hmm` / `shaders/ml/` | HMM forward (f32) |

### 2.4 Collaborative Contributions (Multi-Spring)

| Contribution | Springs | BarraCUDA Location |
|-------------|---------|-------------------|
| `pow_f64` polyfill (S-17) | hotSpring + wetSpring | `shaders/math/math_f64.wgsl` |
| `CrankNicolson` | airSpring + wetSpring + hotSpring | `ops::crank_nicolson` |
| `FusedMapReduceF64` | wetSpring (entropy) + hotSpring (convergence norms) | `ops::fused_map_reduce_f64` |
| `GemmF64` cached | wetSpring (60× taxonomy speedup) | `ops::linalg::gemm_f64` |
| `CyclicReductionF64` | airSpring + wetSpring + hotSpring | `ops::cyclic_reduction_f64` |
| `MovingWindowStats` | airSpring + wetSpring | `ops::moving_window_stats` |

### 2.5 Cross-Spring Evolution Flow

```text
hotSpring precision ──→ df64_core, pow_f64, Welford variance, Lanczos
                        ↓                                      ↓
                     BarraCUDA ←── ToadStool absorption ←── metalForge
                        ↓
wetSpring bio ─────→ HMM forward, fused map-reduce, log_f64 fix, dN/dS
                        ↓
                     BarraCUDA (now has precision + bio)
                        ↓
neuralSpring ML ───→ eigh, batch_fitness, pairwise_l2, spectral density
                        ↓
                     BarraCUDA (now has precision + bio + ML)
                        ↓
               ╔═══════════════════════════╗
               ║  All Springs lean on the  ║
               ║  shared math engine:      ║
               ║  • 645+ WGSL shaders      ║
               ║  • 17 rewired functions    ║
               ║  • 6 shader sources →      ║
               ║    upstream constants     ║
               ║  • 117+ upstream APIs      ║
               ╚═══════════════════════════╝
```

---

## Part 3: Benchmark Results (RTX 4070, `--release`, Session 69)

### Upstream vs Local Shader Dispatch

Same WGSL shader content, different dispatch paths. Measures wrapper overhead:

| Kernel | Origin Paper | Local (µs) | Upstream (µs) | Ratio |
|--------|-------------|-----------|--------------|-------|
| BatchFitness 10k×32 | nS 011-015 | 1,840 | 2,060 | 1.12× ~ |
| Hamming 200×500 | nS 017 (SATé) | 1,807 | 1,947 | 1.08× ≈ |
| Jaccard 100×500 | nS 024 (Pangenome) | 1,972 | 1,849 | 0.94× ≈ |
| LocusVariance 50×500 | nS 025 (MetaPop) | 2,035 | 2,043 | 1.00× ≈ |
| SpatialPayoff 256² | nS 019 (GameTheory) | 1,903 | 1,890 | 0.99× ≈ |
| BatchIPR 1k×256 | nS 022-023 (Anderson) | 1,909 | 2,301 | 1.21× ~ |
| HillGate 100² | nS 021 (Signal) | 2,101 | 2,003 | 0.95× ≈ |
| MultiObjFitness 5k×4 | nS 014 (DirEvo) | 1,978 | 1,943 | 0.98× ≈ |
| PairwiseL2 200×50 | nS 012 (MODES) | 2,031 | 1,940 | 0.96× ≈ |
| SwarmNN 500×20 | nS 015 (Swarm) | 1,990 | 1,999 | 1.00× ≈ |

**Key insight**: Upstream wrapper overhead is within noise for 8/10 ops. BatchIPR
shows 21% overhead — investigate buffer re-creation in `BatchIprGpu::run()`.

### Cross-Spring Typed GPU Op Benchmark

| Op | Size | Median (µs) | Origin Spring | Absorption |
|----|------|-------------|---------------|------------|
| `BatchFitnessGpu` | 1024×64 | 2,000 | neuralSpring (ML) | S-25 |
| `PairwiseL2Gpu` | 128×16 | 1,994 | neuralSpring (MODES) | S-42 |
| `BatchIprGpu` | 32×64 | 2,064 | neuralSpring (Anderson) | S-25 |
| `SpatialPayoffGpu` | 32×32 | 2,102 | neuralSpring (game theory) | S-25 |
| `PairwiseHammingGpu` | 64×100 | 2,027 | neuralSpring (SATé) | S-25 |
| `HmmBatchForwardF64` | 4s×50t×32b | 2,085 | wetSpring (phylo) | S-39 |
| `BatchedEighGpu` | 12×12×40 | 7,497 | hotSpring (nuclear) | S-39 |

### Rewire Evolution (f32 Tensor → f64 Upstream, 10,000 elements)

| Op | f32 Tensor (µs) | f64 Upstream (µs) | Speedup | Cross-Spring Origin |
|----|-----------------|-------------------|---------|---------------------|
| Variance | 9,949 | 2,847 | **3.49×** | hotSpring Welford |
| Pearson | 4,679 | 3,508 | **1.33×** | wetSpring + hotSpring |
| Entropy | 6,317 | 2,468 | **2.56×** | wetSpring fused map-reduce |

---

## Part 4: Lessons for ToadStool Evolution

### 4.1 Expose WGSL Constants Consistently

Several barracuda ops use `include_str!` internally but don't expose the WGSL
source as a public constant. neuralSpring's `validate_gpu_pure_workload.rs` still
uses local `include_str!` for `mean_reduce.wgsl` because barracuda's `Mean` op
doesn't expose `WGSL_MEAN_REDUCE`.

**Pattern to follow** (already used by batch_fitness, pairwise_hamming, etc.):
```rust
pub const WGSL_MEAN_REDUCE: &str = include_str!("../shaders/reduce/mean_reduce.wgsl");
```

### 4.2 Cross-Spring Shader Evolution Benefits All

When hotSpring adds a precision fix (e.g., `log_f64` coefficients), wetSpring and
neuralSpring benefit automatically because they lean on the same upstream shaders.
This is the core value of ToadStool's absorption model — fixes propagate to all
consumers without per-Spring coordination.

| Fix | Origin | Beneficiaries |
|-----|--------|--------------|
| `log_f64` coefficients | wetSpring | All f64 shaders across all Springs |
| `pow_f64` polyfill | hotSpring + wetSpring | HillGate, multi_obj_fitness, regulatory |
| `target` keyword fix | hotSpring | All WGSL shaders |
| Ada Lovelace NVVM | wetSpring | All f64 GPU ops on RTX 40xx |
| Taylor sin/cos (7-term) | hotSpring | All trig-dependent shaders |

### 4.3 BatchIPR Wrapper Overhead

`bench_upstream_vs_local` shows `BatchIprGpu` at 1.21× overhead vs local dispatch.
This is the highest among the 10 benchmarked ops. Likely cause: buffer re-creation
per `run()` call. Consider caching the bind group layout or pre-allocating output
buffers for repeated invocations.

### 4.4 Provenance Tags Are Valuable

barracuda's `provenance.rs` tags (`PROV_CG_SHADERS`, `PROV_BATCH_FITNESS`, etc.)
are invaluable for tracking which Spring contributed what. neuralSpring uses these
to build `bench_cross_spring_evolution` and `validate_cross_spring_evolution`.
**Recommendation**: Continue adding provenance tags for every absorption.

### 4.5 GPU Test Serialization Pattern (from V31)

Still recommended: `OnceLock<Mutex<()>>` + shared GPU instance + sync tests.
Prevents wgpu device contention in parallel test runs. neuralSpring's pattern
(`test_gpu_lock`) works across 505+ tests with zero flakiness.

### 4.6 Tolerance Registry Pattern (from V31)

Still recommended: centralized tolerance constants with runtime introspection
(`all_tolerances()`, `tolerance_by_name()`, `categories()`). neuralSpring has
104+ named tolerances covering CPU, GPU, FFT, numerical, and spectral domains.

---

## Part 5: Updated Absorption Recommendations

### Tier 1 — Ready Now

| Item | Files | Priority | New Since V31 |
|------|-------|----------|---------------|
| `WGSL_MEAN_REDUCE` public constant | `barracuda::ops::mean` | P1 | **Yes** |
| `chi_squared_f64.wgsl` | `metalForge/forge/src/shaders/` | P1 | No |
| `kl_divergence_f64.wgsl` | `metalForge/forge/src/shaders/` | P1 | No |
| HMM chain dispatch | `src/gpu_dispatch/dispatch_ops.rs` | P1 | No |
| FST composed ops | `src/gpu_ops/population.rs` | P2 | No |
| Tolerance registry pattern | `src/tolerances/` | P2 | No |
| GPU test serialization | `src/lib.rs` (test_gpu_lock) | P2 | No |
| CPU parity methodology | `control/generate_cpu_references.py` | P2 | No |

### Tier 2 — Needs Upstream Evolution

| Item | Dependency | Priority |
|------|-----------|----------|
| HMM chain single-encoder | `StatefulPipeline` chain API | P1 |
| df64 HMM forward | `df64` chain support | P2 |
| `BatchIprGpu` buffer caching | Reduce per-call overhead (1.21× → ~1.0×) | P2 |

### Tier 3 — Bug Reports (Existing)

| # | Issue | Status |
|---|-------|--------|
| S-14 | Naive matmul hang (small square matrices) | Workaround: A×B^T |
| S-15 | Matmul hang when elements ≤ 0.1 magnitude | Root-caused: driver bug |
| logsumexp | Buffer-size mismatch in logsumexp driver | Known upstream |

---

## Part 6: Full Metrics (Session 69)

| Gate | Result |
|------|--------|
| `cargo fmt --all -- --check` | **PASS** |
| `cargo clippy --all-targets -D warnings` | **0 warnings** |
| `cargo test --lib` | **505/505 PASS** |
| `cargo test --test integration` | **9/9 PASS** |
| `validate_all` | **147/148 PASS** (1 pre-existing logsumexp) |
| `validate_cross_spring_evolution` | **22/22 PASS** |
| `bench_upstream_vs_local` | **10/10 ≈ or ~ (zero ⚠)** |
| Named tolerances | **104+** |
| Functions rewired to upstream | **17** |
| Validator shader sources rewired | **6** |
| Validation/bench binaries | **159** |
| Total validation checks | **2120+** |
| Coverage | **90.43%** |

---

## Part 7: What neuralSpring Proved for ToadStool

1. **The absorption model works**: 21/21 WGSL shaders absorbed, 17 functions
   rewired, 6 shader sources now referencing upstream constants. Zero behavioral
   regressions across 2120+ checks.

2. **Cross-spring evolution is real**: hotSpring precision fixes (df64_core, pow_f64),
   wetSpring bio features (HMM f64, fused map-reduce), and neuralSpring ML ops
   (eigh, batch fitness, spectral density) all flow through ToadStool and benefit
   every consumer. The shared math engine is stronger than any Spring alone.

3. **Upstream wrapper overhead is negligible**: 8/10 ops within noise (≈), 2/10
   within 12–21% (~). No ops warrant investigation (⚠). The typed wrapper API is
   safe to use in production workloads.

4. **Three-tier hardware coverage holds**: CPU 24/25 (96%), GPU 23/25 (92%),
   mixed 15/15 (100%). Open data confirmed for all 25+5 papers.

5. **The "Lean" phase is nearly complete**: Only 3 local `include_str!` references
   remain (1 blocked on missing public constant, 2 no upstream equiv). All
   function delegations are done. neuralSpring is maximally lean on ToadStool.

---

## Supersedes

- V31: Session 68 — Deep Debt Audit, Tolerance Centralization
  (`wateringHole/handoffs/archive/`)

---

*neuralSpring → ToadStool handoff V32 — AGPL-3.0-or-later*
