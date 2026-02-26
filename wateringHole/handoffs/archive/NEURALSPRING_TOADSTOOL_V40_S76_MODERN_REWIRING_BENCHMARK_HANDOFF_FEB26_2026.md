# neuralSpring → ToadStool/BarraCUDA Handoff V40 — Modern Rewiring + Benchmark Validation

**Session 76 | February 26, 2026**
**Previous**: V39 (Session 75 — Upstream sync S60–S65, 9 stats rewires, 150/150)

---

## Part 1: Executive Summary

Session 76 completes the modern BarraCUDA rewiring pass. A deep scan of all 57
library modules identified remaining local implementations with upstream equivalents.
Two additional Pearson correlation functions in `meta_population.rs` were rewired
to `barracuda::stats::pearson_correlation`. A full benchmark sweep validates that
upstream BarraCUDA wrappers add zero meaningful overhead vs raw metalForge dispatch
(0.85–1.14× across 10 GPU kernels), and cross-spring evolved f64 shaders outperform
naïve f32 Tensor paths by 1.4–3.2×.

### What's New in V40 (Session 76)

| Action | Details |
|--------|---------|
| **+2 functions rewired** | `matrix_correlation`, `thermal_diversity_correlation` → `barracuda::stats::pearson_correlation` |
| **Benchmark validation** | 10/10 GPU kernels show ≈ parity between local and upstream dispatch |
| **Cross-spring f64 gains** | Variance 3.20× (hotSpring Welford), Shannon 2.24× (wetSpring fused), Pearson 1.36× |
| **Documentation sweep** | All root docs, specs, whitePaper, experiments updated to S76 state |
| **Total rewires** | **32 functions + 6 shader sources** (was 30) |

---

## Part 2: Rewiring Details

### New Rewires

```rust
// meta_population.rs — was computing Pearson inline with manual mean/variance/covariance
pub fn matrix_correlation(a: &[f64], b: &[f64], n: usize) -> f64 {
    // ... extract upper triangle ...
    barracuda::stats::pearson_correlation(&xs, &ys).unwrap_or(0.0)
}

pub fn thermal_diversity_correlation(pi_values: &[f64], temperatures: &[f64]) -> f64 {
    barracuda::stats::pearson_correlation(pi_values, temperatures).unwrap_or(0.0)
}
```

Cross-spring origin: `barracuda::stats::pearson_correlation` was absorbed from
airSpring/groundSpring hydrology metrics in ToadStool S64.

### Remaining Local Implementations (By Design)

These are intentionally NOT delegated and should remain local:

| Module | Function | Reason |
|--------|----------|--------|
| `primitives.rs` | `sigmoid`, `rk4_step`, `shannon_entropy`, `hill_activation`, etc. | CPU validation references — independent of upstream for correctness checking |
| `spectral_commutativity.rs` | `frobenius_norm`, `mat_mul`, `commutator` | GPU validation references — must be independent |
| `pangenome_selection.rs` | `spectrum_chi_squared`, `env_association_chi2` | Domain-specific expected-value computation differs from upstream interface |
| `sequence.rs` | LSTM gate dot products | Inline in closure, compiler-inlined — no benefit from delegation |
| `gpu_dispatch/cpu_fallback.rs` | `variance` | Population variance (÷N) vs upstream sample variance (÷(N-1)) |
| Various modules | Inline `iter().sum::<f64>() / n` mean | 3–5 element arrays in hot loops — function call overhead unjustified |

---

## Part 3: Benchmark Results (RTX 4070, Release)

### Upstream Wrappers vs Local metalForge Dispatch

All 10 kernels show negligible overhead — BarraCUDA wrappers are free:

| Kernel | Origin | Local µs | Upstream µs | Ratio |
|--------|--------|----------|-------------|-------|
| BatchFitness 10000×32 | neuralSpring 011-015 | 1624 | 1618 | 1.00× |
| Hamming 200×500 | neuralSpring SATé | 1781 | 2032 | 1.14× |
| Jaccard 100×500 | neuralSpring Pangenome | 2266 | 1918 | 0.85× |
| LocusVariance 50×500 | neuralSpring MetaPop | 1817 | 1914 | 1.05× |
| SpatialPayoff 256×256 | neuralSpring GameTheory | 1886 | 1933 | 1.02× |
| BatchIPR 1000×256 | neuralSpring Anderson | 1891 | 1724 | 0.91× |
| HillGate 100×100 | neuralSpring Signal | 1863 | 1647 | 0.88× |
| MultiObjFitness 5000×4 | neuralSpring DirEvo | 1597 | 1778 | 1.11× |
| PairwiseL2 200×50 | neuralSpring MODES | 1768 | 1720 | 0.97× |
| SwarmNN 500×20 | neuralSpring Swarm | 1627 | 1681 | 1.03× |

### Cross-Spring f64 Evolution (GPU, 10K elements)

| Op | f32 Tensor (µs) | f64 Evolved (µs) | Speedup | Origin |
|----|-----------------|-------------------|---------|--------|
| Variance | 8,221 | 2,570 | **3.20×** | hotSpring Welford |
| Pearson | 4,361 | 3,214 | **1.36×** | wetSpring + hotSpring |
| Shannon | 4,216 | 1,886 | **2.24×** | wetSpring fused map-reduce |

### CPU Stats Throughput (barracuda::stats, 10K elements)

| Metric | Source | µs/iter |
|--------|--------|---------|
| RMSE | airSpring → ToadStool S64 | 4.1 |
| R² | airSpring → ToadStool S64 | 12.4 |
| NSE | airSpring → ToadStool S64 | 12.5 |
| Index of Agreement | airSpring → ToadStool S64 | 14.0 |
| dot | shared → ToadStool S64 | 4.2 |
| l2_norm | shared → ToadStool S64 | 4.1 |
| Shannon | wetSpring → ToadStool S64 | 1.8 |
| Simpson | wetSpring → ToadStool S64 | 0.7 |
| Chao1 | wetSpring → ToadStool S64 | 0.2 |
| Bray-Curtis | wetSpring → ToadStool S64 | 0.1 |
| Pearson r | hotSpring + wetSpring | 26.1 |

---

## Part 4: Cross-Spring Evolution Story

Each spring contributes domain expertise; ToadStool absorbs and GPU-accelerates;
all springs benefit:

```text
hotSpring (precision physics)    → f64 math, Welford variance, spectral theory, lattice QCD
wetSpring (bioinformatics)       → biodiversity metrics, HMM, fused map-reduce, FST decomposition
airSpring (atmospheric)          → hydrology stats (RMSE, R², NSE, IA, hit_rate)
groundSpring (soil hydrology)    → multinomial sampling, MC propagation, noise labels
neuralSpring (ML/neuroevolution) → batch fitness, pairwise ops, swarm NN, 4-tier matmul, validation harness
                                         ↓
                                   ToadStool absorbs
                                         ↓
                              694+ WGSL shaders, all springs benefit
```

**Key cross-spring speedups validated this session**:
- hotSpring's Welford online variance (single-pass f64) → **3.20×** vs 4-dispatch f32
- wetSpring's fused map-reduce (1 GPU dispatch vs 3) → **2.24×** for entropy
- Joint hotSpring+wetSpring f64 correlation → **1.36×** for Pearson r

---

## Part 5: Current State

| Metric | Value |
|--------|-------|
| Papers reproduced | 25 + 5 baseCamp sub-theses |
| Python baselines | 206/206 PASS |
| Rust+GPU checks | 1970+ PASS |
| Total validation | **2180+** checks |
| Library tests | **580/580** PASS |
| Forge tests | **43/43** PASS |
| Integration tests | 9/9 PASS |
| Validation binaries | **163** |
| Named tolerances | **107+** |
| Upstream rewires | **32 functions + 6 shader sources** |
| ToadStool HEAD | `17932267` (S65) |

### Quality Gates (all green)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | **PASS** |
| `cargo clippy --workspace -- -D warnings` | **0 warnings** |
| `cargo test --workspace` | **580 lib + 43 forge + 9 integration PASS** |
| `validate_cross_spring_evolution` | **39/39 PASS** |
| `validate_gpu_pure_workload_all` | **10/10 PASS** |
| `validate_cross_system_dispatch` | **46/46 PASS** |
| `validate_all` | **150/150 PASS** |
| `bench_upstream_vs_local` | **10/10 ≈ parity** |
| `cargo doc --no-deps` | **0 warnings** |
| CPU↔Python parity | **39/39 PASS** (1e-10) |
| Coverage | **94.53%** |
| SPDX compliance | **100%** |

---

## Part 6: Recommendations for ToadStool Team

### Carried from V39 (still relevant)

1. **Add `stats::mae`**: neuralSpring keeps a local MAE (`metrics.rs`) because upstream
   only has `Tensor::mae_loss` (GPU). A CPU `barracuda::stats::mae(obs, sim)` would
   let us retire the local version.

2. **Re-export `WGSL_RK4_PARALLEL`**: S65 refactoring stopped re-exporting the WGSL
   constant. neuralSpring works around this via `include_str!` but a public constant
   would be cleaner.

3. **`shannon(frequencies)` variant**: `barracuda::stats::shannon(counts)` accepts
   count data. A `shannon_from_frequencies()` variant would let neuralSpring retire
   `primitives::shannon_entropy()` (which operates on pre-normalized frequencies).

### New from V40

4. **`pearson_correlation` return type**: Returns `Result<f64, CorrelationError>`.
   All neuralSpring call sites use `.unwrap_or(0.0)` because degenerate inputs
   (constant vectors) are valid in our domain. Consider adding a
   `pearson_correlation_or(default)` convenience or making the error case return
   `NaN` instead of `Err` to match NumPy/SciPy semantics.

5. **Population vs sample variance**: `barracuda::stats::variance` uses sample
   variance (÷(N-1)). neuralSpring's `cpu_fallback::variance` uses population
   variance (÷N) intentionally. Consider adding `variance_population()` or a
   `ddof` parameter to avoid this divergence.

6. **Cross-spring benchmark standardization**: The `bench_cross_spring_evolution`
   pattern (provenance-traced benchmarks with `ValidationHarness`) could become a
   shared `barracuda::bench` module so all springs can emit comparable benchmark
   reports.

---

## Part 7: Document Index

| Document | Location | Purpose |
|----------|----------|---------|
| This handoff | `wateringHole/handoffs/` | V40 modern rewiring + benchmarks |
| BARRACUDA_USAGE | `specs/BARRACUDA_USAGE.md` | Module-level usage inventory |
| CROSS_SPRING_EVOLUTION | `specs/CROSS_SPRING_EVOLUTION.md` | Shader/primitive provenance |
| CROSS_SPRING_SHADER_LINEAGE | `whitePaper/CROSS_SPRING_SHADER_LINEAGE.md` | WGSL shader evolution narrative |
| TOADSTOOL_HANDOFF | `specs/TOADSTOOL_HANDOFF.md` | Shortcoming tracking (all resolved) |
| EVOLUTION_READINESS | `EVOLUTION_READINESS.md` | Module → WGSL → pipeline mapping |
| Experiment 044 | `experiments/README.md` | S76 benchmark validation journal |
| Cross-spring bench | `bench_cross_spring_evolution` | Provenance-traced benchmark (15/15 PASS) |
| Upstream vs local bench | `bench_upstream_vs_local` | Wrapper overhead benchmark (10/10 ≈) |
| V39 (archived) | `wateringHole/handoffs/archive/` | S75 upstream sync S60–S65 |

---

*AGPL-3.0-or-later | neuralSpring → ToadStool V40 | Session 76 | February 26, 2026*
