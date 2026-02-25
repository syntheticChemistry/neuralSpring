# neuralSpring → ToadStool/BarraCUDA Handoff V33

**Session 70 — Deep Audit II, Coverage Evolution, BarraCUDA Usage Inventory, Paper Control Matrix**
**Date**: February 25, 2026
**From**: neuralSpring
**To**: ToadStool / BarraCUDA core team
**License**: AGPL-3.0-or-later
**ToadStool HEAD**: `02207c4a`
**Supersedes**: V32 (Session 69 — Validator Shader Rewiring, Cross-Spring Evolution)

---

## Executive Summary

Session 70 completed the deepest code quality audit yet and delivers actionable
evolution recommendations for the ToadStool/BarraCUDA team:

1. **Coverage 90.43% → 94.53%** (+75 lib tests, 580 total). GPU-path tests for all
   44 Dispatcher operations. Remaining uncovered lines (5.5%) are exclusively GPU
   error-handling branches (`map_err` on device loss) — untestable without hardware
   fault injection.

2. **Tolerance registry macro refactoring**: `tolerance_registry!` declarative macro
   reduced `registry.rs` from 891 to 257 lines while preserving the full runtime
   introspection API. Pattern recommended for BarraCUDA's own tolerance management.

3. **Complete BarraCUDA usage inventory**: 90+ import sites across 60+ files spanning
   20+ barracuda submodules. Zero duplicate math — every operation delegates to upstream.

4. **Paper control matrix verified**: All 25 papers + 5 baseCamp sub-theses have
   controls at BarraCUDA CPU, BarraCUDA GPU, and metalForge mixed-hardware tiers.
   All controls use open data and open systems exclusively.

5. **100% SPDX compliance**: 211/211 source files carry `AGPL-3.0-or-later` headers.

---

## Part 1: What Changed (Session 70)

### Coverage Evolution

| Metric | S69 | S70 | Delta |
|--------|-----|-----|-------|
| Lib tests | 505 | 580 | +75 |
| Line coverage | 90.43% | 94.53% | +4.10pp |
| Named tolerances | 104+ | 105+ | +1 |
| Max file size | 966 lines | 966 lines | No change |
| SPDX compliance | ~100% | 100% (211/211) | Verified |

### Key Changes

| Change | Files | Impact |
|--------|-------|--------|
| GPU-path Dispatcher tests | `gpu_dispatch/tests_gpu.rs` (new, 483 lines) | All 44 GPU ops tested |
| Tolerance registry macro | `tolerances/registry.rs` | 891→257 lines, declarative |
| `SADDLE_EIGENVALUE_THRESHOLD` extracted | `loss_landscape.rs`, `tolerances/mod.rs` | Magic number eliminated |
| `bench.rs` named constants | `bench.rs` | `RATIO_NEGLIGIBLE`, `RATIO_INVESTIGATE`, `NANOS_PER_MICROSECOND` |
| Streaming I/O for JSON loading | `validate_cpu_math_parity.rs` | `BufReader` + `from_reader` |
| `gpu_dispatch/mod.rs` split | `mod.rs` (860 lines) + `tests_gpu.rs` (483 lines) | Under 1000-line limit |
| Python test import fix | `tests/conftest.py`, `test_benchmark_functions.py`, `test_determinism.py` | `generate_synthetic_weather` rename |
| Ruff lint fixes | `generate_cpu_references.py`, `surrogate_validation.py` | B905, F841 |

### Remaining Uncovered Lines (5.5%)

All in GPU error branches — patterns like:

```rust
device.create_buffer(...).map_err(|e| BarracudaError::DeviceLost(e))?;
```

These require hardware fault injection to exercise. No production logic is uncovered.

---

## Part 2: BarraCUDA Usage Inventory (Categorized)

neuralSpring consumes 20+ barracuda submodules across 90+ import sites. This is
the complete inventory — useful for BarraCUDA's own dependency analysis.

### Core Infrastructure

| Module | Usage Sites | Pattern |
|--------|------------|---------|
| `barracuda::device::WgpuDevice` | 40+ (gpu.rs, all validators, benchmarks) | GPU device creation |
| `barracuda::device::driver_profile::GpuDriverProfile` | `gpu_dispatch/mod.rs` | f64 strategy, pow workaround |
| `barracuda::unified_hardware::BandwidthTier` | `gpu_dispatch/mod.rs` | PCIe tier detection |
| `barracuda::error::{BarracudaError, Result}` | `gpu_dispatch`, `evolved`, validators | Error propagation |
| `barracuda::tensor::Tensor` | `gpu_ops/*`, `evolved/*`, validators | GPU tensor operations |

### Math Primitives (CPU)

| Module | Functions Used | Consumers |
|--------|--------------|-----------|
| `barracuda::stats` | `variance`, `pearson_correlation`, `covariance`, `norm_cdf`, `norm_pdf`, `norm_ppf`, `empirical_spectral_density`, `marchenko_pastur_bounds` | 15+ modules |
| `barracuda::special` | `chi_squared_statistic`, `gamma`, `erf`, `bessel_j0` | validators, cpu_fallback |
| `barracuda::numerical` | `numerical_hessian`, `rk45_solve`, `Rk45Config` | loss_landscape, regulatory |
| `barracuda::linalg` | `eigh_f64`, `eigh_householder_qr`, `effective_rank`, `graph_laplacian`, `disordered_laplacian`, `belief_propagation_chain` | spectral, neural_pgm, agent_coordination |
| `barracuda::sample` | `boltzmann_sampling`, `BoltzmannResult` | loss_landscape |
| `barracuda::spectral` | `BatchIprGpu`, `level_spacing_ratio` | weight_spectral, benchmarks |

### GPU Dispatch

| Module | Functions Used | Consumers |
|--------|--------------|-----------|
| `barracuda::dispatch` | `matmul_dispatch`, `frobenius_norm_dispatch`, `transpose_dispatch`, `softmax_dispatch`, `gelu_dispatch`, `l2_distance_dispatch`, `mean_dispatch`, `variance_dispatch`, `hmm_forward_dispatch` | gpu_dispatch/dispatch_ops.rs |

### GPU Operations (Typed Wrappers)

| Module | API | Used In |
|--------|-----|---------|
| `barracuda::ops::bio::*` | `BatchFitnessGpu`, `PairwiseHammingGpu`, `PairwiseJaccardGpu`, `LocusVarianceGpu`, `SpatialPayoffGpu`, `SwarmNnGpu`, `HillGateGpu`, `MultiObjFitnessGpu` | validators, benchmarks |
| `barracuda::ops::logsumexp` | `LogSumExp` | validators |
| `barracuda::ops::rk_stage` | `WGSL_RK4_PARALLEL` | validators (rewired S69) |
| `barracuda::ops::rk45_adaptive` | `WGSL_RK45_ADAPTIVE` | validators (rewired S69) |
| `barracuda::ops::mha` | `MultiHeadAttention` | evolved/mha.rs |
| `barracuda::ops::linalg` | `BatchedEighGpu` | benchmarks |
| `barracuda::staging` | `KernelDispatch`, `StatefulConfig`, `StatefulPipeline` | validators |

### Zero Duplicate Math

Every mathematical operation in neuralSpring either:
1. Delegates to `barracuda::*` (17 rewired functions + 6 shader source references)
2. Is a domain-specific composition of barracuda primitives (e.g., FST = allele_frequencies + variance)
3. Is an independent CPU reference for validation independence (`primitives.rs`)

---

## Part 3: Paper Control Matrix — BarraCUDA CPU → GPU → metalForge

### BarraCUDA CPU Controls (Tier 1)

| Validator | Papers | Checks | Primitives |
|-----------|--------|--------|-----------|
| `validate_cpu_math_parity` | All 25 | 39/39 | 9 primitives + 9 kernels + 6 Dispatcher |
| `validate_barracuda_{domain}` | 24/25 | 203/203 | stats, linalg, special, numerical |
| `validate_barracuda_parity` | 17 domains | 17/17 | CPU vs GPU per domain |

### BarraCUDA GPU Controls (Tier 2)

| Validator | Papers | Checks | GPU Ops |
|-----------|--------|--------|---------|
| `validate_barracuda_gpu_{domain}` | 23/25 | 98+ | Tensor matmul, transpose, tanh, sigmoid |
| `validate_gpu_phase_c` | 016-018, 024-025 | 18/18 | HMM chains, FST, introgression |
| `validate_basecamp_gpu` | baseCamp 01-05 | 14/14 | eigh, variance, Pearson, entropy |

### metalForge Mixed Hardware Controls (Tier 3)

| Validator | Checks | Substrates |
|-----------|--------|-----------|
| `validate_mixed_hardware` | 14/14 | GPU↔NPU↔CPU routing |
| `validate_mixed_dispatch` | 16/16 | PCIe transfer cost model |
| `validate_compute_dispatch` | 16/16 | CPU↔GPU parity |
| `validate_metalforge_pcie` | 23/23 | Bandwidth + latency tiers |

### Open Data Confirmation

| Source | Papers | License |
|--------|--------|---------|
| In-code synthetic (seed=42) | 011-025, baseCamp | N/A (pure math) |
| Open-Meteo ERA5 | Exp 003-004, Study 004-005 | CC BY 4.0 |
| MNIST | Study 003 | CC BY-SA 3.0 |
| GitHub repos | Study 001-002, Paper 012 | MIT / Apache-2.0 |

**No proprietary data. No API keys. All reproducible from scratch.**

---

## Part 4: Evolution Recommendations for ToadStool/BarraCUDA

### 4.1 Tolerance Registry Pattern (Recommended for Absorption)

neuralSpring's `tolerance_registry!` macro reduces boilerplate while maintaining
runtime introspection. The pattern:

```rust
macro_rules! tolerance_registry {
    ($( $cat:literal : [ $($name:ident),+ $(,)? ] ),+ $(,)?) => {
        &[ $($( NamedTolerance { name: stringify!($name), value: $name, category: $cat }, )+)+ ]
    };
}
```

Benefits: compile-time validation, categorized browsing, `tolerance_by_name()` lookup.
105+ named tolerances covering CPU, GPU, FFT, numerical, spectral, and baseCamp domains.

### 4.2 GPU Test Serialization (Still Recommended)

`OnceLock<Mutex<()>>` + shared GPU instance + sync tests prevents wgpu device
contention. neuralSpring's `test_gpu_lock` pattern works across 580 tests with
zero flakiness.

### 4.3 Streaming I/O Pattern

For validation binaries that load JSON references, `BufReader` + `serde_json::from_reader`
avoids buffering entire files into `String`. This matters for large reference datasets.

### 4.4 Macro-Based File Refactoring

When files exceed limits, consider `macro_rules!` to compress repetitive structures
rather than splitting into multiple files. The tolerance registry went from 891→257
lines without any API change.

### 4.5 Updated Absorption Recommendations

| Item | Priority | New Since V32 |
|------|----------|---------------|
| `WGSL_MEAN_REDUCE` public constant | P1 | No (still blocked) |
| `tolerance_registry!` macro pattern | P2 | **Yes** (S70) |
| GPU test serialization pattern | P2 | No (verified at 580 tests) |
| Streaming JSON loading pattern | P3 | **Yes** (S70) |
| `SADDLE_EIGENVALUE_THRESHOLD` concept | P3 | **Yes** (negative tolerance) |

### 4.6 What neuralSpring Needs from BarraCUDA (Unchanged)

| Need | Current Status |
|------|---------------|
| `WGSL_MEAN_REDUCE` public constant | Blocked — no public export |
| `argmax_dim()` for Tensor | Blocked — Viterbi needs CPU argmax |
| `softmax_dim(axis)` for Tensor | Blocked — attention needs row-wise |
| `StatefulPipeline` chain API | Available — not yet leveraged for HMM chains |
| S-14/S-15 matmul hang fix | Workaround in place (A×B^T, data ≥ 0.5) |

---

## Part 5: What neuralSpring Learned for the Ecosystem

### Coverage Insights

- **94.53% is near-ceiling** for GPU-dependent code. The remaining 5.5% is error
  handling that only triggers on device failure. Property-based testing and mocking
  won't help — these are `wgpu` error paths.

- **Smart refactoring beats splitting**: The tolerance registry macro compressed
  891→257 lines while improving readability. The `gpu_dispatch` test extraction
  (1332→860+483) was necessary for the 1000-line limit but the macro approach
  would have been better if applicable.

### BarraCUDA Stability

- **Zero breaking changes** across Sessions 40–70 (30 sessions). Every ToadStool
  sync compiled cleanly. The `barracuda` crate API is remarkably stable.

- **17 functions + 6 shader sources rewired** without any public API changes in
  neuralSpring. The thin-wrapper pattern (`local fn → barracuda::fn`) minimizes
  migration risk.

### Cross-Spring Value

- **hotSpring** precision infrastructure (df64, GpuDriverProfile, Welford) provides
  3.49× variance speedup, hardware-adaptive f64 strategy, and driver workarounds.

- **wetSpring** bio-compute (HMM f64, fused map-reduce, log_f64 fix) provides
  2.56× entropy speedup and 10⁹× HMM precision improvement.

- **neuralSpring** ML validation (eigh, batch fitness, tolerance registry) provides
  trillion-fold eigensolver accuracy, structured validation patterns, and the
  tolerance introspection framework.

---

## Part 6: Full Metrics (Session 70)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | **PASS** |
| `cargo clippy --all-targets -D warnings -W pedantic -W nursery` | **0 warnings** |
| `cargo test --lib` | **580/580 PASS** |
| `cargo test --test integration` | **9/9 PASS** |
| `cargo test --doc` | **9/9 PASS** (3 ignored) |
| `python3 -m pytest tests/` | **48/48 PASS** |
| `ruff check` + `ruff format --check` | **PASS** |
| `cargo llvm-cov --lib` | **94.53% line coverage** |
| Named tolerances | **105+** |
| Functions rewired to upstream | **17** |
| Validator shader sources rewired | **6** |
| SPDX compliance | **211/211 files** |
| Max file size | **966 lines** |
| Validation/bench binaries | **159** |

---

## Supersedes

- V32: Session 69 — Validator Shader Rewiring, Cross-Spring Evolution Benchmarks
  (`wateringHole/handoffs/archive/`)

---

*neuralSpring → ToadStool handoff V33 — AGPL-3.0-or-later*
