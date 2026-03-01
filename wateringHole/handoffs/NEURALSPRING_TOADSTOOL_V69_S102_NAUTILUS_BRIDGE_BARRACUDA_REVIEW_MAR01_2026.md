# neuralSpring → ToadStool/BarraCUDA Handoff V69 — Nautilus Bridge + BarraCUDA Evolution Review

**Date**: March 1, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Sessions 101–102 — ToadStool S71 pin bump (`8dc01a37`), GPU stats parity, 2 upstream shader bugs, Nautilus Shell cross-spring bridge (`bingocube-nautilus`), comprehensive BarraCUDA usage review
**Supersedes**: V68 (GPU Stats Parity + Shader Bugs)

---

## Executive Summary

- **Nautilus Shell integration**: neuralSpring now bridges hotSpring's evolutionary reservoir computing via `bingocube-nautilus`. `SpectralNautilusBridge` maps weight spectral features to Nautilus `BetaObservation`. Feed-forward alternative to recurrent ESN — 27/27 validation PASS
- **ToadStool S71 GPU stats validated**: `KimuraGpu` (max diff 1.11e-16), `HistogramGpu` (correct bins/counts). `JackknifeMeanGpu` and `HargreavesBatchGpu` blocked by upstream shader bugs
- **2 upstream shader bugs** (from V68, still open):
  - `jackknife_mean_f64.wgsl`: `bitcast<f64>(vec2<u32>())` breaks naga DF64 emulation
  - `hargreaves_batch_f64.wgsl`: `enable f64;` directive not supported by naga
- **BarraCUDA footprint**: 198 import sites, 58+ distinct stats functions, 20+ submodules, 47 CPU→GPU ops
- **Quality**: 220 binaries, 753 lib tests, 0 clippy (pedantic+nursery), 0 unsafe, 3590+ total checks

---

## Part 1: Nautilus Shell Cross-Spring Bridge

### What It Is

hotSpring's brain architecture uses a 4-layer multi-substrate system (NPU/GPU/GPU/CPU) with an 11-head ESN for within-run temporal dynamics. The **Nautilus Shell** (from `primalTools/bingoCube/`) is the cross-run structural learning layer — an evolutionary reservoir that replaces temporal recurrence with board-population evolution.

neuralSpring now bridges this via `SpectralNautilusBridge`:

```text
neuralSpring spectral features → BetaObservation mapping → NautilusBrain
  disorder → beta
  level_spacing_ratio → anderson_r
  lambda_min → anderson_lambda_min
  bandwidth → plaquette
  ipr → cg_iters (scaled)
```

### Integration Surface

| API | Purpose |
|-----|---------|
| `observe_spectral(w, lsr, λ_min, bw, ipr)` | Feed spectral features |
| `train()` → `Option<f64>` | Evolve shell on observations |
| `predict(disorder)` → `(ipr, bw, lsr)` | Predict spectral properties |
| `screen_candidates(&[f64])` | Rank disorders by information value |
| `detect_concept_edges()` | LOO-based phase transition finder |
| `is_drifting()` | N_e*s training stability |
| `to_json()` / `from_json()` | Cross-run shell transfer |

### Validation Results

| Check | Result |
|-------|--------|
| Bridge lifecycle | PASS |
| Spectral regime detection | PASS — localized IPR > extended IPR |
| ESN vs Nautilus comparison | PASS |
| Serialization roundtrip | PASS — 1e-10 parity |
| Drift monitoring | PASS |
| Concept edge detection | PASS — 11 edges detected |

### BarraCUDA Relevance

The Nautilus Shell is pure CPU (board evaluation + ridge regression). However:
- The **input features** come from BarraCUDA spectral ops (Anderson, Lanczos, level statistics)
- **GPU acceleration** of board response evaluation is a future ToadStool opportunity
- The `DriftMonitor` pattern (N_e*s) could be generalized for any evolutionary GPU workload

---

## Part 2: Upstream Shader Bugs (Still Open from V68)

### Bug 1: `bitcast<f64>` in DF64 Pipeline

**File**: `crates/barracuda/src/shaders/stats/jackknife_mean_f64.wgsl` line 23

```wgsl
let full_sum = bitcast<f64>(vec2<u32>(params.full_sum_lo, params.full_sum_hi));
```

When `ComputeDispatch::f64()` transforms for DF64 emulation, `bitcast<f64>` produces a type incompatible with array subscript results.

**Fix**: Pass `full_sum` via storage buffer, or add `bitcast<f64>` support to the DF64 transform pipeline.

### Bug 2: `enable f64;` Directive

**File**: `crates/barracuda/src/shaders/science/hargreaves_batch_f64.wgsl` line 7

```wgsl
enable f64;  // naga: "expected global item, found 'enable'"
```

**Fix**: Remove the line. `array<f64>` and `f64()` work without it (proven by `kimura_fixation_f64.wgsl`).

---

## Part 3: Comprehensive BarraCUDA Usage Inventory (S102)

### Import Footprint

| Category | Count |
|----------|-------|
| Total `use barracuda::` sites | 198 |
| Total `use bingocube_nautilus::` sites | 2 |
| Files importing barracuda | 208+ |
| Distinct stats functions used | 58+ |
| Submodules exercised | 20+ |

### Stats Module Usage (58 functions)

**Core stats**: `mean`, `variance`, `covariance`, `correlation`, `pearson_correlation`, `spearman_correlation`, `rmse`, `mae`, `r_squared`, `nash_sutcliffe`, `l2_norm`, `dot`, `norm_pdf`, `norm_cdf`, `norm_ppf`

**Fitting**: `fit_linear`, `fit_quadratic`, `fit_exponential`, `fit_all`, `linear_regression_gpu`, `matrix_correlation_gpu`

**Bootstrap/resampling**: `bootstrap_mean`, `bootstrap_ci`, `rawr_mean`, `jackknife_mean_variance`, `jackknife`

**Diversity**: `shannon`, `shannon_from_frequencies`, `simpson`, `hill`, `alpha_diversity`, `bray_curtis`, `chao1`, `chao1_classic`

**Spectral**: `empirical_spectral_density`, `marchenko_pastur_bounds`, `chi2_decomposed`

**Evolution**: `kimura_fixation_prob`, `error_threshold`, `detection_power`, `detection_threshold`

**Hydrology**: `hargreaves_et0`, `fao56_et0`, `crop_coefficient`, `soil_water_balance`

**Histogram**: `HistogramGpu::dispatch`

**GPU dispatch**: `KimuraGpu::dispatch`, `JackknifeMeanGpu::dispatch` (blocked), `HargreavesBatchGpu::dispatch` (blocked)

### Other Submodules Used

| Submodule | Usage |
|-----------|-------|
| `barracuda::device` | `WgpuDevice`, `ComputeDispatch` |
| `barracuda::tensor` | `Tensor::from_data`, matmul, add, tanh for GPU ESN |
| `barracuda::ops::bio` | `BatchFitnessGpu`, `MultiObjFitnessGpu`, `PairwiseL2Gpu`, `PairwiseHammingGpu`, `PairwiseJaccardGpu`, `LocusVarianceGpu`, `SpatialPayoffGpu`, `BatchIprGpu` |
| `barracuda::ops::linalg` | eigendecomposition, QR |
| `barracuda::ops::fft` | FFT1D, IFFT |
| `barracuda::ops::mha` | `MultiHeadAttention` |
| `barracuda::spectral` | Lanczos, Anderson, level spacing ratio |
| `barracuda::dispatch` | `Dispatcher`, `domain_ops` |
| `barracuda::special` | `gamma`, `erf`, `erfc` |
| `barracuda::nn` | `SimpleMlp` |
| `barracuda::staging` | GPU staging buffers |
| `barracuda::pipeline` | Compute pipeline management |

### What neuralSpring Contributed Upstream

| Contribution | Status |
|-------------|--------|
| `BatchFitnessGpu` | Absorbed into `barracuda::ops::bio` |
| `MultiObjFitnessGpu` | Absorbed |
| `PairwiseL2Gpu`, `PairwiseHammingGpu`, `PairwiseJaccardGpu` | Absorbed |
| `LocusVarianceGpu` | Absorbed |
| `SpatialPayoffGpu` | Absorbed |
| `BatchIprGpu` | Absorbed |
| `empirical_spectral_density` | Absorbed into `barracuda::stats` |
| `marchenko_pastur_bounds` | Absorbed |
| `WeightSpectralResult` extensions | `bandwidth`, `condition_number`, `phase` from hotSpring |
| GPU ESN via Tensor ops | Direct Tensor matmul/tanh pattern |

---

## Part 4: Absorption Opportunities for ToadStool

### Priority 1: Fix Shader Bugs (from V68)

- `jackknife_mean_f64.wgsl`: `bitcast<f64>` DF64 compatibility
- `hargreaves_batch_f64.wgsl`: Remove `enable f64;` line

### Priority 2: Nautilus Shell GPU Acceleration

The `bingocube-nautilus` crate is CPU-only. Board response evaluation (`input → board → response vector`) is embarrassingly parallel — each of 24 boards processes the same input independently. A WGSL shader for batched board response would accelerate training.

### Priority 3: SpectralNautilusBridge as ToadStool Pattern

The bridge pattern (spectral features → evolutionary reservoir → predictions) could be generalized:
- `barracuda::nautilus` module with GPU-accelerated board populations
- `DriftMonitor` as a general evolutionary population health check
- `ConceptEdgeDetector` for automated phase transition discovery

### Priority 4: Continue ComputeDispatch Migration

66/250 ops migrated. neuralSpring's 47 GPU dispatch ops all go through `barracuda::dispatch::Dispatcher` — any ComputeDispatch improvements benefit us automatically.

---

## Part 5: Quality Metrics

| Metric | Value |
|--------|-------|
| Lib tests | **753** (+7 nautilus bridge) |
| Integration tests | 9 |
| Forge tests | 43 |
| Validation binaries | **220** (+2: S71 GPU stats, nautilus bridge) |
| validate_all | 200/200 |
| Clippy warnings | **0** (pedantic + nursery) |
| Unsafe code | **0** |
| Bare unwrap | **0** in library |
| Files > 1000 LOC | **0** |
| SPDX headers | **100%** |
| Named tolerances | 139+ |
| External deps | 10 (9 pure Rust + bingocube-nautilus) |
| ToadStool pin | **`8dc01a37`** |

---

## Part 6: Cross-Spring Evolution Map

```text
hotSpring (brain arch)     → bingoCube Nautilus Shell → neuralSpring SpectralNautilusBridge
hotSpring (proxy.rs)       → bandwidth, condition_number, phase → neuralSpring WeightSpectralResult
hotSpring (esn_v2)         → GPU ESN via Tensor ops → neuralSpring wdm_esn
hotSpring (df64)           → f64 precision shaders → neuralSpring coralForge
wetSpring (DiversityFusion)→ Shannon+Simpson+Pielou → neuralSpring bench_cross_spring
wetSpring (HMM f64)        → Forward algorithm → neuralSpring validate_gpu_hmm_forward
neuralSpring (spectral)    → ESD, level_spacing, MP bounds → barracuda::stats, barracuda::spectral
neuralSpring (batch fitness)→ GPU batch fitness eval → barracuda::ops::bio
neuralSpring (pairwise ops)→ L2, Jaccard, Hamming → barracuda::ops::bio
ToadStool (ComputeDispatch)→ Pure math shaders → all Springs benefit
ToadStool (DF64 transcend) → gamma, erf, trig on f32 GPUs → all Springs
```

---

*neuralSpring Sessions 101–102 — ToadStool S71 GPU stats parity + Nautilus Shell cross-spring bridge.*
*V69 supersedes V68. Archive V68.*
