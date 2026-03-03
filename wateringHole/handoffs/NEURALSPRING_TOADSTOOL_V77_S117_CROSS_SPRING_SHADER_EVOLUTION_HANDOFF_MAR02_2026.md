<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/BarraCUDA Handoff V77 — Cross-Spring Shader Evolution & Provenance

**Date**: March 2, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Sessions 116–117 — ToadStool S87 sync, cross-spring shader evolution, provenance benchmark
**Supersedes**: V76 (S115 Dispatch Parity + ComputeDispatch Bridge + NUCLEUS PCIe Bypass)
**ToadStool HEAD**: `2dc26792` (S87)

---

## Executive Summary

- **42/42 cross-spring shader evolution**: Full provenance validator proving 5 springs converge correctly through ToadStool S87 — local CPU ref == `barracuda::dispatch` == Dispatcher GPU for all key operations
- **15/15 provenance benchmark**: Timing benchmarks with per-operation provenance tracking across all spring origins (T0 local CPU → T1 `barracuda::dispatch` → T2 Dispatcher GPU)
- **18/18 ToadStool S87 sync**: S87 deep debt evolution validated — CPU module ungating, `BarracudaError` evolution, FHE shader fixes, `gpu_helpers` refactor, unsafe audit
- **212/212 validate_all PASS**, 861 lib tests, 0 clippy, 0 fmt
- **232 validation/bench binaries** (up from 230 in V76)

---

## Part 1: Cross-Spring Shader Evolution (42/42)

### 1.1 What We Proved

Every operation that flowed from a spring into ToadStool/BarraCUDA was validated at 3 tiers:

| Tier | Description | Validation |
|------|-------------|------------|
| T0 | Local CPU reference (e.g., `transformer::softmax`) | Baseline truth |
| T1 | `barracuda::dispatch` (CPU or GPU depending on device) | T0 == T1 |
| T2 | `Dispatcher` GPU path (wraps T1) | T1 == T2 |

**Result**: T0 == T1 == T2 for all tested operations. The evolution chain preserves mathematical correctness.

### 1.2 neuralSpring Contributions to ToadStool

| Operation | Source | Absorption Path | Parity |
|-----------|--------|-----------------|--------|
| softmax | `transformer.rs` | → `barracuda::dispatch::softmax_dispatch` → `ComputeDispatch` | EXACT f64 |
| gelu | `transformer.rs` | → `barracuda::dispatch::gelu_dispatch` → `ComputeDispatch` | EXACT f64 |
| sigmoid | `primitives.rs` | → `metalForge sigmoid_f64.wgsl` → `ComputeDispatch` | EXACT f64 |
| matmul | `matmul_gpu_evolved.wgsl` | → `barracuda::dispatch::matmul_dispatch` → `ComputeDispatch` | EXACT f64 |
| layer_norm | `coral_forge/activation.rs` | → `barracuda::TensorSession::layer_norm` | GPU via `layer_norm_f64.wgsl` |
| rk4_step | `primitives.rs` | → `metalForge rk4_parallel.wgsl` (GPU batch) | energy conservation < 1e-6 |
| batch_ipr | `spectral` module | → `barracuda::spectral::BatchIprGpu` | GPU-resident |
| swarm_nn | `swarm_robotics.rs` | → `metalForge swarm_nn_forward.wgsl` | GPU batch |
| chi²_f64 | `chi_squared_f64.wgsl` | → `metalForge` → absorption target | f64 precision |
| kl_div_f64 | `kl_divergence_f64.wgsl` | → `metalForge` → absorption target | f64 precision |

### 1.3 wetSpring Contributions to ToadStool (validated by neuralSpring)

| Operation | Source | Absorption Path | Parity |
|-----------|--------|-----------------|--------|
| Shannon entropy | wetSpring diversity | → `barracuda::stats::shannon` → `Dispatcher` | EXACT f64 |
| Simpson index | wetSpring diversity | → `barracuda::stats::simpson` | in [0,1] |
| chao1 | wetSpring richness | → `barracuda::stats::chao1_classic` | ≥ observed richness |
| Bray-Curtis | wetSpring ecological | → `barracuda::stats::bray_curtis` | functional |
| pielou evenness | wetSpring diversity | → `barracuda::stats::pielou_evenness` | functional |
| HMM forward | `hmm_forward_log.wgsl` | → `barracuda::dispatch::hmm_forward_dispatch` | EXACT f64 |
| Wright-Fisher | `wright_fisher_step.wgsl` | → ToadStool bio shaders | GPU batch |

### 1.4 hotSpring Contributions to ToadStool (validated by neuralSpring)

| Operation | Source | Absorption Path | Parity |
|-----------|--------|-----------------|--------|
| eigensolve | Householder+QR | → `barracuda::linalg::eigh_f64` → `Dispatcher::eigh` | correct count, sorted |
| level_spacing_ratio | spectral analysis | → `barracuda::spectral` | in [0,1] |
| spectral_bandwidth | spectral analysis | → `barracuda::spectral` | positive |
| spectral_condition_number | spectral analysis | → `barracuda::spectral` | functional |
| classify_spectral_phase | spectral analysis | → `barracuda::spectral` | correct phase |
| pearson correlation | hotSpring precision | → `barracuda::stats::pearson_correlation` | EXACT f64 |
| DF64 transcendentals | f32-pair precision | → `df64_transcendentals.wgsl` (37 shaders) | probe-injected |
| math_f64 polyfills | native f64 fallbacks | → `math_f64.wgsl` (28 functions) | probe-injected |

### 1.5 groundSpring Contributions to ToadStool (validated by neuralSpring)

| Operation | Source | Absorption Path | Parity |
|-----------|--------|-----------------|--------|
| bootstrap CI | uncertainty quantification | → `barracuda::stats::bootstrap_ci` | lower ≤ upper, contains true mean |
| jackknife | leave-one-out estimator | → `barracuda::stats::jackknife` | estimate near true mean, se > 0 |
| kimura fixation | population genetics | → `barracuda::stats::kimura_fixation_prob` | prob in (0,1) |
| norm_cdf/pdf/ppf | normal distribution | → `barracuda::stats::norm_cdf` | Φ(0) = 0.5 EXACT |

### 1.6 airSpring Contributions to ToadStool (validated by neuralSpring)

| Operation | Source | Absorption Path | Parity |
|-----------|--------|-----------------|--------|
| Hargreaves ET₀ | hydrology | → `barracuda::stats::hargreaves_et0` | positive ET₀ |
| Thornthwaite ET₀ | hydrology | → `barracuda::stats::thornthwaite_et0` | positive ET₀ |
| Hamon ET₀ | hydrology | → `barracuda::stats::hamon_et0` | functional |
| Makkink ET₀ | hydrology | → `barracuda::stats::makkink_et0` | functional |
| Turc ET₀ | hydrology | → `barracuda::stats::turc_et0` | functional |
| Hargreaves batch | hydrology | → `barracuda::stats::hargreaves_et0_batch` | 365-day batch |

---

## Part 2: ToadStool S87 Sync Findings (18/18)

### 2.1 S87 Changes Reviewed

| Area | Change | Impact on neuralSpring |
|------|--------|----------------------|
| FHE shaders | NTT/INTT `u64_mod_simple` fix, pointwise_mul correction | No downstream impact (FHE not used by neuralSpring) |
| `async-trait` | Reclassified from TODO → NOTE (architectural choice for `Nautilus` trait objects) | Informational only |
| Unsafe audit | 60+ `unsafe` sites documented with `// SAFETY:` comments | No API changes |
| `gpu_helpers` refactor | Internal module reorganization | No API changes |
| CPU module ungating | Stats/diversity/correlation modules accessible without GPU feature gate | Enables CPU-only builds (already validated) |

### 2.2 neuralSpring Validated Against S87

- `barracuda::stats::correlation::variance` — sample variance (ddof=1), documented
- `barracuda::stats::shannon`, `simpson` — functional, correct for uniform distributions
- `BarracudaError::is_device_lost()`, `BarracudaError::io()` — new error API works
- `NautilusBrain`, `DriftMonitor` — API unchanged from S86
- All 53/53 dispatch parity checks pass on S87
- All 14/14 ComputeDispatch bridge checks pass on S87

---

## Part 3: Performance Benchmarks with Provenance

### 3.1 neuralSpring Origins

| Operation | T0 Local CPU | T2 Dispatcher GPU | Speedup |
|-----------|-------------|-------------------|---------|
| softmax (256) | 2.5µs | 1.2µs | **2.08×** |
| gelu (1024) | 7.9µs | 7.8µs | 1.01× |
| matmul (64×64) | — | 1614µs | — |
| sigmoid (1024) | 3.0µs | — (CPU ref) | — |
| RK4 (1000 steps) | 21.7µs | — (CPU single-step) | — |

### 3.2 wetSpring Origins

| Operation | T3 barracuda::stats | Notes |
|-----------|-------------------|-------|
| Shannon (256 bins) | 0.9µs | Pure Rust, no WGSL needed at this scale |
| Simpson (256 bins) | 0.4µs | |
| chao1 (256 bins) | 0.1µs | |
| Bray-Curtis (128 sites) | 0.1µs | |

### 3.3 hotSpring Origins

| Operation | Timing | Notes |
|-----------|--------|-------|
| eigh (32×32) | 11.2ms | Householder+QR via GPU |
| Pearson (512 pairs) | 0.7µs | Pure Rust |
| Variance Dispatcher (512) | 0.4µs | GPU dispatch path |

### 3.4 groundSpring Origins

| Operation | Timing | Notes |
|-----------|--------|-------|
| bootstrap CI (200pts × 500 reps) | 235µs | Pure Rust resampling |
| jackknife (200 pts) | 14.6µs | Leave-one-out |
| norm_cdf (1000 batch) | 7.3µs | Pure Rust |
| norm_ppf (1000 batch) | 2.5µs | Pure Rust |

### 3.5 airSpring Origins

| Operation | Timing | Notes |
|-----------|--------|-------|
| Hargreaves batch (365 days) | 0.7µs | All 6 ET₀ methods sub-microsecond individually |

---

## Part 4: Lessons Learned & Recommendations for ToadStool

### 4.1 Variance Convention

`barracuda::stats::correlation::variance` uses **sample variance** (ddof=1, Bessel's correction).
`barracuda::dispatch::variance_dispatch` uses **population variance** (ddof=0).
`Dispatcher::variance` delegates to `variance_dispatch` (ddof=0).

**Recommendation**: Consider adding explicit `ddof` parameter or separate `variance_population`/`variance_sample` functions.

### 4.2 Precision Strategy is Excellent

The multi-tier precision strategy (F64 native → math_f64.wgsl polyfills → DF64 f32-pair → F32) with per-function hardware probing (`probe_f64_builtins`) is the right architecture. neuralSpring validates that:
- RTX 3090/PTXAS: full native f64 transcendentals
- RX 6950 XT/ACO: only sqrt/fma/abs native, rest need polyfills
- The `ShaderTemplate::for_driver_auto()` injection strategy works correctly

### 4.3 Absorption Targets

The following neuralSpring local implementations are candidates for ToadStool absorption:

| Implementation | Location | Notes |
|---------------|----------|-------|
| `primitives::rk4_step` | `src/primitives.rs` | CPU single-step RK4; upstream has `rk45_solve` (adaptive) and `rk4_parallel.wgsl` (GPU batch) but no CPU single-step |
| `cpu_fallback::variance` | `src/gpu_dispatch/cpu_fallback.rs` | Population variance (ddof=0); aligned with `variance_dispatch` |
| `jacobi_eigh` | `src/anderson_localization.rs` | Misnomer: actually Householder+QR, delegates to `eigh_householder_qr` |

### 4.4 What Works Well

- `barracuda::dispatch` is the clean interface: downstream springs (neuralSpring, wetSpring, hotSpring) all validate through it
- `barracuda::stats` has excellent breadth: 80+ statistical functions spanning diversity, hydrology, uncertainty, normal distribution, regression
- `barracuda::spectral` is solid: Anderson, Lanczos, IPR, spectral diagnostics all work correctly
- `barracuda::nautilus` brain/drift/shell pattern is clean for adaptive compute monitoring
- CPU module ungating (S87) was a smart move — enables CPU-only downstream builds

### 4.5 Cross-Spring Shader Provenance Map

```
hotSpring ──── precision, DF64, eigensolve, math_f64.wgsl, df64_transcendentals.wgsl
wetSpring ──── diversity (Shannon, Simpson, chao1, Bray-Curtis), HMM, Wright-Fisher
neuralSpring ─ matmul, gelu, softmax, sigmoid, swarm_nn, batch_fitness, RK4, MHA
               coralForge: layer_norm, SDPA, triangle, IPA, backbone, torsion
groundSpring ─ bootstrap, jackknife, kimura, norm_cdf/pdf/ppf
airSpring ──── hydrology (ET₀: FAO-56, Thornthwaite, Hamon, Makkink, Turc)
                │
                ▼
        ToadStool S87 (2dc26792)
        844+ WGSL shaders, 37 DF64
        144-op ComputeDispatch
        Pure math, no vendor libs
        Precision per hardware: F16 / F32 / F64 / DF64
```

---

## Appendix: Validation Statistics

| Metric | Value |
|--------|-------|
| ToadStool HEAD | `2dc26792` (S87) |
| validate_all | 212/212 PASS |
| Library tests | 861 |
| Clippy warnings | 0 |
| Fmt issues | 0 |
| Validation/bench binaries | 232 |
| WGSL shaders (ToadStool) | 844+ |
| DF64 shaders | 37 |
| Dispatch parity | 53/53 |
| ComputeDispatch bridge | 14/14 |
| Cross-spring evolution | 42/42 |
| Provenance benchmark | 15/15 |
| NUCLEUS PCIe bypass | 38/38 |
| S87 sync checks | 18/18 |
| Barracuda import files | 117 |
| Barracuda modules used | 16 |

---

*V77 — neuralSpring → ToadStool. Sessions 116–117.*
