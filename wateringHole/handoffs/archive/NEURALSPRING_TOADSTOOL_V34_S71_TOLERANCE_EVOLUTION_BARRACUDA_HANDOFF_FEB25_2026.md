# neuralSpring → ToadStool/BarraCUDA Handoff V34

**Session 71 — Tolerance Evolution, Smart Refactoring, BarraCUDA Absorption Recommendations**
**Date**: February 25, 2026
**From**: neuralSpring
**To**: ToadStool / BarraCUDA core team
**License**: AGPL-3.0-or-later
**ToadStool HEAD**: `02207c4a`
**Supersedes**: V33 (Session 70 — Deep Audit II, Coverage Evolution)

---

## Executive Summary

Session 71 completed the deepest code quality execution to date — eliminating
every ad-hoc numeric tolerance from test assertions across the entire library
codebase. This handoff delivers actionable patterns for the ToadStool/BarraCUDA
team and documents the full barracuda usage evolution.

1. **150+ ad-hoc tolerances → named constants** across 21 library test files.
   Every `assert!(...abs() < 1e-12)` replaced with `tolerances::EXACT_F64`.
   Zero bare numeric tolerance values remain in test assertions.

2. **Smart refactoring**: `gpu_dispatch/mod.rs` reduced from 862→304 lines by
   extracting CPU tests to `tests_cpu.rs`, following the existing `tests_gpu.rs`
   pattern. Not an arbitrary split — semantic: production, CPU tests, GPU tests.

3. **Dependency audit**: All crates verified Pure Rust (ecoBin compliant, zero
   C dependencies). No external deps to evolve.

4. **Coverage ceiling confirmed**: 94.53% is architectural maximum. Below-90%
   files are exclusively GPU error-handling paths and `process::exit()`.

5. **BarraCUDA usage audit**: 90+ import sites, 20+ submodules, zero duplicate
   math. Complete absorption map with recommendations.

---

## Part 1: What Changed (Session 71)

### Tolerance Standardization

Every library module test file now uses named constants from `crate::tolerances`:

| Tolerance Value | Named Constant | Semantic Meaning | Test Files Using |
|----------------|---------------|-----------------|-----------------|
| `1e-15` | `ZERO_DETECTION` | Near-machine-epsilon | 8 files |
| `1e-14` | `ZERO_DETECTION` | Zero-detection threshold | 12 files |
| `1e-12` | `EXACT_F64` | Exact f64 arithmetic | 18 files |
| `1e-10` | `CROSS_LANGUAGE` | Rust↔Python agreement | 14 files |
| `1e-8` | `HMM_POSTERIOR_SUM` | HMM row-sum tolerance | 3 files |
| `1e-6` | `SPECIAL_FUNCTION_F64` | Transcendental functions | 5 files |
| `1e-5` | `HESSIAN_FD_STEP` | FD step size | 1 file |
| `1e-4` | `OPTIMIZER_VALUE_AT_MIN` | Optimizer convergence | 1 file |
| `0.01` | `PINN_BC_TOLERANCE`, `NORM_PPF_TAIL` | Domain-specific | 3 files |

**Files modified** (21 library test modules + 1 new):

`loss_landscape`, `weight_spectral`, `lenet`, `neural_pgm`,
`spectral_commutativity`, `property_tests`, `sequence`, `agent_coordination`,
`information_flow`, `deeponet`, `pinn`, `meta_population`, `primitives`,
`sate_alignment`, `fft`, `eigh`, `quantized`, `pangenome_selection`,
`surrogate`, `transformer`, `introgression`, `anderson_localization`,
`counterdiabatic`, `regulatory_network`, `metrics`, `rng`, `swarm_robotics`,
`gpu_dispatch/tests_cpu.rs` (new)

### What Remains Untouched (Correctly)

| Category | Example | Why |
|----------|---------|-----|
| Doc comments | `/// assert!((val).abs() < 1e-12)` | Changing doctests adds complexity with no benefit |
| Production guards | `if phi_sum.abs() < 1e-30` | Runtime guards, not assertion tolerances |
| Semantic thresholds | `sigmoid(-100.0) < 0.01` | Behavior bounds, not precision checks |
| `f64::EPSILON` | Determinism tests | Bitwise reproducibility — different semantic |
| Function parameters | `dt = 0.01`, `flatness = 0.1` | Experiment configuration, not tolerances |

### Smart Refactoring

| File | Before | After | Method |
|------|--------|-------|--------|
| `gpu_dispatch/mod.rs` | 862 lines | 304 lines | CPU tests → `tests_cpu.rs` |
| `gpu_dispatch/tests_cpu.rs` | (new) | 613 lines | All CPU-path Dispatcher tests |
| `gpu_dispatch/tests_gpu.rs` | 576 lines | 576 lines | Unchanged (GPU-path tests) |

---

## Part 2: BarraCUDA Usage — Full Inventory & Evolution Map

### Production Modules Using BarraCUDA

| neuralSpring Module | barracuda APIs | Purpose |
|-------------------|---------------|---------|
| `gpu_dispatch/mod.rs` | `device::WgpuDevice`, `driver_profile::GpuDriverProfile`, `unified_hardware::BandwidthTier`, `error::BarracudaError` | GPU/CPU dispatch core |
| `gpu_dispatch/dispatch_ops.rs` | `dispatch::{matmul,frobenius_norm,transpose,softmax,gelu,l2_distance,mean,variance,hmm_forward}_dispatch` | 9 ops routed to upstream |
| `gpu_dispatch/cpu_fallback.rs` | `stats::pearson_correlation`, `special::chi_squared_statistic` | CPU reference paths |
| `gpu_ops/*` (6 submodules) | `tensor::Tensor`, `device::WgpuDevice` | 38 GPU-accelerated ops |
| `evolved/mha.rs` | `ops::mha::MultiHeadAttention` | Multi-head attention (thin wrapper) |
| `eigh.rs` | `ops::linalg::eigh_householder_qr` | Eigensolver (delegates upstream) |
| `weight_spectral.rs` | `stats::{empirical_spectral_density,marchenko_pastur_bounds}`, `spectral::level_spacing_ratio` | Spectral analysis |
| `neural_pgm.rs` | `linalg::{effective_rank,belief_propagation_chain}` | PGM inference |
| `agent_coordination.rs` | `linalg::{graph_laplacian,disordered_laplacian}` | Graph operations |
| `loss_landscape.rs` | `numerical::numerical_hessian`, `sample::boltzmann_sampling` | Optimization analysis |
| `gpu.rs` | `device::WgpuDevice` | GPU initialization |

### Validation/Benchmark Modules (Test-Only)

90+ binaries consume `barracuda::tensor::Tensor`, `barracuda::device::WgpuDevice`,
and domain-specific `barracuda::ops::bio::*`, `barracuda::ops::linalg::*`,
`barracuda::staging::*`, `barracuda::pipeline::*` APIs across all 25 papers +
5 baseCamp sub-theses.

### Zero Duplicate Math Guarantee

Every mathematical operation in neuralSpring either:
1. **Delegates to upstream** (17 rewired functions + 6 shader source references)
2. **Composes upstream primitives** (e.g., FST = allele\_frequencies + variance)
3. **Is an independent CPU reference** for validation independence (`primitives.rs`)

No local reimplementation of any barracuda-available operation exists in
production code.

---

## Part 3: Dependency Sovereignty Audit

### External Crates (All Pure Rust, ecoBin Compliant)

| Crate | Purpose | C Deps | Cross-Compile |
|-------|---------|--------|--------------|
| `barracuda` (path) | Ecosystem shared compute | None | ✓ |
| `neural-spring-forge` (path) | Shader forge | None | ✓ |
| `bytemuck` 1.14 | Zero-copy Pod/Zeroable | None | ✓ |
| `serde` 1 + `serde_json` 1 | JSON baseline loading | None | ✓ |
| `tokio` 1.35 | Async runtime (wgpu) | None | ✓ |
| `wgpu` 22 | WebGPU API (Vulkan) | None | ✓ |
| `approx` 0.5 (dev) | Approximate assertions | None | ✓ |

**Conclusion**: Nothing to evolve. The dependency stack is already sovereign.
No C FFI, no system libraries, no proprietary SDKs. Cross-compilable to any
`wgpu`-supported target (Linux, macOS, Windows, Android, WebGPU).

---

## Part 4: Evolution Recommendations for ToadStool/BarraCUDA

### 4.1 Tolerance Standardization Pattern (Recommended for Absorption)

neuralSpring's tolerance infrastructure is now the most evolved in the ecosystem:

```
tolerances/
├── mod.rs      — 105+ named constants with mathematical justifications
├── gpu.rs      — GPU-specific tolerances (tensor, shader, FFT, dispatch)
└── registry.rs — Runtime introspection via tolerance_registry! macro
```

**Pattern**: Every tolerance has:
- A `pub const` with a doc comment explaining the mathematical basis
- A category (machine-precision, cross-language, training, spectral, etc.)
- Registration in the runtime registry for CLI introspection

**Recommendation**: BarraCUDA should absorb this pattern for its own test
infrastructure. The `tolerance_registry!` macro compresses registry boilerplate
from 891→257 lines while maintaining compile-time validation.

### 4.2 Smart Refactoring Over Arbitrary Splitting

When files exceed size limits, prefer **semantic extraction** over line-count
splitting:

| Approach | Example | Outcome |
|----------|---------|---------|
| Semantic extraction | CPU tests → `tests_cpu.rs` (follows `tests_gpu.rs` pattern) | Each file has one responsibility |
| Macro compression | `tolerance_registry!` (891→257 lines) | Same API, less boilerplate |
| Arbitrary split | Split at line 500 | Breaks cohesion, adds friction |

### 4.3 What neuralSpring Needs from BarraCUDA (Updated)

| Need | Priority | Status |
|------|----------|--------|
| `WGSL_MEAN_REDUCE` public constant | P1 | Blocked — no public export |
| `argmax_dim()` for Tensor | P1 | Blocked — Viterbi needs CPU argmax |
| `softmax_dim(axis)` for Tensor | P2 | Blocked — attention needs row-wise |
| `StatefulPipeline` chain API | P3 | Available — not yet leveraged for HMM chains |
| S-14/S-15 matmul hang fix | P3 | Workaround in place |
| `pow(f64,f64)` in transcendental patcher | P2 | S-17 proved fix; one-line change in `patch_exp_log_in_code` |

### 4.4 What neuralSpring Proved for the Ecosystem

| Finding | Sessions | Implication |
|---------|----------|-------------|
| 94.53% coverage is the GPU-code ceiling | S70–71 | Don't chase 100% on GPU error paths — mock `device_lost` isn't realistic |
| Named tolerances scale to 105+ | S68–71 | The registry pattern works at scale; categorized browsing is essential |
| All deps can be Pure Rust | S71 | ecoBin standard is achievable without compromise |
| Smart refactoring preserves cohesion | S71 | Semantic extraction > arbitrary splitting |
| Tolerance standardization is mechanical | S71 | Pattern: `1e-N` → `tolerances::CONSTANT_NAME`, can be automated |

---

## Part 5: Cross-Spring Learnings for ToadStool Evolution

### From hotSpring (Physics Precision)

- **df64 core-streaming**: F64 via f32-pairs on consumer GPUs (RTX 4070: 1:64 FP64:FP32)
- **GpuDriverProfile**: Hardware-adaptive strategy (Native vs Hybrid f64)
- **Welford variance**: Numerically stable single-pass — 3.49× faster
- **Pow polyfill**: `pow_f64` via `exp_f64(n * log_f64(x))` — portable across NVVM/NAK

### From wetSpring (Bio-Compute)

- **HMM f64**: 10⁹× precision improvement over f32 forward/backward
- **Fused map-reduce**: Single-pass Shannon entropy — 2.59× faster
- **log_f64 fix**: Corrected `log(0)` guard in upstream shaders
- **Ada Lovelace NVVM workaround**: f64 builtins need polyfills on RTX 4000 series

### From neuralSpring (ML Validation)

- **eigh Householder+QR**: Trillion-fold accuracy over Jacobi — absorbed verbatim
- **Tolerance registry**: 105+ constants with mathematical justifications + runtime introspection
- **Batch fitness evaluation**: 10k×32 genotypes in single dispatch
- **TensorSession patterns**: Per-op → single-encoder provides 46–78× speedup

### Cross-Spring Synthesis for ToadStool

The three Springs collectively validated:
- **650+ WGSL shaders** across physics, bio, and ML domains
- **14,200+ tests** in ToadStool (not counting Spring-side validation)
- **3 tolerance infrastructure patterns** (hotSpring df64-aware, wetSpring bio-aware, neuralSpring ML-aware)

**Recommendation**: ToadStool should unify the tolerance patterns into a single
`barracuda::tolerances` module following neuralSpring's `tolerance_registry!` pattern,
with categories for each Spring's domain (physics, bio, ML, signal, spectral).

---

## Part 6: Full Metrics (Session 71)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | **PASS** |
| `cargo clippy --all-targets -D warnings -W pedantic -W nursery` | **0 warnings** |
| `cargo test --lib` | **580/580 PASS** |
| `cargo test --test integration` | **9/9 PASS** |
| `cargo doc --no-deps` | **0 warnings** |
| Named tolerances (defined) | **105+** |
| Ad-hoc tolerances in test assertions | **0** (was 150+ before S71) |
| Library test files using `tolerances::*` | **21** (was 3 before S71) |
| Functions rewired to upstream | **17** |
| Validator shader sources rewired | **6** |
| SPDX compliance | **211/211 files** |
| Max library file size | **797 lines** (`validation.rs`) |
| Max file size (any) | **966 lines** (`validate_barracuda_tensor.rs`) |
| All dependencies Pure Rust | **✓** (ecoBin compliant) |

---

## Supersedes

- V33: Session 70 — Deep Audit II, Coverage Evolution
  (`wateringHole/handoffs/archive/`)

---

*neuralSpring → ToadStool handoff V34 — AGPL-3.0-or-later*
