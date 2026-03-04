<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/barraCuda Handoff V78 — Standalone barraCuda Rewire & Revalidation

**Date**: March 3, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/barraCuda team
**License**: AGPL-3.0-or-later
**Covers**: Session 118 — barraCuda standalone extraction rewire, revalidation, S93 API validation
**Supersedes**: V77 (S117 Cross-Spring Shader Evolution & Provenance)
**barraCuda**: v0.3.1 standalone (`../barraCuda/crates/barracuda`)
**Previous pin**: ToadStool S87 (`2dc26792`) via `../phase1/toadstool/crates/barracuda`

---

## Executive Summary

- **Path swap completed**: `Cargo.toml` (root + metalForge/forge) rewired from `../phase1/toadstool/crates/barracuda` to `../barraCuda/crates/barracuda`
- **CI rewired**: 7 checkout blocks in `.github/workflows/rust.yml` updated to reference `barraCuda` standalone repo
- **Zero breakage**: `cargo check` (all targets, all features), `cargo clippy` (pedantic + nursery), `cargo fmt`, `cargo doc` — all clean
- **Full revalidation**: 861/861 lib tests, 9/9 integration, key validators all green:
  - `validate_toadstool_s86_rewire`: 27/27 PASS
  - `validate_cross_spring_rewire`: 41/41 PASS
  - `validate_cross_spring_evolution`: 52/52 PASS
  - `validate_barracuda_stats`: 13/13 PASS
  - `validate_barracuda_linalg`: 17/17 PASS
  - `validate_barracuda_tensor`: 86/86 PASS
  - `validate_nautilus_bridge`: 27/27 PASS
- **New validator**: `validate_toadstool_s93_barracuda_extraction` — **29/29 PASS**
  - S88+ APIs: `tridiag_eigenvectors`, domain tolerance constants (HYDRO_*, PHYSICS_ANDERSON_EIGENVALUE, BIO_DIVERSITY_*)
  - Unified math vocabulary: `MathOp` enum, `ComputeExecutor` trait
  - Precision routing: `Fp64Strategy` (Native/Hybrid/Concurrent)
  - Nautilus continuity: DriftMonitor, NautilusBrain on standalone path
  - Dispatcher continuity: mat_mul, softmax, shannon_entropy on standalone path
- **Docs updated**: EVOLUTION_READINESS.md, specs/BARRACUDA_USAGE.md, specs/BARRACUDA_REQUIREMENTS.md, README.md
- **L-BFGS gap closed**: `barracuda::optimize::LbfgsGpu` now available (was P2 OPEN)

---

## Part 1: What Changed

### 1.1 Cargo.toml Path Swap

```toml
# Before (ToadStool embedded)
barracuda = { path = "../phase1/toadstool/crates/barracuda", features = ["unidirectional"] }

# After (barraCuda standalone)
barracuda = { path = "../barraCuda/crates/barracuda", features = ["unidirectional"] }
```

Both root `Cargo.toml` and `metalForge/forge/Cargo.toml` updated. The `unidirectional` feature is retained — barraCuda v0.3.1 still supports it.

Default features (`gpu`, `domain-models`) are enabled, giving access to all GPU-gated APIs.

### 1.2 CI Workflow Rewire

`.github/workflows/rust.yml` — 7 checkout blocks across all jobs (test, coverage, validate-native, validate-barracuda, validate-barracuda-cpu, cross-validate, benchmarks):

```yaml
# Before
- name: Checkout ecoPrimals (barracuda dependency)
  uses: actions/checkout@v4
  with:
    repository: ${{ github.repository_owner }}/ecoPrimals
    path: ../phase1/toadstool
    sparse-checkout: crates/barracuda

# After
- name: Checkout barraCuda (standalone math primal)
  uses: actions/checkout@v4
  with:
    repository: ${{ github.repository_owner }}/barraCuda
    path: ../barraCuda
    sparse-checkout: crates/barracuda
```

### 1.3 Version Bump

`cargo check` confirmed: `barracuda v0.2.0` → `barracuda v0.3.1`. The version bump was automatic from the path change — barraCuda standalone ships as v0.3.1.

---

## Part 2: New S88–S93 APIs Validated

### 2.1 `barracuda::spectral::tridiag_eigenvectors` (S88)

Tridiagonal eigensolver — validated with known 3×3 tridiagonal matrix:
- Eigenvalues: 2−√2, 2, 2+√2 (all within CROSS_LANGUAGE tolerance)
- Eigenvector count: 9 elements (3×3 matrix)
- Edge cases: empty input → empty output, single element → identity

### 2.2 Domain Tolerance Constants (S88+)

7 new tolerance constants validated accessible with correct abs_tol values:

| Constant | abs_tol | Domain |
|----------|---------|--------|
| `HYDRO_ET0` | 0.05 | Evapotranspiration |
| `HYDRO_SOIL_MOISTURE` | 1e-4 | Soil water |
| `HYDRO_WATER_BALANCE` | 0.1 | Water balance |
| `HYDRO_CROP_COEFFICIENT` | 1e-6 | Crop Kc |
| `PHYSICS_ANDERSON_EIGENVALUE` | 1e-10 | Localization |
| `BIO_DIVERSITY_SHANNON` | 1e-8 | Ecology |
| `BIO_DIVERSITY_SIMPSON` | 1e-10 | Ecology |

### 2.3 `unified_math::MathOp` Vocabulary

Canonical operation enum validated: Negate, Abs, Exp, Sqrt, Square, Reciprocal (unit variants), MatMul{transpose_a, transpose_b} (struct variant), Softmax{dim} (struct variant), ReLU, Add.

### 2.4 `Fp64Strategy` Precision Routing

Three-variant enum for per-hardware f64 execution strategy:
- `Native` — full-rate f64 (Titan V, A100)
- `Hybrid` — DF64 bulk + native reductions (RTX 4070)
- `Concurrent` — dual-path cross-validation

### 2.5 `ComputeExecutor` Trait

Object-safe trait for multi-backend dispatch. Methods: `name()`, `hardware_type()`, `capabilities()`, `can_execute()`, `score_operation()`, `execute()`, `allocate()`, `transfer()`.

### 2.6 L-BFGS Optimizer

`barracuda::optimize::LbfgsGpu` now available (previously P2 OPEN gap). Requires `gpu` feature. Not yet wired into neuralSpring — available for future PINN optimization.

---

## Part 3: Breaking Change Assessment

**Zero breaking changes.** All APIs used by neuralSpring are preserved in barraCuda v0.3.1:

| API | neuralSpring Usage | Status |
|-----|-------------------|--------|
| `WgpuDevice::new()` | `Gpu::new()` | Unchanged |
| `adapter_info()` | `gpu.rs:116` | Unchanged |
| `barracuda::nautilus::*` | 6+ files | Unchanged |
| `barracuda::dispatch::*` | 50+ files | Unchanged |
| `barracuda::tensor::Tensor` | 30+ files | Unchanged |
| `barracuda::ops::bio::*` | 20+ files | Unchanged |
| `MatmulResult` | Not used | N/A |
| `discover_devices()` | Not used (removed in v0.3.1) | N/A |

MSRV bumped 1.80 → 1.87. neuralSpring toolchain confirmed compatible (1.93 stable).

---

## Part 4: Revalidation Evidence

### 4.1 Compile Gates

| Gate | Result |
|------|--------|
| `cargo check --all-targets --all-features` | PASS (barracuda v0.3.1 resolved) |
| `cargo clippy --all-targets --all-features -- -W clippy::pedantic -W clippy::nursery` | 0 warnings |
| `cargo fmt --check` | clean |
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` | 0 warnings |

### 4.2 Test Suite

| Suite | Result |
|-------|--------|
| `cargo test --lib` | 861/861 PASS |
| `cargo test --test integration` | 9/9 PASS |

### 4.3 Key Validators

| Validator | Checks | Result |
|-----------|--------|--------|
| `validate_toadstool_s86_rewire` | 27 | PASS |
| `validate_cross_spring_rewire` | 41 | PASS |
| `validate_cross_spring_evolution` | 52 | PASS |
| `validate_barracuda_stats` | 13 | PASS |
| `validate_barracuda_linalg` | 17 | PASS |
| `validate_barracuda_tensor` | 86 | PASS |
| `validate_nautilus_bridge` | 27 | PASS |
| **`validate_toadstool_s93_barracuda_extraction`** | **29** | **PASS** |

---

## Part 5: Files Modified

| File | Change |
|------|--------|
| `Cargo.toml` | barracuda path: `../phase1/toadstool/crates/barracuda` → `../barraCuda/crates/barracuda` |
| `metalForge/forge/Cargo.toml` | barracuda path: `../../../phase1/toadstool/crates/barracuda` → `../../../barraCuda/crates/barracuda` |
| `.github/workflows/rust.yml` | 7 checkout blocks: repo → `barraCuda`, path → `../barraCuda` |
| `src/bin/validate_toadstool_s93_barracuda_extraction.rs` | New: 29-check S93 rewire validator |
| `EVOLUTION_READINESS.md` | barraCuda v0.3.1 standalone reference |
| `specs/BARRACUDA_USAGE.md` | Version 0.2.0 → 0.3.1, path updated |
| `specs/BARRACUDA_REQUIREMENTS.md` | L-BFGS gap → AVAILABLE |
| `README.md` | barraCuda standalone reference |

---

## Part 6: For the barraCuda Team

### 6.1 Confirmed Working APIs

neuralSpring exercises the following barraCuda v0.3.1 API surface without issue:

- **Device**: `WgpuDevice::new()`, `adapter_info()`, `GpuDriverProfile`, `Fp64Strategy`
- **Tensor**: `from_data`, `to_vec`, `matmul`, `transpose`, `softmax`, `sigmoid`, `tanh`, `gelu_wgsl`, `layer_norm_wgsl`, `log_softmax_wgsl`
- **ops::bio**: 18+ GPU kernels (BatchFitness, HMM, Pairwise*, Spatial, Hill, Swarm, etc.)
- **dispatch**: 9 domain dispatch functions (matmul, frobenius, variance, softmax, etc.)
- **stats**: variance, pearson, shannon, diversity
- **linalg**: solve_f64, eigh_f64, cholesky, graph_laplacian, effective_rank
- **spectral**: tridiag_eigenvectors, BatchIprGpu, Anderson, Lanczos
- **nautilus**: NautilusBrain, DriftMonitor, NautilusShell, BetaObservation
- **tolerances**: All 7 new domain constants + existing constants

### 6.2 New APIs Not Yet Consumed

These are available but not yet wired into neuralSpring:

| API | Potential Use | Priority |
|-----|--------------|----------|
| `LbfgsGpu` | PINN optimization | Medium |
| `SeasonalGpuParams` | Hydrology (airSpring domain) | Low |
| `ComputeExecutor` implementations | Multi-backend dispatch | Future |
| `unified_math::MathOp` dispatch | Op-level routing | Future |

### 6.3 hotSpring Precedent

hotSpring completed this same rewire successfully (716/716 tests pass, single-line path change). neuralSpring confirms the pattern holds for a larger consumer (861 lib + 232 binaries).

---

*neuralSpring V78 handoff — barraCuda standalone extraction rewire. March 3, 2026. 29/29 S93 PASS, 861/861 lib, 9/9 integration, zero breakage.*
