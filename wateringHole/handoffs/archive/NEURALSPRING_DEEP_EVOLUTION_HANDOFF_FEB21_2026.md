# neuralSpring → ToadStool: Deep Evolution Handoff — GPU-Ready Layouts & Consolidated Math

**Date:** 2026-02-21 (evening)
**From:** neuralSpring (ML / isomorphic learning / scholarly reproduction Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Supplements:** `NEURALSPRING_TOADSTOOL_HANDOFF_FEB21_2026.md` (shader inventory)

---

## Executive Summary

neuralSpring underwent a deep structural evolution to align Rust
implementations with the hotSpring Write → Absorb → Lean pattern. Library
modules now use **flat row-major layouts** that match GPU buffer bindings,
**consolidated math primitives** that eliminate duplicated code across 8
modules, and **graceful error handling** that supports cross-backend
validation without panicking on GPU failure.

**Key deliverables:**

- **Flat row-major HMM**: `hmm.rs` transition (N×N), emission (N×M), alpha
  (T×N), posterior (T×N) — direct upload to `hmm_forward_log.wgsl` buffers
- **Flat row-major spectral**: `spectral_commutativity.rs` all matrix ops
  flat with explicit `n` — direct upload to `barracuda::ops::matmul`
- **`primitives.rs` module**: Shannon (6 variants consolidated), Hill (3),
  sigmoid (2), RK4 (2), numerical constants (`LOG_GUARD`, `HILL_EPS`,
  `DIVISION_GUARD`)
- **`require!` macro**: All validation binaries gracefully handle GPU failures
- **198 unit tests + 7 doc-tests** (was 181 + 6)

---

## Part 1: Flat Row-Major Layouts (GPU Buffer Alignment)

### HMM Module (`src/hmm.rs`)

The HMM module now stores all matrices as flat `Vec<f64>` with row-major
indexing. This matches the WGSL shader buffer layout exactly — no conversion
needed for GPU upload.

**Before (nested):**

```rust
pub struct Hmm {
    pub transition: Vec<Vec<f64>>,  // heap-per-row, scattered
    pub emission: Vec<Vec<f64>>,
}
```

**After (flat, GPU-ready):**

```rust
pub struct Hmm {
    pub transition: Vec<f64>,  // N×N row-major, contiguous
    pub emission: Vec<f64>,    // N×M row-major, contiguous
    n: usize,
    m: usize,
}
```

**GPU-native constructor:**

```rust
Hmm::from_flat(transition, emission, initial, n, m)
```

**Buffer layout match with `hmm_forward_log.wgsl`:**

| Rust Field | WGSL Binding | Layout |
|-----------|--------------|--------|
| `hmm.transition` | `@binding(1) trans: array<f32>` | N×N row-major |
| `hmm.emission` (per timestep) | `@binding(2) emiss: array<f32>` | N values |
| `fwd.alpha` (per timestep) | `@binding(0) prev_alpha: array<f32>` | N values |

**Consumer changes:** 5 validation binaries updated to use `chunks(n)`
iteration instead of nested vec indexing.

### Spectral Commutativity Module (`src/spectral_commutativity.rs`)

All matrix operations now take flat `&[f64]` with explicit `n` dimension:

```rust
// Before: fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>>
// After:
pub fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64>
```

**Functions updated:** `frobenius_norm`, `transpose`, `mat_mul`, `commutator`,
`distance_to_normal`, `commutativity_ratio`, `skip_commutativity`,
`identity_matrix`, `random_matrix`, `random_symmetric`, `spectral_gap_approx`.

**GPU benefit:** Flat buffers upload directly to `barracuda::ops::matmul`
without conversion. Cache-friendly layout improves CPU performance.

---

## Part 2: Consolidated Mathematical Primitives (`src/primitives.rs`)

New module centralizing duplicated math across 8 library modules:

### Constants (replacing module-local magic numbers)

| Constant | Value | Replaces | Used In |
|----------|-------|----------|---------|
| `LOG_GUARD` | `1e-300` | Inline `1e-300` | `hmm.rs`, `spectral_commutativity.rs` |
| `HILL_EPS` | `1e-20` | Local `EPS` | `signal_integration.rs`, `regulatory_network.rs` |
| `DIVISION_GUARD` | `1e-15` | Inline `1e-15` | `meta_population.rs`, `modes.rs` |

### Functions

| Function | Replaces | Consolidated From |
|----------|----------|-------------------|
| `shannon_entropy(probs)` | 6 inline variants | `eco_dynamics`, `pangenome_selection`, `modes`, `swarm_robotics` |
| `shannon_equitability(probs)` | Inline equitability | `eco_dynamics`, `modes` |
| `shannon_entropy_from_counts(counts)` | Count-based variant | `regulatory_network` |
| `hill_activation(x, k, n)` | Local `hill_activation` | `regulatory_network`, `signal_integration` |
| `hill_repression(x, k, n)` | Local `hill_repression` | `regulatory_network` |
| `sigmoid(x)` | 2 local variants | `swarm_robotics`, `sequence` |
| `rk4_step::<N>(state, dt, rhs)` | 2 local RK4 implementations | `regulatory_network`, `signal_integration` |

**Generic RK4:** Uses `const N: usize` for compile-time array size and
`FnMut` closure for the right-hand side, supporting mutable captures
(e.g. RNG in stochastic ODEs).

### ToadStool absorption opportunity

These primitives are candidates for `barracuda::numerical` and
`barracuda::stats` expansion:
- `hill_activation` / `hill_repression` → `barracuda::numerical::hill`
- `shannon_entropy` → `barracuda::stats::entropy`

---

## Part 3: Graceful GPU Error Handling

### `ValidationHarness::require()` method

```rust
pub fn require<T, E: Display>(&mut self, label: &str, result: Result<T, E>) -> Option<T>
```

### `require!` macro

```rust
let tensor = require!(h, Tensor::from_data(&data, shape, dev.clone()), "GPU alloc");
```

On `Err`: records a FAIL in the harness and returns from the function.
On `Ok`: unwraps the value.

**Scope:** All `.expect()` calls in 8 validation binaries converted —
approximately 90 call sites total. Removed all `#[allow(clippy::expect_used)]`
attributes.

**CI benefit:** Validation binaries now run cleanly on machines without GPU
adapters (all checks recorded as FAIL, no panic/abort).

---

## Part 4: Absorption Recommendations

### Priority 1: HMM Forward (flat layout ready)

`hmm.rs` now uses flat `Vec<f64>` matching the `hmm_forward_log.wgsl` buffer
layout. The GPU dispatcher in `evolved/hmm_forward_gpu.rs` (270 LOC) can be
absorbed into `barracuda::ops::hmm` with minimal changes — the double-buffered
alpha swap pattern is production-ready.

### Priority 2: Spectral Mat-Mul (flat layout ready)

`spectral_commutativity.rs` flat `mat_mul` is a direct CPU reference for
`barracuda::ops::matmul`. The i-k-j loop order is cache-optimal and matches
the BLAS panel access pattern.

### Priority 3: Entropy / Hill Functions

`primitives::shannon_entropy` and `primitives::hill_activation` are
well-tested (8 new unit tests) and could expand `barracuda::stats` and
`barracuda::numerical` respectively.

### Priority 4: `require!` Pattern

The `require!` macro pattern is useful for any Spring writing validation
binaries. Consider adding a similar mechanism to `barracuda::testing` or
documenting it in the wateringHole standards.

---

## Part 5: Quality Gate (post-evolution)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets -W pedantic -W nursery` | 0 warnings |
| `cargo doc --no-deps` | 0 warnings |
| `cargo test` | 198 unit + 7 doc-tests PASS |
| All validation binaries | 67 binaries compile |
| `unsafe` | Forbidden (`#![forbid(unsafe_code)]`) |

---

## Part 6: Files Changed

| File | Change |
|------|--------|
| `src/primitives.rs` | **NEW** — consolidated math + constants |
| `src/hmm.rs` | Flat row-major layout, `from_flat()`, `ForwardResult::alpha_at()` |
| `src/spectral_commutativity.rs` | Flat row-major layout, explicit `n` dimensions |
| `src/validation.rs` | `require()` method + `require!` macro |
| `src/eco_dynamics.rs` | Zero-copy `&[u8]` genotypes |
| `src/regulatory_network.rs` | Uses `primitives::hill_*`, `rk4_step`, `shannon_entropy_from_counts` |
| `src/signal_integration.rs` | Uses `primitives::hill_activation`, `rk4_step` |
| `src/swarm_robotics.rs` | Uses `primitives::sigmoid`, `shannon_entropy` |
| `src/pangenome_selection.rs` | Uses `primitives::shannon_entropy` |
| `src/modes.rs` | Uses `primitives::shannon_equitability`, `DIVISION_GUARD` |
| `src/sequence.rs` | Uses `primitives::sigmoid` |
| `src/meta_population.rs` | Uses `primitives::DIVISION_GUARD` |
| 8 validation binaries | `.expect()` → `require!` macro |
| 5 HMM consumer binaries | Flat alpha/posterior via `chunks(n)` |
| 3 spectral consumer binaries | Flat matrix ops with explicit `n` |
| 40 Python/shell files | SPDX `AGPL-3.0-or-later` headers |

---

*Following the hotSpring pattern: Write → Absorb → Lean.*
*neuralSpring deep evolution — February 21, 2026.*
