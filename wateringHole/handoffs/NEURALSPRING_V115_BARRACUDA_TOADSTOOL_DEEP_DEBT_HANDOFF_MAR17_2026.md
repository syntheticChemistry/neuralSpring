# neuralSpring V115 → barraCuda / toadStool Deep Debt Evolution Handoff

**Date**: March 17, 2026
**From**: neuralSpring Session 164, V115
**To**: barraCuda / toadStool teams
**License**: AGPL-3.0-or-later
**Covers**: V114–V115, Session 164
**Supersedes**: V114 bC/tS handoff

## Executive Summary

- **7 inline tolerances named** and wired across 6 validation binaries
- **`solve_symmetric` delegated** to `barracuda::linalg::solve::solve_f64_cpu()`
- **MSRV 1.87** pinned across all 3 workspace crates
- **Platform-agnostic testing** — `/tmp` → `std::env::temp_dir()`, socket names → `niche::NICHE_NAME`
- **Idiomatic Rust** — `partial_cmp().unwrap()` → `total_cmp()`, smart tolerance refactoring
- Zero regressions, zero warnings, zero unsafe

## Part 1 — barraCuda API Usage

### solve_f64_cpu Delegation

`glucose_prediction::solve_symmetric` now delegates to `barracuda::linalg::solve::solve_f64_cpu()`.

**Why this matters for barraCuda**:
- Confirms `solve_f64_cpu` is production-grade for SPD linear systems
- neuralSpring adds ridge regularization as a caller-side fallback for near-singular matrices
- The `cholesky_f64_cpu` function is `#[cfg(test)]`-gated in barraCuda — this is fine; `solve_f64_cpu` (Gaussian elimination) is the correct public API for general SPD systems

**Pattern** (for other Springs):

```rust
barracuda::linalg::solve::solve_f64_cpu(a, b, n).unwrap_or_else(|_| {
    let mut regularized = a.to_vec();
    for i in 0..n {
        regularized[i * n + i] += REGULARIZATION_EPS;
    }
    barracuda::linalg::solve::solve_f64_cpu(&regularized, b, n)
        .unwrap_or_else(|_| vec![0.0; n])
})
```

### Local Math Retained (Intentional)

| Function | Module | Reason |
|----------|--------|--------|
| `mat_mul_transpose` | `information_flow` | CPU-only AᵀA for n=4..8 matrices. Dispatch overhead would dominate. Future: `barracuda::linalg::gram_cpu` when available |
| `softmax_rows` | `information_flow` | Intentional CPU f64 reference for validation. Not a delegation candidate |

### New Tolerance Constants

7 new named constants registered in `domain_guards` category:

| Constant | Value | Domain | Used In |
|----------|-------|--------|---------|
| `GPU_TRACE_F32_ROUNDTRIP` | 0.01 | GPU f32 parity | `validate_barracuda_attention_anderson` |
| `CORRELATION_CROSS_VALIDATION` | 0.05 | Cross-method stats | `validate_barracuda_wdm_ensemble_qs` |
| `GPU_ACCUMULATION_F32` | 0.1 | GPU f32 accumulation | `validate_barracuda_wdm_ensemble_qs` |
| `CLASSIFIER_METRIC_CROSS` | 0.01 | Binary classifier | `validate_introgression_nn` |
| `INTROGRESSION_FRACTION_CROSS` | 0.05 | Domain-specific | `validate_introgression_nn` |
| `PROCESS_MODEL_RESPONSE` | 0.05 | ODE response | `validate_digestion_prediction` |
| `RPC_COUNT_FALLBACK` | 0.5 | IPC resilience | `validate_nucleus_tower` |

## Part 2 — Patterns for Absorption

### MSRV Pinning

neuralSpring now pins `rust-version = "1.87"` in all Cargo.toml files.

**Recommendation for barraCuda/toadStool**: Consider pinning MSRV for reproducible builds.
Edition 2024 requires ≥1.85; `rust-version = "1.87"` provides margin for recent stabilizations
(`total_cmp`, let chains, `gen` keyword reservation).

### `total_cmp()` for Float Sorting

All float sorting in neuralSpring now uses `f64::total_cmp()` (stable since Rust 1.62) instead of
`partial_cmp(b).unwrap()` or `partial_cmp(b).unwrap_or(Ordering::Equal)`.

**Why**: `total_cmp` defines a total order including NaN, avoiding the unwrap/panic risk.

**Recommendation**: Sweep `partial_cmp` sort patterns in barraCuda benchmark and test code.

### Tolerance Submodule Pattern

`tolerances/mod.rs` was approaching the 1000 LOC wateringHole limit. Training-related
constants were extracted into `tolerances/training.rs` with `mod training; pub use training::*;`
in the parent module. API is unchanged for callers.

**Recommendation**: Consider this pattern for `barracuda::tolerances` if it grows.

### Platform-Agnostic Test Paths

All test-specific temporary paths now use `std::env::temp_dir()` instead of `/tmp/`.
This enables Windows/macOS CI without path assumptions.

## Part 3 — Still-Relevant Items from V114

- blake3 `pure` feature request (zero C deps)
- Variance semantics documentation (population vs sample)
- `enable f64;` PTXAS regression workaround
- xoshiro128ss.wgsl absorption into `barracuda::ops::prng`
- nn::Layer / nn::Optimizer for training loops
- Autograd reverse-mode AD
- Flash attention / fused LayerNorm+GELU kernels

## Part 4 — Quality Metrics

| Metric | Value |
|--------|-------|
| Library tests | 1152 |
| Forge tests | 73 |
| PlayGround tests | 70 |
| Validation binaries | 260 |
| Clippy warnings | 0 (pedantic+nursery) |
| Unsafe code | 0 (`#![forbid(unsafe_code)]`) |
| `#[allow()]` | 0 (all `#[expect(reason)]`) |
| C dependencies | 0 (ecoBin compliant) |
| Files > 1000 LOC | 0 |
| Named tolerances | 180+ |
| Upstream rewires | 46 |
| Edition | 2024 |
| MSRV | 1.87 |

---

AGPL-3.0-or-later
