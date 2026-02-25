# neuralSpring Session 59 Handoff — V24

**Date**: February 24, 2026
**ToadStool HEAD**: `9404fdb4`
**Previous**: V23 (Session 58 — 7 Dispatcher methods rewired to upstream domain_ops)

---

## What Changed

### 3 Library Functions Rewired to Upstream BarraCUDA

| Local Function | Module | Upstream API | Absorbed In |
|----------------|--------|-------------|-------------|
| `empirical_spectral_density` | `weight_spectral` | `barracuda::stats::empirical_spectral_density` | S54 (M-011) |
| `marchenko_pastur_bounds` | `weight_spectral` | `barracuda::stats::marchenko_pastur_bounds` | S54 (M-012) |
| `effective_rank` | `neural_pgm` | `barracuda::linalg::effective_rank` | S54 (H-009) |

These three functions were contributed by neuralSpring to BarraCUDA in S54 but
neuralSpring continued using local implementations. The local bodies are now
replaced with single-line delegations to the upstream equivalents.

### 2 New Dispatcher Methods via Upstream domain_ops

| Dispatcher Method | Upstream Function | CPU Fallback |
|-------------------|-------------------|-------------|
| `gelu` | `barracuda::dispatch::gelu_dispatch` | `transformer::gelu` |
| `hmm_forward_step` | `barracuda::dispatch::hmm_forward_dispatch` | `cpu_fallback::hmm_forward_step` |

Both follow the existing error-fallback pattern: upstream GPU/CPU routing with
size-based thresholds, falling back to local CPU on any error.

### evolved/ Module Cleanup

| Change | Details |
|--------|---------|
| Removed `WGSL_BATCH_FITNESS_EVAL` re-export | Zero callers — all use `barracuda::ops::batch_gemm` |
| Removed `WGSL_RK4_PARALLEL` re-export | Zero callers — all use `barracuda::ops::rk_stage` |
| Removed `WGSL_MEAN_REDUCE` re-export | Zero callers — all use `barracuda::pipeline::ReduceScalarPipeline` |
| Kept `WGSL_HEAD_SPLIT` / `WGSL_HEAD_CONCAT` | MHA S-03b workaround still active |

### MHA Retirement Assessment

`evolved/mha` assessed for retirement. Upstream `barracuda::ops::mha::MultiHeadAttention`
exists (S52+) with full projection + attention pipeline. However, retirement criteria
require projection shaders to work on RTX 4070 + Vulkan at production sizes
(B=4, S=128, H=8, d=512). This has not been validated yet. Module kept active;
status documented in `evolved/mha.rs` and `evolved/mod.rs`.

## Cumulative Rewiring Status

| Session | Functions Rewired | Target |
|---------|-------------------|--------|
| S56 | `graph_laplacian`, `disordered_laplacian`, `belief_propagation_chain`, `numerical_hessian` | `barracuda::linalg::graph`, `barracuda::numerical` |
| S57 | `patch_pow_to_polyfill` consolidated | `validation::patch_pow_to_polyfill` |
| S58 | `mat_mul`, `frobenius_norm`, `transpose`, `softmax`, `l2_distance`, `mean`, `variance` | `barracuda::dispatch::domain_ops` |
| S59 | `empirical_spectral_density`, `marchenko_pastur_bounds`, `effective_rank` | `barracuda::stats`, `barracuda::linalg` |
| S59 | (new) `gelu`, `hmm_forward_step` | `barracuda::dispatch::domain_ops` |

**Total**: 16 functions delegating to upstream BarraCUDA (up from 11 in V23).

## Validation

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets` (pedantic + nursery) | 0 warnings |
| `cargo test --lib` | 482 PASS |
| `validate_all` | 145/146 PASS |

The single failure (`validate_barracuda_logsumexp`) is a pre-existing upstream
buffer size mismatch in the logsumexp shader — unrelated to this session's changes.

## Files Modified

| File | Change |
|------|--------|
| `src/weight_spectral.rs` | `empirical_spectral_density` and `marchenko_pastur_bounds` bodies replaced with upstream delegation |
| `src/neural_pgm.rs` | `effective_rank` body replaced with upstream delegation |
| `src/gpu_dispatch/dispatch_ops.rs` | Added `gelu` and `hmm_forward_step` methods |
| `src/gpu_dispatch/cpu_fallback.rs` | Added `hmm_forward_step` CPU fallback |
| `src/evolved/mod.rs` | Removed 3 dead WGSL re-exports, updated absorption status docs |
| `src/evolved/mha.rs` | Updated status to reference S59 / upstream `ops::mha` |
| `specs/TOADSTOOL_HANDOFF.md` | Added S59 section, updated rewire count to 16 |
| `specs/BARRACUDA_USAGE.md` | Added `stats::empirical_spectral_density`, `stats::marchenko_pastur_bounds`, `linalg::effective_rank`, updated dispatch count to 9 |
| `specs/CROSS_SPRING_EVOLUTION.md` | Updated provenance table, added S59 section |
| `specs/EVOLUTION_MAPPING.md` | Updated session reference |

## What Remains

| Item | Status | Blocker |
|------|--------|---------|
| `evolved/mha` retirement | Pending | Upstream projection shader validation on RTX 4070 |
| `validate_barracuda_logsumexp` fix | Pre-existing | Upstream buffer size mismatch |
| `evolved/` migration (~2075 lines, D-S20-003) | ToadStool deep debt | Conv2D/Pool wiring incomplete |

---

*neuralSpring V24 handoff — S54-S59 absorption cycle complete, 16 functions upstream.*
