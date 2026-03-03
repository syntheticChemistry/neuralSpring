<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/barraCuda Handoff V79 — Deep Lint Evolution & Shared Validation Helpers

**Date**: March 3, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/barraCuda team
**License**: AGPL-3.0-or-later
**Covers**: Session 119 — `#[allow(` → `#[expect(` migration, shared validation helpers, debris sweep
**Supersedes**: V78 (S118 barraCuda standalone rewire)
**barraCuda**: v0.3.1 standalone (`../barraCuda/crates/barracuda`)

---

## Executive Summary

- **Full `#[allow(` → `#[expect(` migration**: Every lint suppression in production library and bin code now uses `#[expect(lint, reason = "...")]`. This means Rust will warn us if any suppression becomes unnecessary — catching dead code, resolved lint issues, or over-suppression automatically.
- **Zero lib clippy warnings**: 0 warnings under `clippy::pedantic + clippy::nursery`. 0 unfulfilled expectations across all 232 binaries.
- **4 shared validation helpers extracted**: Deduplicates ~25+ inline sites across 13 bin files.
- **869 lib tests** (up from 861): 8 new tests for shared helpers.
- **Zero `#[allow(` in production code**: Only 6 `#[allow(` remain — all in `#[cfg(test)]` modules where `expect_used`/`unwrap_used` lints don't fire.

---

## Part 1: `#[allow(` → `#[expect(` Migration

### Scope

| Target | Module-level | Inline | Total |
|--------|-------------|--------|-------|
| `src/bin/` | 208 `#![allow(` | 31 `#[allow(` | 239 |
| `src/` (lib) | 28 `#![allow(` | 4 `#[allow(` | 32 |
| **Total converted** | **236** | **35** | **271** |

### Unfulfilled Resolution

477+ unfulfilled lint expectations were identified and resolved by removing the specific lints that weren't firing. This exposed widespread over-suppression: many `#[allow(clippy::cast_possible_truncation)]` or `#[allow(clippy::similar_names)]` directives were preemptive and never triggered. Removing them tightens the codebase.

### What This Means for barraCuda

If barraCuda adopts `#[expect(` (which we recommend), any time you refactor code to eliminate a cast, rename variables, or reduce function args, the compiler will tell you the suppression is no longer needed. This prevents suppression debt from accumulating.

### Pattern for Test Code

`clippy::expect_used` and `clippy::unwrap_used` don't fire in `#[cfg(test)]` modules. Use `#[allow(clippy::expect_used)]` (not `#[expect(`) for test infrastructure.

Similarly, `clippy::wildcard_imports` on `use super::*` has cross-compilation-context behavior — use `#[allow(` for imports that are compiled in both lib and test contexts.

---

## Part 2: Shared Validation Helpers

### New Helpers in `neural_spring::validation`

| Helper | Signature | Replaces |
|--------|-----------|----------|
| `max_abs_diff_f64` | `(a: &[f64], b: &[f64]) -> f64` | 3 local `max_diff`/`max_pairwise_diff` + ~25 inline fold patterns |
| `bench_once` | `<F: FnOnce() -> T, T>(label: &str, f: F) -> (T, f64)` | 4 identical `bench` helpers in validators |
| `bench_median` | `<F: FnMut()>(warmup: usize, iters: usize, f: F) -> f64` | `bench_rust` in 8+ benchmark binaries |
| `median_duration_us` | `(times: &mut [Duration]) -> f64` | 6 local `median`/`median_us` implementations |

### Absorption Opportunity

These helpers complement the `ValidationHarness`, `exit_no_gpu`, and `require!` patterns already absorbed into `barracuda::validation`. Consider absorbing:

1. `max_abs_diff_f64` → `barracuda::validation::max_abs_diff_f64` (cross-spring utility)
2. `bench_once` + `bench_median` → `barracuda::validation::bench` module
3. `median_duration_us` → `barracuda::validation::stats`

---

## Part 3: BarraCUDA Usage Summary (S119)

### API Surface (unchanged from V78)

- **~117 files** with barracuda imports
- **25+ submodules** exercised (device, tensor, ops::bio, dispatch, stats, linalg, numerical, special, spectral, nautilus, staging, pipeline, unified_math, unified_hardware, tolerances)
- **44 upstream rewires** (local → barracuda delegate)
- **21/21 WGSL shaders absorbed** upstream
- **4 local shaders remaining**: `xoshiro128ss`, `swarm_nn_scores`, `head_split`, `head_concat`

### Evolution Opportunities for barraCuda

| Area | Detail | Priority |
|------|--------|----------|
| `StatefulPipeline` batching | HMM chain, ODE loops, iterative EA — reduce CPU round-trips | P2 |
| `UnidirectionalPipeline` streaming | Streaming fitness eval — reduce O(T) to O(1) round-trips | P2 |
| `ReduceScalarPipeline` | Log-likelihood, convergence checks | P3 |
| `#[expect(` adoption | Migrate barraCuda's own `#[allow(` to `#[expect(` for drift detection | P3 |
| Shared validation helpers | Absorb `max_abs_diff_f64`, bench helpers from neuralSpring | P4 |

---

## Part 4: Quality State

| Gate | Value |
|------|-------|
| `cargo fmt --check` | Clean |
| `cargo check --all-targets --all-features` | Clean |
| `cargo clippy` (pedantic+nursery) | **0 lib warnings, 0 bin warnings, 0 unfulfilled** |
| `cargo doc --no-deps` | Clean (234 pages) |
| `cargo test --lib` | **869/869 PASS** |
| `#[allow(` in production lib | **0** |
| `#[allow(` in bin/ | **0** |
| `#[allow(` in test modules | 6 (intentional — `expect_used`/`unwrap_used`) |
| `validate_all` | 212/212 PASS |

---

## Action Items for ToadStool/barraCuda

1. **Consider adopting `#[expect(` in barraCuda** — prevents suppression debt, catches resolved lints automatically
2. **Absorb shared validation helpers** — `max_abs_diff_f64`, `bench_once`, `bench_median`, `median_duration_us` into `barracuda::validation`
3. **Pipeline batching** — `StatefulPipeline` and `UnidirectionalPipeline` are the next frontier for GPU-resident acceleration
4. **4 remaining local shaders** — `xoshiro128ss`, `swarm_nn_scores` could be generalized upstream

---

*V79 — neuralSpring Session 119 (March 3, 2026)*
