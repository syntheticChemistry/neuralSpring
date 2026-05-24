# neuralSpring V170 Handoff — Deep Debt Evolution Sprint

**Date:** May 22, 2026
**Session:** S214
**From:** neuralSpring
**To:** primalSpring, downstream consumers
**License:** AGPL-3.0-or-later

---

## Summary

Deep debt evolution sprint resolving 7 categories of code quality improvement with zero regressions across 754 tests.

**45 capabilities** registered (39 → 45). **754 workspace tests** (all pass). Deploy graph `graphs/neuralspring_deploy.toml` aligned to **V170/S214**.

---

## Changes

### Registry Alignment (39 → 45 capabilities)

6 NUCLEUS pipeline capabilities (`science.eigensolve`, `science.digester_anderson_coupling`, `science.isomorphic_reservoir`, `science.wdm_ensemble_qs`, `science.introgression_nn`, `science.attention_anderson`) now advertised via `capability.list`. Previously exercised in dispatch but never registered.

**Files:** `src/config.rs`, `config/capability_registry.toml`, `src/niche.rs`

### Hardcode Elimination

`"skunkbat"` string literal in `handlers.rs` replaced with `primal_names::SKUNKBAT` constant. Zero hardcoded primal names remaining.

**Files:** `src/bin/neuralspring_primal/handlers.rs`

### Discovery Standardization

`probe_capabilities()` now uses canonical `capability.list` with dual-probe fallback to `capabilities.list` (plural) for older primals. IPC discovery model documented as hint-then-probe architecture.

**Files:** `src/validation/composition.rs`, `src/ipc/mod.rs`

### Feature Gate Cleanup

- Documented `composed` feature alias in `Cargo.toml`
- Removed 4 redundant `#[cfg(feature = "barracuda")]` and dead `#[cfg(not(feature = "barracuda"))]` in `loss_landscape.rs` and `weight_spectral/metrics.rs`

**Files:** `Cargo.toml`, `src/loss_landscape.rs`, `src/weight_spectral/metrics.rs`

### IPC Error Typing

`validation/composition.rs` IPC helpers evolved from `Result<_, String>` to `Result<_, IpcError>` with structured `Transport`/`Protocol` variants. Silent IPC fallback in `executor.rs` now logs via `log::warn!`.

**Files:** `src/validation/composition.rs`, `src/nucleus_pipeline/executor.rs`, `src/bin/validate_composition_evolution.rs`

### Rust Idiom Improvements

- 12 `HashMap::insert` chains → `[].into_iter().map().collect()`
- O(n²) `Vec::contains` dedup → `HashSet`
- `inter_population_af_variance` generalized from `&[Vec<f64>]` to `&[impl AsRef<[f64]>]`

**Files:** `src/nucleus_pipeline/dispatch.rs`, `src/ipc/mod.rs`, `src/meta_population/fst.rs`, `src/gpu_dispatch/dispatch_popgen.rs`

### Paper Queue Sync

B3 (Good 2017) and B4 (Blount 2008) status updated from QUEUED to COMPLETE in `specs/PAPER_REVIEW_QUEUE.md`.

---

## Metrics

| Metric | S213 (V169) | S214 (V170) |
|--------|-------------|-------------|
| Capabilities | 39 | 45 |
| Workspace tests | 754 | 754 |
| Hardcoded primal names | 1 | 0 |
| `Result<_, String>` IPC helpers | 4 | 0 |
| Redundant feature gates | 4 | 0 |
| O(n²) dedup sites | 1 | 0 |

---

## Test Verification

```
cargo test --lib → 754 passed, 0 failed
cargo build → OK (default + barracuda features)
cargo build --features barracuda → OK
```

---

## Upstream Impact

- primalSpring `capability_registry.toml` sync: 45 methods advertised
- 6 new `stable` capability entries for pipeline stages
- No breaking changes to IPC wire format
- `probe_capabilities()` return type changed from `Result<_, String>` to `Result<_, IpcError>` — callers using `.is_ok()`, `match`, or `{e}` formatting are unaffected
