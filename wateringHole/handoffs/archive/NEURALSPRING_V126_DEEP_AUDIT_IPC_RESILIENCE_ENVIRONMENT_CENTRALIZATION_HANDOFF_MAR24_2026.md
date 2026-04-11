<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V126 — Deep Audit Execution: IPC Resilience, Environment Centralization, GPU Module Refactor

**Session**: S176 (March 24, 2026)
**Audience**: All springs, all primals, biomeOS integrators
**Previous**: V125 (S175 — ecosystem absorption, ValidationSink, cast deny)
**Status**: Complete
**Supersedes**: V125 (archive V124 + V125)

---

## Summary

S176 executed a comprehensive codebase audit against wateringHole standards
and resolved every finding. Key outcomes: clippy zero-warning gate restored,
provenance environment strings centralized, IPC resilience wired into live
paths, GPU module refactored, integration test coverage expanded to all 49
provenance records, and documentation reconciled with current codebase state.

---

## What Changed

### 1. Clippy Zero-Warning Gate Restored (P0)

6 clippy errors in test code fixed:
- `clippy::similar_names` in `provenance/mod.rs` — renamed `hasher` → `content_hasher`
- `clippy::unwrap_used` in `validation/sink.rs` and `validation/mod.rs` —
  `#[expect(clippy::unwrap_used)]` on test modules with documented reason

**Result**: `cargo clippy --all-targets --all-features -- -D warnings` exits 0.

### 2. Provenance Environment Centralization (P0)

19 literal `"Python 3.10.12, NumPy 2.2.6, seed=42"` strings in
`provenance/experiments.rs` replaced with centralized `WDM_ENVIRONMENT`
constant. One `"Python 3.12, NumPy, seed=42"` replaced with
`ANDERSON_MULTIAGENT_ENVIRONMENT`. Zero non-centralized environment strings
remain in the provenance registry.

**New constants** in `provenance/mod.rs`:
- `WDM_ENVIRONMENT` — WDM surrogates, AlphaFold, baseCamp composition
- `ANDERSON_MULTIAGENT_ENVIRONMENT` — Exp-053 Anderson multi-agent

### 3. Integration Test Expansion (P0)

`tests/integration.rs` expanded from 9 to 12 tests. The provenance test
now iterates over the complete `PROVENANCE_REGISTRY` (all 49 records)
instead of a hand-curated 26-record subset:

- `provenance_registry_all_records_have_valid_commit_and_command` — validates all 49
- `provenance_all_records_have_well_formed_environment` — enforces centralized constants
- `provenance_cpu_parity_records_have_parity_environment` — commit-aware
- `provenance_registry_minimum_record_count` — guards against accidental deletion

### 4. IPC Resilience Wired into Live Paths (P1)

`RetryPolicy` (exponential backoff) and `CircuitBreaker` (3 failures → 10s
cooldown) now wrap every `PetalTonguePushClient` RPC call:

- `send_rpc()` retries on `ConnectionFailed`, records success/failure
- Circuit breaker short-circuits when petalTongue is unreachable
- Extracted `try_send()` and `check_rpc_error()` for clean separation

Previously these primitives existed in `ipc_resilience.rs` but were library-only.

### 5. GPU Module Refactor

`src/gpu.rs` (714 LOC) split into `src/gpu/mod.rs` (475 LOC) +
`src/gpu/tests.rs` (238 LOC). Module path `crate::gpu::tests::shared_gpu`
preserved for 4 downstream consumers. Production code now well under the
1000 LOC ceiling with room for growth.

### 6. Documentation Reconciliation

- `CONTROL_EXPERIMENT_STATUS.md`: capability count 7 → 16 (full list)
- `specs/DATA_PROVENANCE.md`: pretrained models section clarified (nS-01 exception)
- `src/tolerances/`: 7 backtick `BASELINE_COMMIT` placeholders → actual `f9ad0268`
- `src/evolved/mod.rs`: barraCuda v0.3.1 → v0.3.7
- `src/bin/validate_modern_cross_spring/report.rs`: v0.3.5 → v0.3.7
- `specs/BARRACUDA_USAGE.md`: `src/gpu.rs` → `src/gpu/mod.rs`
- `CONTEXT.md`: file count 465 → 466, integration tests 9 → 12

---

## Patterns We Export

### IPC Resilience (New)

`PetalTonguePushClient` now demonstrates the full IPC resilience pattern:
retry with exponential backoff + circuit breaker. Other springs and primals
with IPC clients should adopt `ipc_resilience::{RetryPolicy, CircuitBreaker}`.

### Environment Centralization

All provenance environment strings are now named constants. Integration tests
enforce that no record uses a non-centralized string. Other springs with
Python baselines should centralize their environment strings similarly.

---

## For barraCuda Team

No new absorption requests beyond V125/S175. The 4 generic f64 ops
(gelu/sigmoid/layer_norm/softmax) and `domain-fold` feature pack remain
the active upstream items from V124.

## For All Springs

| Pattern | Where | What's New |
|---------|-------|------------|
| IPC resilience | `src/ipc_resilience.rs` + `visualization/ipc_push.rs` | Now wired into live PetalTongue path |
| Environment centralization | `src/provenance/mod.rs` | `WDM_ENVIRONMENT`, `ANDERSON_MULTIAGENT_ENVIRONMENT` |
| Full-registry integration test | `tests/integration.rs` | Loop over `PROVENANCE_REGISTRY` instead of hand-curated list |
| GPU module split | `src/gpu/` | Test extraction pattern for large modules |

---

## Metrics

| Metric | V125 | V126 |
|--------|------|------|
| Clippy warnings | 0 | 0 (gate restored after test code drift) |
| Integration tests | 9 | **12** |
| Provenance records tested | 26/49 | **49/49** |
| Non-centralized env strings | 20 | **0** |
| IPC clients with resilience | 0 | **1** (petalTongue) |
| `gpu.rs` LOC | 714 | **475** (+ 238 in tests.rs) |
| Total tests | ~1,400 | ~1,403 |
| Validation binaries | 261 | 261 |
| Named tolerances | 232+ | 232+ |

---

## Quality Gates

| Check | Result |
|-------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets --all-features -- -D warnings` | PASS (0 warnings) |
| `cargo test --test integration` | PASS (12/12) |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` | PASS |
| `cargo test --lib --no-run` | PASS |
| unsafe code | None (`#![forbid(unsafe_code)]`) |
| `#[allow()]` | None |
| Files > 1000 LOC | None |
| Production mocks | None |
