# neuralSpring V120 → Ecosystem Cross-Absorption Handoff

**Date**: March 18, 2026
**From**: neuralSpring Session 168b, V120
**To**: barraCuda / toadStool / ecosystem
**License**: AGPL-3.0-or-later
**Covers**: V119–V120, Session 168b
**Supersedes**: V119 Deep Debt Execution Handoff

## Executive Summary

Session 168b absorbs patterns from 6 sibling springs and 12 primals. Five
ecosystem patterns adopted: centralized RPC result extraction (healthSpring),
biomeOS Neural API domain (healthSpring), OnceLock GPU probe cache
(groundSpring), provenance registry with completeness testing (healthSpring),
and cast lint deny (airSpring). Total: 1320 tests, zero clippy, zero unsafe.

**Absorbed from:**
- healthSpring V37: `extract_rpc_result()`, `PRIMAL_DOMAIN`, provenance registry
- groundSpring V116: `OnceLock` GPU probe cache
- airSpring V0.9.0: cast lint deny in Cargo.toml

---

## Part 1 — What neuralSpring Absorbed

### 1.1 `extract_rpc_result()` / `extract_rpc_result_owned()` (healthSpring V37)

Centralized JSON-RPC result extraction that returns `None` when an error
field is present. Replaces ad-hoc `response.get("result")` in
`DispatchOutcome::classify_response`. Includes 5 tests (borrow, error,
neither, owned, fuzz).

### 1.2 `PRIMAL_DOMAIN` (healthSpring V34)

`pub const PRIMAL_DOMAIN: &str = "science.learning"` in `config.rs` for
biomeOS Neural API registration and Songbird capability discovery.

### 1.3 `OnceLock` GPU Probe Cache (groundSpring V116)

`static GPU_PROBE_CACHE: OnceLock<Vec<Substrate>>` in
`metalForge/forge/src/probe.rs`. Caches the result of `wgpu::Instance::new()`
and adapter enumeration, preventing SIGSEGV from concurrent Vulkan
initialization in parallel tests and eliminating redundant probe passes.

### 1.4 `PROVENANCE_REGISTRY` (healthSpring V37 Pattern)

Complete array of all 49 `BaselineProvenance` records with 4 integrity tests:
- `provenance_registry_completeness` — counts `pub const *_PROVENANCE` in
  source and asserts registry length matches
- `provenance_registry_no_duplicate_scripts` — ensures no script appears twice
- `provenance_registry_expected_source_complete` — every record has a
  non-empty `expected_source()`
- `provenance_registry_records_non_empty` — all fields populated, commit valid

### 1.5 Cast Lint Deny (airSpring V0.9.0 Pattern)

All 3 workspace `Cargo.toml` files now warn on:
- `clippy::cast_precision_loss`
- `clippy::cast_possible_truncation`
- `clippy::cast_sign_loss`
- `clippy::cast_lossless`

Modules that require casts already have `#[expect(reason)]` attributes.

---

## Part 2 — Patterns Reviewed But Not Absorbed (and Why)

| Pattern | Source | Decision |
|---------|--------|----------|
| `cast` module as shared crate | airSpring | neuralSpring has `safe_cast.rs` (5 helpers); await barraCuda canonicalization |
| Manifest discovery (tiers 4–5) | primalSpring v0.3.0 | neuralSpring's 5-tier socket resolution is sufficient; manifest discovery adds complexity for primal-to-primal coordination we don't need |
| `ValidationSink` trait | groundSpring V116 | We have `ValidationHarness` which is the upstream pattern; `ValidationSink` is groundSpring's variant |
| Typed `DispatchError` enum | groundSpring V116 | We have `IpcError` for IPC; GPU dispatch uses `Option` (graceful degradation); adding another error type adds complexity without benefit |
| Session-level provenance DAGs | wetSpring V128 | Would require rhizoCrypt dependency; our `BaselineProvenance` + `PROVENANCE_REGISTRY` covers our needs |

---

## Part 3 — What Ecosystem Should Absorb from neuralSpring

### 3.1 `PROVENANCE_REGISTRY` Completeness Pattern

The `include_str!` + `pub const` count technique ensures no provenance
record can be added without also being included in the registry. This
should be ecosystem-standard for any spring with >10 provenance records.

### 3.2 `OnceLock` GPU Probe + `test_gpu_lock` Combination

neuralSpring now has two complementary patterns:
- `OnceLock<Vec<Substrate>>` in probe (process-wide, avoids SIGSEGV)
- `OnceLock<Mutex<()>>` in `test_gpu_lock` (serializes GPU test submissions)

Together they prevent both initialization races and submission races.

### 3.3 Cast Lint Deny in Cargo.toml

All springs should add cast lints to `[lints.clippy]` rather than relying
on `clippy::pedantic` (which only enables `cast_lossless`). The four lints
catch silent precision loss and truncation at compile time.

---

## Part 4 — Quality Metrics

| Category | Count |
|----------|-------|
| Library unit tests | 1167 (+3) |
| Forge unit tests | 73 |
| playGround unit tests | 80 (+5) |
| Property tests | 36 |
| IPC fuzz tests | 5 |
| Named tolerances | 225 |
| Provenance records | 49 (registry-tested) |
| Validation binaries | 238 |
| Benchmark binaries | 18 |
| **Total test artifacts** | **1320 lib + 267 binaries** |

Zero clippy (pedantic+nursery+cast, workspace-wide), zero fmt diffs,
zero doc warnings, zero unsafe, zero C deps, zero `#[allow()]`, zero
mocks in production. MSRV 1.87, Edition 2024.

## Files Changed (S168b)

| File | Change |
|------|--------|
| `playGround/src/ipc_client.rs` | `extract_rpc_result()` + tests, `classify_response` delegated |
| `src/config.rs` | `PRIMAL_DOMAIN` constant |
| `metalForge/forge/src/probe.rs` | `OnceLock` GPU probe cache |
| `src/provenance/mod.rs` | `PROVENANCE_REGISTRY` + 4 completeness tests |
| `Cargo.toml` | cast lint deny |
| `playGround/Cargo.toml` | cast lint deny |
| `metalForge/forge/Cargo.toml` | cast lint deny |
| Root docs | S168b entries |

---

*AGPL-3.0-or-later — neuralSpring → ecosystem*
