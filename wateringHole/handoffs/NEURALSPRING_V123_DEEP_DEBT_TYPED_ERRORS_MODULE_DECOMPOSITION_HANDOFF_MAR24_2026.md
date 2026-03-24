# neuralSpring V123 — Deep Debt: Typed Errors, Module Decomposition, CI Hardening

**Date**: March 24, 2026 (Session S173)
**From**: neuralSpring
**To**: barraCuda / toadStool / ecosystem
**License**: AGPL-3.0-or-later

## Summary

Session S173 executed a comprehensive deep-debt resolution across the neuralSpring
workspace. Key evolution: stringly-typed errors replaced with thiserror enum
hierarchy, three large modules smart-decomposed, barraCuda feature selection
made explicit, CI hardened with cargo-deny and IPC smoke testing.

## Changes

### Typed Error System (thiserror)

- Added `thiserror = "2"` dependency
- Created `src/error.rs`: `GpuError`, `TensorError`, `ParseError`, `Error` enum
- Migrated `gpu.rs` from `Result<T, String>` → `Result<T, GpuError>`
- Migrated `gpu_ops/reduction.rs` → `Result<T, TensorError>`
- Bridge impls (`From<GpuError> for String`) for gradual migration
- **Ecosystem note**: This mirrors the typed-error pattern in barraCuda and
  groundSpring. Springs should converge on `thiserror` over `anyhow` for
  library boundaries.

### Module Decomposition (smart refactors)

| Module | Before | After | Rationale |
|--------|--------|-------|-----------|
| `nucleus_pipeline.rs` | 874 LOC | 5 files (mod/error/report/dispatch/executor) | Separate graph error, report, capability dispatch, execution |
| `glucose_prediction.rs` | 794 LOC | 5 files (mod/cgm/analysis/experiment/tests) | Separate CGM simulation, analysis, LSTM experiment |
| `immunological_anderson/mod.rs` | 779 LOC | 3 new submodules (classification/pharma + existing lattice/matrix) | Separate drug scoring, AD classification, pharmacokinetics |

### barraCuda Feature Selection

Changed from default features to explicit selection:
```toml
barracuda = { path = "../barraCuda/crates/barracuda", default-features = false, features = [
    "gpu", "domain-nn", "domain-esn", "domain-genomics", "domain-timeseries",
] }
```
Drops unused: `domain-pde`, `domain-snn`, `domain-vision`.

### CI Hardening

- **cargo-deny**: Supply chain audit (licenses, advisories, bans, sources)
- **IPC smoke test**: Builds primal, starts on Unix socket, sends `health.liveness`,
  verifies JSON-RPC response
- **rustfmt.toml**: Explicit formatting (edition 2024, max_width 100)

### Dead Code Evolution

- JSON-RPC `INVALID_REQUEST` / `INTERNAL_ERROR` codes wired to actual dispatch
  paths (protocol validation, `catch_unwind` for handler panics)
- `_jsonrpc` field renamed to `jsonrpc_version` (proper naming)
- `Measured` visualization variant wired to env-var-driven runtime selection

### Documentation & Provenance

- Fixed SciPy version drift (1.14.1 → 1.15.3 in tolerance comments)
- Fixed `control/README.md` provenance path
- Pinned `PUBLICATION_ENVIRONMENT` to exact versions
- Added `_provenance` to JSON baselines (`mlp_baseline.json`, `baseline_values.json`)
- Documented coral_forge shader absorption plan in `EVOLUTION_MAPPING.md`
- Flagged dirty `cpu_parity_references.json` for regeneration

## Metrics

| Metric | Value |
|--------|-------|
| Rust source files | 464 (174 lib + 290 bin) |
| `[[bin]]` entries | 261 |
| Tests | ~1,385 (1,199 lib + 72 forge + 80 playground + 9 integration + 25 tokio) |
| Clippy (pedantic+nursery, -D warnings) | 0 |
| Format diffs | 0 |
| Doc warnings | 0 |
| Unsafe blocks | 0 (`#![forbid(unsafe_code)]` workspace-wide) |
| `#[allow()]` in production | 0 |
| Max file LOC | 879 (bench_upstream_vs_local.rs) |
| barraCuda | v0.3.7, `default-features = false` |

## barraCuda Absorption Candidates

### Coral Forge Shaders — Generic Ops for Upstream

These generic activation/normalization shaders should move to `barracuda::ops`:
- `gelu_f64.wgsl` → `ops::activation`
- `sigmoid_f64.wgsl` → `ops::activation`
- `layer_norm_f64.wgsl` → `ops::norm`
- `softmax_f64.wgsl` → `ops::attention`

### Proposed: `domain-fold` Feature

AlphaFold3/structural biology shaders (triangle attention, IPA, backbone, torsion)
are too domain-specific for generic `ops` but would benefit from a named barraCuda
domain feature pack (like `domain-nn`). See `specs/EVOLUTION_MAPPING.md` Group B.

## Open Items for Next Session

1. Regenerate `cpu_parity_references.json` from clean HEAD
2. Continue `Result<T, String>` → typed error migration (remaining ~80+ sites)
3. Evolve `src/evolved/mha.rs` (last thin wrapper) toward upstream absorption
