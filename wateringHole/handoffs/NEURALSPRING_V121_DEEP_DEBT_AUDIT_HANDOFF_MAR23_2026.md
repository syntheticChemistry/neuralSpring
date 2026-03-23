# neuralSpring V121 — Deep Debt Audit Execution Handoff

**Date:** March 23, 2026
**From:** neuralSpring S171
**To:** barraCuda team, toadStool team, ecosystem
**License:** AGPL-3.0-or-later

---

## Executive Summary

neuralSpring S171 completed a deep debt audit and execution pass: typed error
evolution (`PipelineError`), named numerical constants, lint parity across
workspace, version ref freshening (v0.3.5→v0.3.7), 6 new proptests, doc
warning elimination. **1,356 tests** (1,203 lib + 73 forge + 80 playGround),
0 clippy, 0 fmt, 0 doc warnings, 0 unsafe.

---

## Changes This Session (S171)

### Code Quality Evolution

| Change | Impact |
|--------|--------|
| `nucleus_pipeline.rs`: `.expect()` → `Result<_, PipelineError>` | Library boundary no longer panics; typed `CyclicGraph`/`MissingStage` errors |
| `POSITIVE_DATA_GUARD` (1e-10) + `R2_DENOMINATOR_FLOOR` (1e-30) | 9 binaries wired; zero inline guard literals remaining |
| metalForge forge: `unwrap_used` + `expect_used` lints | Parity with root crate lint strictness |
| 2 bench_* removed from `validate_all.rs` | Clean separation: validation ≠ benchmarking |
| 2 doc link fixes | Zero doc warnings on main crate |
| `validate_immunological_anderson` added to `Cargo.toml` `[[bin]]` | Explicit declaration (was autobins-only) |

### Proptest Expansion (6 new)

| Module | Property |
|--------|----------|
| FASTQ (`streaming/fastq.rs`) | Write-then-parse roundtrip fidelity |
| FASTQ | Sequence length ≡ quality length |
| VCF (`streaming/vcf.rs`) | Parsed position always positive |
| VCF | Chrom name preserved through parse |
| WDM surrogate (`wdm_surrogate.rs`) | Predictions always finite |
| WDM surrogate | Predictions deterministic (same input → same output) |

### Documentation Freshening

- barraCuda version refs: v0.3.5 → v0.3.7 across 4 specs + ABSORPTION_TRACKER
- EVOLUTION_READINESS.md → Session 171, 1,356 tests, 227 named tolerances
- CONTROL_EXPERIMENT_STATUS.md → March 23, 2026
- DATA_PROVENANCE.md: baseCamp composition experiments (Exp-096..106) + sub-theses (nS-01..06)
- README.md, CHANGELOG.md, CONTEXT.md, experiments/README.md refreshed

---

## Upstream Asks (barraCuda)

### P0: `DeviceCapabilities` Migration Guide

neuralSpring has **6 `#[expect(deprecated)]` sites** pending the `GpuDriverProfile` → `DeviceCapabilities` migration:

| File | Usage |
|------|-------|
| `src/lib.rs` | Crate-level `gpu_dispatch` module suppress |
| `src/gpu_dispatch/mod.rs` | Core runtime: `fp64_strategy()`, `precision_routing()`, `needs_pow_f64_workaround()`, `f64_zeros_risk()` |
| `src/bin/validate_cross_spring_evolution.rs` | Profile validation |
| `src/bin/diagnose_f64_regression.rs` | Diagnostic |
| `src/bin/validate_coral_forge_gpu.rs` | GPU pipeline |
| `src/bin/validate_coral_forge_gpu_pipeline.rs` | GPU pipeline |

**Request:** Publish documented equivalents for all four `GpuDriverProfile` methods on `DeviceCapabilities`, with timeline for deprecation removal.

### P1: `MultiHeadEsn` Device Accessor

neuralSpring works around the removed `MultiHeadEsn::wgpu_device()` by storing a separate `Arc<WgpuDevice>`. Consider re-exposing a device accessor for downstream ergonomics.

### P2: Shader Catalog Stability

41 WGSL shaders in `metalForge/shaders/` track provenance via `ABSORPTION_TRACKER.md`. A stable upstream shader catalog (names + content hashes) would let consumers verify provenance mechanically rather than via comments.

---

## neuralSpring API Surface Consumed

| Category | Count |
|----------|-------|
| barraCuda submodules used | 45+ |
| barraCuda functions imported | 80+ |
| Files with barraCuda imports | 216 |
| Upstream rewires (local → barraCuda) | 46 |
| Local WGSL shaders | 41 |
| Named tolerances | 227 |
| Property tests | 34 |
| Validation binaries | 267 |

---

## Patterns Worth Sharing

### `PipelineError` typed error evolution

```rust
pub enum PipelineError {
    CyclicGraph { pipeline: String },
    MissingStage { stage_id: String, pipeline: String },
}
```

Replaced `.expect("graph must be a valid DAG")` with `?` propagation. All callers handle the `Result` — binaries use `.expect()` at the entry point (under `#[expect(clippy::expect_used)]`), tests use `.expect()` in `#[cfg(test)]`. Library code has zero panicking paths.

### Named guard constants pattern

Rather than scattering `1e-10` / `1e-30` across binaries, centralize in `primitives.rs` with IEEE 754 justification:

```rust
pub const POSITIVE_DATA_GUARD: f64 = 1e-10;  // ensure positivity for safe log/division
pub const R2_DENOMINATOR_FLOOR: f64 = 1e-30; // prevent zero in R² SS_tot
```

This is the same pattern as `LOG_GUARD` (1e-300) and `DIVISION_GUARD` (1e-15) — domain-appropriate precision levels with documented derivation.

---

## Test Totals

| Category | Count |
|----------|-------|
| Python baselines | 397/397 PASS |
| Rust lib tests | 1,203 |
| Forge tests | 73 |
| playGround tests | 80 |
| Validation binaries | 267 |
| Property tests (proptest) | 34 |
| Grand total checks | 4,900+ |
