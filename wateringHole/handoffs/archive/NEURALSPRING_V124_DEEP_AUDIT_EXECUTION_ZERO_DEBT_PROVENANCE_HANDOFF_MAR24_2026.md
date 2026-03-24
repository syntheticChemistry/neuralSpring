<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V124 — Deep Audit Execution: Zero Debt, Tolerance Fidelity, Self-Knowledge

**Date**: March 24, 2026 (Session S174)
**From**: neuralSpring
**To**: barraCuda / toadStool / ecosystem
**License**: AGPL-3.0-or-later
**Supersedes**: V123 (archived)

## Summary

Session S174 executed the complete action plan from a comprehensive ecosystem
audit against wateringHole standards. Every finding has been resolved: zero
`#[allow()]` in any code path, all tolerance literals replaced with named
constants, primal self-knowledge enforced, provenance headers on all 49 Python
baselines, and community documentation added.

## Changes

### Zero `#[allow()]` — Complete

| File | Before | After |
|------|--------|-------|
| `src/error.rs` | `#![allow(missing_docs)]` blanket | Removed; 20 field doc comments added |
| `src/bin/neuralspring_primal/rpc.rs` | `#[allow(dead_code)]` | `#[expect(dead_code, reason = "...")]` |
| 5× `metalForge/fossils/**/*.rs` | `#[allow(clippy::*)]` | `#[expect(clippy::*, reason = "...")]` |

### Tolerance Fidelity — All Literals Centralized

- **`check_rel` alignment**: `f64::EPSILON` → `tolerances::ZERO_DETECTION` in `validation/mod.rs`
- **New constant**: `GPU_MULTI_OBJ_BESSEL_F64` (3e-3) in `tolerances/gpu.rs` — documents the
  algorithmic difference (sample vs population std) between GPU `MultiObjFitnessGpu` and CPU
  `multi_objective_fitness`. Validated on RTX 4070 + llvmpipe, seeds 42/77/123.
- **4 upstream contract constants** in `tolerances/mod.rs`:
  - `UPSTREAM_HYDRO_CROP_COEFFICIENT` (1e-6)
  - `UPSTREAM_PHYSICS_ANDERSON_EIGENVALUE` (1e-10)
  - `UPSTREAM_BIO_DIVERSITY_SHANNON` (1e-10)
  - `UPSTREAM_BIO_DIVERSITY_SIMPSON` (1e-10)
- **New registry category**: `upstream_contract` (5 entries in `tolerances/registry.rs`)
- **Literal replacement**: all `2e-3` in `validate_gpu_directed.rs` and
  `validate_gpu_pipeline_directed.rs` → `tolerances::GPU_MULTI_OBJ_BESSEL_F64`
- **Centralized toadstool s93**: local `upstream_expected` module in
  `validate_toadstool_s93_barracuda_extraction.rs` → centralized `tolerances::UPSTREAM_*`

### Self-Knowledge Compliance

- **Removed** 3 dead `*_NAME_HINT` constants from `config.rs` (`TOADSTOOL_NAME_HINT`,
  `CORALREEF_NAME_HINT`, `SQUIRREL_NAME_HINT`) — violated self-knowledge principle
- **Neutralized** cross-spring origin strings in `handlers.rs` RPC responses:
  `"hotSpring precision → barraCuda → Dispatcher"` → `"stats.variance → dispatch"`
- **Gated** petalTongue visualization push behind `NEURALSPRING_VISUALIZATION_PUSH`
  env var — no longer unconditional coupling to another primal
- **Updated** playGround clients (`coralreef_client.rs`, `squirrel_client.rs`,
  `toadstool_client.rs`) to use `primal_names::*` directly

### Provenance Alignment

- Added `# Provenance: see src/provenance/experiments.rs` header to all 49
  registered Python baseline scripts in `control/`

### Community Documentation

- **`CONTRIBUTING.md`**: prerequisites, quality standards (clippy, fmt, doc, coverage,
  `forbid(unsafe_code)`, `#[expect]`, file size, SPDX), tolerance policy, validation
  binaries (hotSpring pattern), barraCuda evolution, IPC conventions
- **`SECURITY.md`**: security model (Pure Rust, `cargo-deny`, no vendor lock-in,
  IPC isolation), vulnerability reporting, dependency audit, data provenance

### Code Quality

- `map_or` → `is_ok_and` (modern idiomatic Rust, caught by clippy)

## Metrics

| Metric | Value |
|--------|-------|
| Rust source files | 464 (174 lib + 290 bin) |
| `[[bin]]` entries | 261 |
| Tests | ~1,385 (1,199 lib + 72 forge + 80 playGround + 9 integration + 25 tokio) |
| Named tolerances | 232+ (was 227+) |
| Clippy (pedantic+nursery, -D warnings) | 0 |
| Format diffs | 0 |
| Doc warnings | 0 |
| Unsafe blocks | 0 (`#![forbid(unsafe_code)]` workspace-wide) |
| `#[allow()]` in production | 0 |
| `#[allow()]` in fossils | 0 (all `#[expect()]` with reasons) |
| Tolerance literals in validators | 0 (all named constants) |
| Python scripts with provenance headers | 49/49 |
| barraCuda | v0.3.7, `default-features = false` |

## barraCuda Upstream Contract

S174 introduced explicit contract pinning for 4 barraCuda tolerance values.
These constants are verified at validation time — if barraCuda silently changes
a published tolerance, neuralSpring's `validate_toadstool_s93_barracuda_extraction`
binary will catch it. This pattern should be adopted by other springs.

## Open Items for Next Session

1. Regenerate `cpu_parity_references.json` from clean HEAD
2. Continue `Result<T, String>` → typed error migration (remaining ~80+ sites)
3. Evolve `src/evolved/mha.rs` (last thin wrapper) toward upstream absorption
4. Generic f64 ops absorption (gelu, sigmoid, layer_norm, softmax) — pending barraCuda
5. `domain-fold` feature pack — pending barraCuda decision
