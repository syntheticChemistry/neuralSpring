# neuralSpring V118 → barraCuda / toadStool Deep Audit + Ecosystem Evolution Handoff

**Date**: March 18, 2026
**From**: neuralSpring Session 167, V118
**To**: barraCuda / toadStool teams
**License**: AGPL-3.0-or-later
**Covers**: V117–V118, Session 167
**Supersedes**: V117 Evolution Review Handoff

## Executive Summary

- **Comprehensive 15-dimension audit** of neuralSpring against all wateringHole
  ecosystem standards — zero must-fix items, 5 should-fix items executed
- **Code quality confirmed**: zero unsafe, zero unwrap in library, zero mocks in
  production, zero TODO/FIXME, all files under 1000 LOC, 1156 tests PASS
- **Hardcoding eliminated**: primal names, socket hints, capability definitions all
  evolved to constants and config files
- **ecoBin CI added**: musl + ARM cross-compile + banned C crate detection
- **Upstream absorption wired**: `WGSL_MEAN_REDUCE` re-export, `pearson_r` consolidated
- **L-BFGS evolution path documented**: pinn and loss_landscape modules ready for wiring
- Zero regressions, zero clippy warnings (pedantic+nursery), zero fmt diffs

## Part 1 — What Changed (Session 167)

### 1.1 Centralized `pearson_r` Wrapper

Three domain modules (`wdm_ensemble_qs`, `attention_anderson`, `digester_anderson`)
each had identical `pearson_r` wrappers around `barracuda::stats::correlation::pearson_correlation`.
Consolidated into `primitives::pearson_r` with GPU equivalent documentation
(`barracuda::stats::CorrelationF64`). Domain modules now re-export.

### 1.2 Display-Name Constants (`primal_names::display`)

Added 12 mixed-case display-name constants for UI/presentation contexts:

```rust
pub mod display {
    pub const BARRACUDA: &str = "barraCuda";
    pub const TOADSTOOL: &str = "toadStool";
    pub const CORALREEF: &str = "coralReef";
    // ... 9 more
}
```

Replaced 20+ hardcoded owner strings in `industry_coverage.rs`, label strings
in `kokkos_parity.rs`, and socket hints in `coralreef_bridge.rs`.

### 1.3 Fossil `#[allow()]` → `#[expect(reason)]`

All 24 `#[allow()]` attributes in `metalForge/fossils/` (8 files) evolved to
`#[expect()]` with descriptive reasons. The workspace now has **zero `#[allow()]`
in any code** — fossils included.

### 1.4 ecoBin Cross-Compile CI

New CI job verifies ecoBin compliance:
- `cargo check --target x86_64-unknown-linux-musl` (pure Rust, zero C deps)
- `cargo check --target aarch64-unknown-linux-gnu` (ARM cross-compile)
- Banned C sys crate detection (`openssl-sys`, `ring`, `aws-lc-sys`, etc.)

### 1.5 Capability Registry TOML

Created `config/capability_registry.toml` — canonical capability definitions
per SPRING_AS_NICHE_DEPLOYMENT_STANDARD. 16 capabilities with descriptions.
Sync-tested against `config::ALL_CAPABILITIES` in Rust.

### 1.6 Upstream `WGSL_MEAN_REDUCE` Re-export

`metalForge/forge/src/shaders.rs` now re-exports:
- `MEAN_REDUCE_UPSTREAM` from `barracuda::ops::WGSL_MEAN_REDUCE`
- `MEAN_REDUCE_F64_UPSTREAM` from `barracuda::ops::WGSL_MEAN_REDUCE_F64`

Local `MEAN_REDUCE` retained for 7 validated pipeline binaries (different
binding layout). New GPU pipelines should prefer the upstream variant.

## Part 2 — What barraCuda / toadStool Should Absorb

### 2.1 `display` Name Constants Pattern

neuralSpring's `primal_names::display` module provides canonical display names.
Other springs would benefit from a shared pattern — consider adding display-name
constants to the wateringHole standards.

### 2.2 Capability Registry TOML Pattern

The `config/capability_registry.toml` pattern (TOML file with method names and
descriptions, sync-tested against Rust constants) is reusable across springs.
Consider standardizing this as the canonical capability definition format.

### 2.3 `WGSL_MEAN_REDUCE` Binding Layout Alignment

The local `mean_reduce.wgsl` (single-workgroup, `@binding(2) var<uniform>`)
differs from upstream `WGSL_MEAN_REDUCE` (shared-memory tree reduction).
Aligning the binding layout would allow pipeline binaries to migrate fully
to the upstream shader.

### 2.4 L-BFGS Wiring (Tier A — API exists, wiring effort only)

`barracuda::optimize::lbfgs` (CPU) and `LbfgsGpu` are available.
neuralSpring wiring points:
- **pinn**: PDE residual minimization (Burgers equation collocation)
- **loss_landscape**: gradient-based descent for transition-state analysis

### 2.5 `StatefulPipeline` API Documentation

HMM forward/backward/Viterbi chains and ODE integration loops would benefit
from `staging::StatefulPipeline` for persistent GPU state across iterations.
API documentation (trait bounds, state lifecycle) would unblock wiring.

### 2.6 FMA Adoption Completion

neuralSpring swept 14 `a * b + c` → `mul_add()` sites (S165). barraCuda
Sprint 7 independently swept 10 sites. Recommend systematic sweep across
`stats::`, `linalg::`, `numerical::` CPU reference paths.

## Part 3 — Audit Findings Relevant to Ecosystem

### 3.1 Inline Tolerance False Positive

The audit initially flagged ~80+ inline tolerance literals. On deep inspection,
**all are in `#[cfg(test)]` blocks or doc tests** — production validation paths
exclusively use `tolerances::*` constants. The centralization is complete.

### 3.2 Population vs Sample Variance

`gpu_dispatch::cpu_fallback` uses population variance (÷N) while barraCuda uses
sample variance (÷(N-1)). This is **intentional** — CPU fallback matches the
Python baseline behavior. Documented in `specs/BARRACUDA_USAGE.md`.

### 3.3 Kokkos Baselines Still Estimated

`bench_kokkos_parity` has 9 ops but baselines are from groundSpring V100 handoff
estimates. Verified provenance requires matched-hardware runs. The benchmark
harness is ready — it just needs real Kokkos-CUDA numbers.

### 3.4 Primal Folding MatMul Already Delegates

The audit flagged `neuralspring_primal/folding.rs` local matmul as potential
duplication. On inspection, `matmul_3d` and `matmul_2d` already delegate to
`barracuda::dispatch::matmul_dispatch` with CPU fallback only for resilience.

## Part 4 — Quality Metrics

| Category | Count |
|----------|-------|
| Library unit tests | 1156 (+1) |
| playGround unit tests | 75 |
| Forge unit tests | 73 |
| Integration tests | 9 |
| Property tests | 28 |
| IPC fuzz tests | 5 |
| Named tolerances | 225 |
| Validation binaries | 238 |
| Benchmark binaries | 18 |
| **Total test artifacts** | **1304 lib + 267 binaries** |

Quality: zero clippy (pedantic+nursery), zero fmt diffs, zero unsafe, zero C
deps, zero `#[allow()]` (workspace-wide), zero hardcoded primal names, MSRV
1.87, Edition 2024. ecoBin CI verified.

## Files Changed (S167)

| File | Change |
|------|--------|
| `src/primitives.rs` | +`pearson_r` wrapper |
| `src/primal_names.rs` | +`display` module (12 constants) |
| `src/wdm_ensemble_qs.rs` | Local `pearson_r` → re-export |
| `src/attention_anderson.rs` | Local `pearson_r` → re-export |
| `src/digester_anderson.rs` | Local `pearson_r` → re-export |
| `src/visualization/scenarios/industry_coverage.rs` | Hardcoded → `display::*` |
| `src/visualization/scenarios/kokkos_parity.rs` | Hardcoded → `display::BARRACUDA` |
| `metalForge/forge/src/coralreef_bridge.rs` | `CORALREEF_NAME` constant |
| `metalForge/forge/src/shaders.rs` | +upstream mean reduce re-exports |
| `metalForge/fossils/` (8 files) | `#[allow()]` → `#[expect(reason)]` |
| `.github/workflows/rust.yml` | +ecoBin cross-compile job |
| `config/capability_registry.toml` | New — 16 capabilities |
| `src/config.rs` | Registry reference + sync test |
| `src/pinn.rs` | +L-BFGS evolution doc |
| `src/loss_landscape.rs` | +L-BFGS evolution doc |
| `src/bin/bench_kokkos_parity.rs` | Provenance restructured |
| Root docs | S167 entries |

---

*AGPL-3.0-or-later — neuralSpring → barraCuda / toadStool*
