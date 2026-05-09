# neuralSpring V144 — Deep Debt Sweep Handoff

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Session:** S194 | **Date:** May 9, 2026 | **Supersedes:** V143

## Scope

Post-eukaryotic deep debt sweep: feature gate alignment, inline test expansion,
environment variable centralization, and dependency alignment.

## Deliverables

### 1. Feature Gate Alignment

Three modules that directly import `barracuda::` crate types are now properly
gated behind `#[cfg(feature = "barracuda")]` in `src/lib.rs`:

| Module | barracuda API used |
|--------|-------------------|
| `loss_landscape` | `barracuda::sample`, `barracuda::numerical`, `barracuda::optimize` |
| `weight_spectral` | `barracuda::stats`, `barracuda::spectral` |
| `wdm_esn` | `barracuda::esn_v2`, `barracuda::tensor` |

Default build (`barracuda` feature on) is unaffected. This aligns with the
existing pattern used by `gpu`, `gpu_dispatch`, `gpu_ops`, `evolved`,
`nautilus_bridge`, `training_monitor`, `wdm_surrogate`, and `wdm_transport`.

### 2. Inline Test Expansion (+20 tests, 12 in default build)

| Module | New tests | Coverage added |
|--------|-----------|---------------|
| `ipc::tests` | 9 | Non-numeric extract, non-array extract, missing primal error paths, `with_timeout` override, `PrimalSlot` enum values, liveness report partial/zero |
| `rpc_service::tests` | 8 | Serde round-trips for all wire types (`SpectralAnalysisResult`, `DisorderSweepResult`, `HealthStatus`, `InferenceCompleteRequest` defaults, `InferenceCompleteResponse`, `InferenceEmbedRequest`, `ModelInfo` optional fields, `InferenceModelsResponse`) |
| `config::tests` | 3 | `resolve_family_id()` chain: default fallback, primary precedence, BIOMEOS fallback |

Total workspace tests: 1,295 lib + 73 forge + 80 playGround = **1,448**.

### 3. Centralized BIOMEOS_FAMILY_ID

Added to `src/config.rs`:
- `ENV_BIOMEOS_FAMILY_ID` constant (`"BIOMEOS_FAMILY_ID"`)
- `resolve_family_id()` function: `FAMILY_ID` → `BIOMEOS_FAMILY_ID` → `"default"`

Updated consumers to delegate to the central resolver:
- `playGround/src/discovery.rs` — `get_family_id()` now calls `neural_spring::config::resolve_family_id()`
- `src/bin/neuralspring_primal/discovery.rs` — same delegation

### 4. Dependency Alignment

`temp-env` version aligned to `"0.3.6"` across root and playGround `Cargo.toml`.

## Quality Gates

- `cargo build` — clean
- `cargo fmt --check` — clean
- `cargo clippy --lib` — pre-existing `doc_markdown` suggestions only
- `cargo test --lib` — 1,295 passed, 0 failed
- `cargo test -p neuralspring-playground --lib` — 80 passed

## Evolution Status

| Axis | Status |
|------|--------|
| Feature gate consistency | 11 modules now gated (was 8) |
| Inline test coverage | IPC, rpc_service, config modules now have tests |
| Hardcoded env vars | `BIOMEOS_FAMILY_ID` centralized |
| Dependency drift | `temp-env` aligned |
| Unsafe code | Zero |
| TODO/FIXME/HACK/DEBT | Zero |
| Bare `#[allow]` | Zero |
| Mocks in production | Zero |

## Next Wave Targets

- Gate remaining `barracuda`-dependent files in `bench.rs`, `visualization/scenarios/provenance.rs` for clean `--no-default-features` build
- Migrate playGround binaries off deprecated `PrimalClient` to `CompositionContext`
- Add inline tests to `certification/` modules (requires `guidestone` feature)
- Smart refactor of `tolerances/mod.rs` (665 lines) and `provenance/experiments.rs` (602 lines) if they approach 800
