# NEURALSPRING V122 — Deep Evolution & Ecosystem Absorption Handoff

**Date:** 2026-03-23  
**Session:** S172 (V122)  
**barraCuda:** v0.3.7  
**License:** AGPL-3.0-or-later  

## Summary

Deep debt resolution and ecosystem convergence session. Completed the
`DeviceCapabilities` migration (last spring to finish), unified workspace
lint governance, resolved all 163 playGround missing-docs errors, absorbed
`normalize_method` from barraCuda for IPC, refactored the three largest
validation binaries by responsibility, and centralized all hardcoded env
var names into `config.rs`.

## Changes

### DeviceCapabilities Migration (ecosystem convergence)

Migrated all `GpuDriverProfile` usage to `DeviceCapabilities` from
`barracuda::device::capabilities`. neuralSpring was the last spring with
active deprecated usage. Sites migrated:

- `src/gpu_dispatch/mod.rs` — core `Dispatcher` struct and all accessors
  (`fp64_strategy`, `precision_routing`, `needs_pow_workaround`,
  `check_allocation_safe`)
- `src/bin/diagnose_f64_regression.rs` — diagnostic binary
- `src/bin/validate_coral_forge_gpu.rs` — coral forge GPU validation
- `src/bin/validate_coral_forge_gpu_pipeline.rs` — pipeline validation
- `src/bin/validate_cross_spring_evolution.rs` — renamed `validate_driver_profile`
  → `validate_device_capabilities`
- `src/bin/validate_sovereign_compile.rs` — precision strategy display
- `src/bin/validate_gpu_eigensolve_pipeline.rs` — eigensolve diagnostics
- `src/bin/validate_toadstool_s79_rewire.rs` — type path update
- `src/gpu_dispatch/tests_cpu_metadata.rs` — test references
- `src/gpu_dispatch/tests_gpu.rs` — GPU integration test
- `src/lib.rs` — removed deprecated `#[expect]` on `gpu_dispatch` module

**Migration map:**
- `GpuDriverProfile::from_device(dev)` → `DeviceCapabilities::from_device(dev)`
- `.fp64_strategy()` → same (available on both types)
- `.precision_routing()` → same
- `.needs_pow_f64_workaround()` → `.needs_exp_f64_workaround()` (per-builtin granularity)
- `.f64_zeros_risk()` → `.has_reliable_f64()` (inverted semantics)
- `.check_allocation_safe()` → same (available on `DeviceCapabilities`)

### Workspace Lint Inheritance

Moved `[lints.rust]` and `[lints.clippy]` to `[workspace.lints]` in root
`Cargo.toml`. All three workspace members (`neural-spring`, `neural-spring-forge`,
`neuralspring-playground`) now inherit via `[lints] workspace = true`.

Aligns with wetSpring V132, healthSpring V41, groundSpring V120 pattern.

### playGround Missing-Docs Resolution

Added doc comments to all 163 previously-undocumented public items in
`playGround/src/` including `toadstool_client.rs`, `biomeos_client.rs`,
`coralreef_client.rs`, `hf_hub.rs`, `ipc_client.rs`, `mcp_tools.rs`,
`model_config.rs`, `primal_client.rs`, `secrets.rs`, `songbird_http.rs`,
`squirrel_client.rs`, and `inference/` modules.

Result: `cargo clippy --workspace --all-features -- -D warnings` passes
with **zero warnings** across all four crates.

### `normalize_method` Absorption

Absorbed the `normalize_method` pattern from `barracuda-core::ipc::methods`
into `src/bin/neuralspring_primal/rpc.rs`. The dispatch function now
normalizes incoming JSON-RPC method names, stripping legacy `neuralspring.`
prefix for backward compatibility. Aligns with wetSpring V132, loamSpine
v0.9.8, barraCuda v0.3.7 IPC conventions.

### Smart Refactoring (Responsibility-Based)

Three validation binaries approaching the 1000 LOC ecosystem limit were
refactored into multi-module binaries by domain responsibility:

| Binary | Before | After (max module) |
|--------|--------|--------------------|
| `validate_gpu_pure_workload_all` | 942 LOC | 7 modules, max 209 LOC |
| `validate_cross_spring_evolution` | 913 LOC | 9 modules, max 189 LOC |
| `validate_modern_cross_spring` | 900 LOC | 10 modules, max 137 LOC |

Largest file in `src/` is now 879 LOC (`bench_upstream_vs_local.rs`).

### #[allow] → #[expect] in Forge

Converted `metalForge/forge/src/mixed.rs` from `#![allow(clippy::cast_precision_loss)]`
to `#![expect(clippy::cast_precision_loss, reason = "...")]`. Zero `#[allow]`
in production code across the entire workspace.

### Config Centralization

- Centralized env var names (`ENV_TCP_PORT`, `ENV_IPC_TIMEOUT`,
  `ENV_HEARTBEAT_SECS`, `ENV_FAMILY_ID`, `ENV_GPU_BACKEND_LEGACY`) in
  `src/config.rs`
- Replaced `"127.0.0.1"` with `std::net::Ipv4Addr::LOCALHOST` in TCP binding
- Updated primal binary to use config constants for all env var lookups

### Doctest Fix

Fixed `nucleus_pipeline.rs` doctest — `execute_composition_pipeline()` returns
`Result`, doctest now calls `.expect()` on the result.

## Quality Gates

| Gate | Status |
|------|--------|
| `cargo fmt --all --check` | **PASS** |
| `cargo clippy --workspace --all-features -- -D warnings` | **PASS** (zero warnings) |
| `cargo doc --workspace --all-features --no-deps` | **PASS** (zero warnings) |
| `cargo test --workspace` | **PASS** (1,300 tests: 1,203 lib + 73 forge + 2 playground + 13 doctests, 18 ignored) |
| `#![forbid(unsafe_code)]` | All 3 crates |
| `#[allow()]` in production | Zero |
| Files > 1000 LOC | Zero (max 879) |

## Upstream Asks

### barraCuda (P1)

- **`barracuda::cast`** — shared safe-cast helpers duplicated across springs.
  groundSpring V120 and airSpring v0.10 both request absorption.
- **`MultiHeadEsn` device accessor** — workaround is separate `Arc<WgpuDevice>`.
- **Stable shader catalog** — name + content hash for 41 metalForge WGSL shaders.

### Ecosystem (tracking)

- **loamSpine v0.10** roadmap names neuralSpring for "collision layer validation"
- **`ValidationSink`** — pluggable validation output (wetSpring V132 pattern);
  consider ecosystem-wide absorption

## Absorbed From Other Springs

| Pattern | Source | Applied In |
|---------|--------|------------|
| `DeviceCapabilities` migration | airSpring v0.10, wetSpring V132, groundSpring V120 | `gpu_dispatch/mod.rs` + 10 files |
| Workspace lint inheritance | wetSpring V132, healthSpring V41 | Root `Cargo.toml` |
| `normalize_method` IPC | barraCuda v0.3.7, loamSpine v0.9.8 | `neuralspring_primal/rpc.rs` |
| Responsibility-based binary refactoring | groundSpring V120 dispatch split | 3 validation binaries |
| Env var centralization | healthSpring V41, biomeOS v2.66 | `config.rs` + primal binary |
