# neuralSpring V178 — Wave 76 Parity Alignment Handoff

**Spring:** neuralSpring (southGate)
**Session:** S222 | **Date:** 2026-06-03
**FRAGO:** wave76-parity-sprint-springs (P1)
**Gate:** southGate (AMD Ryzen 7 5800X3D, 128GB DDR4, Pop!_OS 22.04)

---

## Mission

Parity alignment for Wave 75-76 trust infrastructure delivery. Verify
`cargo test --workspace` passes with zero warnings after absorbing
Songbird w75 capability propagation changes. Deep debt sweep.

## Test Results

| Metric | Value |
|--------|-------|
| `cargo test --workspace` | **848 passed, 0 failed, 0 warnings** |
| Lib tests (neural-spring) | 754 passed |
| Playground tests | 73 passed |
| Forge tests | 11 passed |
| Doc tests | 10 passed |
| Clippy warnings (lib + playground + forge) | **0** |
| Clippy warnings (LTEE validator binaries) | 7 (pre-existing `expect()`/`unwrap()` in assertion-style validators) |

## Songbird w75 Absorption

Songbird w75 (`8e6a3288`) changed capability propagation to a **push model**.

**Impact on neuralSpring:** None. neuralSpring compositions use
`CompositionContext::from_live_discovery_with_fallback()` for all
capability routing, which handles both push and pull models transparently.
No code references `discovery.peers` polling behavior directly. The
deprecated `discover_by_capability()` / `discover_primal()` functions
in playGround are already marked deprecated with migration guidance to
`CompositionContext`.

**Verdict:** Compatible. No code changes required.

## Changes Made

### Test Fixes (2 failures → 0)

- **MCP tool parity 43/43** — Added 8 missing tool definitions to
  `playGround/src/mcp_tools.rs` for capabilities added in S213-S214:
  `science.ltee_allele_classifier`, `science.ltee_citrate_esn`,
  `science.eigensolve`, `science.digester_anderson_coupling`,
  `science.isomorphic_reservoir`, `science.wdm_ensemble_qs`,
  `science.introgression_nn`, `science.attention_anderson`.

### Warning Elimination (30+ warnings → 0)

- **Feature-gated imports** — `crate::tolerances`, `crate::rng::Rng`,
  `crate::visualization::scenarios`, `crate::swarm_robotics`,
  `crate::regulatory_network::{GrnParams, integrate_grn}`,
  `crate::signal_integration::{OdeParams, OdeState, integrate_ode}`
  all gated behind `#[cfg(feature = "barracuda")]` to match their
  usage in barracuda-only test functions.
- **`test_gpu_lock`** — Gated behind `#[cfg(all(test, feature = "barracuda"))]`
  since all callers are in barracuda-gated GPU test modules.
- **`provenance_dispatch.rs`** — `IpcError` import gated behind
  `#[cfg(feature = "primalspring")]`.
- **Deprecated API suppression** — Added `#[allow(deprecated)]` to
  playGround client `discover()` methods and re-exports that
  intentionally use the deprecated discovery functions during migration.

### Clippy Deep Debt

- **10x `Ok(expr?)` → `expr`** — Removed needless `Ok()` wrappers in
  `ipc/{squirrel,coralreef,toadstool,skunkbat,barracuda}.rs`.
- **`cast_sign_loss`** — Fixed `discretize_trajectory` to clamp negative
  values before `as usize` cast.
- **`cast_precision_loss`** — `metrics` module gated with `#[expect]`
  (array lengths always below 2^52).
- **Doc backticks** — 13 doc comments in LTEE structs updated with
  proper backtick formatting for field dimension parameters.
- **`sort_by_key`** — Forge pipeline `sort_by` → `sort_by_key` with
  `std::cmp::Reverse`.
- **Trailing comma** — Removed from `provenance/mod.rs` format macro.
- **Unfulfilled lint expectations** — Removed stale `#[expect(clippy::cast_precision_loss)]`
  from two LTEE validator binaries.

## Files Changed

| File | Change |
|------|--------|
| `playGround/src/mcp_tools.rs` | +8 tool definitions, count 35→43 |
| `src/determinism_tests.rs` | Feature-gate imports |
| `src/provenance_dispatch.rs` | Feature-gate `IpcError` import |
| `src/isomorphic_reservoir.rs` | Feature-gate `use super::*` |
| `src/meta_population/geography.rs` | Feature-gate `Rng` import |
| `src/meta_population/mod.rs` | Feature-gate `tolerances` import |
| `src/sequence.rs` | Feature-gate `tolerances` import |
| `src/visualization/ipc_push.rs` | Feature-gate `scenarios` import |
| `src/lib.rs` | Gate `test_gpu_lock`, add metrics expect |
| `src/ipc/{squirrel,coralreef,toadstool,skunkbat,barracuda}.rs` | Remove `Ok(expr?)` |
| `src/ltee_allele_trajectory.rs` | Cast fix, doc backticks, loop style |
| `src/ltee_citrate_esn.rs` | Doc backticks |
| `src/metrics.rs` | (via lib.rs expect) |
| `src/provenance/mod.rs` | Trailing comma |
| `src/bin/validate_ltee_b{3,4}_*.rs` | Remove unfulfilled expects |
| `metalForge/forge/src/pipeline.rs` | `sort_by_key` |
| `playGround/src/{ipc_client,discovery,primal_client,coralreef_client,songbird_http,squirrel_client,toadstool_client}.rs` | Deprecated API suppression |
| `CHANGELOG.md` | S222 entry |
| `graphs/neuralspring_deploy.toml` | S222 version |
| `docs/PRIMAL_GAPS.md` | S222 header |
| `experiments/results/gap-status.json` | S222 session |

## Upstream Notes

- Songbird w75 push-model: fully compatible, no action needed
- primalSpring Wave 75 (`fc8ef4e`): new cross-gate trust validation scenarios absorbed via workspace dep
- FRAGO `wave76-parity-sprint-springs`: neuralSpring parity **COMPLETE**

## ACK

neuralSpring on southGate: parity alignment complete. 848 tests, 0 warnings,
0 failures. Songbird w75 absorbed. MCP 43/43. Ready for cross-gate validation.
