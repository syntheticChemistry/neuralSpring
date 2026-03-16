# neuralSpring V109 — Cross-Ecosystem Absorption + Deep Debt Handoff

**Session**: S158 (2026-03-16)
**Previous**: V108/S157 (Deep Debt + Idiomatic Rust + Tower Atomic)
**barraCuda**: v0.3.5 at `0649cd0`

## Executive Summary

S158 absorbed patterns from all 6 sibling springs and key primals into neuralSpring:

- **wetSpring `#[expect(reason)]` audit**: Zero `#[allow()]` in active code, zero unfulfilled
  lint expectations across `--all-targets`, stale expectations pruned
- **petalTongue/ludoSpring `temp-env` pattern**: 26 `set_var`/`remove_var` → `temp_env::with_var`
  (Rust 2024 safe env var testing)
- **airSpring primal name constants**: All discovery paths use `primal_names::*` or `config::*`
  constants, zero hardcoded primal strings
- **groundSpring zero-panic audit**: All `unwrap()` in library code confirmed test-only, bare
  `unwrap()` → `expect("context")` in diagnostic binary
- **Smart refactoring**: `validate_barracuda_tensor.rs` 918→875 LOC via `check_binary_op` +
  `check_scalar_op` helper extraction

## Quality Gates

| Gate | Status |
|------|--------|
| lib tests | **1128 pass** |
| playGround tests | **61 pass** |
| clippy (pedantic+nursery, -D warnings) | **0 warnings** |
| unfulfilled lint expectations | **0** |
| `#[allow()]` in active code | **0** |
| `#![forbid(unsafe_code)]` | **3/3 crate roots** |
| hardcoded primal names in discovery | **0** |
| production mocks | **0** |
| C dependencies | **0** (blake3/cc is build-time via barraCuda) |
| fmt diffs | **0** |

## Changes Detail

### Lint Evolution

- `src/tolerances/registry.rs`: `#[allow(wildcard_imports)]` → `#[cfg_attr(not(test), expect(...))]`
  to handle cross-cfg boundary (wildcard import lint not triggered under `--test`)
- `src/bin/diagnose_f64_regression.rs`: Removed stale `clippy::unwrap_used` after evolving
  `unwrap()` → `expect("tokio runtime")`

### Environment Safety

- `playGround/Cargo.toml`: Added `temp-env = "0.3.6"` as dev dependency
- `playGround/src/ipc_client.rs`: 9 tests refactored from manual save/restore to
  `temp_env::with_var`/`temp_env::with_vars`
- `playGround/src/biomeos_client.rs`: 1 test refactored to `temp_env::with_var`

### Smart Refactoring

- `src/bin/validate_barracuda_tensor.rs`: Extracted `check_binary_op()` and `check_scalar_op()`
  helpers to consolidate the repeated alloc→dispatch→readback→check pattern
  (918→875 LOC, -43 lines, zero coverage loss)

### Hardcoded Names → Constants

- `src/bin/neuralspring_primal/discovery.rs`: 3× `"biomeos"` → `config::BIOMEOS_SOCKET_SUBDIR`
- `playGround/src/songbird_http.rs`: `SONGBIRD_HINT` → `primal_names::SONGBIRD`
- `playGround/src/primal_client.rs`: `"neuralspring"` → `niche::NICHE_NAME`

## Remaining Evolution Items (for future sessions)

| Item | Source | Priority |
|------|--------|----------|
| toadStool pin S146 → S155b | healthSpring handoff | Medium |
| biomeOS typed `CapabilityClient` SDK | biomeOS SDK | Medium |
| Content Convergence experiment (ISSUE-013) | sweetGrass | Low |
| `WGSL_MEAN_REDUCE` as public constant request | barraCuda team | Low |
| `head_split`/`head_concat` unification | barraCuda team | Low |
| `StatefulPipeline` for HMM/ODE | toadStool team | Low |
| blake3 `pure` feature (eliminate cc build dep) | barraCuda team | Low |
| Pipeline graph streaming (pharmacology, FASTQ) | biomeOS SDK | Low |
