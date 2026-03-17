# neuralSpring V112 — S161 Doc Cleanup + Structured Logging Handoff

**Date**: March 16, 2026
**From**: neuralSpring S161
**Scope**: Hardcoded path elimination, playGround structured logging, doc sync

## Changes

### 1. Hardcoded Path Elimination

Three locations replaced with centralized `config::` constants:

| File | Before | After |
|------|--------|-------|
| `src/bin/neuralspring_primal/main.rs` | `"biomeos/biomeos.sock"` | `config::BIOMEOS_SOCKET_SUBDIR` / `BIOMEOS_ORCHESTRATOR_SOCKET` |
| `playGround/src/biomeos_client.rs` | `"biomeos.sock"` | `neural_spring::config::BIOMEOS_ORCHESTRATOR_SOCKET` |
| `playGround/src/ipc_client.rs` | duplicate `BIOMEOS_SOCKET_SUBDIR` | delegates to `neural_spring::config::BIOMEOS_SOCKET_SUBDIR` |

### 2. Structured Logging Completion

28 `eprintln!` calls replaced with `log::info!/warn!/debug!` across playGround binaries:

| Binary | eprintln → log | Pattern |
|--------|----------------|---------|
| `neuralspring_mcp_adapter` | 18 calls | Startup → `info!`, health/capability warnings → `warn!` |
| `neuralspring_interactive` | 22 calls | User prompts → `println!`, status → `info!`, errors → `error!` |
| `neuralspring_bench_inference` | 5 calls | Setup → `info!`, bench output stays `println!` |
| `biomeos_client.rs` | 1 call | Non-fatal → `warn!` |

**Result**: Zero `eprintln!` remaining in playGround src. All server/library logging is
RUST_LOG-controllable.

### 3. Doc Consolidation

README session history consolidated: S155–S158 condensed into two-line summaries.
All root docs (README, CHANGELOG, EVOLUTION_READINESS, CONTROL_EXPERIMENT_STATUS,
DEPRECATION_MIGRATION), baseCamp, experiments synced to S161.

## Quality Gates

| Metric | Value |
|--------|-------|
| Library tests | 1128 |
| playGround tests | 61 |
| Forge tests | 73 |
| Clippy warnings | 0 (pedantic + nursery) |
| `eprintln!` in playGround | 0 |
| Hardcoded socket paths | 0 |
| `#[allow()]` | 0 |
| Unsafe blocks | 0 |
| C dependencies | 0 |

## Supersedes

V111 (S160 IPC Evolution) — archived.
