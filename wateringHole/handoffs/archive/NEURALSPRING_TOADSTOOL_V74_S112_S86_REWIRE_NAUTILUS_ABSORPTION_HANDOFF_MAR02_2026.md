<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/BarraCUDA Handoff V74 — S86 Rewire + Nautilus Absorption

**Date**: March 2, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Session 112 — ToadStool S86 pin bump, nautilus dependency absorption, DriftMonitor API migration
**Supersedes**: V73 (Paper Queue Validation & GPU Pyramid Complete)
**ToadStool HEAD**: `2fee1969` (S86)

---

## Executive Summary

- **ToadStool pin bumped**: `f97fc2ae` (S79) → `2fee1969` (S86) — 7 commits absorbed
- **Nautilus dependency eliminated**: `bingocube-nautilus` path dep removed; now using `barracuda::nautilus` directly
- **DriftMonitor API migrated**: `record(epoch, pop_size, mean, best)` → `record(&GenerationRecord, pop_size)`
- **DriftMonitor fields evolved**: `history` → `ne_s_history`, `consecutive_drift` removed
- **208/208 validate_all PASS** (was 207/207), 861 lib tests, 0 clippy, 0 fmt
- **New validator**: `validate_toadstool_s86_rewire` (27/27 PASS)

---

## Part 1: What ToadStool Evolved (S79 → S86)

| Session | Key Changes |
|---------|-------------|
| S80 | `barracuda::nautilus` absorbed (7 files, 22 tests). `BatchedEncoder` for multi-op GPU. `fused_mlp`. Batch Nelder-Mead GPU. `StatefulPipeline<S>`. Driver workarounds (sin/cos F64). ComputeDispatch 76→95 |
| S81-82 | Deep debt evolution. ComputeDispatch +16 ops (95→111). OS memory detection. `creation.rs` DRY refactor |
| S84-86 | ComputeDispatch +33 ops (111→144). `hydrology.rs` split to module (CPU/GPU). Experimental stub → real probes. Root docs sweep |

**Net**: 131 files changed, +8199/-8774 lines. ComputeDispatch 76→144 (+68 ops). New dep: `blake3 = "1.5"`.

---

## Part 2: What neuralSpring Rewired

### 2.1 Nautilus Absorption

| Before | After |
|--------|-------|
| `bingocube-nautilus = { path = "../primalTools/bingoCube/nautilus" }` | Dependency removed |
| `use bingocube_nautilus::{...}` | `use barracuda::nautilus::{...}` |
| External crate (primalTools) | BarraCUDA-native (upstream absorption) |

Files changed:
- `Cargo.toml` — dep removed
- `src/nautilus_bridge.rs` — 3 import lines rewired
- `src/training_monitor.rs` — 1 import line rewired

### 2.2 DriftMonitor API Migration

| Before (bingoCube) | After (barracuda) |
|---------------------|-------------------|
| `drift.record(epoch, pop_size, mean_fitness, best_fitness)` | `drift.record(&GenerationRecord{...}, pop_size)` |
| `monitor.history.is_empty()` | `monitor.ne_s_history.is_empty()` |
| `monitor.consecutive_drift == 0` | `!monitor.is_drifting()` |

Files changed:
- `src/training_monitor.rs` — `record()` call adapted to `GenerationRecord`
- `src/bin/validate_nautilus_bridge.rs` — field access updated

### 2.3 New Validator

`validate_toadstool_s86_rewire` (27/27 PASS):
- `NautilusBrain` creation and lifecycle (3 checks)
- `NautilusShell` creation from seed (2 checks)
- `EvolutionConfig` + `SelectionMethod` (1 check)
- `DriftMonitor` lifecycle: empty → record → drift detection (5 checks)
- `SpectralNautilusBridge` via `barracuda::nautilus`: observe → train → predict → screen → serialize roundtrip (12 checks)
- `BetaObservation` field compatibility (4 checks)

---

## Part 3: New ToadStool Capabilities Available

| Capability | Module | neuralSpring Status |
|------------|--------|---------------------|
| `BatchedEncoder` | `barracuda::device` | Available — single CommandEncoder for multi-op GPU pipelines |
| `fused_mlp` | `barracuda::nn` | Available — MLP forward via BatchedEncoder |
| Batch Nelder-Mead GPU | `barracuda::optimize` | Available — N parallel simplex optimizations |
| Brent GPU optimizer | `barracuda::optimize` | Available — scalar root-finding |
| L-BFGS optimizer | `barracuda::optimize` | Available — quasi-Newton optimization |
| Richards PDE GPU | `barracuda::pde` | Available — soil moisture modeling |
| `StatefulPipeline<S>` | `barracuda::pipeline` | Available — day-over-day state tracking |
| Multi-GPU interconnect | `barracuda::multi_gpu` | Available — bandwidth modeling + pipeline dispatch |
| Hydrology extensions | `barracuda::stats::hydrology` | Available — `thornthwaite_et0`, `makkink_et0`, `turc_et0`, `hamon_et0` |
| Anderson acceleration | `barracuda::spectral::anderson` | Available — iterative solver |
| ComputeDispatch 144 ops | throughout | Available — 68 more ops via dispatch |

---

## Appendix: Validation State

| Metric | Before (V73) | After (V74) |
|--------|-------------|-------------|
| ToadStool HEAD | `f97fc2ae` (S79) | `2fee1969` (S86) |
| validate_all | 207/207 | **208/208** |
| lib tests | 861 | 861 |
| clippy warnings | 0 | 0 |
| External deps removed | — | `bingocube-nautilus` |
| New validators | — | `validate_toadstool_s86_rewire` (27/27) |
