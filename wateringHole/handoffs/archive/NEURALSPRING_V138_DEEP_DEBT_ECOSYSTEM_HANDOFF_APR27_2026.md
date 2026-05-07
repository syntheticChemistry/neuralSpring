# neuralSpring V138 — Deep Debt Cleanup + Ecosystem Handoff

**Date**: April 27, 2026
**Session**: S187 (V138)
**From**: neuralSpring
**For**: primalSpring, BarraCUDA, ToadStool, coralReef, Squirrel, all primal teams
**primalSpring**: v0.9.17+
**guideStone**: v0.3.0 / standard v1.2.0 / Level 3

---

## Summary

neuralSpring's codebase has undergone comprehensive deep debt cleanup and is
ready for upstream audit. This handoff documents all structural changes,
confirms architectural compliance across all debt dimensions, and provides
recommendations for primal teams evolving their own codebases.

---

## 1. What Was Done

### 1.1 Six Large-File Smart Refactors

All `.rs` files >800 lines split into logically coherent companion modules:

| File | Lines | Split | Domain Boundary |
|------|-------|-------|-----------------|
| `validate_barracuda_tensor` | 875→424+467 | core activations vs transcendental/extended ops | `main.rs` + `extended.rs` |
| `validate_gpu_pure_wdm_coral` | 834→366+470 | WDM surrogates vs coralForge/AlphaFold3 | `main.rs` + `coral_af3.rs` |
| `validate_metalforge_wdm_coral` | 824→376+465 | NUCLEUS role validations vs mixed routing/PCIe | `main.rs` + `coral_mixed.rs` |
| `bench_upstream_vs_local` | 879→628+250 | core bio benchmarks vs DirEvo/MODES/Swarm | `main.rs` + `extended.rs` |
| `bench_portability_tiers` | 810→527+280 | HMM/Fitness/L2/IPR vs Spatial/Dispatcher/Hamming | `main.rs` + `extended.rs` |
| `validate_barracuda_dispatch_parity` | 805→607+200 | original 32 parity checks vs S115/S127 expanded | `main.rs` + `expanded.rs` |

Pattern: `Cargo.toml` `[[bin]]` path updated to `main.rs` in new directory, companion module loaded via `mod extended;` / `mod expanded;` / `mod coral_af3;` etc.

### 1.2 Centralized biomeOS Socket Discovery

`resolve_biomeos_socket_dir()` extracted to `src/config.rs` with 4-tier resolution:
1. `BIOMEOS_SOCKET_DIR` env var (explicit override)
2. `XDG_RUNTIME_DIR/biomeos/` (freedesktop standard)
3. `/run/user/{uid}/biomeos/` (Linux UID-based, `#[cfg(unix)]`)
4. `std::env::temp_dir()/biomeos/` (universal fallback)

Consumers updated:
- `src/bin/neuralspring_primal/discovery.rs` — delegates to `config::resolve_biomeos_socket_dir()`
- `playGround/src/discovery.rs` — delegates to `neural_spring::config::resolve_biomeos_socket_dir()`
- `metalForge/forge/src/coralreef_bridge.rs` — retains independent copy (correct: independent workspace member, avoids circular deps)

### 1.3 Logging Evolution

4 `eprintln!` calls in `neuralspring_guidestone.rs` replaced with `log::{info, warn}` per ecosystem standard. `metalForge/fossils/` retains `eprintln!` (archived code, not active).

### 1.4 BarraCUDA API Alignment

4 stale tensor method calls fixed in `validate_barracuda_tensor/extended.rs`:
- `tanh_wgsl()` → `tanh()` (renamed upstream)
- `exp()` → `exp_wgsl()` (renamed upstream)
- `log()` → `log_wgsl()` (renamed upstream)
- `sqrt()` → `sqrt_wgsl()` (renamed upstream)

**Recommendation for BarraCUDA team**: Document method renames in BarraCUDA CHANGELOG or migration guide for downstream consumers.

---

## 2. Full Codebase Audit Results

| Dimension | Status | Details |
|-----------|--------|---------|
| Unsafe code | **CLEAN** | `#![forbid(unsafe_code)]` on all 3 workspace crates |
| Mocks in production | **CLEAN** | 2 legitimate fallbacks: Squirrel routing stub in `handlers.rs`, `CoralCompiler::auto()` feature-gate. All test doubles in `#[cfg(test)]` |
| `#[allow()]` | **CLEAN** | Zero. All suppression via `#[expect(lint, reason = "...")]` |
| TODO/FIXME/HACK | **CLEAN** | Zero actionable items |
| External deps | **CLEAN** | All pure Rust except `wgpu` (GPU HAL — irreplaceable) |
| Large files | **CLEAN** | Largest: `tolerances/mod.rs` (776L, pure data tables) |
| Hardcoded paths | **CLEAN** | `/run/user/` centralized in `config.rs`. Linux probes (`/proc/*`, `/sys/*`) properly `#[cfg(unix)]` gated |
| `eprintln!` in active code | **CLEAN** | Zero in `src/` and `playGround/`. Only in `metalForge/fossils/` (archived) |
| `unwrap()` in production | **CLEAN** | Zero outside benchmarks (which have explicit `#[expect]`) and `fossils/diagnostics/` |

---

## 3. Primal Use and Evolution Patterns

### 3.1 Primal Discovery Model

neuralSpring discovers primals at runtime via capability-based socket scanning:
- Constants in `src/primal_names.rs` (lowercase discovery hints only, no compile-time coupling)
- 5-tier socket discovery in `src/validation/composition.rs` and `src/bin/neuralspring_primal/discovery.rs`
- JSON-RPC `capability.list` / `capability.resolve` probes
- Graceful degradation when peers unavailable (exit 2 honest skip)

### 3.2 Primal Integration Surface

| Primal | Integration | Status |
|--------|-------------|--------|
| **BearDog** | `crypto.hash` IPC for DAG signing; Tower startup liveness probe | Live |
| **Songbird** | Discovery mesh; Tower startup liveness probe; neuralAPI capability announcement | Live |
| **Squirrel** | `inference.complete`/`embed`/`models` routing via `try_squirrel_route()` | Live (stub fallback when absent) |
| **ToadStool** | `compute.dispatch` IPC; streaming pattern (upload→compute→readback) | Live |
| **BarraCUDA** | In-process Rust crate (`barracuda::*`) for GPU math, tensor ops, WGSL dispatch | Live (v0.3.12) |
| **coralReef** | Sovereign WGSL compiler pipeline; `coralreef` feature-gated `CoralCompiler` | Live (stub when feature off) |
| **petalTongue** | Visualization IPC (`push_spectral`, `push_render`, `push_scenario`) | Live |
| **NestGate** | DAG provenance (rhizoCrypt), braid audit trail | Available via composition |
| **biomeOS** | Orchestrator registration, heartbeat, socket layout | Live |

### 3.3 NUCLEUS Composition Patterns

neuralSpring validates NUCLEUS composition through 6 composition validators:
1. `validate_nucleus_composition` — proto-nucleate bonding policy, 7-node discovery sweep
2. `validate_inference_composition` — inference chain (neuralSpring→Squirrel→provider)
3. `validate_primal_discovery` — capability-based routing for 7 ecosystem primals
4. `validate_composition_evolution` — 5-phase coherence (capabilities, deploy graph, IPC, inference, health)
5. `validate_nucleus_compute_dispatch` — mixed-hardware dispatch parity
6. `validate_nucleus_pcie_mixed_pipeline` — PCIe bypass and cost validation

### 3.4 neuralAPI and biomeOS Deployment

The primal binary (`neuralspring`) registers with biomeOS via:
- `biomeos::register_with_biomeos` at startup
- Heartbeat loop with configurable interval
- Signal-driven deregistration on shutdown
- 30 advertised capabilities (science, health, inference, provenance, routing)

**neuralAPI** integration is via `NeuralApi` struct in visualization scenarios (enabled/disabled flag for petalTongue rendering). The broader neuralAPI routing happens through Songbird's capability announcement.

---

## 4. Recommendations for Upstream Teams

### For primalSpring
- neuralSpring is audit-ready (V138). All composition validators, deploy graphs, and capability surfaces aligned to `downstream_manifest.toml`
- `PRIMAL_GAPS.md` has 14 gaps documented (Gaps 1-14), many resolved. Remaining open items are upstream-dependent (Squirrel provider registration, rhizoCrypt DAG, Nest atomic)
- Pre-existing test flake: `ipc_dispatch::tests::discover_returns_none_for_absent_primals` fails when any primal socket is running on the test machine. Consider making it environment-aware

### For BarraCUDA
- Method renames (`tanh_wgsl→tanh`, `exp→exp_wgsl`, `log→log_wgsl`, `sqrt→sqrt_wgsl`) caught during S187 audit. Consider documenting these in a migration guide
- neuralSpring exercises 806+ WGSL shaders across 27 paper domains — this is the most comprehensive downstream validation surface

### For ToadStool
- Streaming pattern (unidirectional upload→compute→readback) validated across 7 portability tier benchmarks
- `compute.dispatch.submit` method name confirmed (S181 fix)

### For Squirrel
- `try_squirrel_route()` in `handlers.rs` provides the integration template for inference routing
- `inference.register_provider` is unverified — neuralSpring expects this API for Squirrel to announce inference providers
- Recommendation: `EXTRA_PRIMALS` env var in launcher for specifying additional primal sockets

### For coralReef
- `CoralCompiler::auto()` stub pattern works well for feature-gated integration
- Sovereign compile pipeline validated through `validate_sovereign_compile` and cross-spring shader evolution benchmarks

### For All Teams
- `#[expect(lint, reason = "...")]` is the ecosystem standard (not `#[allow()]`)
- `forbid(unsafe_code)` is enforced workspace-wide
- Socket discovery uses the 4-tier `resolve_biomeos_socket_dir()` pattern
- Graceful degradation (exit 2 honest skip) is the standard for missing peers

---

## 5. Archive Material

`metalForge/fossils/` contains 19 archived files (superseded code, absorbed shaders, legacy benchmarks). This is intentional fossil record per `FOSSIL_RECORD.md`. No cleanup needed.

`control/` contains 80+ Python baselines + 27 JSON reference files — these are the live validation baseline layer (Python→Rust parity). Active and necessary.

---

## 6. Quality Gates (S187)

| Gate | Status |
|------|--------|
| `cargo check --workspace` | PASS |
| `cargo clippy --workspace` | PASS (0 warnings, pedantic+nursery) |
| `cargo fmt --check` | PASS |
| `cargo deny check` | PASS (advisories ok, bans ok, licenses ok, sources ok) |
| `cargo test --workspace --lib` | 1,234 pass, 1 fail (pre-existing env flake), 1 ignored |
| Largest `.rs` file | 776L (`tolerances/mod.rs` — pure data) |

---

*V138 handoff — neuralSpring deep debt cleanup + ecosystem handoff.*
