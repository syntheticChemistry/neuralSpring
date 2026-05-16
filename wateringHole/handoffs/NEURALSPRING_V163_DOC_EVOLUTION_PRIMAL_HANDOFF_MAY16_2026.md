# neuralSpring V163 — Documentation Evolution + Primal Handoff

**From:** neuralSpring (S207c)
**To:** primalSpring (coordination), all spring teams, all primal teams
**Date:** 2026-05-16
**Session:** S207c — comprehensive doc reconciliation + upstream primal handoff

---

## What This Session Did

Systematic documentation reconciliation across 18 files. All version/session/handoff references synced from stale S205b/V158 → current S207b/V162. Capability count corrected from stale 37 → actual 35 (Wave 17 surface: `primal.announce` consolidated 3 methods into 1). barraCuda version refs purged of v0.3.12 → v0.4.0 in active headers. Validation scenario count updated 6 → 7 (`s_signal_dispatch`). Deep debt audit count updated to 5 (added S207b).

### Files Updated (18)

| File | Change |
|------|--------|
| `README.md` | V158→V162, archive range V1–V161, Sessions→207b, capabilities 37→35, scenarios 7, Wave 17 |
| `EVOLUTION_READINESS.md` | Session/handoff header → S207b/V162, capabilities 35, scenarios 7, audits 5 |
| `CONTROL_EXPERIMENT_STATUS.md` | Session header → S207b/V162 |
| `DEPRECATION_MIGRATION.md` | Sessions→207b, V162, Wave 17, audits 5 |
| `CONTEXT.md` | Capabilities 37→35 |
| `CONTRIBUTING.md` | Binary count 264→269 |
| `docs/GUIDESTONE_PROPERTIES.md` | Session S205b→S207b, V162 |
| `docs/FOUNDATION_SEEDING.md` | Session S203→S207b, V162 |
| `whitePaper/README.md` | Session/handoff→S207b/V162, capabilities 35, Wave 17, audits 5 |
| `whitePaper/STUDY.md` | Stale 2918+ → 4700+ Rust+GPU checks, 38.6× geomean |
| `whitePaper/METHODOLOGY.md` | Stale 580 lib → 910 workspace tests |
| `whitePaper/BARRACUDA_EVOLUTION.md` | Header S181+/v0.3.12 → S207b/v0.4.0/V162 |
| `whitePaper/CROSS_SPRING_SHADER_LINEAGE.md` | ToadStool HEAD updated |
| `whitePaper/baseCamp/README.md` | Session/handoff→S207b/V162, capabilities 35, Wave 17, scenarios 7 |
| `whitePaper/baseCamp/extensions.md` | Header S181+/v0.3.12 → S207b/v0.4.0/V162 |
| `whitePaper/baseCamp/EXTENSION_PLAN.md` | Header S181+ → S207b, v0.3.12→v0.4.0, V162 |
| `whitePaper/baseCamp/PAPER_OUTLINES.md` | Header Session 85 → S207b refresh |
| `experiments/README.md` | Session S205b→S207b, capabilities 35, scenarios 7, V162 |
| `specs/README.md` | Session/handoff → S207b/V162 |
| `sporeprint/validation-summary.md` | Date, session S203→S207b |
| `notebooks/NOTEBOOK_PATTERN.md` | Session S205b→S207b |

---

## neuralSpring Primal Use & Evolution — Upstream Handoff

### Current Primal Surface (35 capabilities via IPC)

| Primal | IPC Module | Capabilities Used | Status |
|--------|-----------|-------------------|--------|
| **barraCuda** | `src/ipc/barracuda.rs` | `gpu.matmul`, `gpu.softmax`, `gpu.gelu`, tensor ops | v0.4.0. Feature-gated (`barracuda` feature). 11 modules behind gate. CPU+GPU math engine. |
| **toadStool** | `src/ipc/toadstool.rs` | `toadstool.validate`, `toadstool.list_workloads`, `compute.dispatch` | Orchestration. Workload pre-flight + dispatch. Tier 2 wired. |
| **bearDog** | `src/ipc/beardog.rs` | `crypto.hash`, `crypto.verify` | BLAKE3 hashing for NestGate content-addressed storage. Tower identity. |
| **songBird** | (Tower discovery) | `networking.*` | Tower discovery probes in deploy graphs. |
| **skunkBat** | `src/ipc/skunkbat.rs` | `security.audit_log` | JH-5 forwarding. Triple-first Tower model in all 4 deploy graphs. |
| **nestGate** | `src/ipc/nestgate.rs` | `content.put`, `content.get`, `content.exists` | Weight persistence. `store_to_nestgate` + `load_safetensors_from_nestgate`. Now also via `nest.store` signal dispatch. |
| **coralReef** | `src/ipc/coralreef.rs` | `shader.compile` | Feature-gated bridge (`metalForge/forge/src/coralreef_bridge.rs`). Returns `NotAvailable` when feature off. |
| **squirrel** | `src/ipc/squirrel.rs` | `inference.complete`, `inference.embed`, `inference.models` | ML inference pipeline. `inference_models` facade + `has_squirrel` discovery. |

### Wave 17 Signal API Adoption (primalSpring)

| Signal | Status | Evidence |
|--------|--------|----------|
| `primal.announce` | **ADOPTED** | `biomeos.rs` — single-call registration with legacy fallback. Replaces `nucleus.register` + N × `capability.register`. |
| `nest.store` | **ADOPTED** | `weight_loader.rs` — `store_to_nestgate_signal()` via `ctx.dispatch("nest.store", ...)`. Provenance-tracked weight persistence. |
| `node.compute` | Not applicable | neuralSpring delegates compute to barraCuda/toadStool, not via signal dispatch. |
| `tower.authenticate` | Not applicable | neuralSpring is not auth-heavy. |

### Composition Patterns for NUCLEUS + biomeOS

neuralSpring's 4 deploy graphs (`neuralspring_deploy.toml`, `neuralspring_inference_pipeline.toml`, `neuralspring_spectral_analysis.toml`, `composition/neuralspring_math_pipeline.toml`) all follow the **Tower Atomic** model:

1. **`tower_identity`** — bearDog crypto initialization
2. **`tower_discovery`** — songBird network discovery
3. **`tower_defense`** — skunkBat audit logging
4. Domain nodes depend on Tower completion

This is the canonical **triple-first Tower** pattern. `biomeOS` manages graph execution; springs express dependencies declaratively.

### neuralAPI Deployment

- **UniBin**: `neuralspring_unibin` with `certify`, `validate`, `serve`, `status`, `version` subcommands
- **neuralspring_primal**: JSON-RPC 2.0 over Unix sockets, auto-registers with biomeOS via `primal.announce`
- **MCP adapter**: `neuralspring_mcp_adapter` (35 tool definitions matching capability surface)
- **Certification**: `neuralspring_guidestone` — 19 tests, L0–L5, ALL PASS
- **IPC-first**: `default = []` in `Cargo.toml` — zero compile-time coupling to primals

---

## Hand-Backs to Upstream Primal Teams

### barraCuda team
- **Gap 9** (feature-gate bug) persists in v0.4.0 — `barracuda` feature in spring `Cargo.toml` still activates different code paths than intended. Needs verification in v0.4.1.
- neuralSpring's 15-domain CPU benchmark suite (`validate_barracuda_cpu_bench` + 20 `bench_*.py`) provides 38.6× geomean speedup evidence. Available for barraCuda's own benchmark narrative.
- `PrecisionRoutingAdvice` with `F64NativeNoSharedMem` Ada Lovelace reclassification is working well in production.

### coralReef team
- **Gap 3** (shader compilation routing) — `coralreef_bridge.rs` feature-gated stub returns `NotAvailable` when feature off. v0.1.0 unblocks `compile_shader_universal` routing. strandGate hardware validation (3090 + 6950) needed for dual-vendor coverage.
- 8 neuralSpring WGSL shaders in coralReef corpus.

### nestGate team
- **Gap 5 RESOLVED** — content-addressed weight storage (`content.put`/`content.get`) + Wave 17 `nest.store` signal dispatch both wired and working.
- `store_to_nestgate_signal()` demonstrates the full provenance chain: encode → dispatch → biomeOS graph handles `rhizoCrypt.dag.event.append` → `loamSpine.spine.seal` → `sweetGrass.braid.create`.

### toadStool team
- Tier 2 wired: `toadstool.validate` (workload pre-flight) + `toadstool.list_workloads` + `barracuda.precision.route` all exercised.
- `compute.dispatch` handler added in S207 for Wave 17 alignment.

### squirrel team
- Inference pipeline complete: `inference.complete`, `inference.embed`, `inference.models` all wired via `IpcMathClient`.
- `try_squirrel_route` dynamic fallback for inference routing.
- Provider registration still needs upstream squirrel support (Gap 6 — squirrel must accept `nucleus.register` or `primal.announce` from inference providers).

### primalSpring team
- Wave 17 signal API fully adopted (S207). `primal.announce` + `nest.store` wired with fallback.
- Registry sync: 451-method canonical registry verified.
- `GAP-GS-015` verified fixed — `cargo check --workspace` passes clean.
- 7 validation scenarios (including `s_signal_dispatch` for signal adoption parity).

---

## Debris Review

No debris found. All directories contain purposeful content:
- `metalForge/fossils/` — 19 files, all documented in `FOSSIL_RECORD.md`, intentionally archived
- `scripts/` — 5 files, all actively referenced by docs/CI
- `notebooks/` — 14 files (5 sporePrint + 8 paper notebooks), all indexed
- No `.bak`, `.old`, `.tmp`, `.swp` files found
- No empty directories detected
- Python outside `control/` is purpose-built: `download_pretrained.py`, `openfold3_eval.py`, `pytorch_baseline.py`

---

## Current State

| Metric | Value |
|--------|-------|
| Session | S207c |
| Workspace tests | 910 |
| Capabilities | 35 |
| Validation scenarios | 7 |
| Deploy graphs | 4 |
| Papers | 27/27 (queue CLOSED) |
| Python baselines | 397/397 PASS |
| CPU benchmark domains | 15 (38.6× geomean) |
| Deep debt audits | 5 (all zero-debt) |
| Handoff | V163 |
| Evolution | composing |
| Signal API | Wave 17 |
| Docs synced | 18 files reconciled to S207b/V162 |
