# neuralSpring V143 — Interstadial Eukaryotic Evolution Handoff

**Date:** May 9, 2026
**Session:** S193
**From:** neuralSpring
**To:** primalSpring, all primal teams, all spring teams

---

## Summary

neuralSpring has completed the interstadial eukaryotic evolution, transitioning
from a prokaryotic multi-binary topology to eukaryotic organelle-based
architecture following `primalSpring`'s UniBin pattern.

## Delivered Items

### 1. IPC Tree Graduation (`src/ipc/`)

Monolithic `src/ipc_dispatch.rs` (401 lines) graduated to per-primal modules:

| Module | Primal | Capabilities |
|--------|--------|--------------|
| `ipc/barracuda.rs` | barraCuda | `stats.*`, `tensor.*` |
| `ipc/toadstool.rs` | toadStool | `compute.dispatch` |
| `ipc/beardog.rs` | BearDog | `crypto.hash` |
| `ipc/squirrel.rs` | Squirrel | `inference.*` |
| `ipc/coralreef.rs` | coralReef | `shader.compile.*` (new) |

`IpcMathClient` facade preserved in `ipc/mod.rs` — API-compatible. Old module
marked `#[deprecated]`.

### 2. Certification Organelle (`src/certification/`)

4-layer certification library absorbed from `neuralspring_guidestone` binary:

| Layer | Module | Primals? | Description |
|-------|--------|----------|-------------|
| L0 | `bare.rs` | No | 5 certified properties (determinism, traceability, checksums, env-agnostic, tolerances) |
| L1 | `discovery.rs` | Yes | `CompositionContext` liveness |
| L2 | `parity.rs` | Yes | 7-capability domain science parity |
| L3 | `nucleus.rs` | Yes | BearDog signing, Songbird discovery |

Public API: `certification::certify(max_layer) -> ValidationResult`

### 3. Validation Scenarios (`src/validation/scenarios/`)

6 scenarios absorbed from `validate_*` binaries with `ScenarioMeta` provenance:

| Scenario | Track | Tier | Checks | Source |
|----------|-------|------|--------|--------|
| `nucleus_composition` | NucleusComposition | Live | 22 | `validate_nucleus_composition` |
| `inference_composition` | InferencePipeline | Live | 16 | `validate_inference_composition` |
| `science_composition` | SpectralAnalysis | Both | 9 | `validate_science_composition` |
| `nucleus_tower` | NucleusComposition | Both | 47 | `validate_nucleus_tower` |
| `compute_dispatch` | NucleusComposition | Both | 36 | `validate_nucleus_compute_dispatch` |
| `composition_evolution` | CrossSpring | Both | 30 | `validate_composition_evolution` |

Types: `Tier` (Rust/Live/Both), `Track` (6 domain areas), `ScenarioRegistry`.
Registry: `build_registry()` returning all 6 scenarios.

### 4. UniBin Binary (`neuralspring-unibin`)

Single binary with 5 subcommands:

- `certify --layer N --bare` — certification layers L0-L3
- `validate --track X --scenario Y --tier Z --list` — scenario execution
- `serve` — stub (future: absorb primal server)
- `status` — IPC liveness summary (5 primals + scenario count)
- `version` — semver one-liner

Feature-gated behind `guidestone`.

### 5. Fossilization

3 pre-extinction patterns fossilized with provenance READMEs:

- `fossilRecord/guidestone_prokaryotic_may2026/` — standalone guidestone binary
- `fossilRecord/ipc_dispatch_prokaryotic_may2026/` — monolithic IPC dispatch
- `fossilRecord/primal_server_prokaryotic_may2026/` — primal server directory

### 6. Deprecated Pattern Migration

Deprecated in `playGround/src/`:
- `PrimalClient` → `CompositionContext::from_live_discovery_with_fallback()`
- `discover_primal()` → `CompositionContext` name-based discovery
- `discover_by_capability()` → `CompositionContext` capability discovery

All with `#[deprecated(since = "0.2.0", note = "...")]`.

## Quality Gates

- **Tests:** 1,291 lib + 73 forge + 80 playGround = 1,444 workspace
- **New tests:** 8 (scenario registry + IPC module)
- **Bare `#[allow]`:** 0
- **TODO/FIXME/HACK/DEBT:** 0 in library
- **`cargo build + fmt + clippy + test`:** all clean
- **`primalSpring`:** pinned at v0.9.25 (path dep)

## Evolution Status

| Target | Status |
|--------|--------|
| IPC tree graduation | COMPLETE |
| Certification organelle | COMPLETE |
| Validation scenarios | COMPLETE (6/6 initial) |
| UniBin binary | COMPLETE (serve is stub) |
| Fossilization | COMPLETE |
| Deprecated migrations | COMPLETE |
| Serve subcommand absorption | FUTURE (next wave) |
| Full validator absorption | FUTURE (incremental) |

## Next Wave Targets

1. **Serve absorption**: Move `neuralspring_primal/` logic into UniBin `serve` subcommand
2. **Batch IPC**: Send full layer computations as single `tensor.matmul` calls
3. **Per-trio provenance modules**: Create provenance modules for each primal trio
4. **Additional scenario absorption**: Gradually migrate remaining `validate_*` binaries
5. **coralReef IPC validation**: Wire `shader.compile.*` into parity layer

## Upstream Dependencies

All upstream primal gaps documented in V142 remain active. This handoff is
additive — no new gaps introduced.

---

**Provenance:** neuralSpring V143, Session S193, May 9, 2026.
