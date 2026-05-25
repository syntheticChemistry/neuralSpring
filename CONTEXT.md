<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Context — neuralSpring

## What This Is

neuralSpring is a validation harness (spring / niche) proving [barraCuda](https://github.com/ecoPrimals/barraCuda) Rust and WGSL primitives reproduce Python ML baselines. It is part of the [ecoPrimals](https://github.com/ecoPrimals) sovereign computing ecosystem — self-contained components that coordinate via JSON-RPC 2.0 over Unix sockets, with zero compile-time coupling.

## Role in the Ecosystem

neuralSpring is a **spring** (niche validation domain), **not** a primal. It validates that scientific Python baselines can be faithfully ported to sovereign Rust plus GPU compute. In deployment it participates as a **biomeOS graph** composing real primals — BearDog (crypto), Songbird (networking), ToadStool (orchestration), coralReef (WGSL compiler), barraCuda (math engine), Squirrel (inference), and skunkBat (security audit) — rather than standing alone as a monolith.

## Architecture (Eukaryotic — post-interstadial May 2026)

- **UniBin**: Single `neuralspring_unibin` binary with `certify`, `validate`, `serve`, `status`, `version` subcommands
- **IPC tree**: `src/ipc/` with 7 per-primal modules (`barracuda`, `toadstool`, `beardog`, `squirrel`, `coralreef`, `skunkbat`, `nestgate`) + `IpcMathClient` facade + `CapabilityRouter` (20 hints, 45 capabilities)
- **Certification organelle**: `src/certification/` — 6-layer guidestone (bare/discovery/parity/nucleus/composition/cross-spring)
- **Validation scenarios**: `src/validation/scenarios/` — 10 scenarios with `ScenarioMeta`, `ScenarioRegistry`, tiered execution (Rust + Live)
- **Fossilized patterns**: Migrated to `ecoPrimals/fossilRecord/` (stub README remains)

## Technical Facts

- **Language:** Rust 2024 edition, `rust-version` **1.87**
- **License:** AGPL-3.0-or-later (scyBorg: AGPL code + ORC mechanics + CC-BY-SA creative)
- **Workspace:** 4 crates — `neural-spring` (library), `neural-spring-forge`, `neuralspring-playground`, `neuralspring-exp094`
- **Scale:** 520+ Rust source files; every file under 800 lines
- **Safety:** zero `unsafe` (`#![forbid(unsafe_code)]` workspace-wide); cast lints (`cast_possible_truncation`, `cast_sign_loss`) denied
- **Linting:** Clippy pedantic + nursery, zero warnings, zero `#[allow()]` (all `#[expect(reason)]`)
- **Dependencies:** `barracuda` (math engine, `optional = true`, IPC-first), `wgpu` **28** (GPU), `tokio` (async), `tarpc` (optional RPC), `thiserror` (typed errors). 11 modules feature-gated behind `barracuda`.
- **Supply chain:** `cargo-deny` in CI; `deny.toml` bans ring/openssl/rustls; zero C application dependencies (ecoBin compliant)
- **IPC:** JSON-RPC 2.0 over Unix domain sockets; methods use semantic `domain.verb` naming; `CompositionContext::from_live_discovery_with_fallback()` for all cross-primal calls
- **GPU:** WGSL shaders via barraCuda; f64-canonical precision dispatch; ~97% GPU promotion
- **Config:** Centralized `config.rs` — all env vars, socket resolution (`resolve_biomeos_socket_dir()`), family ID resolution (`resolve_family_id()`), capability constants

## Key Modules

`surrogate`, `transformer`, spectral analysis, Anderson localization, `coral_forge` (protein folding), `streaming` (FASTA / FASTQ / VCF), GPU dispatch, visualization (`petalTongue`), `weight_spectral` (baseCamp), `certification` (guidestone layers), `validation::scenarios` (absorbed validators), `ipc` (per-primal IPC tree).

## Key Capabilities (JSON-RPC)

Forty-five capabilities (`domain.verb`) when composed in biomeOS:

- **Science (14):** `science.spectral_analysis`, `science.anderson_localization`, `science.hessian_eigen`, `science.agent_coordination`, `science.ipr`, `science.disorder_sweep`, `science.training_trajectory`, `science.evoformer_block`, `science.structure_module`, `science.folding_health`, `science.gpu_dispatch`, `science.cross_spring_provenance`, `science.cross_spring_benchmark`, `science.precision_routing`
- **Health (3):** `health.liveness`, `health.readiness`, `health.check`
- **Inference (3):** `inference.complete`, `inference.embed`, `inference.models`
- **Provenance (4):** `provenance.begin`, `provenance.record`, `provenance.complete`, `provenance.status`
- **Routing (7):** `primal.forward`, `primal.discover`, `capability.list`, `identity.get`, `mcp.tools.list`, `compute.dispatch`, `compute.offload`
- **Composition (2):** `composition.status`, `method.register` (biomeOS v3.51)
- **Security (1):** `security.audit_log` (skunkBat JH-5 forwarding)

## Deploy Graphs

4 TOML deploy graphs in `graphs/`:
- `neuralspring_deploy.toml` — full NUCLEUS composition (7 primals incl. skunkBat)
- `neuralspring_spectral_analysis.toml` — science domain (barraCuda)
- `neuralspring_inference_pipeline.toml` — inference chain (Squirrel)
- `composition/neuralspring_math_pipeline.toml` — math composition (barraCuda + toadStool)

## Test Coverage

CI-enforced **~92%** line coverage (`llvm-cov`). **754 lib + 11 integration + 73 forge + 80 playGround + 12 exp094 = 930 workspace tests (IPC-first)** + 19 certification tests (guidestone feature). Suite includes unit tests, property tests (24 proptest), determinism tests, doc tests, integration tests, provenance integrity tests, and 8 composition validators. `ValidationSink` for machine-readable CI output. **guideStone Level 5** (29/29 bare ALL PASS, L4 composition + L5 cross-spring when NUCLEUS live). 8 paper notebooks (72/72 checks, 2 faculties). 233+ named tolerances. Session S219 (May 25, 2026).

## What This Does NOT Do

- Does not replace barraCuda — consumes it for GPU math and WGSL entry points
- Does not compile shaders from scratch — coralReef owns the sovereign compiler pipeline; neuralSpring validates and dispatches through barraCuda
- Is not a general ML training platform — it is parity and harness validation against `control/` baselines

## Related Repositories

- [wateringHole](https://github.com/ecoPrimals/wateringHole) — ecosystem standards and primal registry
- [barraCuda](https://github.com/ecoPrimals/barraCuda) — math engine and WGSL primitives
- [toadStool](https://github.com/ecoPrimals/toadStool) — hardware / workload orchestration
- [coralReef](https://github.com/ecoPrimals/coralReef) — WGSL compiler pipeline
- [skunkBat](https://github.com/ecoPrimals/skunkBat) — defensive network security

## Evolution Path

```text
Layer 1: Python baseline (control/) → Rust validation (src/)
Layer 2: Rust validation → GPU (WGSL via barraCuda)
Layer 3: Rust+Python validate primal IPC (composition validators, deploy graph)
Layer 4: NUCLEUS composition (biomeOS deploy) → sovereign deployment (plasmidBin ecoBin)
Layer 5: Eukaryotic UniBin → certification organelle → validation scenarios → fossilization
```

## Gate Deployment

| Property | Value |
|----------|-------|
| **Gate** | southGate |
| **Hardware** | AMD Ryzen 7 5800X3D (8-core), 128GB DDR4 |
| **OS** | Pop!_OS 22.04 (Linux 6.17) |
| **Composition** | Full NUCLEUS (13 primals), Node Atomic profile |
| **Federation** | Songbird TCP port 7700 (opt-in via `SONGBIRD_FEDERATION_PORT`) |
| **Co-tenants** | wetSpring |
| **NUCLEUS launcher** | `./tools/composition_nucleus.sh start` |
| **Cell graph** | `plasmidBin/cells/neuralspring_cell.toml` |
| **Deployment status** | Operational — 9/13 UDS, 2 TCP/abstract, 2 upstream failures (S219) |
| **Launch** | `SONGBIRD_FEDERATION_PORT=7700 ./tools/composition_nucleus.sh start` then `./tools/cell_launcher.sh neuralspring start` |

## Design Philosophy

Built with AI-assisted constrained evolution: Rust's ownership and type system narrow the search space; springs prove numerical and protocol fidelity before GPU promotion. Primals remain capability-local; graphs compose behavior at runtime. Code evolves through fossilization (old patterns archived with provenance) rather than deletion.

---

## Part of ecoPrimals

This repo is part of the [ecoPrimals](https://github.com/ecoPrimals) sovereign computing ecosystem — pure Rust components that coordinate via JSON-RPC, capability-based routing, and zero compile-time coupling.

See [wateringHole](https://github.com/ecoPrimals/wateringHole) for ecosystem documentation, standards, and the primal registry.
