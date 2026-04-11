<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Context — neuralSpring

## What This Is

neuralSpring is a validation harness (spring / niche) proving [barraCuda](https://github.com/ecoPrimals/barraCuda) Rust and WGSL primitives reproduce Python ML baselines. It is part of the [ecoPrimals](https://github.com/ecoPrimals) sovereign computing ecosystem — self-contained components that coordinate via JSON-RPC 2.0 over Unix sockets, with zero compile-time coupling.

## Role in the Ecosystem

neuralSpring is a **spring** (niche validation domain), **not** a primal. It validates that scientific Python baselines can be faithfully ported to sovereign Rust plus GPU compute. In deployment it participates as a **biomeOS graph** composing real primals — BearDog (crypto), Songbird (networking), ToadStool (orchestration), coralReef (WGSL compiler), and barraCuda (math engine) — rather than standing alone as a monolith.

## Technical Facts

- **Language:** Rust 2024 edition, `rust-version` **1.87**
- **License:** AGPL-3.0-or-later (scyBorg: AGPL code + ORC mechanics + CC-BY-SA creative)
- **Workspace:** 3 crates — `neural-spring` (library), `neural-spring-forge`, `neuralspring-playground`
- **Scale:** 505 Rust source files; every file under 1000 lines
- **Safety:** zero `unsafe` (`#![forbid(unsafe_code)]` workspace-wide); cast lints (`cast_possible_truncation`, `cast_sign_loss`) denied
- **Linting:** Clippy pedantic + nursery, zero warnings, zero `#[allow()]`
- **Dependencies:** `barracuda` (math engine, `default-features = false`), `wgpu` **28** (GPU), `tokio` (async), `tarpc` (optional RPC for the primal binary), `thiserror` (typed errors: `GpuError`, `TensorError`, `ParseError`)
- **Supply chain:** `cargo-deny` in CI; `rustfmt.toml` for formatting consistency; zero C application dependencies (ecoBin compliant)
- **IPC:** JSON-RPC 2.0 over Unix domain sockets; methods use semantic `domain.verb` naming
- **GPU:** WGSL shaders via barraCuda; f64-canonical precision dispatch

## Key Modules

`surrogate`, `transformer`, spectral analysis, Anderson localization, `coral_forge` (protein folding), `streaming` (FASTA / FASTQ / VCF), GPU dispatch, visualization (`petalTongue`), `weight_spectral` (baseCamp).

## Key Capabilities (JSON-RPC)

Thirty capabilities (`domain.verb`) when composed in biomeOS. Method naming follows Semantic Method Naming v2.1: discovery may list or resolve these via `capability.list` (canonical), `identity.get` (T4 discovery), or `mcp.tools.list` (MCP adapter).

- **Science (14):** `science.spectral_analysis`, `science.anderson_localization`, `science.hessian_eigen`, `science.agent_coordination`, `science.ipr`, `science.disorder_sweep`, `science.training_trajectory`, `science.evoformer_block`, `science.structure_module`, `science.folding_health`, `science.gpu_dispatch`, `science.cross_spring_provenance`, `science.cross_spring_benchmark`, `science.precision_routing`
- **Health (3):** `health.liveness`, `health.readiness`, `health.check`
- **Inference (3):** `inference.complete`, `inference.embed`, `inference.models`
- **Provenance (4):** `provenance.begin`, `provenance.record`, `provenance.complete`, `provenance.status`
- **Routing (6):** `primal.forward`, `primal.discover`, `capability.list`, `identity.get`, `mcp.tools.list`, `compute.offload`

## Test Coverage

CI-enforced **≥90%** line coverage (`llvm-cov`). ~1,378 tests (1,225 lib + 73 forge + 80 playGround). Suite includes unit tests, property tests, determinism tests, doc tests, integration tests, and provenance integrity tests. `ValidationSink` for machine-readable CI output (JSON, NDJSON, collecting).

## What This Does NOT Do

- Does not replace barraCuda — consumes it for GPU math and WGSL entry points
- Does not compile shaders from scratch — coralReef owns the sovereign compiler pipeline; neuralSpring validates and dispatches through barraCuda
- Is not a general ML training platform — it is parity and harness validation against `control/` baselines

## Related Repositories

- [wateringHole](https://github.com/ecoPrimals/wateringHole) — ecosystem standards and primal registry
- [barraCuda](https://github.com/ecoPrimals/barraCuda) — math engine and WGSL primitives
- [toadStool](https://github.com/ecoPrimals/toadStool) — hardware / workload orchestration
- [coralReef](https://github.com/ecoPrimals/coralReef) — WGSL compiler pipeline

## Evolution Path

```text
Python baseline (control/) → Rust validation (src/) → GPU (WGSL via barraCuda)
  → NUCLEUS composition (primal IPC via biomeOS) → sovereign deployment
```

## Design Philosophy

Built with AI-assisted constrained evolution: Rust’s ownership and type system narrow the search space; springs prove numerical and protocol fidelity before GPU promotion. Primals remain capability-local; graphs compose behavior at runtime.

---

## Part of ecoPrimals

This repo is part of the [ecoPrimals](https://github.com/ecoPrimals) sovereign computing ecosystem — pure Rust components that coordinate via JSON-RPC, capability-based routing, and zero compile-time coupling.

See [wateringHole](https://github.com/ecoPrimals/wateringHole) for ecosystem documentation, standards, and the primal registry.
