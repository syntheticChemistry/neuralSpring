# neuralSpring V131 — Composition Evolution: Squirrel Routing, Tower Discovery, Tier 3 Validation

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Date:** 2026-04-11  
**Session:** S181  
**Spring:** neuralSpring 0.1.0  
**barraCuda:** v0.3.11  

---

## What changed

### 1. Full capability surface (27 → 30)

`ALL_CAPABILITIES` now includes `health.check`, `identity.get`, and
`mcp.tools.list`. These three methods were already dispatched by the primal
binary but were invisible to `capability.list` and biomeOS
`capability.register`. Now all dispatched methods are discoverable.

Synchronized across: `config.rs`, `niche.rs`, `capability_registry.toml`,
MCP tool definitions (30 tools), plasmidBin metadata.

### 2. Squirrel inference routing

Inference handlers (`inference.complete`, `inference.embed`,
`inference.models`) now attempt runtime Squirrel discovery via
`try_squirrel_route()`. When Squirrel is running, requests are forwarded
via JSON-RPC over UDS. When absent, handlers return stub responses with
`"status": "squirrel_unavailable"` (previously `"not_yet_wired"`).

**Still needed from Squirrel:** `inference.register_provider` wire so
Squirrel discovers neuralSpring as an inference backend.

### 3. Tower Atomic startup probes

New `tower.rs` module in the primal binary probes BearDog and Songbird at
startup via capability-based socket discovery + `health.liveness`. Logs
Tower status (complete/partial/standalone). Non-blocking — neuralSpring
runs independently when Tower is absent.

**Still needed from BearDog/Songbird:** BTSP session establishment,
Songbird mesh discovery, signed capability announcements.

### 4. Tier 3 composition validator

`validate_composition_evolution.rs` validates the full lifecycle:

- Phase 1: Capability surface completeness (niche ↔ config ↔ TOML parity)
- Phase 2: Deploy graph alignment (fragments, bonding, proto-nucleate ref)
- Phase 3: Proto-nucleate node wiring (live IPC discovery + health probes)
- Phase 4: Inference evolution readiness (probe inference.* handlers)
- Phase 5: Health triad (liveness + readiness + check + identity + MCP)

Exit codes: 0 pass, 1 fail, 2 honest skip.

### 5. Audit fixes

- `ToadStoolClient::discover()`: `compute.submit` → `compute.dispatch.submit`
- `check_abs_or_rel`: records actual tolerance mode that passed
- Deploy graph header: includes `nest_atomic` fragment
- Deploy graph provenance session: `S179` → `S181`
- BARRACUDA_REQUIREMENTS.md: version `v0.3.7` → `v0.3.11`
- 3 pre-existing clippy `doc_markdown` warnings fixed
- `composed` Cargo feature added for future IPC composition paths

---

## Questions back

### To Squirrel

Does `inference.register_provider` exist yet? neuralSpring is ready to
call it once the wire is defined. Current routing discovers Squirrel and
forwards `inference.*` calls — but Squirrel needs to know neuralSpring
is a provider, not just a consumer.

### To BearDog

What is the `crypto.btsp_handshake` JSON-RPC surface? neuralSpring
discovers BearDog at startup but cannot establish a BTSP session yet.

### To coralReef / toadStool

neuralSpring's `playGround` has typed IPC clients (`CoralReefClient`,
`ToadStoolClient`). When should the core library start routing shader
compilation and compute dispatch through IPC instead of direct
barraCuda/wgpu calls? Is there a feature-gate pattern to recommend?

### To primalSpring

Upstream proto-nucleate (`neuralspring_inference_proto_nucleate.toml`)
still declares `fragments = ["tower_atomic", "node_atomic", "meta_tier"]`
without `nest_atomic`. Local deploy graph is fixed. Should the upstream
graph add `nest_atomic` since NestGate is in the node list?

---

## Wiring status matrix

| Primal | Discovery | Liveness | Capability probe | Live routing | Status |
|--------|-----------|----------|-----------------|-------------|--------|
| biomeOS | ✓ socket | ✓ heartbeat | — | ✓ register/heartbeat | **wired** |
| BearDog | ✓ startup probe | ✓ probe | — | — | **discovery only** |
| Songbird | ✓ startup probe | ✓ probe | — | — | **discovery only** |
| Squirrel | ✓ per-request | — | — | ✓ inference.* forward | **routed** |
| toadStool | playGround client | — | — | — | **client ready** |
| coralReef | playGround client | — | — | — | **client ready** |
| NestGate | — | — | — | — | **open** |
| barraCuda | direct Rust import | — | — | — | **deferred** |

---

## Validation tiers

| Tier | What it validates | Binary | Status |
|------|-------------------|--------|--------|
| 1 | Python → Rust (science correctness) | `validate_all` (261 science bins) | **green** |
| 2 | Rust → NUCLEUS (primal wiring) | `validate_nucleus_composition`, `validate_inference_composition`, `validate_primal_discovery` | **green** (skip-aware) |
| 3 | Composition evolution (full coherence) | `validate_composition_evolution` | **new (S181)** |
