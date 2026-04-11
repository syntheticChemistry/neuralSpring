<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring V129 — Deploy Graph Proto-Nucleate Alignment

**Date**: April 11, 2026
**Session**: S179
**Spring**: neuralSpring 0.1.0
**barraCuda**: v0.3.11 (path dep, `../../primals/barraCuda/crates/barracuda`)
**License**: AGPL-3.0-or-later

---

## Summary

neuralSpring's deploy graph (`neuralspring_deploy.toml`) is now aligned with
its proto-nucleate composition
(`neuralspring_inference_proto_nucleate.toml` v1.1.0). The three-layer
validation stack — Python baselines validate Rust, Rust and Python baselines
validate NUCLEUS composition patterns — is complete at the graph level.

This handoff documents what changed, what it means for each primal team, and
what remains open.

---

## What Changed (S178 → S179)

| Area | Before (V128/S178) | After (V129/S179) |
|------|--------------------|--------------------|
| Deploy graph nodes | BearDog, Songbird, ToadStool, NestGate, provenance trio, neuralSpring | + coralReef, barraCuda, Squirrel |
| BearDog `by_capability` | `crypto` | `security` (proto-nucleate alignment) |
| ToadStool `by_capability` | `compute` | `compute.dispatch.submit` |
| NestGate `by_capability` | `storage` | `storage.retrieve` |
| `capabilities_provided` | 14 (science only) | 26 (full niche surface) |
| Graph metadata | id, description, coordination, domain | + composition_model, bond_type, trust_model, transport, tcp_ports, encryption_boundary, fragments, proto_nucleate |
| Version string | stale S174 in provenance node | S179 |
| `config::ALL_CAPABILITIES` | 18 entries (subset of niche) | 26 entries (matches niche::CAPABILITIES) |
| `operation_dependencies()` | Missing primal.*, capability.list, compute.offload | All 26 capabilities covered |
| `cost_estimates()` | Missing provenance.status, primal.*, capability.list, compute.offload | All capabilities covered |
| GpuCapabilities doctest | Failed (import ordering) | Fixed — 0 doctest failures |
| `ANDERSON_MULTIAGENT_ENVIRONMENT` | `"Python 3.12, NumPy, seed=42"` | `"Python 3.12.3, NumPy 2.2.6, seed=42"` |
| Kokkos benchmark provenance | `PLACEHOLDER` | `ESTIMATED` (honest reclassification) |

---

## For barraCuda Team

### Deploy Graph Change

barraCuda now appears as a germination node in the deploy graph:

```toml
[[nodes]]
id = "germinate_barracuda"
capabilities = ["math.tensor", "math.stats", "math.activation"]

[nodes.primal]
by_capability = "math.tensor"
```

This is the **composition target** — today neuralSpring calls barraCuda via
direct Rust import (250+ sites, 46 rewires). The germination node documents
the future IPC path when biomeOS orchestrates the full graph. No action
needed from barraCuda until `math.*` JSON-RPC surface exists.

### Open Items (unchanged from V128)

- Feature-gate bug: `special::plasma_dispersion` → `domain-lattice`
- Absorption candidates: `head_split.wgsl`, `head_concat.wgsl` (binding layout)

---

## For coralReef Team

### Deploy Graph Change

coralReef now appears as a germination node:

```toml
[[nodes]]
id = "germinate_coralreef"
capabilities = ["shader.compile"]

[nodes.primal]
by_capability = "shader.compile.wgsl"
```

neuralSpring's `metalForge/forge` has the optional `coralreef` feature with
a `coralreef_bridge` module. The deploy graph makes this discoverable to
biomeOS. Fallback to local `wgpu` when coralReef unavailable.

### What We Need (unchanged)

- `shader.compile.wgsl` IPC endpoint
- `Fp64Strategy` coordination

---

## For Squirrel Team

### Deploy Graph Change

Squirrel now appears as a germination node:

```toml
[[nodes]]
id = "germinate_squirrel"
capabilities = ["ai.query", "inference"]

[nodes.primal]
by_capability = "ai.query"
```

neuralSpring registers `inference.complete`, `inference.embed`,
`inference.models` in the full 26-capability surface. Handlers remain honest
stubs. `playGround/src/squirrel_client.rs` uses `discover_by_capability("ai.query")`
with name fallback.

### What We Need (unchanged)

- `inference.register_provider` — neuralSpring registers as ML inference backend
- Provider health monitoring for Ollama fallback routing

---

## For biomeOS Team

### Deploy Graph Evolution

`neuralspring_deploy.toml` V129 now carries full composition metadata:

```toml
[graph.metadata]
composition_model = "nucleated"
bond_type = "Metallic"
trust_model = "InternalNucleus"
transport = "uds"
tcp_ports = 0
encryption_boundary = "tower"
fragments = ["tower_atomic", "node_atomic", "meta_tier"]
proto_nucleate = "neuralspring_inference_proto_nucleate"
```

All 26 niche capabilities are declared in `capabilities_provided`. The
graph is aligned with the proto-nucleate for biomeOS deployment validation.

### Deployment Path (updated)

```
biomeOS deploys neuralspring_deploy.toml →
  Phase 1: Tower (BearDog + Songbird)
  Phase 2: Node (coralReef + ToadStool + barraCuda — all optional/skip)
  Phase 2b: Nest (NestGate + provenance trio — all optional/skip)
  Phase 2c: Meta (Squirrel — optional/skip)
  Phase 3: neuralSpring (spawn, by_capability = science.spectral_analysis)
  Phase 4: Validate (health check)
  Phase 5: Provenance (optional genesis recording)
```

---

## For Spring Teams

### Three-Layer Validation Stack (complete)

```
Layer 1: Python baseline → Rust validator (science correctness)
  → ValidationHarness + provenance + tolerances + control/*.json

Layer 2: Rust validator → GPU validator (shader promotion)
  → dispatch parity + Tensor API + TensorSession + metalForge WGSL

Layer 3: Rust + Python → Composition validator (NUCLEUS patterns)
  → niche bonding + capability discovery + proto-nucleate graph
  → deploy graph aligned to proto-nucleate (V129)
  → exit-2 honest skip when ecosystem offline
  → ecoBin harvestable to infra/plasmidBin
```

Python was the validation target for Rust. Now Rust and Python are the
validation targets for NUCLEUS composition patterns.

### Capability Surface Reconciliation

`config::ALL_CAPABILITIES` now matches `niche::CAPABILITIES` at 26 entries.
No more split between "subset for config" and "full surface for niche" —
biomeOS consumers see the same set whether they read config or niche.

---

## Metrics (S178 → S179)

| Metric | S178 | S179 |
|--------|------|------|
| Library tests | 1,225 | 1,225 |
| Doctest failures | 1 | 0 |
| Binaries | 264 | 264 |
| `.rs` files | 518 | 518 |
| Capabilities (niche) | 26 | 26 |
| Capabilities (config) | 18 | 26 |
| Deploy graph nodes | 9 | 12 (+coralReef, barraCuda, Squirrel) |
| Deploy graph capabilities_provided | 14 | 26 |
| clippy warnings | 0 | 0 |
| `#[allow()]` | 0 | 0 |
| Deploy graph | V128/S178 | V129/S179 |
| PRIMAL_GAPS resolved | 2 | 4 (+R3 deploy graph, +R4 capability surface) |

---

## Open Items (hand back to primalSpring)

1. **Squirrel provider registration**: `inference.register_provider` does not exist yet
2. **coralReef shader.compile.wgsl**: neuralSpring ready to route, coralReef endpoint needed
3. **NestGate weight storage**: `storage.retrieve` for model weights
4. **BearDog/Songbird BTSP**: Trust session establishment for composed mode
5. **Proto-nucleate fragment inconsistency**: Graph declares `tower_atomic + node_atomic + meta_tier` but includes optional NestGate — should `nest_atomic` be conditional?
6. **barraCuda IPC migration**: Direct import → `math.*` JSON-RPC when biomeOS orchestrates full graph
