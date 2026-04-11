<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring V128 — Composition Validation Phase: NUCLEUS Patterns & Neural API Deployment

**Date**: April 11, 2026
**Session**: S178
**Spring**: neuralSpring 0.1.0
**barraCuda**: v0.3.11 (path dep, `../../primals/barraCuda/crates/barracuda`)
**License**: AGPL-3.0-or-later

---

## Summary

neuralSpring's validation stack is now three layers deep:

1. **Python baselines** → validate science correctness (397/397 PASS)
2. **Rust validators** → validate faithful porting with centralized tolerances and provenance (1,225 lib tests, 261 binaries)
3. **NUCLEUS composition validators** → validate primal composition patterns (3 validators, bonding policy, proto-nucleate graph)

This handoff documents the composition patterns, absorption candidates, and
deployment readiness for primal teams and the biomeOS Neural API.

---

## What Changed (S177 → S178)

| Area | Change |
|------|--------|
| `validate_all.rs` | Composition validators wired with exit-2 honest skip (PASS/SKIP/FAIL) |
| `docs/PRIMAL_GAPS.md` | Reconciled: inference.* `open` → `wip`, binary naming resolved, Resolved section added |
| Version strings | barraCuda v0.3.7 → v0.3.11 across 3 spec files |
| Deploy graph | V124/S174 → V128/S178 |
| `capability_registry.toml` | Expanded to full 26 capabilities |
| `EVOLUTION_READINESS.md` | Primal IPC wiring status, niche self-knowledge summary |
| All root docs | Aligned to S178/V128 |

---

## For barraCuda Team

### Absorption Candidates (Write→Absorb→Lean)

Two local shaders remain in `metalForge/shaders/` with reasons:

| Shader | Reason for Staying Local |
|--------|-------------------------|
| `head_split.wgsl` | Different binding layout from upstream; historical hang notes |
| `head_concat.wgsl` | Paired with head_split; param struct mismatch |

All other shaders (21/21) have been absorbed upstream.

### Feature-Gate Bug (still open)

`special::plasma_dispersion` unconditionally imports from `ops::lattice::cpu_complex::Complex64`
but `ops::lattice` is behind `#[cfg(feature = "domain-lattice")]`. neuralSpring works around
this by enabling `domain-lattice`. Fix belongs upstream.

### Usage Profile

neuralSpring has 250+ barraCuda import sites across 46 upstream rewires.
Zero duplicate math (2 documented intentional divergences: population vs sample
variance, `primitives.rs` as independent CPU reference). Feature set:
`gpu`, `domain-nn`, `domain-esn`, `domain-genomics`, `domain-timeseries`, `domain-lattice`.

---

## For toadStool Team

### Current Integration

- `Dispatcher::new()` at startup — GPU/CPU capability probing
- `Dispatcher::tensor_session()` for fused multi-op pipelines
- `dispatch_for()` for per-op CPU↔GPU routing
- `primal.forward` routing in async dispatch path

### What We Need

- `compute.dispatch.submit` IPC endpoint for composed-mode dispatch delegation
- Hardware discovery via IPC instead of local `wgpu` enumeration
- `gpu.device.info` for runtime capability queries

---

## For Squirrel Team

### Current State

`inference.complete`, `inference.embed`, `inference.models` are fully registered
in niche, config, capability registry, tarpc service trait, and JSON-RPC handlers.
Handlers return honest stubs (`"provider": "stub", "status": "not_yet_wired"`).

### What We Need

- `inference.register_provider` — neuralSpring registers as ML inference
  provider; Squirrel discovers and routes inference requests
- Provider health monitoring (so Squirrel can fall back to Ollama)
- Model discovery protocol (`inference.models` populated from provider inventory)

### Composition Pattern

```
User → biomeOS Neural API → Squirrel → neuralSpring (WGSL inference)
                                     ↘ Ollama (fallback until native ready)
```

---

## For coralReef Team

### Current State

`metalForge/forge` has optional `coralreef` feature with `coralreef_bridge` module.
Library code compiles shaders directly via `wgpu`. playGround has `CoralReefClient`.

### What We Need

- `shader.compile.wgsl` IPC endpoint accepting the same WGSL source format
  neuralSpring produces (f64-enabled, workgroup annotations)
- Fallback to local wgpu when coralReef unavailable (honest skip in validators)
- `Fp64Strategy` coordination (neuralSpring sends precision hints, coralReef
  applies appropriate lowering for target hardware)

---

## For biomeOS Team

### Deploy Graph

`graphs/neuralspring_deploy.toml` (V128/S178) — sequential coordination:
BearDog → Songbird → optional ToadStool/NestGate/provenance trio →
neuralSpring (by capability `science.spectral_analysis`) → validate → provenance.

### Neural API Integration

neuralSpring exposes 26 capabilities via `capability.list`:
- 14 science methods (spectral analysis, Anderson localization, folding, etc.)
- 3 inference methods (complete, embed, models)
- 4 provenance methods (begin, record, complete, status)
- 2 health probes (liveness, readiness)
- 3 cross-primal (forward, discover, capability.list + compute.offload)

### Deployment Path

```
biomeOS deploys neuralspring_deploy.toml →
  Germinate Tower (BearDog + Songbird) →
  Germinate Node (ToadStool + barraCuda + coralReef) →
  Start neuralspring (spawn = true, by_capability = science) →
  Register capabilities →
  Neural API routes requests by capability
```

### Niche Self-Knowledge

- Bond type: Metallic (shared trust domain)
- Trust model: InternalNucleus (no external endpoints)
- Encryption: Tower=full, Node/Nest/Meta=tower_delegated
- Cost estimates and operation dependencies in `niche.rs`

---

## For Spring Teams

### Reusable Patterns

| Pattern | Module | Reuse Guidance |
|---------|--------|----------------|
| `ValidationHarness` | `validation/mod.rs` | Standard pass/fail with `ValidationSink` |
| `composition.rs` | `validation/composition.rs` | 5-tier discovery, JSON-RPC probes, bonding validation |
| `exit_code_skip_aware` | `validation/composition.rs` | 0=pass, 1=fail, 2=honest skip |
| `niche.rs` | `src/niche.rs` | Capability tables, bonding policy, cost estimates |
| `RetryPolicy` + `CircuitBreaker` | `ipc_resilience.rs` | Exponential backoff + state machine |
| `primal_names.rs` | `src/primal_names.rs` | Canonical short/display names, no ad-hoc strings |

### Three-Layer Validation Evolution

```
Layer 1: Python baseline → Rust validator (science correctness)
  → ValidationHarness + provenance + tolerances

Layer 2: Rust validator → GPU validator (shader promotion)
  → dispatch parity + Tensor API + TensorSession

Layer 3: Rust + Python → Composition validator (NUCLEUS patterns)
  → niche bonding + capability discovery + proto-nucleate graph
  → exit-2 honest skip when ecosystem offline
```

### Evolution Path for Other Springs

Any spring adding composition validation should:
1. Read its proto-nucleate graph (`primalSpring/graphs/downstream/`)
2. Create `src/niche.rs` with capability tables and bonding policy
3. Create `src/validation/composition.rs` (or copy neuralSpring's pattern)
4. Build `validate_{spring}_composition` binary
5. Wire into `validate_all` with exit-2 handling
6. Document gaps in `docs/PRIMAL_GAPS.md`

---

## Metrics (S177 → S178)

| Metric | S177 | S178 |
|--------|------|------|
| Library tests | 1,225 | 1,225 |
| Binaries | 264 | 264 |
| `.rs` files | 518 | 518 |
| Capabilities | 19 | 26 |
| Composition validators | 3 | 3 (now in validate_all) |
| barraCuda | v0.3.11 | v0.3.11 |
| clippy warnings | 0 | 0 |
| Deploy graph | V127/S177 | V128/S178 |

---

## Open Items (hand back to primalSpring)

1. **Squirrel provider registration**: `inference.register_provider` does not exist yet
2. **coralReef shader.compile.wgsl**: neuralSpring ready to route, coralReef endpoint needed
3. **NestGate weight storage**: `storage.retrieve` for model weights
4. **BearDog/Songbird BTSP**: Trust session establishment for composed mode
5. **Proto-nucleate fragment inconsistency**: Graph declares `tower_atomic + node_atomic + meta_tier` but includes optional NestGate — should `nest_atomic` be conditional?
6. **Stale songbird sockets**: Dev machines accumulate stale `/tmp/songbird_*.sock` — composition validators discover them and fail liveness. Need socket cleanup protocol or lease-based discovery.
