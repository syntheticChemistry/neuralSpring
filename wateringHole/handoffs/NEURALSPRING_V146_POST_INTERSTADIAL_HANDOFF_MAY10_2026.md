<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V146 — Post-Interstadial River Delta Evolution Handoff

**Date:** May 10, 2026 (Session S196)
**Supersedes:** V145 (S195 — Doc Reconciliation & Upstream Primal Handoff)
**Audit source:** primalSpring post-interstadial river delta spring evolution guidance (May 10, 2026)

---

## 1. Audit Response Summary

This handoff documents neuralSpring's response to the primalSpring post-interstadial audit. All 6 shared targets addressed; all 3 neuralSpring-specific items resolved.

| Target | Status | Action |
|--------|--------|--------|
| Tier 4 rewiring (JH-11) | **Progressed** | barraCuda already `optional = true`; 11 modules feature-gated; `CompositionContext` wired |
| CI cross-sync (403 methods) | **Resolved** | Updated 389→403 in test comment; cross-sync test validates shared caps against canonical registry |
| skunkBat audit logging (JH-5) | **Wired** | `primal_names::SKUNKBAT` + `src/ipc/skunkbat.rs` + `germinate_skunkbat` deploy graph node |
| composition.status absorption | **Absorbed** | Added to `ALL_CAPABILITIES`, `niche::CAPABILITIES`, `capability_registry.toml` |
| method.register absorption | **Absorbed** | Added to `ALL_CAPABILITIES`, `niche::CAPABILITIES`, `capability_registry.toml` |
| Push sovereignty | **Maintained** | BearDog TLS, Songbird NAT traversal already in deploy graphs; zero external service deps |
| CONTEXT.md alignment | **Resolved** | Full rewrite: eukaryotic architecture, UniBin, 6-module IPC tree, Layer 5 evolution |
| ToadStool/barraCuda sync | **Tracked** | IPC tree modules track upstream dispatch contracts; per-primal isolation |
| Evoformer/folding IPC hooks | **Validated** | Structural tests for evoformer_block, structure_module, folding_health capabilities |

## 2. Code Changes (S196)

### New files
- `src/ipc/skunkbat.rs` — `security.audit_log` IPC surface

### Modified files
- `src/config.rs` — 3 new capabilities, 389→403 comment fix, expanded skip list for biomeOS-originated methods
- `src/primal_names.rs` — `SKUNKBAT` constant + `display::SKUNKBAT`
- `src/ipc/mod.rs` — 6-slot `IpcMathClient` (was 5), `PrimalSlot::Skunkbat`, `audit_log()` method, `has_skunkbat()`, clippy backtick fixes
- `src/niche.rs` — 3 new capabilities, 2 new structural tests
- `config/capability_registry.toml` — 3 new entries (composition.status, method.register, security.audit_log)
- `graphs/neuralspring_deploy.toml` — `germinate_skunkbat` Tower phase node
- `CONTEXT.md` — full rewrite for eukaryotic architecture

### Test delta
- 1,295 → 1,297 lib tests (+2: `evoformer_folding_capabilities_present`, `composition_and_security_capabilities_present`)
- 1,448 → 1,450 workspace tests

## 3. Capability Surface (33 methods)

| Domain | Count | Methods |
|--------|-------|---------|
| Science | 14 | `science.spectral_analysis`, `science.anderson_localization`, `science.hessian_eigen`, `science.agent_coordination`, `science.ipr`, `science.disorder_sweep`, `science.training_trajectory`, `science.evoformer_block`, `science.structure_module`, `science.folding_health`, `science.gpu_dispatch`, `science.cross_spring_provenance`, `science.cross_spring_benchmark`, `science.precision_routing` |
| Health | 3 | `health.liveness`, `health.readiness`, `health.check` |
| Inference | 3 | `inference.complete`, `inference.embed`, `inference.models` |
| Provenance | 4 | `provenance.begin`, `provenance.record`, `provenance.complete`, `provenance.status` |
| Routing | 6 | `primal.forward`, `primal.discover`, `capability.list`, `identity.get`, `mcp.tools.list`, `compute.offload` |
| Composition | 2 | `composition.status`, `method.register` |
| Security | 1 | `security.audit_log` |

## 4. IPC Tree (6 per-primal modules)

| Module | Primal | Capabilities |
|--------|--------|-------------|
| `barracuda` | barraCuda | `stats.*`, `tensor.*` |
| `toadstool` | toadStool | `compute.dispatch` |
| `beardog` | BearDog | `crypto.hash` |
| `squirrel` | Squirrel | `inference.*` |
| `coralreef` | coralReef | `shader.compile.*` |
| `skunkbat` | skunkBat | `security.audit_log` |

## 5. Deploy Graph Primals

`neuralspring_deploy.toml` now germination-orders 7 primals:

1. **Tower**: BearDog (security) → Songbird (discovery) → skunkBat (audit)
2. **Node**: coralReef (shader) → ToadStool (compute) → barraCuda (math)
3. **Nest**: NestGate (storage) + provenance trio (rhizoCrypt, loamSpine, sweetGrass)
4. **Meta**: Squirrel (inference)
5. **Niche**: neuralSpring (science)

## 6. Upstream Audit Data Corrections

The primalSpring audit subsection for neuralSpring (§4 in `SPRING_NUCLEUS_AUDIT_MAY2026.md`) contains stale information:

| Audit claim | Actual state |
|-------------|-------------|
| "No `src/ipc/` directory" | `src/ipc/` exists with 6 per-primal modules (since S193) |
| "Only 1 deploy graph" | 4 deploy graph TOMLs (since S193) |
| "V138" | S196 (current) |
| "IPC in `ipc_dispatch.rs`" | Fossilized; `src/ipc/` is the active IPC tree |
| "~1,225+ tests" | 1,297 lib tests / 1,450 workspace |

The **matrix row** (top-level table) is more current than the narrative subsection.

## 7. Remaining Interstadial Targets

| Target | Priority | Notes |
|--------|----------|-------|
| guideStone L4 (NUCLEUS deploy) | Medium | Requires live biomeOS graph execution validation |
| guideStone L5 (primal proof) | Low | Proto-nucleate is declared; graph execution pending |
| petalTongue wiring | Low | Visualization primal — neuralSpring has `visualization::stream` but no live petalTongue composition |
| sweetGrass wiring | Low | Provenance braids — trio modules exist but braids not composed |
| Tier 4 full IPC-first | Medium | 11 modules feature-gated; remaining library imports to convert |

---

**Quality gate:** 1,297 lib + 73 forge + 80 playGround = 1,450 workspace tests. `cargo build + clippy + test` all clean. Zero `#[allow()]`.
