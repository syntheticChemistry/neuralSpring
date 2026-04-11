# NEURALSPRING V130 — Composition Evolution: Deployment Triad, MCP, Identity

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

> **Session:** S180 | **Date:** 2026-04-11 | **Spring:** neural-spring 0.1.0
> **Scope:** Absorb hardened composition patterns from primalSpring, wateringHole, plasmidBin

---

## Summary

Comprehensive audit revealed 10 action items across neuralSpring, primalSpring
graphs, and plasmidBin metadata. All executed in this session.

### What changed

#### neuralSpring (this repo)

1. **Clippy fix** — `metalForge/forge/src/coralreef_bridge.rs`: added
   `#[expect(clippy::unwrap_used)]` to `compile_without_feature_returns_not_available`
   test. Clippy now passes clean across the workspace.

2. **MCP tool definitions parity** — `playGround/src/mcp_tools.rs`: added 8
   MCP tool definitions for `provenance.begin`, `provenance.record`,
   `provenance.complete`, `provenance.status`, `primal.forward`,
   `primal.discover`, `capability.list`, `compute.offload`. Updated test
   domain list and count assertion (19 → 27). All playGround tests pass.

3. **Deployment health triad** — `src/bin/neuralspring_primal/handlers.rs`:
   added `handle_health_check` implementing `health.check` per
   `DEPLOYMENT_VALIDATION_STANDARD.md`. Returns combined liveness + readiness
   for benchScale and plasmidBin smoke tests.

4. **Identity endpoint** — same file: added `handle_identity_get` implementing
   `identity.get` per Ecosystem Compliance Matrix T4 (discovery tier). Returns
   primal name, niche, version, domain, license, and full capability list.

5. **MCP tools on primal surface** — same file: added `handle_mcp_tools_list`
   implementing `mcp.tools.list` per hotSpring composition pattern. Returns
   all 27 capabilities as discoverable tools with domain parsed from
   `domain.verb` naming convention.

6. **Dispatcher wiring** — `src/bin/neuralspring_primal/main.rs`: wired
   `health.check`, `identity.get`, `mcp.tools.list` into `dispatch_sync`.

7. **Method normalization** — `src/bin/neuralspring_primal/rpc.rs`: evolved
   `normalize_method` from single-prefix strip to iterative multi-prefix loop
   handling `neuralspring.`, `neural-spring.`, `neural_spring.` per
   `SPRING_COMPOSITION_PATTERNS` §1.

8. **Deploy graph evolution** — `graphs/neuralspring_deploy.toml`: added
   `nest_atomic` to fragment list; added `health.check`, `identity.get`,
   `mcp.tools.list` to `capabilities_provided`; bumped to V130/S180.

9. **Gap log** — `docs/PRIMAL_GAPS.md`: documented 6 new resolved items
   (R5–R10) covering MCP parity, deployment triad, fragment alignment,
   upstream graph reconciliation, plasmidBin metadata, method normalization.

#### primalSpring (upstream)

10. **Pipeline graph fix** — `graphs/neuralspring_inference_pipeline.toml`:
    fixed `binary = "neuralspring_primal"` → `"neuralspring"` and
    `health_method = "neural.health"` → `"health.liveness"`.

11. **Deploy graph alignment** — `graphs/spring_deploy/neuralspring_deploy.toml`:
    fixed `binary = "neuralspring_primal"` → `"neuralspring"`, replaced stale
    capability set with the actual 14 science + 3 inference capabilities
    neuralSpring advertises, updated `by_capability` from `"neural"` to
    `"inference"`.

#### plasmidBin (infra)

12. **Metadata refresh** — `neuralspring/metadata.toml`: updated version
    `0.7.0` → `0.1.0`, domain `ml` → `science.learning`, capabilities
    from 2 stale `ml.*` entries to full 30-capability surface, `built_at`
    to `2026-04-11`, UniBin modes to actual subcommands.

13. **Manifest lock** — `manifest.lock`: version and domain updated to match.

---

## Validation

| Check | Result |
|-------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --workspace --all-targets -- -D warnings` | PASS |
| `cargo test --workspace --lib` | PASS (80/80) |
| `cargo doc --workspace --no-deps` (RUSTDOCFLAGS=-D warnings) | PASS |

---

## Patterns absorbed from upstream

| Pattern | Source | Applied |
|---------|--------|---------|
| Health triad (`liveness` + `readiness` + `check`) | `DEPLOYMENT_VALIDATION_STANDARD.md` | handlers.rs |
| `identity.get` (T4 discovery) | `ECOSYSTEM_COMPLIANCE_MATRIX.md` | handlers.rs |
| `mcp.tools.list` on primal | hotSpring V0632 handoff | handlers.rs |
| Iterative multi-prefix method normalization | `SPRING_COMPOSITION_PATTERNS` §1 | rpc.rs |
| Full MCP tool coverage of capability surface | `SPRING_COMPOSITION_PATTERNS` §2 | mcp_tools.rs |
| `nest_atomic` fragment declaration | primalSpring deploy graph | deploy.toml |

---

## Remaining open gaps (hand back)

| Gap | Owner | Status |
|-----|-------|--------|
| §1: Inference stub → live Squirrel | Squirrel + neuralSpring | wip |
| §2: barraCuda direct → IPC | barraCuda + biomeOS | deferred |
| §3: coralReef via IPC | coralReef | open |
| §4: toadStool via IPC | toadStool | open |
| §5: NestGate weight storage | NestGate | open |
| §6: BearDog/Songbird Tower | BearDog + Songbird | open |
| §9: barraCuda feature-gate bug | barraCuda | open (workaround) |
| §10: Shader absorption (29 WGSL) | barraCuda | tracking |
| Proto-nucleate `nest_atomic` fragment | primalSpring | open (§7) |

---

## Evolution narrative

Python was the validation target for our Rust. Now we have both Rust and
Python validation targets for our ecoPrimal NUCLEUS composition patterns.
This session moves neuralSpring from "Rust validation spring" toward
"composition-validated primal" — the primal binary now speaks the full
deployment standard (health triad, identity, MCP tool listing) and the
deploy graphs across all three sources (neuralSpring, primalSpring,
plasmidBin) are reconciled on binary name, health method, capability
set, and fragment declarations.

Next step: wire `inference.*` through Squirrel (gap §1) and begin
TensorSession adoption for fused multi-op GPU pipelines.
