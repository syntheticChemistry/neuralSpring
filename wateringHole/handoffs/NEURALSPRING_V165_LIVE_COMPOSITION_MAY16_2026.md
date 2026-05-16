# neuralSpring V165 — Live Composition + Live Data Chains

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**From:** neuralSpring (S209, V165)
**To:** primalSpring, upstream primals
**Date:** 2026-05-16
**Supersedes:** V164 (Wave 20 Schema Standardization)

---

## Summary

neuralSpring evolves from structural/validation-only composition to **live composition with provenance-tracked data chains**. All new code is gated behind `#[cfg(feature = "primalspring")]`.

## What Changed

### Phase 1: Provenance Chain Wiring

1. **`nest.commit` signal dispatch** — `commit_session_signal()` in `weight_loader.rs`
   - Dispatches session finalization through biomeOS: `rhizoCrypt.event.append → bearDog.crypto.sign → nestGate.content.put → loamSpine.session.commit → sweetGrass.braid.create`
   - Use after `nest.store` dispatches to seal training/experiment sessions

2. **Science result provenance** — `store_science_result()` in `weight_loader.rs`
   - Wraps science computation results in `nest.store` provenance
   - Includes method name, domain, and spring metadata
   - Enables provenance tracking for spectral analysis, inference, etc.

3. **Nest commit validation scenario** — `s_nest_commit` (scenario 8/9)
   - Tier 1: structural checks for signal graph awareness, `nest.commit`/`nest.store` wiring, deploy graph fragments
   - Tier 2: live dispatch probes for `nest.store` → `nest.commit` chain, `store_science_result()` exercise
   - Skip-tolerant for missing primals

### Phase 2: Signal Dispatch Evolution

4. **`node.compute` signal dispatch** — `IpcMathClient::dispatch_compute_signal()`
   - Routes GPU compute workloads through biomeOS `node.compute` signal
   - biomeOS decomposes: `toadStool.compute.dispatch → coralReef.shader.compile.wgsl → barraCuda.tensor.matmul`
   - Existing `compute_dispatch()` (direct toadStool IPC) retained for non-composed mode

5. **Schema standard validation scenario** — `s_schema_standard` (scenario 9/9)
   - Tier 1: `capability.list` envelope compliance (count/primal/capabilities), `ALL_CAPABILITIES` naming convention, `primal.list` in local registry
   - Tier 2: live probe of `primal.list` and `capability.list` response shapes

6. **Live IPC pipeline executor** — `execute_graph_live()` in `nucleus_pipeline/executor.rs`
   - Routes pipeline stages through `CompositionContext` instead of local function calls
   - Falls back to local dispatch when capabilities are not resolvable via composition
   - Provenance recording is identical to local executor for uniform `PipelineReport`

## Files Modified

| File | Change |
|------|--------|
| `src/weight_loader.rs` | +`commit_session_signal()`, +`store_science_result()` |
| `src/ipc/mod.rs` | +`dispatch_compute_signal()` on `IpcMathClient` |
| `src/nucleus_pipeline/executor.rs` | +`execute_graph_live()`, +`dispatch_capability_live()` |
| `src/nucleus_pipeline/mod.rs` | re-export `execute_graph_live` |
| `src/validation/scenarios/s_nest_commit.rs` | **NEW** — provenance chain scenario |
| `src/validation/scenarios/s_schema_standard.rs` | **NEW** — schema standard scenario |
| `src/validation/scenarios/mod.rs` | register 2 new scenarios (7 → 9) |
| `docs/PRIMAL_GAPS.md` | Gaps 15–18 resolved |
| `CHANGELOG.md` | S209 entry |
| `README.md` | V165 summary |

## PRIMAL_GAPS Status

- **Gap 15** (`nest.commit`): `candidate` → **RESOLVED**
- **Gap 16** (schema validation): `candidate` → **RESOLVED**
- **Gap 17** (`node.compute` dispatch): **NEW → RESOLVED**
- **Gap 18** (live IPC pipeline): **NEW → RESOLVED**

## Upstream Hand-backs

### primalSpring
- neuralSpring now exercises `nest.store`, `nest.commit`, and `node.compute` signal graphs — all signal graph definitions validated structurally
- `execute_graph_live()` provides a reference pattern for other springs evolving from local to live composition
- `s_schema_standard` validates the Wave 20 envelope standard in a niche spring context

### biomeOS
- `dispatch_compute_signal()` depends on biomeOS routing `node.compute` to the correct toadStool → coralReef → barraCuda graph
- `execute_graph_live()` routes via `ctx.call(domain, capability, params)` — the domain is extracted from the `capability` string

### nestGate / rhizoCrypt / loamSpine / sweetGrass
- `nest.commit` dispatch now exercised from neuralSpring; live scenario probes the full chain

## Quality

- `cargo check --workspace` — clean
- `cargo clippy --workspace` — clean (no new warnings)
- `cargo test --workspace` — 732 passed, 2 pre-existing env-dependent skips
- Scenario count: 9, all feature-gated behind `guidestone`
- All new code behind `#[cfg(feature = "primalspring")]`
