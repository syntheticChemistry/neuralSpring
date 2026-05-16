# neuralSpring V161 — Wave 17 Signal API Adoption Handoff

**From:** neuralSpring (S207)
**To:** primalSpring, biomeOS, all spring teams
**Date:** 2026-05-16
**Session:** S207 — Wave 17 signal API adoption per primalSpring audit

---

## Summary

neuralSpring has adopted the Wave 17 signal API from primalSpring:

- **`primal.announce`**: Single-call registration replaces `nucleus.register` + N × `capability.register`
- **`nest.store` dispatch**: Weight persistence via `ctx.dispatch("nest.store", ...)` with biomeOS managing the provenance graph
- **Validation scenario**: `s_signal_dispatch` validates structural and live signal adoption
- **GAP-GS-015**: Verified fixed — `cargo check --workspace` passes against primalSpring HEAD

---

## Changes Made

### 1. Registration: `primal.announce` with Fallback

**`src/bin/neuralspring_primal/biomeos.rs`** — the primary registration
function now sends a single `primal.announce` JSON-RPC call to biomeOS:

```json
{
  "method": "primal.announce",
  "params": {
    "primal_id": "neural-spring",
    "transport": "/run/user/.../neuralspring-family.sock",
    "methods": ["science.spectral_analysis", "inference.complete", ...],
    "lifecycle": { "state": "running" },
    "signal_tiers": ["node", "nest", "meta"],
    "version": "0.1.0"
  }
}
```

If `primal.announce` fails (pre-v3.57 biomeOS), falls back to the legacy
`nucleus.register` + per-capability `capability.register` loop.

**`playGround/src/biomeos_client.rs`** — `BiomeOsClient::announce()` method
added with same fallback behavior.

### 2. Weight Persistence: `nest.store` Signal Dispatch

**`src/weight_loader.rs`** — new `store_to_nestgate_signal()`:

```rust
pub fn store_to_nestgate_signal(
    path: &Path,
    ctx: &mut CompositionContext,
    author: &str,
) -> Result<serde_json::Value, IpcError>
```

When running inside a biomeOS composition, this dispatches:
`ctx.dispatch("nest.store", { content, content_type, author, filename })`

biomeOS decomposes into: `NestGate.content.put → rhizoCrypt.dag.event.append → loamSpine.spine.seal → sweetGrass.braid.create`

The direct `content.put` path (`store_to_nestgate()`) is retained for
standalone/fallback use.

### 3. Capability Surface: 35 Methods

Added `primal.announce` to:
- `config::ALL_CAPABILITIES` (35 entries)
- `niche::CAPABILITIES`
- `capabilities.rs` constants
- `config/capability_registry.toml`
- `playGround/src/mcp_tools.rs` tool definitions (35 tools)

### 4. RPC Handlers

Added 5 handlers to the primal dispatch table:
- `primal.announce` — acknowledges announcements (niche, not orchestrator)
- `composition.status` — returns composing status + capability count + signal API version
- `method.register` — legacy handler, returns note to use `primal.announce`
- `compute.dispatch` — dispatch readiness check
- `security.audit_log` — skunkBat JH-5 forwarding stub

### 5. Validation: `s_signal_dispatch` Scenario

New scenario (Track::Signal, Tier::Both, 12 checks):

**Tier 1 (Rust structural):**
- `primal.announce` in registry, ALL_CAPABILITIES, and niche CAPABILITIES
- skunkBat triple-first in deploy graph + inference graph
- Node + nest atomic fragments in deploy graph

**Tier 2 (Live):**
- `primal.info` responds via orchestration
- `nest.store` dispatch succeeds or SKIP if composition unavailable

---

## Signal Adoption Summary (per SIGNAL_ADOPTION_STANDARD.md)

| Glacial Priority | Status |
|-----------------|--------|
| Pull primalSpring HEAD for 451-method registry sync | **DONE** — `registry_methods_in_primalspring_canonical` passes |
| Replace registration with `ctx.announce()` | **DONE** — `primal.announce` with fallback |
| `nest.store` for weight/result persistence | **DONE** — `store_to_nestgate_signal()` |
| GAP-GS-015 (`ALL_CAPS`/`BTSP_EXTRA_CAPS` re-export) | **VERIFIED** — `cargo check --workspace` clean |
| ML surrogates for lithoSpore modules 3, 4, 6 | MEDIUM priority — not yet started |
| Threads 5+7 → foundation | Not yet started |

### Signal Adoption Depth

| Signal | neuralSpring Status |
|--------|-------------------|
| `nest.store` | **WIRED** — `store_to_nestgate_signal()` via `ctx.dispatch` |
| `nest.retrieve` | CANDIDATE — weight loading via `load_safetensors_from_nestgate` |
| `nest.commit` | CANDIDATE — session finalization |
| `node.compute` | CANDIDATE — toadStool dispatch pipeline |
| `meta.observe` | CANDIDATE — inference monitoring |
| `meta.intent` | CANDIDATE — squirrel-powered inference planning |
| `tower.*` | Structural (deploy graphs, not runtime dispatch yet) |

---

## Current State

| Metric | Value |
|--------|-------|
| Session | S207 |
| Workspace tests | 910 (2 env-dependent skips) |
| Clippy errors | 0 |
| Capabilities | 35 |
| Validation scenarios | 7 |
| PRIMAL_GAPS open | 3 (Gap 6 BTSP/upstream, Gap 9 barraCuda/upstream, Gap 10 tracking) |
| Signal API | Wave 17 adopted (announce + nest.store) |
| Evolution | composing |
| Handoff | V161 |
