# neuralSpring V182 Handoff — Wave 107 Inference Provider Registration

**Date:** 2026-06-10
**Session:** S226
**Gate:** southGate
**Commit:** 15fdf9d

---

## Summary

Wired Squirrel `inference.register_provider` / `inference.unregister_provider`
so neuralSpring can register itself as an inference provider. Promoted
`nest.store` / `nest.commit` from string literals to typed capability constants.
Capability surface expanded 47→51.

## Changes

### Squirrel inference provider lifecycle

- `src/ipc/squirrel.rs`: Added `register_provider()` and `unregister_provider()`
  matching upstream Squirrel's wire format (provider_id, socket, capabilities)
- `src/ipc/mod.rs`: Added `register_as_provider()` and `unregister_provider()`
  to `IpcMathClient` facade; 2 new capability hints (24→26 total)
- `src/capabilities.rs`: `INFERENCE_REGISTER_PROVIDER`, `INFERENCE_UNREGISTER_PROVIDER`

### NestGate signal constants

- `src/capabilities.rs`: `NEST_STORE`, `NEST_COMMIT` constants
- `src/provenance_dispatch.rs`: 3 string literal call sites migrated to constants

### Capability surface (47→51)

| New capability | Owner | Purpose |
|----------------|-------|---------|
| `inference.register_provider` | Squirrel | Provider lifecycle registration |
| `inference.unregister_provider` | Squirrel | Provider lifecycle cleanup |
| `nest.store` | biomeOS (signal) | Content-addressed storage + provenance trio |
| `nest.commit` | biomeOS (signal) | Session commit + provenance finalization |

All synced: `ALL_CAPABILITIES`, `niche::CAPABILITIES`, `capability_registry.toml`,
MCP tools (51 total, `nest` domain added).

### Remaining WIP (from Wave 107 blurb)

- **Startup registration call**: `register_as_provider` is wired but not yet
  called from `neuralspring_primal` startup. Needs Squirrel to be running
  for live E2E.
- **WGSL tokenization pipeline**: Deferred — requires coralReef → toadStool →
  barraCuda shader composition (full pipeline not yet available).
- **NestGate weight persistence live E2E**: Direct `content.put` path works;
  signal path (`nest.store`) needs live NUCLEUS with biomeOS orchestrator.
- **TransportEndpoint adoption**: songBird Wave 107 ships topology-aware
  `TransportEndpoint` / `MeshRelay`. neuralSpring is UDS-only. P2 adoption
  per transport evolution impulse.

## Verification

- **934 workspace tests** (758 lib + 11 integration + 73 forge + 80
  playGround + 12 exp094), 0 failures
- **0 clippy warnings** (pedantic + nursery)
- **51 capabilities** synced across all registries
