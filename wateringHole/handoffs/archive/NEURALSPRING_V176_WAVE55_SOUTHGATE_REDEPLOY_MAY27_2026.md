# neuralSpring V176 — Wave 55 southGate Redeploy

**Date:** 2026-05-27
**Session:** S220
**Gate:** southGate (Ryzen 7 5800X3D, RTX 4060 + 3090s, 128GB DDR4, Pop!_OS 22.04)
**Upstream:** primalSpring Wave 55 — southGate redeploy (Songbird socket hardening)

---

## Summary

Absorbed Wave 55 directive: force-fetched hardened plasmidBin binaries
(v2026.05.27), redeployed NUCLEUS on southGate. All 13 primals start
successfully for the first time. loamSpine Tokio runtime-in-runtime crash
**resolved upstream** in v0.9.16.

## What changed

### 1. Hardened binary fetch

Force-fetched all 13 primal binaries from plasmidBin v2026.05.27. Killed
stale NUCLEUS processes from prior co-resident sessions (wetSpring),
cleaned sockets and sled DB, synced to git checkout.

### 2. NUCLEUS 13/13 started

First time all 13 primals start on southGate:

| Primal | Status |
|--------|--------|
| biomeos | ALIVE |
| beardog | ALIVE — health.liveness OK |
| songbird | ALIVE — health.liveness OK, federation *:7700 |
| skunkbat | ALIVE |
| toadstool | ALIVE |
| barracuda | STARTED then auto-exit (no GPU) |
| coralreef | ALIVE — health.liveness OK |
| nestgate | ALIVE — health.liveness OK |
| squirrel | ALIVE |
| rhizocrypt | ALIVE |
| loamspine | ALIVE — health.liveness OK, BTSP Phase 2 |
| sweetgrass | ALIVE — health.liveness OK |
| petaltongue | ALIVE |

### 3. loamSpine Tokio crash resolved

loamSpine v0.9.16 fixed the `infant_discovery` path that previously panicked
with "Cannot start a runtime from within a runtime." Now gracefully falls
back when no discovery service is configured:

```
WARN: Infant discovery failed: No discovery service found. Continuing without discovery.
INFO: Service state: STARTING → READY
INFO: Service state: READY → RUNNING
```

Clean `STOPPED → STARTING → READY → RUNNING` lifecycle. BTSP Phase 2 active,
UDS JSON-RPC server listening, domain and legacy symlinks created.

### 4. barracuda GPU-less auto-exit

barracuda starts in degraded mode (cpu-shader only), announces to Neural API,
then auto-exits ~34s later. Creates `math-southgate.sock` during its brief
runtime. Upstream environmental limitation — requires GPU/DRM access.
Steady state: 12/13 ALIVE.

## Deployment results

| Metric | Value |
|--------|-------|
| Primals started | 13/13 |
| Steady state | 12/13 (barracuda auto-exit) |
| UDS sockets | 32 (primaries + capability aliases) |
| Federation | *:7700 (all interfaces) |
| TCP health | {"status":"alive"} |
| Cross-gate eastGate | Reachable, mesh.init OK |
| SONGBIRD_PEERS | 192.168.1.144:7700 seeded |

## Upstream notes

- loamSpine Tokio crash: **CLOSED** — resolved in v0.9.16
- barracuda: needs `--no-gpu-exit=false` or `BARRACUDA_KEEP_ALIVE=true` for
  headless compositions without GPU
- Songbird `discovery.peers` still returns empty after `mesh.init` (v0.2.1)

## Files changed

- `graphs/neuralspring_deploy.toml` — V176/S220
- `docs/PRIMAL_GAPS.md` — Gap 33 added
- `CHANGELOG.md` — S220 entry
- All doc headers synced to S220/V176

## Response

neuralSpring Wave 55: NUCLEUS 13/13 started (12/13 steady) on southGate, peers seeded, loamSpine Tokio fix confirmed
