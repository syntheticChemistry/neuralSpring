# neuralSpring V175 — Wave 50 Covalent HPC

**Date:** 2026-05-25
**Session:** S219
**Gate:** southGate (Ryzen 7 5800X3D, RTX 4060 + 3090s, 128GB DDR4, Pop!_OS 22.04)
**Upstream:** primalSpring Wave 50 — Post-Primordial Absorption + Covalent HPC

---

## Summary

Absorbed primalSpring Wave 50 audit. Fixed the last primordial `target/release/`
hardcode (petalTongue), wired `SONGBIRD_PEERS` for cross-gate mesh seeding,
verified bidirectional cross-gate connectivity with eastGate, and confirmed
NUCLEUS 12/13 ALIVE on southGate.

## What changed

### 1. petalTongue primordial hardcode fixed

`composition_nucleus.sh:396` hardcoded
`$ECO_ROOT/primals/petalTongue/target/release/petaltongue` with `find_binary`
as fallback. Replaced with `find_binary petaltongue` exclusively. This was the
last `target/release/` reference in the script — `grep -rn 'target/release'
tools/` now returns zero hits for primal names.

### 2. SONGBIRD_PEERS wired

After Songbird starts and socket is ready, if `SONGBIRD_PEERS` env var is set,
the script sends a `mesh.init` JSON-RPC call with `bootstrap_peers` array.
Matches primalSpring `nucleus_launcher.sh` pattern exactly. Documented in
script header. Startup banner shows peer addresses.

### 3. Cross-gate mesh verified

| From | To | Method | Result |
|------|----|--------|--------|
| southGate (192.168.4.29:7700) | eastGate (192.168.1.144:7700) | `health.liveness` | `{"status":"alive"}` |
| eastGate (192.168.1.144:7700) | southGate (192.168.4.29:7700) | `mesh.init` | `{"initialized":true}` |
| southGate → eastGate | `mesh.init` | `{"initialized":true}` |
| Both gates | `discovery.peers` | `{"peers":[],"total_count":0}` |

Cross-subnet routing works. `mesh.init` succeeds both directions.
`discovery.peers` returns empty — Songbird v0.2.1 feature gap (initializes
mesh state but does not populate the peer list in this version).

## Deployment results

| Metric | Value |
|--------|-------|
| Primals started | 12/13 (loamSpine: upstream Tokio panic) |
| UDS sockets created | toadstool, barracuda, coralreef, nestgate, sweetgrass, petaltongue |
| Federation | `*:7700` (all interfaces) |
| TCP health | `{"status":"healthy"}` |
| Cross-gate | eastGate reachable, bidirectional |
| SONGBIRD_PEERS | 192.168.1.144:7700 seeded, mesh.init succeeded |

## Known issues

| Issue | Status |
|-------|--------|
| loamSpine Tokio runtime-in-runtime | Upstream bug, persistent |
| rhizocrypt EADDRINUSE on UDS | Recovered to TCP 9400/9401 |
| discovery.peers empty after mesh.init | Songbird v0.2.1 feature gap |
| socat not installed on southGate | Used curl/HTTP for seeding instead |
| SONGBIRD_PEERS UDS seeding WARN | Expected — UDS socat path; TCP seeding works |

## Next steps (from Wave 50 guidance)

- [ ] Begin agentic covalent: explore `science.*` method dispatch across gates
- [ ] Wait for Songbird update with `discovery.peers` population
- [ ] Test cross-gate `capability.call` when peer discovery is functional

## Files changed

- `tools/composition_nucleus.sh` — petalTongue hardcode fix, SONGBIRD_PEERS wiring, banner update
- `graphs/neuralspring_deploy.toml` — V175/S219
- `docs/PRIMAL_GAPS.md` — Gap 32 added
- `CHANGELOG.md` — S219 entry
- All doc headers synced to S219/V175

## Response

neuralSpring Wave 50: NUCLEUS 12/13 on southGate, peers seeded, covalent ready
