<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V173 — Wave 48 Covalent Spring Mesh (southGate Sound-Off)

**Date**: May 25, 2026
**Session**: S217
**Gate**: southGate
**Hardware**: AMD Ryzen 7 5800X3D (8-core), 128GB DDR4, Pop!_OS 22.04
**Co-tenants**: wetSpring

---

## Gate Declaration

neuralSpring declares **southGate** as its deployment gate. Gate Deployment
section added to `CONTEXT.md` per Wave 48 requirements.

| Property | Value |
|----------|-------|
| Gate | southGate |
| Hardware | Ryzen 7 5800X3D, 128GB DDR4 |
| Composition | Full NUCLEUS (13 primals) |
| Federation | Songbird TCP port 7700 |
| Cell graph | `plasmidBin/cells/neuralspring_cell.toml` |

## Federation Enablement

`composition_nucleus.sh` upgraded with `SONGBIRD_FEDERATION_PORT` support:
- When set, Songbird binds TCP for cross-gate LAN discovery
- Default: UDS-only (no TCP, zero ports bound)
- Launch: `SONGBIRD_FEDERATION_PORT=7700 ./tools/composition_nucleus.sh start`

### Verification

```
# Federation port bound
LISTEN 0 128 *:7700 *:* users:(("songbird",...))

# Health check on TCP
curl -s -X POST http://127.0.0.1:7700/jsonrpc \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"health.liveness"}'
→ {"jsonrpc":"2.0","result":{"status":"healthy"},"id":1}

# Peers (0 — first federation gate on this LAN segment)
curl -s -X POST http://127.0.0.1:7700/jsonrpc \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"discovery.peers"}'
→ {"jsonrpc":"2.0","result":{"peers":[],"total_count":0},"id":1}
```

**Note**: Songbird JSON-RPC on TCP is at path `/jsonrpc`, not root `/`.

## Upstream API Sync

`CompositionContext::dispatch()` signature changed in primalSpring (`Value` → `&Value`).
8 call sites updated:
- `src/ipc/mod.rs` — `node.compute` dispatch
- `src/nucleus_pipeline/executor.rs` — `node.compute` dispatch
- `src/provenance_dispatch.rs` — `nest.store` (2×) + `nest.commit` (1×)
- `src/validation/scenarios/s_nest_commit.rs` — `nest.store` + `nest.commit`
- `src/validation/scenarios/s_signal_dispatch.rs` — `nest.store`

## Songbird Sled Database Issue

Prior unclean shutdowns left corrupted sled task lifecycle database at
`~/.local/share/songbird/`. This caused `Error: task lifecycle database:
Failed to create task storage` on startup. **Fix**: clean
`~/.local/share/songbird/task_lifecycle*` before first federation launch.

Also cleaned: hundreds of stale `songbird_security_provider_*.sock` files
and `songbird_bad_pid_*.pid` files from prior crash loops in `/tmp/`.

## Deployment Status

| Primal | Transport | Status |
|--------|-----------|--------|
| biomeOS | UDS | UP |
| bearDog | UDS | UP |
| Songbird | UDS + TCP:7700 | UP (federation) |
| skunkBat | TCP (no UDS) | UP |
| toadStool | UDS | UP |
| barraCuda | UDS | UP |
| coralReef | UDS | UP |
| NestGate | UDS | UP |
| Squirrel | abstract socket | UP |
| rhizoCrypt | TCP only | UP |
| loamSpine | — | FAIL (upstream Tokio bug) |
| sweetGrass | UDS | UP |
| petalTongue | UDS | UP |

## Test Results

- **754/754** lib tests pass (IPC-first, `--no-default-features`)
- **guideStone** 29/29 PASS (4 SKIP — offline primals)
- **neuralspring binary** builds clean with `--features "primal,barracuda,guidestone"`, zero warnings

## Files Changed

- `CONTEXT.md` — Gate Deployment section added
- `tools/composition_nucleus.sh` — `SONGBIRD_FEDERATION_PORT` support, federation banner
- `src/ipc/mod.rs` — dispatch `Value` → `&Value`
- `src/nucleus_pipeline/executor.rs` — dispatch `Value` → `&Value`
- `src/nucleus_pipeline/dispatch.rs` — unused `mut` removed
- `src/provenance_dispatch.rs` — dispatch `Value` → `&Value` (3 sites)
- `src/validation/scenarios/s_nest_commit.rs` — dispatch `Value` → `&Value` (2 sites)
- `src/validation/scenarios/s_signal_dispatch.rs` — dispatch `Value` → `&Value`
- `graphs/neuralspring_deploy.toml` — S217/V173
- `CHANGELOG.md` — S217 entry
- `validation/CHECKSUMS` — date updated
- Doc headers synced to S217/V173
