# neuralSpring V177 — Wave 67 Glacial Cutover P0 Investigation

**Session**: S221 | **Date**: 2026-06-01 | **Wave**: 67
**Gate**: southGate (Ryzen 7 5800X3D, 128GB DDR4, Pop!_OS 22.04)

---

## Summary

Absorbed Wave 67 glacial cutover plan. Investigated all three P0 blockers
assigned to southGate: Songbird security socket fix, biomeOS `capability.call`
RPC, and bearDog S4 auth configuration. Deployed NUCLEUS via canonical
`plasmidBin/nucleus_launcher.sh` (10/10 started, 6/10 HEALTHY). Updated all
docs to V177/S221.

---

## P0 Investigation Results

### P0-1: Songbird security socket fix (BLOCKER)

**Finding**: Songbird v0.2.1 exposes `--security-socket` CLI flag with
3-tier capability discovery chain (`$SECURITY_PROVIDER_SOCKET` env var,
`$XDG_RUNTIME_DIR/biomeos/security.sock`, `$BEARDOG_SOCKET` legacy). The
flag is parsed but the internal `songbird_http_client` module hardcodes
`/tmp/neural-api-*.sock` for socket construction. Source fix required in
Songbird — repo not cloned on southGate.

**Action needed**: Fix `songbird_http_client` to read the `--security-socket`
value. Ship new binary to plasmidBin. Wire `--security-socket` in
`nucleus_launcher.sh`.

### P0-2: biomeOS `capability.call` RPC (BLOCKER)

**Finding**: biomeOS `crates/biomeos-api/src/unix_server.rs` already has
the proxy infrastructure:
- `NEURAL_API_PROXY_METHODS` const includes `"capability.call"` (line 169)
- `dispatch_jsonrpc_line_async()` routes to `proxy_to_neural_api()`
- `proxy_to_neural_api()` discovers Neural API socket and forwards

The -32601 occurs because:
1. TCP path enforces BTSP auth, rejecting raw JSON-RPC
2. Sync dispatch fallback (`dispatch_jsonrpc_line`) lacks the proxy
3. Neural API socket must be running for the proxy to succeed

**Action needed**: Ensure TCP handler uses async dispatch (or allow
local inter-primal JSON-RPC without BTSP). Ensure `biomeos neural-api`
runs as part of the composition.

### P0-3: bearDog S4 auth config

**Finding**: bearDog v0.9.0 on southGate (0.0.0.0:9100) passes Phase 4
health sweep. `configs/production.toml` has Ed25519 auth, strict policy,
MFA, and sovereignty compliance. No source fix needed — ironGate should
schedule the formal 7-day shadow validation window.

**Action needed**: ironGate to begin formal S4 validation against
`southgate:9100`.

---

## Deployment

| Item | Value |
|------|-------|
| Launcher | `plasmidBin/nucleus_launcher.sh` (canonical) |
| Composition | `nucleus` (10 primals) |
| Started | 10/10 |
| Healthy | 6/10 (beardog, songbird, toadstool, coralreef, nestgate, loamspine) |
| Peers | `--peers 192.168.1.144:7700` (eastGate) |
| Federation port | Songbird :9200 TCP (federation port not bound in this session — `--port` only) |

---

## Cross-Gate Mesh Readiness

southGate is ready for Phase 1 mesh validation:
- `niche.rs` `science_semantic_mappings()` provides capability routing maps
- `primal_names.rs` `domains::` module ready for `capability.call` dispatch
- Federation peer seeding tested in prior waves (eastGate reachable)
- Awaiting Songbird security socket fix and biomeOS `capability.call` proxy
  before `discovery.peers` and `s_covalent_mesh` live tests

---

## Files Changed

| File | Change |
|------|--------|
| `docs/PRIMAL_GAPS.md` | Gap 34 added (Wave 67 P0 investigation) |
| `experiments/results/gap-status.json` | Gap 34 entry, total=34, wip=1 |
| `CHANGELOG.md` | S221/V177 entry (glacial cutover investigation) |
| `graphs/neuralspring_deploy.toml` | Version bumped to S221/V177 |
| `README.md` | Header updated, tools/ tree corrected (fossilized) |
| `CONTEXT.md` | Launcher path updated to plasmidBin canonical |
| `EVOLUTION_READINESS.md` | Header updated S221/V177 |
| `CONTROL_EXPERIMENT_STATUS.md` | Header updated S221/V177 |
| `DEPRECATION_MIGRATION.md` | Session range updated |
| `specs/README.md` | Header updated S221/V177 |
| `experiments/results/*.json` | All 6 session/date stamps updated |
| `wateringHole/handoffs/` | V176 archived, V177 active |

---

## Upstream Notes for primalSpring

1. **Songbird**: Source fix in `songbird_http_client` — must honor
   `--security-socket` / env var chain. New plasmidBin binary needed.
2. **biomeOS**: Proxy code exists but is unreachable via TCP due to BTSP
   enforcement. Either (a) add `--allow-local-rpc` for inter-primal TCP,
   or (b) ensure the async dispatch is used for all transports.
3. **bearDog S4**: Ready for formal 7-day gate. ironGate should schedule.
4. **`discovery.peers`**: Still empty in Songbird v0.2.1 after `mesh.init`.
   Separate from the security socket fix — peer list population is a
   feature gap.
5. **Canonical launcher**: `composition_nucleus.sh` fossilized in Wave 63.
   All southGate deployments now use `plasmidBin/nucleus_launcher.sh`.

---

*V177. Wave 67. Glacial blockers investigated — ready for upstream fixes.*
