# neuralSpring V180 — Forward Evolution Handoff

**Spring:** neuralSpring (southGate)
**Session:** S224 | **Date:** 2026-06-04
**Gate:** southGate (AMD Ryzen 7 5800X3D, 128GB DDR4, Pop!_OS 22.04)
**Supersedes:** V179 (S223 — deep debt evolution)

---

## Mission

Forward evolution per primalSpring Wave 77 directive: cross-gate composition
scenarios, ML IPC wiring, MCP tools expansion, deep debt cleanup.

## Test Results

| Metric | Value |
|--------|-------|
| `cargo test --workspace` | **932 passed, 0 failed** |
| Lib tests (neural-spring) | 756 passed |
| Playground tests | 80 passed |
| Forge tests | 73 passed |
| Integration tests | 11 passed |
| Exp094 tests | 12 passed |
| Clippy (workspace, all-features) | **0 warnings** |

## Changes — S224 Forward Evolution

### Cross-Gate Dispatch Scenario (new)

`s_cross_gate_dispatch.rs` — 12-check scenario validating cross-gate
dispatch patterns:

| Check | Description |
|-------|-------------|
| Tier 1 (7) | Capability constants defined, dotted notation, all 4 new caps in registry, count = 47 |
| Tier 2 (5) | Live `ml.mlp_infer` → barraCuda, `crypto.btsp_handshake` → BearDog, `discovery.peers` → Songbird, `mesh.init` → Songbird, BLAKE3 integrity |

New `CrossGate` track in scenario registry. 11 total scenarios (was 10).

### ML IPC Wiring

`ml.mlp_infer` capability routed to barraCuda via IPC:
- `src/capabilities.rs`: `ML_MLP_INFER` constant
- `src/ipc/barracuda.rs`: `ml_mlp_infer()` function (input vector + layer spec)
- `CAPABILITY_HINTS`: `ml.mlp_infer → barracuda`

### Mesh + Trust IPC

| Capability | Primal | IPC Module |
|------------|--------|------------|
| `discovery.peers` | Songbird | hint only (no dedicated module yet) |
| `mesh.init` | Songbird | hint only |
| `crypto.btsp_handshake` | BearDog | `src/ipc/beardog.rs`: `btsp_handshake()` |

### MCP Tools Expansion (43 → 47)

| New Tool | Domain | Description |
|----------|--------|-------------|
| `ml.mlp_infer` | ml | barraCuda MLP forward inference |
| `discovery.peers` | discovery | Songbird mesh peer discovery |
| `mesh.init` | mesh | Songbird mesh initialization |
| `crypto.btsp_handshake` | crypto | BearDog BTSP trust handshake |

### Deep Debt

- `target/release/` fallbacks removed from `scripts/visualize.sh` and
  `scripts/validate_clean_machine.sh` — error with `plasmidbin install` guidance
- `specs/NUCLEUS_TOWER_INTEGRATION.md` systemd example → `~/.local/bin/`
- 10+ docs reconciled from stale S209-S216 stamps to S224

## Capability Surface

47 capabilities (was 43). 4 new marked `experimental` stability:
`ml.mlp_infer`, `discovery.peers`, `mesh.init`, `crypto.btsp_handshake`.

## Upstream Gaps

| # | Gap | Owner | Priority |
|---|-----|-------|----------|
| G-01 | Songbird IPC module (dedicated `src/ipc/songbird.rs`) | neuralSpring | P2 |
| G-02 | `ml.mlp_infer` upstream validation (barraCuda must expose JSON-RPC method) | barraCuda | P2 |
| G-03 | Discovery triplication consolidation | neuralSpring | P2 |

## ACK

neuralSpring on southGate: S224 forward evolution complete. 932 tests,
0 warnings. Cross-gate dispatch scenario wired. ML inference routable via
capability discovery. MCP surface expanded for mesh + trust. Ready for
cross-gate live validation.
