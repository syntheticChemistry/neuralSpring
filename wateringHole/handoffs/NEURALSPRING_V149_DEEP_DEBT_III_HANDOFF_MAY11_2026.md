<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V149 — Deep Debt Sweep III & Doc Reconciliation Handoff

**Date:** May 11, 2026 (Sessions S198–S199)
**Supersedes:** V147 (S197 — Deep Debt Sweep II, now in `archive/`)
**Upstream:** primalSpring v0.9.25+ | 413 canonical methods | interstadial open

---

## 1. Summary

Sessions S198–S199 complete two sweeps: (1) S198 responded to the primalSpring
post-interstadial river delta audit (CI cross-sync 413, `compute.dispatch`,
UniBin binary alignment, foundation seeding manifest), and (2) S199 eliminated
production mocks, evolved bench error handling, and isolated deprecated APIs.

| Area | Action | Impact |
|------|--------|--------|
| Inference stubs → errors | `handle_inference_{complete,embed,models}` now return `SERVICE_UNAVAILABLE` (-32001) | No mocks in production — honest JSON-RPC error when Squirrel absent |
| Bench unwrap() → expect() | 7 bare `.unwrap()` in `bench_cross_spring_shader_evolution.rs` evolved | Dispatch context in error messages |
| Deprecated API isolation | `#![expect(deprecated)]` on `neuralspring_interactive` + `neuralspring_mcp_adapter` | PrimalClient→CompositionContext migration documented |
| CI cross-sync 403→413 | All docs updated to 413 canonical methods | Zero drift vs primalSpring registry |
| `compute.dispatch` | Added to capability registry, niche, config, MCP tools | 34 capabilities (was 33) |
| UniBin rename | `neuralspring-unibin` → `neuralspring_unibin` | Matches projectNUCLEUS workload TOMLs |
| Foundation seeding | `docs/FOUNDATION_SEEDING.md` with Thread 5 + Thread 7 | Ready for foundation contribution |
| FOUNDATION_SEEDING paths | Corrected stale `src/validate_*.rs` → `src/bin/validate_*.rs` | Accurate validator references |
| Doc reconciliation | All living docs → S199/V149 | 15+ files synchronized |

## 2. Code Changes

### S198 — Post-Interstadial Audit Response II

- `config/capability_registry.toml` — added `composition.status`, `method.register`, `security.audit_log`, `compute.dispatch`
- `src/config.rs` — `ALL_CAPABILITIES` array expanded to 34; CI cross-sync comment 403→413
- `src/niche.rs` — `CAPABILITIES` array expanded to 34; structural tests added
- `playGround/src/mcp_tools.rs` — 4 new `McpToolDef` entries; count 30→34; domains expanded
- `Cargo.toml` — `[[bin]]` name `neuralspring-unibin` → `neuralspring_unibin`
- `src/bin/neuralspring_unibin/cli.rs` — command name aligned to underscore
- `validation/CHECKSUMS` — regenerated

### S199 — Deep Debt Sweep III

- `src/bin/neuralspring_primal/rpc.rs` — `SERVER_ERROR` (dead_code) → `SERVICE_UNAVAILABLE` (-32001)
- `src/bin/neuralspring_primal/handlers.rs` — three inference handlers evolved from success-with-stub to error response; `error_code::` import unified
- `src/bin/bench_cross_spring_shader_evolution.rs` — 7 `.unwrap()` → `.expect()` with context
- `playGround/src/bin/neuralspring_interactive.rs` — `#![expect(deprecated)]` added
- `playGround/src/bin/neuralspring_mcp_adapter.rs` — `#![expect(deprecated)]` added
- `docs/FOUNDATION_SEEDING.md` — validator paths corrected, session bumped

### Test delta

- 1,297 lib + 73 forge + 80 playGround = **1,450 workspace tests**
- 13 certification tests (guidestone feature-gated)
- Zero clippy warnings (pedantic + nursery + cast deny, workspace-wide)
- `neuralspring_unibin validate`: **21/21 ALL PASS**
- `neuralspring_unibin certify`: **29/29 ALL PASS**

## 3. Primal Evolution Review

### IPC tree (6 modules, all `&Path` idiomatic)

| Module | Primal | Capabilities | Status |
|--------|--------|--------------|--------|
| `ipc/barracuda` | barraCuda | `stats.*`, `tensor.*` | Live, 11 functions |
| `ipc/toadstool` | toadStool | `compute.dispatch` | Live, 5 functions |
| `ipc/beardog` | BearDog | `crypto.hash` | Live, 2 functions |
| `ipc/squirrel` | Squirrel | `inference.*` | Live, 3 functions |
| `ipc/coralreef` | coralReef | `shader.compile.*` | Live, 2 functions |
| `ipc/skunkbat` | skunkBat | `security.audit_log` | Live, 1 function |

### Primal dependencies (neuralSpring consumes)

| Primal | How used | Optional? | Feature gate |
|--------|----------|-----------|--------------|
| barraCuda | GPU compute: tensor ops, stats, dispatch | Yes (`optional = true`) | `barracuda` |
| toadStool | GPU workload dispatch orchestration | IPC-only | N/A (IPC) |
| BearDog | TLS hash computation | IPC-only | N/A (IPC) |
| Squirrel | AI inference routing (complete/embed/models) | IPC-only | N/A (IPC) |
| coralReef | Sovereign shader compilation | IPC-only | N/A (IPC) |
| skunkBat | Audit event logging | IPC-only | N/A (IPC) |
| biomeOS | Socket discovery, method registration, composition | Runtime | N/A (env) |

### Production mock elimination (S199)

Before S199, the neuralSpring primal returned `"provider": "stub"` success
responses with empty data when Squirrel was absent. This disguised an
unavailable service as a successful result. Now:

- `inference.complete` → `SERVICE_UNAVAILABLE` error (-32001)
- `inference.embed` → `SERVICE_UNAVAILABLE` error (-32001)
- `inference.models` → `SERVICE_UNAVAILABLE` error (-32001)

Callers get an honest error code and can react accordingly.

## 4. NUCLEUS Composition Patterns

### Deploy graphs (4 active)

| Graph | Purpose | Nodes |
|-------|---------|-------|
| `neuralspring_deploy.toml` | Full spring deploy | 15 nodes |
| `neuralspring_inference.toml` | Squirrel inference composition | 5 nodes |
| `exp094_composition_parity.toml` | Composition parity experiment | 5 nodes |
| `neuralspring_proto_nucleate.toml` | Proto-nucleate validation | 4 nodes |

### NUCLEUS workload readiness

projectNUCLEUS has two workload TOMLs waiting for our UniBin:
- `workloads/neuralspring/neuralspring-ml-validation.toml` → `neuralspring_unibin validate`
- `workloads/neuralspring/neuralspring-certification.toml` → `neuralspring_unibin certify`

**plasmidBin release binary**: `cargo build --release --bin neuralspring_unibin --features guidestone` produces the artifact. Ready for publication.

### Cell membrane mapping

| Zone | neuralSpring component |
|------|----------------------|
| Extracellular (CDN) | sporePrint notebooks, validation-summary.md |
| Membrane (tunnel) | `neuralspring_unibin validate/certify` via toadStool dispatch |
| Intracellular (sovereign) | src/ library + barraCuda GPU compute |

## 5. Guidance for Upstream Primal Teams

### barraCuda team
- neuralSpring validates 34 capabilities over IPC with 1,450 tests
- Tridiagonal eigensolver on GPU (`tridiag_eigh.wgsl`) is the remaining open item
- Our `bench_industry_gpu_parity.rs` validates cuBLAS/cuDNN/cuFFT/FlashAttention parity; Kokkos baselines remain estimated (on-iron measurement is the gap)

### toadStool team
- `compute.dispatch` now registered in neuralSpring capability registry
- IPC dispatch via `ipc/toadstool.rs` uses 5 functions, all `&Path`
- Main evolution axis: absorb dispatch contract changes as they evolve

### skunkBat team
- neuralSpring has full Rust IPC wiring (`ipc/skunkbat.rs`)
- `security.audit_log` registered in capability registry + deploy graphs
- Ready for Phase 3 cross-primal forwarding (rhizoCrypt DAG + sweetGrass braid)

### Squirrel team
- Inference routing is live (`try_squirrel_route` in handlers.rs)
- Service-unavailable error responses are now honest (-32001), not fake success
- Ready to absorb Squirrel provider discovery when it ships

### biomeOS team
- `composition.status` and `method.register` absorbed (S196)
- neuralSpring registers 34 capabilities via `method.register`
- Socket discovery via `BIOMEOS_SOCKET_SUBDIR` (capability-based)

## 6. Guidance for Downstream Springs

### Patterns to absorb

1. **Inference stub → error evolution**: if your spring fakes success when a primal is absent, evolve to `SERVICE_UNAVAILABLE` errors
2. **IPC `&Path` idiom**: all socket-accepting functions should take `&Path`, not `&PathBuf`
3. **`#![expect(deprecated)]` isolation**: playground binaries that use deprecated APIs should carry explicit `#![expect(deprecated, reason = "...")]`
4. **Bench `.expect()` over `.unwrap()`**: benchmark binaries should use `.expect()` with context describing which dispatch failed
5. **Capability registry parity**: validate local registry against primalSpring canonical 413 methods; `compute.dispatch` should be present
6. **UniBin underscore naming**: NUCLEUS workloads expect `{spring}_unibin` (underscore), not hyphen

### Foundation seeding

neuralSpring's validated contributions (see `docs/FOUNDATION_SEEDING.md`):
- **Thread 5** (LTEE): 5 Dolson papers — NK fitness, MODES, eco dynamics, directed evo, swarm
- **Thread 7** (Anderson Math): Lyapunov exponent, IPR, level spacing, Anderson transition, Evoformer

Springs with validated public data should create similar seeding manifests.

## 7. Quality Gates

| Metric | Value |
|--------|-------|
| Workspace tests | 1,450 (1,297 lib + 73 forge + 80 playGround) |
| Certification tests | 13 (guidestone feature-gated) |
| Capabilities | 34 |
| CI cross-sync | 413 methods, zero drift |
| Clippy warnings | 0 (pedantic + nursery + cast deny) |
| `unsafe` blocks | 0 (`forbid(unsafe_code)`) |
| `#[allow()]` | 0 (all `#[expect()]`) |
| Mocks in production | 0 |
| `todo!()`/`unimplemented!()` | 0 |
| Large files (>800L) | 0 (max 776: tolerances/mod.rs) |
| UniBin validate | 21/21 ALL PASS |
| UniBin certify | 29/29 ALL PASS |
| Deploy graphs | 4 |
| IPC modules | 6 (all `&Path` idiomatic) |
| Named tolerances | 233+ |
| Paper notebooks | 8 (72/72 checks) |
| CHECKSUMS | BLAKE3, 15 files, all valid |

---

*neuralSpring V149 | Sessions S198–S199 | AGPL-3.0-or-later*
