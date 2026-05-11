<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V147 — Deep Debt Sweep II & Doc Reconciliation Handoff

**Date:** May 10, 2026 (Session S197)
**Supersedes:** V146 (S196 — Post-Interstadial River Delta Evolution)
**Upstream:** primalSpring v0.9.25+ | 403 canonical methods | interstadial open

---

## 1. Summary

Session S197 completes a deep debt sweep focused on idiomatic Rust evolution, certification test coverage, integrity validation, citation accuracy, and full documentation reconciliation. All work builds on the S196 post-interstadial audit response (V146).

| Area | Action | Impact |
|------|--------|--------|
| IPC `&PathBuf` → `&Path` | All 6 per-primal modules evolved to idiomatic `&Path` | Zero clippy warnings; 33 pre-existing warnings resolved |
| Certification tests | 13 new tests across `certification/` organelle | `bare.rs` (4), `discovery.rs` (3), `parity.rs` (2), `nucleus.rs` (1), `mod.rs` (3) |
| CHECKSUMS | Regenerated via `gen_checksums` | L0 29/29 PASS (was failing due to stale hashes) |
| Waters citations | Papers 019-021 corrected | Aligned with canonical `specs/PAPER_REVIEW_QUEUE.md` |
| paper-baselines.json | Malformed JSON fixed | Notebooks array was broken at entry 016; now valid |
| Doc reconciliation | All living docs updated to S197 | README, CONTEXT, PRIMAL_GAPS, experiments, whitePaper, sporeprint, deploy graph, notebooks |

## 2. Code Changes (S197)

### Modified files (IPC evolution)
- `src/ipc/barracuda.rs` — `socket: &PathBuf` → `socket: &Path`, `# Errors` doc sections
- `src/ipc/toadstool.rs` — same
- `src/ipc/beardog.rs` — same
- `src/ipc/squirrel.rs` — same
- `src/ipc/coralreef.rs` — same
- `src/ipc/skunkbat.rs` — same (created in S196, already used `&Path`)

### Modified files (certification)
- `src/certification/mod.rs` — 3 structural tests (max_layer, certify return, clamp)
- `src/certification/bare.rs` — 4 tests (validate_all, deterministic_rng, provenance, tolerances)
- `src/certification/discovery.rs` — 3 tests (nonempty, lowercase, core domains)
- `src/certification/parity.rs` — 2 tests (nonempty, core methods)
- `src/certification/nucleus.rs` — 1 test (two-capability validation)

### Modified files (docs + data)
- `validation/CHECKSUMS` — regenerated (BLAKE3, 15 files)
- `whitePaper/baseCamp/waters.md` — citations 019-021 corrected
- `playGround/src/mcp_tools.rs` — 3 new tool definitions, domain list expanded
- `experiments/results/paper-baselines.json` — malformed JSON fixed, session updated

### Test delta
- 1,297 lib + 73 forge + 80 playGround = **1,450 workspace tests**
- +13 certification tests (guidestone feature-gated)
- Zero clippy warnings (pedantic + nursery + cast deny, workspace-wide)

## 3. Current State

| Metric | Value |
|--------|-------|
| Workspace tests | 1,450 (1,297 lib + 73 forge + 80 playGround) |
| Certification tests | 13 (guidestone feature) |
| Capabilities | 34 (14 science + 3 health + 3 inference + 4 provenance + 2 cross_primal + 2 compute + 1 capability + 1 identity + 1 mcp + 1 composition + 1 method + 1 security) |
| IPC modules | 6 (barracuda, toadstool, beardog, squirrel, coralreef, skunkbat) |
| Deploy graphs | 4 TOMLs |
| Clippy warnings | 0 |
| `#[allow()]` | 0 |
| Unsafe code | 0 (`#![forbid(unsafe_code)]`) |
| GuideStone | Level 3 (29/29 bare ALL PASS) |
| Papers | 27 implemented |
| Named tolerances | 233+ |
| barraCuda | v0.3.13 (optional, IPC-first) |
| CI cross-sync | 413 canonical methods |
| Edition | 2024, MSRV 1.87 |

## 4. Primal Evolution & Use Review

### IPC Tree (6 per-primal modules)

| Module | Primal | Key Methods | Status |
|--------|--------|-------------|--------|
| `barracuda` | barraCuda | `compute.dispatch.submit`, tensor ops | `optional = true`, IPC-first |
| `toadstool` | toadStool | `gpu.dispatch`, `shader.compile` | Discovery wired |
| `beardog` | bearDog | TLS, signing, identity verification | Tower phase |
| `squirrel` | squirrel | `inference.complete/embed/models` | Dynamic fallback routing |
| `coralreef` | coralReef | PCIe broker, sovereign dispatch | Node phase |
| `skunkbat` | skunkBat | `security.audit_log` | JH-5 forwarding (S196) |

All modules use idiomatic `&Path` for socket arguments (evolved from `&PathBuf` in S197).

### Composition Patterns for NUCLEUS

neuralSpring's NUCLEUS composition follows the 4-tier deploy graph pattern:

1. **Tower phase**: bearDog TLS + Songbird discovery + skunkBat audit
2. **Node phase**: barraCuda compute + toadStool dispatch + coralReef PCIe
3. **Nest phase**: rhizoCrypt provenance + sweetGrass braid
4. **Cross-Atomic**: Full science pipeline with capability routing

Deploy via `neuralAPI` from `biomeOS`:
- `biomeOS` resolves neuralSpring via socket directory (`BIOMEOS_SOCKET_SUBDIR`)
- 33 JSON-RPC capabilities registered dynamically (`method.register` from biomeOS v3.51)
- `composition.status` provides `{ active_users, primal_health, resource_pressure }`
- `CompositionContext::from_live_discovery_with_fallback()` for sovereign IPC

### Upstream Guidance for Primal Teams

**barraCuda**: neuralSpring consumes 11 feature-gated modules. Dispatch contract evolution should maintain backward compatibility for `compute.dispatch.submit` and tensor ops. 18 surface gaps still tracked (PRIMAL_GAPS appendix).

**toadStool**: Pipeline graph DAG absorbed upstream. neuralSpring tracks discovery contract changes via `src/ipc/toadstool.rs`.

**bearDog**: TLS and signing are deployed in Tower phase. No evolution needed from neuralSpring side.

**squirrel**: Dynamic routing via `try_squirrel_route()` fallback. Inference contract is stable.

**coralReef**: PCIe broker integration via `src/ipc/coralreef.rs`. 8 neuralSpring shaders in coralReef corpus.

**skunkBat**: JH-5 forwarding wired (S196). `security.audit_log` calls route to rhizoCrypt DAG + sweetGrass braid. When Phase 3 ships, cross-primal audit forwarding activates automatically.

### Downstream Pattern Absorption

**projectNUCLEUS** (sporeGarden): neuralSpring's deploy graphs, capability registry, and certification organelle serve as reference patterns for NUCLEUS composition validation.

**foundation** (sporeGarden): Threads 5 (intelligence) and 7 (sovereignty) consume neuralSpring's inference routing and IPC-first patterns.

## 5. Remaining Evolution Targets

| Target | Priority | Notes |
|--------|----------|-------|
| guideStone L4 (NUCLEUS deployment) | Medium | Requires live NUCLEUS composition environment |
| Tier 4 IPC-first for 11 barraCuda modules | Medium | JH-11 resolved; feature-gating in place |
| 18 barraCuda surface gaps | Low | Tracked in PRIMAL_GAPS appendix; dependent on upstream |
| Evoformer live IPC validation | Medium | Structural tests pass; live barraCuda IPC pending |
| TBD items in planning docs | Low | `MSA_DATABASE_PLAN.md`, `MIXED_HARDWARE_DESIGN.md` — planning-phase placeholders |

## 6. Archive & Debris Status

- **fossilRecord/**: 3 fossils with provenance READMEs (ipc_dispatch, guidestone, primal_server — all prokaryotic-era)
- **No** backup/temp/`.bak`/`.pyc` files
- **No** empty directories
- **No** orphaned test data
- **Zero** `#[deprecated]` in `src/` except `ipc_dispatch` (fossilized, kept for migration path)
- **Clean** `.gitignore` coverage for build artifacts

---

*neuralSpring V147 | Session S197 | May 10, 2026 | AGPL-3.0-or-later*
