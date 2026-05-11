<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V151 — Doc Reconciliation + Upstream Gap Analysis

**Session**: S200b | **Date**: May 11, 2026 | **Prior**: V150 (S200: guideStone L5 + plasmidBin)

---

## Summary

V151 completes a deep documentation reconciliation pass across all living docs,
updates frozen validation artifacts (JSON), and delivers a comprehensive upstream
gap analysis for primal and spring teams. All code debt from S200b is resolved.

### Changes

1. **Deploy graph version sync** — 3 stale graphs (inference, spectral, math
   pipeline) updated V140/S190 → V150/S200.
2. **Handler doc alignment** — `handlers.rs` comment corrected from "stub
   response" to "`SERVICE_UNAVAILABLE` error" matching S199 implementation.
3. **16 living doc updates** — README, CONTEXT, EVOLUTION_READINESS,
   CONTROL_EXPERIMENT_STATUS, PRIMAL_GAPS, FOUNDATION_SEEDING, specs/README,
   experiments/README, whitePaper/README, baseCamp/README, NOTEBOOK_PATTERN,
   fossilRecord/README, sporeprint/validation-summary — all to S200b/V151.
4. **GuideStone Level 3→5** — corrected in sporeprint, whitePaper, baseCamp,
   EVOLUTION_READINESS, gap-status.json, validation-state.json.
5. **Certification 13→19** — corrected in whitePaper, baseCamp,
   EVOLUTION_READINESS, experiments/README.
6. **validation-state.json** — level 3→5, L4/L5 PENDING→PASS, date corrected.
7. **gap-status.json** — gap 13 (guideStone) resolved, scorecard level 5.
8. **FOUNDATION_SEEDING.md** — stale path `control/spectral_analysis/` →
   `control/anderson_localization/`.
9. **specs/README.md** — stale V135/S184 cross-reference → V151/S200b.

---

## Quality Gates

| Metric | Value |
|--------|-------|
| Library tests | 1,297 (0 failures) |
| Forge tests | 73 |
| playGround tests | 80 |
| Workspace total | 1,450 |
| Certification tests | 19 (L0-L5 ALL PASS) |
| Clippy warnings | 0 |
| Files >800 lines | 0 |
| Production unwraps | 0 |
| Mocks in production | 0 |
| unsafe blocks | 0 |
| Capabilities | 34 |
| guideStone Level | 5 |
| Paper queue | 27/27 closed |

---

## Upstream Gap Analysis — Priority Items for Primal Teams

### P0: NestGate Not Live (Blocks Full Data Chain)

**Owner**: NestGate primal team + primalSpring

neuralSpring deploy graphs declare `germinate_nestgate` with `storage.retrieve`
and `storage.put` capabilities. Spectral analysis and inference pipelines
reference NestGate for weight persistence. However:

- **No `src/ipc/nestgate.rs`** — no typed Rust IPC client exists
- **`discover_data_primal_and_forward`** uses capability-based discovery for
  `data.*` methods but does not implement `storage.retrieve` for model weights
- Without NestGate live, the full data → compute → provenance chain is broken

**Ask**: NestGate team to publish JSON-RPC surface with at minimum:
`storage.put`, `storage.retrieve`, `storage.list`, `health.check`.

### P1: Provenance Trio Not Wired in Rust

**Primals**: rhizoCrypt, loamSpine, sweetGrass

Deploy graphs germinate all three (`dag.session.create`, `commit.session`,
`provenance.create_braid`), but neuralSpring has **no Rust IPC modules** for
them. Current provenance flows are shell-level (`nucleus_composition_lib.sh`).

**Ask**: When these primals stabilize their JSON-RPC surfaces, neuralSpring
needs `src/ipc/{rhizocrypt,loamspine,sweetgrass}.rs` modules to close the
Rust-native provenance pipeline.

### P2: Advertised-but-Undispatched RPC Methods

neuralSpring advertises these capabilities in its registry and niche but
**does not dispatch them** in `dispatch_sync`:

| Method | Advertised | Dispatched |
|--------|-----------|------------|
| `composition.status` | Yes | No |
| `method.register` | Yes | No |
| `compute.dispatch` | Yes | No |
| `security.audit_log` | Yes (library client) | No (RPC) |

These methods work via `primal.forward` delegation to their owner primals,
but a direct RPC call to neuralSpring returns `METHOD_NOT_FOUND`.

**Resolution path**: Either implement local dispatch handlers that delegate
to owner primals, or remove from the advertised capability list and document
as "available via composition only."

### P3: Tier 4 IPC-First (barracuda `optional = true`)

14 library files import `neural_spring_forge` without `#[cfg(feature = "barracuda")]`
gates. The UniBin cannot build without the barraCuda source tree present.

**Pattern**: ludoSpring's dual-path `crate::math` dispatch and wetSpring's
`primal-proof` handler-level feature gating.

**Scope**: Multi-session refactor. Each of the 14 files needs handler-level
dual-path dispatch (in-process compute vs IPC delegation).

---

## For Spring Teams

### Patterns to Absorb

1. **Certification organelle** (6 layers, 19 tests) — `src/certification/`
   implements L0-L5 with sequential validation and early exit. Other springs
   targeting L4+ can reference this structure.

2. **IPC tree** (6 per-primal modules) — `src/ipc/` with `IpcMathClient`
   facade demonstrates clean separation of primal communication concerns.

3. **Inference error honesty** — `SERVICE_UNAVAILABLE` (-32001) instead of
   stub data when upstream primals are absent. All springs should adopt this.

4. **Foundation seeding manifest** — `docs/FOUNDATION_SEEDING.md` documents
   thread contributions with validator paths and tolerance specifications.

### NUCLEUS Composition Patterns

- **Deploy graph**: `neuralspring_deploy.toml` germinates 12 primals with
  `by_capability` discovery. `fallback = "skip"` for optional nodes.
- **biomeOS integration**: `biomeos.rs` uses `nucleus.register`,
  `capability.register`, `nucleus.heartbeat` — failures are non-fatal warnings.
- **Cell membrane**: UniBin `validate` is the Tier 1 entry point for
  NUCLEUS science validation. Deploy graphs feed `composition.deploy(graph)`.
- **plasmidBin**: 2.8M stripped release binary at
  `target/release/neuralspring_unibin`. Two NUCLEUS workloads ready
  (`neuralspring-ml-validation.toml`, `neuralspring-certification.toml`).

---

## Foundation Seeding

Ready to contribute to two threads:

- **Thread 5** (Evolutionary Biology / LTEE): 5 Dolson papers, evolutionary
  dynamics validation, 8 notebooks
- **Thread 7** (Anderson Mathematics): IPR, localization length, level spacing
  statistics, Lyapunov exponents from `control/anderson_localization/`

Contribution path documented in `docs/FOUNDATION_SEEDING.md`.

---

## Next Evolution Targets

1. Close NestGate gap (P0) when upstream ships
2. Wire Rust IPC for provenance trio when surfaces stabilize
3. Resolve advertised-but-undispatched methods (P2)
4. Continue Tier 4 IPC-first refactor (14 files, multi-session)
5. Close remaining open primal gaps (5, 6, 9, 11)

---

*neuralSpring V151 | Session S200b | AGPL-3.0-or-later*
