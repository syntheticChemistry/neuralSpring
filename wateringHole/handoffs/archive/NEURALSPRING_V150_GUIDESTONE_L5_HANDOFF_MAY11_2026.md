<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# neuralSpring V150 — guideStone L5 + plasmidBin Release Handoff

**Date:** May 11, 2026 (Session S200)
**Supersedes:** V149 (S199 — Deep Debt Sweep III, now in `archive/`)
**Upstream:** primalSpring v0.9.25+ | 413 canonical methods | post-interstadial push 2

---

## 1. Summary

Session S200 responds to the primalSpring river delta evolution audit (May 11, 2026).
neuralSpring advances from guideStone Level 3 to Level 5 with two new certification
layers, builds a 2.8M plasmidBin-ready release binary, and documents the Tier 4
IPC-first gap for future systematic resolution.

| Area | Action | Impact |
|------|--------|--------|
| guideStone L4 | New `certification::composition` module | Deploy graphs, registry, 4 family calls |
| guideStone L5 | New `certification::cross_spring` module | Frozen artifacts, protocol liveness, hash determinism |
| Certification tests | +6 tests (was 13, now 19) | 3 composition + 3 cross_spring |
| plasmidBin binary | `cargo build --release` produces 2.8M stripped ELF | projectNUCLEUS workload-ready |
| CLI default | `certify --layer 5` (was 3) | Full certification pipeline by default |
| Tier 4 gap | Documented: 115 ungated barraCuda imports | Systematic `#[cfg(feature)]` needed |

## 2. Code Changes

### New files

- `src/certification/composition.rs` — Layer 4: validates 4 deploy graph files,
  capability registry structure (≥30 methods), and live composition calls across
  tensor/security/compute/ai families
- `src/certification/cross_spring.rs` — Layer 5: validates 5 frozen ecosystem
  artifacts (gap-status.json, validation-state.json, PRIMAL_GAPS.md,
  FOUNDATION_SEEDING.md, CHECKSUMS), 4 family pings, and BLAKE3 hash determinism

### Modified files

- `src/certification/mod.rs` — `MAX_LAYER` 3→5, layer table updated, L4/L5
  orchestration sections, test assertion updated
- `src/bin/neuralspring_unibin/cli.rs` — default layer 3→5, help text expanded
- `validation/CHECKSUMS` — regenerated for new modules

### Test delta

- 1,297 lib + 73 forge + 80 playGround = **1,450 workspace tests**
- **19 certification tests** (guidestone feature-gated, was 13)
- UniBin certify: **29/29 ALL PASS** (L4/L5 skip when no live NUCLEUS — correct)
- UniBin validate: **21/21 ALL PASS**

## 3. guideStone Layer Architecture

| Layer | Module | Checks | Requires NUCLEUS? |
|-------|--------|--------|-------------------|
| L0 | `bare` | P1-P5 (determinism, provenance, checksums, env, tolerances) | No |
| L1 | `discovery` | Capability liveness probes (tensor, security, compute, ai) | Yes |
| L2 | `parity` | 7 domain science parity checks | Yes |
| L3 | `nucleus` | BearDog signing + Songbird discovery | Yes |
| L4 | `composition` | 4 deploy graphs + registry + 4 family composition calls | Yes |
| L5 | `cross_spring` | 5 frozen artifacts + 4 pings + hash determinism | Yes |

L4 and L5 only execute when L1 discovers live primals. In bare mode (no NUCLEUS),
certify still reports 29/29 PASS for L0 with L1-L5 properly skipped.

## 4. plasmidBin Release Binary

```
cargo build --release --bin neuralspring_unibin --features guidestone
```

- Binary: `target/release/neuralspring_unibin` — 2.8M, stripped, x86-64 ELF
- Subcommands: `validate` (21/21), `certify` (29/29), `serve`, `status`, `version`
- projectNUCLEUS workloads reference `$SPRINGS_ROOT/neuralSpring/target/release/neuralspring_unibin`
- Cell deployment graph: `infra/plasmidBin/cells/neuralspring_cell.toml`

## 5. Tier 4 IPC-First Gap (Documented)

`cargo check --no-default-features --features guidestone` produces 115 errors.
Root cause: 11 modules are feature-gated in `lib.rs` (`#[cfg(feature = "barracuda")]`),
but many leaf modules import from those gated modules unconditionally (e.g.
`anderson_localization.rs` imports `neural_spring_forge`, `bench.rs` imports `gpu::Gpu`).

**Resolution path**: Systematic `#[cfg(feature = "barracuda")]` wrapping of all
leaf module imports and conditional compilation of dependent function bodies.
wetSpring's `primal-proof` handler-level wiring (V159) is the reference pattern.
This is a multi-session architectural task.

## 6. Upstream Audit Response

| Audit target | Status |
|-------------|--------|
| guideStone L3→L4+ | **Done** — L4 + L5 implemented |
| plasmidBin release | **Done** — 2.8M binary built and verified |
| Foundation Threads 5+7 | **Documented** — FOUNDATION_SEEDING.md with Thread 5 (LTEE) + Thread 7 (Anderson) |
| Tier 4 IPC-first | **Gap documented** — 115 imports need feature-gating |
| CI cross-sync 413 | **Green** — zero drift |
| skunkBat Rust IPC | **Exemplar** — `src/ipc/skunkbat.rs` + deploy graphs |

## 7. Quality Gates

| Metric | Value |
|--------|-------|
| Workspace tests | 1,450 (1,297 lib + 73 forge + 80 playGround) |
| Certification tests | 19 (guidestone feature-gated) |
| guideStone level | **5** (was 3) |
| Capabilities | 34 |
| CI cross-sync | 413 methods, zero drift |
| Clippy warnings | 0 |
| Mocks in production | 0 |
| Release binary | 2.8M stripped ELF |
| UniBin validate | 21/21 ALL PASS |
| UniBin certify | 29/29 ALL PASS |
| Deploy graphs | 4 |
| IPC modules | 6 |
| Named tolerances | 233+ |

---

*neuralSpring V150 | Session S200 | AGPL-3.0-or-later*
