# neuralSpring guideStone — Certified Properties

**Standard**: `primalSpring/wateringHole/GUIDESTONE_COMPOSITION_STANDARD.md` v1.2.0
**Binary**: `neuralspring_guidestone` v0.4.0 (feature-gated: `guidestone`)
**Level**: 5 (6-layer certification: bare + discovery + parity + nucleus + composition + cross-spring)
**Date**: May 25, 2026 — Session S218 (live southGate deployment, 30/37 PASS, 45 capabilities, 754 tests, V174 handoff)

---

## Overview

A guideStone carries 5 certified properties that hold **without any primals running**
(bare guideStone, L0). Six additive layers validate increasingly complex composition:

| Layer | Module           | Requires primals? | Description |
|-------|------------------|-------------------|-------------|
| L0    | `bare`           | No                | 5 certified properties (P1–P5) |
| L1    | `discovery`      | Yes               | `CompositionContext` liveness probes |
| L2    | `parity`         | Yes               | Domain science parity (7 capabilities via IPC) |
| L3    | `nucleus`        | Yes               | Additive NUCLEUS (BearDog signing, Songbird discovery) |
| L4    | `composition`    | Yes               | NUCLEUS composition (deploy graphs, registry, families) |
| L5    | `cross_spring`   | Yes               | Cross-spring validation (frozen artifacts, protocol liveness, hash determinism) |

Physics and science output is IDENTICAL either way. NUCLEUS adds metadata, not math.

---

## Property 1: Deterministic Output

> Same binary, same results, any architecture.

**Status**: CERTIFIED

| Aspect | Evidence |
|--------|----------|
| Seeded RNG | All 27 papers use `Rng::new(42)` (xoshiro256++ via SplitMix64 seeding) |
| Named tolerances | 228+ constants in `tolerances/` — no ad-hoc magic numbers |
| CPU-only path | Full validation coverage without GPU — CPU path is the reference |
| Cross-substrate | Python/CPU/GPU parity documented for all active experiments |

**guideStone check**: `P1:deterministic_rng` — seeds xoshiro, runs twice, exact bitwise match.

---

## Property 2: Reference-Traceable

> Every number traces to a paper or proof.

**Status**: PARTIAL (documented, not yet machine-readable JSON output)

| Aspect | Evidence |
|--------|----------|
| Provenance records | 49 entries in `PROVENANCE_REGISTRY` (provenance/mod.rs) |
| Structure | Each record: label, script, commit, date, command, environment, value, unit |
| Script existence | Test suite validates every `script` path exists on disk |
| SPDX headers | Test suite validates every script has `SPDX-License-Identifier:` |
| Expected sources | `expected_source()` method maps records to reference constants |

**guideStone checks**: `P2:provenance_registry_populated`, `P2:provenance_all_labelled`,
`P2:provenance_all_scripted`, `P2:provenance_all_committed`.

**Gap**: Not yet machine-readable JSON output (Level 3 requirement). The data exists
in Rust constants but the guideStone does not emit a structured provenance manifest.

---

## Property 3: Self-Verifying

> Tampered inputs detected, non-zero exit.

**Status**: CERTIFIED (BLAKE3 CHECKSUMS via `primalspring::checksums`)

| Aspect | Evidence |
|--------|----------|
| Exit codes | 0 = all pass, 1 = regression, 2 = bare only (no NUCLEUS) |
| BLAKE3 CHECKSUMS | `validation/CHECKSUMS` — 15 validation-critical files hashed |
| Checksum verification | `primalspring::checksums::verify_manifest()` in Phase 1 bare checks |
| Manifest generation | `examples/gen_checksums.rs` generates manifest (feature-gated) |
| ValidationHarness | Every validation binary uses `check_abs` / `check_rel` / `check_bool` |
| Parity checks | 7 `PROTO_NUCLEATE_VALIDATION_CAPABILITIES` validated against baselines |
| Domain integrity | 1,300 lib tests with `#![forbid(unsafe_code)]` |

**Checksummed files**: guideStone binary, tolerances (5 modules), provenance (3 modules),
validation (2 modules), RNG, capability registry, Python baseline tolerances, Cargo.toml.

---

## Property 4: Environment-Agnostic

> Pure Rust, ecoBin, no network, no sudo.

**Status**: CERTIFIED

| Aspect | Evidence |
|--------|----------|
| Pure Rust | `#![forbid(unsafe_code)]` at crate and binary level |
| Zero C deps | `deny.toml` bans `ring`, `openssl-sys`, `cc` (except blake3 build) |
| No network | No runtime downloads, no HTTP calls, offline-capable |
| No root | Runs as unprivileged user, no `sudo` or capability requirements |
| Cross-compile | Static musl targets supported via `rust-toolchain.toml` (stable) |
| ecoBin compliant | Binary deployable from `plasmidBin/` without source tree |

**guideStone checks**: `P4:ecobin_compliant`, `P4:pure_rust_forbid_unsafe`,
`P4:no_network_required`.

---

## Property 5: Tolerance-Documented

> Every tolerance has a derivation.

**Status**: CERTIFIED

| Aspect | Evidence |
|--------|----------|
| Named constants | 228+ in `tolerances/` with justification comments |
| Categories | machine, cross-language, spectral, training, literature, gpu, evolutionary |
| Registry introspection | `all_tolerances()` returns `NamedTolerance` with name/value/category |
| Finite values | Test suite validates no NaN/Inf in any tolerance |
| Ordered | Composition parity tolerances form a strict ordering chain |

**guideStone checks**: `P5:tolerance_count`, `P5:tolerances_all_finite`,
`P5:tolerances_all_named`, `P5:tolerances_all_categorized`.

**Gap**: Tolerance derivation metadata is in source comments, not yet machine-readable.
A structured `ToleranceDerivation` type with paper DOIs would strengthen this for Level 3+.

---

## Readiness Matrix

| Level | Module         | Description | Status |
|-------|----------------|-------------|--------|
| L0    | `bare`         | 5 properties certified without primals (29/29 checks) | DONE |
| L1    | `discovery`    | CompositionContext liveness probes | DONE |
| L2    | `parity`       | Domain science parity via IPC (7 capabilities) | DONE |
| L3    | `nucleus`      | Additive NUCLEUS (BearDog signing, Songbird discovery) | DONE |
| L4    | `composition`  | NUCLEUS composition (deploy graphs, capability registry, family calls) | DONE |
| L5    | `cross_spring` | Cross-spring validation (frozen artifacts, protocol liveness, hash determinism) | DONE |

**19 certification tests** across 6 layers. Run via:
```
cargo test --features barracuda,guidestone -p neural-spring --lib certification
```

### L0 Evidence (S184 → S185)

- `neuralspring_guidestone` v0.3.0: 29/29 bare checks PASS (4 SKIP for missing NUCLEUS)
- P3 BLAKE3 CHECKSUMS: 15 files verified via `primalspring::checksums::verify_manifest()`
- `v.section()` structured output for Phase 1–4
- `FAMILY_ID` env support for family-isolated socket discovery
- Protocol tolerance: `is_skip_error()` unified skip classification (v0.9.17 pattern)
- Exit code 2 correctly returned for bare-only mode
- S185: absorbed `primalspring::composition::is_skip_error` — replaces 7 manual error arms

### L1–L3 Evidence (S193 → S197)

- Discovery: `CompositionContext` liveness probes for all 12 primals
- Parity: 7 `PROTO_NUCLEATE_VALIDATION_CAPABILITIES` validated via IPC against baselines
- NUCLEUS: BearDog signing + Songbird discovery integrated
- 13 certification tests across 4 layers (expanded to 19 across 6 in S200+)

### L4–L5 Evidence (S200 → S201b)

- Composition: deploy graphs validated, capability registry cross-sync (35 capabilities)
- Cross-spring: frozen artifact hashes, protocol liveness, deterministic hash comparison
- 19 certification tests across 6 layers
- IPC-first defaults (`default = []`) — all validation runs without GPU by default
- `IpcError` typed hierarchy replaces stringly-typed errors
- 241 `required-features` bins ensure GPU-dependent binaries only build with `barracuda`

---

## References

- guideStone Standard: `primalSpring/wateringHole/GUIDESTONE_COMPOSITION_STANDARD.md` (v1.2.0)
- Composition Guidance: `primalSpring/wateringHole/PRIMALSPRING_COMPOSITION_GUIDANCE.md`
- Downstream Manifest: `primalSpring/graphs/downstream/downstream_manifest.toml`
- plasmidBin Depot: `primalSpring/wateringHole/PLASMINBIN_DEPOT_PATTERN.md`
- hotSpring reference: `hotSpring-guideStone-v0.7.0` (Level 5 certified)
- primalSpring reference: `primalspring certify` (Level 8 — live NUCLEUS certification, absorbed as UniBin organelle)
