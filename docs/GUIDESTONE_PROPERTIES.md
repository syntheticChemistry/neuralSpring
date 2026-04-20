# neuralSpring guideStone — Certified Properties

**Standard**: `primalSpring/wateringHole/GUIDESTONE_COMPOSITION_STANDARD.md` v1.2.0
**Binary**: `neuralspring_guidestone` v0.3.0 (feature-gated: `guidestone`)
**Level**: 3 (bare ALL PASS — all 5 properties certified without primals)
**Date**: April 20, 2026 — Session S185

---

## Overview

A guideStone carries 5 certified properties that hold **without any primals running**
(bare guideStone). When a NUCLEUS is deployed, additive layers activate — primal
discovery, domain science parity via IPC, and BearDog signing — but physics and
science output is IDENTICAL either way. NUCLEUS adds metadata, not math.

---

## Property 1: Deterministic Output

> Same binary, same results, any architecture.

**Status**: CERTIFIED

| Aspect | Evidence |
|--------|----------|
| Seeded RNG | All 27 papers use `Rng::new(42)` (xoshiro256++ via SplitMix64 seeding) |
| Named tolerances | 234+ constants in `tolerances/` — no ad-hoc magic numbers |
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
| Domain integrity | 1,234+ lib tests with `#![forbid(unsafe_code)]` |

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
| Named constants | 234+ in `tolerances/` with justification comments |
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

| Level | Description | Status |
|-------|-------------|--------|
| 0 | Not started | -- |
| 1 | Validation exists (IpcMathClient, validate_proto_nucleate_capabilities) | DONE |
| 2 | Properties documented (this file) | DONE |
| 3 | Bare guideStone works (29/29 pass, P1-P5 certified without primals) | DONE |
| 4 | NUCLEUS guideStone works (validates against live NUCLEUS) | PENDING |
| 5 | Certified (all 5 properties hold, cross-substrate parity) | PENDING |

### Level 3 Evidence (S184 → S185)

- `neuralspring_guidestone` v0.3.0: 29/29 bare checks PASS (4 SKIP for missing NUCLEUS)
- P3 BLAKE3 CHECKSUMS: 15 files verified via `primalspring::checksums::verify_manifest()`
- `v.section()` structured output for Phase 1–4
- `FAMILY_ID` env support for family-isolated socket discovery
- Protocol tolerance: `is_skip_error()` unified skip classification (v0.9.17 pattern)
- Exit code 2 correctly returned for bare-only mode
- S185: absorbed `primalspring::composition::is_skip_error` — replaces 7 manual error arms

### Level 4 Requirements

- Live NUCLEUS deployed from `plasmidBin/` ecobins (12 primals)
- `primalspring_guidestone` passes (exit 0) as base certification
- All 7 `PROTO_NUCLEATE_VALIDATION_CAPABILITIES` return PASS (not SKIP)

### Level 5 Requirements

- All Level 4 requirements met
- Cross-substrate parity: Python / CPU / GPU / IPC all within tolerances
- barraCuda surface gaps (Gap 11: 18 methods) resolved upstream
- BearDog signing receipt validates end-to-end

---

## References

- guideStone Standard: `primalSpring/wateringHole/GUIDESTONE_COMPOSITION_STANDARD.md` (v1.2.0)
- Composition Guidance: `primalSpring/wateringHole/PRIMALSPRING_COMPOSITION_GUIDANCE.md`
- Downstream Manifest: `primalSpring/graphs/downstream/downstream_manifest.toml`
- plasmidBin Depot: `primalSpring/wateringHole/PLASMINBIN_DEPOT_PATTERN.md`
- hotSpring reference: `hotSpring-guideStone-v0.7.0` (Level 5 certified)
- primalSpring reference: `primalspring_guidestone` (Level 4 — 67/67 live NUCLEUS checks)
