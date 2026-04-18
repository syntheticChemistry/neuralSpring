# neuralSpring guideStone — Certified Properties

**Standard**: `primalSpring/wateringHole/GUIDESTONE_COMPOSITION_STANDARD.md`
**Binary**: `neuralspring_guidestone` (feature-gated: `guidestone`)
**Level**: 2 (properties documented, partial Level 3: bare guideStone compiles and validates)
**Date**: April 18, 2026 — Session S183

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

**Status**: PARTIAL (Level 2 — documented, not yet machine-readable JSON)

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

**Status**: PARTIAL (Level 2 — exit code semantics, no CHECKSUMS file)

| Aspect | Evidence |
|--------|----------|
| Exit codes | 0 = all pass, 1 = regression, 2 = bare only (no NUCLEUS) |
| ValidationHarness | Every validation binary uses `check_abs` / `check_rel` / `check_bool` |
| Parity checks | 7 `PROTO_NUCLEATE_VALIDATION_CAPABILITIES` validated against baselines |
| Domain integrity | 1,234+ lib tests with `#![forbid(unsafe_code)]` |

**Gap**: No `CHECKSUMS` file for binary integrity verification. This is a Level 3
requirement — the guideStone binary should carry a manifest of known-good hashes
for its inputs and report tampering as a non-zero exit.

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
| 3 | Bare guideStone works (compiles, validates P1-P5 without primals) | PARTIAL |
| 4 | NUCLEUS guideStone works (validates against live NUCLEUS) | PENDING |
| 5 | Certified (all 5 properties hold, cross-substrate parity) | PENDING |

### Level 3 Blockers

- **CHECKSUMS file** for Property 3 (Self-Verifying)
- **Machine-readable provenance** for Property 2 (JSON output)
- **Machine-readable tolerance derivations** for Property 5

### Level 4 Requirements

- Live NUCLEUS deployed from `plasmidBin/` ecobins
- `primalspring_guidestone` passes (exit 0) as base certification
- All 7 `PROTO_NUCLEATE_VALIDATION_CAPABILITIES` return PASS (not SKIP)

### Level 5 Requirements

- All Level 3 + Level 4 blockers resolved
- Cross-substrate parity: Python / CPU / GPU / IPC all within tolerances
- barraCuda surface gaps (Gap 11: 18 methods) resolved upstream
- BearDog signing receipt validates end-to-end

---

## References

- guideStone Standard: `primalSpring/wateringHole/GUIDESTONE_COMPOSITION_STANDARD.md`
- Composition Guidance: `primalSpring/wateringHole/PRIMALSPRING_COMPOSITION_GUIDANCE.md`
- Downstream Manifest: `primalSpring/graphs/downstream/downstream_manifest.toml`
- hotSpring reference: `hotSpring-guideStone-v0.7.0` (Level 5 certified)
