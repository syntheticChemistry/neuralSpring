# neuralSpring V181 Handoff — Wave 82c Domain Profile

**Date:** 2026-06-06
**Session:** S225
**Gate:** southGate
**Commit:** 155ff38

---

## Summary

Wave 82c compliance: created root `domain_profile.toml` for `litho
emit-pseudospore` ecosystem classification. Cleaned stale session stamps
across 7 living docs. Migrated remaining raw-string primal discovery
hints to constants.

## Changes

### P1: domain_profile.toml (Wave 82c requirement)

- Created `domain_profile.toml` at repo root
- Domain: `computational-science`
- 6 subdomains: ml-surrogates, transfer-learning, scholarly-reproduction,
  biophysical-ai, warm-dense-matter, isomorphic-patterns
- 10 translation entity groups covering full neuralSpring scope
- 4 derivation pipelines (Python baselines, Rust validation, GPU parity,
  cross-gate dispatch)
- 7 audit checks, 6 figure definitions
- Reference format: wetSpring / healthSpring

### Primal name hygiene

- 3 raw-string discovery hints (`"biomeos"`, `"rhizocrypt"`) in
  `handlers.rs` migrated to `primal_names::BIOMEOS` /
  `primal_names::RHIZOCRYPT` constants
- All primal discovery hints now use constants from `primal_names.rs`

### Documentation reconciliation

7 living docs updated from stale session stamps to S225:

| File | From | To |
|------|------|----|
| `CONTEXT.md` | S218 | S225 |
| `specs/NUCLEUS_TOWER_INTEGRATION.md` | S132 | S225 |
| `specs/BARRACUDA_USAGE.md` | S181 | S225 |
| `specs/ECOSYSTEM_LEVERAGE_GUIDE.md` | S175 | S225 |
| `whitePaper/BARRACUDA_EVOLUTION.md` | S213 | S225 |
| `whitePaper/baseCamp/extensions.md` | S215 | S225 |
| `sporeprint/validation-summary.md` | S223 | S225 |

- CONTEXT.md: capabilities 45→47, scenarios 10→11, deployment 9/13→13/13
- NUCLEUS_TOWER_INTEGRATION: 14 methods → 47 capabilities

## Verification

- **932 workspace tests** (756 lib + 11 integration + 73 forge + 80
  playGround + 12 exp094), 0 failures
- **0 clippy warnings** (pedantic + nursery)
- **47 capabilities** in sync across `ALL_CAPABILITIES`, `niche::CAPABILITIES`,
  and `capability_registry.toml`

## Status

neuralSpring is fully compliant with Wave 82c requirements. No
remaining P0/P1 items.
