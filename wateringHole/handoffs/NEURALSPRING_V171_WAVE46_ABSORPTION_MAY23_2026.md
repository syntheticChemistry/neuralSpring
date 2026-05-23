# neuralSpring V171 Handoff — Wave 46 Absorption Sprint

**Date:** May 23, 2026
**Session:** S215
**From:** neuralSpring
**To:** primalSpring, downstream consumers
**License:** AGPL-3.0-or-later

---

## Summary

Absorption of primalSpring Wave 46 (v0.9.27, 458 methods) upstream audit. Registry sync, BLAKE3 graph backfill, sporePrint content refresh, and guideStone version reconciliation.

**45 capabilities** registered. **754 workspace tests** (all pass). Deploy graph `graphs/neuralspring_deploy.toml` aligned to **V171/S215**.

---

## Changes

### Registry Sync (445 -> 458, Wave 38 -> 46)

primalSpring registry count updated from 445 (Wave 38) to 458 (Wave 46, v0.9.27) across all living documentation and the `src/config.rs` cross-sync test doc comment. The test itself is substring-based and count-agnostic — no code changes needed.

### sporePrint Content Refresh

`sporeprint/validation-summary.md` refreshed from S209 to S215:
- 910 tests -> 754 (IPC-first count post-S208 restructure)
- 37 capabilities -> 45 (S214 registry alignment)
- Performance geomean: 83.6x -> 38.6x (honest 15-domain measure)
- 14 gaps (2 resolved) -> 28 gaps (28 resolved)
- primalSpring v0.9.25+ -> v0.9.27+
- B3/B4 validation binaries added to key binary list

### BLAKE3 Graph Backfill (FN-1)

All 4 deploy/graph TOMLs now carry `blake3_hash` content hashes in `[graph.metadata]`:
- `graphs/neuralspring_deploy.toml`
- `graphs/neuralspring_spectral_analysis.toml`
- `graphs/neuralspring_inference_pipeline.toml`
- `graphs/composition/neuralspring_math_pipeline.toml`

Aligns with upstream FN-1 (BLAKE3 backfill, 10/25 sources hashed). All graph versions updated to S215.

### guideStone Version Reconciliation

Binary `neuralspring_guidestone.rs` version bumped from 0.3.0 to 0.4.0, matching the certification organelle SSoT (`src/certification/mod.rs`).

---

## Wave 46 Assessment — Items Not Absorbed (By Design)

| Item | Reason |
|------|--------|
| NeuralBridge observatory | primalSpring-internal abstraction; springs consume via CompositionContext |
| Ionic contract wiring | healthSpring Track 4 scope; neuralSpring is Metallic/InternalNucleus |
| `guidestone_readiness = 2` in manifest | Upstream responsibility to update to match our L5 claim |
| 6 new `neural_api.*` methods | biomeOS observatory domain; not in neuralSpring's niche |

---

## Test Verification

```
cargo test --lib -> 754 passed, 0 failed
cargo build -> OK
```

---

## Upstream Impact

- primalSpring cross-sync: neuralSpring references updated to 458 methods / Wave 46
- BLAKE3 hashes on all 4 graph TOMLs (FN-1 contribution)
- sporePrint content current for primals.eco publishing pipeline
- guideStone binary/organelle version parity restored
