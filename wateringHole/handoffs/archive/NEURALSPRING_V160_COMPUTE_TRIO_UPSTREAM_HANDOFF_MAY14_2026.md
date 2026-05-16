# neuralSpring V160 — Compute Trio Wave Absorption Handoff

**From:** neuralSpring (S206)
**To:** primalSpring, barraCuda, coralReef, toadStool, plasmidBin, all spring teams
**Date:** 2026-05-14
**Session:** S206 — Upstream audit absorption (May 14 plasmidBin + compute trio wave)

---

## Summary

neuralSpring absorbed the May 14 ecosystem status update and compute trio wave
(barraCuda v0.4.0, coralReef v0.1.0). All deploy graphs now honor triple-first
Tower (bearDog + songBird + skunkBat). NestGate aligned to Wave 7. Gap
reconciliation complete. Clippy cast safety fixes applied for Rust 1.94.

---

## Changes Made

### 1. Deploy Graphs: skunkBat Triple-First Tower

All domain deploy graphs now include skunkBat in the Tower phase:

| Graph | Change |
|-------|--------|
| `neuralspring_deploy.toml` | Already compliant — confirmed |
| `neuralspring_inference_pipeline.toml` | Added `tower_defense` (skunkBat) node after `tower_discovery` |
| `neuralspring_spectral_analysis.toml` | Added `tower_discovery` (songBird) + `tower_defense` (skunkBat); node deps updated to chain through discovery |
| `composition/neuralspring_math_pipeline.toml` | Version stamp only — minimal graph, no Tower |

All graph version stamps updated from S200b to S206.

### 2. NestGate Wave 7 Alignment

Deploy graph NestGate `by_capability` changed from `"storage.retrieve"` to
`"content.get"` per Wave 7 guidance that springs adopt content-addressed
routing alongside storage.

### 3. PRIMAL_GAPS.md Reconciliation

| Gap | Change |
|-----|--------|
| **Gap 5** (NestGate) | Marked **RESOLVED** — was stale "wip" despite S205 completion |
| **Gap 7** (Proto-nucleate vs spring-deploy) | Explicitly documented as **design decision** per CROSS_SPRING_PARITY_SCORECARD |
| **Gap 3** (coralReef) | Noted coralReef v0.1.0 unblocks `compile_shader_universal` routing |
| **Gap 9** (barraCuda plasma_dispersion) | Re-verified still present in v0.4.0 — flagged for compute trio wave |

### 4. Clippy Cast Safety (Rust 1.94)

5 cast errors in `glucose_prediction/experiment.rs` fixed with proper `#[expect]`
annotations: `cast_possible_truncation`, `cast_sign_loss` for domain-specific
numeric patterns. Zero clippy errors on both default and `--all-features` builds.

---

## Hand-Backs to Upstream Teams

### → barraCuda

- **Gap 9**: `special/plasma_dispersion.rs` line 23 imports
  `crate::ops::lattice::cpu_complex::Complex64` without `#[cfg(feature = "domain-lattice")]`.
  neuralSpring works around this by enabling `domain-lattice`. Fix belongs upstream.
  v0.4.0 still has this — compute trio wave is the right time.

### → coralReef

- **Gap 3 evolution**: `compile_shader_universal` routing through coralReef IPC
  is now unblocked by v0.1.0 release (Blackwell, naga::Module, dual-vendor).
  neuralSpring has `shader.compile.wgsl` and `shader.compile.capabilities` IPC
  already wired in `src/ipc/coralreef.rs`. Next step: route the actual
  `gpu/mod.rs::compile_shader_universal` through this IPC path when coralReef is
  available in composition.

### → plasmidBin

- **Cell graph missing skunkBat**: `cells/neuralspring_cell.toml` has beardog +
  songbird but not skunkBat. Per the atomic model (Tower = bearDog + songBird +
  skunkBat), a `skunkBat` node should be added between songbird (order 2) and
  toadstool (order 3) with `by_capability = "defense"` and
  `capabilities = ["security.audit_log", "defense.threat_baseline", "defense.metadata_lineage"]`.

### → primalSpring

- `wateringHole/README.md` still lists neuralSpring at S201b with older gap
  descriptions. Current truth: S206, V160, Gap 11 CLOSED, Gap 5 RESOLVED,
  proto-nucleate vs spring-deploy documented as design decision.
- `downstream_manifest.toml` `guidestone_readiness = 2` for neuralSpring — actual
  status is Level 5 (19 certification tests ALL PASS, L0-L5).

### → Squirrel

- `inference.register_provider` — neuralSpring still cannot self-register as an
  inference backend. Not blocking current evolution but needed for full composition.

---

## Current State

| Metric | Value |
|--------|-------|
| Session | S206 |
| Workspace tests | 910 |
| Clippy errors | 0 (default + `--all-features`) |
| PRIMAL_GAPS open | 3 (Gap 6 BTSP/upstream, Gap 9 barraCuda/upstream, Gap 10 tracking) |
| Deploy graphs | 4, all S206, all triple-first Tower |
| guideStone | Level 5 (19/19 certification tests) |
| Evolution | composing |
| Handoff | V160 |

---

## Hold Items (unchanged)

- Full NUCLEUS compositions — hold until `plasmidBin` deployment tooling live
- Squirrel provider registration — upstream Squirrel dependency
- WGSL tokenization pipeline — coralReef + toadStool + barraCuda chain
- Matched-hardware GPU benchmarks — barraCuda precision/E2E validation
