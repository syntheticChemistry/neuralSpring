# neuralSpring v5 — Evolution Handoff

**Date:** February 22, 2026
**From:** neuralSpring (ML validation & evolutionary computation biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Supersedes:** `archive/NEURALSPRING_V4_ABSORPTION_HANDOFF_FEB22_2026.md`

---

## Executive Summary

neuralSpring has completed a full documentation sync, aligning all root docs,
whitePaper, specs, and metalForge manifests to ToadStool `77f70b2e`. All 12
shortcomings (S-01..S-12) are confirmed absorbed. 8 of 16 WGSL shaders now
source from upstream barracuda. The remaining 8 local shaders are documented
with specific absorption recommendations in the companion handoff.

---

## 1. What Changed Since V4

| Category | Action |
|----------|--------|
| Root docs sync | README, CONTROL_EXPERIMENT_STATUS aligned to `77f70b2e` / 12 shortcomings |
| EVOLUTION_READINESS | Shader check counts corrected (modes 15, directed 6, swarm 9, signal 9) |
| ABSORPTION_MANIFEST | S-12 added, ToadStool HEAD updated, eigh.rs delegation noted |
| whitePaper | README, STUDY, BARRACUDA_EVOLUTION updated for S-12 absorption |
| BARRACUDA_EVOLUTION | S-12 section updated to ABSORBED status with NAK eigensolve note |
| V4 handoff | Archived to `archive/` |
| ToadStool absorption request | New companion handoff with 8-shader absorption sequence |

---

## 2. Current Metrics

| Metric | Value |
|--------|-------|
| Python baselines | 206/206 PASS (25 experiments) |
| Rust lib tests | 237 unit + 9 doc-tests |
| Line coverage | 94.9% |
| Validation binaries | 81 |
| Bench binaries | 5 |
| Modules | 29 + 3 evolved |
| WGSL shaders | 16 (8 upstream, 8 local) |
| Clippy | 0 warnings (pedantic + nursery) |
| ToadStool shortcomings | 12/12 absorbed |
| Grand total checks | 1300+ (206 Python + 1100+ Rust+GPU) |

---

## 3. Absorption State

### Fully absorbed (delegated to upstream)

| Item | Upstream API |
|------|-------------|
| S-01..S-11 | Various (see `specs/TOADSTOOL_HANDOFF.md`) |
| S-12 eigensolver | `barracuda::ops::linalg::eigh_householder_qr` |
| 8 WGSL shaders | `forge` re-exports from `barracuda::ops::bio::*`, `spectral::*`, `ops::rk_stage` |

### Active local workarounds

| Item | Issue | Binary | Checks |
|------|-------|--------|--------|
| S-03b MHA projection | GPU dispatch hangs | `validate_mha_gpu` | 10/10 |
| S-13 PooledBuffer race | Drop before completion | `evolved::tensor_sync` | — |
| 8 WGSL shaders | Pending absorption | Various | 78 total |

---

## 4. Document Alignment Audit

All documents now reference ToadStool `77f70b2e` and 12 absorbed shortcomings:

| Document | SHA | Shortcomings | Status |
|----------|-----|-------------|--------|
| README.md | `77f70b2e` | 12 absorbed | Current |
| CONTROL_EXPERIMENT_STATUS.md | `77f70b2e` | 12 absorbed | Current |
| EVOLUTION_READINESS.md | `77f70b2e` | 12 absorbed | Current |
| metalForge/ABSORPTION_MANIFEST.md | `77f70b2e` | 12 absorbed (S-12 added) | Current |
| specs/TOADSTOOL_HANDOFF.md | `77f70b2e` | 12 absorbed | Current |
| specs/CROSS_SPRING_EVOLUTION.md | `77f70b2e` | — | Current |
| whitePaper/README.md | `77f70b2e` | 12 absorbed | Current |
| whitePaper/STUDY.md | `77f70b2e` | 12 absorbed | Current |
| whitePaper/BARRACUDA_EVOLUTION.md | `77f70b2e` | 12 absorbed | Current |

---

## 5. Companion Handoff

See `NEURALSPRING_TOADSTOOL_ABSORPTION_V5_FEB22_2026.md` for the
ToadStool-directed absorption request with:
- 8 WGSL shaders ready for absorption (with priority order)
- 2 bug fixes (S-03b, S-13)
- Cross-spring evolution learnings
- Benchmark data for absorption prioritization

---

*neuralSpring v5 evolution handoff — all docs aligned, all shortcomings absorbed.*
