# neuralSpring v6 — Evolution Handoff

**Date:** February 22, 2026
**From:** neuralSpring (ML validation & evolutionary computation biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Supersedes:** `archive/NEURALSPRING_V5_EVOLUTION_HANDOFF_FEB22_2026.md`

---

## Executive Summary

neuralSpring completed Phase 5a: GPU `Tensor` validation across 7 scientific
domains (spectral, ecology, HMM, evolution, neural networks, pairwise distance,
Anderson localization). This expanded GPU coverage from 2 domains / 16 checks
to 7 domains / 43 checks. Three new BarraCUDA bugs discovered (S-14, S-15, S-16),
blocking 10 of 43 checks.

Root docs, whitePaper, experiments, and wateringHole handoffs synchronized.
experiments/ directory created following hotSpring journal pattern.

---

## 1. What Changed Since V5

| Category | Action |
|----------|--------|
| Phase 5a GPU Tensor | 7 validators created: spectral, eco, hmm, fitness, nn, pairwise, anderson |
| Bug discovery | S-15 (matmul hang, negative/sparse data), S-16 (transpose dispatch divisor) |
| Root docs sync | README, EVOLUTION_READINESS, DEPRECATION_MIGRATION, CONTROL_EXPERIMENT_STATUS updated |
| whitePaper | README, BARRACUDA_EVOLUTION updated with Phase 5a findings |
| experiments/ | Created with experiment journal index (hotSpring pattern) |
| specs/TOADSTOOL_HANDOFF | Added S-14/S-15/S-16, updated canonical handoff reference |
| wateringHole | V5 archived, V6 GPU handoff + evolution handoff created |
| DEPRECATION_MIGRATION | Updated ToadStool HEAD from `dc540afd` to `77f70b2e`, added S-15/S-16 |

---

## 2. Current Metrics

| Metric | Value |
|--------|-------|
| Python baselines | 206/206 PASS (25 experiments) |
| Rust lib tests | 255 unit + 9 doc-tests |
| Line coverage | 94.9% |
| Validation binaries | 93 |
| Bench binaries | 5 |
| Modules | 31 + 3 evolved |
| WGSL shaders | 16 (8 upstream, 8 local) |
| GPU Tensor validators | 7 (33/43 PASS) |
| Clippy | 0 warnings (pedantic + nursery) |
| ToadStool shortcomings absorbed | 12/12 (S-01..S-12) |
| New shortcomings reported | 3 (S-14, S-15, S-16) |
| Grand total checks | 1442+ (206 Python + 1236+ Rust+GPU) |

---

## 3. Absorption State

### Fully absorbed (delegated to upstream)

| Item | Upstream API |
|------|-------------|
| S-01..S-11 | Various (see `specs/TOADSTOOL_HANDOFF.md`) |
| S-12 eigensolver | `barracuda::ops::linalg::eigh_householder_qr` |
| 8 WGSL shaders | `forge` re-exports from `barracuda::ops::bio::*`, `spectral::*`, `ops::rk_stage` |

### Active local workarounds

| Item | Issue | Status |
|------|-------|--------|
| S-03b MHA projection | GPU dispatch hangs | Local `head_split`/`head_concat` WGSL |
| S-13 PooledBuffer race | Drop before completion | Local `evolved::tensor_sync` |
| S-14 Naive matmul | Small square matrices hang | Non-square shapes |
| S-15 Negative data | Matmul hang with negative/sparse f32 | Positive-only data |
| S-16 Transpose dispatch | Dimensions > 16 produce partial output | Avoid transpose for Gram (use direct `from_data`) |
| 8 WGSL shaders | Pending absorption | Various (78 checks) |

### Pending upstream (new in V6)

| # | Shortcoming | Severity | Fix Effort |
|---|-------------|----------|------------|
| S-15 | Matmul hang (negative/sparse data) | Critical | Investigation needed |
| S-16 | Transpose dispatch divisor | High | One line |
| S-14 | Naive matmul hang | Medium | Retire Naive tier or investigate driver |

---

## 4. Document Alignment Audit

| Document | Status | Updates in V6 |
|----------|--------|---------------|
| `README.md` | Current | Phase 5a table, new shortcomings, updated counts |
| `CONTROL_EXPERIMENT_STATUS.md` | Current | Phase 5a section, 7-domain table, S-14/S-15/S-16 |
| `EVOLUTION_READINESS.md` | Current | Phase 5a GPU Tensor section, new shortcomings table |
| `DEPRECATION_MIGRATION.md` | Current | ToadStool HEAD `77f70b2e`, S-12, S-15/S-16 |
| `specs/TOADSTOOL_HANDOFF.md` | Current | Canonical handoff → V6, S-14/S-15/S-16 section |
| `whitePaper/README.md` | Current | Phase 5a section, updated counts |
| `whitePaper/BARRACUDA_EVOLUTION.md` | Current | S-15/S-16 sections |
| `experiments/README.md` | **NEW** | Experiment journal index (hotSpring pattern) |

---

## 5. Companion Handoffs

| Document | Content |
|----------|---------|
| `NEURALSPRING_V6_BARRACUDA_GPU_HANDOFF_FEB22_2026.md` | S-14/S-15/S-16 diagnosis, reproduction, fixes, audit |
| `archive/NEURALSPRING_TOADSTOOL_ABSORPTION_V5_FEB22_2026.md` | 8-shader absorption request (still current) |

---

*neuralSpring v6 evolution handoff — Phase 5a complete, 3 new bugs reported, all docs aligned.*
