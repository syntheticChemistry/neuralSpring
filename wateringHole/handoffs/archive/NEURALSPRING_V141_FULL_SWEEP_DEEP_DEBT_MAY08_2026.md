# neuralSpring V141 — Full Sweep: Deep Debt, Coverage, and Downstream Review

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Date:** May 8, 2026
**Session:** S191
**From:** neuralSpring
**To:** primalSpring, all primal teams, all spring teams, sporeGarden teams
**Prior:** V140 (cross-spring parity response, same day)

---

## Summary

Full sweep across all remaining evolution axes following the V140 parity
response. Deep debt audit confirms codebase is remarkably clean (zero
unsafe, zero TODOs, zero mocks in prod, zero unwraps in lib, zero files
>800L). Work focused on coverage expansion, downstream alignment, and
continued notebook pipeline.

| Work Item | Status |
|-----------|--------|
| projectNUCLEUS + foundation review | **DONE** |
| Inline test coverage expansion | **DONE** — 45 new tests across 5 modules |
| Liu faculty paper notebooks | **DONE** — 3 notebooks, 26/26 checks |
| Benchmark gap roadmap | **DONE** — gaps documented and prioritized |
| Tier 4 IPC validator audit | **DONE** — 160 checks documented |
| Documentation + handoff | **DONE** |

## Phase 1: Downstream Project Review

### projectNUCLEUS (sporeGarden)

Reviewed `gardens/projectNUCLEUS` — the deployable NUCLEUS infrastructure:
- 13/13 primals deployed and healthy on ironGate
- BTSP Phase 3 AEAD, 5-tier discovery hierarchy, full provenance chain
- JupyterHub at `lab.primals.eco` with ABG tiered access
- MethodGate (JH-0) adopted by all 13/13 primals
- neuralSpring's deploy graphs feed directly into projectNUCLEUS

### foundation (sporeGarden)

Reviewed `gardens/foundation` — the scientific knowledge layer:
- 10 domain threads mapping 70+ reproduced papers
- neuralSpring appears in Thread 5 (Evolutionary Biology, Dolson/Waters)
  and Thread 7 (Anderson Mathematics, Kachkovskiy)
- 100 data sources across 5 threads, 36 validation targets
- foundation defines WHAT to validate; projectNUCLEUS defines HOW to deploy

## Phase 2: Test Coverage Expansion

45 new inline unit tests across 5 library modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| `src/error.rs` | 14 | Error construction, Display, From conversions |
| `src/streaming/mod.rs` | 8 | Newline trimming, buffer capacity invariants |
| `src/search/mod.rs` | 5 | K-mer index build, lookup, N-base skipping |
| `src/provenance/references.rs` | 8 | Softmax sum-to-one, GELU signs, benchmark globals |
| `src/visualization/types.rs` | 10 | DataChannel serialization, ScenarioNode skip-empty |

Lib tests: 1,279 (up from 1,234 pre-S191)

## Phase 3: Liu Faculty Paper Notebooks

3 new publishable notebooks (8 total, 72/72 checks):

| Notebook | Paper | Checks |
|----------|-------|--------|
| `paper-016-hmm-phylo.ipynb` | Liu et al. (2014) PLoS Comp Bio — PhyloNet-HMM | 10/10 |
| `paper-017-sate-alignment.ipynb` | Liu et al. (2009) Science — SATe | 8/8 |
| `paper-018-introgression.ipynb` | Liu et al. (2015) PNAS — Introgression | 8/8 |

All notebooks execute cleanly, have full inline implementations, matplotlib
visualizations, and provenance links. Updated `paper-baselines.json`.

## Phase 4: Benchmark Gap Audit

Updated `specs/BENCHMARK_ANALYSIS.md` with roadmap:

| Category | Status |
|----------|--------|
| Python CPU vs Rust CPU | Complete (83.6x geomean) |
| GPU vs CPU | Complete (5-scale benchmark) |
| cuBLAS/cuDNN/cuFFT/FlashAttn | Complete |
| Kokkos | Partial (estimated, not matched-hardware) |
| Polybench/GPU | Not started (linear algebra subset relevant) |
| SPEC/Rodinia/Parboil | Out of scope (application-level) |

## Phase 5: Tier 4 IPC Validator Audit

8 composition validators with 160 total checks (11 skip-when-offline):

| Validator | Checks | Skips |
|-----------|--------|-------|
| `validate_nucleus_composition` | 22 | 2 |
| `validate_inference_composition` | 16 | 3 |
| `validate_science_composition` | 9 | 2 |
| `validate_composition_evolution` | 30 | 4 |
| `validate_nucleus_tower` | 47 | 0 |
| `validate_nucleus_compute_dispatch` | 36 | 0 |
| `validate_mixed_composition_pipeline` | — | — |
| `validate_nucleus_pcie_mixed_pipeline` | — | — |

Updated `experiment-catalog.json` with detailed Tier 4/5 status.

## Deep Debt Audit Results

| Area | Status |
|------|--------|
| Files >800 lines | **None** (largest: tolerances/mod.rs 776L) |
| `unsafe` code | **Zero** (#![forbid(unsafe_code)] workspace-wide) |
| `#[allow(`] | **Zero** (all migrated to #[expect(]) |
| TODO/FIXME/HACK | **Zero** |
| Mocks in production | **Zero** (inference stubs are intentional degradation) |
| `.unwrap()` in lib | **Zero** (all in test modules) |
| `extern crate` | **Zero** |
| External non-Rust deps | **None** (wgpu GPU drivers are inherent) |

## Codebase Stats

- **Tests:** 1,279 lib + 73 forge + 80 playGround = 1,432
- **Experiments:** 134
- **Binaries:** 269
- **Deploy graphs:** 4
- **Experiment crates:** 1
- **Paper notebooks:** 8 (72/72 checks, 2/8 faculties)
- **Named tolerances:** 233
- **GuideStone:** L3 (29/29 bare checks)
- **barracuda:** optional (S190)
- **Deny clean:** yes

## Remaining Paper Notebook Batches

| Batch | Faculty | Papers | Status |
|-------|---------|--------|--------|
| 1 | Dolson | 011-015 | **Done** (S189) |
| 2 | Liu | 016-018 | **Done** (S191) |
| 3 | Waters | 019-021 | Pending |
| 4 | Kachkovskiy | 022-023 | Pending |
| 5 | Anderson/Campbell | 024-025 | Pending |
| 6 | Liao/Wang | 026-027 | Pending |

---

*License: AGPL-3.0-or-later*
