+++
title = "neuralSpring Validation Summary"
description = "ML primitives and sovereign structure prediction — 4,900+ checks, Isomorphism Theorem, 38.6x faster than Python"
date = 2026-06-03

[taxonomies]
primals = ["barracuda", "toadstool", "biomeos", "squirrel"]
springs = ["neuralspring", "hotspring", "wetspring", "groundspring"]
+++

# neuralSpring — sporePrint Validation Summary

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

> **Session:** S225 | **Date:** Jun 6, 2026 | **Version:** 0.1.0 | **Handoff:** V181
> **Gate:** southGate | **Live validation:** 9/13 primals via UDS
> **Tier:** 2 (sporePrint: frozen data + notebooks + paper baselines)

---

## Headline Numbers

| Metric | Value |
|--------|-------|
| **Workspace tests (IPC-first)** | 754 |
| **Proptest properties** | 24 |
| **Python baselines** | 397/397 PASS |
| **Rust+GPU checks** | 4,500+ |
| **Total validation checks** | 4,900+ |
| **Binaries** | 269 (244 validate, 18 bench, 7 other) |
| **Experiments** | 134 across 11 domains |
| **Papers reproduced** | 27 (6 faculties) |
| **Capabilities** | 45 (12 domains) |
| **Named tolerances** | 233 |
| **guideStone** | 30/37 PASS, 6 SKIP (live southGate deployment) |
| **BTSP** | 13/13 mandatory |
| **PRIMAL_GAPS** | 29 main (29 resolved) |

---

## Code Quality

| Check | Status |
|-------|--------|
| `cargo clippy --workspace` (pedantic+nursery) | **0 warnings** |
| `cargo fmt --check` | **0 diffs** |
| `cargo doc --workspace --no-deps` | **0 warnings** |
| `cargo deny check` | **clean** |
| `#![forbid(unsafe_code)]` | **workspace-wide** |
| `#[allow()]` attributes | **0** |
| TODO/FIXME/HACK | **0** |
| Mocks in production | **0** |

---

## Performance

| Metric | Value |
|--------|-------|
| Rust vs Python geomean | **38.6x** (15 domains) |
| Fastest speedup | **1,104x** (multi-objective) |
| CPU-Python parity | **41/41 PASS** (1e-10) |
| GPU max speedup | **104x** (transformer medium) |
| GPU coverage | **~97%** |
| Multi-GPU parity | **384/384 bit-identical** |
| Dispatch overhead | **<=1.04x** (9/10 ops) |

---

## Key Validation Binaries

- `validate_isomorphism` — 6-primitive decomposition
- `validate_gemm_attention` — core neural primitives
- `validate_dispatch_parity` — multi-GPU bit-identical
- `validate_helixvision` — Evoformer, IPA, diffusion
- `validate_all` — full validation suite (244 binaries)
- `neuralspring_guidestone` — guideStone Level 5 (19 certification tests)
- `validate_ltee_b3_allele_trajectory` — LSTM+HMM+ESN allele classifier (16/16)
- `validate_ltee_b4_citrate_esn` — ESN citrate early-warning (16/16)

---

## Notebooks — sporePrint

| # | Notebook | Focus |
|---|----------|-------|
| 01 | `01-composition-validation.ipynb` | Deploy graphs, bond types, capabilities, discovery tiers |
| 02 | `02-benchmark-comparison.ipynb` | Rust vs Python timing, GPU speedups, guideStone phases |
| 03 | `03-ecosystem-evidence.ipynb` | 134 experiments, gap resolution, security posture |
| 04 | `04-cross-spring-connections.ipynb` | Primal consumption matrix, ecosystem flows |
| 05 | `05-btsp-security-deep-dive.ipynb` | Per-primal BTSP posture, security convergence arc |

---

## Notebooks — Paper Baselines (2 faculties, 8 notebooks)

Publishable-grade Jupyter notebooks with full inline Python/NumPy implementations of
peer-reviewed science. Each notebook is the **math validation base** — the foundation
layer that Rust, GPU, and primal IPC are validated against. Self-contained; executable
on JupyterHub without the neuralSpring repo.

### Batch 1: Dolson Faculty (Evolutionary Computation)

| Paper | Notebook | Citation | Checks |
|-------|----------|----------|--------|
| 011 | [`paper-011-counterdiabatic-evolution.ipynb`](../notebooks/papers/paper-011-counterdiabatic-evolution.ipynb) | Iram, Dolson et al. (2020) *Nature Physics* 17:135-142 | 11/11 |
| 012 | [`paper-012-modes-toolbox.ipynb`](../notebooks/papers/paper-012-modes-toolbox.ipynb) | Dolson et al. (2019) *Artificial Life* 25(1):50-73 | 9/9 |
| 013 | [`paper-013-eco-dynamics.ipynb`](../notebooks/papers/paper-013-eco-dynamics.ipynb) | Dolson & Ofria (2018) *GECCO '18 Companion* | 7/7 |
| 014 | [`paper-014-directed-evolution.ipynb`](../notebooks/papers/paper-014-directed-evolution.ipynb) | Dolson, Banzhaf, Ofria (2022) *eLife* 11:e79665 | 8/8 |
| 015 | [`paper-015-swarm-robotics.ipynb`](../notebooks/papers/paper-015-swarm-robotics.ipynb) | Foreback, Bohm, Dolson (2025) IEEE | 11/11 |

### Batch 2: Liu Faculty (HMM & Phylogenetic Inference)

| # | Notebook | Paper | Checks |
|---|----------|-------|--------|
| 016 | [`paper-016-hmm-phylo.ipynb`](../notebooks/papers/paper-016-hmm-phylo.ipynb) | Liu et al. (2014) PLoS Comp Bio | 10/10 |
| 017 | [`paper-017-sate-alignment.ipynb`](../notebooks/papers/paper-017-sate-alignment.ipynb) | Liu et al. (2009) Science | 8/8 |
| 018 | [`paper-018-introgression.ipynb`](../notebooks/papers/paper-018-introgression.ipynb) | Liu et al. (2015) PNAS | 8/8 |

**Total:** 8 notebooks, 72/72 checks PASS, 3,337 lines of validated Python source.
Faculties: Emily Dolson (Evolutionary Computation), Kevin Liu (Phylogenetic Inference).
Remaining batches: 19 papers across 4 additional faculties.

---

## Frozen Data

| File | Contents |
|------|----------|
| `validation-state.json` | Test counts, capabilities, code quality, guideStone |
| `experiment-catalog.json` | 134 experiments, 6 faculties, validation tiers |
| `security-posture.json` | BTSP, cargo-deny, unsafe, BLAKE3 checksums |
| `cross-spring-matrix.json` | 8 primal dependencies, proto-nucleate |
| `benchmark-data.json` | Rust vs Python, GPU, multi-GPU, isomorphic primitives |
| `gap-status.json` | 28 gaps, 28 resolved |
| `paper-baselines.json` | 8 paper notebooks, 72 checks, 2 faculties, BarraCUDA mappings |

---

## Ecosystem

- **Edition**: 2024
- **MSRV**: 1.87
- **barraCuda**: v0.4.0
- **primalSpring**: v0.9.27+ (Wave 46, 458 methods)
- **genomeBin**: v5.1 (46 binaries, 6 target triples)
- **Bond type**: Metallic
- **Trust model**: InternalNucleus
- **Proto-nucleate**: 7 validation capabilities, 6 primal dependencies
- **Isomorphism Theorem**: all neural architectures decompose into 6 primitives (GEMM, Attention, Normalization, Nonlinearity, Reduction, Gating)
- **helixVision**: sovereign AlphaFold2/3 structure prediction primitives

---

## See Also

- [Spring Catalog](https://primals.eco/architecture/spring-catalog-status-science-and-evolution/) on primals.eco
- [baseCamp Papers 01, 02, 04, 05, 06, 07](https://primals.eco/science/)

---

**Provenance:** [primals.eco](https://primals.eco) | neuralSpring Session S224
