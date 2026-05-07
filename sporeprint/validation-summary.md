+++
title = "neuralSpring Validation Summary"
description = "ML primitives and sovereign structure prediction — 4,900+ checks, Isomorphism Theorem, 83.6x faster than Python"
date = 2026-05-07

[taxonomies]
primals = ["barracuda", "toadstool", "biomeos", "squirrel"]
springs = ["neuralspring", "hotspring", "wetspring", "groundspring"]
+++

# neuralSpring — sporePrint Validation Summary

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

> **Session:** S188 | **Date:** May 7, 2026 | **Version:** 0.1.0
> **Tier:** 2 (sporePrint: frozen data + notebooks)

---

## Headline Numbers

| Metric | Value |
|--------|-------|
| **Workspace lib tests** | 1,387 (1,234 lib + 73 forge + 80 playGround) |
| **Proptest properties** | 24 |
| **Python baselines** | 397 PASS |
| **Rust+GPU checks** | 4,500+ |
| **Total validation checks** | 4,900+ |
| **Binaries** | 269 (244 validate, 18 bench, 7 other) |
| **Experiments** | 134 across 11 domains |
| **Papers reproduced** | 27 (6 faculties) |
| **Capabilities** | 30 (9 domains) |
| **Named tolerances** | 233 |
| **guideStone** | Level 3 — 29/29 bare ALL PASS (P1-P5) |
| **BTSP** | 13/13 mandatory |
| **PRIMAL_GAPS** | 14 main (2 resolved, 13 appendix resolved) |

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
| Rust vs Python geomean | **83.6x** (11 domains) |
| Fastest speedup | **1,104x** (multi-objective) |
| CPU↔Python parity | **39/39 PASS** (1e-10) |
| GPU max speedup | **104x** (transformer medium) |
| GPU coverage | **~97%** |
| Multi-GPU parity | **384/384 bit-identical** |
| Dispatch overhead | **≤1.04x** (9/10 ops) |

---

## Key Validation Binaries

- `validate_isomorphism` — 6-primitive decomposition
- `validate_gemm_attention` — core neural primitives
- `validate_dispatch_parity` — multi-GPU bit-identical
- `validate_helixvision` — Evoformer, IPA, diffusion
- `validate_all` — full validation suite (244 binaries)
- `neuralspring_guidestone` — guideStone Level 3 (29/29 bare)

---

## Notebooks

| # | Notebook | Focus |
|---|----------|-------|
| 01 | `01-composition-validation.ipynb` | Deploy graphs, bond types, capabilities, discovery tiers |
| 02 | `02-benchmark-comparison.ipynb` | Rust vs Python timing, GPU speedups, guideStone phases |
| 03 | `03-ecosystem-evidence.ipynb` | 134 experiments, gap resolution, security posture |
| 04 | `04-cross-spring-connections.ipynb` | Primal consumption matrix, ecosystem flows |
| 05 | `05-btsp-security-deep-dive.ipynb` | Per-primal BTSP posture, security convergence arc |

---

## Frozen Data

| File | Contents |
|------|----------|
| `validation-state.json` | Test counts, capabilities, code quality, guideStone |
| `experiment-catalog.json` | 134 experiments, 6 faculties, validation tiers |
| `security-posture.json` | BTSP, cargo-deny, unsafe, BLAKE3 checksums |
| `cross-spring-matrix.json` | 8 primal dependencies, proto-nucleate |
| `benchmark-data.json` | Rust vs Python, GPU, multi-GPU, isomorphic primitives |
| `gap-status.json` | 14 gaps, 13 resolved, 5 composition evolution |

---

## Ecosystem

- **Edition**: 2024
- **MSRV**: 1.87
- **barraCuda**: v0.3.12
- **primalSpring**: v0.9.17+
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

**Provenance:** [primals.eco](https://primals.eco) | neuralSpring Session S188
