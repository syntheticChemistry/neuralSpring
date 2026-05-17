# neuralSpring V166 — Deep Debt Re-Audit (6th Pass)

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**From:** neuralSpring (S210, V166)
**To:** primalSpring, upstream primals
**Date:** 2026-05-17
**Supersedes:** V165 (Live Composition + Live Data Chains)

---

## Summary

6th comprehensive deep debt audit across all 7 priority areas. Upstream Wave 20 audit confirmed zero code debt. One LOC policy violation fixed.

## Deep Debt Audit Results (7 Priorities)

| Priority | Area | Status |
|----------|------|--------|
| 1 | TODO/FIXME/HACK/STUB markers | **ZERO** — no actionable markers in `src/**/*.rs` |
| 2 | Modern idiomatic Rust | **CLEAN** — `unwrap_used`/`expect_used` warned via workspace lints, test-only usage with `#[expect]` |
| 3 | External dependencies | **MANAGED** — all via workspace `Cargo.toml`, semver specs, `cargo-deny` in CI |
| 4 | Large files (>800 LOC) | **FIXED** — `weight_loader.rs` (805→710) via `provenance_dispatch.rs` extraction |
| 5 | Unsafe code | **ZERO** — `#![forbid(unsafe_code)]` on all crate roots + workspace lint |
| 6 | Hardcoding | **ZERO** — all primal names via `primal_names.rs` + `CapabilityRouter` hints |
| 7 | Production mocks | **ZERO** — `CoralCompiler` stub is feature-gated `NotAvailable`, not fake success |

## Code Change

`weight_loader.rs` (805 LOC) split into:
- `weight_loader.rs` (710 LOC) — safetensors/JSON weight loading + NestGate storage
- `provenance_dispatch.rs` (107 LOC) — `store_to_nestgate_signal()`, `commit_session_signal()`, `store_science_result()`

## Audit Questions Answered

### Q1: Python benchmarks for barraCuda CPU (Rust)?

**Yes.** `validate_barracuda_cpu_bench` + 15 `control/**/bench_*.py` scripts benchmark Python/NumPy timing vs pure Rust barraCuda CPU across 15 domains (HMM, NK fitness, pairwise metrics, ecology, introgression, regulatory ODEs, spectral/Anderson, signal integration, evolution/swarm, LSTM-glucose). Additionally, `specs/BENCHMARK_ANALYSIS.md` documents Python vs WGSL-on-CPU (llvmpipe) vs GPU scaling for large GEMM/transformer.

### Q2: Industry GPU parity benchmarks?

**Partial.** PyTorch/CUDA GPU parity exists (`bench_industry_gpu_parity` + `control/industry_gpu/bench_cublas_gemm.py`, `bench_cudnn_ops.py`, `bench_cufft.py`, `bench_flash_attention.py`). Kokkos estimated baselines exist (`bench_kokkos_parity`). **Gaps**: Polybench/GPU, oneDNN not present. Galaxy N/A (workflow engine). SPEC/Rodinia explicitly out of scope.

### Q3: What have we NOT implemented?

Open evolution items (not blocking debt):
- LTEE queue: B2–B9, E2–E5 (QUEUED in `PAPER_REVIEW_QUEUE.md`)
- Tower BTSP session, Songbird-centric discovery (wip in PRIMAL_GAPS)
- Upstream barraCuda `special::plasma_dispersion` feature-gate (local workaround active)
- coralForge Phase D/E (multi-molecule tokenization, full MSA pipeline)
- All 9 validation scenarios implemented; Tier 2 live checks are environment-dependent

### Q4: Papers not reviewed from queue?

Main **27-paper queue: CLOSED** (all 27 complete). Outstanding:
- LTEE GuideStone subsection: B2–B9 (7 papers QUEUED)
- Eaves/Woldring bridge: E2–E5 (4 papers QUEUED)
- Immunological Anderson: baseCamp Sub-thesis 06 (proposal, awaiting wetSpring Exp 270–274)
- coralForge nF-* Phase D/E (structure prediction roadmap phases)

### Q5: Datasets to examine?

Priority datasets as the project matures:
- **OhioT1DM / OpenAPS** — real CGM data for glucose prediction (Paper 026)
- **MODES toolbox CSVs** — published artifact tables for Paper 012
- **UniRef90, BFD, MGnify, PDB/mmCIF** — full MSA/structure stack for coralForge
- **LTEE public data** — fitness/mutation time series for B2–B9 queue
- **Real phylogenetic alignments** — integration-scale inputs for Papers 016–018
- **CIFAR-10** — diversify landscape/Hessian studies beyond MNIST subsets
- **Independent EOS/opacity tables** — generalize WDM surrogate claims

## Quality

- `cargo check --workspace` — clean
- `cargo clippy --workspace` — 0 errors
- `cargo test --workspace` — 732 passed, 2 pre-existing env-dependent skips
- All files ≤800 LOC (verified via `find | wc`)
