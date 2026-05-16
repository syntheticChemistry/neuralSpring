# neuralSpring V162 — Deep Debt Re-Audit (3rd Pass)

**From:** neuralSpring (S207b)
**To:** primalSpring (coordination), all spring teams
**Date:** 2026-05-16
**Session:** S207b — comprehensive deep debt re-audit per primalSpring directive

---

## Deep Debt Status: ZERO across all 7 priority areas

| Priority | Count | Evidence |
|----------|-------|----------|
| TODO/FIXME/HACK/STUB markers | **0** | Workspace-wide grep `src/**/*.rs` — zero actionable markers |
| Modern idiomatic Rust | **YES** | Edition 2024, MSRV 1.87, `#[expect]` everywhere (0 `#[allow]`), typed error hierarchy (`IpcError`) |
| External deps evolved | **All modern** | `wgpu 28`, `tokio 1.49`, `thiserror 2`, `clap 4.5`. No stale pins, no candidates for Rust-native replacement |
| Large files (>800L) refactored | **0 files >800L** | Largest: `tolerances/mod.rs` (776L). 158,166 total LOC across workspace |
| `unsafe` code → safe Rust | **0 unsafe** | `#![forbid(unsafe_code)]` on all library crates |
| Hardcoding → capability-based | **Done** | `config.rs` centralizes runtime resolution. Primal discovery is capability-based (`CapabilityRouter`, env-driven sockets). No hardcoded IPs/ports/paths |
| Mocks isolated to testing | **Done** | All `mock_*`/`FAKE_SOCKET` patterns inside `#[cfg(test)]`. CoralReef bridge stub is correct feature-gated capability absence (Tier 4 IPC-first) |

### Additional metrics

| Metric | Value |
|--------|-------|
| `unimplemented!()`/`todo!()`/`unreachable!()` | **0** |
| `panic!()` in production | **0** (only inside `#[cfg(test)]`) |
| `.unwrap()` in library production | **0** (36 in benchmark binary with `#[expect]`, 12 in fossils diagnostics) |

---

## Audit Question 1: Python benchmarks for barraCuda CPU (Rust)?

**YES — 15 CPU benchmark domains with 38.6× geometric mean speedup.**

### Infrastructure

- `scripts/run_all_baselines.sh` — orchestrates all 84 baseline scripts
- `src/bin/validate_barracuda_cpu_bench.rs` — authoritative Python/NumPy timing vs Rust comparison
- `src/validation/cpu_bench.rs` — subprocess execution framework (single-thread BLAS: `OPENBLAS_NUM_THREADS=1`)
- `control/generate_cpu_references.py` → `control/cpu_parity_references.json` — NumPy-derived reference values
- `src/bin/validate_cpu_math_parity.rs` — Rust vs Python/NumPy cross-language numerical parity

### 20 Python benchmark scripts (`bench_*.py`)

| Script | Domain | Paper |
|--------|--------|-------|
| `control/hmm_phylo/bench_hmm_forward.py` | HMM forward | 016-018 |
| `control/counterdiabatic/bench_nk_fitness.py` | NK fitness | 011 |
| `control/modes/bench_pairwise_l2.py` | Pairwise L2 | 012 |
| `control/eco_dynamics/bench_eco.py` | Eco batch fitness | 013 |
| `control/sate_alignment/bench_hamming.py` | Pairwise Hamming | 017 |
| `control/pangenome_selection/bench_jaccard.py` | Jaccard | 024 |
| `control/game_theory/bench_replicator.py` | Replicator dynamics | 019 |
| `control/regulatory_network/bench_rk4.py` | RK4 GRN | 020 |
| `control/spectral_commutativity/bench_commutator.py` | Commutator | 022 |
| `control/anderson_localization/bench_anderson.py` | Anderson IPR | 023 |
| `control/signal_integration/bench_hill_gate.py` | Hill gate | 021 |
| `control/directed_evolution/bench_multi_obj.py` | Multi-objective | 014 |
| `control/swarm_robotics/bench_swarm_nn.py` | Swarm NN | 015 |
| `control/meta_population/bench_meta_pop.py` | Global FST | 026 |
| `control/glucose_prediction/bench_glucose_lstm.py` | LSTM glucose | 026 |
| `control/industry_gpu/bench_cublas_gemm.py` | cuBLAS GEMM | GPU parity |
| `control/industry_gpu/bench_cudnn_ops.py` | cuDNN ops | GPU parity |
| `control/industry_gpu/bench_cufft.py` | cuFFT | GPU parity |
| `control/industry_gpu/bench_flash_attention.py` | Flash Attention | GPU parity |
| `control/industry_gpu/bench_cuda_common.py` | CUDA warmup/timing | GPU parity |

---

## Audit Question 2: Industry GPU benchmarks for barraCuda GPU parity?

### What exists

| Benchmark | Status | Evidence |
|-----------|--------|----------|
| **Kokkos** | Harness exists (`bench_kokkos_parity.rs`) | Baselines labeled ESTIMATED — not matched hardware |
| **cuBLAS** | Python control exists | `bench_cublas_gemm.py` via PyTorch |
| **cuDNN** | Python control exists | `bench_cudnn_ops.py` via PyTorch |
| **cuFFT** | Python control exists | `bench_cufft.py` via PyTorch |
| **Flash Attention** | Python control exists | `bench_flash_attention.py` |
| **Industry GPU parity binary** | Exists | `bench_industry_gpu_parity.rs` — WGSL vs CUDA via PyTorch |
| **Galaxy** | N/A | Workflow engine, not comparable kernel benchmark |
| **CUTLASS/Triton/TensorRT/NCCL** | Not referenced | Not applicable to sovereign WGSL shader stack |
| **PolyBench/SPEC** | Roadmap | Mentioned in `specs/BENCHMARK_ANALYSIS.md` — not implemented |

### What's needed for deeper parity

- **Matched-hardware Kokkos runs** (current baselines are estimated)
- **strandGate hardware validation** (3090 + 6950) for dual-vendor coverage
- coralReef v0.1.0 unlocks `compile_shader_universal` routing

---

## Audit Question 3: What have we not implemented/verified/tested?

### Coverage model

| Layer | Count | Evidence |
|-------|-------|---------|
| Library unit tests | 734 | `cargo test --lib` |
| Integration tests | 11 | `tests/integration.rs` |
| Forge tests | 73 | `metalForge/forge` |
| Playground tests | 80 | `playGround` |
| Doc tests | 12 | `cargo test --doc` |
| **Validation binaries** | **130+** | `Cargo.toml [[bin]]` entries — **not run by `cargo test`** |

### Known gaps

- **`validate_*` binaries**: 130+ exist but are **not part of `cargo test`** — require `cargo run --bin` or `make validate`
- **Live IPC tests**: All `#[ignore]` with reasons (requires daemon, GPU, network)
- **Integration test % of capability surface**: Not computed (would need `cargo tarpaulin`)
- **LTEE B2-B9**: QUEUED in paper tracker
- **baseCamp B-16 to B-21 (immunological Anderson)**: Proposal only, 0 computational checks

---

## Audit Question 4: Papers not reviewed from queue?

**Paper queue is CLOSED — 27/27 complete** (`specs/PAPER_REVIEW_QUEUE.md`)

| Phase | Papers | Status |
|-------|--------|--------|
| Phase 0 (synthetic, exps 1-5) | 5 | Complete |
| Phase 0+ (PINN, DeepONet, LeNet, ERA5, quant) | 5 | Complete |
| Papers 11-26 | 16 | Complete |
| Paper 27 (Wang/Liao digestion) | 1 | Complete |
| **Total** | **27** | **All complete** |

### Additional paper-adjacent items

| Item | Status |
|------|--------|
| baseCamp B-01 to B-15 | Primitives validated via `nS-*` experiments |
| baseCamp B-16 to B-21 (immunological Anderson) | **Proposal only** — 0 computational checks |
| LTEE B1 (Barrick) | **COMPLETE** — Python + `validate_ltee_b1_mutation_accumulation` |
| LTEE B2-B9 | **QUEUED** |
| Notebooks `papers/paper-011..018` | Executable reproduction summaries with DOIs |

---

## Audit Question 5: Datasets to examine?

### Currently used

| Dataset | Domain | Source |
|---------|--------|--------|
| Synthetic benchmarks | All domains | Generated (NK, swarm, Hamiltonians) |
| ERA5 (Open-Meteo) | Weather/LSTM | `control/lstm_weather/era5_east_lansing_daily.npz` |
| MNIST | Vision | torchvision |
| GPT-2 safetensors | Spectral analysis | HuggingFace (read-only `nS-01`) |
| WDM/FPEOS | Warm dense matter | Militzer et al. |
| MODES CSVs | Phylogenetics | GitHub (Paper 12) |
| LTEE B1 | Evolution | `control/ltee_mutation_accumulation/expected_values.json` |

### Roadmap datasets (documented in EXTENSION_PLAN.md)

| Dataset | Domain | Priority |
|---------|--------|----------|
| **NOAA GHCND** | Climate/weather | airSpring chaining |
| **LTEE assembled genomes** (SRA) | Evolution | LTEE B2+ queue |
| **OhioT1DM / OpenAPS** | CGM/glucose | healthSpring chaining (Paper 026 extension) |
| **UniProt / AlphaFold DB** | Protein structure | coralForge pipeline |
| **FAO-56 ET₀** | Agriculture | airSpring cross-validation |

---

## Current State

| Metric | Value |
|--------|-------|
| Session | S207b |
| Workspace tests | 910 |
| Clippy errors | 0 |
| Capabilities | 35 |
| Validation scenarios | 7 |
| Deep debt | Zero across all 7 priorities |
| Files >800L | 0 (max 776L) |
| unsafe code | 0 (forbid) |
| Evolution | composing |
| Signal API | Wave 17 (primal.announce + nest.store) |
| Handoff | V162 |
