# neuralSpring → ToadStool/BarraCUDA Handoff V28

**Session 63 — BandwidthTier Wiring + Cross-Spring Evolution Benchmark**
**Date**: February 25, 2026
**ToadStool HEAD**: `02207c4a`

---

## Executive Summary

Session 63 wires upstream `BandwidthTier` detection and NVK allocation guard into
neuralSpring's `Dispatcher`, then runs the full cross-spring benchmark suite against
the S62 `ToadStool`. This completes the S62 sync cycle with infrastructure wiring
and empirical performance validation across all three Springs' shader contributions.

---

## Current State

| Metric | Value |
|--------|-------|
| `cargo test --lib` | **500 PASS** |
| `validate_all` | **145/146 PASS** (1 pre-existing logsumexp) |
| `validate_cross_spring_evolution` | **22/22 PASS** |
| `cargo clippy --all-targets` (pedantic + nursery) | **0 warnings** |
| WGSL shaders absorbed | **21/21** (zero local WGSL remaining) |
| `ToadStool` HEAD | `02207c4a` |
| Named tolerances | **101+** |
| Coverage | **93.17%** |
| `BandwidthTier` detected | `PciE4x16` (RTX 4070) |

---

## What Changed (S63)

### 1. `BandwidthTier` Detection

Wired `barracuda::unified_hardware::BandwidthTier::detect_from_adapter_name()` into
the `Dispatcher`. On initialization:

```text
[dispatch] GPU available: NVIDIA GeForce RTX 4070 (DiscreteGpu, Vulkan, f64=Hybrid, pcie=PciE4x16)
```

Added `Dispatcher::bandwidth_tier()` public accessor for downstream decision-making
(transfer cost modelling, mixed-hardware dispatch).

### 2. NVK Allocation Guard

Added `Dispatcher::check_allocation_safe(total_bytes)` which delegates to upstream
`GpuDriverProfile::check_allocation_safe()`. Protects against NVK (TITAN V)
PTE-faulting at ~1.2 GB combined GPU allocation.

### 3. Cross-Spring Benchmark Suite (S63 Data)

#### Typed GPU Ops — All Three Springs (RTX 4070, `--release`)

| Op | Size | Median (µs) | Origin Spring | Session |
|----|------|-------------|---------------|---------|
| `BatchFitnessGpu` | 1024×64 | 3,033 | neuralSpring (ML) | S-25 |
| `PairwiseL2Gpu` | 128×16 | 3,154 | neuralSpring (MODES) | S-42 |
| `BatchIprGpu` | 32×64 | 2,364 | neuralSpring (Anderson) | S-25 |
| `SpatialPayoffGpu` | 32×32 | 2,901 | neuralSpring (game theory) | S-25 |
| `PairwiseHammingGpu` | 64×100 | 2,678 | neuralSpring (SATé) | S-25 |
| `HmmBatchForwardF64` | 4s×50t×32b | 3,325 | wetSpring (phylo) | S-39 |
| `BatchedEighGpu` | 12×12×40 | 7,402 | hotSpring (nuclear) | S-39 |

#### Rewire Evolution — f32 Tensor → f64 Upstream (10,000 elements)

| Op | f32 Tensor (µs) | f64 Upstream (µs) | Speedup | Cross-Spring Origin |
|----|-----------------|-------------------|---------|---------------------|
| Variance | 9,949 | 2,847 | **3.49×** | hotSpring Welford |
| Pearson | 4,679 | 3,508 | **1.33×** | wetSpring + hotSpring |
| Entropy | 6,317 | 2,468 | **2.56×** | wetSpring fused map-reduce |

#### Rewired Dispatcher Methods — Upstream vs CPU

| Method | Origin | n | Upstream (µs) | CPU (µs) |
|--------|--------|---|---------------|----------|
| `matmul` | hotSpring precision | 128 | 2,714 | 325 |
| `softmax` | hotSpring numerics | 1024 | 4.7 | 4.7 |
| `gelu` | neuralSpring ML | 1024 | 19.4 | 15.1 |
| `mean` | hotSpring reduce | 1024 | 0.4 | 0.4 |
| `hmm_forward` | wetSpring bio | 32 | 0.5 | 0.5 |

---

## Cross-Spring Provenance Summary

### What neuralSpring evolved → absorbed upstream

| Contribution | Category | Upstream API | Status |
|-------------|----------|-------------|--------|
| `eigh_householder_qr` | Eigensolve | `ops::linalg::eigh_f64` | Absorbed |
| `batch_fitness_eval.wgsl` | EA fitness | `ops::bio::batch_fitness` | Absorbed |
| `pairwise_hamming/jaccard/l2.wgsl` | Distance | `ops::bio::pairwise_*` | Absorbed |
| `batch_ipr.wgsl` | Spectral | `spectral::batch_ipr` | Absorbed |
| `spatial_payoff.wgsl` | Game theory | `ops::bio::spatial_payoff` | Absorbed |
| `head_split/head_concat.wgsl` | MHA reshape | `ops::mha` | Absorbed (S-03b) |
| `empirical_spectral_density` | Stats | `stats::empirical_spectral_density` | Absorbed (S54) |
| `effective_rank` | Linear algebra | `linalg::effective_rank` | Absorbed (S54) |
| 4-tier `KernelRouter` | Matmul | `ops::matmul` | Absorbed (S-02) |

### What neuralSpring leans on from hotSpring

| From hotSpring | Speedup / Benefit |
|---------------|------------------|
| `VarianceReduceF64` (Welford) | **3.49×** faster variance |
| `GpuDriverProfile` | Hardware-adaptive f64 strategy |
| `BandwidthTier` | PCIe tier detection (wired S63) |
| `BatchedEighGpu` (NAK) | Single-dispatch eigensolve |
| `df64_core` | Double-float f32-pair emulation |
| `pow_f64` polyfill | Transcendental workaround |

### What neuralSpring leans on from wetSpring

| From wetSpring | Speedup / Benefit |
|---------------|------------------|
| `FusedMapReduceF64` | **2.56×** faster entropy |
| `CorrelationF64` | **1.33×** faster Pearson + f64 precision |
| `HmmBatchForwardF64` | 10⁹× precision over f32 HMM |
| `log_f64` coefficient fix | All f64 shader math |
| Ada Lovelace workaround | RTX 4070 GPU support |

---

## Outstanding Absorption Targets

### Still Open

| Issue | Category | Notes |
|-------|----------|-------|
| `LogSumExp` (S-16) | Buffer mismatch | `validate_barracuda_logsumexp` still fails |
| `Tensor::mean()` dim | API gap | Upstream `mean()` is scalar-only |
| HMM backward/Viterbi dispatch | Missing | `hmm_backward_dispatch` / `hmm_viterbi_dispatch` not yet in `domain_ops` |
| Tridiagonal eigensolver | Missing | Lanczos tridiagonalization + eigenvector extraction |

### Opportunities from S62 Features

| Feature | Potential |
|---------|-----------|
| `Conv2dGpu` | Full GPU LeNet-5 pipeline (currently Tensor API) |
| `SpMM f64` | Sparse graph kernels |
| `TransE f64` | Knowledge graph scoring |
| `PeakDetectF64` | Signal processing validation |
| `ComputeDispatch` builder | Reduce validation binary boilerplate |

---

## Modified Files (S63)

| File | Change |
|------|--------|
| `src/gpu_dispatch/mod.rs` | `BandwidthTier` import, `bandwidth_tier()`, `check_allocation_safe()`, PCIe tier logging |
| `whitePaper/CROSS_SPRING_SHADER_LINEAGE.md` | S63 benchmark data, S-03b section, validation evidence |
| `specs/CROSS_SPRING_EVOLUTION.md` | S62-S63 section with benchmarks |
| `experiments/README.md` | Experiment 031 entry |
| `README.md` | Sessions 40–63, V28 handoff reference |
| `wateringHole/handoffs/` | V27 archived, V28 created |

---

*The absorption cycle completes: neuralSpring evolved workarounds, `ToadStool`
absorbed them, neuralSpring now leans on the shared engine. Every benchmark run
proves the three Springs' contributions work in concert through unified APIs.*
