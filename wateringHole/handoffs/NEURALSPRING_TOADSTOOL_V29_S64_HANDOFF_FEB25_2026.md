# neuralSpring → ToadStool/BarraCUDA Handoff V29

**Session 64 — Forge Evolution: Substrate Discovery, Workload Tracking, Write-Phase Extensions**
**Date**: February 25, 2026
**ToadStool HEAD**: `02207c4a`
**Forge version**: `neural-spring-forge` v0.2.0

---

## Executive Summary

Session 64 evolves neuralSpring's metalForge/forge crate to match hotSpring and
wetSpring: substrate discovery, workload tracking with `ShaderOrigin`, and new
write-phase WGSL extensions. This makes neuralSpring's contributions absorbable
by ToadStool in the same way hotSpring and wetSpring contribute.

---

## Current State

| Metric | Value |
|--------|-------|
| `cargo test --lib` | **500 PASS** |
| `neural-spring-forge` tests | **43 PASS** (was 30) |
| `cargo clippy --all-targets` (pedantic + nursery) | **0 warnings** |
| WGSL shaders in forge | **23** (was 21) |
| Workloads: Absorbed / Local / CPU-only | **20 / 6 / 2** |
| Forge version | **v0.2.0** (was v0.1.0) |

---

## What Changed (S64)

### 1. Forge Crate Evolution — Matching hotSpring/wetSpring Pattern

| New Module | Purpose | Pattern From |
|------------|---------|-------------|
| `substrate.rs` | `Substrate`, `SubstrateKind`, `Identity`, `Properties`, `Capability` | hotSpring |
| `probe.rs` | `probe_gpus()` via wgpu, `probe_cpu()` via procfs | hotSpring |
| `inventory.rs` | `discover()` → all substrates on machine | hotSpring |
| `workloads.rs` | `MlWorkload` with `ShaderOrigin` (Absorbed/Local/CpuOnly) | wetSpring |

### 2. Write-Phase WGSL Extensions

| Shader | Purpose | Replaces |
|--------|---------|----------|
| `chi_squared_f64.wgsl` | Fused `sum((o-e)²/e)` — single-pass f64 reduction | CPU elementwise loop in `gpu_ops::reduction::chi_squared_gpu` |
| `kl_divergence_f64.wgsl` | Fused `sum(p*ln(p/q))` — single-pass f64 reduction | CPU log-ratio loop in `gpu_ops::reduction::kl_divergence_gpu` |

Both are absorption candidates for `barracuda::ops::fused_chi_squared_f64` and
`barracuda::ops::fused_kl_divergence_f64`.

### 3. Workload Absorption Inventory

#### Absorbed (20 workloads — upstream `BarraCUDA` APIs)

| Workload | Upstream Primitive | Cross-Spring Origin |
|----------|-------------------|---------------------|
| matmul | `matmul_dispatch` | hotSpring precision |
| softmax | `softmax_dispatch` | hotSpring numerics |
| gelu | `gelu_dispatch` | neuralSpring ML → S52 |
| mean | `mean_dispatch` | hotSpring reduce |
| variance | `VarianceReduceF64` | hotSpring Welford |
| pearson_correlation | `CorrelationF64` | wetSpring + hotSpring |
| shannon_entropy | `FusedMapReduceF64` | wetSpring fused |
| hmm_forward | `hmm_forward_dispatch` | wetSpring bio → S52 |
| frobenius_norm | `frobenius_norm_dispatch` | hotSpring reduction |
| transpose | `transpose_dispatch` | hotSpring precision |
| l2_distance | `l2_distance_dispatch` | neuralSpring MODES |
| multi_head_attention | `MultiHeadAttention` | neuralSpring → S-03b |
| batch_fitness | `BatchFitnessGpu` | neuralSpring EA S-25 |
| pairwise_l2 | `PairwiseL2Gpu` | neuralSpring MODES S-42 |
| pairwise_hamming | `PairwiseHammingGpu` | neuralSpring SATé S-25 |
| pairwise_jaccard | `PairwiseJaccardGpu` | neuralSpring pangenome S-25 |
| spatial_payoff | `SpatialPayoffGpu` | neuralSpring game theory S-25 |
| batch_ipr | `BatchIprGpu` | neuralSpring Anderson S-25 |
| eigensolve | `BatchedEighGpu` | hotSpring nuclear S-39 |
| hmm_batch_forward_f64 | `HmmBatchForwardF64` | wetSpring phylo S-39 |

#### Local — Write Phase (6 workloads for ToadStool absorption)

| Workload | What It Needs | Status |
|----------|-------------|--------|
| `chi_squared_gpu` | Fused GPU op (`chi_squared_f64.wgsl` written) | **WGSL ready** |
| `kl_divergence_gpu` | Fused GPU op (`kl_divergence_f64.wgsl` written) | **WGSL ready** |
| `hmm_backward` | `hmm_backward_dispatch` in `domain_ops` | Needs upstream |
| `hmm_viterbi` | `hmm_viterbi_dispatch` in `domain_ops` | Needs upstream |
| `pairwise_l2_matrix` | `PairwiseL2MatrixF64` typed op (f64 variant) | Needs upstream f64 |
| `replicator_step` | Small GEMV + update (domain-specific) | Low priority |

#### CPU-Only (2 workloads — inherently sequential)

| Workload | Why CPU-only |
|----------|-------------|
| `pareto_front` | O(n²) dominance check, branching-heavy |
| `mantel_test` | Permutation + correlation, sequential |

---

## Absorption Recommendations for ToadStool

### Immediate (WGSL ready, can absorb now)

1. **`chi_squared_f64.wgsl`** → `barracuda::ops::fused_chi_squared_f64`
   - Binding layout: uniform{n,pad}, storage{observed}, storage{expected}, storage{partials}
   - Workgroup size: 256, tree reduction
   - Pattern: identical to `fused_map_reduce_f64.wgsl`

2. **`kl_divergence_f64.wgsl`** → `barracuda::ops::fused_kl_divergence_f64`
   - Binding layout: uniform{n,pad}, storage{p}, storage{q}, storage{partials}
   - Workgroup size: 256, tree reduction with 1e-30 guard
   - Pattern: identical to `fused_map_reduce_f64.wgsl`

### Medium Term (needs Rust API work)

3. **`hmm_backward_dispatch`** — Mirror `hmm_forward_dispatch` for backward pass
4. **`hmm_viterbi_dispatch`** — Argmax + traceback dispatch
5. **`PairwiseL2MatrixF64`** — f64 variant of `PairwiseL2Gpu` (currently f32-only)
6. **`BatchIprF64`** — f64 variant of `BatchIprGpu` (currently f32-only)

---

## Modified Files (S64)

| File | Change |
|------|--------|
| `metalForge/forge/Cargo.toml` | v0.1.0 → v0.2.0 |
| `metalForge/forge/src/lib.rs` | New module exports: substrate, probe, inventory, workloads |
| `metalForge/forge/src/substrate.rs` | **New**: Runtime compute device abstraction |
| `metalForge/forge/src/probe.rs` | **New**: GPU + CPU probing |
| `metalForge/forge/src/inventory.rs` | **New**: Substrate assembly |
| `metalForge/forge/src/workloads.rs` | **New**: 28 ML workloads with `ShaderOrigin` tracking |
| `metalForge/forge/src/shaders.rs` | +2 write-phase shaders (23 total) |
| `metalForge/shaders/chi_squared_f64.wgsl` | **New**: Fused chi-squared f64 |
| `metalForge/README.md` | Updated with forge v0.2.0 module table |
| `metalForge/ABSORPTION_MANIFEST.md` | Updated HEAD, session range |
| `experiments/README.md` | Experiment 032 entry |
| `README.md` | Sessions 40–64, V29 handoff reference |

---

*Following hotSpring/wetSpring: evolve locally, validate rigorously, hand off
cleanly. The forge now speaks the same language as its sibling Springs.*
