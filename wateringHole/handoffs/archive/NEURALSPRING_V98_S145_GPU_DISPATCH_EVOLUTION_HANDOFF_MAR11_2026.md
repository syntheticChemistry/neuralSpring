# neuralSpring V98 S145 → GPU Dispatch Evolution Handoff

**Date:** March 11, 2026
**From:** neuralSpring S145 (1115 lib tests, 73 forge tests, 0 clippy warnings)
**To:** barraCuda, toadStool, coralReef teams
**Supersedes:** `NEURALSPRING_V97_S144_TOADSTOOL_BARRACUDA_EVOLUTION_HANDOFF_MAR10_2026.md` (archived)
**License:** AGPL-3.0-only

---

## Executive Summary

neuralSpring S145 is a GPU infrastructure evolution sprint that:

1. **Synced to barraCuda v0.3.5** (`0649cd0`) — gains `ReduceScalarPipeline` f64 fix, `BatchedComputeDispatch<B>`, `CoralReefDevice` `GpuBackend`, `FmaPolicy`, `stable_gpu` specials, `tridiag_eigh_gpu`, 36 tolerances
2. **Rewired 5 workloads** from local to absorbed (20→25): `FusedChiSquaredGpu`, `FusedKlDivergenceGpu`, `hmm_backward`, `hmm_viterbi`, `PairwiseL2Gpu` (matrix)
3. **Evolved NUCLEUS pipeline** to GPU dispatch — `eigensolve` and `attention_anderson` stages route through `Dispatcher` (GPU when available, CPU fallback)
4. **Mixed-hardware composition DAG** — `composition_pipeline()` now marks `eigensolve` and `attention_anderson` as `GpuOnly`, with provenance recording actual execution substrate
5. **4 GPU experiments** (Exp 103-106): eigensolve pipeline, batched spectral, sovereign compile validation, mixed-hardware composition

---

## Part 1: Upstream Sync

### barraCuda v0.3.5 (`0649cd0`)

| Feature | Status in neuralSpring |
|---------|----------------------|
| `ReduceScalarPipeline` f64 fix | Available — DF64 accumulation replaces broken f64 shared memory |
| `BatchedComputeDispatch<B>` | Available — batch multiple dispatches into one submission |
| `CoralReefDevice` `GpuBackend` | Available (feature-gated `sovereign-dispatch`) |
| `FmaPolicy` + `domain_requires_separate_fma()` | Available for FMA-sensitive domains |
| `tridiag_eigh_gpu` (`BatchedTridiagEighGpu`) | Available for GPU eigensolve in NUCLEUS pipeline |
| `FusedChiSquaredGpu` | Rewired — replaces local chi-squared |
| `FusedKlDivergenceGpu` | Rewired — replaces local KL divergence |
| `hmm_backward` | Rewired — full backward pass GPU WGSL shader |
| 36 tolerances + introspection | Available via `by_name()`, `all_tolerances()` |
| `has_df64_spir_v_poisoning()` (renamed) | Available — replaces old `has_nvvm_df64_poisoning_risk()` |

**API break resolved:** `ESNConfig` gained 3 new SGD fields (`sgd_learning_rate`, `sgd_min_iterations`, `sgd_max_iterations`). neuralSpring updated with default values.

### toadStool S146 (`751b3849`)

Acknowledged: `nvvm_transcendental_risk` in `gpu.info`, `PrecisionBrain` in `compile_wgsl_multi`, VRAM-aware routing, 19 `SpringDomain` variants, `PcieTopologyGraph` (stable), `HealthSpring`.

### coralReef Iter 33 (`b783217`)

Acknowledged: 46/46 sovereign compile, NVVM poisoning bypass validated, 4 DRM ioctl struct ABI fixes, Nouveau UAPI migration (VM_INIT + VM_BIND + EXEC).

---

## Part 2: Workload Rewire (5 local → absorbed)

| Workload | Previous | Now | Upstream Primitive |
|----------|----------|-----|-------------------|
| `chi_squared_gpu` | Local | **Absorbed** | `FusedChiSquaredGpu` |
| `kl_divergence_gpu` | Local | **Absorbed** | `FusedKlDivergenceGpu` |
| `hmm_backward` | Local | **Absorbed** | `barracuda::ops::bio::hmm_backward` |
| `hmm_viterbi` | Local | **Absorbed** | `barracuda::ops::bio::hmm_viterbi` |
| `pairwise_l2_matrix` | Local | **Absorbed** | `PairwiseL2Gpu` |

**Remaining local:** Only `replicator_step` (small GEMV, no upstream equivalent).

**Workload summary:** 25 absorbed, 1 local, 2 CPU-only.

---

## Part 3: NUCLEUS Pipeline GPU Dispatch

The NUCLEUS pipeline executor (`nucleus_pipeline.rs`) now supports mixed-hardware execution:

### New API

```rust
pub fn execute_composition_pipeline_gpu(dispatcher: &Dispatcher) -> PipelineReport;
pub fn execute_graph_gpu(graph: &PipelineGraph, dispatcher: &Dispatcher) -> PipelineReport;
```

### Dispatch Strategy

| Stage | Substrate | Dispatcher Method |
|-------|-----------|------------------|
| `eigensolve` | `GpuOnly` | `dispatcher.eigh()` — `BatchedEighGpu` on GPU, `eigh_householder_qr` on CPU fallback |
| `attention_anderson` | `GpuOnly` | `dispatcher.attention_spectral_analysis()` — GPU eigensolve + IPR + LSR |
| `digester_anderson` | `CpuOnly` | Direct function call |
| `isomorphic_reservoir` | `CpuOnly` | Direct function call |
| `wdm_ensemble_qs` | `CpuOnly` | Direct function call |
| `introgression_nn` | `CpuOnly` | Direct function call |

### Provenance

`PipelineReport` now includes `gpu_stages` and `cpu_stages` counts. Each `StageResult` records the actual execution substrate, enabling cross-comparison between GPU and CPU paths.

---

## Part 4: GPU Experiments (Exp 103-106)

| Exp | Binary | Description |
|-----|--------|-------------|
| 103 | `validate_gpu_eigensolve_pipeline` | CPU vs GPU eigensolve precision sweep (N=8,16,32,64) |
| 104 | `validate_batched_spectral` | Sequential vs GPU-dispatch pipeline comparison, cross-experiment spectral summary |
| 105 | `validate_sovereign_compile` | Compute triangle readiness report — hardware, precision, sovereign dispatch status |
| 106 | `validate_mixed_composition_pipeline` | Full mixed-hardware DAG execution with per-stage substrate breakdown and transfer cost |

---

## Part 5: Compute Triangle Integration

neuralSpring is now wired to the full compute triangle:

```
toadStool S146 (discovery) → capabilities → NUCLEUS pipeline
barraCuda v0.3.5 (math)   → ComputeDispatch → Dispatcher
coralReef Iter 33 (compile) → sovereign binary → Dispatcher (feature-gated)
```

### RTX 4070 Readiness

Following hotSpring's pilot work (sovereign dispatch roadmap):
- The RTX 4070 (Ada/GSP-only) is the highest-ROI test target for sovereign compute
- neuralSpring's `Dispatcher` already probes GPU capabilities at runtime
- When `CoralReefDevice` sovereign dispatch matures, the `execute_graph_gpu` executor can route through it with zero code changes in the pipeline stages

---

## Part 6: Action Items

### For barraCuda (P2)

1. Consider exposing `ESNConfig` SGD defaults as public constants (currently crate-private)
2. `replicator_step` — small GEMV + update op; consider a lightweight `GemvUpdateGpu` for game theory

### For toadStool (P2)

1. Acknowledge neuralSpring NUCLEUS pipeline GPU dispatch — `PipelineGraph` is ready for `toadstool::orchestration` absorption
2. Spring pin update: neuralSpring S145, barraCuda v0.3.5

### For coralReef (P3)

1. RTX 4070 sovereign dispatch test — neuralSpring's eigensolve and spectral shaders are the first non-MD workload candidates

---

## Metrics

| Metric | S144 | S145 |
|--------|------|------|
| barraCuda | v0.3.3 (`83aa08a`) | v0.3.5 (`0649cd0`) |
| toadStool | S142 (`a86bc546`) | S146 (`751b3849`) |
| coralReef | Iter 29 (`2779c88`) | Iter 33 (`b783217`) |
| Lib tests | 1112 | **1115** |
| Forge tests | 73 | 73 |
| Modules | 47 | 47 |
| Binaries | 254 | **258** |
| Absorbed workloads | 20 | **25** |
| Local workloads | 6 | **1** |
| GPU pipeline stages | 0 | **2** (eigensolve, attention_anderson) |
| GPU experiments | — | **4** (Exp 103-106) |
| Clippy warnings | 0 | 0 |
