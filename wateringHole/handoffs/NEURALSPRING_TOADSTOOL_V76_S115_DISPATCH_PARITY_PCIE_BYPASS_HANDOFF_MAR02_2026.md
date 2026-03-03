<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/BarraCUDA Handoff V76 — Dispatch Parity + ComputeDispatch Bridge + NUCLEUS PCIe Bypass

**Date**: March 2, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Session 115 — Full dispatch parity, ComputeDispatch evolution bridge, NUCLEUS mixed-hardware PCIe bypass
**Supersedes**: V75 (S113 Cross-Spring Evolution Benchmark)
**ToadStool HEAD**: `2dc26792` (S87)

---

## Executive Summary

- **53/53 dispatch parity**: Every `Dispatcher` method with a GPU path now has proven CPU↔GPU mathematical parity (+23 checks from V75's 30/30)
- **14/14 ComputeDispatch bridge**: New validator proves `neuralSpring Dispatcher` math is bit-identical to `barracuda::dispatch` functions (which ToadStool's 144-op `ComputeDispatch` wraps)
- **38/38 NUCLEUS PCIe bypass**: Full mixed-hardware pipeline validation — GPU↔NPU PCIe P2P bypass, Tower→Node→Nest atomic chain, biomeOS graph coordination
- **212/212 validate_all PASS**, 861 lib tests, 0 clippy, 0 fmt
- **232 validation/bench binaries** (up from 226 in V75)

---

## Part 1: Expanded Dispatch Parity (53/53)

### 1.1 New Operations Validated in S115

| Operation | CPU Path | GPU Path | Tolerance | Result |
|-----------|----------|----------|-----------|--------|
| `multi_obj_fitness` | `cpu_fallback::multi_obj_fitness` | `multi_obj_fitness_gpu` (WGSL) | `TENSOR_MATMUL_F32` (1e-3) | PASS |
| `swarm_nn_forward` | `cpu_fallback::swarm_nn_forward` | `swarm_nn_forward_gpu` (WGSL) | `TENSOR_EXACT_F32` | PASS |
| `integrate_ode_batch` | `cpu_fallback::integrate_ode_batch` | `Rk45AdaptiveGpu` (WGSL) | `GPU_RK4_F32` | PASS |
| `inter_population_af_variance` | `cpu_fallback::inter_pop_af_var` | `inter_population_af_variance_gpu` | `GPU_VARIANCE_F64` | PASS |
| `hmm_backward_step` | `cpu_fallback::hmm_backward_step` | `hmm_backward_step_gpu` (WGSL) | `TENSOR_TRANSCENDENTAL_F32` | PASS |
| `hmm_viterbi_step` | `cpu_fallback::hmm_viterbi_step` | `hmm_viterbi_step_gpu` (WGSL) | `TENSOR_TRANSCENDENTAL_F32` | PASS |
| `hmm_forward_chain` + `hmm_viterbi_chain` | CPU chain | GPU chain | `GPU_MEAN_DISPATCH_F32` / `TENSOR_TRANSCENDENTAL_F32` | PASS |
| `detect_introgression` | `cpu_fallback::detect_introgression` | GPU dispatch | `GPU_ENTROPY_F64` | PASS |

### 1.2 Previously Validated (V75 and earlier, 30 ops)

All 30 previously validated operations continue to pass: matmul, transpose, softmax, gelu, sigmoid, variance, mean, entropy, l2_distance, frobenius_norm, hmm_forward, rk4_step, eigensolve, diversity (Shannon/Simpson/BrayCurtis/chao1), hill_gate, thermal_diversity, global_fst_variance_decomposition, pairwise_fst_full, etc.

### 1.3 Coverage

Every `Dispatcher` method that has both a CPU fallback and a GPU path now has proven parity. No gaps remain.

---

## Part 2: ComputeDispatch Evolution Bridge (14/14)

New binary: `validate_compute_dispatch_evolution`

This validator proves that neuralSpring's `Dispatcher` operations produce identical results to calling the underlying `barracuda::dispatch` functions directly — the same functions that ToadStool's 144-op `ComputeDispatch` wraps.

| Check | What It Proves | Result |
|-------|---------------|--------|
| matmul bridge | `Dispatcher::mat_mul` == `barracuda::dispatch::matmul_dispatch` | PASS (EXACT_F64) |
| transpose bridge | `Dispatcher::transpose` == `barracuda::dispatch::transpose_dispatch` | PASS (EXACT_F64) |
| softmax bridge | `Dispatcher::softmax` == `barracuda::dispatch::softmax_dispatch` | PASS (EXACT_F64) |
| gelu bridge | `Dispatcher::gelu` == `barracuda::dispatch::gelu_dispatch` | PASS (EXACT_F64) |
| mean bridge | `Dispatcher::mean` == `barracuda::dispatch::mean_dispatch` | PASS (EXACT_F64) |
| variance bridge | `Dispatcher::variance` == `barracuda::dispatch::variance_dispatch` | PASS (EXACT_F64) |
| l2_distance bridge | `Dispatcher::l2_distance` == `barracuda::dispatch::l2_distance_dispatch` | PASS (EXACT_F64) |
| hmm_forward bridge | `Dispatcher::hmm_forward` == `barracuda::dispatch::hmm_forward_dispatch` | PASS (EXACT_F64) |
| frobenius_norm bridge | `Dispatcher::frobenius_norm` == `barracuda::dispatch::frobenius_norm_dispatch` | PASS (EXACT_F64) |
| threshold routing (small) | CPU path for data < 64 elements | PASS |
| threshold routing (large) | GPU path for data ≥ 4096 elements | PASS |
| threshold routing (medium) | Device-dependent routing for 64–4095 | PASS |
| determinism (repeated exec) | Repeated matmul yields bit-identical results | PASS |
| determinism (cross-function) | Softmax deterministic across repeated calls | PASS |

### ToadStool Absorption Implication

This bridge validator confirms that when ToadStool absorbs neuralSpring's dispatch patterns, the math is already proven identical to `barracuda::dispatch`. No adaptation layer needed — neuralSpring's `Dispatcher` is a direct consumer of the same API that `ComputeDispatch` wraps.

---

## Part 3: NUCLEUS PCIe Bypass + Mixed Pipeline (38/38)

New binary: `validate_nucleus_pcie_mixed_pipeline`

### 3.1 PCIe Bridge Cost Model (8 checks)

| Check | What It Validates | Result |
|-------|------------------|--------|
| PCIe bypass cost (1 KB) | `PcieBridge::transfer_cost` scaling | PASS |
| PCIe bypass cost (1 MB) | Sub-linear overhead at scale | PASS |
| PCIe bypass cost (1 GB) | Large transfer costing | PASS |
| Cost scaling (1 MB > 1 KB) | Monotonic cost increase | PASS |
| Multi-hop chain (GPU→NPU→CPU) | `chained_transfer_cost` > direct | PASS |
| Multi-hop overhead | Chain adds measurable overhead | PASS |
| Small transfer (64 B) | Minimum viable transfer | PASS |
| Large transfer (100 MB) | High-bandwidth tier | PASS |

### 3.2 NPU Routing Decisions (6 checks)

| Check | Workload Properties | Expected Route | Result |
|-------|-------------------|----------------|--------|
| Realtime + NPU available | `needs_realtime: true, npu_available: true` | GpuToNpu | PASS |
| Large compute, no NPU | `compute_us: 50_000, npu_available: false` | GpuOnly | PASS |
| Small compute, no NPU | `compute_us: 100, npu_available: false` | CpuOnly | PASS |
| Large + realtime, no NPU | `needs_realtime: true, npu_available: false` | GpuOnly (fallback) | PASS |
| Mixed substrate returned | All routes return valid `MixedSubstrate` | PASS |
| Routing determinism | Same workload → same route | PASS |

### 3.3 GPU→NPU Bypass Science Pipeline (6 checks)

End-to-end execution of science operations across mixed substrates:

| Check | Pipeline Stage | Result |
|-------|---------------|--------|
| GPU variance computation | `Dispatcher::variance` on GPU | PASS |
| CPU variance fallback | `cpu_fallback::variance` | PASS |
| GPU↔CPU variance parity | < `GPU_VARIANCE_F64` tolerance | PASS |
| GPU mean computation | `Dispatcher::mean` on GPU | PASS |
| CPU mean fallback | `cpu_fallback::mean` | PASS |
| GPU↔CPU mean parity | < `GPU_MEAN_DISPATCH_F32` tolerance | PASS |

### 3.4 NUCLEUS Tower→Node→Nest Chain (10 checks)

| Check | NUCLEUS Atomic | What It Proves | Result |
|-------|---------------|----------------|--------|
| Tower discovery | `inventory::discover_local()` | Substrate inventory works | PASS |
| Tower substrate count | ≥ 1 substrate found | At least CPU available | PASS |
| Tower has CPU | `SubstrateKind::Cpu` in inventory | CPU always available | PASS |
| Node eigensolve | `Dispatcher::eigensolve_symmetric` | GPU compute dispatch | PASS |
| Node eigenvalue count | Correct number returned | Dimension preserved | PASS |
| Node eigenvalue variance | Computed from GPU eigenvalues | Science metric valid | PASS |
| Nest provenance count | N results stored | All results tracked | PASS |
| Nest Shannon entropy | Provenance entropy > 0 | Non-trivial provenance | PASS |
| Nest provenance ordering | Results maintain insertion order | Causal chain preserved | PASS |
| Tower→Node→Nest chain | End-to-end completion | Full atomic pipeline works | PASS |

### 3.5 biomeOS Multi-Stage Graph Coordination (8 checks)

Spectral → Population → Information pipeline with dynamic routing:

| Check | Stage | What It Validates | Result |
|-------|-------|------------------|--------|
| Spectral eigensolve | Stage 1 | Graph Laplacian eigenvalues | PASS |
| Eigenvalue count | Stage 1 | Correct dimensions | PASS |
| Population variance (dispatch) | Stage 2 | GPU variance of eigenvalues | PASS |
| Population variance (CPU) | Stage 2 | CPU reference computation | PASS |
| Dispatch↔CPU variance parity | Stage 2 | Cross-substrate math identity | PASS |
| Information entropy (dispatch) | Stage 3 | Shannon entropy of eigenvalues | PASS |
| Information entropy (CPU) | Stage 3 | CPU reference computation | PASS |
| Dispatch↔CPU entropy parity | Stage 3 | Cross-substrate math identity | PASS |

---

## Part 4: What neuralSpring Contributes Back to ToadStool

### 4.1 New in S115

| Contribution | Impact |
|-------------|--------|
| Full dispatch parity proof (53 ops) | ToadStool can confidently expose all 53 ops via `ComputeDispatch` knowing CPU↔GPU parity is proven |
| `barracuda::dispatch` bridge proof (9 core ops) | Validates that the `dispatch` layer ToadStool wraps produces identical results to direct API calls |
| Threshold routing validation | Confirms ToadStool's routing heuristics (small→CPU, large→GPU) work correctly |
| PCIe cost model validation | metalForge's cost model for GPU↔NPU transfers is proven realistic and monotonic |
| NUCLEUS atomic chain proof | Tower→Node→Nest pipeline works end-to-end — ready for ToadStool daemon integration |
| biomeOS graph coordination | Multi-stage science pipelines can be dynamically routed across substrates |

### 4.2 Cumulative (V75 and earlier)

15+ shaders absorbed upstream, 6-spring provenance tracked, 68/68 cross-spring evolution benchmark, 14/14 cross-spring bench. See V75 for full history.

---

## Part 5: Rewire Opportunities (Updated from V75)

| Priority | Item | Recommendation |
|----------|------|---------------|
| **Done** | All 53 Dispatcher ops with GPU path | Parity proven — no gaps |
| **Done** | ComputeDispatch bridge (9 core ops) | Bit-identical to barracuda::dispatch |
| **Done** | NUCLEUS PCIe mixed pipeline | 38/38 checks, GPU→NPU→CPU validated |
| Low | `cpu_fallback::variance` | Could rewire to `barracuda::stats::variance_ddof(data, 0)` |
| Low | `jacobi_eigh` naming | Misleading name (uses Householder); already delegates to barracuda |
| Future | BatchedEncoder for SpectralNautilusBridge | 950ms bottleneck is ESN training — GPU BatchedEncoder path |
| Future | `barracuda::pde` for Richards | Soil moisture modeling if added |
| Future | `barracuda::multi_gpu` | Large tensor operations across devices |

---

## Part 6: Lessons Learned (S115 Additions)

1. **f32 GPU tolerance requires care**: `multi_obj_fitness` on GPU uses f32 WGSL; the CPU path uses f64. Tolerances must account for this (TENSOR_MATMUL_F32 at 1e-3, not TENSOR_EXACT_F32 at 1e-6). ToadStool should document which ops use f32 vs f64 shaders.

2. **Determinism is provable**: Repeated GPU dispatch of the same data produces bit-identical results. This is important for reproducible science — ToadStool can guarantee this.

3. **PCIe bypass cost scales sub-linearly**: Transfer overhead grows slower than data size, making large GPU→NPU transfers efficient. The cost model in metalForge accurately predicts this.

4. **NUCLEUS atomics compose cleanly**: Tower (discovery) → Node (compute) → Nest (provenance) is a natural pipeline. Each atomic is independently testable and composable.

5. **biomeOS graph routing preserves math**: Multi-stage pipelines routed across different substrates produce identical results to single-substrate execution. The routing layer is transparent to the science.

---

## Appendix: Validation State

| Metric | V75 (S113) | V76 (S115) |
|--------|-----------|-----------|
| ToadStool HEAD | `2fee1969` (S86) | `2dc26792` (S87) |
| validate_all | 208/208 | **212/212** |
| Dispatch parity | 30/30 | **53/53** |
| ComputeDispatch bridge | — | **14/14** |
| NUCLEUS PCIe bypass | — | **38/38** |
| lib tests | 861 | 861 |
| binaries | 226 | **232** |
| clippy warnings | 0 | 0 |
| Cross-spring provenance | 6 springs | 6 springs |
| Paper queue coverage | 25 papers full-stack | 25 papers full-stack |

---

## Action Items for ToadStool

- **toadStool action**: Consider exposing `ComputeDispatch` threshold routing parameters as configurable (neuralSpring validated the default 64/4096 thresholds work well)
- **toadStool action**: Document f32 vs f64 shader variants per op for downstream tolerance selection
- **toadStool action**: NUCLEUS Tower→Node→Nest atomic chain is ready for daemon-side integration (all APIs validated)
- **toadStool action**: PCIe cost model in metalForge is validated — can be used for intelligent dispatch routing in ToadStool's streaming pipeline
