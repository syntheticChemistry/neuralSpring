# metalForge — Cross-System Dispatch Strategy

**Parent**: ecoPrimals/neuralSpring/metalForge
**License**: AGPL-3.0-or-later
**Date**: February 20, 2026

---

## Vision

The final workload validation shows the same math running on PURE GPU,
then demonstrates mixed-system dispatch: GPU → CPU → NPU (AKD1000).
ToadStool's `UnidirectionalPipeline` reduces round-trips; BarraCUDA's
`DeviceCapabilities` routes each workload to the optimal substrate.

---

## Dispatch Tiers

| Tier | Substrate | When | Example |
|------|-----------|------|---------|
| **GPU-only** | RTX 4070 (Vulkan) | Large tensor ops, GEMM, FFT | MLP inference, HMM forward chain |
| **CPU-fallback** | llvmpipe (LLVM) | CI, small tensors, correctness | Unit tests, validation binaries |
| **GPU→CPU** | GPU dispatch + CPU readback | Iterative algorithms with scalar feedback | EA fitness → selection (CPU) → GPU mutation |
| **GPU→NPU** | GPU compute + NPU inference | Real-time decision (low-latency NPU) | ESN surrogate on AKD1000 for fast prediction |
| **Mixed** | All three | Full pipeline | GPU physics → NPU decision → CPU coordination |

---

## Current Validated Dispatch Paths

### GPU-only (Phase 3c — validated)

| Workload | Shader | Binary | Status |
|----------|--------|--------|--------|
| HMM forward chain | `hmm_forward_log.wgsl` | `validate_gpu_hmm_forward` | **PASS** |
| Batch fitness eval | `batch_fitness_eval.wgsl` | `validate_gpu_batch_fitness` | **PASS** |
| Parallel RK4 ODE | `rk4_parallel.wgsl` | `validate_gpu_rk4` | **PASS** |
| FFT (f32+f64+Rfft) | BarraCUDA `ops::fft` | `validate_barracuda_fft` | **24/24 PASS** |
| ML inference | evolved MHA + native ops (S-03b) | `validate_barracuda_ml_inference` | **PASS** |
| Pairwise Jaccard | `pairwise_jaccard.wgsl` | `validate_gpu_pangenome` | **6/6 PASS** |
| Locus variance | `locus_variance.wgsl` | `validate_gpu_meta_pop` | **7/7 PASS** |

### CPU-fallback (Phase 1–2 — validated)

| Workload | BarraCUDA Module | Binary | Status |
|----------|-----------------|--------|--------|
| All 24 paper modules | `stats`, `linalg`, `numerical`, `special`, `tensor` | `validate_barracuda_*` (24 binaries) | **203/203 PASS** |
| Tensor API (90 ops) | `tensor::Tensor` | `validate_barracuda_tensor` | **90/90 PASS** |
| f64 GPU ops | `ops::*_f64` | `validate_barracuda_tensor_f64` | **35/35 PASS** |

### GPU↔CPU Cross-Dispatch (Phase 3d+ — validated)

| Workload | Shader | Binary | Status |
|----------|--------|--------|--------|
| EA fitness routing | `batch_fitness_eval.wgsl` | `validate_cross_dispatch` | **8/8 PASS** |
| Genomics (Jaccard + variance) | `pairwise_jaccard.wgsl` + `locus_variance.wgsl` | `validate_cross_dispatch_genomics` | **8/8 PASS** |
| Extended (Papers 017, 019, 022–023) | Hamming, spatial_payoff, batch_ipr | `validate_cross_dispatch_extended` | **12/12 PASS** |
| Phase 4e (Papers 012, 014, 015, 021) | pairwise_l2, multi_obj_fitness, swarm_nn_forward, hill_gate | `validate_cross_dispatch_phase4e` | **13/13 PASS** |

### GPU→CPU (Phase 1d — validated)

| Workload | Pattern | Binary | Status |
|----------|---------|--------|--------|
| 3-way benchmark | GPU compute → CPU readback → compare | ~~`bench_scaling`~~ (fossilized) | **✓ GPU < CPU < Py at crossover** |
| Fused pipeline | GPU 9-18 passes → single readback | ~~`bench_fused_inference`~~ (fossilized) | **46–78× speedup** |

---

## Planned Cross-System Paths

### GPU→NPU (Phase 4)

Following hotSpring's metalForge NPU characterization:

| Workload | GPU Role | NPU Role | Path |
|----------|----------|----------|------|
| EA + surrogate | Batch fitness eval (GPU) | ESN decision (AKD1000) | GPU population → NPU predict → CPU select |
| Regulatory network | GPU RK4 integration | NPU steady-state predict | GPU dynamics → NPU fast inference |
| Real-time HMM | GPU forward chain | NPU emission predict | GPU transitions → NPU observation model |

### Requirements from hotSpring metalForge NPU findings

- AKD1000 supports ~15k neurons per inference
- Quantize-aware training required (MetaTF → QuantizeML → Akida)
- 10 SDK assumptions overturned (see hotSpring `metalForge/npu/akida/BEYOND_SDK.md`)
- Direct deploy possible for ESN models < 500 neurons

---

## Relationship to ToadStool

ToadStool's `UnidirectionalPipeline` is the key enabler for cross-system
dispatch.  Data streams in one direction: GPU-resident state updates,
scalar-only readback.  This reduces CPU-GPU round-trips from O(T) to O(1)
for iterative workloads.

| ToadStool Feature | neuralSpring Use | Status |
|-------------------|-----------------|--------|
| `TensorSession` ML ops | Fused MLP/Transformer inference | **Available** (S-01/S-11 absorbed) |
| `StatefulPipeline` | HMM chain, ODE loops | **Validated** (10/10 PASS) |
| `ReduceScalarPipeline` | Log-likelihood, convergence | Available (local mean_reduce validated) |
| `KernelRouter` | 4-tier matmul | **Absorbed** (S-02) — replaces local shaders |
| `DispatchConfig` | CPU/GPU routing | **Validated** (8+8+12+13 = 41 cross-dispatch PASS) |
| `DeviceCapabilities` | Per-substrate dispatch | Already used |

---

*Cross-system dispatch strategy — GPU, CPU, NPU via ToadStool/BarraCUDA.*
