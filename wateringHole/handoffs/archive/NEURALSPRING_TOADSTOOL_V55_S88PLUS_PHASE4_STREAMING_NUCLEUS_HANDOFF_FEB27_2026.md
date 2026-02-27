# neuralSpring → ToadStool/BarraCUDA Handoff: V55 Phase 4 WGSL + Streaming Pipeline + NUCLEUS Atomics

**Date**: February 27, 2026
**From**: neuralSpring (Session 88+)
**To**: ToadStool/BarraCUDA team
**ToadStool pin**: `e96576ee` (S68 reviewed)
**neuralSpring**: 172 binaries, 668 lib + 43 forge tests, 171/172 validate\_all
**Supersedes**: V54 (barracuda evolution audit, now in archive/)
**License**: AGPL-3.0-or-later

---

## Executive Summary

neuralSpring's validation stack is complete across all tiers. This handoff
documents the final Phase 4 WGSL shader validation, ToadStool streaming proof,
and comprehensive barracuda evolution audit — everything ToadStool needs to
absorb neuralSpring's spectral science pipeline.

**New since V54:**
- `validate_gpu_shader_phase4`: **22/22 PASS** — direct WGSL dispatch for
  HMM backward/Viterbi, matrix correlation, linear regression
- `validate_streaming_spectral_pipeline`: **28/28 PASS** — ToadStool
  unidirectional streaming proof with Anderson disorder sweep
- Documentation sweep — all counts aligned (2970+ checks, 172 binaries)

**From V53–V54 (included for absorption context):**
- NUCLEUS compute dispatch: **39/39 PASS** (Tower→Node→Nest)
- ToadStool absorption readiness: **294/294 PASS** (CPU+GPU+batch+mixed)
- Publication mixed-hardware: **43/43 PASS** (PCIe bridge, substrate routing)
- Barracuda evolution audit: 90+ imports, 39 rewires, zero duplicate math

---

## Part 1: How neuralSpring Uses BarraCUDA

### Usage Inventory (90+ import sites, 60+ files)

| Category | Primitives | Files |
|----------|-----------|-------|
| **Stats** | variance, pearson, ESD, marchenko\_pastur, r\_squared, rmse, nash\_sutcliffe, dot, L2, shannon | 15+ |
| **Linalg** | eigh\_f64, solve\_f64, cholesky, LU, SVD, effective\_rank, tridiag\_solve | 12+ |
| **Numerical** | rk45\_solve | 5+ |
| **Special** | chi\_squared\_sf/cdf, gamma, erf, bessel\_\*, legendre, hermite | 8+ |
| **Tensor** | matmul, transpose, sigmoid, tanh, add, mul, conv2d, maxpool2d | 20+ |
| **GPU dispatch** | Dispatcher (44 CPU→GPU ops), BatchIprGpu, HmmBatchForwardF64 | 30+ |
| **Shaders** | 42 WGSL (41 local + 3 re-exports), 15 df64 sovereign folding | 42 |
| **Device** | WgpuDevice, GpuCapabilities, GpuDriverProfile | 10+ |

### Validation Tiers (2970+ checks)

| Tier | Coverage | Checks |
|------|----------|--------|
| Python (Py) | 25/25 papers + 5 WDM + 3 pub exp | 263 |
| Rust CPU (Rs) | 25/25 + baseCamp + WDM + pub exp | 668 lib + 9 integration |
| BarraCUDA CPU (bC) | 24/25 papers (96%) | 203 |
| BarraCUDA GPU (gT) | 23/25 papers (92%) | 98+ |
| metalForge WGSL (mF) | 15/25 papers + Phase 4 shaders | 130 |
| GPU Pipeline (gP) | 15/25 papers | 94 |
| Cross-dispatch (xD) | 15/15 Phase 0++ papers | 49 |
| Mixed hardware (mH) | baseCamp + pub experiments | 100+ |
| Multi-GPU | RTX 4070 + TITAN V NVK | 384 bit-identical |
| Streaming | Batch eigensolve→IPR→stats | 28 |

---

## Part 2: What to Absorb

### Priority 1: Phase 4 WGSL Shaders (validated, ready)

| Shader | Algorithm | Absorption Target | Precision |
|--------|-----------|-------------------|-----------|
| `hmm_backward_log.wgsl` | HMM backward (log logsumexp) | `barracuda::ops::bio::hmm_backward` | 1.19e-7 |
| `hmm_viterbi.wgsl` | Viterbi decoding (log argmax) | `barracuda::ops::bio::hmm_viterbi` | exact (0.0) |
| `matrix_correlation.wgsl` | Pearson (N×N upper triangle) | `barracuda::stats::matrix_correlation_gpu` | <1e-6 |
| `linear_regression.wgsl` | OLS normal equations | `barracuda::stats::linear_regression_gpu` | slope diff <0.003 |

All four validated via direct `dispatch_shader` in `validate_gpu_shader_phase4` (22/22).

### Priority 2: Streaming Pipeline Pattern

The `validate_streaming_spectral_pipeline` proves ToadStool's unidirectional
streaming pattern preserves scientific conclusions:

```
Hamiltonian assembly (CPU)
  ↓ batch upload
eigensolve (GPU via eigh_gpu)
  ↓ eigenvectors stay GPU-side
BatchIprGpu dispatch (GPU, scalar readback)
  ↓ IPR scalars
Dispatcher::variance / ::mean (GPU)
  ↓ scalar readback
Anderson diagnostic (CPU threshold check)
```

Key results:
- 8 Hamiltonians × eigensolve → IPR: max diff < 1e-6
- Anderson disorder sweep (W=0.5→16): IPR 0.09→0.79, clear transition
- Eigenvalue parity (sorted): 1.6e-14 (machine ε)

### Priority 3: Sovereign Folding df64 Shaders (15 shaders, validated)

Still pending absorption from V52. All 15 WGSL shaders use df64 core streaming:
- Arithmetic tier: 3.6e-8 to 5.6e-7 (tol 1e-6)
- Transcendental tier: 1.7e-4 to 3.4e-4 (tol 5e-4)
- `compile_shader_f64_hybrid` entry point for absorption

### Not Yet Used (Low Priority)

| Function | Why Relevant | Blocker |
|----------|-------------|---------|
| `ops::logsumexp` | HMM backward (currently inlined in WGSL) | None |
| `ops::pairwise_distance` | Anderson interaction graphs | None |
| `ops::batched_eigh_gpu` | Batch eigensolve for disorder sweep | NAK tridiag (S-12) |

---

## Part 3: Evolution Learnings for ToadStool

### What Worked Well

1. **Capability-based dispatch**: `Dispatcher` routes 44 ops transparently.
   GPU when available, CPU fallback otherwise. ≤1.04× overhead for 9/10 ops.
2. **df64 core streaming**: ~9.9× throughput vs native f64 on consumer GPUs.
   Two-zone tolerance (arithmetic / transcendental) is principled and robust.
3. **Named tolerances**: 129+ constants in centralized registry. Zero inline
   magic numbers. Every tolerance traces to a physical or numerical origin.
4. **Deterministic seeds**: All experiments use seed=42. Python and Rust
   produce identical results at 1e-10 cross-language parity.

### What to Evolve

1. **Full GPU command encoder chaining**: Currently eigensolve→IPR uses
   separate dispatches. ToadStool's `StatefulPipeline` can fuse these.
2. **Batch eigensolve**: `batched_eigh_gpu` would eliminate the per-matrix
   host loop in `validate_streaming_spectral_pipeline`.
3. **Ring buffer readback**: GPU-resident ring buffers (hotSpring pattern)
   would reduce latency for streaming disorder sweeps.
4. **Transcendental precision**: df64 transcendentals use degree-6 Horner
   polynomials. Degree-10+ would close the gap to arithmetic tier.

### Cross-Spring Alignment

| Spring | Version | Key Contribution |
|--------|---------|-----------------|
| wetSpring | V61 | 79 barracuda primitives, nanopore field genomics |
| hotSpring | V0614 | df64 strategy, NVK patterns, 22 papers |
| neuralSpring | V55 | 42 WGSL shaders, spectral science, streaming proof |

All three Springs on ToadStool `e96576ee`. Unified patterns:
- df64 core streaming (hotSpring origin, neuralSpring adopted)
- Capability-based dispatch (neuralSpring `Dispatcher` pattern)
- Named tolerance registry (neuralSpring origin, cross-spring adopted)
- Python→Rust→GPU validation chain (shared pattern)

---

## Part 4: Full Validation Matrix

| Validator | Checks | What It Proves |
|-----------|--------|----------------|
| `validate_gpu_shader_phase4` | 22/22 | Phase 4 WGSL direct dispatch |
| `validate_streaming_spectral_pipeline` | 28/28 | ToadStool streaming pattern |
| `validate_publication_mixed_hardware` | 43/43 | metalForge mixed-hardware tier |
| `validate_nucleus_compute_dispatch` | 39/39 | NUCLEUS Tower→Node→Nest |
| `validate_toadstool_spectral_absorption` | 294/294 | Full absorption readiness |
| `validate_publication_gpu_pipeline` | 13/13 | BatchIprGpu + cross-system |
| `validate_barracuda_training_trajectory` | 9/9 | Exp-050 GPU |
| `validate_barracuda_hessian_eigen` | 10/10 | Exp-052 GPU |
| `validate_barracuda_anderson_multiagent` | 11/11 | Exp-053 GPU |
| `validate_cpu_math_parity` | 39/39 | Rust = Python/NumPy |
| `validate_gpu_pure_workload_all` | 10/10 | Pure GPU all 15 domains |
| `validate_cross_system_dispatch` | 46/46 | GPU→NPU→CPU |
| **Full suite** | **171/172** | 1 pre-existing WDM damping assertion |

---

*End of V55 handoff. Previous handoffs archived in `wateringHole/handoffs/archive/`.*
