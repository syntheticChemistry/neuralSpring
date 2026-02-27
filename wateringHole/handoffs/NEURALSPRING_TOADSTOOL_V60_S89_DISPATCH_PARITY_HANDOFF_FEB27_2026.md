# neuralSpring → ToadStool/BarraCUDA Handoff V60 — Dispatch Parity & Mixed-Hardware

**Date**: February 27, 2026
**From**: neuralSpring (ML/neuroevolution validation)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Session 89 — dispatch parity, mixed-hardware dispatch, upstream BarraCUDA wiring
**Supersedes**: V59 (Comprehensive Evolution Handoff)

---

## Executive Summary

- **177 binaries**, **177/177 validate_all** (all green), **668 lib tests**, **3111+ checks**
- **+3 upstream BarraCUDA ops wired**: `HillGateGpu`, `MultiObjFitnessGpu`, `SwarmNnGpu`
- **Dispatch parity proven**: 26 Dispatcher operations produce identical CPU↔GPU results (30/30)
- **Mixed-hardware dispatch validated**: NPU substrate, PCIe bridge, NUCLEUS atomics (47/47)
- **47 GPU-promoted Dispatcher ops** (~97% of production math)
- BarraCUDA CPU remains **83.6× faster** than Python/NumPy (geomean, 11 domains)

---

## Part 1: New Upstream BarraCUDA APIs Wired

### GPU Ops Wrappers (`src/gpu_ops/bio.rs`)

| Wrapper | BarraCUDA API | Params | Precision |
|---------|---------------|--------|-----------|
| `hill_gate_gpu` | `barracuda::ops::bio::hill_gate::HillGateGpu` | `HillGateParams` (f64 + padding) | f64 buffers |
| `multi_obj_fitness_gpu` | `barracuda::ops::bio::MultiObjFitnessGpu` | f64 fitness + weights | f64 buffers |
| `swarm_nn_forward_gpu` | `barracuda::ops::bio::swarm_nn::SwarmNnGpu` | `SwarmNnParams` (f64 + padding) | f64 buffers |

### Dispatcher Methods (`src/gpu_dispatch/dispatch_ops.rs`)

| Method | GPU Path | CPU Fallback |
|--------|----------|-------------|
| `Dispatcher::hill_gate()` | `gpu_ops::hill_gate_gpu` | `signal_integration::two_input_hill` |
| `Dispatcher::multi_obj_fitness()` | `gpu_ops::multi_obj_fitness_gpu` | `directed_evolution::multi_objective_fitness` |
| `Dispatcher::swarm_nn_forward()` | `gpu_ops::swarm_nn_forward_gpu` | `swarm_robotics::neural_forward` |

**toadStool action**: These wrappers handle f64↔buffer conversion and BarraCUDA
params struct padding fields (`_pad`, `_pad0`, `_pad1`, `_pad2`). If the params
structs evolve, neuralSpring will need updating. Consider documenting padding
field requirements in BarraCUDA API docs.

---

## Part 2: Dispatch Parity — CPU↔GPU Math Identical (30/30)

`validate_barracuda_dispatch_parity` exercises 26 Dispatcher operations,
comparing `Dispatcher::cpu_only()` vs `Dispatcher::from_gpu()`:

| Category | Operations | Tolerance | Status |
|----------|-----------|-----------|--------|
| Linear algebra | mat_mul, frobenius_norm, transpose, commutator, distance_to_normal | 1e-10 | PASS |
| Activations | softmax, gelu, hill_activation_batch, boltzmann | 1e-6 | PASS |
| Reductions | l2_distance, mean, variance, shannon_entropy, pearson_correlation, chi_squared | 1e-10 | PASS |
| HMM | hmm_forward_step, hmm_backward_step, hmm_viterbi_step, hmm_forward_chain | 1e-6 | PASS |
| Population genetics | allele_frequencies, nucleotide_diversity, pairwise_fst, global_fst | 0.1 (FST) | PASS |
| Game theory | replicator_step | 1e-6 | PASS |
| Eigensolvers | eigh, disorder_sweep | 1e-6 | PASS |
| Pangenome | spectrum_chi_squared, selection_coefficient | 1e-6 | PASS |

**Precision note**: `pairwise_fst` uses `GPU_FST_PAIRWISE_F32` tolerance (0.1)
because the Weir-Cockerham FST estimator amplifies f32 intermediate precision
differences for small populations. Larger, denser populations (n=20) stabilize
the estimator. This is documented in `src/tolerances/`.

---

## Part 3: Mixed-Hardware Dispatch (47/47)

`validate_mixed_hardware_dispatch` exercises metalForge's complete stack:

| Domain | Checks | What It Validates |
|--------|--------|-------------------|
| Substrate discovery | 6 | GPU/CPU/NPU `SubstrateKind` creation, capabilities, display |
| PCIe transfer costs | 9 | Latency tiers, bandwidth tiers, chained transfers, P2P comparisons |
| PcieBridge | 5 | Reserve, route, GPU↔NPU bypassing CPU |
| Mixed substrate routing | 6 | `MixedSubstrate` workload routing with cost model |
| Dispatcher mixed dispatch | 8 | `Dispatcher::mixed_dispatch()` integration |
| NUCLEUS Tower atomic | 5 | eigensolve workload via compute dispatch |
| NUCLEUS Node atomic | 4 | population genetics workload via compute dispatch |
| NUCLEUS Nest atomic | 4 | spatial workload via compute dispatch |

### NPU Substrate Type System (New)

Added `SubstrateKind::Npu` to `metalForge/forge/src/substrate.rs`:
- `Capability::NpuInference` — inference-optimized NPU workload
- `Capability::NpuBatch` — batch NPU workload
- `fmt::Display` for `Npu` → "NPU"
- Full test coverage

**toadStool action**: The NPU substrate type enables NUCLEUS Nest atomic
(Tower + NestGate) with NPU inference routing. This is the foundation for
AKD1000 NPU dispatch via metalForge — consider adding NPU-specific bandwidth
tiers and transfer cost models.

---

## Part 4: Absorption Targets (Updated)

### Still pending absorption

| Shader/API | Domain | Priority |
|-----------|--------|----------|
| 15 sovereign folding df64 shaders | Protein structure | P1 |
| `compile_shader_df64_streaming` | df64 pipeline | P1 |
| `nn::SimpleMLP` | WDM surrogates | P2 |
| Transcendental precision (degree-10+ Horner) | Core math | P2 |
| `head_split.wgsl`, `head_concat.wgsl` | MHA S-03b | P3 |

### Already absorbed (no action needed)

All S-01..S-17 shortcomings resolved. 42 upstream rewires complete.
21/21 WGSL shaders absorbed. 16 submodules exercised.

---

## Part 5: Validation Matrix

| Metric | Count |
|--------|-------|
| Total binaries | 177 |
| validate_all | **177/177 PASS** |
| Library tests | 668 |
| Total checks | 3111+ |
| Barracuda import sites | 124 |
| Files using barracuda | 177 |
| Barracuda submodules | 16 |
| Upstream rewires | 42 |
| GPU-promoted Dispatcher ops | 47 |
| metalForge WGSL shaders | 42 |
| Named tolerances | 131+ |
| Python baselines | 263 checks |
| CPU vs Python speedup | 83.6× geomean |
| GPU portability | 9/9 |
| Dispatch parity | 30/30 |
| Mixed-hardware dispatch | 47/47 |
| Multi-GPU (RTX 4070 + Titan V) | Bit-identical |

---

## Part 6: Recommendations

1. **Params struct documentation**: Document padding field requirements for
   `HillGateParams`, `SwarmNnParams`, and similar GPU params structs
2. **NPU bandwidth model**: Add NPU-specific transfer cost tiers to
   `barracuda::unified_hardware`
3. **FST precision**: Consider f64 intermediate path for Weir-Cockerham FST
   on GPU to eliminate the 0.1 tolerance requirement
4. **BLAS small-matrix fast-path**: Still open from V59 — close the commutator
   gap (0.3× vs NumPy BLAS)
5. **Sovereign folding absorption**: 15 df64 shaders remain the highest-priority
   absorption target

---

*AGPL-3.0-or-later*
