# neuralSpring V15 — Session 47 Handoff

**Date**: February 23, 2026
**From**: neuralSpring → ToadStool / BarraCUDA
**Session**: 47
**ToadStool HEAD**: `b41ee5f4` (Session 47 + S45/S46/S49 absorption)
**Previous**: V14 (Sessions 45–46 — pure GPU promotion, gpu_ops + gpu_dispatch)
**License**: AGPL-3.0-or-later

---

## Executive Summary

Session 47 completed the **typed op migration** and absorbed upstream ToadStool
fixes. 10 validation binaries rewired from raw wgpu dispatch to typed BarraCUDA
ops. MHA S-03b (projection shader z-dimension dispatch) **FIXED** upstream.
`evolved/hmm_forward_gpu.rs` retired — HmmBatchForwardF64 (wetSpring) is now the
primary HMM path. Eigensolvers and pangenome ops promoted to GPU. ~95% production
math on GPU.

**Key numbers**: ToadStool `b41ee5f4`. 10 validators typed. 4 new gpu_ops functions.
HMM fossil (351 LOC). MHA S-03b FIXED. bench_cross_spring_evolution added.

---

## Part 1: Upstream Absorption

### ToadStool Commits Absorbed

| Commit | Session | Key Changes |
|--------|---------|-------------|
| `c8076a2d` | S45 | Deep debt evolution (typed errors, shader fixes) |
| `fe573095` | S46 | Cross-project absorption (lattice QCD, MD transport, bio ODE, **MHA S-03b fix**) |
| `9bd71391` | S49 | Shader-first architecture (645+ shaders, zero CPU-only production math) |

### MHA S-03b Fix

The native `Tensor::multi_head_attention` projection shader z-dimension dispatch
bug was **FIXED upstream** in ToadStool S46. The fix flows to neuralSpring via
path dependency. `evolved::mha` remains active until full native MHA validation
is completed.

---

## Part 2: Typed Op Migration — 10 Validators Rewired

Migrated from raw wgpu dispatch to typed BarraCUDA ops:

| Validator | Typed Op | Domain |
|-----------|----------|--------|
| `validate_gpu_batch_fitness` | BatchFitnessGpu | Batch fitness |
| `validate_gpu_sate` | PairwiseHammingGpu | SATé alignment |
| `validate_gpu_pangenome` | PairwiseJaccardGpu | Pangenome |
| `validate_gpu_meta_pop` | LocusVarianceGpu | Meta-population |
| `validate_gpu_game_theory` | SpatialPayoffGpu | Game theory |
| `validate_gpu_directed` | MultiObjFitnessGpu | Directed evolution |
| `validate_gpu_modes` | PairwiseL2Gpu | MODES novelty |
| `validate_gpu_anderson` | BatchIprGpu | Anderson localization |
| `validate_gpu_swarm` | SwarmNnGpu | Swarm robotics |
| `validate_gpu_signal` | HillGateGpu | Signal integration |

Cross-spring absorption is **complete** — all 10 domain validators use typed ops.

---

## Part 3: HMM Forward Retirement

- **Retired**: `evolved/hmm_forward_gpu.rs` (351 lines)
- **Fossil**: `metalForge/fossils/evolved_hmm_forward_gpu/`
- **Primary path**: `HmmBatchForwardF64` (wetSpring origin) — f64, batch, BarraCUDA

All HMM callers now use upstream wetSpring path.

---

## Part 4: New GPU Promotions (gpu_ops + gpu_dispatch)

| Function | Purpose | Op |
|----------|---------|-----|
| `eigh_gpu` | BatchedEighGpu (single-dispatch for n≤32) | hotSpring |
| `disorder_sweep_gpu` | Batch eigensolve + mean IPR | hotSpring |
| `spectrum_chi_squared_gpu` | Pangenome chi-squared | neuralSpring |
| `selection_coefficient_gpu` | Pangenome selection coefficient | neuralSpring |

---

## Part 5: API Migration Absorbed

| API | Change |
|-----|--------|
| `solve_f64`, `cholesky_f64`, `gen_eigh_f64` | Now take `Arc<WgpuDevice>` (GPU-first) |
| `HillGateParams` | Removed `_pad3`/`_pad4` (f64 alignment) |

---

## Part 6: bench_cross_spring_evolution

New benchmark demonstrating the full cross-spring cycle:

- **neuralSpring**: BatchFitnessGpu, PairwiseL2Gpu, BatchIprGpu, SpatialPayoffGpu, PairwiseHammingGpu
- **wetSpring**: HmmBatchForwardF64
- **hotSpring**: BatchedEighGpu

Binary: `src/bin/bench_cross_spring_evolution.rs`

---

## Part 7: Pure GPU Roadmap Update

| Category | Before S47 | After S47 |
|----------|------------|-----------|
| Eigensolvers | 0/2 modules | **2/2** (eigh_gpu, disorder_sweep_gpu) |
| Chi-squared (pangenome) | Pending | **spectrum_chi_squared_gpu** |
| Selection coefficient | Pending | **selection_coefficient_gpu** |
| Production math on GPU | ~90% | **~95%** |

---

## Related Handoffs

- **V14** (Sessions 45–46): Pure GPU promotion, 38 ops — archived
- **V13** (Session 44): Multi-GPU portability, benchmarks, 2 upstream bug fixes
- `whitePaper/CROSS_SPRING_SHADER_LINEAGE.md`: Session 47 section
- `specs/BARRACUDA_USAGE.md`: Session 47 — Typed Op Migration
- `specs/PURE_GPU_ROADMAP.md`: Coverage gap updated
