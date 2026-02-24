# neuralSpring V16 — Session 48 Handoff

**Date**: February 23, 2026
**From**: neuralSpring → ToadStool / BarraCUDA
**Session**: 48
**ToadStool HEAD**: `b41ee5f4` (Session 47 + S45/S46/S49 absorption)
**Previous**: V15 (Session 47 — typed op migration, 10 validators)
**License**: AGPL-3.0-or-later

---

## Executive Summary

Session 48 completed the **mass typed-op rewiring**: 28 validation/benchmark binaries
converted from raw wgpu dispatch (include_str! local shaders + manual pipeline/
bindgroup/encoder creation) to modern BarraCUDA typed op APIs. This removes thousands
of lines of boilerplate and validates the upstream ToadStool/BarraCUDA APIs directly.

**Key numbers**: 28 binaries rewired. f32→f64 data type alignment for 6 ops. HillGateGpu
f64 graceful skip on RTX 4070. Validation: 132/133 (only pre-existing logsumexp
driver issue).

---

## Part 1: Mass Typed Op Rewiring — 28 Binaries

| Category | Count | Examples |
|----------|-------|----------|
| Standalone validators | 2 | wright_fisher → WrightFisherGpu (f64), stencil → StencilCooperationGpu (f64) |
| Pipeline validators | 15 | All use typed BarraCUDA ops + CPU mean instead of raw wgpu shader chains |
| Cross-dispatch validators | 6 | All use typed BarraCUDA ops |
| Benchmarks | 1 | bench_gpu_kernels.rs — 5 benchmarks now use typed ops |

---

## Part 2: f32→f64 Data Type Alignment

These ops moved from f32 to f64 (upstream ToadStool S49 sync):

- BatchFitnessGpu
- LocusVarianceGpu
- MultiObjFitnessGpu
- WrightFisherGpu
- StencilCooperationGpu
- SwarmNnGpu

---

## Part 3: HillGateGpu f64 Graceful Skip

On RTX 4070, HillGateGpu f64 triggers a driver limitation. Validators skip the f64
path gracefully; f32 path remains validated.

---

## Part 4: Validation Score — 132/133

Only pre-existing `validate_barracuda_logsumexp` driver issue remains. All other
validators PASS.

---

## Part 5: Cross-Spring Benchmark Results (RTX 4070, Vulkan)

### bench_cross_spring_evolution

| Op | Origin | Time (ms) |
|----|--------|-----------|
| BatchFitnessGpu | neuralSpring | 44 |
| PairwiseL2Gpu | neuralSpring | 8.7 |
| BatchIprGpu | neuralSpring | 7 |
| SpatialPayoffGpu | neuralSpring | 5.3 |
| PairwiseHammingGpu | neuralSpring | 5.2 |
| HmmBatchForwardF64 | wetSpring | 7.2 |
| BatchedEighGpu | hotSpring | 17.5 |

### bench_gpu_kernels (typed ops)

| Kernel | GPU | Rust CPU | GPU Advantage |
|--------|-----|----------|---------------|
| Large Hamming | 5.6 ms | 8.2 ms | 1.4× |
| Large Jaccard | 5.6 ms | 13.3 ms | 2.4× |
| Large Fitness (50000×64) | 6 ms | — | — |

---

## Part 6: Remaining Raw wgpu (3 Binaries + 1 Intentional)

| Binary | Reason |
|--------|--------|
| `bench_upstream_vs_local` | **Intentional** — compares local vs upstream dispatch |
| `validate_gpu_pipeline_swarm` | No upstream equivalent for scores variant |
| `validate_gpu_pipeline_regulatory` | ODE structure mismatch with upstream |
| `validate_cross_dispatch_ode` | Same ODE structure mismatch |

---

## Related Handoffs

- **V15** (Session 47): Typed op migration (10 validators) — archived
- **V14** (Sessions 45–46): Pure GPU promotion, 38 ops — archived
- `whitePaper/CROSS_SPRING_SHADER_LINEAGE.md`: Session 48 section
- `specs/BARRACUDA_USAGE.md`: Session 48 — Mass Typed Op Rewiring
- `specs/PURE_GPU_ROADMAP.md`: Raw wgpu coverage note
- `whitePaper/BARRACUDA_EVOLUTION.md`: Session 48 section
- `EVOLUTION_READINESS.md`: Session 48 rows
