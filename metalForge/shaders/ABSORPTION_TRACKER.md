# metalForge Shader Absorption Tracker

**Parent**: ecoPrimals/neuralSpring/metalForge
**License**: AGPL-3.0-or-later
**Pattern**: Evolve locally → validate → handoff → ToadStool absorbs → retire
**ToadStool HEAD**: `dc540afd` (Session 25, Feb 20, 2026)

---

## Active Shaders

| Shader | Domain | Status | Validation Binary | Absorption Target |
|--------|--------|--------|-------------------|-------------------|
| `hmm_forward_log.wgsl` | Phylogenetics (016–018) | **Validated** | `validate_gpu_hmm_forward` | `barracuda::ops::hmm` or `StatefulPipeline` |
| `batch_fitness_eval.wgsl` | Evolution (011–015) | **Validated** | `validate_gpu_batch_fitness` | `barracuda::ops::batch_gemm` |
| `rk4_parallel.wgsl` | Regulatory/Signal (020–021) | **Validated** | `validate_gpu_rk4` | `barracuda::ops::ode` |
| `mean_reduce.wgsl` | Fitness aggregation | **Validated** | `validate_gpu_pure_workload` | `barracuda::pipeline::ReduceScalarPipeline` |
| `pairwise_jaccard.wgsl` | Pangenome (024) | **Validated** | `validate_gpu_pangenome` | `barracuda::ops::pairwise_distance` |
| `locus_variance.wgsl` | Meta-population (025) | **Validated** | `validate_gpu_meta_pop` | `barracuda::ops::VarianceReduceF64` |
| `spatial_payoff.wgsl` | Game theory (019) | **Validated** | `validate_gpu_game_theory` | `barracuda::ops::stencil` |
| `batch_ipr.wgsl` | Spectral (022–023) | **Validated** | `validate_gpu_anderson` | `barracuda::ops::batch_reduce` |
| `pairwise_hamming.wgsl` | Alignment (017) | **Validated** | `validate_gpu_sate` | `barracuda::ops::pairwise_distance` |
| `xoshiro128ss.wgsl` | Stochastic (PRNG) | **Validated** | `validate_gpu_prng` | `barracuda::ops::prng` |
| `head_split.wgsl` | MHA (attention) | **Validated** | `validate_mha_gpu` | `barracuda` or `TensorSession` head ops |
| `head_concat.wgsl` | MHA (attention) | **Validated** | `validate_mha_gpu` | `barracuda` or `TensorSession` head ops |

## Planned Shaders

| Shader | Domain | Priority | Dependency |
|--------|--------|----------|------------|
| `tridiag_eigensolver.wgsl` | Spectral (022–023) | P3 | Needs Householder → bisection design |
| `logsumexp_reduce.wgsl` | HMM/phylogenetics | P2 | Complements `hmm_forward_log.wgsl` |

## Retired (Absorbed by ToadStool `dc540afd`)

### Evolved Modules (S-01 through S-11 — all absorbed)

| Module | LOC | ToadStool Fix | Absorbed In | Replacement API |
|--------|-----|---------------|-------------|-----------------|
| `fused_pipeline.rs` | 680 | S-01: `TensorSession` single-encoder | `fbedd222` | `TensorSession::run()` |
| `fused_mlp.rs` | 356 | S-01/S-11: ML ops in session | `fbedd222` | `TensorSession::{matmul, relu, gelu}` |
| `fused_transformer.rs` | 725 | S-01/S-11: ML ops in session | `fbedd222` | `TensorSession::{head_split, attention, layer_norm}` |
| `matmul_cpu_tiled.wgsl` | 270 | S-02: 4-tier `KernelRouter` | `82f953c8` | `ops::matmul` CpuTiled32 |
| `matmul_gpu_evolved.wgsl` | 306 | S-02: 4-tier `KernelRouter` | `82f953c8` | `ops::matmul` GpuEvolved32 |
| `layer_norm.rs` | 268 | S-08: `from_pooled_buffer` | `81a6fd4b` | `Tensor::layer_norm_wgsl()` |
| `log_softmax.rs` | 259 | S-09: `from_pooled_buffer` | `81a6fd4b` | `Tensor::log_softmax_wgsl()` |

**Status**: All fossilized in `metalForge/fossils/evolved_s01_s11/`.
Code removed from active compilation Feb 20, 2026.

### Still Active (Not Yet Absorbed)

| Module | LOC | Issue | Status |
|--------|-----|-------|--------|
| `mha.rs` | 182 | S-03b: native projection shaders hang | Active in `src/evolved/` |

---

## Validation Workflow

Each shader follows this lifecycle:

1. **Evolve**: Write WGSL in `metalForge/shaders/`, targeting a specific paper's workload
2. **Orchestrate**: Create Rust dispatch code in `src/evolved/` using `wgpu::Buffer` + single encoder
3. **Validate**: Write `src/bin/validate_gpu_*.rs` with `ValidationHarness` against Python controls
4. **Benchmark**: Add to `metalForge/gpu/nvidia/` dispatch characterization
5. **Handoff**: Document in `wateringHole/handoffs/NEURALSPRING_*.md` for ToadStool
6. **Retire**: When ToadStool absorbs, update this tracker and remove local code

---

## BarraCUDA Capabilities Available for Leverage

| Module | Potential Use | Papers | Status |
|--------|--------------|--------|--------|
| `staging::StatefulPipeline` | Iterative RK4, HMM chains | 016–021 | **Validated** (10/10 PASS) |
| `pipeline::ReduceScalarPipeline` | Fitness aggregation | 011–015 | Available (local `mean_reduce` validated) |
| `TensorSession` ML ops | Fused MLP/Transformer inference | All ML | **Validated** by ToadStool (S-01/S-11) |
| `ops::logsumexp` | HMM log-domain numerics | 016–018 | **Validated** (5/5 PASS) |
| `KernelRouter` 4-tier matmul | Replace local shaders | All GEMM | **Absorbed** (S-02) |
| `NAK` eigensolver | Anderson localization GPU | 022–023 | Available |
| `Fft1DF64` | f64 FFT | — | New (Session 25) |
| `GemmF64::WGSL` | f64 GEMM shader source | — | New (wetSpring v4) |

---

*Shader evolution tracker — following the hotSpring metalForge pattern.*
