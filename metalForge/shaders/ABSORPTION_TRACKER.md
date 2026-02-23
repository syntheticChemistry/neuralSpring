# metalForge Shader Absorption Tracker

**Parent**: ecoPrimals/neuralSpring/metalForge
**License**: AGPL-3.0-or-later
**Pattern**: Evolve locally → validate → handoff → ToadStool absorbs → retire
**ToadStool HEAD**: `5437c170` (Session 42+43, Feb 22, 2026)

---

## Active Shaders (still local — no upstream equivalent or upstream differs significantly)

| Shader | Domain | Status | Validation Binary | Absorption Target |
|--------|--------|--------|-------------------|-------------------|
| `head_split.wgsl` | MHA (attention) | **Validated** | `validate_mha_gpu` | `barracuda::ops::mha` (fix S-03b) |
| `head_concat.wgsl` | MHA (attention) | **Validated** | `validate_mha_gpu` | `barracuda::ops::mha` (fix S-03b) |
| `xoshiro128ss.wgsl` | Stochastic (PRNG) | **Validated** | `validate_gpu_prng` | `barracuda::ops::prng` |
| `swarm_nn_scores.wgsl` | Swarm (015) | **Validated** | `validate_gpu_pipeline_swarm` | No upstream equivalent |
| `logsumexp_reduce.wgsl` | HMM/phylo (016–018) | **Validated** | `validate_gpu_logsumexp` | `barracuda::ops::reduce` (batched logsumexp) |
| `stencil_cooperation.wgsl` | Game theory (019) | **Validated** | `validate_gpu_stencil` | `barracuda::ops::stencil` (Fermi imitation) |
| `rk45_adaptive.wgsl` | Regulatory ODE (020–021) | **Validated** | `validate_gpu_rk45` | `barracuda::ops::ode` (injectable RHS) |
| `wright_fisher_step.wgsl` | PopGen (024–025) | **Validated** | `validate_gpu_wright_fisher` | `barracuda::ops::popgen` (drift+selection) |

**Note**: `head_split`/`head_concat` have upstream equivalents at `barracuda::shaders::tensor/`
but use different param structs (`HeadSplitParams` vs local `Params`). `xoshiro128ss` differs
from `barracuda::shaders::misc::prng_xoshiro` in state model (persistent vs one-shot).

## Planned Shaders

| Shader | Domain | Priority | Dependency |
|--------|--------|----------|------------|
| `tridiag_eigensolver.wgsl` | Spectral (022–023) | P3 | Needs Householder → bisection design |
| `logsumexp_reduce.wgsl` | HMM/phylogenetics | P2 | Complements `hmm_forward_log.wgsl` |

## Retired (Absorbed by ToadStool `5437c170`)

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

### Shaders Absorbed (Session 42, `5437c170` — generalized variants)

ToadStool absorbed 5 neuralSpring shaders as generalized upstream variants.
Local copies remain for validation (our validators depend on the local binding layouts).

| Shader | Upstream Path | Semantic Differences |
|--------|---------------|---------------------|
| `pairwise_l2.wgsl` | `barracuda::shaders::math::pairwise_l2` | Closed-form pair decoding (upstream) vs linear search (local). Different struct names (`PairwiseParams` vs `Params`) |
| `multi_obj_fitness.wgsl` | `barracuda::shaders::bio::multi_obj_fitness` | Bessel correction `n-1` (upstream) vs population `n` (local). Different param names (`pop`/`n_obj` vs `pop_size`/`n_objectives`) |
| `hill_gate.wgsl` | `barracuda::shaders::bio::hill_gate` | Mode 0/1 generalization (upstream) vs 2D-grid only (local). `HillGateParams` vs `HillParams` |
| `swarm_nn_forward.wgsl` | `barracuda::shaders::bio::swarm_nn_forward` | Generic MLP with `SwarmParams{input_dim,hidden_dim,output_dim}` (upstream) vs fixed 1→4→5 (local). Clamped sigmoid |
| `mean_reduce.wgsl` | `barracuda::shaders::reduce::mean_reduce` | Effectively identical (upstream credits neuralSpring as origin) |

### Shaders Absorbed (Pre–Session 39, `77f70b2e` — identical copies)

| Shader | Upstream API |
|--------|-------------|
| `hmm_forward_log.wgsl` | `barracuda::ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32` |
| `batch_fitness_eval.wgsl` | `barracuda::ops::bio::batch_fitness::WGSL_BATCH_FITNESS_EVAL` |
| `rk4_parallel.wgsl` | `barracuda::ops::rk_stage::WGSL_RK4_PARALLEL` |
| `pairwise_jaccard.wgsl` | `barracuda::ops::bio::pairwise_jaccard::WGSL_PAIRWISE_JACCARD` |
| `pairwise_hamming.wgsl` | `barracuda::ops::bio::pairwise_hamming::WGSL_PAIRWISE_HAMMING` |
| `locus_variance.wgsl` | `barracuda::ops::bio::locus_variance::WGSL_LOCUS_VARIANCE` |
| `spatial_payoff.wgsl` | `barracuda::ops::bio::spatial_payoff::WGSL_SPATIAL_PAYOFF` |
| `batch_ipr.wgsl` | `barracuda::spectral::batch_ipr::WGSL_BATCH_IPR` |

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
| `ops::nn::conv2d.wgsl` | Batched Conv2D (LeNet-5 layers) | Study 003 | **New** (Session 39) — not yet wired to executor |
| `ops::nn::maxpool2d.wgsl` | MaxPool2D (LeNet-5 pooling) | Study 003 | **New** (Session 39) — not yet wired to executor |
| `ops::nn::avgpool2d.wgsl` | AvgPool2D (alternative pooling) | — | **New** (Session 39) — not yet wired to executor |
| `cpu_conv_pool` | CPU Conv2D/MaxPool2D/AvgPool2D | Study 003 | **New** (Session 39) — CPU fallback |
| `esn_v2::export_weights/import_weights` | GPU-train → NPU-deploy ESN | — | **New** (Session 39) |

---

## Absorption Readiness (February 21, 2026)

### GPU-Ready Library Modules

Library modules now use flat row-major layouts that match shader binding
layouts, reducing the conversion barrier for ToadStool absorption:

| Module | Layout | Shader Buffer Match |
|--------|--------|---------------------|
| `hmm.rs` | Flat `Vec<f64>` (T×N) | `hmm_forward_log.wgsl` `@binding(2) prev_alpha: array<f32>` |
| `spectral_commutativity.rs` | Flat `Vec<f64>` (N×N) | `barracuda::ops::matmul` flat buffer input |
| `primitives.rs` | Centralized math constants | Shader `const` declarations (e.g. `LOG_GUARD`) |
| `anderson_localization.rs` | Flat `Vec<f64>` (N×N) | `batch_ipr.wgsl` `@binding(0) hamiltonians: array<f32>` |
| `directed_evolution.rs` | Flat `Vec<f64>` (pop×genome) | `multi_obj_fitness.wgsl` `@binding(0) genotypes: array<f32>` |
| `sate_alignment.rs` | Flat `Vec<u8>` (n×len) | `pairwise_hamming.wgsl` `@binding(0) sequences: array<u32>` |
| `pinn.rs` | Scalar + grid ops | `barracuda::tensor` matmul+tanh buffer input |
| `deeponet.rs` | Scalar + polynomial | `barracuda::tensor` matmul buffer input |

### Validation Robustness

All validation binaries use `require!` macro for GPU operations, enabling
graceful degradation when specific adapters or features are unavailable.
This supports the cross-backend validation pattern required by ToadStool
(GPU → CPU → NPU parity testing).

### Next Absorption Targets

Following the hotSpring lifecycle (evolve → validate → handoff → absorb → retire):

| Shader | Status | Next Step |
|--------|--------|-----------|
| `head_split.wgsl` / `head_concat.wgsl` | Validated (10/10). Upstream variant exists with different params | Unify param structs; handoff to `barracuda::ops::mha` |
| `xoshiro128ss.wgsl` | Validated (5/5). Upstream has different state model | Reconcile persistent-state vs one-shot; handoff to `barracuda::ops::prng` |
| `swarm_nn_scores.wgsl` | Validated (pipeline PASS). No upstream equivalent | New handoff to `barracuda::ops::bio` |

**Previously absorbed** (Session 39): `pairwise_l2`, `multi_obj_fitness`, `hill_gate`,
`swarm_nn_forward`, `mean_reduce` — upstream has generalized variants. Local copies retained
for validation compatibility; future migration to upstream binding layouts.

---

*Shader evolution tracker — following the hotSpring metalForge pattern.*
