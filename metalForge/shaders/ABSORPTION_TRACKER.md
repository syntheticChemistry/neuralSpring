# metalForge Shader Absorption Tracker

**Parent**: ecoPrimals/neuralSpring/metalForge
**License**: AGPL-3.0-or-later
**Pattern**: Evolve locally → validate → handoff → ToadStool absorbs → retire
**ToadStool HEAD**: `1dd7e338` (Sessions 50–70+++, S70+++: cross-spring absorption, DF64 ML shaders, SimpleMlp, matmul_ref, architecture safety, Feb 26, 2026)

---

## Active Shaders (still local — no upstream equivalent or upstream differs)

| Shader | Domain | Status | Validation Binary | Absorption Target |
|--------|--------|--------|-------------------|-------------------|
| `head_split.wgsl` | MHA (attention) | **Validated** | `validate_mha_gpu` | `barracuda::ops::mha` (fix S-03b) |
| `head_concat.wgsl` | MHA (attention) | **Validated** | `validate_mha_gpu` | `barracuda::ops::mha` (fix S-03b) |

**Note**: `head_split`/`head_concat` have upstream equivalents at `barracuda::shaders::tensor/`
but use different param structs (`HeadSplitParams` vs local `Params`). Upstream MHA projection
shaders still hang on RTX 4070 — these remain the only truly local shaders.

## Recently Absorbed (ToadStool Sessions 50–53, `9abd6857`)

| Shader | Upstream API | Absorption Session | Provenance Tag |
|--------|-------------|-------------------|----------------|
| `xoshiro128ss.wgsl` | `barracuda::ops::prng_xoshiro` | S51 (H-004) | `PROV_RK45_ADAPTIVE` |
| `logsumexp_reduce.wgsl` | `barracuda::ops::LogsumexpWgsl` | S51 (H-004) | — |
| `stencil_cooperation.wgsl` | `barracuda::StencilCooperationGpu` | S52 | — |
| `wright_fisher_step.wgsl` | `barracuda::WrightFisherGpu` | S52 | — |
| `rk45_adaptive.wgsl` | `barracuda::ops::rk45_adaptive` | S51 | `PROV_RK45_ADAPTIVE` |
| `swarm_nn_scores.wgsl` | `barracuda::SwarmNnGpu` (scores path) | S52 (L-009) | `PROV_SWARM_NN` |

Local copies retained for raw WGSL validation binaries (our validators depend on local binding layouts).

## Write-Phase Extensions (Session 64)

| Shader | Domain | Status | Origin |
|--------|--------|--------|--------|
| `chi_squared_f64.wgsl` | ML validation | **Validated** (forge tests) | neuralSpring S-64 |
| `kl_divergence_f64.wgsl` | ML validation | **Validated** (forge tests) | neuralSpring S-64 |

## Planned Shaders

| Shader | Domain | Priority | Dependency |
|--------|--------|----------|------------|
| `tridiag_eigensolver.wgsl` | Spectral (022–023) | P3 | Needs Householder → bisection design |

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

ToadStool S68 evolved all shaders to f64 canonical with runtime downcast
via `LazyLock<String>`. Several public `const &str` constants became private.
Local copies now used for validation where upstream constants are inaccessible.

| Shader | Upstream API | S68 Status |
|--------|-------------|------------|
| `hmm_forward_log.wgsl` | `barracuda::ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32` | Still `pub const` |
| `batch_fitness_eval.wgsl` | `barracuda::ops::bio::batch_fitness::WGSL_BATCH_FITNESS_EVAL` | Still `pub const` |
| `rk4_parallel.wgsl` | Upstream f64: `rk4_parallel_f64.wgsl` (requires polyfill) | Local f32 copy |
| `pairwise_jaccard.wgsl` | `WGSL_PAIRWISE_JACCARD` now private `LazyLock` | Local copy |
| `pairwise_hamming.wgsl` | `WGSL_PAIRWISE_HAMMING` now private `LazyLock` | Local copy |
| `locus_variance.wgsl` | `WGSL_LOCUS_VARIANCE_F64` (new f64 pub const) | Re-export f64 |
| `spatial_payoff.wgsl` | `WGSL_SPATIAL_PAYOFF` now private `LazyLock` | Local copy |
| `batch_ipr.wgsl` | `WGSL_BATCH_IPR` now `pub static LazyLock<String>` | Local copy |

### Still Active (Thin Wrapper)

| Module | LOC | Issue | Status |
|--------|-----|-------|--------|
| `mha.rs` | 182 | S-03b resolved upstream (`0c998992`) | Thin wrapper delegating to `barracuda::ops::mha` |

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

`xoshiro128ss.wgsl` and `swarm_nn_scores.wgsl` **ABSORBED** by ToadStool S51/S52.

**Previously absorbed** (Session 39): `pairwise_l2`, `multi_obj_fitness`, `hill_gate`,
`swarm_nn_forward`, `mean_reduce` — upstream has generalized variants. Local copies retained
for validation compatibility; future migration to upstream binding layouts.

---

*Shader evolution tracker — following the hotSpring metalForge pattern.*
