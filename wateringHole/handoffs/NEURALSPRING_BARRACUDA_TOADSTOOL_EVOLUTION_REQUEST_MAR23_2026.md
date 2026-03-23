# NEURALSPRING → BARRACUDA/TOADSTOOL — Evolution & Absorption Request

**Date:** 2026-03-23  
**neuralSpring:** V122 (S172), 1,380 tests, 0 clippy/fmt/doc  
**barraCuda:** v0.3.7  
**License:** AGPL-3.0-or-later  

## Context

neuralSpring is the ML validation spring for the ecoPrimals ecosystem. It
validates barraCuda's primitives against 27 scholarly reproductions + 5 novel
compositions spanning evolutionary computation, phylogenetics, game theory,
spectral analysis, population genetics, regulatory biology, biomedical time-series,
and cross-domain reservoir computing. This handoff summarizes what neuralSpring
has evolved locally that is ready for upstream absorption, and what gaps remain.

## Completed Migrations

### DeviceCapabilities (V122)

neuralSpring completed migration from deprecated `GpuDriverProfile` to
`DeviceCapabilities` across 11 files. neuralSpring was the **last spring**
to finish this migration. No deprecated imports remain.

**Migration map for other springs still migrating:**
- `GpuDriverProfile::from_device(dev)` → `DeviceCapabilities::from_device(dev)`
- `.fp64_strategy()` → same method name on `DeviceCapabilities`
- `.precision_routing()` → same method name
- `.needs_pow_f64_workaround()` → `.needs_exp_f64_workaround()` (per-builtin granularity)
- `.f64_zeros_risk()` → `.has_reliable_f64()` (inverted semantics)
- `.check_allocation_safe()` → same method name on `DeviceCapabilities`

### normalize_method (V122)

Absorbed the `normalize_method` IPC pattern from `barracuda-core::ipc::methods`.
neuralSpring's primal binary now normalizes incoming JSON-RPC method names,
stripping legacy `neuralspring.` prefix. All springs should adopt this pattern.

## WGSL Shaders Ready for Upstream Absorption (41 total)

### Tier A — Direct absorption candidates (well-validated, stable API)

These shaders have extensive validation coverage and stable interfaces:

| Shader | Domain | Validation Coverage |
|--------|--------|-------------------|
| `attention_apply_f64.wgsl` | coralForge SDPA | validate_coral_forge_gpu_pipeline |
| `backbone_update_f64.wgsl` | coralForge structure | validate_coral_forge_gpu_pipeline |
| `chi_squared_f64.wgsl` | Statistics | validate_barracuda_dispatch_parity |
| `gelu_f64.wgsl` | Activation | validate_coral_forge_gpu + benches |
| `ipa_scores_f64.wgsl` | coralForge IPA | validate_coral_forge_gpu_pipeline |
| `kl_divergence_f64.wgsl` | Statistics | validate_barracuda_dispatch_parity |
| `layer_norm_f64.wgsl` | Normalization | validate_coral_forge_gpu + benches |
| `linear_regression.wgsl` | Statistics | validate_barracuda_dispatch_parity |
| `msa_col_attention_scores_f64.wgsl` | coralForge MSA | validate_coral_forge_gpu |
| `msa_row_attention_scores_f64.wgsl` | coralForge MSA | validate_coral_forge_gpu |
| `outer_product_mean_f64.wgsl` | coralForge Evoformer | validate_coral_forge_gpu |
| `sdpa_scores_f64.wgsl` | coralForge SDPA | validate_coral_forge_gpu_pipeline |
| `sigmoid_f64.wgsl` | Activation | validate_coral_forge_gpu |
| `softmax_f64.wgsl` | Activation | validate_coral_forge_gpu + benches |
| `torsion_angles_f64.wgsl` | coralForge structure | validate_coral_forge_gpu_pipeline |
| `triangle_attention_f64.wgsl` | coralForge Evoformer | validate_coral_forge_gpu |
| `triangle_mul_incoming_f64.wgsl` | coralForge Evoformer | validate_coral_forge_gpu |
| `triangle_mul_outgoing_f64.wgsl` | coralForge Evoformer | validate_coral_forge_gpu |

### Tier B — Adapt/merge candidates (overlap with upstream)

These have upstream equivalents but neuralSpring has f64 or domain-specific variants:

| Shader | Domain | Upstream Equivalent |
|--------|--------|-------------------|
| `batch_fitness_eval.wgsl` | Evolution | `barracuda::ops::bio::batch_fitness` (re-exported) |
| `batch_ipr.wgsl` | Spectral | `barracuda::spectral::BatchIprGpu` |
| `hmm_backward_log.wgsl` | Phylogenetics | `barracuda::ops::bio::hmm` (partial) |
| `hmm_viterbi.wgsl` | Phylogenetics | `barracuda::ops::bio::hmm` (partial) |
| `locus_variance.wgsl` | Population genetics | `barracuda::ops::bio::locus_variance` (re-exported) |
| `mean_reduce.wgsl` | Reduction | `barracuda::ops::WGSL_MEAN_REDUCE` (layout mismatch) |
| `logsumexp_reduce.wgsl` | Reduction | `barracuda::ops::logsumexp` |
| `matrix_correlation.wgsl` | Statistics | Needs new upstream home |

### Tier C — New shader needed (no upstream equivalent)

| Shader | Domain | Notes |
|--------|--------|-------|
| `head_concat.wgsl` | MHA | Multi-head attention merge |
| `head_split.wgsl` | MHA | Multi-head attention split |
| `hill_gate.wgsl` | Regulatory biology | Two-input Hill AND gate |
| `multi_obj_fitness.wgsl` | Evolution | Multi-objective fitness stats |
| `pairwise_hamming.wgsl` | Genomics | Distance matrix |
| `pairwise_jaccard.wgsl` | Genomics | Distance matrix |
| `pairwise_l2.wgsl` | Genomics | Distance matrix |
| `rk4_parallel.wgsl` | ODE | Parallel RK4 step (f32 local copy; f64 upstream) |
| `rk45_adaptive.wgsl` | ODE | Adaptive step-size |
| `spatial_payoff.wgsl` | Game theory | Payoff grid |
| `stencil_cooperation.wgsl` | Game theory | Fermi imitation |
| `swarm_nn_forward.wgsl` | Neural | Swarm NN MLP forward |
| `swarm_nn_scores.wgsl` | Neural | Max-activation scores |
| `wright_fisher_step.wgsl` | Population genetics | W-F simulation |
| `xoshiro128ss.wgsl` | PRNG | GPU random (used by several) |

## Upstream Requests (Priority Order)

### P0 — Blocking

1. **`GpuDriverProfile` removal timeline**: neuralSpring has completed migration.
   When will barraCuda remove the deprecated type? All springs should be migrated
   by now (airSpring v0.10, wetSpring V132, groundSpring V120, neuralSpring V122).

### P1 — High Value

2. **`barracuda::cast` module**: Safe-cast helpers duplicated across springs
   (groundSpring V120, airSpring v0.10 both request). Centralize `safe_cast<T>()`
   with checked overflow.

3. **`MultiHeadEsn` device accessor**: neuralSpring works around missing
   `wgpu_device()` on `MultiHeadEsn` by keeping a separate `Arc<WgpuDevice>`.
   Adding the accessor unblocks cleaner `TensorSession` wiring.

4. **Stable shader catalog API**: neuralSpring has 41 WGSL shaders in
   `metalForge/shaders/` — many mirror upstream but with different binding
   layouts or `LazyLock` patterns. A stable name + content_sha256 catalog
   would enable automated promotion tracking.

### P2 — Evolution

5. **HMM backward + Viterbi absorption**: neuralSpring has validated f64
   HMM backward (`hmm_backward_log.wgsl`) and Viterbi (`hmm_viterbi.wgsl`)
   shaders. Upstream `barracuda::ops::bio::hmm` has forward but not backward/Viterbi.

6. **coralForge shader set**: 10 structure prediction shaders (SDPA, IPA,
   backbone update, torsion angles, triangle attention/multiply, MSA attention,
   outer product mean, layer norm) — all validated against OpenFold Python.
   Consider absorption into `barracuda::ops::structural` or similar.

7. **Pairwise distance GPU ops**: `pairwise_hamming`, `pairwise_jaccard`,
   `pairwise_l2` — validated with Python baselines. Could join
   `barracuda::ops::pairwise_distance`.

### P3 — Nice to Have

8. **`ValidationSink` ecosystem pattern**: wetSpring V132 introduced pluggable
   `ValidationSink` for structured validation output. Consider ecosystem-wide
   absorption so springs can emit JSON/CSV validation reports.

9. **`mixed_dispatch` absorption**: neuralSpring's `Dispatcher::mixed_dispatch()`
   routes workloads across GPU/CPU/NPU using `metalForge` cost model. Candidate
   for `toadStool` → `barracuda::unified_hardware` absorption.

## TensorSession / StatefulPipeline Usage

neuralSpring actively uses both:

- `Dispatcher::tensor_session()` → fused multi-op GPU pipelines (matmul+GELU+softmax+layer_norm)
- `Dispatcher::stateful_pipeline()` → iterative GPU kernels (ODE integrators, eigensolvers)
- playGround transformer inference uses `TensorSession` for GPT-2 forward
- `validate_gpu_stateful_pipeline` exercises `StatefulPipeline` + `KernelDispatch`

## Quality State

| Metric | Value |
|--------|-------|
| Tests | 1,380 (0 failures) |
| Clippy | 0 warnings (pedantic + nursery, workspace-wide) |
| Fmt | 0 diff |
| Doc | 0 warnings |
| Unsafe | `#![forbid(unsafe_code)]` on all 3 crates |
| `#[allow]` | Zero in production |
| Max file LOC | 879 |
| WGSL shaders | 41 |
| Named tolerances | 227+ |
| Python baselines | 397 PASS |
