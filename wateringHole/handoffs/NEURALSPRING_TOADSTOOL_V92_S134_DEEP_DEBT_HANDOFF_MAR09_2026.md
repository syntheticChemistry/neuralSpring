<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/BarraCUDA V92 Handoff

| Field | Value |
|-------|-------|
| **Date** | 2026-03-09 |
| **From** | neuralSpring S134 |
| **To** | ToadStool, BarraCUDA |
| **License** | AGPL-3.0-or-later |
| **Supersedes** | V91 (S133) |
| **Pins** | ToadStool S130+, BarraCUDA v0.3.3, coralReef Iteration 10 |

## Executive Summary

Session 134 deep debt resolution — activation consolidation, tolerance promotion, and
comprehensive doc alignment. All 966 tests pass, 91.66% line coverage, zero clippy.

### Key Metrics

| Metric | Value |
|--------|-------|
| Lib tests | 966 |
| Forge tests | 71 |
| Integration tests | 9 |
| validate_all | 220/220 PASS |
| Binaries | 246 |
| Line coverage | 91.66% |
| Named tolerances | 150+ |
| Upstream rewires | 46 |
| Clippy pedantic+nursery | 0 warnings |

## Part 1: Activation Consolidation

Seven duplicate activation functions (sigmoid, gelu, gelu_f32, relu, relu_f32, softmax,
relu_vec, relu_inplace) consolidated into `primitives.rs` as the single canonical CPU
reference. All call sites in library code and validation binaries now delegate to
`primitives::*`.

**Affected modules:**
- `transformer.rs` — softmax, gelu delegate to primitives
- `coral_forge/activation.rs` — gelu, gelu_vec delegate to primitives
- `lenet.rs` — relu delegates to primitives::relu_vec
- `coral_forge/structure/backbone.rs` — relu_inplace delegates to primitives
- `coral_forge/pairformer.rs` — sigmoid delegates to primitives
- `coral_forge/confidence.rs` — sigmoid delegates to primitives
- `neural_pgm.rs` — weight_to_transition uses primitives::softmax per row

**ToadStool action:** When promoting activation shaders, the canonical CPU reference for
numerical comparison is always `neural_spring::primitives::*`. GPU shader results should
match within `tolerances::GPU_F32` or `tolerances::GPU_F64_RELAXED` as appropriate.

## Part 2: Tolerance Promotion

Sixteen inline numeric tolerance literals in validation binaries replaced with 5 new
named constants in `tolerances/mod.rs`:

| Constant | Value | Domain |
|----------|-------|--------|
| `GLUCOSE_CGM_STAT_TOL` | 0.5 | CGM glucose prediction |
| `GLUCOSE_TAU_TOL` | 1e-6 | Glucose time constant |
| `PLDDT_DEGENERACY_THRESHOLD` | 0.01 | AlphaFold3 pLDDT |
| `GPU_KIMURA_BATCH_DIFF` | 0.02 | GPU Kimura population genetics |
| `TENSOR_RELU_DETERMINISM_F32` | 0.0 | ReLU f32 exact determinism |

All registered in `tolerances/registry.rs` under `domain_validation` category.
Total named tolerances: 150+.

**ToadStool action:** All tolerance values are available via
`neural_spring::tolerances::registry::all_tolerances()` for runtime introspection.
Use these when comparing GPU shader output against CPU reference.

## Part 3: Code Quality

- Fixed 6 clippy pedantic/nursery errors (cast_precision_loss, cast_possible_truncation, doc backticks, const fn promotion)
- Replaced `unwrap()` in tests with `expect()` (descriptive messages)
- Replaced `panic!()` with `assert!(matches!(...))` for idiomatic testing
- `validate_gpu_shader_phase4.rs`: standardized no-GPU exit via `validation::exit_no_gpu()`
- `coralreef_bridge.rs`: replaced hardcoded primal namespaces with `BIOMEOS_NAMESPACES` env

## Part 4: Provenance

Added provenance triplets (Python script, commit, date, exact command) to 5 validation
binary docblocks: `validate_barracuda_glucose_prediction`, `validate_barracuda_hmm_f64`,
`validate_barracuda_wdm_sqw`, `validate_gpu_hmm_forward`, `validate_cross_dispatch_hmm`.

## Part 5: Evolution Gaps for ToadStool/BarraCUDA

### Absorption-ready shaders (still local to metalForge)
- `xoshiro128ss.wgsl` — GPU PRNG
- `swarm_nn_scores.wgsl` — swarm neural network scoring
- `logsumexp_reduce.wgsl` — log-sum-exp reduction
- `stencil_cooperation.wgsl` — spatial stencil cooperation
- `rk45_adaptive.wgsl` — adaptive Runge-Kutta
- `wright_fisher_step.wgsl` — Wright-Fisher population genetics

### GPU training stack gaps
- Autograd / backpropagation — not yet in BarraCUDA
- `nn::Layer` trait — no GPU-side layer abstraction
- `nn::Optimizer` (SGD, Adam) — no GPU optimizer

### CPU activation exposure
- BarraCUDA has private scalar activations in `dispatch/domain_ops.rs` (sigmoid, relu, gelu)
- **BarraCUDA action:** Consider exposing these as `barracuda::activations::*` public API
  to allow springs to use upstream activations directly

### Benchmark gaps
- No Python → BarraCUDA CPU timing benchmarks exist
- No Kokkos / Galaxy GPU parity benchmark document found
- Recommend creating `bench/python_parity/` and `bench/gpu_kokkos/` for formal comparison

*This handoff is unidirectional: neuralSpring → ecosystem. No response expected.*
