<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/barraCuda Handoff V81 — SimpleMlp Rewire + HMM f64 ComputeDispatch + Cross-Spring Benchmark

**Date**: March 4, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/barraCuda team
**License**: AGPL-3.0-or-later
**Covers**: Session 121 — `SimpleMlp` rewire of WDM surrogates, HMM Viterbi f64 `ComputeDispatch` rewire, cross-spring modern benchmark with 5-spring provenance
**Supersedes**: V80 (S120 deep debt audit + CI hardening)
**barraCuda**: v0.3.1 standalone (`../barraCuda/crates/barracuda`)

---

## Executive Summary

- **WDM surrogates rewired to `barracuda::nn::SimpleMlp`**: `wdm_surrogate.rs` (EOS 2→128→128→2) and `wdm_transport.rs` (Transport 3→64→64→3) eliminated their local `MlpLayer` struct (~300 LOC). MLP forward passes now delegate to `SimpleMlp::forward()`. Domain-specific normalization and output transforms (signed-log, log-power) preserved in wrapper logic.
- **HMM Viterbi chain rewired to f64 `ComputeDispatch`**: `gpu_ops/bio/hmm.rs::hmm_viterbi_chain_gpu` replaced per-step f32 `Tensor` loop with single-dispatch f64 `barracuda::ops::bio::hmm_viterbi` (`hmm_viterbi_f64.wgsl`). Linear→log domain conversion handled at call site. Precision and efficiency both improved.
- **Upstream rewires**: 44 → **46** total (SimpleMlp + hmm_viterbi).
- **New validation binary**: `validate_barracuda_s121_rewire` — **80/80 PASS** (SimpleMlp layer counts, I/O sizes, prediction finiteness, determinism, JSON roundtrip, HMM Viterbi/forward CPU parity).
- **New benchmark binary**: `bench_cross_spring_modern` — **28/28 PASS** (SimpleMlp, HMM, stats, linalg, Dispatcher evolved ops; 5-spring provenance documented per section).
- **Quality gates**: fmt ✓ · clippy ✓ (0 warnings, pedantic+nursery, all-features) · test ✓ (869/869 lib) · doc ✓ · 213/213 validate\_all.

---

## Part 1: WDM Surrogate Rewire — `SimpleMlp`

### What Changed

| File | Before | After |
|------|--------|-------|
| `src/wdm_surrogate.rs` | Local `MlpLayer { weights, biases }` + manual matmul loop | `barracuda::nn::SimpleMlp` with `DenseLayer` (Relu hidden, Identity output) |
| `src/wdm_transport.rs` | Same local `MlpLayer` pattern | Same `SimpleMlp` delegation |
| `src/bin/validate_barracuda_wdm_eos.rs` | `surr.layers` iteration | `surr.mlp.layers` iteration with `DenseLayer` weight extraction |
| `src/bin/validate_barracuda_wdm_transport.rs` | Same | Same |
| `src/bin/validate_wdm_eos.rs` | `!surrogate.layers.is_empty()` | `!surrogate.mlp.layers.is_empty()` |
| `src/bin/validate_wdm_transport.rs` | Same | Same |

### Design Decision: Domain Logic Preserved

`SimpleMlp::forward()` handles pure matrix math (affine + activation). Domain-specific pre/post-processing remains in neuralSpring:

- **EOS**: log10(ρ)+guard → normalize → MLP → denormalize → signed-log output transform
- **Transport**: (log_rho, log_t, z_star) → normalize → MLP → denormalize → 10^x power transform

This separation keeps `SimpleMlp` universal (barraCuda's responsibility) while keeping domain semantics in the Spring (neuralSpring's responsibility).

### JSON Weight Loading

Python baselines store weights as flat row-major arrays. `DenseLayer` expects `Vec<Vec<f64>>` (rows × cols). The `load_surrogate_from_json` / `load_transport_from_json` functions perform the reshape, setting `Activation::Relu` for hidden layers and `Activation::Identity` for output.

---

## Part 2: HMM Viterbi f64 ComputeDispatch Rewire

### What Changed

| Before | After |
|--------|-------|
| Per-step f32 `Tensor` loop: create tensors → GPU matmul → readback → argmax, for each timestep | Single `barracuda::ops::bio::hmm_viterbi(device, &log_trans, &log_emit, &log_init, t_steps, n_states)` |
| f32 precision (Tensor API) | f64 precision (ComputeDispatch, `hmm_viterbi_f64.wgsl`) |
| CPU round-trips per timestep | Zero CPU round-trips (full Viterbi on GPU) |

### Input Domain Conversion

`hmm_viterbi_chain_gpu` receives probabilities in linear domain (consistent with neuralSpring's CPU HMM API). The upstream shader expects log-domain. Conversion is done at the call site:

- `log_trans[i*S+j] = transition[i*S+j].max(LOG_GUARD).ln()`
- `log_init[j] = initial[j].max(LOG_GUARD).ln()`
- `log_emit[t*S+j] = emission[j*n_obs + obs[t]].max(LOG_GUARD).ln()` (pre-extracted per timestep)

### Output Mapping

`ViterbiResult { path: Vec<u32>, delta: Vec<f64>, psi: Vec<u32> }` → `(Vec<usize>, f64)` where the score is `max(delta[last_step])`.

---

## Part 3: Cross-Spring Evolution — Provenance Map

The `bench_cross_spring_modern` binary documents where each capability originated and how it evolved through the ecosystem:

| Capability | Origin Spring | Evolution Path | barraCuda Module |
|------------|--------------|----------------|-----------------|
| SimpleMlp (MLP inference) | neuralSpring (WDM surrogates) | neuralSpring local → barraCuda `nn::SimpleMlp` | `barracuda::nn` |
| HMM Viterbi f64 | neuralSpring (phylogenetics) → wetSpring (bio) | per-step Tensor → fused `ComputeDispatch` | `barracuda::ops::bio` |
| R², Pearson, RMSE | neuralSpring (metrics) | neuralSpring local → barraCuda `stats` | `barracuda::stats` |
| Shannon entropy | wetSpring (diversity) | wetSpring → barraCuda `stats::shannon` | `barracuda::stats` |
| `eigh_f64` (eigendecomp) | hotSpring (spectral) | Householder+QR, S-12 resolved | `barracuda::linalg` |
| Softmax dispatch | neuralSpring (transformer) | local → `barracuda::dispatch::softmax_dispatch` | `barracuda::dispatch` |
| GELU dispatch | neuralSpring (transformer) | local → `barracuda::dispatch::gelu_dispatch` | `barracuda::dispatch` |
| MatMul dispatch | multi-spring | 4-tier router: naive → tiled → gpu-evolved | `barracuda::dispatch` |

---

## Part 4: Absorption Recommendations for ToadStool/barraCuda

### High Priority — Absorb from neuralSpring

| Item | What | Why | Effort |
|------|------|-----|--------|
| `hmm_forward_chain` patterns | neuralSpring's `hmm_forward_chain_gpu` still uses per-step Tensor loop | Create `barracuda::ops::bio::hmm_forward` ComputeDispatch analogous to `hmm_viterbi` | Medium |
| `hmm_backward_step` patterns | Similar per-step pattern | Create `barracuda::ops::bio::hmm_backward` ComputeDispatch | Medium |
| `SimpleMlp` GPU inference | neuralSpring GPU validation binaries manually unpack `DenseLayer` into `Tensor` ops | `SimpleMlp::forward_gpu(device)` that runs on GPU natively | Medium |
| `integrate_ode_batch_gpu` | Uses local `neural_spring_forge::shaders::RK4_PARALLEL` | Could use `barracuda::ops::rk_stage::WGSL_RK4_PARALLEL` if exposed | Low |

### Medium Priority — Cross-Spring Intelligence

| Observation | Source | Recommendation |
|-------------|--------|---------------|
| `DenseLayer` JSON serde | neuralSpring had to manually reshape flat row-major to `Vec<Vec<f64>>` | Consider adding `SimpleMlp::from_flat_json()` or a serde-compatible format for Python baseline interop |
| `log_emit` pre-extraction | `hmm_viterbi` shader expects `[T*S]` log emissions, not `[S*Obs]` emission matrix | Document this layout requirement clearly in barraCuda API docs — neuralSpring had to read the WGSL to figure it out |
| f64 HMM precision | f64 Viterbi produces strictly better results than f32 per-step Tensor approach | Consider deprecating f32 HMM paths in favor of f64 ComputeDispatch |

### Low Priority — Future Evolution

| Item | Status | Notes |
|------|--------|-------|
| 4 local WGSL shaders (`xoshiro128ss`, `swarm_nn_scores`, `head_split`, `head_concat`) | Active in metalForge | Absorption candidates when barraCuda adds GPU PRNG module and native 3D attention ops |
| Population genetics GPU ops (`allele_frequencies_gpu`, `nucleotide_diversity_gpu`) | Tensor-based wrappers | Could compose from upstream `fst_variance_decomposition` + `LocusVarianceGpu` |

---

## Part 5: What neuralSpring Learned — Useful for All Springs

1. **SimpleMlp interop**: The `Activation` enum (`Relu`, `Tanh`, `Sigmoid`, `Gelu`, `Identity`) covers all neuralSpring use cases. No custom activations needed. This is a good API surface.

2. **ComputeDispatch precision**: The f64 `hmm_viterbi` ComputeDispatch produces better results than the f32 Tensor per-step approach. Springs should prefer `ComputeDispatch` over `Tensor` loops when a full-pass shader exists.

3. **log_emit layout**: The `hmm_viterbi_f64.wgsl` shader expects pre-extracted per-timestep log emissions (`[T*S]`), not the full emission matrix (`[S*Obs]`). This is efficient but non-obvious. Document clearly.

4. **Cross-spring benchmark pattern**: `bench_cross_spring_modern` demonstrates how to document provenance (origin spring + evolution path) inline with benchmarks. Other springs should adopt this for their own cross-spring validation.

---

## Quality Gates (S121)

| Gate | Result |
|------|--------|
| `cargo fmt -- --check` | Clean |
| `cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery` | 0 warnings |
| `cargo test --workspace` | 869 lib + 9 integration + 43 forge PASS |
| `validate_barracuda_s121_rewire` | 80/80 PASS |
| `bench_cross_spring_modern` | 28/28 PASS |
| `validate_all` | 213/213 PASS |
| `#[allow(` in codebase | 0 (all `#[expect(`) |
| unsafe code | 0 |
| Files > 1000 lines | 0 |
| SPDX headers | 337/337 |

---

*V81 — neuralSpring Session 121 (March 4, 2026)*
