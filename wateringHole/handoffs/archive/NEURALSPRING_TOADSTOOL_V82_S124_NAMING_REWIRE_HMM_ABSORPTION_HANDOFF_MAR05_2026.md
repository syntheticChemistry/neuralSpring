<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/barraCuda Handoff V82 — airSpring V069 Naming Rewire + HMM Forward ComputeDispatch Absorption + Paper 026

**Date**: March 5, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/barraCuda team
**License**: AGPL-3.0-or-later
**Covers**: Sessions 122–124 — airSpring V069 naming rewire, HMM forward chain ComputeDispatch absorption, Paper 026 Chuna LSTM glucose prediction, documentation alignment
**Supersedes**: V81 (S121 SimpleMlp rewire + HMM Viterbi f64 ComputeDispatch + cross-spring benchmark)
**barraCuda**: v0.3.1 standalone (`../barraCuda/crates/barracuda`)

---

## Executive Summary

- **airSpring V069 naming rewire**: Swept 20 library `.rs` files, 38 binary `.rs` files, and 10+ `specs/` docs to enforce the naming convention: `ToadStool` = hardware dispatch/orchestration/streaming, `BarraCUDA` = math/shaders/ops. Historical `ToadStool` session references preserved with clarifying context (e.g., "`BarraCUDA` (via `ToadStool` S76)").
- **HMM forward chain → `ComputeDispatch` absorption**: `gpu_ops/bio/hmm.rs::hmm_forward_chain_gpu` now routes through `barracuda::ops::bio::HmmBatchForwardF64` (single-dispatch f64 log-domain GPU). Falls back to legacy per-step `Tensor` loop on error. This completes the HMM GPU absorption alongside V81's Viterbi rewire.
- **Paper 026 — Chuna LSTM blood glucose prediction**: Full 4-tier implementation (Py 5/5, Rs 8/8, bC 4/4, gT 4/4). LSTM reservoir on CGM time series, R²=0.88–0.93. `barracuda::nn::SimpleMlp` for readout, `BarraCUDA` Tensor for GPU path.
- **`validate_all` gap closure**: Added `validate_toadstool_s79_rewire` and `validate_toadstool_s93_barracuda_extraction` — now **217/217**.
- **Quality gates**: fmt ✓ · clippy ✓ (0 warnings, pedantic+nursery, all-features) · test ✓ (880/880 lib) · doc ✓ · 217/217 validate\_all.

---

## Part 1: airSpring V069 Naming Rewire

### Scope

| Category | Files | Changes |
|----------|-------|---------|
| Library `.rs` | 20 | Comment/doc updates: math provenance → `BarraCUDA`, dispatch role → `ToadStool` |
| Binary `.rs` | 38 | `eprintln!` labels, `bench_once` tags, provenance comments |
| Specs `.md` | 10+ | Backticking, role clarification, roadmap item updates |
| Root docs | 3 | `README.md`, `CHANGELOG.md`, `EVOLUTION_READINESS.md` |

### Naming Convention Applied

| Reference Type | Name Used | Example |
|---------------|-----------|---------|
| Math/shader/op provenance | `BarraCUDA` | "absorbed via `BarraCUDA` (ToadStool S76)" |
| Hardware dispatch/streaming | `ToadStool` | "`ToadStool` three-zone pattern" |
| Historical session commits | Both, clarified | "`BarraCUDA` (ToadStool S87, `2dc26792`)" |
| Standalone crate reference | `barraCuda` | "`barracuda::nn::SimpleMlp`" |

### Key Decisions

1. **Preserved historical ToadStool commit references** — S60, S64, S66, S76, S87, S89 all kept with their original session numbers since these were `ToadStool` commits at the time. Context added to show math now lives in `BarraCUDA`.
2. **`ToadStool` kept for dispatch functions** — `mixed_dispatch()`, `Fp64Strategy`, hardware promotion, streaming pipeline, `DeviceCapabilities`. These are orchestration, not math.
3. **Backticked all primal names** — `ToadStool`, `BarraCUDA`, `barraCuda` consistently backticked in prose for clarity.

---

## Part 2: HMM Forward Chain ComputeDispatch Absorption

### What Changed

| Before | After |
|--------|-------|
| Per-step f32 `Tensor` loop: create tensor → GPU matmul → readback per timestep | Primary: `barracuda::ops::bio::HmmBatchForwardF64` single dispatch (f64, log-domain) |
| f32 precision (Tensor API) | f64 precision (ComputeDispatch) |
| N CPU↔GPU round-trips (one per timestep) | Zero round-trips (full forward pass on GPU) |
| — | Fallback: legacy per-step path if fused dispatch fails |

### Implementation: `hmm_forward_chain_gpu_fused`

```
fn hmm_forward_chain_gpu_fused(
    initial, transition, emission, observations,
    n_states, n_obs, device
) -> Result<f64, String>
```

1. Converts linear-domain inputs to log-domain (`v.max(LOG_GUARD).ln()`)
2. Creates GPU storage buffers via `device.create_buffer_init()`
3. Allocates output buffers: `log_alpha [n_seqs × T × S]`, `log_lik [n_seqs]`
4. Single `HmmBatchForwardF64::dispatch(ns, no, nt, n_seqs, ...)` call
5. Reads back `log_lik[0]` via staging buffer + `map_async`

### Orchestration

`hmm_forward_chain_gpu` tries `_fused` first. On any error, falls back to `_perstep` (the legacy per-step Tensor implementation, renamed from the original function).

---

## Part 3: Paper 026 — Chuna LSTM Blood Glucose Prediction

### Implementation

| Tier | File | Tests | Status |
|------|------|-------|--------|
| Python | `control/glucose_prediction/baseline_glucose.py` | 5/5 | PASS |
| Rust | `src/bin/validate_glucose_prediction.rs` | 8/8 | PASS |
| BarraCUDA CPU | `src/bin/validate_barracuda_glucose.rs` | 4/4 | PASS |
| GPU Tensor | `src/bin/validate_barracuda_gpu_glucose.rs` | 4/4 | PASS |

LSTM reservoir with readout via `barracuda::nn::SimpleMlp`. CGM synthetic time series. R²=0.88–0.93 across tiers. Baseline JSON stored in `control/wdm/glucose_prediction_baseline.json`.

---

## Part 4: Quality Gates (S124)

| Gate | Result |
|------|--------|
| `cargo fmt -- --check` | Clean |
| `cargo clippy --workspace --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery` | 0 warnings |
| `cargo test --lib` | 880/880 PASS |
| `cargo doc --no-deps` | 0 warnings |
| `validate_all` | 217/217 PASS |
| `#[allow(` in codebase | 0 (all `#[expect(`) |
| unsafe code | 0 |
| Files > 1000 lines | 0 |
| SPDX headers | 340+ |

---

## Part 5: BarraCUDA Evolution Observations for the Team

### What Works Well

1. **`HmmBatchForwardF64`** — Clean API, single dispatch for full forward pass. Log-domain is the right choice for numerical stability. Paired with V81's `hmm_viterbi`, neuralSpring now has both HMM algorithms on single-dispatch GPU.

2. **`SimpleMlp::forward()`** — Covers all neuralSpring use cases (EOS, transport, glucose). The `Activation` enum (`Relu`, `Tanh`, `Sigmoid`, `Gelu`, `Identity`) is complete for our needs.

3. **`ComputeDispatch` pattern** — f64 single-dispatch beats f32 per-step Tensor in both precision and throughput. Springs should always prefer `ComputeDispatch` when a full-pass shader exists.

### Absorption Opportunities

| Item | What | Why | Effort |
|------|------|-----|--------|
| `SimpleMlp::forward_gpu(device)` | GPU-native MLP forward | neuralSpring GPU validators manually unpack `DenseLayer` → Tensor matmul. A native GPU path would eliminate this boilerplate | Medium |
| `HmmBatchForwardF64` batch helper | CPU-side data helpers for single-sequence use | Creating `[n_seqs=1]` buffers + log-domain conversion is ~40 LOC boilerplate. A `hmm_forward_single(device, trans, emit, init, obs)` convenience would help | Low |
| Tridiagonal eigensolver | Papers 022-023 need eigendecomposition of tridiagonal matrices | neuralSpring uses `eigh_f64` (dense Householder+QR) which works but is O(n³). A Sturm bisection or divide-and-conquer `ComputeDispatch` would be valuable | High |
| GPU PRNG dispatch | `xoshiro128ss.wgsl` exists in metalForge but has no `ComputeDispatch` wrapper | Stochastic GPU algorithms (Monte Carlo, evolutionary ops) would benefit from a `PrngDispatch` | Medium |

### Remaining Local WGSL Shaders (Absorption Candidates)

| Shader | Location | What It Does | Blocker |
|--------|----------|-------------|---------|
| `xoshiro128ss.wgsl` | metalForge/shaders | GPU PRNG (parallel Xoshiro128**) | No upstream PRNG module |
| `swarm_nn_scores.wgsl` | metalForge/shaders | Batch NN forward for swarm fitness | Domain-specific, low reuse value |
| `head_split.wgsl` | metalForge/shaders | MHA head dimension splitting | Partially in barraCuda's MHA; needs 2D↔3D adapter |
| `head_concat.wgsl` | metalForge/shaders | MHA head concatenation | Same as above |

### Precision Observations

- **f64 HMM** (ComputeDispatch) produces strictly better results than **f32 HMM** (Tensor per-step). The difference is measurable at T>100 timesteps.
- **`SimpleMlp`** uses f64 weights throughout. GPU Tensor validation at f32 shows ≤1e-3 relative error vs CPU f64 — acceptable for inference, not for training.
- **Eigensolvers**: Householder+QR via `eigh_f64` achieves 1.75e-14 at n=32. Divide-and-conquer would push to 1e-14 for larger matrices.

---

## Counts

| Metric | Value |
|--------|-------|
| Library tests | 880 |
| Validation/bench binaries | 238 |
| `validate_all` | 217/217 |
| Upstream rewires | 46 |
| WGSL shaders absorbed | 21 + 15 coralForge df64 |
| Local WGSL shaders | 4 (absorption candidates) |
| metalForge WGSL shaders | 42 |
| CPU→GPU dispatch ops | 47 (~97%) |
| Named tolerances | 139+ |
| clippy warnings | 0 (pedantic+nursery) |
| doc warnings | 0 |
| Papers implemented | 26 (25 + Paper 026 glucose) |

---

*V82 — neuralSpring Sessions 122–124 (March 5, 2026)*
