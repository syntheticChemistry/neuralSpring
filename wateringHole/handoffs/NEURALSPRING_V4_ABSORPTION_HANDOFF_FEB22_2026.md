# neuralSpring → ToadStool Handoff V4 (Feb 22, 2026)

**neuralSpring SHA**: (current HEAD)
**ToadStool SHA**: `77f70b2e` (Session 31h)
**BarraCUDA**: 0.2+ with `ops::bio::*`, `spectral::*`, `ops::linalg::eigh_f64`

---

## What changed since V3

ToadStool `77f70b2e` absorbed a large batch of neuralSpring primitives across
Sessions 25–31h. neuralSpring has rewired to lean on upstream.

### Newly absorbed by ToadStool

| Category | What was absorbed | neuralSpring action |
|----------|-------------------|---------------------|
| **S-12 eigensolver** | `eigh_householder_qr` → `barracuda::ops::linalg::eigh_f64` | `src/eigh.rs` delegates to upstream; local fossil preserved |
| **8 WGSL shaders** | HMM, batch_fitness, RK4, Jaccard, Hamming, locus_var, spatial, batch_ipr | Forge crate re-exports from barracuda instead of `include_str!` |
| **HMM GPU dispatch** | `WGSL_HMM_FORWARD_LOG_F32` + `HmmBatchForwardF64` | Local `hmm_forward_gpu` now uses upstream shader source |
| **Spectral module** | `spectral::{anderson_*, hofstadter_*, lanczos, BatchIprGpu}` | Available for future migration |
| **Bio primitives** | `ops::bio::{Felsenstein, Gillespie, SmithWaterman, ANI, dN/dS, SNP, pangenome}` | Available for future paper extensions |

### State of neuralSpring absorption

| Item | Status |
|------|--------|
| S-01 through S-11 | Absorbed (`dc540afd`), fossilized |
| S-12 (eigh accuracy) | **Absorbed** (`77f70b2e`), delegated |
| S-03b (MHA projection hang) | Still local (`evolved::mha` + `head_split.wgsl`/`head_concat.wgsl`) |
| S-13 (PooledBuffer race) | Still local (`evolved::tensor_sync`) |
| 8 of 16 WGSL shaders | Sourced from upstream barracuda |
| 8 of 16 WGSL shaders | Still local (pairwise_l2, multi_obj_fitness, swarm_nn, hill_gate, head_split, head_concat, xoshiro128ss, mean_reduce) |

---

## Current neuralSpring metrics

| Metric | Value |
|--------|-------|
| Lib tests | 237 unit + 9 doc-tests |
| Line coverage | 94.9% |
| Validation binaries | 81 |
| Bench binaries | 5 |
| Modules | 29 + 3 evolved |
| WGSL shaders | 16 (8 upstream, 8 local) |
| Clippy | 0 warnings (pedantic + nursery) |
| Forge crate | `neural-spring-forge` 0.1.0 |

---

## Outstanding shortcomings (2 remain)

### S-03b: MHA projection shader hang

`barracuda::Tensor::multi_head_attention` hangs on projection dispatch.
neuralSpring works around this via `evolved::mha` which uses separate
`head_split.wgsl`/`head_concat.wgsl` shaders sandwiching TensorSession matmul.

**Fix path**: Debug `project_with_head_split` GPU execution in ToadStool.

### S-13: PooledBuffer drop-before-completion race

`barracuda::device::PooledBuffer::drop` can return a buffer to the pool
before its GPU commands complete. neuralSpring works around this via
`evolved::tensor_sync::{gpu_fence, fenced_matmul, materialize}`.

**Fix path**: Add `device.poll(Wait)` in `PooledBuffer::drop` before
returning to pool.

---

## 8 WGSL shaders still pending absorption

| Shader | Domain | Papers | Suggested upstream module |
|--------|--------|--------|--------------------------|
| `pairwise_l2.wgsl` | MODES novelty | 012 | `barracuda::ops::bio::pairwise_l2` |
| `multi_obj_fitness.wgsl` | Directed evolution | 014 | `barracuda::ops::bio::multi_obj_fitness` |
| `swarm_nn_forward.wgsl` | Swarm robotics | 015 | `barracuda::ops::bio::swarm_nn` |
| `hill_gate.wgsl` | Signal integration | 021 | `barracuda::ops::bio::hill_gate` |
| `mean_reduce.wgsl` | Fitness aggregation | — | `barracuda::pipeline::ReduceScalarPipeline` |
| `head_split.wgsl` | MHA | — | `barracuda::ops::mha` (fix S-03b first) |
| `head_concat.wgsl` | MHA | — | `barracuda::ops::mha` (fix S-03b first) |
| `xoshiro128ss.wgsl` | GPU PRNG | — | `barracuda::ops::prng` |

---

## BarraCUDA API usage summary

### Already wired (re-exported or delegated)

| API | neuralSpring consumer |
|-----|----------------------|
| `ops::linalg::eigh_householder_qr` | `src/eigh.rs` (delegation) |
| `ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32` | `evolved::hmm_forward_gpu`, forge `HMM_FORWARD_LOG` |
| `ops::bio::batch_fitness::WGSL_BATCH_FITNESS_EVAL` | forge `BATCH_FITNESS_EVAL` |
| `ops::rk_stage::WGSL_RK4_PARALLEL` | forge `RK4_PARALLEL` |
| `ops::bio::pairwise_jaccard::WGSL_PAIRWISE_JACCARD` | forge `PAIRWISE_JACCARD` |
| `ops::bio::pairwise_hamming::WGSL_PAIRWISE_HAMMING` | forge `PAIRWISE_HAMMING` |
| `ops::bio::locus_variance::WGSL_LOCUS_VARIANCE` | forge `LOCUS_VARIANCE` |
| `ops::bio::spatial_payoff::WGSL_SPATIAL_PAYOFF` | forge `SPATIAL_PAYOFF` |
| `spectral::batch_ipr::WGSL_BATCH_IPR` | forge `BATCH_IPR` |

### Previously wired (unchanged from V3)

| API | neuralSpring consumer |
|-----|----------------------|
| `Tensor::from_data`, `to_vec` | All validation binaries |
| `Tensor::layer_norm_wgsl` | ML inference validation |
| `Tensor::log_softmax_wgsl` | ML inference validation |
| `session::TensorSession` | `evolved::mha`, bench binaries |
| `ops::fft::{Fft1D, Ifft1D, Fft1DF64, Rfft}` | `validate_barracuda_fft` |
| `staging::StatefulPipeline` | `validate_gpu_stateful_pipeline` |
| `dispatch::{dispatch_for, DispatchTarget}` | 4 cross-dispatch binaries |
| `stats::*`, `linalg::*`, `numerical::*`, `special::*` | 17 CPU port binaries |

---

*Handoff V4 — neuralSpring → ToadStool Session 31h absorption sync*
