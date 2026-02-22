# ToadStool Handoff — neuralSpring Local Evolutions

This document catalogues BarraCUDA / ToadStool shortcomings that
`neuralSpring` evolved around locally, following the `hotSpring` pattern.

**Last reviewed:** ToadStool commit `dc540afd` (Session 25, Feb 20, 2026)
**Canonical handoff:** `wateringHole/handoffs/NEURALSPRING_TOADSTOOL_HANDOFF_FEB21_2026.md`

---

## Resolution Status

**All 11 neuralSpring shortcomings (S-01 through S-11) are now ABSORBED by
ToadStool `dc540afd`.** Key absorption commits:

| Commit | What It Did |
|--------|-------------|
| `fbedd222` | S-01/S-11 — `TensorSession` extended with `{MatMul, ReLU, GELU, Softmax, LayerNorm}`, single-encoder batch |
| `7c302d7b` | Deep debt — futures eliminated, async fix, vendor-ID-first |
| `82f953c8` | S-03 z-dispatch fix, S-04 softmax uniform size, S-05/S-06 Params fix |
| `81a6fd4b` | S-07 `from_buffer` public, S-08/S-09 round-trip elimination, S-10 `new_cpu_relaxed()` |
| `cce8fe7c` | wetSpring absorption — `GemmF64::WGSL` |
| `1ffe8b1a` | GPU FFT f64 validation + error system debt |

| # | Shortcoming | Severity | ToadStool Fix | Status |
|---|-------------|----------|---------------|--------|
| 1 | Per-op submission (S-01) | **Critical** | `TensorSession` single-encoder batch | **ABSORBED** |
| 2 | Naive matmul (S-02) | **Critical** | 4-tier `KernelRouter` (Naive/Tiled16/CpuTiled32/GpuEvolved32) | **ABSORBED** |
| 3 | MHA z-dispatch bug (S-03) | **High** | `workgroups_z = params.seq_len` | **ABSORBED** |
| 4 | Softmax pooled buffers (S-04) | **Medium** | `params.size` uniform (not `arrayLength`) | **ABSORBED** |
| 5 | `leaky_relu` Params (S-05) | **Low** | `{size: u32, negative_slope: f32}` (8 bytes) | **ABSORBED** |
| 6 | `elu` Params (S-06) | **Low** | `{size: u32, alpha: f32}` (8 bytes) | **ABSORBED** |
| 7 | `from_buffer` `pub(crate)` (S-07) | **High** | `pub fn from_buffer()` | **ABSORBED** |
| 8 | `layer_norm` round-trip (S-08) | **Medium** | `Tensor::from_pooled_buffer()` (no `read_buffer`) | **ABSORBED** |
| 9 | `log_softmax` round-trip (S-09) | **Medium** | `Tensor::from_pooled_buffer()` (no `read_buffer`) | **ABSORBED** |
| 10 | `science_limits()` CPU (S-10) | **Medium** | `WgpuDevice::new_cpu_relaxed()` | **ABSORBED** |
| 11 | `TensorSession` limited (S-11) | **High** | `SessionOp::{MatMul, ReLU, GELU, Softmax, LayerNorm, Attention, HeadSplit, HeadConcat}` | **ABSORBED** |

### neuralSpring rewiring (this session)

| Action | Details |
|--------|---------|
| `validate_barracuda_tensor` | Rewired from `evolved::layer_norm` / `evolved::log_softmax` to native `Tensor::layer_norm_wgsl()` / `Tensor::log_softmax_wgsl()` — 90/90 PASS |
| `validate_barracuda_tensor` | Added `leaky_relu` (S-05) and `elu` (S-06) tests — both now passing natively |
| `gpu.rs` | CPU path rewired to `WgpuDevice::new_cpu_relaxed()` (S-10 absorption) |
| `evolved/mod.rs` | Documented deprecation status for all workaround modules |

### What we absorbed from ToadStool

| ToadStool Evolution | neuralSpring Action |
|---|---|
| `WORKGROUP_SIZE_1D` / `WORKGROUP_SIZE_2D` constants | Imported into dispatch functions |
| `GpuDriverProfile` | Captured in `MatmulConfig` |
| wgpu v22 API | Already matched since initial port |
| `probe::seed_cache_from_heuristics()` | Called automatically by `WgpuDevice::from_existing()` |
| WGSL shader stability | All `include_str!` shaders verified unchanged |
| `ops::fft::{Fft1D, Ifft1D, Fft1DF64, Rfft}` | 24/24 checks PASS (f32 + f64 + Rfft) |
| `Tensor::from_buffer()` now public (S-07) | Validation rewired to native ops |
| `Tensor::layer_norm_wgsl()` no round-trip (S-08) | Validation rewired to native ops |
| `Tensor::log_softmax_wgsl()` no round-trip (S-09) | Validation rewired to native ops |
| `WgpuDevice::new_cpu_relaxed()` (S-10) | `gpu.rs` rewired |
| `leaky_relu_wgsl` / `elu_wgsl` Params fix (S-05/S-06) | Now validated in `validate_barracuda_tensor` |
| `TensorSession` ML ops (S-01/S-11) | Available for fused pipeline migration |

### New ToadStool capabilities available for leverage

| Capability | API | neuralSpring Use Case |
|------------|-----|----------------------|
| `TensorSession` ML ops | `session::{matmul, relu, gelu, softmax, layer_norm, run}` | Replace `evolved::fused_mlp` / `fused_transformer` |
| `StatefulPipeline` | `staging::StatefulPipeline::run_iterations()` | EA loops, ODE integration, HMM chains |
| `ReduceScalarPipeline` | `pipeline::ReduceScalarPipeline::sum_f64()` | Fitness aggregation, log-likelihood |
| `KernelRouter` 4-tier matmul | `ops::matmul` with `MatMulTier` | Replace `evolved::matmul_*.wgsl` |
| NAK eigensolve | `batched_eigh_nak_optimized_f64.wgsl` | Anderson localization eigensolver |
| `Fft1DF64` | `ops::fft::Fft1DF64` | f64 FFT — **now validated** (8/8, SHADER_F64) |
| `GemmF64::WGSL` | `ops::linalg::gemm_f64` | f64 GEMM shader source |
| `Tensor::from_arc_buffer` / `try_arc_buffer` | `tensor::Tensor` | Zero-copy buffer sharing |

---

## Evolved Module Retirement Plan

`src/evolved/` contains ~3375 LOC of workarounds. All non-metalForge modules
are now superseded by native BarraCUDA APIs. Documented in `evolved/mod.rs`.

### Fossilized (removed from active code — `metalForge/fossils/`)

| Module | LOC | Replacement |
|--------|-----|-------------|
| `fused_pipeline` | 680 | `TensorSession` single-encoder batch |
| `fused_mlp` | 356 | `TensorSession::{matmul, relu/gelu, run}` |
| `fused_transformer` | 725 | `TensorSession::{head_split, attention, head_concat, layer_norm}` |
| `layer_norm` | 268 | `Tensor::layer_norm_wgsl()` (no round-trip) |
| `log_softmax` | 259 | `Tensor::log_softmax_wgsl()` (no round-trip) |
| `matmul_cpu_tiled.wgsl` | 270 | `ops::matmul` CpuTiled32 tier |
| `matmul_gpu_evolved.wgsl` | 306 | `ops::matmul` GpuEvolved32 tier |
| `bench_fused_inference` | 688 | Deep fused pipeline coupling |
| `bench_scaling` | 439 | Deep fused pipeline coupling |
| **Total** | **~3991** | |

### Active (not yet absorbed)

| Module | LOC | Issue | Binary | Checks |
|--------|-----|-------|--------|--------|
| `mha` | 182 | S-03b: native projection shaders hang — **GPU `head_split.wgsl`/`head_concat.wgsl` now available** | `validate_barracuda_ml_inference`, `bench_transformer_block` | 17 |
| `hmm_forward_gpu` | 270 | No BarraCUDA equivalent | `validate_gpu_hmm_forward` | 13/13 |

### S-03b: MHA — PARTIAL FIX (GPU head split/concat shaders)

**Root cause**: Native MHA fuses matmul into projection shaders (heavy per-thread nested loops → GPU watchdog timeout).

**Local fix**: `metalForge/shaders/head_split.wgsl` and `head_concat.wgsl` — pure data movement:
- `head_split.wgsl`: [B,S,D] → [B,H,S,D/H]
- `head_concat.wgsl`: [B,H,S,D/H] → [B,S,D]

Validated at production sizes: B=4, S=128, H=8, d_head=64 (d_model=512). Validation: `validate_mha_gpu` (10/10 PASS).

**Fix**: Decompose into `matmul` (validated) + `head_split.wgsl` / `head_concat.wgsl`.  
**Absorption**: Replace `mha_projection.wgsl` with `matmul` + `head_split.wgsl`; replace `mha_output.wgsl` with `head_concat.wgsl` + `matmul`.  
**Status**: `evolved::mha` CPU workaround still active until ToadStool absorbs GPU shaders.

### Rewired to native APIs

| Binary | Previous | Now |
|--------|----------|-----|
| `bench_barracuda_tensor` | `evolved::layer_norm`/`log_softmax` | `Tensor::layer_norm_wgsl()`/`log_softmax_wgsl()` |
| `validate_barracuda_ml_inference` | Uses `evolved::mha` (S-03b, cannot rewire yet) | Kept |
| `bench_transformer_block` | Uses `evolved::mha` (S-03b, cannot rewire yet) | Kept |

---

## Benchmark Data

### ML Inference — 3-Way Comparison

Full analysis: `specs/BENCHMARK_ANALYSIS.md`

**MLP (4→64→64→10):**

| Backend | Median | Throughput |
|---------|--------|------------|
| Python/NumPy | 23 µs | 42,965 inf/s |
| BarraCUDA CPU (llvmpipe) | 4.7 ms | 211 inf/s |
| BarraCUDA GPU (RTX 4070) | 4.0 ms | 247 inf/s |

**Transformer encoder block (d=32, h=4, seq=8):**

| Backend | Median | Throughput |
|---------|--------|------------|
| Python/NumPy | 77 µs | 13,044 blk/s |
| BarraCUDA CPU (llvmpipe) | 11.0 ms | 91 blk/s |
| BarraCUDA GPU (RTX 4070) | 13.8 ms | 73 blk/s |

**Root cause:** Per-op `queue.submit()`. GPU ≈ CPU because tensors are
too small to amortize launch latency.

### Fused Pipeline (Single-Encoder)

| Pipeline | MLP | Transformer | Speedup vs Per-Op |
|----------|-----|-------------|-------------------|
| Per-op | 4.5 ms | 12.8 ms | 1.0× |
| **Fused** | **97 µs** | **164 µs** | **46× / 78×** |

### 3-Way Scaling (Fused + Evolved Shaders)

| Scale | Py(1t) | CPU | GPU | CPU/Py | GPU/Py |
|-------|--------|-----|-----|--------|--------|
| MLP large (3.1M) | 3.0 ms | **2.7 ms** | **178 µs** | **1.1× faster** | 16.8× |
| TF medium (103M) | 59 ms | **15.1 ms** | **566 µs** | **3.9× faster** | 104× |
| TF xlarge (6.6B) | 232 ms | 1.42 s | **17.8 ms** | — | **13.1× faster** |

---

## Phase 2 — BarraCUDA CPU Port Findings

**Date**: February 21, 2026
**Status**: All 15 Phase 0++ modules (+PINN, +DeepONet) ported to BarraCUDA CPU math. 170/170 checks PASS.

### Primitives Validated

| Primitive | Modules | Precision | Status |
|-----------|---------|-----------|--------|
| `numerical::rk45_solve` | regulatory, signal, game | Machine ε | **Excellent** |
| `linalg::solve_f64` | hmm, swarm | Machine ε | **Excellent** |
| `linalg::eigh_f64` | spectral, anderson | 1.75e-14 (n=32) | **S-12 RESOLVED** — Householder+QR |
| `special::chi_squared_sf` | introgression | 1e-10 | **Excellent** |
| `stats::variance` | all 13 modules | Machine ε | **Excellent** |
| `stats::pearson_correlation` | modes | Machine ε | **Excellent** |

### S-12: eigh_f64 — RESOLVED (Householder+QR)

**Local fix**: `src/eigh.rs` — Householder tridiagonalization + QL implicit shifts (Wilkinson) replaces Jacobi.

**Accuracy table**:

| n | Householder+QR | Jacobi | Improvement |
|---|----------------|--------|-------------|
| 4 | 1.13e-14 | 2.21e-14 | 2× |
| 8 | 3.05e-14 | 1.27e-1 | 4.2 trillion × |
| 16 | 5.28e-14 | 1.64e+1 | 312 trillion × |
| 32 | 1.83e-13 | 7.03e+1 | 383 trillion × |
| 64 | 5.43e-13 | 1.69e+2 | 311 trillion × |

- Anderson Hamiltonian n=32: 1.75e-14 (vs Jacobi's ~70)
- Validation: `validate_eigh_accuracy` (9/9 PASS)
- **Absorption target**: `barracuda::linalg::eigh_f64` — replace Jacobi with Householder+QR

---

## metalForge Shader Evolutions

Following the hotSpring pattern, these WGSL shaders are developed in
`metalForge/shaders/` with Rust orchestration in `src/evolved/`:

| Shader | Rust Module | Papers | Absorption Target |
|--------|-------------|--------|-------------------|
| `hmm_forward_log.wgsl` | `evolved::hmm_forward_gpu` | 016–018 | `barracuda::ops::hmm` |
| `head_split.wgsl` | *(bin-level)* | — | `barracuda::ops::mha::head_split` |
| `head_concat.wgsl` | *(bin-level)* | — | `barracuda::ops::mha::head_concat` |
| `batch_fitness_eval.wgsl` | *(bin-level)* | 011–015 | `barracuda::ops::batch_gemm` |
| `rk4_parallel.wgsl` | *(bin-level)* | 020–021 | `barracuda::ops::ode` |
| `mean_reduce.wgsl` | *(bin-level)* | 011–015 | `barracuda::pipeline::ReduceScalarPipeline` |

See `metalForge/shaders/ABSORPTION_TRACKER.md` for lifecycle tracking.

### New BarraCUDA Primitives Suggested

| Primitive | Use Case | Papers |
|-----------|----------|--------|
| `ops::mha::head_split` / `head_concat` | [B,S,D]↔[B,H,S,D/H] — decompose native MHA projection (S-03b fix) | — |
| `linalg::batch_matmul` | HMM forward/backward chain | 016–018 |
| `ea::batch_fitness` | Population-parallel fitness evaluation | 011–015 |
| `numerical::batch_rk45` | Multi-system ODE integration | 020–021 |
| `linalg::pairwise_distance` | O(N²) distance matrix | 017 |
| `ea::tournament_select` | GPU-parallel tournament selection | 011–015 |
| `stencil::neighborhood_scan` | Spatial cooperation model | 019 |
