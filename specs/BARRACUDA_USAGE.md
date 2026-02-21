# BarraCUDA Usage Audit — neuralSpring

**Last Updated**: February 21, 2026 (post deep-evolution audit)
**BarraCUDA version**: `0.2.0` (path dep: `../phase1/toadstool/crates/barracuda`)
**Purpose**: Map every barracuda capability we use, what we're missing, and the evolution path

---

## What We Use

### Device & GPU Infrastructure

| Module | Where Used | Purpose |
|--------|-----------|---------|
| `device::WgpuDevice` | `gpu.rs`, all FFT/tensor/ML binaries | GPU device creation and management |
| `device::capabilities::WORKGROUP_SIZE_*` | `evolved/mha.rs`, `evolved/hmm_forward_gpu.rs` | Shader workgroup sizing |

### Statistics

| Module | Where Used (13 binaries) | Purpose |
|--------|-------------------------|---------|
| `stats::correlation::variance` | counterdiabatic, modes, eco, directed, swarm, sate, game, hmm | Population variance for statistical checks |
| `stats::pearson_correlation` | modes | Correlation between diversity metrics |

### Linear Algebra

| Module | Where Used | Purpose |
|--------|-----------|---------|
| `linalg::solve_f64` | hmm, swarm, linalg validation | Linear system solve |
| `linalg::eigh_f64` | spectral, anderson, linalg validation | Eigendecomposition |
| `linalg::cholesky_f64` | linalg validation | Cholesky factorization |
| `linalg::lu_det`, `lu_solve` | linalg validation | LU decomposition |
| `linalg::tridiagonal_solve` | linalg validation | Tridiagonal solver |
| `ops::linalg::svd::*` | linalg_ext validation | SVD |
| `linalg::gen_eigh::*` | linalg_ext validation | Generalized eigendecomposition |

### Numerical

| Module | Where Used | Purpose |
|--------|-----------|---------|
| `numerical::rk45_solve` | regulatory, signal, game | Adaptive ODE integration |

### Special Functions

| Module | Where Used | Purpose |
|--------|-----------|---------|
| `special::chi_squared_sf/cdf` | introgression | Chi-squared hypothesis testing |
| `special::gamma, erf, erfc, bessel_*, legendre, hermite, laguerre, factorial` | special validation | Mathematical special functions |

### Tensor API

| Module | Where Used | Purpose |
|--------|-----------|---------|
| `tensor::Tensor` | 84+ op validation, FFT, ML inference, benchmarks | Core GPU tensor type |
| `ops::fft::{Fft1D, Ifft1D}` | FFT validation | Cooley-Tukey radix-2 FFT |
| `ops::*_f64` (7 ops) | tensor_f64 validation | Double-precision GPU reductions |

### Shaders

| Module | Where Used | Purpose |
|--------|-----------|---------|
| `shaders::precision::cpu` | precision validation | CPU f64 precision checks |
| `shaders::quantized` | quantized validation | Q4/Q8 dequantization + GEMV |

---

## What We DON'T Use (But Should)

### High Priority — Direct Replacements for Remaining Evolutions

| BarraCUDA Module | Replaces | Impact | Blocker |
|-----------------|----------|--------|---------|
| `ops::logsumexp` / `logsumexp_wgsl` | `hmm_forward_gpu` manual logsumexp | Correctness + perf | Need to verify API compatibility |
| Native `ops::mha` | `evolved::mha` (S-03b) | Retire 182 LOC | Projection shaders hang on RTX 4070 |
| `ops::pairwise_distance` | Hand-rolled distance in SATé (017) | Correctness | None — ready to integrate |
| `linalg::batched_eigh_gpu` | `eigh_f64` accuracy gap (S-12) | Fix 1e-3 gap at n=8 | Verify NAK solver accuracy |

### Medium Priority — Streaming and Pipeline

| BarraCUDA Module | Use Case | Impact |
|-----------------|----------|--------|
| `staging::StatefulPipeline` | HMM chain, ODE loops, iterative EA | Eliminate CPU loop over GPU dispatches |
| `staging::UnidirectionalPipeline` | Streaming fitness eval | Reduce round-trips from O(T) to O(1) |
| `pipeline::ReduceScalarPipeline` | Log-likelihood, convergence checks | Scalar-only readback |

### Low Priority — Future Features

| BarraCUDA Module | Potential Use | When |
|-----------------|---------------|------|
| `ops::rnn_cell` / `lstm_cell` | Sequence forecasting GPU port | Phase 4 |
| `ops::rope` | Rotary embeddings for Transformer | Phase 4 |
| `nn::Layer` / `nn::Optimizer` | GPU training | Phase 5 |
| `compute_graph` | Lazy execution | Phase 5 |
| `session::TensorSession` | Session management | Phase 5 |
| `esn_v2` | ESN surrogate | Phase 4 (NPU path) |

---

## Feature Flags

| Feature | Status | Why |
|---------|--------|-----|
| `default` | Enabled | Basic tensor/device |
| `unidirectional` | **Not enabled** | Needed for `UnidirectionalPipeline` |
| `parallel` | Not enabled | Not needed yet (rayon-based cascade) |
| `benchmarks` | Not enabled | Use our own bench binaries |
| `serde` | Not enabled | Could enable for calibration caching |

**Recommendation**: Enable `unidirectional` when integrating Phase 3b streaming.

---

## Evolution Path

### Phase 3b — Streaming (Next)

1. Enable `unidirectional` feature flag
2. Replace `hmm_forward_gpu` manual timestep loop with `StatefulPipeline`
3. Replace `validate_gpu_batch_fitness` direct dispatch with `UnidirectionalPipeline`
4. Add `ReduceScalarPipeline` for log-likelihood extraction

### Phase 3d — Retire Remaining Evolutions

S-01, S-02, S-08, S-09 absorbed and fossilized. Remaining:

1. When ToadStool fixes S-03b (projection shader hang): retire `evolved::mha`
2. When BarraCUDA adds `ops::hmm`: retire `evolved::hmm_forward_gpu`

### Phase 4 — Cross-System

1. Integrate `ops::pairwise_distance` for SATé (017)
2. Integrate `linalg::batched_eigh_gpu` for Anderson/spectral (022–023)
3. Use `ops::logsumexp` in HMM forward to replace manual implementation
4. NPU path via AKD1000 for ESN surrogates

---

---

## Deep Evolution Alignment (February 21, 2026)

### GPU-Ready Modules (flat row-major layout)

These modules now use flat `Vec<f64>` layouts that match GPU buffer bindings
directly — no conversion needed for `Tensor::from_data` or raw `wgpu::Buffer`:

| Module | Before | After | BarraCUDA Target |
|--------|--------|-------|------------------|
| `hmm.rs` | `Vec<Vec<f64>>` | Flat `Vec<f64>` (N×N, N×M, T×N) | `ops::hmm` / `StatefulPipeline` |
| `spectral_commutativity.rs` | `Vec<Vec<f64>>` | Flat `Vec<f64>` (N×N) | `ops::matmul` (GEMM f64) |
| `directed_evolution.rs` | `Vec<Vec<f64>>` | Flat `Vec<f64>` (pop×genome, pop×obj) | `ops::batch_gemm` |
| `sate_alignment.rs` | `Vec<Vec<u8>>` / `Vec<Vec<f64>>` | Flat `Vec<u8>` (n×len), `Vec<f64>` (n×n) | `ops::pairwise_distance` |
| `anderson_localization.rs` | `Vec<Vec<f64>>` | Flat `Vec<f64>` (N×N) | `linalg::eigh_f64` |
| `pinn.rs` | Scalar + grid | `Vec<f64>` flat grid | `tensor::{matmul, tanh}` |
| `deeponet.rs` | Scalar + poly | `Vec<f64>` flat grid | `tensor::{matmul, dot}` |

### Consolidated Primitives (candidates for barracuda expansion)

| neuralSpring Primitive | BarraCUDA Equivalent | Status |
|----------------------|---------------------|--------|
| `primitives::shannon_entropy` | `stats::entropy` | **New candidate** |
| `primitives::hill_activation` | `numerical::hill` | **New candidate** |
| `primitives::sigmoid` | `ops::sigmoid` (f32 GPU exists) | f64 CPU candidate |
| `primitives::rk4_step::<N>` | `numerical::rk45_solve` | Complementary (fixed vs adaptive) |
| `primitives::LOG_GUARD` | `numerical::constants` | **New candidate** |
| `primitives::DIVISION_GUARD` | `numerical::constants` | **New candidate** |

### Next Absorption Targets (ordered by readiness)

1. **HMM flat layout → `ops::hmm`**: `Hmm::from_flat()` provides GPU-native
   entry; `evolved/hmm_forward_gpu.rs` is the reference dispatcher (270 LOC)
2. **Spectral flat layout → `ops::matmul`**: `mat_mul(a, b, n)` is the CPU
   reference for GEMM f64 validation
3. **`require!` pattern → `barracuda::testing`**: Reusable across all Springs
4. **Shannon entropy → `stats::entropy`**: Well-tested, 8 unit tests
5. **Hill functions → `numerical::hill`**: Used by regulatory biology + signal
   integration across neuralSpring and potentially hotSpring

*Barracuda usage audit — neuralSpring, February 21, 2026.*
