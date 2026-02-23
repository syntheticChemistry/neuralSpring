# BarraCUDA Usage Audit — neuralSpring

**Last Updated**: February 23, 2026 (Sessions 40–48)
**BarraCUDA version**: `0.2.0` (path dep: `../phase1/toadstool/crates/barracuda`)
**Purpose**: Map every barracuda capability we use, what we're missing, and the evolution path

---

## What We Use

### Device & GPU Infrastructure

| Module | Where Used | Purpose |
|--------|-----------|---------|
| `device::WgpuDevice` | `gpu.rs`, all FFT/tensor/ML binaries | GPU device creation and management |
| `device::capabilities::WORKGROUP_SIZE_*` | `evolved/mha.rs` | Shader workgroup sizing (legacy) |
| `device.limits()` / `device.features()` | `gpu.rs` (`GpuCapabilities`) | Runtime hardware discovery — workgroup limits, f64/f16 support, buffer sizes |

### Statistics

| Module | Where Used (13 binaries) | Purpose |
|--------|-------------------------|---------|
| `stats::correlation::variance` | counterdiabatic, modes, eco, directed, swarm, sate, game, hmm | Population variance for statistical checks |
| `stats::pearson_correlation` | modes | Correlation between diversity metrics |

### Linear Algebra

| Module | Where Used | Purpose |
|--------|-----------|---------|
| `linalg::solve_f64` | hmm, swarm, linalg validation | Linear system solve (takes `Arc<WgpuDevice>`) |
| `linalg::eigh_f64` | spectral, anderson, linalg validation | Eigendecomposition (takes `Arc<WgpuDevice>`) |
| `linalg::cholesky_f64` | linalg validation | Cholesky factorization (takes `Arc<WgpuDevice>`) |
| `linalg::lu_det`, `lu_solve` | linalg validation | LU decomposition |
| `linalg::tridiagonal_solve` | linalg validation | Tridiagonal solver |
| `ops::linalg::svd::*` | linalg_ext validation | SVD |
| `linalg::gen_eigh::*` | linalg_ext validation | Generalized eigendecomposition (takes `Arc<WgpuDevice>`) |

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
| `linalg::batched_eigh_gpu` | `eigh_f64` — **S-12 ABSORBED** (`77f70b2e`) | Householder+QR upstream | NAK GPU eigensolve also available |

### Medium Priority — Streaming and Pipeline

| BarraCUDA Module | Use Case | Impact |
|-----------------|----------|--------|
| `staging::StatefulPipeline` | HMM chain, ODE loops, iterative EA | Eliminate CPU loop over GPU dispatches |
| `staging::UnidirectionalPipeline` | Streaming fitness eval | Reduce round-trips from O(T) to O(1) |
| `pipeline::ReduceScalarPipeline` | Log-likelihood, convergence checks | Scalar-only readback |

### New in Session 42 (`5437c170`) — NN Compute and Bug Fixes

| BarraCUDA Module | Use Case | Status |
|-----------------|----------|--------|
| `ops::nn::conv2d.wgsl` | Batched Conv2D (LeNet-5 conv layers) | Available — not yet wired to executor |
| `ops::nn::maxpool2d.wgsl` | MaxPool2D (LeNet-5 pooling) | Available — not yet wired to executor |
| `ops::nn::avgpool2d.wgsl` | AvgPool2D (alternative pooling) | Available — not yet wired to executor |
| `cpu_conv_pool::{conv2d, max_pool2d, avg_pool2d}` | CPU reference implementations | Available — used by CpuExecutor |
| `esn_v2::export_weights/import_weights` | GPU-train → NPU-deploy pipeline | Available |
| S-13 PooledBuffer race fix | Deferred return + device poll | **Flows automatically** via path dep |
| TS-003 trig precision | 7-term Taylor + Cody-Waite range reduction | **Flows automatically** via path dep |
| TS-001 pow_f64 precision | Extended exp/log polynomials | **Flows automatically** via path dep |
| TS-004 FusedMapReduceF64 fix | Single command encoder | **Flows automatically** via path dep |

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

S-01, S-02, S-08, S-09 absorbed and fossilized. **Session 47**:

1. **S-03b FIXED** upstream (z-dimension dispatch) — `evolved::mha` kept until full native MHA validation
2. **evolved::hmm_forward_gpu RETIRED** — HmmBatchForwardF64 (wetSpring) is primary HMM path

### Phase 4 — Cross-System

1. Integrate `ops::pairwise_distance` for SATé (017)
2. Integrate `linalg::batched_eigh_gpu` for Anderson/spectral (022–023)
3. Use `ops::logsumexp` in HMM forward to replace manual implementation
4. NPU path via AKD1000 for ESN surrogates

### Phase 5b — Full-Stack GPU Tensor Validation (February 22, 2026)

Phase 5b exercises the unified `Tensor` API (matmul, transpose, tanh, sigmoid, add)
on a live GPU (RTX 4070, Vulkan backend) across **23 papers**. ALL GREEN.

**S-16 FIXED.** S-15 root-caused. 98+ GPU Tensor checks PASS.

| ID | Issue | Resolution |
|----|-------|------------|
| S-14 | Naive matmul hang (small square, N < 32) | **Workaround**: A×B^T pattern |
| S-15 | Matmul hang when f32 elements ≤ 0.1 magnitude | **Root-caused**: WGPU/Vulkan driver bug. Data ≥ 0.5 avoids hang |
| S-16 | Transpose dispatches 256 instead of 16 | **FIXED**: `const TILE: u32 = 16` |

**What we learned for usage:**

1. `Tensor::matmul` works for ALL data with magnitude ≥ 0.5. Use `rng.uniform() * 0.5 + 0.5`
   for test data generation. Covers: spectral, eco, HMM, fitness, NN forward passes, Anderson.
2. `Tensor::transpose` works correctly after S-16 fix. All pairwise validators PASS.
3. `Tensor::tanh`, `Tensor::sigmoid`, and `Tensor::add` work correctly across all domains.
4. f32 GPU vs f64 CPU agreement is excellent (< 1e-3 for most operations).
5. The A×B^T pattern (non-square intermediates) reliably avoids S-14 Naive tier hang.

---

## Session Evolution Narrative

| Session | Key Evolution |
|---------|----------------|
| **Session 39** | First upstream wrappers — 6 bio ops (HMM, pairwise, batch fitness, etc.) |
| **Session 42** | Full rewire — 10/10 upstream parity, LeNet-5 `cpu_conv_pool` |
| **Session 43** | Upstream expansion (Gillespie, wetSpring trio, chi²), CPU vs GPU parity (bit-identical), mixed-hardware dispatch design |

### Session 43 — Upstream Expansion & Mixed Hardware (February 22, 2026)

**New APIs wired:**

| API | Module | Validator | Purpose |
|-----|--------|-----------|---------|
| `GillespieGpu` | `ops::bio::gillespie` | `validate_gpu_gillespie` | f64 parallel SSA (stochastic simulation) |
| `TaxonomyFcGpu` | `ops::bio::taxonomy_fc` | `validate_upstream_taxonomy` | f64 metagenomics |
| `KmerHistogramGpu` | `ops::bio::kmer_histogram` | `validate_upstream_kmer` | k-mer histograms |
| `UniFracPropagateGpu` | `ops::bio::unifrac_propagate` | `validate_upstream_unifrac` | tree propagation |
| `chi_squared::*` | `special::chi_squared` | `validate_barracuda_chi_squared` | PDF/CDF/moments (within 1e-4 of SciPy) |

**New dispatch heuristics** (`metalForge/forge/src/dispatch.rs`):

| Heuristic | Threshold | Purpose |
|-----------|-----------|---------|
| `logsumexp_substrate(batch, width)` | batch×width > 20k → GPU | Batched logsumexp (HMM/phylo) |
| `stochastic_substrate(n_pops, n_loci, two_n)` | n_pops×n_loci×two_n > 100k → GPU | Wright-Fisher / Gillespie |

**CPU vs GPU parity:** `validate_cpu_gpu_parity` (17/17 PASS) — MatMul, ReLU, Sigmoid, Tanh, Sum
bit-identical across GPU and CPU Tensor paths.

**Mixed-hardware:** `metalForge/mixed.rs`, `pcie_bridge.rs` — transfer cost model, PCIe P2P
DMA design; `validate_mixed_dispatch` (16/16 PASS).

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
   entry; `HmmBatchForwardF64` (wetSpring) is primary path (evolved/hmm_forward_gpu retired S47)
2. **Spectral flat layout → `ops::matmul`**: `mat_mul(a, b, n)` is the CPU
   reference for GEMM f64 validation
3. **`require!` pattern → `barracuda::testing`**: Reusable across all Springs
4. **Shannon entropy → `stats::entropy`**: Well-tested, 8 unit tests
5. **Hill functions → `numerical::hill`**: Used by regulatory biology + signal
   integration across neuralSpring and potentially hotSpring

---

## Session 44 — Multi-GPU Portability & Upstream Bug Fixes (February 23, 2026)

**New hardware:** TITAN V 12 GB (NVK open-source Vulkan driver, Volta GV100)
**Key result:** 131/131 validators PASS on both RTX 4070 and TITAN V — bit-identical

### New APIs Exercised

| API | Validator | Checks | Finding |
|-----|-----------|--------|---------|
| `Tensor::conv2d()` | `validate_barracuda_gpu_lenet` | 8/8 | Conv2d WGSL shader works via Tensor API |
| `Tensor::maxpool2d()` | `validate_barracuda_gpu_lenet` | (incl.) | MaxPool2d WGSL shader works via Tensor API |
| `Tensor::softmax()` | `validate_barracuda_transformer` | (incl.) | Global softmax (not row-wise — document for attention usage) |
| `Tensor::mean()` | `validate_barracuda_tensor` | (incl.) | Fixed: entry point + double-divide bug |

### Upstream Bugs Fixed

| Bug | Location | Fix |
|-----|----------|-----|
| mean_reduce entry point | `ops/mean.rs` | `"main"` → `"mean_reduce"`, single-f32 readback |
| chi-squared expected values | `validate_barracuda_chi_squared.rs` | Textbook-rounded → full-precision computed |

### Benchmark Findings (Rust vs Python)

Pure Rust is **178.5× faster** than single-thread Python/NumPy across 11 Phase 0++ kernels.
Exception: commutator (64×64) — NumPy's BLAS matmul is 2.5× faster than pure Rust loops.
This validates the reverse pipeline: prove math on GPU, then optimize CPU with BLAS techniques.

### Multi-GPU Adapter Selection

`NEURALSPRING_BACKEND=titan` selects TITAN V; default selects RTX 4070.
Implemented via `Gpu::new()` adapter name-substring matching in `src/gpu.rs`.

---

## Sessions 45–46 — Pure GPU Promotion via Tensor API (February 23, 2026)

### New Modules

| Module | Purpose | Tensor Methods Used |
|--------|---------|-------------------|
| `gpu_ops.rs` | 38 GPU-accelerated functions | All major Tensor ops |
| `gpu_dispatch.rs` | Capability-based runtime dispatch | `WgpuDevice` detection |

### New Tensor API Usage Patterns

| Pattern | Functions | Key Methods |
|---------|----------|-------------|
| GEMV chain | hmm_forward_step, hmm_backward_step, replicator_step | `matmul`, `mul`, `transpose` |
| Broadcast + reduce | hmm_viterbi_step | `broadcast`, `add`, `max_dim` |
| Column reduction | allele_frequencies | `sum_dim(0)`, `div_scalar` |
| Transcendental pipeline | hill_activation_batch | `log_wgsl`, `mul_scalar`, `exp_wgsl` |
| Elementwise compose | nucleotide_diversity, pearson_correlation | `mul`, `sub`, `add`, `mean` |
| Dimension reduction | variance, logsumexp | `mean_dim`, `sum_dim` |

### API Gaps Identified

| Gap | Impact | Workaround |
|-----|--------|------------|
| No `argmax_dim()` | Viterbi needs indices, not just max values | CPU argmax after `max_dim` readback |
| No `pow_scalar(n)` | Hill activation `x^n` | `exp(n * ln(x))` pipeline |
| No `softmax_dim(axis)` | Row-wise attention softmax | `ScaledDotProductAttention` |
| No `div(other)` (elementwise) | Ratio computation | Uploaded reciprocal + `mul` |

### Ownership Model (Documented)

**Consuming** (`self`): `matmul`, `softmax`, `sigmoid`, `gelu_wgsl`, `log_wgsl`, `exp_wgsl`, `sqrt_wgsl`, `broadcast`

**Borrowing** (`&self`): `transpose`, `add`, `sub`, `mul`, `sum`, `mean`, `max`, `norm`, `mul_scalar`, `add_scalar`, `div_scalar`, `sum_dim`, `mean_dim`, `max_dim`, `min_dim`, `reshape`, `to_vec`

### Validation

| Validator | Checks | Status |
|-----------|--------|--------|
| `validate_gpu_promotion` | 27/27 | PASS (both GPUs) |
| `validate_gpu_phase_b` | 20/20 | PASS (both GPUs) |
| `validate_all` | 133/133 | ALL GREEN |

---

## Session 47 — Typed Op Migration (February 23, 2026)

### 10 Validators Rewired to Typed BarraCUDA Ops

Migrated from raw wgpu dispatch to typed BarraCUDA ops:

| Validator | Typed Op |
|-----------|----------|
| `validate_gpu_batch_fitness` | BatchFitnessGpu |
| `validate_gpu_sate` | PairwiseHammingGpu |
| `validate_gpu_pangenome` | PairwiseJaccardGpu |
| `validate_gpu_meta_pop` | LocusVarianceGpu |
| `validate_gpu_game_theory` | SpatialPayoffGpu |
| `validate_gpu_directed` | MultiObjFitnessGpu |
| `validate_gpu_modes` | PairwiseL2Gpu |
| `validate_gpu_anderson` | BatchIprGpu |
| `validate_gpu_swarm` | SwarmNnGpu |
| `validate_gpu_signal` | HillGateGpu |

### 4 New gpu_ops Functions

| Function | Purpose |
|----------|---------|
| `eigh_gpu` | BatchedEighGpu (single-dispatch for n≤32) |
| `disorder_sweep_gpu` | Batch eigensolve + mean IPR |
| `spectrum_chi_squared_gpu` | Pangenome chi-squared |
| `selection_coefficient_gpu` | Pangenome selection coefficient |

### API Changes Absorbed (Upstream S45/S46/S49)

| API | Change |
|-----|--------|
| `solve_f64`, `cholesky_f64`, `gen_eigh_f64` | Now take `Arc<WgpuDevice>` (GPU-first) |
| `HillGateParams` | Removed `_pad3`/`_pad4` (f64 alignment) |

### MHA S-03b Fix Status

**FIXED** upstream in ToadStool S46 (`fe573095`). Z-dimension dispatch bug resolved.
Fix flows to neuralSpring via path dependency. `evolved::mha` remains for now until
full native MHA validation.

---

## Session 48 — Mass Typed Op Rewiring (February 23, 2026)

### 28 Binaries Rewired to Typed BarraCUDA Ops

All 28 binaries migrated from raw wgpu (include_str! local shaders + manual pipeline/
bindgroup/encoder creation) to typed BarraCUDA op APIs. Validates upstream ToadStool/
BarraCUDA APIs directly.

**Key API patterns used**:

| Typed Op | Domain | Validators |
|----------|--------|------------|
| BatchFitnessGpu | Batch fitness (011–015) | validate_gpu_batch_fitness, pipeline_fitness |
| PairwiseHammingGpu | SATé alignment (017) | validate_gpu_sate, pipeline_sate |
| PairwiseJaccardGpu | Pangenome (024) | validate_gpu_pangenome, pipeline_genomics |
| PairwiseL2Gpu | MODES novelty (012) | validate_gpu_modes, pipeline_modes |
| LocusVarianceGpu | Meta-population (025) | validate_gpu_meta_pop, pipeline_meta_pop |
| SpatialPayoffGpu | Game theory (019) | validate_gpu_game_theory, pipeline_ecology |
| MultiObjFitnessGpu | Directed evolution (014) | validate_gpu_directed, pipeline_directed |
| BatchIprGpu | Anderson (022–023) | validate_gpu_anderson, pipeline_spectral |
| SwarmNnGpu | Swarm robotics (015) | validate_gpu_swarm, pipeline_modes |
| WrightFisherGpu (f64) | Pop genetics (024–025) | validate_gpu_wright_fisher, pipeline_wright_fisher |
| StencilCooperationGpu (f64) | Game theory Fermi | validate_gpu_stencil |
| HillGateGpu | Signal (021) | validate_gpu_signal (f32 path; f64 graceful skip) |

### f64 Data Type Changes

f32→f64 alignment with upstream (ToadStool S49): BatchFitnessGpu, LocusVarianceGpu,
MultiObjFitnessGpu, WrightFisherGpu, StencilCooperationGpu, SwarmNnGpu.

### HillGateGpu f64 Graceful Skip Pattern

On RTX 4070, HillGateGpu f64 triggers a driver limitation. Validators skip f64 path
gracefully; f32 path remains validated.

### Validation Score: 132/133

Only pre-existing `validate_barracuda_logsumexp` driver issue remains. All other
validators PASS.

---

*Barracuda usage audit — neuralSpring, February 23, 2026. Phase 5e: bC 24/25, gT 23/25, xD 15/15, mG 132/133 (RTX 4070 + TITAN V NVK; logsumexp driver issue). Session 48: 28 binaries rewired raw wgpu → typed ops. Session 47: typed op migration, MHA fix, HMM retirement. Sessions 45–46: pure GPU promotion. Session 44: multi-GPU, 178.5×. Session 43: upstream expansion, mixed-hardware. Session 42: deep audit.*
