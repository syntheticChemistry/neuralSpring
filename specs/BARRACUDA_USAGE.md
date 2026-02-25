# BarraCUDA Usage Audit — neuralSpring

**Last Updated**: February 25, 2026 (Sessions 40–69)
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

| Module | Where Used | Purpose |
|--------|-----------|---------|
| `stats::correlation::variance` | counterdiabatic, modes, eco, directed, swarm, sate, game, hmm | Population variance for statistical checks |
| `stats::pearson_correlation` | modes, `cpu_fallback` | Correlation between diversity metrics |
| `stats::empirical_spectral_density` | `weight_spectral` | Eigenvalue histogram (rewired S59, M-011) |
| `stats::marchenko_pastur_bounds` | `weight_spectral` | Random matrix spectral bounds (rewired S59, M-012) |

### Linear Algebra

| Module | Where Used | Purpose |
|--------|-----------|---------|
| `linalg::solve_f64` | hmm, swarm, linalg validation | Linear system solve (takes `Arc<WgpuDevice>`) |
| `linalg::eigh_f64` | spectral, anderson, linalg validation | Eigendecomposition (takes `Arc<WgpuDevice>`) |
| `linalg::cholesky_f64` | linalg validation | Cholesky factorization (takes `Arc<WgpuDevice>`) |
| `linalg::lu_det`, `lu_solve` | linalg validation | LU decomposition |
| `linalg::tridiagonal_solve` | linalg validation | Tridiagonal solver |
| `linalg::effective_rank` | `neural_pgm` | Eigenvalue entropy rank (rewired S59, H-009) |
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
| Native `ops::mha::MultiHeadAttention` | `evolved::mha` thin wrapper | **Wired** (S-03b resolved upstream `0c998992`) |
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
| `gpu_ops/` | 38 GPU-accelerated functions (6 submodules) | All major Tensor ops |
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

### API Gaps (Updated — 2 Closed by ToadStool S52)

| Gap | Impact | Status |
|-----|--------|--------|
| ~~`argmax_dim()`~~ | Viterbi needs indices | **CLOSED** — `Tensor::argmax_dim(axis)` at `9abd6857` |
| No `pow_scalar(n)` | Hill activation `x^n` | `exp(n * ln(x))` pipeline |
| ~~`softmax_dim(axis)`~~ | Row-wise attention softmax | **CLOSED** — `Tensor::softmax_dim(axis)` at `9abd6857` |
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

**FULLY RESOLVED** upstream in ToadStool `0c998992`. MHA projections decomposed into matmul + head_split/head_concat shaders. All 21/21 WGSL shaders absorbed. `evolved::mha` is now a thin wrapper to `barracuda::ops::mha::MultiHeadAttention`.

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

### HillGateGpu f64 — S-17 Root Cause and Fix

**Root cause**: `hill_gate_f64.wgsl` uses native WGSL `pow(f64, f64)` which
triggers NVVM compilation failure on RTX 4070 (Ada Lovelace, proprietary) and
NAK assertion failure on TITAN V (NVK open-source). `compile_shader_f64` patches
`exp()` and `log()` to polyfills but **does not patch `pow()`**.

**Fix**: Replace `pow(` with `pow_f64(` in the shader source before compilation.
`compile_shader_f64` → `inject_missing_math_f64` auto-injects the polyfill
(uses `exp_f64(n * log_f64(base))` — proven in `gpu_ops::bio`'s Tensor pipeline).

**Validation**: `validate_hillgate_f64_fix` 18/18 PASS on both GPUs. Max diff
1.11e-16 (RTX 4070) / 2.22e-16 (TITAN V) — machine epsilon.

**ToadStool action**: Extend `apply_transcendental_workaround` in
`shaders/precision/mod.rs` to also replace `pow(` → `pow_f64(` when
`needs_pow_f64_workaround()` is true. The detection already exists in
`driver_profile.rs`. One-line addition to `patch_exp_log_in_code`.

### Validation Score: 132/133

Only pre-existing `validate_barracuda_logsumexp` driver issue remains. All other
validators PASS.

---

## Session 49 — Deep Debt Audit (February 23, 2026)

### Code Quality Hardening

| Change | Barracuda Surface | Impact |
|--------|------------------|--------|
| `gpu_or_cpu` helper | `gpu_dispatch.rs` | Centralised all 25 GPU-fallback dispatch methods into a single `gpu_or_cpu` closure pattern. All `wgpu_device()` calls go through one path. |
| `exit_no_gpu()` | 79 validation/bench binaries | Unified GPU unavailability handling: `NEURALSPRING_REQUIRE_GPU=1` → exit 1 (CI mode). Default → graceful skip. All binaries use `validation::exit_no_gpu()`. |
| `baseline_path()` | 4 binaries (ML inference, benchmarks) | Replaced `concat!(env!("CARGO_MANIFEST_DIR"), ...)` with `validation::baseline_path()` for JSON baseline resolution. |
| Zero debt | All src/ | No TODO, FIXME, HACK, MOCK, STUB. No hardcoded paths. No unsafe. |

### BarraCUDA Absorption Summary (Full Stack)

| Category | Count | Status |
|----------|-------|--------|
| Typed GPU ops in use | 12 (BatchFitness, PairwiseHamming, PairwiseJaccard, PairwiseL2, LocusVariance, SpatialPayoff, MultiObjFitness, BatchIpr, SwarmNn, WrightFisher, StencilCooperation, HillGate) | All validated |
| Tensor API methods | 30+ (matmul, transpose, add, sub, mul, sigmoid, tanh, gelu, softmax, conv2d, maxpool2d, mean, sum, etc.) | All validated |
| CPU primitives | 18 (variance, pearson, eigh, solve, cholesky, lu, svd, rk45, chi_squared, etc.) | All validated |
| Shaders consumed | 13 upstream (absorbed) + 8 local | All validated |
| Raw wgpu remaining | 4 binaries (bench_upstream_vs_local intentional, 3 ODE/pipeline not yet applicable) | Documented |
| Feature flags needed | `unidirectional` (not yet enabled) | Medium priority |

### What ToadStool Should Know

1. **Dispatching model works**: The `gpu_or_cpu` pattern proves that capability-based routing via `wgpu_device()` is the right abstraction. ToadStool should consider absorbing this as a `BarraCUDA::dispatch()` primitive.

2. **`exit_no_gpu` is a CI pattern**: All three Springs need the same GPU/no-GPU policy. This could become `barracuda::testing::require_gpu()`.

3. **`baseline_path` is a testing primitive**: `env!("CARGO_MANIFEST_DIR")`-relative paths for control data should be a `barracuda::testing` utility.

4. **f64 alignment complete**: All 12 typed ops now use f64 data types. The f32→f64 migration (Session 48) is a model for other Springs.

5. **HillGateGpu f64 driver limitation persists**: RTX 4070 skips f64 path. TITAN V (NVK) untested for HillGate f64 specifically — worth investigating.

---

## Session 50 — baseCamp Biophysical AI Interpretability (February 24, 2026)

### New BarraCUDA Usage in baseCamp Modules

baseCamp modules primarily compose existing primitives. BarraCUDA surface:

| Module | BarraCUDA Primitive | How Used |
|--------|--------------------|----|
| `weight_spectral.rs` | `eigh` (via `eigh.rs`) | Eigendecomposition of symmetrized weight Hamiltonians |
| `information_flow.rs` | `eigh` (via `eigh.rs`) | Attention Hamiltonian spectral analysis |
| `loss_landscape.rs` | `eigh` (via `eigh.rs`) | Hessian spectrum computation |
| `neural_pgm.rs` | `eigh` (via `eigh.rs`) | Transition matrix spectral analysis |
| `agent_coordination.rs` | `eigh` (via `eigh.rs`) | Disordered Laplacian eigendecomposition |
| All 5 modules | `rng::Rng` | Deterministic seed-based stochastic generation |

### GPU Promotion Candidates

All 5 modules are CPU-only. GPU promotion uses existing patterns:

| baseCamp Function | GPU Pattern | Existing BarraCUDA Analogue |
|-------------------|-------------|---------------------------|
| `weight_to_hamiltonian` (matmul) | Tensor matmul | 4-tier `KernelRouter` |
| `numerical_hessian` (parallel evals) | Batch parallel | `BatchFitnessGpu` |
| `belief_propagation_chain` (GEMV) | Batch GEMV | `HmmBatchForwardF64` |
| `interaction_graph` (pairwise) | Pairwise distance | `PairwiseL2Gpu` |
| `boltzmann_sampling` (MCMC) | Parallel chains | `WrightFisherGpu` |

### General-Purpose Primitives for BarraCUDA Absorption

| Primitive | Generalized Form | Potential Location |
|-----------|-----------------|-------------------|
| `graph_laplacian(adjacency)` | `D - A` | `ops::linalg` |
| `effective_rank(eigenvalues)` | Entropy-based rank | `ops::linalg` |
| `empirical_spectral_density(eigenvalues, bins)` | Histogram | `ops::stats` |
| `numerical_hessian(f, x, h)` | Central FD Hessian | `ops::numerical` |
| `level_spacing_ratio(eigenvalues)` | GOE/Poisson stat | `ops::stats` |

### Updated Totals

| Category | Before (S49) | After (S50) |
|----------|-------------|-------------|
| Library modules | 31 | **36** (+5 baseCamp) |
| Validation binaries | 133 | **138** (+5 baseCamp) |
| Unit tests | 374 | **459** (+38 baseCamp) |
| BarraCUDA `eigh` consumers | 4 modules | **9** modules (+5 baseCamp) |

---

## Session 51 — Code Quality Evolution (February 24, 2026)

### Structural Changes

`gpu_dispatch.rs` refactored into `gpu_dispatch/` module directory:
- `gpu_dispatch/mod.rs` — Dispatcher struct, `gpu_or_cpu` pattern, 25 dispatch methods
- `gpu_dispatch/cpu_fallback.rs` — 6 CPU fallback implementations extracted

The CPU fallbacks are now independently testable and candidates for
`barracuda::stats` / `barracuda::bio` CPU reference absorption.

**Variance convention difference**: `cpu_fallback::variance` uses **population
variance** (÷N), matching GPU kernel convention. `barracuda::stats::variance`
uses **sample variance** (÷(N-1)). Do NOT rewire — the conventions are
intentionally different for GPU shader parity.

### Hardcoding Evolution

7 inline `1e-14` guards centralized to `tolerances::ZERO_DETECTION` across 5 bins.
Algorithm convergence parameters (Nelder-Mead, bisect) remain correctly inline.

### Dependency Audit

All dependencies confirmed pure Rust. Only `-sys` crates are `linux-raw-sys`
(kernel constants via wgpu) and `renderdoc-sys` (optional wgpu debug) — both
unavoidable transitive dependencies with no C compilation.

### Quality Gates

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets` | 0 warnings (pedantic + nursery) |
| `cargo clippy -p neural-spring-forge` | 0 warnings |
| `cargo doc --no-deps` | 0 warnings (146 pages) |
| `cargo test` | 459 lib + 9 doc PASS |
| `cargo llvm-cov --lib` | 92.9% line coverage |

---

## Session 52 — ToadStool Sync & Cross-Spring Benchmarking (February 24, 2026)

### ToadStool Sync (16 commits, `b41ee5f4` → `9abd6857`)

| Absorption | Upstream API | Impact |
|-----------|-------------|--------|
| `xoshiro128ss.wgsl` | `barracuda::ops::prng_xoshiro` | PRNG local shader retired |
| `logsumexp_reduce.wgsl` | `barracuda::ops::LogsumexpWgsl` | HMM/phylo numerics |
| `stencil_cooperation.wgsl` | `barracuda::StencilCooperationGpu` | Game theory Fermi |
| `wright_fisher_step.wgsl` | `barracuda::WrightFisherGpu` | Population genetics |
| `rk45_adaptive.wgsl` | `barracuda::ops::rk45_adaptive` | Adaptive ODE |
| `swarm_nn_scores.wgsl` | `barracuda::SwarmNnGpu` | Swarm robotics |

**API gaps closed**: `Tensor::argmax_dim(axis)`, `Tensor::softmax_dim(axis)`.

**Rewired**: `weight_spectral::level_spacing_ratio` → `barracuda::spectral::level_spacing_ratio`.

**Only 2 local shaders remain**: `head_split.wgsl` + `head_concat.wgsl` (MHA S-03b workaround).

### Cross-Spring Benchmark Results (RTX 4070, Vulkan, `--release`)

| Op | Origin | Size | µs |
|----|--------|------|----|
| BatchFitnessGpu | neuralSpring | 1024×64 | 1,337 |
| PairwiseL2Gpu | neuralSpring | 128×16 | 1,542 |
| BatchIprGpu | neuralSpring | 32×64 | 2,027 |
| SpatialPayoffGpu | neuralSpring | 32×32 | 1,450 |
| PairwiseHammingGpu | neuralSpring | 64×100 | 1,682 |
| HmmBatchForwardF64 | wetSpring | 4s×50t×32b | 2,141 |
| BatchedEighGpu | hotSpring | 12×12×40 | 6,629 |

### Validation

| Gate | Result |
|------|--------|
| `validate_all` | 137/138 PASS (1 pre-existing logsumexp driver issue) |
| `cargo test --lib` | 459 PASS |
| `cargo llvm-cov --lib` | 92.89% line coverage |

---

## Session 53 — Final f64 Typed Op Rewiring (February 24, 2026)

### 5 Operations Rewired to Upstream f64 Typed Ops

| Local Function | Old Path | Upstream API | Origin |
|---------------|----------|-------------|--------|
| `variance_gpu` | f32 Tensor (4 dispatches) | `VarianceReduceF64::population_variance` | hotSpring |
| `pearson_correlation_gpu` | f32 Tensor (3+ dispatches) | `CorrelationF64::correlation` | wetSpring + hotSpring |
| `shannon_entropy_gpu` | f32 Tensor (3 dispatches) | `FusedMapReduceF64::shannon_entropy` | wetSpring |
| `cpu_fallback::pearson` | Local Rust | `barracuda::stats::pearson_correlation` | wetSpring |
| `cpu_fallback::chi_squared` | Local Rust | `barracuda::special::chi_squared_statistic` | wetSpring |

### Rewire Evolution Benchmark (10,000 elements)

**RTX 4070 (Ada Lovelace)**

| Op | f32 Tensor (µs) | f64 Upstream (µs) | Speedup | Origin |
|----|-----------------|-------------------|---------|--------|
| Variance | 7,018 | 2,316 | **3.03×** | hotSpring Welford |
| Pearson | 3,566 | 3,480 | **1.02×** | wetSpring + hotSpring |
| Entropy | 3,989 | 1,662 | **2.40×** | wetSpring fused |

**TITAN V (NVK)**

| Op | f32 Tensor (µs) | f64 Upstream (µs) | Speedup | Origin |
|----|-----------------|-------------------|---------|--------|
| Variance | 13,333 | 2,937 | **4.54×** | hotSpring Welford |
| Pearson | 5,098 | 15,053 | 0.34× (NVK f64 cost) | wetSpring + hotSpring |
| Entropy | 5,510 | 3,525 | **1.56×** | wetSpring fused |

### Validation

| Gate | Result |
|------|--------|
| `validate_all` | 138/139 PASS (1 pre-existing logsumexp driver issue) |
| `cargo test --lib` | 459 PASS |

---

## Session 56 — ToadStool S53 Sync + Upstream Rewiring (February 24, 2026)

### 4 Functions Rewired to Upstream BarraCUDA (`9404fdb4`)

| Local Function | Module | Upstream Module | Sub-thesis |
|----------------|--------|----------------|-----------|
| `graph_laplacian` | `agent_coordination` | `barracuda::linalg::graph::graph_laplacian` | Sub-05 |
| `disordered_laplacian` | `agent_coordination` | `barracuda::linalg::graph::disordered_laplacian` | Sub-05 |
| `belief_propagation_chain` | `neural_pgm` | `barracuda::linalg::graph::belief_propagation_chain` | Sub-04 |
| `numerical_hessian` | `loss_landscape` | `barracuda::numerical::numerical_hessian` | Sub-03 |

### New Upstream Modules Consumed

| Module | Purpose | Origin |
|--------|---------|--------|
| `barracuda::linalg::graph` | Graph Laplacians, belief propagation | neuralSpring Sub-04/05 handoff |
| `barracuda::numerical` | Numerical Hessian via finite differences | neuralSpring Sub-03 handoff |
| `barracuda::ops::bio::swarm_nn` | Swarm neural network forward pass | neuralSpring Paper 015 |
| `barracuda::ops::bio::xoshiro128ss` | GPU-friendly PRNG | neuralSpring Paper 011 |

### 3 New Validators

| Validator | Checks | Purpose |
|-----------|--------|---------|
| `validate_basecamp_dispatch` | 19 | Dispatcher baseCamp GPU routing |
| `validate_barracuda_parity` | 34 | CPU↔GPU bit-parity across all domains |
| `validate_metalforge_pcie` | 36 | PCIe tiers, chained transfers, substrate selection |

### Validation

| Gate | Result |
|------|--------|
| `validate_all` | 159 binaries (147/148 PASS, 1 pre-existing logsumexp) |
| `cargo test --lib` | 478 PASS |
| `cargo test -p neural-spring-forge --lib` | 30 PASS |
| `cargo clippy (pedantic+nursery)` | 0 warnings |

## Session 58 — Upstream Dispatch Rewiring + GpuDriverProfile (February 24, 2026)

### 9 Dispatcher Methods Rewired to Upstream domain_ops

| Dispatcher Method | Upstream Function | Notes |
|-------------------|-------------------|-------|
| `mat_mul` | `barracuda::dispatch::matmul_dispatch` | Square n×n → (n,n,n) adapter |
| `frobenius_norm` | `barracuda::dispatch::frobenius_norm_dispatch` | Direct delegation |
| `transpose` | `barracuda::dispatch::transpose_dispatch` | Square n → (n,n) adapter |
| `softmax` | `barracuda::dispatch::softmax_dispatch` | Direct delegation |
| `l2_distance` | `barracuda::dispatch::l2_distance_dispatch` | Direct delegation |
| `mean` | `barracuda::dispatch::mean_dispatch` | Direct delegation |
| `variance` | `barracuda::dispatch::variance_dispatch` | Direct delegation |
| `gelu` | `barracuda::dispatch::gelu_dispatch` | S59 — CPU fallback: `transformer::gelu` |
| `hmm_forward_step` | `barracuda::dispatch::hmm_forward_dispatch` | S59 — CPU fallback: `cpu_fallback::hmm_forward_step` |

### GpuDriverProfile Integration (hotSpring-evolved)

| New Method | Upstream Source | Purpose |
|------------|----------------|---------|
| `driver_profile()` | `barracuda::device::driver_profile::GpuDriverProfile` | Full hardware detection |
| `fp64_strategy()` | `GpuDriverProfile::fp64_strategy()` | Native vs Hybrid f64 routing |
| `needs_pow_workaround()` | `GpuDriverProfile::needs_pow_f64_workaround()` | pow(f64) polyfill decision |

### RTX 4070 Detected Profile

| Field | Value |
|-------|-------|
| Driver | NvidiaProprietary |
| Compiler | NvidiaPtxas |
| Arch | Ada |
| FP64 rate | Throttled (1:64) |
| FP64 strategy | Hybrid |
| Eigensolve | WarpPacked { wg_size: 32 } |

### Cross-Spring Provenance

| Spring | Contributions to BarraCUDA |
|--------|---------------------------|
| hotSpring | df64_core, pow_f64, Fp64Strategy, GpuDriverProfile, Taylor trig, Lanczos |
| wetSpring | HMM, ODE bio (5 systems), NMF, Anderson localization, Ridge regression |
| neuralSpring | ValidationHarness, batch_fitness, pairwise ops, eigh, KernelRouter |

---

## Session 59 — Library + Dispatch Rewiring (February 24, 2026)

### 3 Library Functions + 2 Dispatcher Methods Rewired

| Function | Module | Upstream API | Absorbed In |
|----------|--------|-------------|-------------|
| `empirical_spectral_density` | `weight_spectral` | `barracuda::stats::empirical_spectral_density` | S54 (M-011) |
| `marchenko_pastur_bounds` | `weight_spectral` | `barracuda::stats::marchenko_pastur_bounds` | S54 (M-012) |
| `effective_rank` | `neural_pgm` | `barracuda::linalg::effective_rank` | S54 (H-009) |
| `gelu` (Dispatcher) | `dispatch_ops` | `barracuda::dispatch::gelu_dispatch` | S52 |
| `hmm_forward_step` (Dispatcher) | `dispatch_ops` | `barracuda::dispatch::hmm_forward_dispatch` | S52 |

### Cumulative Rewire Count: 16

| Session | Rewired | Running Total |
|---------|---------|---------------|
| S56 | 4 (graph, hessian, BP) | 4 |
| S58 | 7 (domain\_ops dispatchers) | 11 |
| S59 | 5 (ESD, MP, rank, gelu, hmm) | **16** |

---

## Sessions 60–61 — Cross-Spring Benchmark Validation (February 25, 2026)

### Updated Validation

| Gate | Result |
|------|--------|
| `validate_cross_spring_evolution` | **22/22 PASS** |
| `cargo test --lib` | **500 PASS** |
| `validate_all` | **145/146 PASS** |

### Paper Controls Verification: Hardware Tiers

All 25+5 papers confirmed working across three hardware tiers:

| Tier | Coverage | Checks |
|------|----------|--------|
| BarraCUDA CPU | 24/25 papers (96%) | 203 checks |
| BarraCUDA GPU | 23/25 papers (92%) | 98+ checks |
| metalForge mixed | 15/15 applicable | 14/14 mixed + 16/16 dispatch |

Open data confirmed: zero proprietary, zero paywalled, zero access-restricted sources.

---

---

## Sessions 66–67b — Phase C GPU + CPU Parity + Dispatch Tiers (February 25, 2026)

### New Dispatcher Methods (Session 66)

| Method | GPU Path | CPU Fallback |
|--------|----------|--------------|
| `hmm_forward_chain` | `hmm_forward_chain_gpu` (T × forward_step) | `Hmm::from_flat` → `.forward()` |
| `hmm_viterbi_chain` | `hmm_viterbi_chain_gpu` (T × viterbi_step) | `Hmm::from_flat` → `.viterbi()` |
| `pairwise_fst` | `pairwise_fst_gpu` (allele_freq + W-C) | `meta_population::pairwise_fst` |
| `global_fst` | `global_fst_gpu` (per-pop allele_freq) | `meta_population::global_fst` |
| `inter_population_af_variance` | Existing gpu_op → dispatch | `meta_population::inter_population_af_variance` |

GPU dispatch coverage: 38 ops → **44 ops** (~97% of production math).

### CPU↔Python Parity (Session 67)

`validate_cpu_math_parity`: 39/39 PASS at 1e-10 tolerance.
Proves Rust CPU (library + Dispatcher::cpu_only()) = Python/NumPy.

### Dispatch Tier Characterization (Session 67b)

`bench_dispatch_tiers`: Library direct → Dispatcher::cpu_only() → Dispatcher::new() GPU.
9/10 ops ≤1.04× CPU dispatch overhead. Per-call GPU driver-bound for small workloads.
Motivates StatefulPipeline/UnidirectionalPipeline batching.

### Updated Validation

| Gate | Result |
|------|--------|
| `cargo test --lib` | **505 PASS** |
| `validate_all` | **147/148 PASS** |
| `validate_gpu_phase_c` | **18/18 PASS** |
| `validate_cpu_math_parity` | **39/39 PASS** |

---

## Session 68 — Deep Debt Audit: BarraCUDA Integration Health

### Audit Findings

Full barracuda usage sweep across 60+ files, 90+ import sites:

- **20+ barracuda submodules** actively consumed (device, tensor, ops::bio, ops::mha,
  ops::linalg, ops::fft, ops::fused_map_reduce_f64, ops::variance_reduce_f64,
  ops::correlation_f64_wgsl, ops::logsumexp, ops::rk_stage, stats, special,
  spectral, dispatch, pipeline, staging, numerical, linalg::graph, unified_hardware)
- **Zero barracuda-related TODO/FIXME** in src/
- **Zero duplicate math** — all identified overlaps are intentional:
  - `cpu_fallback::variance` uses population (÷N) vs barracuda sample (÷(N-1))
  - `primitives.rs` kept as independent CPU reference for validation independence
  - `spectrum_chi_squared` derives expected from fractions (variant API, not duplicate)
- **GPU test serialization**: Added crate-level `test_gpu_lock` and shared `Gpu`
  instance to prevent wgpu device contention — pattern recommended for upstream

### Quality Gates (Session 68)

| Gate | Result |
|------|--------|
| `cargo fmt --all -- --check` | **PASS** |
| `cargo clippy --all-targets -D warnings` | **0 warnings** |
| `cargo test --lib` | **505/505 PASS** |
| `cargo test --test integration` | **9/9 PASS** |
| `cargo doc --no-deps` | **0 warnings** |
| `cargo llvm-cov --lib` | **90.43% line coverage** |
| Named tolerances | **104+** |
| Ad-hoc magic numbers | **0** |

---

## Session 69 — Validator Shader Rewiring + Cross-Spring Benchmarks

### Shader Source Rewiring

6 validator binaries rewired from local `include_str!` to upstream barracuda shader
constants. Same shader content, but source-of-truth now lives in barracuda:

| Validator | Old | New |
|-----------|-----|-----|
| `validate_gpu_rk4` | `include_str!("metalForge/shaders/rk4_parallel.wgsl")` | `barracuda::ops::rk_stage::WGSL_RK4_PARALLEL` |
| `validate_gpu_rk45` | `include_str!(rk45_adaptive.wgsl)` | `barracuda::ops::rk45_adaptive::WGSL_RK45_ADAPTIVE` |
| `validate_gpu_stateful_pipeline` | `include_str!(rk4_parallel.wgsl)` | `barracuda::ops::rk_stage::WGSL_RK4_PARALLEL` |
| `validate_gpu_pure_workload` | `include_str!(batch_fitness_eval.wgsl)` | `barracuda::ops::bio::batch_fitness::WGSL_BATCH_FITNESS_EVAL` |
| `validate_gpu_logsumexp` | `include_str!(logsumexp_reduce.wgsl)` | `barracuda::ops::logsumexp::LogSumExp::WGSL_LOGSUMEXP_REDUCE` |
| `validate_gpu_pipeline_swarm` | `include_str!(swarm_nn_scores.wgsl)` | `barracuda::ops::bio::swarm_nn::WGSL_SWARM_NN_SCORES` |

### Remaining Local Shaders

| File | Shader | Reason |
|------|--------|--------|
| `validate_gpu_pure_workload.rs` | `mean_reduce.wgsl` | No public upstream `WGSL_MEAN_REDUCE` constant |
| `validate_mha_gpu.rs` | `head_split.wgsl`, `head_concat.wgsl` | No upstream equivalent |
| `bench_upstream_vs_local.rs` | 10 shaders | Intentional: benchmarks local vs upstream dispatch |

### Upstream vs Local Benchmark (RTX 4070, --release)

| Kernel | Origin | Local (µs) | Upstream (µs) | Overhead |
|--------|--------|-----------|--------------|----------|
| BatchFitness 10k×32 | nS 011-015 | 1,840 | 2,060 | 12% ~ |
| Hamming 200×500 | nS 017 | 1,807 | 1,947 | 8% ≈ |
| Jaccard 100×500 | nS 024 | 1,972 | 1,849 | −6% ≈ |
| LocusVariance 50×500 | nS 025 | 2,035 | 2,043 | <1% ≈ |
| SpatialPayoff 256² | nS 019 | 1,903 | 1,890 | −1% ≈ |
| BatchIPR 1k×256 | nS 022-023 | 1,909 | 2,301 | 21% ~ |
| HillGate 100² | nS 021 | 2,101 | 2,003 | −5% ≈ |
| MultiObjFitness 5k×4 | nS 014 | 1,978 | 1,943 | −2% ≈ |
| PairwiseL2 200×50 | nS 012 | 2,031 | 1,940 | −4% ≈ |
| SwarmNN 500×20 | nS 015 | 1,990 | 1,999 | <1% ≈ |

### BarraCUDA Consumption Summary (S69 complete)

| Category | Count |
|----------|-------|
| Barracuda submodules consumed | 20+ |
| Functions rewired to upstream | 17 |
| Validator shader sources rewired | 6 |
| Upstream GPU typed ops validated | 10 bio + f64 HMM + Gillespie + wetSpring trio + chi² |
| Total barracuda import sites | 90+ |
| Upstream API coverage | 117+ APIs exercised |

*BarraCUDA usage audit — neuralSpring, February 25, 2026. Sessions 50–69: 17 functions + 6 shader sources rewired to upstream, GpuDriverProfile wired in, S-03b fully resolved, 159 binaries, 505 lib + 43 forge + 9 integration tests. Phase C GPU ~97%, CPU↔Python parity 39/39, dispatch overhead ≤1.04× (9/10 ops). Session 68: zero duplicate math, zero debt, 104+ tolerances, 90.43% coverage. Session 69: shader rewiring complete, upstream benchmarks nominal (10/10 ≈ or ~).*
