# neuralSpring Absorption Manifest

**Parent**: ecoPrimals/neuralSpring
**License**: AGPL-3.0-or-later
**Pattern**: Evolve locally → validate → handoff → ToadStool absorbs → retire
**ToadStool HEAD**: `02207c4a` (Sessions 50–70, Feb 25, 2026)
**Last Updated**: February 25, 2026 (Sessions 60–70 — forge v0.2.0 with substrate/probe/inventory/workloads, 2 write-phase WGSL extensions, 23 shaders, 43 forge tests)

---

## Already Absorbed by ToadStool

S-01 through S-12 — all resolved in ToadStool `77f70b2e`. Local workarounds
fossilized in `metalForge/fossils/evolved_s01_s11/` (~3.4k LOC, incl. eigh_local.rs).

| Shortcoming | Fix | ToadStool Commit | Replacement API |
|-------------|-----|-----------------|-----------------|
| S-01 Per-op dispatch | `TensorSession` single-encoder | `fbedd222` | `TensorSession::run()` |
| S-02 Naive matmul | 4-tier `KernelRouter` | `82f953c8` | `ops::matmul` CpuTiled32/GpuEvolved32 |
| S-03 MHA z-dispatch | `workgroups_z = seq_len` | `dc540afd` | Native MHA |
| S-04 Softmax pooled | `params.size` uniform | `dc540afd` | `Tensor::softmax_wgsl()` |
| S-05 leaky_relu Params | `{size, negative_slope}` | `dc540afd` | `Tensor::leaky_relu_wgsl()` |
| S-06 elu Params | `{size, alpha}` | `dc540afd` | `Tensor::elu_wgsl()` |
| S-07 from_buffer pub | `pub fn from_buffer()` | `dc540afd` | `Tensor::from_buffer()` |
| S-08 layer_norm round-trip | `from_pooled_buffer` | `81a6fd4b` | `Tensor::layer_norm_wgsl()` |
| S-09 log_softmax round-trip | `from_pooled_buffer` | `81a6fd4b` | `Tensor::log_softmax_wgsl()` |
| S-10 science_limits CPU | `new_cpu_relaxed()` | `dc540afd` | `Gpu::new_cpu_relaxed()` |
| S-11 TensorSession limited | ML ops in SessionOp | `fbedd222` | `TensorSession::{matmul, relu, gelu}` |
| S-12 eigh_f64 accuracy | Householder+QR eigensolver | `77f70b2e` | `barracuda::ops::linalg::eigh_householder_qr` |

---

## Absorbed Shaders (Tier A — validated, upstream)

### Identical copies (absorbed at `77f70b2e`)

| WGSL Shader | Upstream API | Domain | Validation | Checks |
|-------------|-------------|--------|------------|--------|
| `hmm_forward_log.wgsl` | `barracuda::ops::bio::hmm` | Phylogenetics 016–018 | `validate_gpu_hmm_forward` | 13/13 |
| `pairwise_jaccard.wgsl` | `barracuda::ops::bio::pairwise_jaccard` | Pangenome 024 | `validate_gpu_pangenome` | 6/6 |
| `locus_variance.wgsl` | `barracuda::ops::bio::locus_variance` | Meta-pop 025 | `validate_gpu_meta_pop` | 7/7 |
| `spatial_payoff.wgsl` | `barracuda::ops::bio::spatial_payoff` | Game theory 019 | `validate_gpu_game_theory` | 5/5 |
| `batch_ipr.wgsl` | `barracuda::spectral::batch_ipr` | Spectral 022–023 | `validate_gpu_anderson` | 5/5 |
| `pairwise_hamming.wgsl` | `barracuda::ops::bio::pairwise_hamming` | Alignment 017 | `validate_gpu_sate` | 5/5 |

### Generalized variants (absorbed at `5437c170` Session 42)

Local copies retained for validation compatibility (different binding layouts).

| WGSL Shader | Upstream Path | Validation | Checks | Key Difference |
|-------------|---------------|------------|--------|----------------|
| `pairwise_l2.wgsl` | `shaders::math::pairwise_l2` | `validate_gpu_modes` | 15/15 | O(1) pair decode |
| `multi_obj_fitness.wgsl` | `shaders::bio::multi_obj_fitness` | `validate_gpu_directed` | 6/6 | Bessel correction |
| `hill_gate.wgsl` | `shaders::bio::hill_gate` | `validate_gpu_signal` | 9/9 | Mode generalization |
| `swarm_nn_forward.wgsl` | `shaders::bio::swarm_nn_forward` | `validate_gpu_swarm` | 9/9 | Generic MLP dims |
| `mean_reduce.wgsl` | `shaders::reduce::mean_reduce` | `validate_gpu_pure_workload` | 7/7 | Identical |

### Upstream rewiring (Session 56 — ToadStool `9404fdb4`)

neuralSpring baseCamp functions now delegate to upstream BarraCUDA modules
(ToadStool Sessions 51–53 absorbed these from our handoffs):

| Local Function | Upstream Delegation | Module | Checks |
|----------------|-------------------|--------|--------|
| `agent_coordination::graph_laplacian` | `barracuda::linalg::graph::graph_laplacian` | Sub-05 | 23/23 PASS |
| `agent_coordination::disordered_laplacian` | `barracuda::linalg::graph::disordered_laplacian` | Sub-05 | 23/23 PASS |
| `neural_pgm::belief_propagation_chain` | `barracuda::linalg::graph::belief_propagation_chain` | Sub-04 | 21/21 PASS |
| `loss_landscape::numerical_hessian` | `barracuda::numerical::numerical_hessian` | Sub-03 | 27/27 PASS |

New upstream capabilities available (not yet wired):

| Upstream Module | API | Potential Use |
|----------------|-----|---------------|
| `barracuda::linalg::graph::effective_rank` | `effective_rank(eigenvalues)` | Weight spectral analysis |
| `barracuda::stats::spectral_density` | `empirical_spectral_density`, `marchenko_pastur_bounds` | Weight matrix RMT |
| `barracuda::sample::metropolis` | `boltzmann_sampling` | Loss landscape MCMC exploration |
| `barracuda::numerical::WGSL_HESSIAN_COLUMN` | GPU Hessian column shader | GPU-accelerated Hessian |
| `barracuda::shaders::linalg::laplacian.wgsl` | GPU graph Laplacian | GPU-accelerated Sub-05 |
| `barracuda::shaders::linalg::symmetrize.wgsl` | GPU symmetrization | Hessian/adjacency cleanup |

### Newly absorbed (ToadStool S51–S52)

| WGSL Shader | Upstream API | Absorption Session |
|-------------|-------------|-------------------|
| `xoshiro128ss.wgsl` | `barracuda::ops::prng_xoshiro` | S51 (H-004) |
| `logsumexp_reduce.wgsl` | `barracuda::ops::LogsumexpWgsl` | S51 (H-004) |
| `stencil_cooperation.wgsl` | `barracuda::StencilCooperationGpu` | S52 |
| `wright_fisher_step.wgsl` | `barracuda::WrightFisherGpu` | S52 |
| `rk45_adaptive.wgsl` | `barracuda::ops::rk45_adaptive` | S51 |
| `swarm_nn_scores.wgsl` | `barracuda::SwarmNnGpu` | S52 (L-009) |

### Still local (pending absorption)

| WGSL Shader | Library Export | Domain | Validation | Checks | Absorption Target |
|-------------|--------------|--------|------------|--------|-------------------|
| `head_split.wgsl` | `evolved::WGSL_HEAD_SPLIT` | MHA (S-03b) | `validate_mha_gpu` | 5/5 | `barracuda::ops::mha` |
| `head_concat.wgsl` | `evolved::WGSL_HEAD_CONCAT` | MHA (S-03b) | `validate_mha_gpu` | 5/5 | `barracuda::ops::mha` |

### Cross-Dispatch Validators

| Binary | Shaders | Papers | Checks | Status |
|--------|---------|--------|--------|--------|
| `validate_cross_dispatch` | EA fitness routing | — | 8 | 8/8 PASS |
| `validate_cross_dispatch_genomics` | Jaccard + variance | 024, 025 | 8 | 8/8 PASS |
| `validate_cross_dispatch_extended` | Hamming, spatial_payoff, batch_ipr | 017, 019, 022–023 | 12 | 12/12 PASS |
| `validate_cross_dispatch_phase4e` | pairwise_l2, multi_obj_fitness, swarm_nn_forward, hill_gate | 012, 014, 015, 021 | 13 | 13/13 PASS |

### BarraCUDA GPU Tensor Validation

| Binary | Domain | Checks | Status |
|--------|--------|--------|--------|
| `validate_barracuda_gpu_spectral` | GPU Tensor matmul for commutator (Paper 022) | 8 | **8/8 PASS** |
| `validate_barracuda_gpu_eco` | GPU Tensor matmul for eco dynamics (Paper 013) | 6 | **6/6 PASS** |

### Cross-Domain Shaders (multi-paper — also absorbed at 77f70b2e)

| WGSL Shader | Upstream API | Domain | Validation | Checks |
|-------------|-------------|--------|------------|--------|
| `batch_fitness_eval.wgsl` | `barracuda::ops::bio::batch_fitness` | Evolution 011–015 | `validate_gpu_batch_fitness` | 20/20 |
| `rk4_parallel.wgsl` | `barracuda::ops::rk_stage` | Regulatory 020–021 | `validate_gpu_rk4` | 8/8 |

---

## Ready for Absorption (Tier B — validated GPU pipelines)

Pure GPU end-to-end pipelines that chain domain shaders with `mean_reduce.wgsl`
in a single `wgpu::CommandEncoder`. No CPU readback until final scalar.

| Pipeline | Shaders Chained | Validation | Checks | Status |
|----------|----------------|------------|--------|--------|
| HMM → mean | `hmm_forward_log` + `mean_reduce` | `validate_gpu_pipeline_hmm` | 5/5 | PASS |
| Ecology → mean | `spatial_payoff` + `mean_reduce` | `validate_gpu_pipeline_ecology` | 5/5 | PASS |
| Spectral → mean | `batch_ipr` + `mean_reduce` | `validate_gpu_pipeline_spectral` | 5/5 | PASS |
| Genomics → mean | `pairwise_jaccard` + `mean_reduce` | `validate_gpu_pipeline_genomics` | 5/5 | PASS |
| MODES L2 → mean | `pairwise_l2` + `mean_reduce` | `validate_gpu_pipeline_modes` | 4/4 | PASS |
| Directed → mean | `multi_obj_fitness` + `mean_reduce` | `validate_gpu_pipeline_directed` | 4/4 | PASS |
| Signal → mean | `hill_gate` + `mean_reduce` | `validate_gpu_pipeline_signal` | 4/4 | PASS |

---

## Stays Local (neuralSpring-specific)

These components are neuralSpring test infrastructure; not candidates for
ToadStool absorption.

| Component | Purpose | LOC |
|-----------|---------|-----|
| `validation.rs` | `ValidationHarness` pass/fail framework | ~120 |
| `tolerances/` | Centralized tolerance constants + runtime introspection (20+ named) | ~1037 |
| `provenance.rs` | Python baseline metadata | ~80 |
| `metrics.rs` | R², RMSE, MAE, NSE | ~150 |
| `fft.rs` | Analytical DFT reference values | ~100 |
| `eigh.rs` | Eigensolver → delegates to `barracuda` (S-12 absorbed) | ~40 |
| 142 validation binaries | Correctness proof suite | ~9k |
| 5 benchmark binaries | Performance comparison suite | ~1k |

---

## Active Evolutions (not yet absorbed)

| Module | LOC | Issue | Status | Path to Absorption |
|--------|-----|-------|--------|-------------------|
| `evolved::mha` | 182 | S-03b: native projection shaders hang | Active in `src/evolved/` | ToadStool: matmul + head_split/head_concat WGSL |
| `evolved::hmm_forward_gpu` | 270 | No `barracuda::ops::hmm` | Active in `src/evolved/` | ToadStool: new `ops::hmm` op |

---

## GPU-Ready Module Layouts

All domain modules use flat row-major `Vec<f64>` or `Vec<u8>` layouts
that match GPU buffer bindings directly:

| Module | Layout | WGSL Buffer Match |
|--------|--------|-------------------|
| `hmm.rs` | Flat `Vec<f64>` (T×N) | `hmm_forward_log.wgsl @binding(2)` |
| `spectral_commutativity.rs` | Flat `Vec<f64>` (N×N) | `barracuda::ops::matmul` |
| `directed_evolution.rs` | Flat `Vec<f64>` (pop×genome) | `multi_obj_fitness.wgsl @binding(0)` |
| `sate_alignment.rs` | Flat `Vec<u8>` (n×len) | `pairwise_hamming.wgsl @binding(0)` |
| `anderson_localization.rs` | Flat `Vec<f64>` (N×N) | `batch_ipr.wgsl @binding(0)` |
| `pinn.rs` | Scalar + flat grid | `barracuda::tensor` matmul buffer |
| `deeponet.rs` | Scalar + flat grid | `barracuda::tensor` matmul buffer |
| `primitives.rs` | Centralized constants | Shader `const` declarations |

---

## Planned Shaders (not yet implemented)

| Shader | Domain | Priority | Dependency |
|--------|--------|----------|------------|
| `tridiag_eigensolver.wgsl` | Spectral 022–023 | P3 | Householder → bisection design |
| `logsumexp_reduce.wgsl` | HMM/phylogenetics | P2 | Complements `hmm_forward_log.wgsl` |

---

## BarraCUDA APIs Used

| Category | APIs | Checks |
|----------|------|--------|
| Statistics | `variance`, `pearson_correlation`, `covariance`, `norm_cdf` | 13 |
| Linear Algebra | `solve_f64`, `eigh_f64`, `cholesky_f64`, `lu_*`, `tridiag`, `svd_*`, `gen_eigh_f64` | 34 |
| Special Functions | `gamma`, `erf`, `bessel_*`, `legendre`, `hermite`, `laguerre`, `chi_squared_*` | 26 |
| Numerical | `rk45_solve` | 10 |
| Optimization | `nelder_mead`, `bisect`, `brent` | 10 |
| Tensor (f32) | 90 ops via `Tensor` API | 90 |
| Tensor (f64) | 7 ops via f64 GPU reductions | 35 |
| Precision | CPU f64 shaders | 12 |
| Quantized | Q4/Q8 dequant + GEMV | 15 |
| ML Inference | MLP + Transformer end-to-end | 13 |
| FFT | f32/f64/Rfft, Parseval, inverse | 12 |
| LogSumExp | log-domain stability | 5 |

**Total BarraCUDA primitive checks**: 275
**Total BarraCUDA CPU port checks**: 170 (17 modules)
**Total BarraCUDA GPU Tensor validation**: 14 (spectral 8, eco 6)
**Total GPU shader checks**: 108 (17 WGSL — 13 upstream, 4 local)
**Total GPU pipeline checks**: 32 (7 pipelines)
**Total cross-dispatch checks**: 41 (8+8+12+13)
**Total dispatch + parity checks**: 89 (16+14+19+17+23, Session 55–56)
**Total lib tests**: 505 lib + 9 integration + 43 forge tests
**Upstream rewired**: 4 functions delegating to `barracuda::linalg::graph` + `barracuda::numerical`
**Grand total validation**: 2010+ (206 Python + 1810+ Rust+GPU)

---

## BarraCUDA API Usage Summary

| Category | APIs Used | Source Files |
|----------|----------|-------------|
| **Device** | `WgpuDevice`, `WgpuDevice::new_cpu_relaxed()` | `gpu.rs`, 15+ validation binaries |
| **Tensor** | `Tensor::from_data`, `matmul`, `relu`, `gelu`, `softmax_wgsl`, `layer_norm_wgsl`, `log_softmax_wgsl`, `leaky_relu_wgsl`, `elu_wgsl`, `from_buffer` | `validate_barracuda_tensor.rs`, `bench_*.rs` |
| **Statistics** | `variance`, `pearson_correlation`, `covariance`, `norm_cdf`, `norm_pdf`, `norm_ppf` | `validate_barracuda_stats.rs`, 8 CPU port binaries |
| **Linear Algebra** | `solve_f64`, `eigh_f64`, `cholesky_f64`, `lu_*`, `tridiag`, `svd_*`, `gen_eigh_f64` | `validate_barracuda_linalg*.rs`, `validate_eigh_accuracy.rs` |
| **Special Functions** | `gamma`, `erf`, `bessel_*`, `legendre`, `hermite`, `laguerre`, `chi_squared_*` | `validate_barracuda_special.rs` |
| **Optimization** | `nelder_mead`, `bisect`, `brent` | `validate_barracuda_optimize.rs` |
| **Tensor f64** | `SumReduceF64`, `FusedMapReduceF64`, `NormReduceF64`, `VarianceReduceF64`, `MaxAbsDiffF64`, `CosineSimilarityF64`, `WeightedDotF64` | `validate_barracuda_tensor_f64.rs` |
| **FFT** | `Fft1D`, `Ifft1D`, `Fft1DF64`, `Rfft` | `validate_barracuda_fft.rs` |
| **LogSumExp** | `LogSumExp` | `validate_barracuda_logsumexp.rs` |
| **Shaders** | `quantized::{dequant_q4, dequant_q8, gemv_q4, gemv_q8}`, `precision::cpu::*` | `validate_barracuda_quantized.rs`, `validate_barracuda_precision.rs` |
| **Staging** | `StatefulPipeline`, `KernelDispatch`, `StatefulConfig` | `validate_gpu_stateful_pipeline.rs` |
| **Dispatch** | `dispatch_for`, `DispatchTarget` | 4 cross-dispatch binaries |
| **Error** | `BarracudaError` | `evolved/mha.rs`, `validate_barracuda_stats.rs` |

---

---

## Session 43 — New Shaders + Upstream Wrappers (February 22, 2026)

### New Local Shaders (4 — evolving for ToadStool absorption)

| Shader | Lines | Entry Point | Workgroup | Domain |
|--------|-------|-------------|-----------|--------|
| `logsumexp_reduce.wgsl` | 42 | `logsumexp_reduce` | 256 | Batched numerically-stable logsumexp |
| `stencil_cooperation.wgsl` | 73 | `stencil_update` | 256 | Fermi imitation dynamics (Moore neighborhood) |
| `rk45_adaptive.wgsl` | 141 | `rk45_step` | 64 | Dormand-Prince RK45 with Hill function RHS |
| `wright_fisher_step.wgsl` | 89 | `wright_fisher` | 256 | Binomial drift + selection + inline xoshiro128** |

### New Upstream Wrappers Wired

| API | BarraCuda Module | Validator | Checks |
|-----|------------------|-----------|--------|
| `GillespieGpu` | `ops::bio::gillespie` | `validate_gpu_gillespie` | 20/20 |
| `TaxonomyFcGpu` | `ops::bio::taxonomy_fc` | `validate_upstream_taxonomy` | 3/3 |
| `KmerHistogramGpu` | `ops::bio::kmer_histogram` | `validate_upstream_kmer` | 3/3 |
| `UniFracPropagateGpu` | `ops::bio::unifrac_propagate` | `validate_upstream_unifrac` | 2/2 |
| `chi_squared::*` | `special::chi_squared` | `validate_barracuda_chi_squared` | 13/13 |

### New Infrastructure

| Component | Location | Tests |
|-----------|----------|-------|
| `mixed.rs` | `metalForge/forge/src/mixed.rs` | 5 unit tests |
| `pcie_bridge.rs` | `metalForge/forge/src/pcie_bridge.rs` | 3 unit tests |
| `MIXED_HARDWARE_DESIGN.md` | `metalForge/MIXED_HARDWARE_DESIGN.md` | Design doc |

### Totals

- **Shader count**: 17 → 21 (4 new local)
- **Forge tests**: 18 → 26 (8 new)
- **Validation binaries**: 115 → 127 (12 new)
- **New validation checks**: 108 across 12 validators

---

### ToadStool S58–S59 Cross-Spring Absorptions (Confirmed in Session 57)

ToadStool absorbed the following from neuralSpring and sibling Springs:

| Absorbed | Origin | BarraCUDA Module |
|----------|--------|-----------------|
| `ValidationHarness` | neuralSpring | `barracuda::validation` |
| `exit_no_gpu` / `gpu_required` | neuralSpring | `barracuda::validation` |
| `require!` macro | neuralSpring | `barracuda::validation` |
| `anderson_3d_correlated` | wetSpring | `barracuda::spectral::anderson` |
| `anderson_sweep_averaged` | wetSpring | `barracuda::spectral::anderson` |
| `find_w_c` | wetSpring | `barracuda::spectral::anderson` |
| `ridge_regression` | wetSpring ESN | `barracuda::linalg::ridge` |
| NMF (Euclidean + KL) | wetSpring | `barracuda::linalg::nmf` |
| 5 ODE bio systems | wetSpring | `barracuda::numerical::ode_bio` |
| `df64_core.wgsl` | hotSpring | `barracuda::shaders::math` |
| `Fp64Strategy` / `GpuDriverProfile` | hotSpring | `barracuda::device::driver_profile` |
| `pow_f64` polyfill fix | hotSpring | `needs_pow_f64_workaround()` |
| Dispatch domain ops | cross-spring | `barracuda::dispatch::domain_ops` |

**neuralSpring impact**: Our `ValidationHarness`, `exit_no_gpu`, and `require!` are
now upstream. We keep our local copy (which adds `check_abs_or_rel`, `baseline_path`,
GPU tensor helpers, `NEURALSPRING_REQUIRE_GPU` env var). Consolidated 4 duplicate
`patch_pow_to_polyfill` functions into `validation::patch_pow_to_polyfill`.

---

*Absorption manifest — neuralSpring, following the hotSpring pattern.*
*Lifecycle: evolve → validate → export WGSL → handoff → ToadStool absorbs → retire.*
