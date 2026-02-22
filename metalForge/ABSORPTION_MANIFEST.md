# neuralSpring Absorption Manifest

**Parent**: ecoPrimals/neuralSpring
**License**: AGPL-3.0-or-later
**Pattern**: Evolve locally → validate → handoff → ToadStool absorbs → retire
**ToadStool HEAD**: `77f70b2e` (Session 31h, Feb 22, 2026)
**Last Updated**: February 22, 2026

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

## Ready for Absorption (Tier A — validated, WGSL exported, flat layouts)

These are complete, validated, and ready for ToadStool to absorb. Each has:
- A `pub const WGSL_*` export from its domain library module
- A validation binary proving GPU-CPU parity
- Flat row-major data layouts matching GPU buffer bindings

| WGSL Shader | Library Export | Domain | Validation | Checks | Absorption Target |
|-------------|--------------|--------|------------|--------|-------------------|
| `hmm_forward_log.wgsl` | `hmm::WGSL_HMM_FORWARD_LOG` | Phylogenetics 016–018 | `validate_gpu_hmm_forward` | 13/13 | `barracuda::ops::hmm` |
| `pairwise_jaccard.wgsl` | `pangenome_selection::WGSL_PAIRWISE_JACCARD` | Pangenome 024 | `validate_gpu_pangenome` | 6/6 | `barracuda::ops::pairwise_distance` |
| `locus_variance.wgsl` | `meta_population::WGSL_LOCUS_VARIANCE` | Meta-pop 025 | `validate_gpu_meta_pop` | 7/7 | `barracuda::ops::VarianceReduceF64` |
| `spatial_payoff.wgsl` | `game_theory::WGSL_SPATIAL_PAYOFF` | Game theory 019 | `validate_gpu_game_theory` | 5/5 | `barracuda::ops::stencil` |
| `batch_ipr.wgsl` | `anderson_localization::WGSL_BATCH_IPR` | Spectral 022–023 | `validate_gpu_anderson` | 5/5 | `barracuda::ops::batch_reduce` |
| `pairwise_hamming.wgsl` | `sate_alignment::WGSL_PAIRWISE_HAMMING` | Alignment 017 | `validate_gpu_sate` | 5/5 | `barracuda::ops::pairwise_distance` |
| `pairwise_l2.wgsl` | `modes::WGSL_PAIRWISE_L2` | MODES 012 | `validate_gpu_modes` | 15/15 | `barracuda::ops::pairwise_distance` |
| `multi_obj_fitness.wgsl` | `directed_evolution::WGSL_MULTI_OBJ_FITNESS` | Directed evo 014 | `validate_gpu_directed` | 6/6 | `barracuda::ops::batch_gemm` |
| `swarm_nn_forward.wgsl` | `swarm_robotics::WGSL_SWARM_NN_FORWARD` | Swarm robotics 015 | `validate_gpu_swarm` | 9/9 | `barracuda::ops::batch_gemm` |
| `hill_gate.wgsl` | `signal_integration::WGSL_HILL_GATE` | Signal 021 | `validate_gpu_signal` | 9/9 | `barracuda::ops::elementwise` |

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

### Cross-Domain Shaders (multi-paper)

| WGSL Shader | Library Export | Domain | Validation | Checks | Absorption Target |
|-------------|--------------|--------|------------|--------|-------------------|
| `batch_fitness_eval.wgsl` | `evolved::WGSL_BATCH_FITNESS_EVAL` | Evolution 011–015 | `validate_gpu_batch_fitness` | 20/20 | `barracuda::ops::batch_gemm` |
| `rk4_parallel.wgsl` | `evolved::WGSL_RK4_PARALLEL` | Regulatory 020–021 | `validate_gpu_rk4` | 8/8 | `barracuda::ops::ode` |
| `mean_reduce.wgsl` | `evolved::WGSL_MEAN_REDUCE` | Aggregation | `validate_gpu_pure_workload` | 7/7 | `barracuda::pipeline::ReduceScalarPipeline` |
| `head_split.wgsl` | `evolved::WGSL_HEAD_SPLIT` | MHA (S-03b) | `validate_mha_gpu` | 10/10 | `barracuda::ops::mha` |
| `head_concat.wgsl` | `evolved::WGSL_HEAD_CONCAT` | MHA (S-03b) | `validate_mha_gpu` | 10/10 | `barracuda::ops::mha` |
| `xoshiro128ss.wgsl` | `rng::WGSL_XOSHIRO128SS` | PRNG | `validate_gpu_prng` | 5/5 | `barracuda::ops::prng` |

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
| `tolerances.rs` | Centralized tolerance constants (58 named) | ~450 |
| `provenance.rs` | Python baseline metadata | ~80 |
| `metrics.rs` | R², RMSE, MAE, NSE | ~150 |
| `fft.rs` | Analytical DFT reference values | ~100 |
| `eigh.rs` | Eigensolver → delegates to `barracuda` (S-12 absorbed) | ~40 |
| 81 validation binaries | Correctness proof suite | ~9k |
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
**Total GPU shader checks**: 108 (16 WGSL)
**Total GPU pipeline checks**: 32 (7 pipelines)
**Total cross-dispatch checks**: 41 (8+8+12+13)
**Total lib tests**: 237 unit + 9 doc (94.9% line coverage)
**Grand total validation**: 1307 (206 Python + 1101 Rust+GPU)

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

*Absorption manifest — neuralSpring, following the hotSpring pattern.*
*Lifecycle: evolve → validate → export WGSL → handoff → ToadStool absorbs → retire.*
