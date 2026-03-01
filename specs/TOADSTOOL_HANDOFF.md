# ToadStool Handoff — neuralSpring Local Evolutions

This document catalogues BarraCUDA / ToadStool shortcomings that
`neuralSpring` evolved around locally, following the `hotSpring` pattern.

**Last reviewed:** ToadStool commit `8dc01a37` (Sessions 50–101, Mar 1, 2026) — **ALL 17 shortcomings RESOLVED, 44 upstream rewires, 130+ barracuda import sites, 20+ submodules exercised, 219 binaries, 200/200 validate\_all, 3560+ checks. S101: ToadStool S71 pin bump (6 commits, ComputeDispatch migration, DF64 transcendentals), GPU stats parity (KimuraGpu+HistogramGpu PASS, JackknifeMeanGpu+HargreavesBatchGpu blocked by upstream shader bugs), +1 binary (219 total). 746 lib tests, 0 clippy.**
**Canonical handoff:** `wateringHole/handoffs/NEURALSPRING_TOADSTOOL_V68_S101_GPU_STATS_PARITY_SHADER_BUGS_MAR01_2026.md`
**Session 97c sync (S70+++ pin bump):** ToadStool pin bumped `e96576ee`→`1dd7e338` (13 commits, S68++→S70+++). **Key absorptions**: 7 new DF64 WGSL shaders (gelu, sigmoid, softmax, layer_norm, sdpa, brent, seasonal_pipeline), `SimpleMlp` (JSON-serde MLP), `matmul_ref` (non-consuming matmul for recurrent architectures), `SymmetrizeGpu`, `LaplacianGpu`, `stats::evolution/jackknife/hydrology`, `chao1_classic`, preferred_workgroup_size. **Rewired**: 2 `matmul_ref` sites (ESN validator + tensor benchmark). **Not rewired** (by design): SimpleMlp in validators (test specific Tensor ops), SymmetrizeGpu (small matrices), LaplacianGpu (keep CPU path). ToadStool now at 668 WGSL shaders, 4700+ workspace tests, 0 clippy warnings, 45 documented unsafe. Full re-validation: 200/200 validate_all, 685 lib tests, 0 clippy. V64 handoff updated.
**Session 56 sync:** 4 baseCamp functions rewired to upstream `barracuda::linalg::graph` + `barracuda::numerical`
**Session 58 sync:** 7 Dispatcher methods rewired to upstream `barracuda::dispatch::domain_ops` + GpuDriverProfile wired in
**Session 57 sync:** S58–S59 confirmed: ValidationHarness/exit_no_gpu/require! absorbed; pow polyfill consolidated; new upstream: anderson correlated, ridge, NMF, ODE bio, dispatch domain_ops, Fp64Strategy
**Session 59 sync:** 5 new rewires — `empirical_spectral_density`, `marchenko_pastur_bounds`, `effective_rank` to upstream stats/linalg; `gelu` + `hmm_forward_step` added to Dispatcher via upstream `domain_ops`; 3 dead WGSL re-exports removed from `evolved/`
**Session 60 sync:** Benchmark validation pass — 22/22 cross-spring evolution checks, f64 typed ops benchmarked (Variance 2.46× hotSpring, Entropy 2.59× wetSpring), 500 lib tests, 145/146 validate_all
**Session 61 sync:** V26 handoff, code quality sweep, 101+ tolerances, property tests, comprehensive evolution handoff
**Session 62 sync:** S-03b **FULLY RESOLVED** upstream. ToadStool `0c998992` decomposed MHA projections into matmul + head_split/head_concat shaders. All 21/21 WGSL shaders absorbed. `evolved/mha.rs` now thin wrapper to `barracuda::ops::mha::MultiHeadAttention`. 500 lib tests, 145/146 validate_all.
**Session 64 sync:** V29 handoff. BandwidthTier + NVK guard wired into Dispatcher. Cross-spring benchmarks: Variance 3.49×, Entropy 2.56×, Pearson 1.33×.
**Session 66 sync:** Phase C GPU promotion — HMM chains, FST, introgression, AF variance. 44 CPU→GPU ops (~97%). validate_gpu_phase_c 18/18 PASS.
**Session 67 sync:** V30 handoff. CPU↔Python parity 39/39 PASS (1e-10). Dispatch tier benchmarks: ≤1.04× CPU overhead (9/10 ops). Per-call GPU driver-bound → motivates pipeline batching.
**Session 68 sync:** V31 handoff. Deep debt audit — 104+ tolerances centralized, zero ad-hoc magic numbers, zero bare `unwrap()`, 90.43% coverage, all files ≤1000 lines. BarraCUDA usage audited (90+ imports, 20+ submodules, zero duplicates). GPU test serialization pattern documented. Rewired `boltzmann_sampling` → `barracuda::sample::boltzmann_sampling` (17th upstream rewire). Total: **17 functions rewired to upstream**.

**Session 69 sync:** 6 validator shader sources rewired from local `include_str!` to upstream barracuda constants (RK4, RK45, batch fitness, logsumexp, swarm NN scores). Upstream-vs-local benchmark: 10/10 ≈ or ~ (zero ⚠). Cross-spring evolution benchmark refreshed. Total: **17 functions + 6 shader sources rewired to upstream**.

**Session 70 sync:** Deep audit II — 93.5% coverage (580 tests, +75). tolerance_registry! macro (891→257 lines). gpu_dispatch/mod.rs split (1332→860+483). SADDLE_EIGENVALUE_THRESHOLD extracted. Streaming I/O for JSON loading. 100% SPDX compliance (211/211 files). V33 handoff crafted. All files ≤1000 lines. Zero debt. Remaining 5.5% uncovered lines are GPU error branches.

**Session 71 sync:** Deep audit execution — 150+ ad-hoc tolerances → named constants across 21 library test files. Smart refactored `gpu_dispatch/mod.rs` 862→304 lines. Dependency audit: all crates Pure Rust (ecoBin compliant). V34 handoff crafted.

**Session 72 sync:** Full ToadStool commit review — 47 commits (`77f70b2e`..`02207c4a`, ToadStool sessions S39–S62) audited. **ALL 17 shortcomings now RESOLVED upstream**: S-14/S-15/S-16 fixed at `a4996b34` (S39: Naive tier removed, matmul hang fixed, transpose dispatch fixed). S-17 fixed at `c82c23d1` (S58: `patch_transcendentals_in_code` covers `pow`). Previously blocked Tensor APIs now available upstream: `argmax_dim(axis)`, `softmax_dim(axis)`. New upstream APIs: `fst_variance_decomposition`, `Conv2dGpu`, `PeakDetectF64`, `MovingWindowStats`, `SparseGemmF64`, `TranseScoreF64`, `ridge_regression`, `NMF`. ToadStool caught up to neuralSpring handoff V16/V18; V33/V34 not yet consumed. `barracuda::validation::ValidationHarness` absorbed (subset of neuralSpring's — missing GPU helpers). Validator workarounds (positive-only data, A×B^T) retained as defense-in-depth. V35 handoff crafted.

**Session 73 sync:** Cross-spring rewiring — 4 new upstream rewires using newly available Tensor APIs. (1) Viterbi `argmax_dim(0)` replaces CPU argmax loop in `hmm_viterbi_step_gpu`. (2) `Dispatcher::softmax_row_wise` via `Tensor::softmax_dim(1)`. (3) `fst_single_locus` wraps `barracuda::ops::bio::fst_variance_decomposition` for F-statistics (θ, f_is, f_it). (4) `pairwise_fst_full` uses upstream per-locus decomposition for multi-locus F-statistics. New tolerances: `DISPATCH_F32_ROUNDTRIP` (1e-6), `DISPATCH_VITERBI_F32` (1e-5). Cross-spring evolution validator: 39/39 PASS. Total: **21 functions + 6 shader sources rewired to upstream**. V36 handoff crafted.

**Session 74 sync:** Pure GPU all-domains — `validate_gpu_pure_workload_all` 10/10 PASS (9 typed BarraCUDA GPU ops: BatchFitnessGpu, MultiObjFitnessGpu, HmmBatchForwardF64, SpatialPayoffGpu, BatchIprGpu, PairwiseHammingGpu, PairwiseL2Gpu, PairwiseJaccardGpu, LocusVarianceGpu + determinism check). `bench_evolution_tiers` measures CPU→GPU portability for 8 domains. f32/f64 precision boundary documented: domain ops f32, HMM/baseCamp f64. IPR requires pre-normalized eigenvectors, Jaccard outputs upper-triangle. GPU dispatch overhead ~186µs per submit (structural floor). Also `validate_cross_system_dispatch` 46/46 PASS (full metalForge stack: hardware discovery → domain heuristics → multi-substrate parity → transfer cost hierarchy → NPU routing → crossover sweep). Discovered 3 GPUs (RTX 4070 Vulkan, TITAN V NVK, RTX 4070 OpenGL) + i9-12900K CPU. CPU→GPU crossover at ~1946µs. validate_all: 150/150 PASS. 166 binaries. V39 handoff updated.

**Session 76 sync:** Modern BarraCUDA rewiring + benchmark validation. Rewired `matrix_correlation` and `thermal_diversity_correlation` in `meta_population.rs` → `barracuda::stats::pearson_correlation` (cross-spring origin: airSpring/groundSpring hydrology S64). Full benchmark sweep on RTX 4070: upstream wrappers add 0 meaningful overhead (0.85–1.14× across 10 kernels). Cross-spring evolved f64 shaders outperform naïve f32 Tensor: Variance 3.20×, Pearson 1.36×, Shannon 2.24×. All quality gates green: 580+43+9 tests, 150/150 validate_all, 39/39 cross-spring, 15/15 bench. Total: **32 functions + 6 shader sources rewired to upstream**.

**Session 77 sync:** WDM surrogates + baseCamp GPU pure + coralForge shaders. 3 WDM Python baselines (nW-01 transport via Stanton-Murillo, nW-02 EOS via Militzer FPEOS, nW-04 classical-to-WDM transfer learning). `wdm_surrogate.rs` module with `EosSurrogate::predict()`. 2 Rust validators (CPU + BarraCUDA GPU). `validate_basecamp_gpu_pure` validates all 5 sub-theses on GPU with scalar readback. `bench_basecamp_gpu_pure` benchmarks GPU vs CPU for all sub-theses. 9 new f64 WGSL shaders for coralForge (layer\_norm, GELU, sigmoid, SDPA scores/softmax/apply, triangle mul outgoing/incoming, triangle attention). 604 lib tests. V41 handoff.

**Session 78–79 sync:** ToadStool S66 absorption + complete cross-spring rewiring. Deep S66 review: `compile_shader_df64` convention with `Df64` struct, 6 new function rewires (mae → `barracuda::stats::mae`, shannon → `shannon_from_frequencies`, hill×2 → `barracuda::stats::hill`, l2\_distance → `l2_distance_dispatch`, complexity → `fit_linear`). All 9 metalForge f64 shaders aligned to `compile_shader_df64` convention. Population vs sample variance clarified. Cross-spring validator expanded to 52/52 PASS, benchmark to 19/19 PASS. Total: **38 functions + 6 shader sources rewired to upstream**. V42→V43→V44 handoffs.

**Session 86 sync:** WDM surrogate buildout complete. `wdm_transport.rs` new module (MLP 3→H→3 transport surrogate). 4 new validators (nW-01 transport 30/30, nW-02 EOS wired 36/36+GPU 15/15, nW-04 transfer 6/6) added to `validate_all` (154 total). `check_drift.sh` expanded to 29 baselines. 611 lib + 43 forge + 9 integration tests. Key learning: `barracuda::nn::SimpleMLP` with JSON weight loading would replace ~400 LOC across 3 WDM surrogates — highest-priority absorption target. V50 handoff crafted.

**Session 87 sync:** WDM surrogate queue closed — nW-03 (LSTM S(q,ω) peak predictor) and nW-05 (ESN regime classifier) complete. 156 total validators, 31 baselines, 668 lib + 43 forge + 9 integration tests.

**Session 91 sync:** Full ToadStool S66–S68 evolution review. ToadStool achieved **ZERO f32-only shaders** (296 deleted, 291 converted to f64 canonical). Dual-layer universal precision architecture: `Precision::op_preamble()` (Layer 1: abstract ops) + `sovereign/df64_rewrite.rs` (Layer 2: naga-guided f64→df64 infix rewrite). `compile_shader_universal(source, precision)` now exposed in `gpu.rs` for callers to compile at F16/F32/F64/Df64 per-use/hardware. `Precision` enum re-exported from `gpu.rs`. Primal Evoformer matmul helpers (`matmul_2d`, `matmul_3d`) rewired to upstream `barracuda::dispatch::matmul_dispatch` (m, k, n non-square support). NUCLEUS Tower validator: 22/22 PASS. All quality gates green: 669 lib tests, 0 clippy warnings, 181/181 validate\_all. ToadStool metrics: 700 WGSL shaders (497 f32 via LazyLock downcast, 182 f64, 19 Df64), 2608 barracuda tests, 122 shader tests (unit + e2e + chaos + fault). Total: **44 functions + 6 shader sources rewired to upstream**. V61 handoff crafted.

---

## Resolution Status

**All 12 neuralSpring shortcomings (S-01 through S-12) are now ABSORBED by
ToadStool `77f70b2e`.** S-13 **FIXED** upstream in Session 42 (`5437c170`).
S-03b **FULLY RESOLVED** upstream (ToadStool `0c998992`). Key absorption commits:

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
| `ReduceScalarPipeline` | `pipeline::ReduceScalarPipeline::sum_f64()` | **Wired** — Anderson mean IPR (5.55e-17 diff) |
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
| `mha` | ~50 | **Thin wrapper** to `barracuda::ops::mha::MultiHeadAttention` (S-03b resolved upstream) | `validate_barracuda_ml_inference`, `bench_transformer_block` | 17 |
| `hmm_forward_gpu` | 270 | `HmmBatchForwardF64` validated (11/11 PASS) — local retained for f32 fallback | `validate_barracuda_hmm_f64` | 11/11 |

### S-03b: MHA — FULLY RESOLVED (ToadStool `0c998992`)

**Root cause**: Native MHA fuses matmul into projection shaders (heavy per-thread nested loops → GPU watchdog timeout).

**Upstream fix**: ToadStool `0c998992` decomposed MHA projections into matmul + head_split/head_concat shaders. All 21/21 WGSL shaders now absorbed upstream.

**Status**: `evolved::mha` is now a thin wrapper delegating to `barracuda::ops::mha::MultiHeadAttention`.

### Rewired to native APIs

| Binary | Previous | Now |
|--------|----------|-----|
| `bench_barracuda_tensor` | `evolved::layer_norm`/`log_softmax` | `Tensor::layer_norm_wgsl()`/`log_softmax_wgsl()` |
| `validate_barracuda_ml_inference` | Uses `evolved::mha` (thin wrapper to upstream) | Kept |
| `bench_transformer_block` | Uses `evolved::mha` (thin wrapper to upstream) | Kept |

---

## Session 59 — ToadStool S59 Sync + Rewiring (February 24, 2026)

### 5 Functions Rewired to Upstream BarraCUDA

| Local Function | Module | Upstream API | Absorbed In |
|----------------|--------|-------------|-------------|
| `empirical_spectral_density` | `weight_spectral` | `barracuda::stats::empirical_spectral_density` | S54 (M-011) |
| `marchenko_pastur_bounds` | `weight_spectral` | `barracuda::stats::marchenko_pastur_bounds` | S54 (M-012) |
| `effective_rank` | `neural_pgm` | `barracuda::linalg::effective_rank` | S54 (H-009) |
| (new) `gelu` | `gpu_dispatch/dispatch_ops` | `barracuda::dispatch::gelu_dispatch` | S52 |
| (new) `hmm_forward_step` | `gpu_dispatch/dispatch_ops` | `barracuda::dispatch::hmm_forward_dispatch` | S52 |

### evolved/ Module Cleanup

| Change | Details |
|--------|---------|
| Removed `WGSL_BATCH_FITNESS_EVAL` | Dead re-export — all callers use `barracuda::ops::batch_gemm` directly |
| Removed `WGSL_RK4_PARALLEL` | Dead re-export — all callers use `barracuda::ops::rk_stage` directly |
| Removed `WGSL_MEAN_REDUCE` | Dead re-export — all callers use `barracuda::pipeline::ReduceScalarPipeline` directly |
| Removed `WGSL_HEAD_SPLIT` / `WGSL_HEAD_CONCAT` | S-03b resolved — upstream `barracuda::ops::mha` |

### MHA Retirement Assessment

`evolved/mha` is now a **thin wrapper** delegating to `barracuda::ops::mha::MultiHeadAttention`.
S-03b fully resolved upstream (ToadStool `0c998992`): MHA projections decomposed into matmul + head_split/head_concat. All 21/21 WGSL shaders absorbed.

### Validation

| Gate | Result |
|------|--------|
| `cargo test --lib` | 500 PASS |
| `validate_all` | 145/146 PASS (1 pre-existing logsumexp driver issue) |
| `cargo clippy (pedantic+nursery)` | 0 warnings |

### Rewire Count

Total rewired functions: **16** (11 from S56/S58 + 5 from S59)

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

**Date**: February 22, 2026
**Status**: 24/25 papers ported to BarraCUDA CPU math. 203/203 checks PASS (96% coverage).

### Primitives Validated

| Primitive | Modules | Precision | Status |
|-----------|---------|-----------|--------|
| `numerical::rk45_solve` | regulatory, signal, game | Machine ε | **Excellent** |
| `linalg::solve_f64` | hmm, swarm | Machine ε | **Excellent** |
| `linalg::eigh_f64` | spectral, anderson | 1.75e-14 (n=32) | **S-12 RESOLVED** — Householder+QR |
| `special::chi_squared_sf` | introgression | 1e-10 | **Excellent** |
| `stats::variance` | all 13 modules | Machine ε | **Excellent** |
| `stats::pearson_correlation` | modes | Machine ε | **Excellent** |

### S-12: eigh_f64 — ABSORBED (`77f70b2e`)

**Upstream fix**: `barracuda::ops::linalg::eigh_householder_qr` absorbed neuralSpring's
Householder+QR implementation verbatim. `src/eigh.rs` now delegates to upstream.
Local fossil: `metalForge/fossils/evolved_s01_s11/eigh_local.rs`.

- Validation: `validate_eigh_accuracy` (9/9 PASS, delegated)
- ToadStool also added `WGSL_BATCHED_EIGH_NAK_OPTIMIZED` for GPU-native eigensolve

---

## metalForge Shader Evolutions

17 WGSL shaders in `metalForge/shaders/`. 13 now have upstream equivalents in
barracuda (8 identical at `77f70b2e`, 5 generalized variants at `5437c170`).
4 remain local-only.

### Absorbed (identical — `77f70b2e`)

| Shader | Upstream API | Status |
|--------|-------------|--------|
| `hmm_forward_log.wgsl` | `barracuda::ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32` | **Absorbed** |
| `batch_fitness_eval.wgsl` | `barracuda::ops::bio::batch_fitness::WGSL_BATCH_FITNESS_EVAL` | **Absorbed** |
| `rk4_parallel.wgsl` | `barracuda::ops::rk_stage::WGSL_RK4_PARALLEL` | **Absorbed** |
| `pairwise_jaccard.wgsl` | `barracuda::ops::bio::pairwise_jaccard::WGSL_PAIRWISE_JACCARD` | **Absorbed** |
| `pairwise_hamming.wgsl` | `barracuda::ops::bio::pairwise_hamming::WGSL_PAIRWISE_HAMMING` | **Absorbed** |
| `locus_variance.wgsl` | `barracuda::ops::bio::locus_variance::WGSL_LOCUS_VARIANCE` | **Absorbed** |
| `spatial_payoff.wgsl` | `barracuda::ops::bio::spatial_payoff::WGSL_SPATIAL_PAYOFF` | **Absorbed** |
| `batch_ipr.wgsl` | `barracuda::spectral::batch_ipr::WGSL_BATCH_IPR` | **Absorbed** |

### Absorbed (generalized variants — Session 42, `5437c170`)

| Shader | Upstream Path | Key Differences |
|--------|---------------|-----------------|
| `pairwise_l2.wgsl` | `barracuda::shaders::math::pairwise_l2` | Closed-form pair decode, different struct |
| `multi_obj_fitness.wgsl` | `barracuda::shaders::bio::multi_obj_fitness` | Bessel correction (n-1), different params |
| `hill_gate.wgsl` | `barracuda::shaders::bio::hill_gate` | Mode 0/1 generalization, `HillGateParams` |
| `swarm_nn_forward.wgsl` | `barracuda::shaders::bio::swarm_nn_forward` | Generic MLP, `SwarmParams`, clamped sigmoid |
| `mean_reduce.wgsl` | `barracuda::shaders::reduce::mean_reduce` | Effectively identical |

Local copies retained for validation (validators depend on local binding layouts).

### Upstream Parity Verification (10/10 PASS)

All 10 absorbed shaders have dual-path validators comparing local metalForge
dispatch output vs upstream BarraCuda wrapper output.

| Shader | Wrapper | Parity Diff | Status |
|--------|---------|-------------|--------|
| `batch_fitness_eval.wgsl` | `BatchFitnessGpu` | 0.00e0 | **bit-identical** |
| `pairwise_hamming.wgsl` | `PairwiseHammingGpu` | 0.00e0 | **bit-identical** |
| `pairwise_jaccard.wgsl` | `PairwiseJaccardGpu` | 0.00e0 | **bit-identical** |
| `locus_variance.wgsl` | `LocusVarianceGpu` | 0.00e0 | **bit-identical** |
| `spatial_payoff.wgsl` | `SpatialPayoffGpu` | 0.00e0 | **bit-identical** |
| `batch_ipr.wgsl` | `BatchIprGpu` | 0.00e0 | **bit-identical** |
| `hill_gate.wgsl` | `HillGateGpu` | 0.00e0 | **bit-identical** |
| `pairwise_l2.wgsl` | `PairwiseL2Gpu` | 0.00e0 | **bit-identical** |
| `multi_obj_fitness.wgsl` | `MultiObjFitnessGpu` | 1.95e-3 | **PASS** (Bessel n-1 vs n) |
| `swarm_nn_forward.wgsl` | `SwarmNnGpu` | 0 (u32) | **bit-exact** |

### Still local (no upstream equivalent or significant API differences)

| Shader | Domain | Suggested upstream module |
|--------|--------|--------------------------|
| `xoshiro128ss.wgsl` | GPU PRNG | `barracuda::ops::prng` |
| `swarm_nn_scores.wgsl` | Swarm (015) | New — no upstream equivalent |

---

## Phase 5a/5b Shortcomings — ALL RESOLVED UPSTREAM

Discovered during GPU `Tensor` validation across 23 scientific domains.
**All resolved** at ToadStool `a4996b34` (S39) and `c82c23d1` (S58).

| # | Shortcoming | Severity | Root Cause | Status |
|---|-------------|----------|------------|--------|
| S-14 | Naive matmul hang (small square matrices) | Medium | Driver/binary complexity interaction (RTX 4070 Vulkan) | **RESOLVED** upstream (`a4996b34` S39: Naive tier removed) |
| S-15 | Matmul hang when f32 elements ≤ 0.1 magnitude | Critical | WGPU/Vulkan driver bug (RTX 4070) — not sign or sparsity | **RESOLVED** upstream (`a4996b34` S39) |
| S-16 | 2D transpose dispatch uses wrong divisor | High | `execute_2d()` uses 256 instead of tile size 16 | **RESOLVED** upstream (`a4996b34` S39: `const TILE: u32 = 16`) |

Validators retain conservative data patterns (positive-only, A×B^T) as defense-in-depth.
Full diagnosis: `wateringHole/handoffs/`

---

## Capability-Based Dispatch (Sessions 40, 42)

All 12 core GPU validators (batch_fitness, anderson, game_theory, sate, pangenome,
meta_pop, modes, directed, swarm, signal, rk4) plus the evolved `hmm_forward_gpu`
module now use `Gpu::dispatch_1d()` instead of hardcoded `.div_ceil(256)`.

`dispatch_1d` validates the shader's `@workgroup_size(N)` against runtime-discovered
`max_compute_workgroup_size_x` (panics on incompatible hardware) and clamps
workgroup counts to `max_compute_workgroups_per_dimension`.

Capabilities are logged at startup for observability:
```
capabilities: wg_x=256, dispatch_max=65535, buffers=12, f64=true, f16=true
```

### Cross-Eigensolver Validation

`validate_barracuda_spectral_theory` now includes `eigh_vs_sturm` checks:
- Dense Householder+QR (`eigh_householder_qr`) vs tridiag Sturm bisection
  (`find_all_eigenvalues`) on Anderson Hamiltonians
- n=64 W=3: max eigval diff `2.89e-15` (machine epsilon agreement)
- n=200 W=6: max eigval diff `1.42e-14`
- **17/17 checks PASS** (up from 14)

---

## Deep Audit (February 22, 2026 — Sessions 41–42)

Comprehensive codebase audit and debt resolution:

### Code Quality

- **`cargo fmt`**: Fixed 33 files (2,521 lines of diff) — now zero violations
- **`cargo clippy`**: Fixed 123 warnings (pedantic + nursery + `unwrap_used` + `expect_used`) — now zero warnings
- **`cargo doc`**: Fixed 1 broken rustdoc link — now zero warnings
- **All `#[allow]` attributes audited**: `dead_code` (1, field used but not read yet), cast lints (deliberate numeric narrowing), `float_cmp` (determinism tests), `expect_used` (test-only) — all justified
- **Zero mocks, stubs, `todo!`, or `unimplemented!`** in production code

### Deduplication

- Extracted shared GPU validation helpers into `src/validation.rs`: `gpu_readback`, `max_abs_diff_gpu_vs_cpu`, `check_gpu_points`, `gpu_tensor!` macro
- Migrated 24 validation binaries from local copies to shared helpers (~400 LOC removed)
- Eliminated local `fn readback`, `fn max_abs_diff`, `macro_rules! tensor` from all binaries

### Tolerance Centralization

- Added 3 new constants: `INTROGRESSION_FRACTION_ABS`, `INTROGRESSION_FPR_MAX`, `GENE_TREE_CONCORDANT_MIN`
- Registered 18 previously unregistered tolerances in the runtime `NamedTolerance` registry
- Replaced all inline magic numbers in validation binaries with named tolerances
- Split tolerances/ module (1028 lines) into `tolerances/mod.rs` (696) + `tolerances/registry.rs` (341)

### Provenance

- Added explicit `python3` commands and environment details to `SOFTMAX_1_TO_5`, `GELU_REFERENCE`, `RASTRIGIN_REFERENCE`

### Test Coverage

- **9 new determinism tests**: introgression, regulatory_network, pangenome_selection, meta_population, sate_alignment, signal_integration, game_theory, spectral_commutativity, anderson_localization (total: 16)
- **9 new integration tests** (`tests/integration.rs`): cross-module consistency, provenance round-trip, tolerance registry lookup, validation harness, HMM/softmax/GELU/benchmark provenance verification
- Library tests: **459 lib + 9 integration tests** (up from 264 lib)

### Dependency Analysis

All external dependencies are pure Rust — zero C/C++ wrapper crates:
- `barracuda` (workspace), `bytemuck` (GPU Pod), `serde`/`serde_json` (JSON baselines, 3 files), `tokio` (wgpu async), `wgpu` (GPU), `approx` (dev-only)

### Idiomatic Rust Evolution

- `unwrap()` → `let-else` / `unwrap_or(Ordering::Equal)` across all non-test code
- Casts: `as u64` → `u64::from()`, `as f64` → `f64::from()` (infallible)
- Arithmetic: `a * b + c` → `a.mul_add(b, c)` (FMA)
- `const fn` where applicable
- `non_upper_case_globals` → proper naming convention

### Session 42 Deep Audit (February 22, 2026)

- fmt/clippy/doc all clean (0 warnings)
- GPU validation helpers deduplicated (23 binaries → shared validation.rs)
- Tolerance module split (mod.rs + registry.rs, 18 new registry entries)
- 9 new determinism tests, 9 new integration tests
- Python drift detection script (control/check_drift.sh)
- Pure Rust dependency tree verified

---

## Session 43 — New Absorption Targets (February 22, 2026)

neuralSpring built 4 new WGSL shaders validated and ready for ToadStool absorption:

### New Shaders (4 files, all validated)

| Shader | Entry Point | Domain | Absorption Target | Validator |
|--------|-------------|--------|-------------------|-----------|
| `logsumexp_reduce.wgsl` | `logsumexp_reduce` | Batched logsumexp (HMM/phylo) | `barracuda::ops::reduce` | `validate_gpu_logsumexp` 5/5 |
| `stencil_cooperation.wgsl` | `stencil_update` | Fermi imitation dynamics | `barracuda::ops::stencil` | `validate_gpu_stencil` 3/3 |
| `rk45_adaptive.wgsl` | `rk45_step` | Dormand-Prince RK45 (Hill RHS) | `barracuda::ops::ode` | `validate_gpu_rk45` 6/6 |
| `wright_fisher_step.wgsl` | `wright_fisher` | Drift+selection+xoshiro | `barracuda::ops::popgen` | `validate_gpu_wright_fisher` 4/4 |

### New Upstream Wrappers Successfully Wired

| API | Checks | Key Finding |
|-----|--------|-------------|
| `GillespieGpu` | 20/20 | Perfect conservation (f64 integer stoichiometry) |
| `TaxonomyFcGpu` | 3/3 | f64 log-posterior bit-exact GPU vs CPU |
| `KmerHistogramGpu` | 3/3 | u32 histogram exact match |
| `UniFracPropagateGpu` | 2/2 | f64 leaf init exact (1e-12 tolerance) |
| `chi_squared::*` | 13/13 | PDF/CDF/moments/test all within 1e-4 of SciPy |

### Mixed-Hardware Infrastructure (for future ToadStool absorption)

| Component | Purpose | Absorption Target |
|-----------|---------|-------------------|
| `mixed.rs` | MixedSubstrate enum + TransferCost model | `barracuda::unified_hardware` |
| `pcie_bridge.rs` | PcieBridge + P2P detection | `barracuda::unified_hardware::transfer` |
| `logsumexp_substrate()` | Dispatch heuristic for logsumexp | `barracuda::dispatch` |
| `stochastic_substrate()` | Dispatch heuristic for popgen | `barracuda::dispatch` |

### CPU vs GPU Parity Validation

`validate_cpu_gpu_parity` (17/17 PASS): Tensor API produces identical results on GPU (RTX 4070 Vulkan)
and CPU (llvmpipe). Cross-hardware MatMul and ReLU are **bit-identical**.

---

## Session 44 — Multi-GPU Portability & Bug Fixes (February 23, 2026)

### Upstream Bug Fixes (P0 — action required for ToadStool)

| Bug | File | Fix | Impact |
|-----|------|-----|--------|
| `Tensor::mean()` crash | `ops/mean.rs` | `entry_point: "main"` → `"mean_reduce"`, single-f32 readback, remove double-divide | Any caller of `Tensor::mean()` |
| Chi-squared precision | validator-side only | Updated expected values to full precision | Documentation quality |

### Multi-GPU Validation

| GPU | Driver | Architecture | Result |
|-----|--------|-------------|--------|
| RTX 4070 | NVIDIA proprietary | Ada Lovelace | **131/131 PASS** |
| TITAN V | NVK open-source | Volta (GV100) | **143+ PASS** |

All results **bit-identical** across driver stacks. WGSL shaders are fully portable.

### New Validators (4 files)

| Validator | Checks | Absorption Relevance |
|-----------|--------|---------------------|
| `validate_gpu_pipeline_wright_fisher` | 4/4 | Chains `wright_fisher_step.wgsl` → `mean_reduce.wgsl` in single encoder |
| `validate_gpu_pipeline_gillespie` | 6/6 | Chains `GillespieGpu` output → `mean_reduce.wgsl` |
| `validate_barracuda_gpu_lenet` | 8/8 | First exercise of `Tensor::conv2d()` + `Tensor::maxpool2d()` |
| `validate_barracuda_transformer` | 12/12 | Full transformer layer via Tensor API; found global-only softmax behavior |

### Tensor API Findings for ToadStool

| Finding | Recommendation |
|---------|----------------|
| `Tensor::softmax()` is global (all elements), not row-wise | Add `Tensor::softmax_dim(axis)` or document that row-wise requires `ScaledDotProductAttention` |
| `Tensor::mean()` was broken (wrong entry point + double-divide) | Already fixed locally; needs upstream merge |
| No fused `Tensor::layer_norm()` method | Shader exists but no Tensor API method |

### Performance Data for ToadStool Optimization

Pure Rust is 178.5× faster than single-thread Python/NumPy. Exception: dense matmul
(commutator 64×64) where NumPy BLAS is 2.5× faster. Opportunity for BarraCUDA's CPU
matmul to adopt tiling/SIMD techniques from `whitePaper/BARRACUDA_EVOLUTION.md`.

### NVK Compatibility for ToadStool CI

NVK handles all 21 neuralSpring WGSL shaders without modification on Volta hardware.
Consider adding NVK to ToadStool CI for open-source driver compatibility testing.

---

## Sessions 45–46 — Pure GPU Promotion (Phase A+B)

### What Changed

neuralSpring created `gpu_ops/` (6 submodules) and `gpu_dispatch.rs` — a capability-based
runtime dispatch layer that routes 38 previously CPU-bound operations to GPU
via the BarraCUDA `Tensor` API. The `Dispatcher` detects GPU availability at
construction and falls back to CPU when hardware is unavailable.

### Tensor API Learnings for ToadStool

| Finding | Impact | Recommendation |
|---------|--------|----------------|
| `matmul`, `softmax`, `sigmoid`, `gelu_wgsl`, `log_wgsl`, `exp_wgsl`, `sqrt_wgsl`, `broadcast` consume `self` | Requires cloning or careful ownership chains | Document consuming vs borrowing methods in Tensor API docs |
| `max_dim` returns values only, no argmax indices | Viterbi requires CPU argmax after GPU max | Add `argmax_dim()` to Tensor API |
| `x^n` via `exp(n * ln(x))` is numerically stable with guard | Good pattern for GPU power functions | Consider `Tensor::pow_scalar()` method |
| 2×2 GEMV works through `matmul` with `[1,2] × [2,2]` | Correct but dispatch overhead dominates at small sizes | Fused elementwise-matmul for small matrices |
| Column sum via `sum_dim(0)` works for allele frequencies | Clean API for reductions along arbitrary dimension | Already good |
| `broadcast` + `add` for outer-product-like patterns | Viterbi score matrix construction works | Already good |
| Global `softmax` vs per-row softmax remains | Attention still needs `ScaledDotProductAttention` | Add `softmax_dim(axis)` |

### GPU-Promoted Operations (38 total)

**Phase A (27 ops, Session 45):**
matmul, transpose, frobenius_norm, softmax, l2_distance, pearson_correlation,
variance, mean, neural_forward, hmm_forward_step, rk4_step, fitness_evaluation,
diversity_metrics, tree_distance, logsumexp, log_likelihood, pca_project,
chi_squared, hamming_distance, jaccard_similarity, batch_fitness, locus_variance,
spatial_payoff, geographic_distances, pairwise_distances, and more.

**Phase B (11 ops, Session 46):**
hmm_backward_step, hmm_viterbi_step, allele_frequencies, nucleotide_diversity,
matrix_correlation, geographic_distance_matrix, thermal_diversity_correlation,
inter_population_af_variance, replicator_step, hill_activation_batch.

### Remaining CPU-Only (~10% of production math)

| Operation | Why CPU-Only | Absorption Path |
|-----------|-------------|----------------|
| Full ODE loops (`integrate_ode`, `integrate_grn`) | Sequential time-stepping with state dependency | `StatefulPipeline` + GPU PRNG for stochastic terms |
| FST variance decomposition | Multi-step between/within variance | Custom `fst_decompose.wgsl` shader |
| Introgression HMM chain | Full forward → backward → Viterbi sequence | Compose `hmm_forward_step` + `hmm_backward_step` + `hmm_viterbi_step` |
| Viterbi argmax | `max_dim` returns values only | Needs `argmax_dim()` in Tensor API |

### Validation

| Validator | Checks | Hardware |
|-----------|--------|----------|
| `validate_gpu_promotion` | **27/27 PASS** | RTX 4070 + TITAN V NVK |
| `validate_gpu_phase_b` | **20/20 PASS** | RTX 4070 + TITAN V NVK |
| `validate_all` | **133/133 PASS** | RTX 4070 |

---

## ToadStool Sync: `5437c170` → `6ee71f07` (2 commits)

neuralSpring synced to ToadStool HEAD `9abd6857` (Feb 23, 2026). Two bug-fix
commits since our last tracked commit:

| Commit | Fix | Origin | neuralSpring Impact |
|--------|-----|--------|-------------------|
| `b53dd2f6` | SNP BGL binding mismatch (6→5 storage); ODE f64 builtins (max/pow/clamp polyfills); Jacobi eigenvector rotation (all rows) | wetSpring Exp098 | **None** — neuralSpring doesn't use SNP, ODE f64, or Jacobi eigenvectors |
| `6ee71f07` | loop_unroller `substitute_loop_var` emits `u32` suffix (`"0"` → `"0u"`) | hotSpring v0.6.7 | **None** — affects `BatchedEighGpu` single-dispatch (not used by neuralSpring) |

**Build**: `cargo check` clean, zero new warnings.
**Validation**: 459 lib + 9 integration tests PASS. `validate_all`: **133/133 PASS** (RTX 4070).

### Still Pending Absorption (neuralSpring → ToadStool)

| Fix | File | Applied Locally | Status |
|-----|------|----------------|--------|
| `Tensor::mean()` entry point + double-divide | `ops/mean.rs` | Session 44 | **Pending** — needs ToadStool commit |
| Chi-squared expected value precision | neuralSpring validator only | Session 44 | **N/A** — validator-side only |

---

## Session 50: baseCamp Primitives for ToadStool Absorption

Session 50 added 5 baseCamp modules (82/82 PASS) implementing Biophysical
AI Interpretability. These introduce general-purpose primitives suitable
for upstream absorption.

### Absorption Candidates

| Primitive | Current Location | Generalized Form | BarraCUDA Target |
|-----------|-----------------|-----------------|-----------------|
| `graph_laplacian` | `agent_coordination.rs` | `D - A` from any adjacency matrix | `ops::linalg::laplacian` |
| `effective_rank` | `neural_pgm.rs` | Entropy of normalized eigenvalues | `ops::linalg::effective_rank` |
| `empirical_spectral_density` | `weight_spectral.rs` | Eigenvalue histogram | `ops::stats::histogram` |
| `numerical_hessian` | `loss_landscape.rs` | Central finite differences | `ops::numerical::hessian` |
| `level_spacing_ratio` | `weight_spectral.rs` | GOE/Poisson spectral stat | `ops::stats::level_spacing_ratio` |

### GPU Shader Candidates for ToadStool

| Shader | Description | Template |
|--------|-------------|----------|
| `symmetrize.wgsl` | `out[i,j] = (A[i,j] + A[j,i]) / 2` | Adapt from `transpose.wgsl` |
| `histogram.wgsl` | Atomic histogram binning of eigenvalues | New pattern (workgroup atomics) |
| `hessian_column.wgsl` | Parallel finite differences per dimension | Adapt from `batch_fitness_eval.wgsl` |
| `laplacian.wgsl` | Row-sum → diagonal, subtract adjacency | Adapt from `spatial_payoff.wgsl` |
| `metropolis.wgsl` | Parallel MCMC chains with acceptance | Adapt from `wright_fisher_step.wgsl` |

### baseCamp eigh Usage (5 new consumers)

All 5 baseCamp modules use `eigh_f64` (via `eigh.rs` → `barracuda::ops::linalg`):

| Module | Input Matrix | Typical Size | Use |
|--------|-------------|-------------|-----|
| `weight_spectral` | Symmetrized `W^T W` | 64×64–512×512 | Spectral analysis of weight Hamiltonians |
| `information_flow` | Attention Hamiltonian | seq_len × seq_len | Information localization |
| `loss_landscape` | Numerical Hessian | n_params × n_params | Curvature analysis |
| `neural_pgm` | Symmetrized transition | n_states × n_states | Effective rank |
| `agent_coordination` | Disordered Laplacian | n_agents × n_agents | Coordination spectral analysis |

**Note**: No new shortcomings discovered. S-15 (matmul magnitude ≤ 0.1) does not
affect baseCamp — all matrices are synthetic with controllable magnitude.

**Updated totals**: 36 modules, 142 binaries, 459 unit tests (up from 31/133/374).

---

## Session 55: Mixed-Hardware Dispatch Wiring

Session 55 wired `metalForge::mixed::mixed_substrate()` into `Dispatcher::mixed_dispatch()`,
creating an end-to-end dispatch path from science operation → substrate routing → GPU/CPU/NPU
execution. This is ready for `ToadStool` to absorb into `barracuda::unified_hardware`.

### New Absorption Candidates

| Component | Current Location | `BarraCUDA` Target |
|-----------|-----------------|-------------------|
| `Dispatcher::mixed_dispatch()` | `gpu_dispatch/mod.rs` | `barracuda::unified_hardware::dispatch` |
| `mixed_substrate()` | `metalForge/forge/src/mixed.rs` | `barracuda::unified_hardware::routing` |
| `PcieBridge` | `metalForge/forge/src/pcie_bridge.rs` | `barracuda::unified_hardware::transfer` |

### New Validators

| Validator | Checks | Status |
|-----------|--------|--------|
| `validate_compute_dispatch` | 16 (routing + CPU↔GPU parity for 6 ops) | **PASS** |
| `validate_mixed_hardware` | 14 (mixed routing + PCIe bridge + crossover) | **PASS** |

---

## S-17: HillGate f64 `pow()` Fix — RESOLVED UPSTREAM (`c82c23d1` S58)

### Root Cause

`hill_gate_f64.wgsl` uses native WGSL `pow(f64, f64)`. On both:
- **RTX 4070 (Ada Lovelace, proprietary)**: NVVM compilation failure → device lost
- **TITAN V (NVK, open-source)**: NAK assertion `alu.def.bit_size() == 32` → device lost

### Resolution

**RESOLVED upstream** at ToadStool `c82c23d1` (S58: cross-spring absorption).
`patch_transcendentals_in_code` now covers `pow(` → `pow_f64(` in addition to
`exp(` and `log(`. The fix was proven by neuralSpring (18/18 PASS on both RTX 4070
and TITAN V NVK) and absorbed verbatim.

neuralSpring retains `validation::patch_pow_to_polyfill()` as defense-in-depth
for any WGSL loaded outside the barracuda shader compilation pipeline.

### Validation

| Adapter | Max GPU-CPU Diff | Checks |
|---------|-----------------|--------|
| RTX 4070 (Vulkan, proprietary) | 1.11e-16 | 18/18 PASS |
| TITAN V (NVK, open-source) | 2.22e-16 | 18/18 PASS |
