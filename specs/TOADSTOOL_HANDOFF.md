# ToadStool Handoff — neuralSpring Local Evolutions

This document catalogues BarraCUDA / ToadStool shortcomings that
`neuralSpring` evolved around locally, following the `hotSpring` pattern.

**Last reviewed:** ToadStool commit `d45fdfb3` (Session 40, Session 42 deep audit, Feb 22, 2026)
**Canonical handoff:** `wateringHole/handoffs/NEURALSPRING_V10_TOADSTOOL_BARRACUDA_HANDOFF_FEB22_2026.md`

---

## Resolution Status

**All 12 neuralSpring shortcomings (S-01 through S-12) are now ABSORBED by
ToadStool `77f70b2e`.** S-13 **FIXED** upstream in Session 39 (`d45fdfb3`).
S-03b has local workaround. Key absorption commits:

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
| `mha` | 182 | S-03b: native projection shaders hang — **GPU `head_split.wgsl`/`head_concat.wgsl` now available** | `validate_barracuda_ml_inference`, `bench_transformer_block` | 17 |
| `hmm_forward_gpu` | 270 | `HmmBatchForwardF64` validated (11/11 PASS) — local retained for f32 fallback | `validate_barracuda_hmm_f64` | 11/11 |

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
barracuda (8 identical at `77f70b2e`, 5 generalized variants at `d45fdfb3`).
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

### Absorbed (generalized variants — Session 39, `d45fdfb3`)

| Shader | Upstream Path | Key Differences |
|--------|---------------|-----------------|
| `pairwise_l2.wgsl` | `barracuda::shaders::math::pairwise_l2` | Closed-form pair decode, different struct |
| `multi_obj_fitness.wgsl` | `barracuda::shaders::bio::multi_obj_fitness` | Bessel correction (n-1), different params |
| `hill_gate.wgsl` | `barracuda::shaders::bio::hill_gate` | Mode 0/1 generalization, `HillGateParams` |
| `swarm_nn_forward.wgsl` | `barracuda::shaders::bio::swarm_nn_forward` | Generic MLP, `SwarmParams`, clamped sigmoid |
| `mean_reduce.wgsl` | `barracuda::shaders::reduce::mean_reduce` | Effectively identical |

Local copies retained for validation (validators depend on local binding layouts).

### Still local (no upstream equivalent or significant API differences)

| Shader | Domain | Suggested upstream module |
|--------|--------|--------------------------|
| `head_split.wgsl` | MHA | `barracuda::ops::mha` (fix S-03b first) |
| `head_concat.wgsl` | MHA | `barracuda::ops::mha` (fix S-03b first) |
| `xoshiro128ss.wgsl` | GPU PRNG | `barracuda::ops::prng` |
| `swarm_nn_scores.wgsl` | Swarm (015) | New — no upstream equivalent |

---

## Phase 5a/5b Shortcomings

Discovered during GPU `Tensor` validation across 23 scientific domains.

| # | Shortcoming | Severity | Root Cause | Status |
|---|-------------|----------|------------|--------|
| S-14 | Naive matmul hang (small square matrices) | Medium | Driver/binary complexity interaction (RTX 4070 Vulkan) | **Workaround**: A×B^T pattern |
| S-15 | Matmul hang when f32 elements ≤ 0.1 magnitude | Critical | WGPU/Vulkan driver bug (RTX 4070) — not sign or sparsity | **Root-caused**: data ≥ 0.5 avoids hang |
| S-16 | 2D transpose dispatch uses wrong divisor | High | `execute_2d()` uses 256 instead of tile size 16 | **FIXED**: `const TILE: u32 = 16` |

**S-15 clarification**: Phase 5a initially attributed the hang to negative values
or sparsity. Phase 5b root-caused it to element *magnitude*: any matrix where many
elements have `|x| ≤ 0.1` triggers the hang. The workaround (`rng.uniform() * 0.5 + 0.5`)
ensures all elements ≥ 0.5. This affects all matmul tiers, not just Naive.

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
- Library tests: **264 lib + 9 integration tests** (up from 255 lib)

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
