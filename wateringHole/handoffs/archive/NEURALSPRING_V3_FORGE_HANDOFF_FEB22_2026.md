# neuralSpring V3 Forge Handoff

**Date**: February 22, 2026
**From**: neuralSpring (ecoPrimals)
**To**: ToadStool / BarraCUDA team
**Supersedes**: `NEURALSPRING_V2_CONSOLIDATED_HANDOFF_FEB21_2026.md` (archived)

---

## Summary

neuralSpring now packages all 16 WGSL shaders, binding layouts, and dispatch
routing in a dedicated `metalForge/forge/` Rust crate (`neural-spring-forge`).
This follows the hotSpring `metalForge/forge` pattern: a single crate that
ToadStool can absorb from directly.

**What changed since V2:**

1. Created `metalForge/forge/` Rust crate with 4 modules and 18 unit tests
2. Migrated all 16 WGSL `pub const` exports from scattered library modules to `forge::shaders`
3. Documented binding layouts and dispatch geometry as Rust code in `forge::bindings`
4. Codified GPU vs CPU crossover heuristics in `forge::dispatch`
5. Formalized the `Gpu -> WgpuDevice` bridge in `forge::bridge`
6. Updated all root docs, whitePaper, and `ABSORPTION_MANIFEST.md` with current counts

---

## Forge Crate Structure

```
metalForge/forge/
  Cargo.toml          # deps: barracuda (path), wgpu 22, bytemuck
  src/
    lib.rs            # Crate root — absorption-friendly layout for ToadStool
    shaders.rs        # 16 WGSL shader sources (single source of truth)
    bindings.rs       # Binding layout structs + dispatch geometry per shader
    dispatch.rs       # GPU vs CPU crossover routing (empirical thresholds)
    bridge.rs         # Gpu <-> barracuda::device::WgpuDevice bridge
```

### How to Absorb a Shader

1. Copy the WGSL source from `forge::shaders::SHADER_NAME`
2. Copy the binding layout from `forge::bindings::SHADER_NAME`
3. Create a `barracuda::ops::*` wrapper using the layout's `entry_point`, `workgroup_size`, and `dispatch_note`
4. neuralSpring switches to the upstream op and removes the local shader

---

## Shader Catalog (16 shaders, all validated)

| Shader | Entry Point | WG Size | Bindings | Validation | Checks | Target |
|--------|------------|---------|----------|------------|--------|--------|
| `HMM_FORWARD_LOG` | `hmm_forward_log` | 256 | 5 (4 storage + 1 uniform) | `validate_gpu_hmm_forward` | 13/13 | `ops::hmm` |
| `BATCH_FITNESS_EVAL` | `batch_fitness_linear` | 256 | 4 | `validate_gpu_batch_fitness` | 20/20 | `ops::batch_gemm` |
| `RK4_PARALLEL` | `rk4_step` | 64 | 5 (incl. scratch) | `validate_gpu_rk4` | 8/8 | `ops::ode` |
| `MEAN_REDUCE` | `mean_reduce` | 1 | 3 | `validate_gpu_pure_workload` | 7/7 | `pipeline::ReduceScalarPipeline` |
| `PAIRWISE_JACCARD` | `pairwise_jaccard` | 256 | 3 | `validate_gpu_pangenome` | 6/6 | `ops::pairwise_distance` |
| `LOCUS_VARIANCE` | `locus_variance` | 256 | 3 | `validate_gpu_meta_pop` | 7/7 | `ops::VarianceReduceF64` |
| `SPATIAL_PAYOFF` | `spatial_payoff` | 256 | 3 | `validate_gpu_game_theory` | 5/5 | `ops::stencil` |
| `BATCH_IPR` | `batch_ipr` | 256 | 3 | `validate_gpu_anderson` | 5/5 | `ops::batch_reduce` |
| `PAIRWISE_HAMMING` | `pairwise_hamming` | 256 | 3 | `validate_gpu_sate` | 5/5 | `ops::pairwise_distance` |
| `PAIRWISE_L2` | `pairwise_l2` | 256 | 3 | `validate_gpu_modes` | 15/15 | `ops::pairwise_distance` |
| `MULTI_OBJ_FITNESS` | `multi_obj_fitness` | 256 | 3 | `validate_gpu_directed` | 6/6 | `ops::batch_gemm` |
| `SWARM_NN_FORWARD` | `swarm_nn_forward` | 256 | 4 | `validate_gpu_swarm` | 9/9 | `ops::batch_gemm` |
| `HILL_GATE` | `hill_gate` | 256 | 4 | `validate_gpu_signal` | 9/9 | `ops::elementwise` |
| `HEAD_SPLIT` | `head_split` | 256 | 3 | `validate_mha_gpu` | 5/5 | `ops::mha` |
| `HEAD_CONCAT` | `head_concat` | 256 | 3 | `validate_mha_gpu` | 5/5 | `ops::mha` |
| `XOSHIRO128SS` | `generate` | 256 | 3 | `validate_gpu_prng` | 5/5 | `ops::prng` |

---

## Outstanding Shortcomings

| ID | Description | Status | Impact |
|----|-------------|--------|--------|
| S-03b | Native `Tensor::multi_head_attention` GPU hang | Active workaround in `evolved::mha` | MHA uses matmul + head_split/concat shaders |
| S-12 | `eigh_f64` Jacobi accuracy gap at n>=8 | Local `eigh.rs` Householder+QR fix | Spectral analysis, Anderson localization |
| S-13 | `PooledBuffer` drop-before-completion race | Local `evolved::tensor_sync` fix | Sequential tensor ops |
| S-14 | Naive matmul driver hang (small square) | Documented, no local fix needed | Recommendation: remove Naive tier |

---

## Current Project State

| Metric | Value |
|--------|-------|
| Python checks | 206/206 PASS |
| Rust lib tests | 237 + 9 doc = 246 |
| Validation binaries | 81 |
| Bench binaries | 5 |
| Line coverage | 94.9% |
| WGSL shaders | 16 (in forge) |
| Forge tests | 18 |
| clippy | 0 warnings (pedantic + nursery) |
| Grand total | 1300+ validation checks |

---

## BarraCUDA APIs Used

| Category | Key APIs |
|----------|---------|
| Device | `WgpuDevice`, `new_cpu_relaxed()`, `new_gpu()` |
| Tensor | `Tensor`, `from_data`, `matmul`, `relu`, `gelu`, `softmax_wgsl`, `layer_norm_wgsl` |
| Statistics | `variance`, `pearson_correlation`, `covariance`, `norm_cdf/pdf/ppf` |
| Linear Algebra | `solve_f64`, `eigh_f64`, `cholesky_f64`, `lu_*`, `svd_*` |
| Special | `gamma`, `erf`, `bessel_*`, `chi_squared_*` |
| Optimization | `nelder_mead`, `bisect`, `brent` |
| Tensor f64 | `SumReduceF64`, `FusedMapReduceF64`, `NormReduceF64`, `VarianceReduceF64` |
| FFT | `Fft1D`, `Ifft1D`, `Fft1DF64`, `Rfft` |
| Staging | `StatefulPipeline`, `KernelDispatch` |
| Dispatch | `dispatch_for`, `DispatchTarget` |

---

## Absorption Priority

1. **Pairwise distance** (Hamming, Jaccard, L2) — 3 shaders, common pattern
2. **Batch fitness / NN forward** — 3 shaders, batch GEMM variant
3. **Head split/concat** — 2 shaders, fixes S-03b hang
4. **RK4 parallel** — 1 shader, multi-system ODE
5. **HMM forward** — 1 shader, log-domain forward pass
6. **Remaining** — spatial payoff, batch IPR, locus variance, Hill gate, mean reduce, PRNG

---

*neuralSpring forge handoff — following the hotSpring metalForge pattern.*
*Lifecycle: evolve → validate → export (forge crate) → handoff → ToadStool absorbs → retire.*
