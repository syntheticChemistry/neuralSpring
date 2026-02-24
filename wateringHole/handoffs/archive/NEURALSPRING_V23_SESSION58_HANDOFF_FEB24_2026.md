# neuralSpring Session 58 Handoff — V23

**Date**: February 24, 2026
**ToadStool HEAD**: `9404fdb4`
**Previous**: V22 (Session 57 — S58/S59 sync + pow polyfill consolidation)

---

## What Changed

### 7 Dispatcher Methods Rewired to Upstream `domain_ops`

neuralSpring's `Dispatcher` now delegates 7 core math operations to
`barracuda::dispatch::domain_ops` instead of local `gpu_ops::*` implementations.
The upstream functions handle GPU/CPU routing with size-based thresholds:

| Dispatcher Method | Upstream Function |
|-------------------|-------------------|
| `mat_mul` | `barracuda::dispatch::matmul_dispatch` |
| `frobenius_norm` | `barracuda::dispatch::frobenius_norm_dispatch` |
| `transpose` | `barracuda::dispatch::transpose_dispatch` |
| `softmax` | `barracuda::dispatch::softmax_dispatch` |
| `l2_distance` | `barracuda::dispatch::l2_distance_dispatch` |
| `mean` | `barracuda::dispatch::mean_dispatch` |
| `variance` | `barracuda::dispatch::variance_dispatch` |

Each method falls back to local CPU implementation on upstream error.

### GpuDriverProfile Wired into Dispatcher

`GpuDriverProfile` (from `barracuda::device::driver_profile`, hotSpring-evolved)
is now built at Dispatcher initialization and exposes:

- `driver_profile()` — full driver/compiler/arch/FP64 detection
- `fp64_strategy()` — `Native` on compute-class GPUs, `Hybrid` on consumer
- `needs_pow_workaround()` — whether pow(f64,f64) needs polyfill

RTX 4070 profile: Ada arch, NvidiaPtxas, Throttled FP64 (1:64 ratio),
Hybrid strategy, WarpPacked eigensolve, pow workaround needed.

### New Validator

`validate_cross_spring_evolution` (10/10 PASS):
- 7 rewired method parity checks (CPU ↔ upstream dispatch)
- 2 driver profile detection checks
- Cross-spring throughput benchmark
- Cross-spring evolution lineage report

## Cumulative Rewiring Status

| Session | Functions Rewired | Target |
|---------|-------------------|--------|
| S56 | `graph_laplacian`, `disordered_laplacian`, `belief_propagation_chain`, `numerical_hessian` | `barracuda::linalg::graph`, `barracuda::numerical` |
| S57 | `patch_pow_to_polyfill` consolidated | `validation::patch_pow_to_polyfill` |
| S58 | `mat_mul`, `frobenius_norm`, `transpose`, `softmax`, `l2_distance`, `mean`, `variance` | `barracuda::dispatch::domain_ops` |

**Total**: 11 functions delegating to upstream BarraCUDA.

## Cross-Spring Evolution Provenance

| Spring | Contributions to BarraCUDA | Used by neuralSpring |
|--------|---------------------------|---------------------|
| hotSpring | df64_core, pow_f64, Fp64Strategy, GpuDriverProfile, Taylor trig, Lanczos | GpuDriverProfile, Fp64Strategy, pow polyfill |
| wetSpring | HMM, ODE bio (5), NMF, Anderson, Ridge | Anderson localization (spectral), HMM (phylo) |
| neuralSpring | ValidationHarness, batch_fitness, pairwise ops, eigh, KernelRouter | Absorbed upstream; local still uses own ValidationHarness |

## Quality Gates

| Gate | Result |
|------|--------|
| `cargo test --lib --release` | 478 PASS |
| `cargo test -p neural-spring-forge --lib` | 30 PASS |
| `cargo clippy (pedantic+nursery)` | 0 warnings |
| `cargo fmt --check` | clean |
| `validate_all` | 145/146 PASS (1 pre-existing logsumexp) |

## For ToadStool/BarraCUDA Team

### Observations

1. **Size-based dispatch thresholds**: upstream `domain_ops` uses `DispatchConfig`
   with size thresholds — for small inputs (n < ~256), CPU is used even when GPU
   is available. This is good behavior for small science workloads.

2. **GPU matmul parity**: max diff 2.3e-4 for 64x64 due to accumulation order.
   Expected for parallel reduction vs sequential. Tolerance 1e-3 is appropriate.

3. **Dispatcher init now logs f64 strategy**: `[dispatch] GPU available: NVIDIA
   GeForce RTX 4070 (DiscreteGpu, Vulkan, f64=Hybrid)`.

### Potential Future Upstream Items

- `boltzmann_dispatch` — Boltzmann distribution (temperature-scaled softmax)
- `pearson_dispatch` — Pearson correlation
- `shannon_entropy_dispatch` — Shannon entropy
- `chi_squared_dispatch` — Chi-squared statistic
- `eigh_dispatch` — Symmetric eigensolve with driver-adaptive strategy

---

*neuralSpring V23 Session 58 — 7 methods rewired to upstream domain_ops, GpuDriverProfile wired in, 11 total functions delegating to upstream BarraCUDA.*
