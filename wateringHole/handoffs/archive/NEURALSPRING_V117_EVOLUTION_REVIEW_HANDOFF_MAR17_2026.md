# neuralSpring V117 → barraCuda / toadStool Evolution Review Handoff

**Date**: March 17, 2026
**From**: neuralSpring Session 166, V117
**To**: barraCuda / toadStool teams
**License**: AGPL-3.0-or-later
**Covers**: V116–V117, Sessions 165–166
**Supersedes**: V116 Ecosystem Absorption Handoff

## Executive Summary

- **Full evolution review** against barraCuda Sprint 6-7 and toadStool S146+
- **neuralSpring at scale**: 1155 lib + 75 playGround + 73 forge tests (1303 total), 267 binaries, 67 modules, 225 named tolerances
- **barraCuda usage**: 45+ submodules, 80+ functions, 216+ import files across neuralSpring
- **New wiring opportunities**: `WGSL_MEAN_REDUCE` re-export, `StatefulPipeline`, L-BFGS optimizer
- Zero regressions, zero clippy warnings (pedantic+nursery), zero unsafe, zero C dependencies

## Part 1 — barraCuda Sprint 6-7 Review

### What's New in barraCuda

| Sprint | Feature | neuralSpring Impact |
|--------|---------|---------------------|
| 7 | Smart module refactor (ode_bio split, gpu_hmc_types) | No action — internal |
| 7 | 28 new unit tests (3772 total) | Ecosystem confidence |
| 7 | 10 `mul_add()` FMA evolutions (RK45, spline, tridiag) | Confirms ecosystem-wide FMA adoption |
| 7 | Hardcoding → named constants (transport, discovery, quotas) | Pattern alignment with neuralSpring S164 |
| 6 | **`execute_gemm_ex(trans_a, trans_b)`** | Available for large-matrix A^T*A; neuralSpring keeps small n≤8 local |
| 6 | **`WGSL_MEAN_REDUCE` / `WGSL_MEAN_REDUCE_F64` re-export** | Future GPU pipelines can compose with upstream reduction |
| 6 | FAMILY_ID socket paths | Already used in `discover_socket()` since S162 |
| 6 | blake3 ecoBin compliance | Aligned — neuralSpring is ecoBin compliant |
| 5 | Typed `BarracudaError` | Replace remaining `Result<T, String>` at boundaries |
| 5 | `poll_until_ready` `&mut self` → `&self` | Simpler async readback in forge pipelines |

### neuralSpring's barraCuda Usage (Current)

| Category | Count | Key APIs |
|----------|-------|----------|
| Stats | 15+ | `variance`, `pearson_correlation`, `r_squared`, `rmse`, `nash_sutcliffe`, `dot`, `l2_norm`, `shannon`, `diversity`, `hydrology`, `evolution`, `jackknife` |
| Linalg | 12+ | `solve_f64_cpu`, `eigh_f64`, `cholesky_f64`, `lu_det`, `lu_solve`, `tridiagonal_solve`, `effective_rank`, `graph_laplacian`, `BatchedEighGpu`, `svd::*`, `gen_eigh::*` |
| Special | 10+ | `chi_squared_sf/cdf`, `gamma`, `erf/erfc`, `bessel_*`, `legendre`, `hermite`, `laguerre`, `factorial` |
| Tensor | 8+ | `matmul`, `transpose`, `sigmoid`, `tanh`, `softmax_dim`, `argmax_dim`, `layer_norm_wgsl` |
| ops::bio | 18+ | `BatchFitnessGpu`, `HmmBatchForwardF64`, `PairwiseHammingGpu`, `PairwiseJaccardGpu`, `PairwiseL2Gpu`, `LocusVarianceGpu`, `SpatialPayoffGpu`, `MultiObjFitnessGpu`, `BatchIprGpu`, `SwarmNnGpu`, `WrightFisherGpu` |
| FFT | 4 | `Fft1D`, `Fft1DF64`, `Ifft1D`, `Rfft` |
| Dispatch | 6+ | `matmul_dispatch`, `frobenius_norm_dispatch`, `variance_dispatch`, `transpose_dispatch`, `softmax_dispatch` |
| NN | 3 | `SimpleMlp`, `esn_v2`, `BandwidthTier` |

**Import scale**: 216+ files import from `barracuda::*`.

## Part 2 — What neuralSpring Needs (Action Items)

### barraCuda action: FMA adoption in CPU reference code

neuralSpring swept 14 `a * b + c` → `mul_add()` sites in S165. barraCuda Sprint 7 independently swept 10 sites. Recommend systematic sweep across `stats::`, `linalg::`, `numerical::` CPU reference paths to ensure all accumulation loops use FMA. Pattern:

```rust
// Before
sum += a * b;
// After
sum = a.mul_add(b, sum);
```

### barraCuda action: `WGSL_MEAN_REDUCE` absorption confirmation

neuralSpring's local `mean_reduce.wgsl` reference shader (in `validate_gpu_pure_workload.rs`) can be replaced by `barracuda::ops::{WGSL_MEAN_REDUCE, WGSL_MEAN_REDUCE_F64}` once neuralSpring evolves that binary. This is a future migration — documenting the availability.

### barraCuda action: L-BFGS wiring path

`barracuda::optimize::LbfgsGpu` exists but neuralSpring hasn't wired it yet. Target modules: `pinn` (Burgers equation optimizer), `loss_landscape` (Metropolis→L-BFGS for gradient-based exploration). This is a Tier A rewire opportunity.

### toadStool action: IPC proptest pattern adoption

neuralSpring added 8 property/fuzz tests for IPC resilience primitives (RetryPolicy bounds, CircuitBreaker state machine, rapid cycling, parse_capability_list fuzz, DispatchOutcome classify fuzz, extract_rpc_error fuzz). Recommend toadStool adopt similar invariant testing for its IPC server-side parsing. Key invariant: `delay ≤ max_delay` for all retry attempts.

### toadStool action: `StatefulPipeline` documentation

neuralSpring has HMM forward/backward/Viterbi chains and ODE integration loops that could benefit from `StatefulPipeline` for persistent GPU state across iterations. Current implementation uses discrete dispatch calls. Documentation of the `StatefulPipeline` API (expected trait bounds, state management lifecycle) would unblock neuralSpring wiring.

## Part 3 — Evolution Readiness Summary

### Ready for immediate GPU promotion (Tier A)

All 27 paper reproductions + 5 novel compositions have CPU+GPU validated paths. The remaining ~3% of math that isn't GPU-promoted consists of:
- Small-matrix operations (n≤8) intentionally kept as CPU references
- Seed/RNG initialization (inherently sequential)
- File I/O and validation harness scaffolding

### What stays local by design

| Item | Reason |
|------|--------|
| `mat_mul_transpose` (n≤8) | Dispatch overhead exceeds compute for tiny matrices |
| `softmax_rows` CPU reference | Analytical verification target for GPU softmax |
| Validation harness (`ValidationHarness`) | Test infrastructure, not compute |
| Tolerance registry | Compile-time constants, no runtime compute |
| IPC client/discovery | Networking, not numerical |

### What should migrate upstream (future)

| Item | Target | Blocker |
|------|--------|---------|
| Local WGSL mean reduce | `barracuda::ops::WGSL_MEAN_REDUCE` | None — available Sprint 6 |
| PINN L-BFGS optimizer | `barracuda::optimize::LbfgsGpu` | Wiring effort |
| HMM chain state | `staging::StatefulPipeline` | API documentation |
| Large matrix A^T*A | `execute_gemm_ex(trans_a=true)` | Only for n>8 workloads |

## Test Summary

| Category | Count |
|----------|-------|
| Library unit tests | 1155 |
| playGround unit tests | 75 |
| Forge unit tests | 73 |
| Integration tests | 9 |
| Doc tests | 13 |
| Validation binaries | 238 |
| Benchmark binaries | 18 |
| Property tests | 28 |
| IPC fuzz tests | 5 |
| Named tolerances | 225 |
| **Total test artifacts** | **1303 lib + 267 binaries** |

Quality: zero clippy (pedantic+nursery), zero fmt diffs, zero unsafe, zero C deps, MSRV 1.87, Edition 2024.

## Files Changed (S166)

| File | Change |
|------|--------|
| `README.md` | Count corrections, S166 entry |
| `CHANGELOG.md` | S166 entry |
| `EVOLUTION_READINESS.md` | S166 entry, count corrections |
| `CONTROL_EXPERIMENT_STATUS.md` | Count corrections |
| `DEPRECATION_MIGRATION.md` | Count corrections |
| `experiments/README.md` | S166 header, Exp 122 |
| `whitePaper/baseCamp/README.md` | S166 entry, count corrections |
| `specs/*.md` | Count corrections |
| `wateringHole/` | V116→archive, V117 published |

---

*AGPL-3.0-or-later — neuralSpring → barraCuda / toadStool*
