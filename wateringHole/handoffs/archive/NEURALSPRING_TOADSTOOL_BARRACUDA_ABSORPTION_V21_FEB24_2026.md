# neuralSpring → ToadStool/BarraCUDA Absorption Handoff V21

**Date**: February 24, 2026 (Session 56)
**From**: neuralSpring
**To**: ToadStool / BarraCUDA core team
**ToadStool HEAD**: `9404fdb4`
**Previous**: [V20 (Sessions 50–55)](archive/NEURALSPRING_TOADSTOOL_BARRACUDA_ABSORPTION_V20_FEB24_2026.md)
**License**: AGPL-3.0-or-later
**Purpose**: Confirm upstream absorption of 4 baseCamp functions, present new
absorption targets, and document learnings for BarraCUDA evolution.

---

## Executive Summary

Session 56 confirms that ToadStool `9404fdb4` correctly absorbed 4 neuralSpring
baseCamp functions into BarraCUDA. neuralSpring has rewired to use upstream and
all 478 lib tests + 155 validation binaries pass. We present 3 new absorption
targets (Dispatcher patterns, PCIe cost model, baseCamp GPU shaders) and
document 4 learnings for BarraCUDA's continued evolution.

---

## Part 1: Confirmed Absorptions (Session 56 Verified)

### 1.1 Functions Absorbed and Rewired

| Function | BarraCUDA Module | Tests Passing | API Match |
|----------|-----------------|---------------|-----------|
| `graph_laplacian` | `linalg::graph::graph_laplacian` | 23/23 | Exact |
| `disordered_laplacian` | `linalg::graph::disordered_laplacian` | 23/23 | Exact (heterogeneity vec) |
| `belief_propagation_chain` | `linalg::graph::belief_propagation_chain` | 21/21 | Exact |
| `numerical_hessian` | `numerical::numerical_hessian` | 27/27 | Exact |

### 1.2 Shaders Absorbed

| Shader | BarraCUDA Location | Status |
|--------|-------------------|--------|
| `xoshiro128ss.wgsl` | `ops::bio::xoshiro128ss` | Absorbed (was local) |
| `swarm_nn_scores.wgsl` | `ops::bio::swarm_nn` | Absorbed (was local) |

### 1.3 Still Local (neuralSpring-only)

| Shader | Reason |
|--------|--------|
| `batch_rk45.wgsl` | Domain-specific ODE integration |

---

## Part 2: New Absorption Targets

### Target A: Dispatcher Routing Patterns (HIGH)

neuralSpring's `Dispatcher` implements a `gpu_or_cpu` pattern that all Springs
need. Proposed BarraCUDA API:

```rust
pub async fn gpu_or_cpu<T>(
    device: Option<&WgpuDevice>,
    workload_bytes: u64,
    gpu_fn: impl AsyncFn(&WgpuDevice) -> T,
    cpu_fn: impl Fn() -> T,
) -> T
```

**Validation**: 29 Dispatcher methods, 89 dispatch-specific checks, all pass.

### Target B: PCIe Bandwidth Tier Model (MEDIUM)

Generic infrastructure for mixed-hardware cost modeling:

```rust
pub enum BandwidthTier { Pcie4X16, Pcie4X4, Pcie5X16, SharedMemory }
pub fn transfer_cost_for_tier(bytes: u64, tier: BandwidthTier) -> TransferCost;
pub fn chained_transfer_cost(bytes: u64, hop1: BandwidthTier, hop2: BandwidthTier) -> TransferCost;
pub fn compare_transfer_paths(bytes: u64, direct: BandwidthTier, hop1: BandwidthTier, hop2: BandwidthTier) -> (TransferCost, TransferCost, bool);
```

**Validation**: 36 checks in `validate_metalforge_pcie`, all pass.

### Target C: baseCamp GPU Shaders (EVOLVING)

These are validated via Dispatcher but could benefit from dedicated WGSL shaders:

| Primitive | Current Path | Shader Benefit |
|-----------|-------------|---------------|
| Symmetric matmul (W^T·W) | Tensor matmul (2 dispatches) | Fused single-dispatch |
| Numerical Hessian | CPU finite differences | GPU parallel O(n) differences |
| Graph Laplacian | CPU degree-adjacency | Sparse GPU (row-parallel) |
| Batch belief propagation | CPU GEMV chain | GPU batch GEMV pipeline |

---

## Part 3: Learnings for BarraCUDA Evolution

### 3.1 The Thin-Wrapper Pattern

When BarraCUDA absorbs a function, the Spring should keep a thin wrapper
that preserves the public API and delegates to upstream:

```rust
pub fn numerical_hessian(loss_fn: &dyn Fn(&[f64]) -> f64, params: &[f64], epsilon: f64) -> Vec<f64> {
    barracuda::numerical::numerical_hessian(loss_fn, params, epsilon)
}
```

This gives zero-cost migration, preserves test coverage, and allows Springs
to add domain-specific documentation or parameter validation.

### 3.2 Tolerance Constants Need Ecosystem-Wide Registry

neuralSpring uses 95 named, justified tolerance constants. Each has:
- A descriptive name (`GPU_VARIANCE_F64`, `PGM_NORMALIZATION_SUM`)
- A justified value with documentation explaining why
- Registration in a central registry with duplicate detection

BarraCUDA should consider hosting a shared tolerance registry so all Springs
use the same constants for the same operations.

### 3.3 f64 Graph Operations Are Critical

The `linalg::graph` module is used heavily by Sub-04 (PGMs) and Sub-05
(multi-agent coordination). Both require f64 for:
- Laplacian eigenvalue separation (f32 merges close eigenvalues)
- Belief propagation normalization (f32 underflows on long chains)
- Algebraic connectivity detection (second-smallest eigenvalue of Laplacian)

### 3.4 PCIe P2P Is Faster But Not Always Available

`detect_p2p()` currently returns `false` because `wgpu` doesn't expose PCI
BDF (Bus/Device/Function) information. When `wgpu` adds this, BarraCUDA
should expose P2P capability detection and auto-select P2P vs CPU-staged
routing. The cost model is already validated.

---

## Part 4: Cumulative Absorption Inventory

| Category | Count | Detail |
|----------|-------|--------|
| GPU typed ops used | 15 | matmul, variance, pearson, entropy, chi², eigh, softmax, boltzmann, hill, mean, sum, max, L2, KL, replicator |
| f64 reduction ops | 3 | VarianceReduceF64, CorrelationF64, FusedMapReduceF64 |
| CPU primitives | 20+ | linalg, stats, spectral, numerical, graph, bio |
| Functions rewired to upstream | 9 | 5 f64 ops (S53) + 4 baseCamp (S56) |
| Shaders absorbed | 20 | 19 upstream + 1 local (batch_rk45) |
| Shortcomings filed | 17 | 13 absorbed/fixed, 2 root-caused, 1 polyfill, 1 pending |
| Total validation checks | 2010+ | 206 Python + 1810+ Rust/GPU |

---

## Appendix: Validation Evidence

```
$ cargo test --lib -q
test result: ok. 478 passed; 0 failed; 0 ignored

$ cargo test -p neural-spring-forge --lib -q
test result: ok. 30 passed; 0 failed; 0 ignored

$ cargo clippy --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery
# 0 warnings

$ cargo run --release --bin validate_basecamp_dispatch
# 19/19 PASS

$ cargo run --release --bin validate_barracuda_parity
# 34/34 PASS — CPU↔GPU parity across all science domains

$ cargo run --release --bin validate_metalforge_pcie
# 36/36 PASS — PCIe tiers, chaining, substrate selection, bridge, live dispatch
```
