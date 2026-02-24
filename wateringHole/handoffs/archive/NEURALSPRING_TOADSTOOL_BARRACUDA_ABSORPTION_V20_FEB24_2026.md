# neuralSpring → ToadStool/BarraCUDA Absorption Handoff V20

**Date**: February 24, 2026 (Sessions 50–55)
**From**: neuralSpring
**To**: ToadStool / BarraCUDA core team
**License**: AGPL-3.0-or-later
**Purpose**: Comprehensive BarraCUDA evolution review, absorption targets, and
learnings relevant to ToadStool's evolution.

---

## Executive Summary

neuralSpring has evolved from 0→142 validation binaries using BarraCUDA across
25 scholarly papers + 5 baseCamp sub-theses. We consume 12 typed GPU ops,
3 f64 reduction ops, 18+ CPU primitives, and the full Tensor API. We've
discovered 17 shortcomings (S-01 through S-17), of which 13 are absorbed/fixed
upstream. We've wired metalForge mixed-hardware dispatch into the `Dispatcher`
and proven CPU↔GPU parity at machine epsilon across all baseCamp math.

**What ToadStool should absorb**: Mixed-hardware dispatch, 5 baseCamp primitives,
4 GPU shader candidates, S-17 `pow()` fix, and the `gpu_or_cpu` / `mixed_dispatch`
patterns.

---

## Part 1: BarraCUDA Evolution — What We Learned

### 1.1 The f32→f64 Journey

neuralSpring started with f32 Tensor ops and hit precision walls in:
- Eigendecomposition (spectral analysis of weight matrices)
- Pearson correlation (small-effect-size detection)
- Shannon entropy (probability distributions with near-zero values)
- Chi-squared tests (expected vs observed comparison)

**Lesson**: Science workloads need f64. The 3 f64 reduction ops
(`VarianceReduceF64`, `CorrelationF64`, `FusedMapReduceF64`) were critical.
ToadStool should prioritize f64 typed ops for science domains.

### 1.2 The Dispatch Crossover

Empirical finding (RTX 4070, Vulkan): **GPU dispatch overhead is ~1.5ms**.
This is the fixed cost of `queue.submit()` + buffer readback. GPU compute
time is negligible at all tested scales (5888 CUDA cores).

**Crossover rule**: CPU wins when total CPU work < 1.5ms; GPU wins above.
`StatefulPipeline` and `TensorSession` amortize the fixed cost for multi-pass
workloads.

**Lesson**: The dispatch overhead, not compute, determines the CPU/GPU
crossover. For small matrices (< 64×64), CPU is always faster unless the
workload is batched.

### 1.3 The Multi-GPU Portability Story

141 validators produce bit-identical results on:
- RTX 4070 (Ada Lovelace, proprietary NVIDIA Vulkan)
- TITAN V (GV100, NVK open-source Vulkan)

Two generations apart, different driver stacks, same WGSL source, same results.
The exceptions are:
- `pow(f64)` crashes on both (S-17) — needs polyfill
- Naive matmul hangs for small N or low-magnitude elements (S-14/S-15)
- `logsumexp` has a pre-existing driver issue (1 validator)

**Lesson**: WGSL portability works. The few failure modes are all in
transcendental functions and edge-case dispatch, not in the core compute.

### 1.4 The Mixed-Hardware Cost Model

We modeled GPU↔CPU and GPU↔NPU transfers using PCIe 4.0 bandwidth constants:

| Path | Bandwidth | Latency | 1MB Transfer |
|------|-----------|---------|-------------|
| GPU↔CPU (x16) | 31.5 GB/s | 2 µs | ~35 µs |
| GPU↔NPU (x4, staged) | 7.9 GB/s | 7 µs | ~140 µs |
| GPU↔NPU (x4, P2P) | 7.9 GB/s | 2 µs | ~135 µs |

P2P detection is stubbed (wgpu doesn't expose PCI BDF). When it does,
real IOMMU group comparison via `/sys/bus/pci/devices/{BDF}/iommu_group`
will enable true P2P.

**Lesson**: CPU staging adds 5µs latency per hop. For large workloads this
is negligible; for realtime inference chains it matters. P2P bypass
(GPU→NPU direct via PCIe) eliminates the CPU round-trip entirely.

---

## Part 2: What ToadStool Should Absorb

### 2.1 Immediate (One-Line Fixes)

**S-17: `pow(f64)` polyfill** — Add `.replace("pow(", "pow_f64(")` to
`patch_exp_log_in_code` in `barracuda/src/shaders/precision/mod.rs`:

```rust
fn patch_exp_log_in_code(code: &str) -> String {
    code.replace("exp(", "exp_f64(")
        .replace("log(", "log_f64(")
        .replace("pow(", "pow_f64(")  // S-17: native pow(f64) crashes NVVM/NAK
}
```

Also fix `hill_f64.wgsl` (element-wise Hill) — same native `pow(f64)` pattern.

**`Tensor::mean()` double-divide** — The current `mean()` implementation
divides twice. Fix in `ops/mean.rs`.

### 2.2 Mixed-Hardware Dispatch Module

Absorb `metalForge/forge/src/{dispatch,mixed,pcie_bridge}.rs` into
`barracuda::unified_hardware`:

| Source File | Target Module | Lines |
|-------------|--------------|-------|
| `dispatch.rs` | `barracuda::dispatch` | ~205 |
| `mixed.rs` | `barracuda::unified_hardware::routing` | ~152 |
| `pcie_bridge.rs` | `barracuda::unified_hardware::transfer` | ~97 |

The `Dispatcher::mixed_dispatch()` pattern in `gpu_dispatch/mod.rs` shows
how to wire these into an ergonomic API. Key insight: return both the result
and the substrate decision for observability.

### 2.3 General-Purpose Science Primitives

From baseCamp — these are tested across 128 checks and ready for upstream:

| Primitive | Signature | Domain |
|-----------|-----------|--------|
| `graph_laplacian` | `(adjacency: &[f64], n: usize) -> Vec<f64>` | `D - A` from any adjacency matrix |
| `effective_rank` | `(eigenvalues: &[f64]) -> f64` | Entropy of normalized eigenvalues |
| `empirical_spectral_density` | `(eigenvalues: &[f64], n_bins: usize) -> (Vec<f64>, Vec<f64>)` | Eigenvalue histogram |
| `numerical_hessian` | `(f: impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64>` | Central finite differences |
| `belief_propagation_chain` | `(transitions: &[Vec<f64>], ...) -> Vec<f64>` | Multi-layer message passing |

### 2.4 GPU Shader Candidates

| Shader | Description | Priority | Template |
|--------|-------------|----------|----------|
| `symmetrize.wgsl` | `out[i,j] = (A[i,j] + A[j,i]) / 2` — Hamiltonian construction | High | `transpose.wgsl` |
| `hessian_column.wgsl` | Parallel finite differences per dimension — loss landscapes | High | `batch_fitness_eval.wgsl` |
| `histogram.wgsl` | Atomic histogram binning of eigenvalues — spectral analysis | Medium | New (workgroup atomics) |
| `laplacian.wgsl` | Row-sum diagonal, subtract adjacency — graph analysis | Medium | `spatial_payoff.wgsl` |
| `metropolis.wgsl` | Parallel MCMC chains with acceptance — Boltzmann sampling | Low | `wright_fisher_step.wgsl` |

---

## Part 3: Learnings for ToadStool's Evolution

### 3.1 What Worked

1. **The typed op pattern** — `VarianceReduceF64`, `CorrelationF64`, etc. are
   the right abstraction. Each takes a `device: Arc<WgpuDevice>` + data,
   returns a scalar. No configuration, no pipelines, no bind groups exposed
   to consumers. This is what Springs want.

2. **The `gpu_or_cpu` pattern** — A single closure-based dispatch function
   that tries GPU first and falls back to CPU. Every Spring (hot, wet, neural)
   independently evolved this pattern. ToadStool should provide it as a
   first-class utility.

3. **The `compile_shader_f64` + polyfill pipeline** — Automatically injecting
   `math_f64.wgsl` functions when f64 transcendentals are detected is elegant.
   But it needs to cover `pow()` (S-17) and eventually `atan2()`, `sinh()`, etc.

4. **Cross-spring shader absorption** — The "evolve locally → validate → hand off
   → absorb upstream → retire" lifecycle works. 13 of neuralSpring's 21 shaders
   are now upstream. The 2 remaining local shaders (`head_split`, `head_concat`)
   are MHA workarounds that will retire when native MHA stabilizes.

5. **ValidationHarness + exit 0/1** — Battle-tested across 142 binaries on 2 GPUs.
   `check_bool`, `check_abs`, `check_rel`, `finish()` is the right minimal API
   for validation binaries.

### 3.2 What We'd Do Differently

1. **Start with f64** — We wasted time debugging f32 precision issues that
   vanished when switching to f64. For science workloads, f64 should be the
   default, with f32 as an optimization for deployment.

2. **Test the polyfill pipeline exhaustively** — S-17 (`pow()`) was only
   discovered in Session 52 because no shader had used `pow(f64)` until
   HillGate. A systematic test of all WGSL transcendentals through the
   polyfill pipeline would have caught this earlier.

3. **Expose PCI topology** — `detect_p2p()` returns `false` because wgpu
   doesn't expose PCI BDF. For mixed-hardware dispatch, real P2P detection
   is essential. Consider `VK_EXT_external_memory_host` or sysfs probing.

### 3.3 Patterns for All Springs

| Pattern | Description | Validated Across |
|---------|------------|-----------------|
| `gpu_or_cpu(name, gpu_fn, cpu_fn)` | Closure-based GPU/CPU dispatch | 142 binaries |
| `mixed_dispatch(name, compute_us, bytes, ...)` | Cost-model substrate routing | 14 checks |
| `exit_no_gpu()` | `REQUIRE_GPU=1` → exit 1, else skip | All CI |
| `baseline_path(rel)` | `CARGO_MANIFEST_DIR`-relative paths | All data loaders |
| `ValidationHarness` | check_bool / check_abs / finish | 142 binaries |
| `tolerances::*` | Centralized named constants | 90+ constants |

---

## Part 4: BarraCUDA Shortcomings Status

| # | Shortcoming | Severity | Status |
|---|-------------|----------|--------|
| S-01 | Per-op submission | Critical | **ABSORBED** (`TensorSession`) |
| S-02 | Naive matmul | Critical | **ABSORBED** (`KernelRouter`) |
| S-03 | MHA z-dispatch | High | **ABSORBED** |
| S-04 | Softmax uniform size | Medium | **ABSORBED** |
| S-05 | Params f32-only | Medium | **ABSORBED** |
| S-06 | Params struct | Medium | **ABSORBED** |
| S-07 | `from_buffer` private | Low | **ABSORBED** |
| S-08 | CPU→GPU round-trip | Medium | **ABSORBED** |
| S-09 | Double allocation | Medium | **ABSORBED** |
| S-10 | No CPU adapter | Low | **ABSORBED** (`new_cpu_relaxed()`) |
| S-11 | Missing fused ops | Medium | **ABSORBED** |
| S-12 | Shader compile errors | High | **ABSORBED** |
| S-13 | PooledBuffer race | Critical | **FIXED** upstream |
| S-14 | Naive matmul hang (small N) | Medium | **WORKAROUND** (A×B^T pattern) |
| S-15 | Matmul hang (low magnitude) | Medium | **WORKAROUND** (data ≥ 0.5) |
| S-16 | Transpose dispatch | Low | **FIXED** |
| S-17 | `pow(f64)` crash | High | **WORKAROUND** (polyfill) — needs one-line fix |

---

## Part 5: Open Data & Reproducibility

Every paper and experiment in neuralSpring uses open data:

| Category | Count | Data Source |
|----------|-------|-------------|
| Synthetic/analytical | 20 papers | In-code generation, deterministic seed (42) |
| Open API | 4 studies | Open-Meteo ERA5 (CC BY 4.0) |
| Public dataset | 1 study | MNIST (CC BY-SA 3.0) |
| Open-source reference | 3 studies | GitHub (MIT / Apache-2.0) |
| baseCamp | 5 sub-theses | All synthetic, deterministic seed (42) |

**Zero** proprietary models. **Zero** API keys. **Zero** paywalled data.
Every result is reproducible with `cargo run --release --bin validate_all`.

---

*neuralSpring V20 ToadStool absorption handoff — 142 binaries, 1950+ checks,
12+3 typed GPU ops, 18+ CPU primitives, mixed-hardware dispatch wired,
S-17 fix ready, 5 baseCamp primitives ready for upstream. All open data.*
