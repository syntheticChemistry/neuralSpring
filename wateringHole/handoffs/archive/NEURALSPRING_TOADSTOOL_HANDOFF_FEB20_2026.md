# neuralSpring → ToadStool: Consolidated Handoff — 11 Shortcomings, Shader Designs, GPU Promotion

**Date:** 2026-02-20 (consolidated)
**From:** neuralSpring (ML / isomorphic learning / scholarly reproduction Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**ToadStool reviewed:** commit `dc540afd` (Session 25, Feb 20, 2026)
**Supersedes:** All three prior handoffs (Feb 19 ML, Feb 19 Shader, Feb 20 Phase 0++)

---

## Executive Summary

neuralSpring has completed **23 experiments** across 5 scientific disciplines
with **728/728 PASS** (190 Python + 167 Rust native + 248 BarraCUDA primitives
+ 123 BarraCUDA CPU ports). Phase 2 (BarraCUDA CPU ports) is **complete** —
all 13 Phase 0++ modules validated against BarraCUDA CPU math primitives.

**UPDATE (Feb 20, 2026 — ToadStool `dc540afd`):**

**All 11 neuralSpring shortcomings (S-01 through S-11) are now ABSORBED.**
Key commit: `fbedd222` (TensorSession ML ops — S-01/S-11). neuralSpring
has rewired `validate_barracuda_tensor` from evolved workarounds to native
BarraCUDA APIs (90/90 PASS), rewired `gpu.rs` CPU path to
`WgpuDevice::new_cpu_relaxed()`, and documented `src/evolved/` (~3046 LOC)
for retirement. The remaining active evolution is `hmm_forward_gpu` (metalForge
shader, not a workaround). S-12 (`eigh_f64` accuracy gap) is still outstanding.

**New ToadStool capabilities relevant to neuralSpring:**
- `StatefulPipeline` — iterative GPU-resident simulation (directly usable for EA loops, ODE integration)
- `ReduceScalarPipeline` — two-pass GPU reduction returning 8 bytes (usable for fitness evaluation)
- `KernelRouter` — workload→hardware routing infrastructure (wire matmul tiering into this)
- `GpuDriverProfile` — already imported into `MatmulConfig`

---

## Part 1: The 11 Shortcomings (Priority-Ordered)

### Tier 1 — Critical Performance

#### S-01: Per-Op Command Submission (46–78× penalty)

Each `Tensor` operation creates its own `CommandEncoder`, dispatches one
compute pass, and calls `queue.submit()`. A 9-op MLP submits 9 command
buffers at ~200 µs each = 1.8 ms dispatch for ~5 µs compute.

**Local fix:** `src/evolved/fused_pipeline.rs` — pre-compile all shaders,
pre-allocate all intermediate buffers, pre-create all bind groups once.
Record all compute passes into one `CommandEncoder`, submit once.

**Result:** MLP 92 µs (43.6×), Transformer 174 µs (76.6×) vs per-op.

**Suggested upstream:** Extend `TensorSession` ops from `{Add, Mul, Fma, Scale}`
to include `{MatMul, ReLU, GELU, LayerNorm, Softmax, Attention}`. The session
already batches into one encoder — just wire the missing ops. This retires
`fused_pipeline.rs`, `fused_mlp.rs`, and `fused_transformer.rs`.

**ToadStool note:** `StatefulPipeline` demonstrates this exact pattern for MD
workloads. The `KernelDispatch` struct is the right abstraction — extend it to
ML ops.

---

#### S-02: Naive Matmul — Zero Cache Reuse (CPU 3× slower than Python)

`matmul.wgsl` reads K elements from global memory per output element — zero
tile reuse. NumPy calls OpenBLAS with hand-tuned cache-tiled GEMM. The naive
shader compiles to LLVM IR that thrashes cache at every K-step.

**Local fix:** 4-tier `DeviceCapabilities`-driven shader router in
`fused_pipeline.rs`:

| Condition | Shader | Tile | Key Optimization |
|-----------|--------|------|-----------------|
| M or N < threshold | `matmul.wgsl` (naive) | none | Safe for small M |
| CPU, large M,N | `matmul_cpu_tiled.wgsl` | 32×32 | Double-buffered, 8×4 µkernel |
| GPU, small M,N | `matmul_tiled.wgsl` | 16×16 | High SM occupancy |
| GPU, large M,N (≥256) | `matmul_gpu_evolved.wgsl` | 32×32 | Double-buffered, 2×2 µkernel |

**Results (CPU beats single-thread Python at crossover):**

| Scale | FLOPs | Py(1t) | CPU | GPU | CPU/Py | GPU/Py |
|-------|-------|--------|-----|-----|--------|--------|
| MLP large | 3.1M | 3.0 ms | **2.7 ms** | **178 µs** | **1.1× faster** | 16.8× |
| TF medium | 103M | 59 ms | **15.1 ms** | **566 µs** | **3.9× faster** | 104× |

**Suggested upstream:**
1. Add `matmul_cpu_tiled.wgsl` and `matmul_gpu_evolved.wgsl` to `barracuda/src/shaders/math/`
2. Wire `DeviceCapabilities` into `KernelRouter::route()` for matmul variant selection
3. Both shaders use the same binding layout as existing `matmul.wgsl` — drop-in

**Shader designs delivered:** `neuralSpring/src/evolved/matmul_cpu_tiled.wgsl` (263 lines),
`neuralSpring/src/evolved/matmul_gpu_evolved.wgsl` (302 lines). Ready for copy.

---

### Tier 2 — Correctness Bugs

#### S-03: MHA Projection Dispatch Bug (z-dimension)

**File:** `barracuda/src/ops/mha/projections.rs` lines 165–167

```rust
let workgroups_z = params.seq_len.div_ceil(16);  // BUG
```

Shader uses `@workgroup_size(16, 16, 1)` — z workgroup size is 1. With
`seq_len=8`, `div_ceil(16)=1` dispatches only `global_id.z=0`. Positions 1–7
produce zeros.

**Fix (one line each):**
```rust
// project_with_head_split:
let workgroups_z = params.seq_len;       // was .div_ceil(16)
// concat_and_project:
let workgroups_z = params.d_model;       // was .div_ceil(16)
```

**Local fix:** `src/evolved/mha.rs` decomposes MHA into matmul + CPU head-split
+ `attention()` + matmul. Fused pipeline uses GPU-resident head-split/concat shaders.

---

#### S-04: Softmax on Pooled Buffers (incorrect normalization)

**File:** `barracuda/src/shaders/activation/softmax_simple.wgsl`

```wgsl
let N = arrayLength(&input);  // physical buffer, not logical tensor
```

When pool returns an oversized buffer (64 elements for a 10-element tensor),
softmax normalizes over 64 elements. Extra zeros contribute `exp(0 - max_logit)`
to the denominator, corrupting probabilities.

**Fix:** Pass logical size via uniform buffer:
```wgsl
struct Params { logical_size: u32, }
@group(0) @binding(2) var<uniform> params: Params;
// Use params.logical_size instead of arrayLength(&input)
```

**Local fix:** Re-upload logits before softmax to force exact-size buffer.

---

#### S-05: `leaky_relu_wgsl` Params Mismatch (wgpu panic)

**File:** `barracuda/src/ops/leaky_relu_wgsl.rs` line 46

Rust `Params` has `{ size: u32 }` (4 bytes). WGSL expects
`{ size: u32, negative_slope: f32 }` (8 bytes). Causes wgpu validation panic.

**Fix:**
```rust
#[repr(C)]
struct Params { size: u32, negative_slope: f32 }
let params = Params { size: size as u32, negative_slope };
```

---

#### S-06: `elu_wgsl` Params Mismatch (same pattern)

**File:** `barracuda/src/ops/elu_wgsl.rs` line 46

Same as S-05. Rust has `{ size: u32 }`, WGSL expects `{ size: u32, alpha: f32 }`.

**Fix:** Add `alpha: f32` to Rust `Params`.

---

### Tier 3 — Performance (Round-Trips)

#### S-07: `Tensor::from_buffer` is `pub(crate)` (forces 2 round-trips)

**File:** `barracuda/src/tensor.rs` line 90

External crates cannot construct a `Tensor` from an existing `wgpu::Buffer`.
Forces `layer_norm_wgsl` and `log_softmax_wgsl` to `read_buffer` (GPU→CPU)
then `Tensor::new()` (CPU→GPU) per op.

**Fix (one character):**
```rust
pub fn from_buffer(buffer: wgpu::Buffer, shape: Vec<usize>, device: Arc<WgpuDevice>) -> Self
```

This single change retires S-08 and S-09 below.

---

#### S-08: `layer_norm_wgsl` GPU→CPU→GPU Round-Trip (5× penalty)

**File:** `barracuda/src/ops/layer_norm_wgsl.rs` lines 179–182

```rust
let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;
Ok(Tensor::new(output_data, shape.to_vec(), device.clone()))
```

**Fix:** `Ok(Tensor::from_buffer(output_buffer, shape.to_vec(), device.clone()))`
(requires S-07 first)

**Local fix:** `src/evolved/layer_norm.rs` — same shader, returns raw buffer.
Stock: 1.7 ms GPU. Evolved: 329 µs GPU.

---

#### S-09: `log_softmax_wgsl` GPU→CPU→GPU Round-Trip (same pattern)

**File:** `barracuda/src/ops/log_softmax_wgsl.rs` lines 175–178

Same as S-08. Same fix. Same local evolution in `src/evolved/log_softmax.rs`.

---

#### S-10: `WgpuDevice::new_cpu()` Requires `science_limits()` (blocks CPU)

**File:** `barracuda/src/device/wgpu_device/creation.rs` line 30

`science_limits()` requests 512 MB `max_storage_buffer_binding_size`. llvmpipe
caps at 128 MB. `new_cpu()` always fails on standard CPU software rasterizers.

**Fix:** Add fallback:
```rust
pub async fn new_cpu_relaxed() -> Result<Self> {
    // Use adapter.limits() instead of science_limits()
}
```

Or make `new_cpu()` fall back to adapter limits when `science_limits()` fails.

**Local fix:** `src/gpu.rs` `create_relaxed()` uses `Limits::downlevel_defaults()`.

---

### Tier 4 — Low Priority

#### S-11: `TensorSession` Limited to `{Add, Mul, Fma, Scale}`

**File:** `barracuda/src/session.rs` lines 92–118

`SessionOp` enum only has 4 variants. ML inference needs `MatMul`, `ReLU`,
`GELU`, `LayerNorm`, `Softmax`, `Attention`.

**Fix:** Extend enum and wire through session batching. The session already
batches into one encoder — just add the missing op variants and shader pipelines.

This is the highest-impact upstream change — it would retire the entire
`src/evolved/` directory (7 modules, ~1500 lines).

---

## Part 2: WGSL Shader Designs for Upstream

### 2.1 Delivered Shaders (ready to copy)

| Shader | Location | Lines | Purpose |
|--------|----------|-------|---------|
| `matmul_cpu_tiled.wgsl` | `src/evolved/` | 263 | CPU: 32×32 double-buffered, vec4, 8×4 µkernel |
| `matmul_gpu_evolved.wgsl` | `src/evolved/` | 302 | GPU: 32×32 double-buffered, vec4, 2×2 µkernel |
| `HEAD_SPLIT_WGSL` | `fused_pipeline.rs` inline | ~30 | `[seq, d_model]` → `[heads, seq, d_head]` |
| `HEAD_CONCAT_WGSL` | `fused_pipeline.rs` inline | ~30 | Reverse of head-split |
| `BATCHED_ATTENTION_WGSL` | `fused_pipeline.rs` inline | ~60 | Fused QK^T/√d → softmax → ·V |

All use same binding layout as existing BarraCUDA shaders. Drop-in additions.

### 2.2 Phase 0++ GPU Promotion Shader Designs (proposed)

These are the 7 new algorithmic patterns from the 23-paper catalog. Each maps
to a concrete WGSL kernel design.

#### `batch_gemv.wgsl` — Population Fitness Evaluation (Papers 011–015)

```wgsl
// Dispatch: (pop_size, genome_len/16, niche_count)
// Each thread computes fitness[pop][niche] = Σ_j genome[pop][j] * landscape[niche][j]
@group(0) @binding(0) var<storage, read> genomes: array<f32>;      // [pop_size × genome_len]
@group(0) @binding(1) var<storage, read> landscape: array<f32>;    // [niche_count × genome_len]
@group(0) @binding(2) var<storage, read_write> fitness: array<f32>; // [pop_size × niche_count]
@group(0) @binding(3) var<uniform> params: BatchGemvParams;        // pop_size, genome_len, niche_count

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // gid.x = population index, gid.y = niche index (tiled)
    // Shared-memory tiling of landscape rows for reuse across population
}
```

**Serves:** All 5 Dolson papers. Population 100–500, genome 8–50, niches 1–5.
**BarraCUDA integration:** Wire into `KernelRouter` as `ComputeWorkload::BatchGemv`.

#### `hmm_forward_log.wgsl` — HMM Forward Chain (Papers 016–018)

```wgsl
// Log-domain forward: log(α_t) = log(A^T · α_{t-1}) + log(B · o_t)
// Sequential over T, parallel over states. Uses log-sum-exp for numerical stability.
// For T=500, N=2 states: 500 sequential 2×2 matmuls.
// Key optimization: ping-pong between two N-element buffers (no allocation per step).
@group(0) @binding(0) var<storage, read> log_trans: array<f32>;     // [N × N] log transition
@group(0) @binding(1) var<storage, read> log_emit: array<f32>;      // [T × N] log emission
@group(0) @binding(2) var<storage, read_write> alpha: array<f32>;   // [2 × N] ping-pong
@group(0) @binding(3) var<uniform> params: HmmParams;               // T, N, current_t

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Each thread computes one state's forward probability
    // log-sum-exp over all source states, add emission
}
```

**Serves:** Papers 016–018. Reuses `ReduceScalarPipeline` for final log-likelihood.
**Integration:** Dispatch T times from `StatefulPipeline::run_iterations()`.

#### `pairwise_distance.wgsl` — Sequence Distance Matrix (Paper 017)

```wgsl
// One thread per pair (i,j), i < j. Computes Hamming or Jukes-Cantor distance.
// For N=100 sequences of length L=500: 4950 pairs, trivially parallel.
@group(0) @binding(0) var<storage, read> sequences: array<u32>;     // [N × L] packed bases
@group(0) @binding(1) var<storage, read_write> distances: array<f32>; // [N × N]
@group(0) @binding(2) var<uniform> params: DistParams;              // N, L, metric

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    // Decode (i, j) from triangular index
    // Count mismatches, apply JC correction: d = -3/4 * ln(1 - 4p/3)
}
```

**Serves:** Paper 017 (SATé alignment). No synchronization needed. Pure embarrassingly parallel.

#### `rk4_batch.wgsl` — Parallel ODE Integration (Papers 020–021)

```wgsl
// Batch RK4 for M independent ODE systems, each with D state variables.
// Each thread handles one system. 4 stages per kernel launch.
// RHS is elementwise: Hill functions + linear degradation.
@group(0) @binding(0) var<storage, read_write> state: array<f32>;   // [M × D]
@group(0) @binding(1) var<storage, read> params_ode: array<f32>;    // [M × P] per-system parameters
@group(0) @binding(2) var<uniform> params: Rk4Params;               // M, D, P, dt

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sys = gid.x;
    // k1 = dt * f(state)
    // k2 = dt * f(state + k1/2)
    // k3 = dt * f(state + k2/2)
    // k4 = dt * f(state + k3)
    // state += (k1 + 2*k2 + 2*k3 + k4) / 6
}
```

**Serves:** Papers 020–021. Dispatch N_steps times from `StatefulPipeline`.
State stays GPU-resident. Only scalar readback for convergence check.

#### `stencil_1d.wgsl` — Spatial Neighborhood Average (Paper 019)

```wgsl
// 1D stencil: out[i] = average(in[i-r..i+r]) for cooperation/defection grids.
// Shared-memory halo exchange pattern.
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: StencilParams;  // N, radius

var<workgroup> tile: array<f32, 288>;  // 256 + 2*16 halo

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(global_invocation_id) gid: vec3<u32>) {
    // Load center + halo into shared memory
    // Compute average of radius-neighborhood
}
```

**Serves:** Paper 019 spatial cooperation. Could reuse BarraCUDA's conv1d.

#### `tridiag_eigh.wgsl` — Tridiagonal Eigensolver (Papers 022–023)

```wgsl
// Bisection + inverse iteration for symmetric tridiagonal eigenvalue problem.
// Much faster than general Jacobi for tridiagonal structure.
// Paper 023: N=50–200 Aubry-André/Anderson Hamiltonians.
@group(0) @binding(0) var<storage, read> diag: array<f32>;         // [N] diagonal
@group(0) @binding(1) var<storage, read> offdiag: array<f32>;      // [N-1] off-diagonal
@group(0) @binding(2) var<storage, read_write> eigenvalues: array<f32>; // [N]
@group(0) @binding(3) var<storage, read_write> eigenvectors: array<f32>; // [N × N]
@group(0) @binding(4) var<uniform> params: TridiagParams;          // N, tol, max_iter

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Each thread finds one eigenvalue via bisection on Sturm count
    // Then inverse iteration for eigenvector
}
```

**Serves:** Papers 022–023. Specialized for tridiagonal structure — O(N²) vs O(N³) Jacobi.

#### `xoshiro256ss.wgsl` — GPU PRNG (All stochastic algorithms)

```wgsl
// Xoshiro256** for GPU-side parallel random number generation.
// Each thread uses jump() to get an independent stream from shared seed.
struct Rng { s0: u32, s1: u32, s2: u32, s3: u32, }

fn rotl(x: u32, k: u32) -> u32 { return (x << k) | (x >> (32u - k)); }

fn next(rng: ptr<function, Rng>) -> u32 {
    let result = rotl((*rng).s1 * 5u, 7u) * 9u;
    let t = (*rng).s1 << 9u;
    (*rng).s2 ^= (*rng).s0;
    (*rng).s3 ^= (*rng).s1;
    (*rng).s1 ^= (*rng).s2;
    (*rng).s0 ^= (*rng).s3;
    (*rng).s2 ^= t;
    (*rng).s3 = rotl((*rng).s3, 11u);
    return result;
}

fn uniform_f32(rng: ptr<function, Rng>) -> f32 {
    return f32(next(rng) >> 8u) / 16777216.0;
}
```

**Serves:** All stochastic Phase 0++ algorithms (EA mutation, MC sampling, initialization).
Foundation for GPU-side population evaluation without CPU round-trips.

---

## Part 3: Recommended Absorption Order

| Priority | Item | Effort | Impact | Retires |
|----------|------|--------|--------|---------|
| 1 | **`from_buffer` → `pub`** (S-07) | 1 char | High | S-08, S-09 |
| 2 | **MHA projection shader hang** (S-03b — z-dispatch fixed but native MHA hangs on RTX 4070/Vulkan) | Investigation | Correctness | `evolved::mha` |
| 3 | **`leaky_relu`/`elu` Params** (S-05, S-06) | 4 lines | Correctness | — |
| 4 | **Softmax logical size** (S-04) | 10 lines | Correctness | Re-upload workaround |
| 5 | **`new_cpu_relaxed()`** (S-10) | 20 lines | Unblocks CPU | `gpu::create_relaxed()` |
| 6 | **Add matmul shaders** (S-02) | Copy + wire | Performance | Evolved matmul shaders |
| 7 | **Extend `TensorSession`** (S-01, S-11) | Medium | 46–78× perf | All `evolved/` modules |
| 8 | **Absorb attention shaders** | Copy | Completeness | Inline WGSL in fused_pipeline |

---

## Part 4: neuralSpring Evolved Code Inventory

### Fossilized (Feb 20, 2026 — `metalForge/fossils/`)

| Module | Lines | Workaround For | Status |
|--------|-------|----------------|--------|
| `evolved/layer_norm.rs` | 268 | S-08 | Fossilized, rewired to native |
| `evolved/log_softmax.rs` | 259 | S-09 | Fossilized, rewired to native |
| `evolved/fused_pipeline.rs` | 680 | S-01 | Fossilized |
| `evolved/fused_mlp.rs` | 356 | S-01/S-11 | Fossilized |
| `evolved/fused_transformer.rs` | 725 | S-01/S-11 | Fossilized |
| `evolved/matmul_cpu_tiled.wgsl` | 270 | S-02 | Fossilized |
| `evolved/matmul_gpu_evolved.wgsl` | 306 | S-02 | Fossilized |
| **Total** | **~2864** | | |

### Still Active

| Module | Lines | Issue | Status |
|--------|-------|-------|--------|
| `evolved/mha.rs` | 182 | S-03b (native MHA hangs) | Active |
| `evolved/hmm_forward_gpu.rs` | 270 | No BarraCUDA equiv | Active |

---

## Part 5: ToadStool Capabilities for neuralSpring GPU Promotion

New ToadStool infrastructure directly usable for Phase 0++ GPU promotion:

| ToadStool Capability | neuralSpring Use Case | Integration Point |
|---------------------|----------------------|-------------------|
| `StatefulPipeline` | EA generation loops (Papers 011–015), ODE integration (020–021), HMM chains (016–018) | `run_iterations()` for N GPU steps, scalar readback for convergence |
| `ReduceScalarPipeline` | Fitness aggregation (sum/max over population), log-likelihood (HMM) | `scalar_buffer()` for zero-copy pipeline chaining |
| `KernelRouter` | 4-tier matmul selection, future per-op device routing | Wire `MatmulConfig` logic into `KernelRouter::route()` |
| `GpuDriverProfile` | Per-driver shader specialization (NAK vs proprietary) | Already captured in `MatmulConfig` |
| NAK-optimized `batched_eigh_nak_optimized_f64.wgsl` | Anderson localization (Paper 023) eigensolver | Drop-in replacement for Jacobi iteration |

---

## Part 6: Cross-Paper Primitive Convergence

From 23 papers, the primitive usage confirms the isomorphism thesis:

| Primitive | Papers Using It | BarraCUDA Shader |
|-----------|----------------|-----------------|
| GEMM | 18/23 | `matmul.wgsl` + evolved variants |
| `reduce_sum` | 20/23 | `sum_reduce_f64.wgsl` |
| `elementwise` | 23/23 | Various activation shaders |
| Softmax | 8/23 | `softmax_simple.wgsl` |
| ODE integration | 3/23 | Proposed `rk4_batch.wgsl` |
| Eigendecomposition | 3/23 | `batched_eigh_*.wgsl` + proposed `tridiag_eigh.wgsl` |
| HMM chain | 3/23 | Proposed `hmm_forward_log.wgsl` |

---

## Reproduction

```bash
# Full Python baselines (190/190 PASS, ~10 min)
bash scripts/run_all_baselines.sh

# Full Rust validation (532/532 PASS, ~15 sec)
make validate

# BarraCUDA CPU ports only (123/123 PASS)
make validate-barracuda-cpu

# All quality gates
make check
```

---

## Phase 2 Addendum: BarraCUDA CPU Port Findings (Feb 20, 2026)

All 13 Phase 0++ modules ported to BarraCUDA CPU math. 123/123 checks PASS.

### Primitives Validated

| Primitive | Modules | Precision |
|-----------|---------|-----------|
| `numerical::rk45_solve` | regulatory, signal, game | Machine ε — direct RK4 replacement |
| `linalg::solve_f64` | hmm, swarm | Machine ε — stationary distributions |
| `linalg::eigh_f64` | spectral, anderson | ~1e-3 (n=8) — see S-12 below |
| `special::chi_squared_sf` | introgression | 1e-10 — LRT p-values |
| `stats::variance` | all 13 modules | Machine ε — cross-validation |
| `stats::pearson_correlation` | modes | Machine ε — trend analysis |

### S-12: eigh_f64 Accuracy Gap (NEW)

| Matrix Size | Reconstruction Error | LAPACK Reference |
|-------------|---------------------|------------------|
| n=4 | ~1e-6 | 1e-14 |
| n=8 | ~1e-3 | 1e-14 |
| n=16 | ~0.01 | 1e-14 |
| n=32 | ~0.7 | 1e-14 |

Jacobi iteration converges slowly for larger matrices. ToadStool's NAK
eigensolver may resolve this on GPU. Suggested CPU fix: Householder →
tridiagonal → bisection.

### New Absorption Candidates from Phase 2

| Primitive | Use Case | Papers |
|-----------|----------|--------|
| `linalg::batch_matmul` | HMM forward/backward chain | 016–018 |
| `ea::batch_fitness` | Population-parallel fitness evaluation | 011–015 |
| `numerical::batch_rk45` | Multi-system ODE integration | 020–021 |

---

*neuralSpring: 23 papers, 722/722 PASS, 12 shortcomings documented (11 original
+ S-12 eigh accuracy gap), 7 GPU shader designs proposed, 5 delivered,
13 BarraCUDA CPU ports complete. Phase 2 done — ready for GPU evolution.
ToadStool `StatefulPipeline` + `ReduceScalarPipeline` + `KernelRouter`
provide the infrastructure for Phase 3 GPU promotion.*
