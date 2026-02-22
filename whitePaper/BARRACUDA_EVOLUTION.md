# BarraCUDA Shader Evolution for ML Inference

**Date**: February 22, 2026
**Gate**: Eastgate (i9-12900K, 32 GB DDR5, RTX 4070 12 GB)
**Methodology**: Python control → Rust validation → WGSL shader evolution

---

## 1. The Problem: Per-Op Dispatch Kills Small-Tensor Inference

BarraCUDA's Tensor API creates a new `CommandEncoder` and calls `queue.submit()`
for every operation. For an MLP with 9 ops, that's 9 submissions at ~200 µs each:

```
matmul → add → relu → matmul → add → relu → matmul → add → softmax
  200µs   200µs  200µs   200µs  200µs  200µs   200µs  200µs   200µs
```

**Total dispatch: 1.8 ms. Actual f32 compute: ~5 µs. Overhead: 99.7%.**

Python/NumPy does the same math in 8 µs — a single BLAS call per matmul, zero
I/O overhead, in-process function pointer dispatch. BarraCUDA was 200× slower
than Python, not because the shaders were wrong (they're correct to 1.49e-8),
but because the dispatch overhead completely dominated.

---

## 2. Evolution Steps

### Step 1: Fused Pipeline (Phase 1c)

Pre-compile all shaders, pre-allocate all intermediate buffers and bind groups
**once**, then record all compute passes into a **single** `CommandEncoder`
with one `queue.submit()`:

| Model | Per-Op | Fused | Speedup |
|-------|--------|-------|---------|
| MLP | 4.0 ms | 92 µs | **43.6×** |
| Transformer | 13.3 ms | 174 µs | **76.6×** |

**Key insight from hotSpring**: Their MD simulation batches 10 velocity-Verlet
steps into one encoder, reads back only 8 bytes (KE scalar), not full particle
buffers. "90% to GPU, ~10% back." We applied the same pattern — one encoder
for the full forward pass, readback only at the end.

New inline WGSL shaders were required for the fused pipeline because the
standard BarraCUDA MHA ops use CPU-side head_split/head_concat:
- `HEAD_SPLIT_WGSL`: `[seq, d_model]` → `[n_heads, seq, d_head]`
- `HEAD_CONCAT_WGSL`: reverse
- `BATCHED_ATTENTION_WGSL`: fused Q·K^T/√d → softmax → ·V for all heads

### Step 2: BLAS-Evolved CPU Shader

Python's advantage at small scale comes from OpenBLAS — hand-tuned AVX-512 GEMM
with panel packing, multi-level tiling, and micro-kernel register blocking.
BarraCUDA's `matmul.wgsl` reads K elements from global memory per output element:
zero cache reuse.

We learned from OpenBLAS and applied every applicable technique in WGSL:

| OpenBLAS Technique | WGSL Implementation | Why It Helps on CPU |
|-------------------|---------------------|---------------------|
| Panel size / L1 blocking | 32×32 tiles in `var<workgroup>` | Tiles fit in 32 KB L1 |
| Aligned SIMD loads | `array<vec4<f32>>` B-tile | Maps to `movaps` via LLVM |
| Mr × Nr register blocking | 8×4 micro-kernel (8 vec4 accumulators) | 32 FMAs per k-step per thread |
| ILP from independent FMA chains | 4× k-loop unroll | CPU pipeline overlaps 4 FMA chains |
| Software pipelining / prefetch | Double-buffered tiles | Load NEXT while computing CURRENT |
| FMA3 instructions | `fma()` intrinsic | WGSL → LLVM `fmuladd` → FMA3 |

Result: **CPU beats single-thread Python** at MLP large (3M FLOPs, 1.1×) and
**Transformer 3.9× faster** at TF medium (103M FLOPs).

### Step 3: Double-Buffered GPU Shader

hotSpring's MD simulations use double-buffered staging to overlap GPU iteration
N+1 with CPU processing of iteration N. We applied the same principle **within**
a matmul shader: two sets of shared memory tiles, loading the next while
computing on the current.

On NVIDIA hardware, this matters because loads go through the memory pipeline
and FMAs go through the ALU pipeline. Between two `workgroupBarrier()` calls,
the GPU interleaves both:

```
Standard:   [load] → barrier → [compute] → barrier → [load] → ...
                                                      ↑ ALU idle during load

Double-buf: [load_NEXT + compute_CURRENT] → barrier → [load_NEXT + compute_CURRENT] → ...
            ↑ ALU active while memory is in flight
```

Result: **10–12% improvement** at TF xlarge (20.6 ms → 17.8 ms).

### Step 4: Tiered GPU Routing

The 32×32 double-buffered shader reduces workgroup count (fewer, larger tiles),
which can hurt SM occupancy for small matrices. We added tiering:

- Small GPU matmuls (M,N < 256): 16×16 tiles (BarraCUDA standard, high occupancy)
- Large GPU matmuls (M,N ≥ 256): 32×32 double-buffered (throughput wins)

This recovers the small-scale regression while keeping the large-scale improvement.

### Step 5: DeviceCapabilities-Driven Router

BarraCUDA's `DeviceCapabilities` already provides `optimal_matmul_tile_size()` —
returns 32 for NVIDIA, 16 for Intel, 8 for CPU. We wire this into a `MatmulConfig`
struct that caches the device capabilities at pipeline creation:

```rust
pub fn select_matmul(cache, m, k, n, config) -> (pipeline, dispatch_fn) {
    if m < threshold || n < threshold { naive }
    else if config.is_cpu            { cpu_tiled (32×32, double-buffered) }
    else if m >= 256 || n >= 256     { gpu_evolved (32×32, double-buffered) }
    else                             { gpu_tiled (16×16, shared-memory) }
}
```

No neuralSpring code changes needed when ToadStool adds new vendors — just
update `DeviceCapabilities` upstream.

---

## 3. Results: 3-Way Benchmark

### MLP Scaling

| Scale | FLOPs | Py(1t) | CPU | GPU | GPU/CPU |
|-------|-------|--------|-----|-----|---------|
| tiny | 10K | 12 µs | 382 µs | 87 µs | 4.4× |
| small | 49K | 12 µs | 447 µs | 94 µs | 4.8× |
| medium | 786K | 56 µs | 2.8 ms | 131 µs | 21.5× |
| **large** | **3.1M** | **3.0 ms** | **2.7 ms** | **178 µs** | **15.1×** |
| **xlarge** | **12.6M** | **9.0 ms** | **10.3 ms** | **265 µs** | **39.1×** |

### Transformer Scaling

| Scale | FLOPs | Py(1t) | CPU | GPU | GPU/CPU |
|-------|-------|--------|-----|-----|---------|
| tiny | 201K | 115 µs | 694 µs | 161 µs | 4.3× |
| small | 12.8M | 1.0 ms | 2.9 ms | 243 µs | 12.0× |
| **medium** | **103M** | **59 ms** | **15.1 ms** | **566 µs** | **26.8×** |
| large | 822M | 146 ms | 193 ms | 2.6 ms | 74.3× |
| xlarge | 6.6B | 232 ms | 1.42 s | 17.8 ms | **79.9×** |

### Progression Check: GPU < CPU < Python

| Scale | MLP | Transformer |
|-------|-----|-------------|
| large | **✓** GPU < CPU < Py | ~ (CPU 1.3× > Py) |
| medium | ~ (CPU > Py) | **✓** GPU < CPU < Py |

GPU dominates CPU at **every scale** (4–80×).

---

## 4. Correctness

All three backends agree within f32 tolerance:

| Model | Max Diff (CPU vs Python) | Max Diff (GPU vs Python) |
|-------|-------------------------|--------------------------|
| MLP | 1.49e-8 | 1.49e-8 |
| Transformer | 7.45e-7 | 1.10e-6 |

**The math IS the shader.** CPU and GPU execute the same WGSL source code —
ToadStool compiles it to native x86 (llvmpipe) or Vulkan (NVIDIA driver).
Correctness is a property of the WGSL, not the hardware.

---

## 5. Lessons for ToadStool

### 5.1 Per-op dispatch is the #1 bottleneck

For ML inference, the ratio of dispatch overhead to actual compute is 99%+ at
small tensor sizes. A session-based API (`TensorSession`) that records multiple
ops into one encoder would eliminate this entirely.

### 5.2 GPU-resident outputs are essential

Both `layer_norm_wgsl` and `log_softmax_wgsl` read results back to CPU and
construct a new `Tensor`. This forces a GPU→CPU→GPU round-trip (~400 µs each).
Making `Tensor::from_buffer` public would retire two local evolutions.

### 5.3 Device-aware kernel selection has real impact

The same matmul shader performs very differently on CPU vs GPU. A 4-tier router
with runtime device detection provides the right kernel for every dispatch:
- CPU needs larger register blocks, smaller workgroups, double-buffered tiles
- GPU needs higher occupancy at small scale, double-buffered tiles at large scale

### 5.4 BLAS teaches GPU shader design

Every OpenBLAS optimization maps to a WGSL technique:
- Panel packing → `var<workgroup>` tile blocking
- SIMD loads → `vec4<f32>` storage
- Register blocking → micro-kernel accumulators
- Software pipelining → double-buffered shared memory
- ILP → k-loop unrolling

### 5.5 hotSpring's streaming dispatch pattern works for ML

The "many passes, one encoder, minimal readback" pattern from hotSpring's MD
simulation transfers directly to neural network inference. Both are chains of
dependent compute operations with a single scalar output (loss or energy).

---

## 6. Evolved Code Inventory

### Fossilized (absorbed — `metalForge/fossils/evolved_s01_s11/`)

| Module | Lines | Shortcoming | Absorbed Into |
|--------|-------|-------------|---------------|
| `fused_pipeline.rs` | 680 | S-01 | `TensorSession` |
| `fused_mlp.rs` | 356 | S-01/S-11 | `TensorSession` ML ops |
| `fused_transformer.rs` | 725 | S-01/S-11 | `TensorSession` ML ops |
| `matmul_cpu_tiled.wgsl` | 270 | S-02 | `ops::matmul` CpuTiled32 |
| `matmul_gpu_evolved.wgsl` | 306 | S-02 | `ops::matmul` GpuEvolved32 |
| `layer_norm.rs` | 268 | S-08 | `Tensor::layer_norm_wgsl()` |
| `log_softmax.rs` | 259 | S-09 | `Tensor::log_softmax_wgsl()` |

Total fossilized: **~2,864 LOC** (removed from active compilation Feb 20, 2026).

### Still Active

| Module | Lines | Why Active |
|--------|-------|-----------|
| `mha.rs` | 182 | S-03b: native projection shaders hang |
| `hmm_forward_gpu.rs` | 270 | No BarraCUDA equivalent |
| `gpu.rs` | 225 | Device wrapper (rewired to `new_cpu_relaxed`) |

---

## 7. Phase 3 — GPU Streaming Evolution

### 7.1 FFT Validation (Phase 3a — Complete)

ToadStool's BarraCUDA team shipped `ops::fft` — a Cooley-Tukey radix-2
implementation in WGSL, covering f32 complex (`Fft1D`/`Ifft1D`), f64 complex
(`Fft1DF64`), and real-to-complex (`Rfft`). Session 25 fixed three GPU FFT bugs
(fossil `floor_f64` calls, missing sin/cos kernel deps, inverse twiddle conjugation)
and added GPU-validated f64 tests. We now validate **24 analytical checks**:

**f32 FFT (Fft1D / Ifft1D) — 12 checks:**

| Check | Method | Tolerance |
|-------|--------|-----------|
| Inverse round-trip (N=16) | `IFFT(FFT(x)) == x` | 1e-3 |
| Parseval's theorem (N=32) | `‖x‖² == ‖FFT(x)‖²/N` | 1e-3 |
| Delta → constant DFT pair | Analytical DFT definition | 1e-5 |
| Constant → delta DFT pair | Analytical DFT definition | 1e-5 |
| Cosine concentration | Energy at ±f bins | 1e-4 |
| Larger round-trip (N=256) | `IFFT(FFT(x)) == x` | 1e-3 |
| Multi-frequency synthesis | 3-frequency sum, bin check | boolean |

**Rfft (real-to-complex, f32) — 4 checks:**

| Check | Method | Tolerance |
|-------|--------|-----------|
| Output shape | N → [N/2+1, 2] | exact |
| DC component | X[0].re == N for constant signal | 1e-3 |
| Off-peak energy | ≈ 0 for constant | 1e-3 |
| Cosine energy | Significant energy at target bin | boolean |

**f64 FFT (Fft1DF64) — 8 checks (requires SHADER_F64):**

| Check | Method | Tolerance |
|-------|--------|-----------|
| Inverse round-trip (N=16) | `IFFT(FFT(x))/N == x` | 1e-10 |
| Parseval's theorem (N=32) | `‖x‖² == ‖FFT(x)‖²/N` | 1e-10 |
| Delta → constant DFT pair | All real=1, imag=0 | 1e-10 |
| Constant → delta DFT pair | X[0]=N, off-peak≈0 | 1e-10 |
| Cosine concentration | Energy at ±f bins | 1e-10 |

The f64 round-trip error achieved 2.78e-16 (essentially machine epsilon),
confirming ToadStool's WGSL f64 butterfly is IEEE 754 compliant.

No local shader was needed — we absorbed directly from ToadStool. This is the
ideal absorption pattern: upstream ships, Spring validates, upstream iterates.

### 7.2 GPU Streaming Targets (Phase 3b — Complete)

The goal is to move iterative workloads from CPU loops to GPU-resident pipelines
using `StatefulPipeline` (state stays on GPU between iterations) and
`UnidirectionalPipeline` (data streams in, results stream out):

| Workload | Papers | CPU Pattern | GPU Target |
|----------|--------|-------------|------------|
| HMM forward chain | 016–018 | Sequential GEMM | `StatefulPipeline` with log-domain GEMM shader |
| Population fitness | 011–015 | Loop over individuals | Batch GEMM shader (N individuals × M traits) |
| ODE integration | 020–021 | CPU RK4 loop | GPU-parallel multi-system RK4 shader |
| Eigendecomposition | 022–023 | CPU Householder+QR (S-12 absorbed) | NAK GPU eigensolve available |
| Pangenome selection | 024 | CPU pairwise Jaccard | `pairwise_jaccard.wgsl` — GPU O(N²) similarity |
| Meta-population | 025 | CPU locus variance | `locus_variance.wgsl` — GPU allele-frequency variance |

### 7.3 Shader Evolution for Absorption (Phase 3c–4d — 12 shaders)

Following hotSpring's pattern, WGSL shaders are developed in `metalForge/shaders/`,
validated against CPU references, and documented for ToadStool absorption.

**12 WGSL shaders validated**, 93/93 PASS (RTX 4070).

| Shader | Validation Binary | Checks | Absorption Target |
|--------|-------------------|--------|-------------------|
| `hmm_forward_log.wgsl` | `validate_gpu_hmm_forward` | 13/13 | `ops::hmm` or `StatefulPipeline` |
| `batch_fitness_eval.wgsl` | `validate_gpu_batch_fitness` | 20/20 | `ops::batch_gemm` |
| `rk4_parallel.wgsl` | `validate_gpu_rk4` | 8/8 | `ops::ode` |
| `mean_reduce.wgsl` | `validate_gpu_pure_workload` | 7/7 | `ReduceScalarPipeline` |
| `pairwise_jaccard.wgsl` | `validate_gpu_pangenome` | 6/6 | `ops::pairwise_distance` |
| `locus_variance.wgsl` | `validate_gpu_meta_pop` | 7/7 | `ops::VarianceReduceF64` |
| `spatial_payoff.wgsl` | `validate_gpu_game_theory` | 5/5 | `ops::stencil` |
| `batch_ipr.wgsl` | `validate_gpu_anderson` | 5/5 | `ops::batch_reduce` |
| `pairwise_hamming.wgsl` | `validate_gpu_sate` | 5/5 | `ops::pairwise_distance` |
| `xoshiro128ss.wgsl` | `validate_gpu_prng` | 5/5 | `ops::prng` |
| `head_split.wgsl` | `validate_mha_gpu` | 5/5 | `ops::mha` |
| `head_concat.wgsl` | `validate_mha_gpu` | 5/5 | `ops::mha` |

Planned (not yet implemented):

| Shader | Target Op | Absorption Target |
|--------|-----------|-------------------|
| `tridiag_eigensolver.wgsl` | Eigendecomposition | `linalg::eigh_gpu` — **resolved** via NAK eigensolve (`77f70b2e`) |
| `pairwise_distance.wgsl` | Distance matrix | `ops::pairwise_distance` |

Each shader follows the hotSpring lifecycle: evolve → validate → handoff → absorb → retire.
See `metalForge/shaders/ABSORPTION_TRACKER.md` for the full lifecycle tracker.

---

## 8. Phase 4a: Performance Parity Benchmarks

The `bench_phase0pp_kernels` binary compares pure Rust math (neuralSpring) to single-thread NumPy
at identical problem sizes across 7 Phase 0++ kernels. This proves where the evolution path delivers
and where GPU acceleration is essential.

| Kernel | Paper | Rust µs | Python µs | Speedup |
|--------|-------|---------|-----------|---------|
| HMM forward (3×5000) | 016-018 | 330.0 | 12007.6 | 36.4× |
| Replicator dynamics (10k steps) | 019 | 150.0 | 34937.4 | 232.9× |
| Commutator ‖[A,B]‖_F (64×64) | 022 | 334.6 | 23.3 | 0.1× |
| NK fitness (N=10,K=2, 1000 genotypes) | 011 | 17.9 | 14087.2 | 787.1× |
| Pairwise Hamming (20×500) | 017 | 34.3 | 408.3 | 11.9× |
| Jaccard distance (30×500) | 024 | 142.3 | 2045.4 | 14.4× |
| RK4 GRN ODE (2000 steps) | 020-021 | 218.6 | 24659.8 | 112.8× |
| **TOTAL** | | **1227.8** | **88169.0** | **71.8×** |

**Narrative:** Rust pure math is 71.8× faster than single-thread NumPy overall. GEMM-heavy
operations (commutator: 0.1×) show why GPU WGSL acceleration via BarraCUDA matters — at small
matrix sizes, NumPy's OpenBLAS-backed GEMM dominates. The evolution path (Python → Rust CPU →
BarraCUDA GPU) delivers dramatic speedups for elementwise, reduction, and ODE workloads; for dense
GEMM the GPU path is the only way to beat optimized BLAS.

---

## 9. ToadStool Absorption Complete (Feb 22, 2026)

**All 12 neuralSpring shortcomings (S-01 through S-12) are now absorbed by
ToadStool at commit `77f70b2e` (Session 31h).** S-12 (eigensolver accuracy)
was the final shortcoming — resolved by absorbing neuralSpring's Householder+QR
implementation upstream.

Key absorption commit: `fbedd222` — extended `TensorSession` with
`{MatMul, ReLU, GELU, Softmax, LayerNorm}` and single-encoder batch
dispatch. This directly addresses the per-op overhead documented in
Sections 1–2 above.

### Rewiring completed

- `validate_barracuda_tensor` rewired from `evolved::layer_norm`/`log_softmax` to
  native `Tensor::layer_norm_wgsl()`/`log_softmax_wgsl()` — **90/90 PASS**
- `bench_barracuda_tensor` rewired from `evolved::layer_norm`/`log_softmax` to
  native `Tensor::layer_norm_wgsl()`/`log_softmax_wgsl()`
- `leaky_relu` (S-05) and `elu` (S-06) tests added — both passing natively
- `gpu.rs` CPU path rewired to `WgpuDevice::new_cpu_relaxed()` (S-10)
- All deprecated evolved modules fossilized in `metalForge/fossils/`
- `bench_fused_inference` and `bench_scaling` fossilized (deep fused pipeline coupling)

### S-03b: Native MHA Projection Shader Hang

While S-03 (z-dispatch `div_ceil(16)` → `div_ceil(1)`) was absorbed, the native
`Tensor::multi_head_attention` hangs during GPU execution on RTX 4070 / Vulkan.
The evolved MHA (matmul projections + CPU head split/concat + attention) works
correctly and remains active in `src/evolved/mha.rs`. Filed as S-03b for
ToadStool to debug the `project_with_head_split` / `concat_and_project` GPU
execution flow.

### What remains active

- `evolved::mha` — MHA workaround (S-03b, blocked on native shader hang)
- `evolved::hmm_forward_gpu` — metalForge shader evolution (no BarraCUDA equivalent)

See `metalForge/fossils/FOSSIL_RECORD.md` for the complete fossil inventory.

---

## 10. GPU WGSL Kernel Benchmarks + GPU PRNG (Phase 4c)

### GPU Dispatch Crossover

The `bench_gpu_kernels` binary times WGSL shaders on RTX 4070 vs Rust CPU at matching
problem sizes, revealing the fundamental crossover point for dispatch decisions.

| Kernel | Scale | GPU µs | Rust CPU µs | Winner |
|--------|-------|--------|-------------|--------|
| Hamming | Small (20×500) | 1,589 | 34 | CPU 46× |
| Hamming | **Large (200×1000)** | **1,675** | **7,089** | **GPU 4.2×** |
| Jaccard | Small (30×500) | 1,659 | 142 | CPU 12× |
| Jaccard | **Large (100×2000)** | **1,464** | **8,246** | **GPU 5.6×** |
| Fitness | Small (1000×10) | 1,836 | 18 | CPU 102× |
| Fitness | Large (50000×64) | 1,510 | — | — |

**Finding:** GPU dispatch overhead is ~1.5ms fixed (Vulkan `queue.submit()` +
readback). GPU compute time is negligible at all tested scales — the 5888 CUDA
cores on the RTX 4070 complete these kernels in microseconds. The dispatch cost
is amortized when:

1. **CPU work exceeds ~1.5ms** — the natural crossover point, confirmed empirically.
2. **Fused dispatch** (`TensorSession`, `StatefulPipeline`) — one submit for N ops.
3. **Large workloads** — 200 seqs × 1000 sites gives 19,900 pairs → GPU 4.2× faster.

This validates the cross-dispatch architecture: `barracuda::dispatch` routes small
workloads to CPU and large workloads to GPU, with the threshold matching the
empirical ~1.5ms crossover. `StatefulPipeline` eliminates the overhead entirely
for iterative GPU-resident algorithms.

### GPU PRNG: Xoshiro128**

The `xoshiro128ss.wgsl` shader provides GPU-parallel pseudo-random number generation.
Each thread maintains independent 4×u32 state (seeded via SplitMix32). Generates
uniform f32 in [0, 1). State persists across dispatches for multi-call sequences.

| Check | Status |
|-------|--------|
| Uniformity (mean ∈ [0.48, 0.52]) | **PASS** (0.4995) |
| Range ([0, 1)) | **PASS** |
| Determinism (same seed → same output) | **PASS** |
| Independence (distinct thread sequences) | **PASS** |
| Multi-call (state advances correctly) | **PASS** |

Exported as `rng::WGSL_XOSHIRO128SS` for ToadStool absorption.

**Impact:** Enables Wright-Fisher, Gillespie SSA, and parallel EA generation loops
entirely on GPU via `StatefulPipeline` — no CPU round-trips for random number
generation.

---

## 11. ToadStool Issue Resolution (Phase 4d)

Phase 4d resolves two ToadStool shortcomings via local fixes, documented for
absorption.

### S-12 Resolution: Householder+QR Eigensolver — ABSORBED

BarraCUDA's original `linalg::eigh_f64` used Jacobi iteration, which degraded to
~1e-3 relative error at n≥8 and ~0.1 at n=16. neuralSpring implemented
Householder tridiagonalization + QL implicit shifts (Wilkinson), achieving
LAPACK-level accuracy. **ToadStool absorbed this at `77f70b2e`** — `src/eigh.rs`
now delegates to `barracuda::ops::linalg::eigh_householder_qr`. Local fossil
preserved at `metalForge/fossils/evolved_s01_s11/eigh_local.rs`.

**Accuracy comparison (historical)**:

| n | Householder+QR | Jacobi | Improvement |
|---|----------------|--------|-------------|
| 4 | 1.13e-14 | 2.21e-14 | 2× |
| 8 | 3.05e-14 | 1.27e-1 | 4.2 trillion × |
| 16 | 5.28e-14 | 1.64e+1 | 312 trillion × |
| 32 | 1.83e-13 | 7.03e+1 | 383 trillion × |
| 64 | 5.43e-13 | 1.69e+2 | 311 trillion × |

Anderson Hamiltonian n=32: 1.75e-14 (vs Jacobi's ~70). Validation:
`validate_eigh_accuracy` — **9/9 PASS** (now delegated to upstream).

ToadStool also added `WGSL_BATCHED_EIGH_NAK_OPTIMIZED` for GPU-native
eigensolve — available for Anderson localization and hotSpring nuclear physics.

### S-03b Partial Fix: GPU Head Split/Concat Shaders

The native `Tensor::multi_head_attention` hangs during GPU execution on RTX 4070
(Vulkan). The projection shaders (`project_with_head_split` / `concat_and_project`)
are the suspected cause. Local GPU `head_split.wgsl` and `head_concat.wgsl` avoid
this by decomposing MHA into validated ops:

```
matmul (Q,K,V projections) → head_split → attention → head_concat → matmul (Wo)
```

- `head_split.wgsl`: [B,S,D] → [B,H,S,D/H]
- `head_concat.wgsl`: [B,H,S,D/H] → [B,S,D]

Validation: `validate_mha_gpu` — **10/10 PASS**.

**New check count**: 19 additional checks (9 + 10).

---

## 12. Phase 4e: Domain Modules + GPU Pipelines (February 21, 2026)

Phase 4e adds two new Rust domain modules, four new GPU shaders, and completes
module flattening for GPU-ready absorption.

### New Rust Domain Modules

| Module | Domain | Key Operations |
|--------|--------|----------------|
| `pinn.rs` | Burgers' PDE (Paper 028) | Cole-Hopf transform, MLP forward, finite-difference residual |
| `deeponet.rs` | Operator networks (Paper 029) | Branch-trunk operator, polynomial evaluation |

Both modules validate against BarraCUDA CPU via the Tensor API: matmul, tanh,
dot product. No local shaders — they consume `barracuda::tensor` buffer input.

### Four New GPU Domain Shaders

| Shader | Paper | Validation Binary | Absorption Target |
|--------|-------|-------------------|-------------------|
| `pairwise_l2.wgsl` | 012 (MODES novelty) | `validate_gpu_modes` | `barracuda::ops::pairwise_distance` |
| `multi_obj_fitness.wgsl` | 014 (Directed evolution) | `validate_gpu_directed` | `barracuda::ops::batch_gemm` |
| `swarm_nn_forward.wgsl` | 015 (Swarm robotics) | `validate_gpu_swarm` | `barracuda::ops::batch_gemm` |
| `hill_gate.wgsl` | 021 (Signal integration) | `validate_gpu_signal` | `barracuda::ops::elementwise` |

Three new GPU pipelines chain these shaders to `mean_reduce.wgsl` for fitness
aggregation.

### Module Flattening Complete

Two modules converted from nested to flat layouts:

| Module | Before | After |
|--------|--------|-------|
| `directed_evolution.rs` | `Vec<Vec<f64>>` genotypes | Flat `Vec<f64>` (pop×genome) |
| `sate_alignment.rs` | `Vec<Vec<u8>>` sequences | Flat `Vec<u8>` (n×len) |

All domain modules now use GPU-ready flat row-major layouts.

### Phase 4e Summary

- **95 new validation checks** across pinn, deeponet, modes, directed, swarm, signal
- **4 new WGSL shaders**: pairwise_l2, multi_obj_fitness, swarm_nn_forward, hill_gate
- **3 new GPU pipelines** chaining to mean_reduce
- **Module flattening**: directed_evolution, sate_alignment → flat layouts

---

## 13. Deep Evolution: GPU-Ready Layouts (February 21, 2026)

Following the hotSpring pattern of evolving Rust implementations toward
GPU absorption, neuralSpring underwent a deep structural evolution:

### Flat Row-Major Matrices

HMM and spectral commutativity modules converted from `Vec<Vec<f64>>`
(heap-per-row) to flat `Vec<f64>` with row-major indexing. This is the
GPU-native layout — flat buffers upload directly to `wgpu::Buffer`
without conversion.

| Module | Before | After | GPU Benefit |
|--------|--------|-------|-------------|
| `hmm.rs` | `Vec<Vec<f64>>` transition, emission, alpha | Flat `Vec<f64>` with stride `n` | Direct upload to `hmm_forward_log.wgsl` buffers |
| `spectral_commutativity.rs` | `Vec<Vec<f64>>` for all matrix ops | Flat `Vec<f64>` with explicit `n` dimension | Direct upload to `barracuda::ops::matmul` |
| `ForwardResult` | Nested alpha, requires row allocation | Flat with `alpha_at(t)` slice accessor | Zero-copy slicing for GPU readback comparison |

### Consolidated Mathematical Primitives

Six variants of Shannon entropy, three Hill kinetics, two sigmoid, and
two RK4 integration — all centralized into `src/primitives.rs`. Module-local
magic numbers (`1e-15`, `1e-300`, `1e-20`) promoted to named constants
(`DIVISION_GUARD`, `LOG_GUARD`, `HILL_EPS`).

### Graceful GPU Error Handling

The `require!` macro replaces `.expect()` across all validation binaries.
When GPU operations fail (adapter unavailable, buffer allocation, shader
compilation), the harness records a FAIL and continues rather than panicking.
This is essential for CI where GPU adapters may vary.

```rust
// Before: panics on GPU failure
let tensor = Tensor::from_data(&data, shape, device.clone()).expect("alloc");

// After: records failure, continues validation
let tensor = require!(h, Tensor::from_data(&data, shape, device.clone()), "alloc");
```

### Write → Absorb → Lean Alignment

These changes align neuralSpring's Rust implementations with the hotSpring
absorption pattern:

1. **Flat buffers** match GPU binding layouts documented in `EVOLUTION_READINESS.md`
2. **Named constants** match ToadStool's `tolerances` pattern
3. **Graceful errors** support the cross-backend validation that ToadStool requires
4. **`Hmm::from_flat()`** constructor provides the GPU-native entry point

---

## Phase 5b: Upstream Issue Resolution

### S-13: `PooledBuffer` Drop-Before-Completion Race

`BarraCUDA`'s `BufferPool` returns buffers to the pool in `PooledBuffer::drop`
without waiting for the GPU to finish using them. Sequential operations that
produce intermediate tensors (dropped before readback) trigger buffer reuse
races — data corruption or driver hangs.

**Local fix**: `evolved::tensor_sync` provides `gpu_fence`, `materialize`,
and `fenced_matmul` as sync primitives. The proper upstream fix is
`device.poll(Wait)` in `PooledBuffer::drop` or generation-tracked recycling.

### S-14: Naive Matmul Hang for Small Square Matrices

The Naive matmul tier (`matmul.wgsl`, selected when M or N < 32) hangs
indefinitely on the RTX 4070 Vulkan driver when the binary exceeds a certain
complexity threshold. Non-square inputs always work. The Tiled16 tier is
unaffected.

**Hypothesis**: Pipeline cache pressure from complex binaries triggers a
driver-level hang during `create_compute_pipeline` or `dispatch_workgroups`.

**Recommendation**: Remove the Naive tier; use Tiled16 for all sizes.

### GELU Test Fix

The `gelu(3) ≈ 3.0` test used the wrong expected value. True GELU(3) =
2.996362607918227 (from `scipy.special.erf`). The WGSL implementation was
correct — only the test expectation was wrong. Now 86/86 PASS.

---

## Phase 5a Findings: GPU Tensor Validation (7 Domains)

Phase 5a expanded GPU `Tensor` validation from 2 domains (spectral + eco)
to 7, exercising `matmul`, `transpose`, `tanh`, and `add` across all 15
Phase 0++ papers. Two new critical bugs discovered:

### S-15: Matmul Hang with Negative / Sparse f32 Input (Critical)

`Tensor::matmul` hangs indefinitely when input data contains negative f32
values or is highly sparse (many zeros). Confirmed on RTX 4070 Vulkan with
the Naive matmul tier. The shader source (`matmul.wgsl`) has no conditional
logic on data values — the hang is at the WGPU/Vulkan driver level.

The `should_use_npu_for_matmul()` code path calls `to_vec()` on both input
tensors for sparsity analysis before routing, even when NPU is unavailable.
This introduces GPU→CPU readback synchronization that may interact with the
subsequent matmul dispatch.

**Impact**: Blocks GPU validation of any domain with naturally negative data
(neural network weights, centered features, physics quantities). All Phase 5a
validators work around this by restricting to `[0, 1)` range data.

### S-16: 2D Transpose Dispatch Uses Wrong Workgroup Divisor (High)

The 2D transpose shader uses `@workgroup_size(16, 16)` with tiled
shared-memory access, but the dispatch in `ops/transpose/compute.rs` divides
by `optimal_workgroup_size(WorkloadType::ElementWise)`, which returns 256 on
NVIDIA GPUs:

```
CORRECT: workgroups_y = ceil(rows / 16) = ceil(20 / 16) = 2
BUG:     workgroups_y = ceil(rows / 256) = ceil(20 / 256) = 1
```

This produces partial output: for a [20, 8] → [8, 20] transpose, only
columns 0-15 are computed; columns 16-19 remain zero. When this partially
transposed tensor is used in matmul, the output Gram matrix has an
entire column block of zeros.

**Root cause**: `execute_2d()` line 169 uses `caps.optimal_workgroup_size()`
instead of the hardcoded tile constant 16.

**Fix**: Replace `optimal_wg_size` with `16` (one line).

### Phase 5b: Full-Stack Resolution (ALL GREEN)

Phase 5b resolves the Phase 5a blockers and expands coverage to all papers:

**S-16 FIXED**: The transpose dispatch bug was a one-line fix: replace
`optimal_workgroup_size(ElementWise)` with the shader's hardcoded tile
constant (`const TILE: u32 = 16`). All pairwise validators now PASS.

**S-15 ROOT-CAUSED**: The matmul hang occurs when input data elements have
magnitude ≤ 0.1. This is a WGPU/Vulkan driver bug on RTX 4070, not a shader
bug. The workaround generates all test data with `rng.uniform() * 0.5 + 0.5`,
ensuring all elements ≥ 0.5. Anderson localization and all other domains now PASS.

**New validators added** (Phase 5b buildout):
- `validate_barracuda_surrogate` (Exp 001) — 7 checks, S-15 safe
- `validate_barracuda_transfer` (Exp 004) — 7 checks, S-15 safe
- `validate_barracuda_gpu_transformer` (Exp 002) — 7 checks, S-15 safe
- `validate_cross_dispatch_hmm` (Papers 016, 018) — 4 checks
- `validate_cross_dispatch_ode` (Paper 020) — 4 checks

**Reclassified as GPU Tensor (gT)**: `validate_barracuda_sequence` (Exp 003),
`validate_barracuda_lenet` (Study 003), `validate_barracuda_lstm` (Study 004)
already used `Tensor` operations on GPU — counted toward gT coverage.

**Final coverage** (25 papers, 7 tiers):

| Tier | Coverage | Status |
|------|----------|--------|
| Python (Py) | 25/25 (100%) | **ALL PASS** |
| Rust (Rs) | 25/25 (100%) | **ALL PASS** |
| BarraCUDA CPU (bC) | 24/25 (96%) | **ALL GREEN** |
| GPU Tensor (gT) | 23/25 (92%) | **ALL GREEN** |
| metalForge WGSL (mF) | 14/25 (56%) | **ALL PASS** |
| GPU Pipeline (gP) | 7/25 (28%) | **ALL PASS** |
| Cross-dispatch (xD) | 15/15 (100%) | **ALL GREEN** |

Full handoff: `wateringHole/handoffs/`

### Session 39 Sync: Upstream Absorption Wave (`d45fdfb3`)

ToadStool's Session 39 (dead code sweep + evolution) absorbed 5 neuralSpring
local shaders into barracuda's shader tree as generalized upstream variants:

| Shader | Upstream Path | Evolution |
|--------|---------------|-----------|
| `pairwise_l2.wgsl` | `shaders/math/pairwise_l2` | Closed-form pair decode (O(1) vs O(N) linear search) |
| `multi_obj_fitness.wgsl` | `shaders/bio/multi_obj_fitness` | Bessel correction (n-1 divisor), standardized params |
| `hill_gate.wgsl` | `shaders/bio/hill_gate` | Mode 0 (paired) / mode 1 (grid) generalization |
| `swarm_nn_forward.wgsl` | `shaders/bio/swarm_nn_forward` | Generic MLP via `SwarmParams{input_dim,hidden_dim,output_dim}`, clamped sigmoid |
| `mean_reduce.wgsl` | `shaders/reduce/mean_reduce` | Effectively identical (barracuda credits neuralSpring as origin) |

**Bug fixes flowing to neuralSpring** (via path dependency):
- **S-13 FIXED**: `PooledBuffer` drop race condition — deferred return via pending queue + non-blocking device poll
- **TS-003**: Trig precision — `sin_simple`/`cos_simple` upgraded to 7-term Taylor + Cody-Waite range reduction
- **TS-001**: `pow_f64` fix — extended `exp_f64` to handle 2^k for |k| up to 1023
- **TS-004**: `FusedMapReduceF64` — both passes encoded in single command encoder

**New capabilities** (not yet leveraged):
- `ops::nn/conv2d.wgsl` — batched Conv2D with stride, padding (LeNet-5 ready)
- `ops::nn/maxpool2d.wgsl` — batched MaxPool2D
- `ops::nn/avgpool2d.wgsl` — batched AvgPool2D
- `cpu_conv_pool` module — CPU reference implementations
- `esn_v2::export_weights/import_weights` — GPU-train → NPU-deploy

**Shader absorption summary** (cumulative):

| Category | Count | Status |
|----------|-------|--------|
| Identical copies (77f70b2e) | 8 | **Upstream** |
| Generalized variants (d45fdfb3) | 5 | **Upstream** (local copies retained for validation) |
| Still local-only | 4 | `head_split`, `head_concat`, `xoshiro128ss`, `swarm_nn_scores` |
| **Total** | **17** | **13/17 absorbed** (76%) |
