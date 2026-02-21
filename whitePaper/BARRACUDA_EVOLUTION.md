# BarraCUDA Shader Evolution for ML Inference

**Date**: February 21, 2026
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
| Eigendecomposition | 022–023 | CPU Jacobi | Tridiagonal + bisection shader |
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
| `tridiag_eigensolver.wgsl` | Eigendecomposition | `linalg::eigh_gpu` |
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

## 9. ToadStool Absorption Complete (Feb 20, 2026)

**All 11 neuralSpring shortcomings (S-01 through S-11) are now absorbed by
ToadStool at commit `dc540afd` (Session 25).**

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

### S-12 Resolution: Householder+QR Eigensolver

BarraCUDA's `linalg::eigh_f64` uses Jacobi iteration, which degrades to ~1e-3
relative error at n≥8 and ~0.1 at n=16. The local `src/eigh.rs` implements
Householder tridiagonalization + QL implicit shifts (Wilkinson), achieving
LAPACK-level accuracy at all matrix sizes.

**Accuracy comparison**:

| n | Householder+QR | Jacobi | Improvement |
|---|----------------|--------|-------------|
| 4 | 1.13e-14 | 2.21e-14 | 2× |
| 8 | 3.05e-14 | 1.27e-1 | 4.2 trillion × |
| 16 | 5.28e-14 | 1.64e+1 | 312 trillion × |
| 32 | 1.83e-13 | 7.03e+1 | 383 trillion × |
| 64 | 5.43e-13 | 1.69e+2 | 311 trillion × |

Anderson Hamiltonian n=32: 1.75e-14 (vs Jacobi's ~70). Validation:
`validate_eigh_accuracy` — **9/9 PASS**.

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
