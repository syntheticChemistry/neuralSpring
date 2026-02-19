# BarraCUDA Shader Evolution for ML Inference

**Date**: February 19, 2026
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

| Module | Lines | Purpose | Upstream Recommendation |
|--------|-------|---------|------------------------|
| `fused_pipeline.rs` | 408 | ShaderCache + 4-tier router + dispatch helpers | Integrate into `TensorSession` |
| `fused_mlp.rs` | 253 | Fused MLP (9 passes, 1 submit) | Template for session-based ML ops |
| `fused_transformer.rs` | 409 | Fused Transformer (18 passes, 1 submit) | Template for session-based ML ops |
| `matmul_cpu_tiled.wgsl` | 263 | Double-buffered CPU matmul | Add to `barracuda/src/shaders/math/` |
| `matmul_gpu_evolved.wgsl` | 302 | Double-buffered GPU matmul | Add to `barracuda/src/shaders/math/` |
| `layer_norm.rs` | 199 | GPU-resident layer norm | Make `from_buffer` public |
| `log_softmax.rs` | 192 | GPU-resident log-softmax | Make `from_buffer` public |
| `mha.rs` | 116 | MHA projection workaround | Fix z-dim dispatch bug |
| `gpu.rs` | 225 | Device wrapper with CPU/GPU creation | Relax `science_limits()` for llvmpipe |

Total locally evolved: **~2,367 lines of Rust + 565 lines of WGSL**.
All retireable once ToadStool absorbs the upstream changes.
