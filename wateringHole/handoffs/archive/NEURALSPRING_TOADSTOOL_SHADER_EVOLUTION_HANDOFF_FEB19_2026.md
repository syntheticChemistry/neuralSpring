# [SUPERSEDED] neuralSpring → ToadStool: Shader Evolution & 3-Way Benchmark Handoff

> **Superseded by:** `NEURALSPRING_TOADSTOOL_HANDOFF_FEB20_2026.md` (consolidated)
> This document is a fossil record. See the consolidated handoff for current status.

**Date:** 2026-02-19
**From:** neuralSpring (ML / isomorphic learning Spring)
**To:** ToadStool / BarraCUDA core team
**Builds-on:** NEURALSPRING_TOADSTOOL_HANDOFF_FEB19_2026.md (ML validation + fused pipeline)
**License:** AGPL-3.0-or-later

---

## Executive Summary

neuralSpring has evolved BarraCUDA's matmul shaders from naive global-memory
reads to BLAS-grade double-buffered tiled kernels with device-aware routing.
A 3-way benchmark (Python/NumPy vs BarraCUDA CPU vs BarraCUDA GPU) validates
the hotSpring progression: **GPU < CPU < Python** at crossover scales.

**Key deliverables:**

- **2 evolved WGSL shaders**: `matmul_cpu_tiled.wgsl` (CPU-optimized) and
  `matmul_gpu_evolved.wgsl` (GPU-optimized), both using double-buffered tiles
- **4-tier `MatmulConfig` router** driven by `DeviceCapabilities` at runtime
- **3-way benchmark binary** (`bench_scaling`) that runs Python, CPU, and GPU
  in a single invocation with automatic progression checking
- **GPU 104× faster** than Python at Transformer medium (103M FLOPs)
- **CPU 3.9× faster** than single-thread Python at the same scale
- **GPU dominates CPU 4–80×** at every measured scale

All outputs are numerically correct: max diff 1.49e-8 (MLP), 1.10e-6 (Transformer)
against Python baselines. Same WGSL source, different compilation targets.

---

## 1. What Changed Since Last Handoff

| Item | Before (Phase 1c) | After (Phase 1d) |
|------|--------------------|-------------------|
| GPU matmul | `matmul_tiled.wgsl` (16×16) | 4-tier: naive / tiled / **gpu-evolved** |
| CPU matmul | `matmul_cpu_tiled.wgsl` (single-buffered) | **Double-buffered**, same 8×4 µkernel |
| Benchmark | Single-backend, manual env var | **3-way in one binary** |
| Shader router | `select_matmul(cache, m, k, n, config)` | + tiered GPU (16×16 vs 32×32) |
| `Gpu` struct | `new()` only | + **`new_cpu()`**, **`new_gpu()`** |

---

## 2. Deliverables

| # | File | Lines | Description |
|---|------|-------|-------------|
| 1 | `src/evolved/matmul_cpu_tiled.wgsl` | 263 | CPU: 32×32 double-buffered, vec4 B-tile, 8×4 µkernel, 4× k-unroll |
| 2 | `src/evolved/matmul_gpu_evolved.wgsl` | 302 | GPU: 32×32 double-buffered, vec4 B-tile, 2×2 µkernel, 4× k-unroll |
| 3 | `src/evolved/fused_pipeline.rs` | 408 | ShaderCache + MatmulConfig + 4-tier select_matmul |
| 4 | `src/evolved/fused_mlp.rs` | 253 | Fused MLP with MatmulConfig routing |
| 5 | `src/evolved/fused_transformer.rs` | 409 | Fused Transformer with MatmulConfig routing |
| 6 | `src/bin/bench_scaling.rs` | 288 | 3-way benchmark (Python vs CPU vs GPU) |
| 7 | `src/gpu.rs` | 225 | + `new_cpu()`, `new_gpu()` methods |
| 8 | `control/ml_inference/bench_scaling.py` | 158 | Python baseline with single-thread variant |
| 9 | `specs/BENCHMARK_ANALYSIS.md` | 146 | Full 3-way analysis and recommendations |
| 10 | `whitePaper/BARRACUDA_EVOLUTION.md` | 180 | Shader evolution narrative |

---

## 3. What ToadStool Should Absorb

### 3.1 Add evolved matmul shaders to BarraCUDA (High Priority)

**Files**: `matmul_cpu_tiled.wgsl`, `matmul_gpu_evolved.wgsl`
**Target**: `barracuda/src/shaders/math/`

Both shaders use the same binding layout as existing `matmul.wgsl`:
- `@binding(0)` A: `array<f32>` (read)
- `@binding(1)` B: `array<f32>` (read)
- `@binding(2)` C: `array<f32>` (read_write)
- `@binding(3)` params: `MatmulParams { M, K, N }` (uniform)

They are drop-in additions — no changes to existing shaders required.

**CPU shader key techniques:**
- Workgroup `(8, 4)` = 32 threads (matches llvmpipe's single-core execution)
- 8×4 micro-kernel: 8 rows × 4 cols per thread, all in `vec4<f32>` accumulators
- Double-buffered `var<workgroup>` tiles: load NEXT while computing CURRENT
- 4× k-loop unroll for ILP
- vec4 B-tile storage for aligned 16-byte loads

**GPU shader key techniques:**
- Workgroup `(16, 16)` = 256 threads
- 2×2 micro-kernel: 32×32 output tile per workgroup
- Double-buffered tiles: overlap memory pipeline with ALU pipeline
- vec4 B-tile for coalesced 128-bit transactions
- 4× k-loop unroll for warp-level ILP

### 3.2 Wire `DeviceCapabilities` into `KernelRouter` for matmul (High Priority)

**Current state**: `KernelRouter` routes between NPU and WGSL but doesn't select
between WGSL matmul variants.

**Recommended change**: Add a `select_matmul` function (similar to neuralSpring's)
that uses `DeviceCapabilities::optimal_matmul_tile_size()` and `device_type` to
choose between naive, cpu-tiled, gpu-tiled, and gpu-evolved variants.

Reference implementation: `neuralSpring/src/evolved/fused_pipeline.rs` lines 347–377.

**Tiering logic:**
```
M,N < threshold       → matmul.wgsl (naive)
CPU, large M,N        → matmul_cpu_tiled.wgsl
GPU, small M,N        → matmul_tiled.wgsl
GPU, M or N ≥ 256     → matmul_gpu_evolved.wgsl
```

### 3.3 Extend `TensorSession` to support ML ops (Medium Priority)

neuralSpring's fused pipeline manually manages encoders, bind groups, and
compute passes. This pattern should become a first-class BarraCUDA API.

**Current `TensorSession`**: Supports `Add`, `Mul`, `Fma`, `Scale`.
**Needed**: `MatMul`, `ReLU`, `GELU`, `LayerNorm`, `Softmax`, `Attention`.

The key pattern: pre-compile pipelines, pre-allocate buffers, record N passes
into one `CommandEncoder`, submit once. neuralSpring's `ShaderCache` + helper
functions provide the template.

### 3.4 `DeviceCapabilities::optimal_matmul_tile_size()` for CPU (Low Priority)

Currently returns 8 for CPU. neuralSpring's CPU shader uses 32×32 tiles
(much larger). Consider a separate `optimal_cpu_matmul_tile_size()` that
returns 32 for modern x86 with AVX-512 and 16 for ARM/NEON.

The current return value of 8 is used as the `min_tiled` threshold (below
which we fall back to naive matmul). neuralSpring uses `.max(16)` to ensure
the threshold is at least 16, but a more accurate value from BarraCUDA would
eliminate this workaround.

---

## 4. Benchmark Results

### 4.1 3-Way MLP Scaling

| Scale | FLOPs | Py(1t) | CPU | GPU | CPU/Py | GPU/Py | GPU/CPU |
|-------|-------|--------|-----|-----|--------|--------|---------|
| tiny | 10K | 12 µs | 382 µs | 87 µs | 32.9× slower | 7.5× slower | 4.4× faster |
| small | 49K | 12 µs | 447 µs | 94 µs | 35.8× slower | 7.5× slower | 4.8× faster |
| medium | 786K | 56 µs | 2.8 ms | 131 µs | 50.1× slower | 2.3× slower | 21.5× faster |
| **large** | **3.1M** | **3.0 ms** | **2.7 ms** | **178 µs** | **1.1× faster** | **16.8× faster** | **15.1×** |
| **xlarge** | **12.6M** | **9.0 ms** | **10.3 ms** | **265 µs** | ~borderline | **34.0× faster** | **39.1×** |

### 4.2 3-Way Transformer Scaling

| Scale | FLOPs | Py(1t) | CPU | GPU | CPU/Py | GPU/Py | GPU/CPU |
|-------|-------|--------|-----|-----|--------|--------|---------|
| tiny | 201K | 115 µs | 694 µs | 161 µs | 6.0× slower | 1.4× slower | 4.3× faster |
| small | 12.8M | 1.0 ms | 2.9 ms | 243 µs | 2.8× slower | 4.3× faster | 12.0× faster |
| **medium** | **103M** | **59 ms** | **15.1 ms** | **566 µs** | **3.9× faster** | **104× faster** | **26.8×** |
| large | 822M | 146 ms | 193 ms | 2.6 ms | 1.3× slower | 56.2× faster | 74.3× faster |
| xlarge | 6.6B | 232 ms | 1.42 s | 17.8 ms | 6.1× slower | 13.1× faster | **79.9×** |

### 4.3 Double-Buffering Impact (GPU)

| Scale | Before DB | After DB | Improvement |
|-------|-----------|----------|-------------|
| TF large | 3.0 ms | 2.6 ms | 13% |
| TF xlarge | 20.6 ms | 17.8 ms | 14% |

### 4.4 Progression Check

| Scale | MLP | Transformer |
|-------|-----|-------------|
| medium | ~ GPU < CPU (CPU > Py) | **✓ GPU < CPU < Py** |
| large | **✓ GPU < CPU < Py** | ~ GPU < CPU (CPU 1.3× > Py) |

---

## 5. Lessons Learned

### 5.1 BLAS is the teacher

Every optimization in Python's NumPy (via OpenBLAS) maps to a WGSL technique:

| Python/OpenBLAS | WGSL Implementation |
|-----------------|---------------------|
| Panel packing (L1 blocking) | `var<workgroup>` tile arrays |
| Aligned AVX loads | `array<vec4<f32>>` storage → 16-byte coalesced |
| Mr × Nr register blocking | Micro-kernel (8×4 CPU, 2×2 GPU) |
| Software pipelining | Double-buffered tiles |
| FMA3 intrinsics | `fma()` WGSL builtin |
| ILP from independent chains | k-loop unrolling (4×) |

**Recommendation**: When BarraCUDA needs to optimize a shader, study how the
equivalent operation is implemented in OpenBLAS/BLIS first. The same principles
apply to WGSL shaders compiled to both CPU (via LLVM) and GPU (via Vulkan).

### 5.2 CPU and GPU need different shaders

The same WGSL *language* runs on both, but optimal parameters differ:

| Parameter | CPU (llvmpipe) | GPU (RTX 4070) |
|-----------|---------------|----------------|
| Workgroup size | 32 threads (8×4) | 256 threads (16×16) |
| Tile size | 32×32 | 32×32 (large), 16×16 (small) |
| Micro-kernel | 8×4 (larger blocks, more reuse) | 2×2 (more threads, less per-thread) |
| Double-buffering benefit | Modest (7–11%) | Significant (10–14%) |
| Bottleneck | Dispatch overhead + memory bandwidth | Dispatch overhead (tiny), throughput (large) |

**Recommendation**: `DeviceCapabilities` should route to different shader variants
based on `device_type`, not just different tile sizes.

### 5.3 hotSpring's streaming dispatch pattern transfers to ML

hotSpring's MD simulation: "Many compute passes in one encoder, minimal readback."
neuralSpring's fused inference: exactly the same pattern. Both are chains of
dependent compute operations producing a single output.

**Recommendation**: The BarraCUDA `TensorSession` API should support this pattern
natively — a `session.run()` that submits one encoder for N recorded ops.

### 5.4 Double-buffered tiles should be the default for large matmuls

The improvement is consistent across both CPU (LLVM scheduling) and GPU
(memory/ALU pipeline overlap). For matrices where K > TILE, double-buffering
is strictly better — the extra shared memory (~5 KB) is negligible.

**Recommendation**: Make double-buffered tiling the default for the evolved
matmul shaders in BarraCUDA, falling back to single-buffered only when shared
memory pressure is a concern (e.g., very large tile sizes on mobile GPUs).

### 5.5 Dispatch overhead is structural for tiny tensors

At MLP tiny (10K FLOPs), both CPU and GPU are 8–33× slower than Python.
This is the wgpu dispatch floor: ~80 µs (GPU) / ~300 µs (CPU) minimum per
submission, vs Python's ~10 µs for a single in-process BLAS call.

**No shader optimization can fix this.** The solution is either:
1. Batch many small inferences into one submission (streaming)
2. Use BarraCUDA's CPU-native BLAS path for tiny tensors (bypass wgpu entirely)
3. Accept the overhead for single-inference latency and optimize throughput

### 5.6 The math IS the shader

The most important observation: correctness is a property of the WGSL source,
not the hardware. Max diff between CPU and GPU outputs is 1.49e-8 (MLP) and
~1e-6 (Transformer). The same WGSL → SPIR-V compiles to:
- x86 native (llvmpipe/LLVM) for CPU
- Vulkan bytecode (NVIDIA driver) for GPU

**This is BarraCUDA's core value proposition**: write the math once, get
hardware-portable correctness. The evolved shaders in this handoff maintain
this property — they're still standard WGSL, compilable on any vendor.

---

## 6. Remaining Bottlenecks

| Bottleneck | Impact | Suggested Fix |
|------------|--------|---------------|
| CPU dispatch overhead (wgpu+llvmpipe) | ~300 µs min per submit | TensorSession / batch API |
| CPU memory bandwidth (large TF) | 1.3–6× slower than Py at 822M+ | Panel packing, multi-WG parallelism |
| GPU dispatch overhead (tiny tensors) | ~80 µs min per submit | Streaming / CPU BLAS fallback |
| `layer_norm` / `log_softmax` readback | ~400 µs × 2 per TF block | Make `from_buffer` public |
| MHA z-dim dispatch bug | Incorrect workgroup count | Fix `div_ceil(16)` → `div_ceil(1)` |

---

## 7. Files Changed (This Handoff)

```
 M  src/evolved/fused_pipeline.rs     (+ matmul_gpu_evolved, 4-tier router, MatmulConfig)
 M  src/evolved/fused_mlp.rs          (MatmulConfig integration)
 M  src/evolved/fused_transformer.rs  (MatmulConfig integration)
 M  src/evolved/matmul_cpu_tiled.wgsl (double-buffered tiles)
 A  src/evolved/matmul_gpu_evolved.wgsl (new GPU-evolved shader)
 M  src/bin/bench_scaling.rs          (3-way benchmark)
 M  src/gpu.rs                        (+ new_cpu, new_gpu methods)
 M  control/ml_inference/bench_scaling.py (+ single-thread variant)
 M  specs/BENCHMARK_ANALYSIS.md       (3-way results + 4-tier router)
 M  specs/TOADSTOOL_HANDOFF.md        (#11 updated: 4-tier + GPU evolved)
 A  whitePaper/BARRACUDA_EVOLUTION.md  (shader evolution narrative)
 M  whitePaper/README.md              (updated with Phase 1d)
 M  whitePaper/STUDY.md               (updated with 3-way results)
 M  CONTROL_EXPERIMENT_STATUS.md      (updated with Phase 1d)
 M  README.md                         (updated with 3-way table)
 M  wateringHole/handoffs/NEURALSPRING_TOADSTOOL_HANDOFF_FEB19_2026.md (updated)
```

---

## 8. Reproduction

```bash
# Full 3-way benchmark (Python + CPU + GPU, ~6 min)
cargo run --release --bin bench_scaling

# Validate correctness on both backends
NEURALSPRING_BACKEND=cpu cargo run --release --bin validate_barracuda_ml_inference
NEURALSPRING_BACKEND=gpu cargo run --release --bin validate_barracuda_ml_inference

# All quality gates
make check
```
