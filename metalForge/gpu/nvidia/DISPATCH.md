# RTX 4070 — ML Dispatch Characterization

**Device**: NVIDIA GeForce RTX 4070 (AD104), 12 GB GDDR6X
**Driver**: Vulkan (proprietary), via wgpu v22
**Date**: February 20, 2026

---

## Dispatch Overhead

### Per-Op Submission (queue.submit per op)

| Operation | Latency | Notes |
|-----------|---------|-------|
| `queue.submit()` | ~500 µs | Fixed cost per submission |
| Shader compilation | ~2 ms | First use only; cached thereafter |
| Buffer allocation | ~50 µs | Pooled after first allocation |
| Buffer readback | ~200 µs | GPU→CPU synchronization barrier |

### Fused Submission (single encoder, single submit)

| Pipeline | Ops | Per-Op Total | Fused Total | Speedup |
|----------|-----|-------------|-------------|---------|
| MLP (4→64→64→10) | 9 | 4.5 ms | 97 µs | 46× |
| Transformer (d=32, h=4, seq=8) | 18+ | 12.8 ms | 164 µs | 78× |

**Conclusion**: Dispatch overhead dominates for small-to-medium tensors.
Fused dispatch eliminates this entirely.

---

## Matmul Throughput by Scale

### GPU (RTX 4070, Vulkan)

| M×K×N | Naive | Tiled (16×16) | Evolved (32×32) | GFLOP/s |
|-------|-------|--------------|----------------|---------|
| 64×64×64 | 23 µs | 18 µs | 15 µs | 35 |
| 256×256×256 | 89 µs | 42 µs | 31 µs | 1,078 |
| 1024×1024×1024 | 8.2 ms | 1.1 ms | 0.8 ms | 2,684 |

### CPU (llvmpipe)

| M×K×N | Naive | CPU-Tiled (32×32) | NumPy (OpenBLAS) |
|-------|-------|------------------|------------------|
| 64×64×64 | 1.2 ms | 0.4 ms | 0.1 ms |
| 256×256×256 | 45 ms | 9.8 ms | 3.2 ms |
| 1024×1024×1024 | 12.8 s | 1.42 s | 0.23 s |

**Key finding**: CPU-tiled WGSL beats naive by 3–9× but still trails
OpenBLAS by 3–6× at scale. This is expected — OpenBLAS uses AVX-512 and
L1/L2 cache-tuned micro-kernels that software rasterization cannot match.

---

## Workgroup Occupancy

### Matmul Dispatch Grid

| Workgroup Size | Best For | Why |
|---------------|----------|-----|
| 8×8 (naive) | M,N ≤ 16 | Single workgroup, zero overhead |
| 16×16 (tiled) | M,N ≤ 512 | High occupancy, small shared memory |
| 32×32 (evolved) | M,N > 512 | Maximum shared memory reuse, lower occupancy but higher throughput |

### Shared Memory Pressure

| Shader | Shared Memory | Tiles Loaded | Reuse Factor |
|--------|--------------|-------------|-------------|
| naive | 0 B | — | 1× (no reuse) |
| tiled (16×16) | 2 KB | 1 tile pair | K/16× reuse |
| evolved (32×32) | 8 KB (2 buffers) | 2 tile pairs (double-buffered) | 2×K/32× reuse |

---

*RTX 4070 dispatch characterization — neuralSpring metalForge*
