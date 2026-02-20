# metalForge — ML Dispatch & Hardware Characterization

**Parent**: ecoPrimals/neuralSpring
**License**: AGPL-3.0-or-later

---

## Philosophy

Every ML inference result in ecoPrimals originates in shader math (WGSL).
ToadStool dispatches that math to whatever silicon is available. metalForge
exists to **characterize how hardware affects ML dispatch** — where per-op
overhead dominates, where memory bandwidth saturates, and where cache tiling
unlocks real throughput.

Following the hotSpring pattern: we don't just benchmark. We probe
dispatch latency, measure workgroup occupancy, profile shared memory
pressure, and find the crossover points where CPU software rasterization
beats GPU for small tensors.

---

## Hardware Inventory

| Substrate | Device | Driver | Key Characteristic |
|-----------|--------|--------|-------------------|
| **GPU** | NVIDIA RTX 4070 (12 GB) | Vulkan (proprietary) | 5888 CUDA cores, 186 GB/s bandwidth |
| **CPU** | llvmpipe (software) | Mesa | Single-threaded WGSL, useful for CI and correctness |

---

## Key Findings

### 1. Per-Op Dispatch Overhead (S-01)

The single biggest performance bottleneck in BarraCUDA ML inference is
**per-op command submission**. Each `Tensor` operation creates its own
`CommandEncoder`, dispatches one compute pass, and submits individually.

| Pipeline | Per-Op | Fused | Speedup |
|----------|--------|-------|---------|
| MLP (4→64→64→10) | 4.5 ms | **97 µs** | **46×** |
| Transformer (d=32, h=4) | 12.8 ms | **164 µs** | **78×** |

**Root cause**: `queue.submit()` on Vulkan has ~500 µs fixed overhead.
An MLP with 9 ops pays 9 × 500 µs = 4.5 ms in submission alone.

**Local fix**: `evolved::fused_pipeline` — single encoder, single submit.

### 2. Matmul Cache Tiling (S-02)

BarraCUDA's `matmul.wgsl` reads K elements from global memory per output
element. No shared memory, no tiling, no vectorization.

| Shader | Backend | Technique | Result |
|--------|---------|-----------|--------|
| `matmul.wgsl` (stock) | CPU | Global reads | 3× slower than NumPy |
| `matmul_cpu_tiled.wgsl` | CPU | 32×32, vec4, 8×4 µkernel | **1.1× faster than NumPy** |
| `matmul_gpu_evolved.wgsl` | GPU | 16×16, 2×2 µkernel, double-buf | **104× faster than NumPy** |

**Key insight**: CPU tiling matters even for software rasterizers because
it dramatically reduces global memory traffic via shared memory reuse.

### 3. GPU Crossover Points

| Scale (FLOPs) | Python | CPU | GPU | Winner |
|---------------|--------|-----|-----|--------|
| 3.1M (MLP large) | 3.0 ms | 2.7 ms | 178 µs | GPU |
| 103M (TF medium) | 59 ms | 15.1 ms | 566 µs | GPU |
| 6.6B (TF xlarge) | 232 ms | 1.42 s | 17.8 ms | GPU |

**Crossover**: GPU always wins when using fused dispatch. CPU beats Python
only at ≥3M FLOPs. Below ~1M FLOPs, Python's NumPy (calling OpenBLAS)
wins due to zero dispatch overhead.

### 4. BarraCUDA CPU Math Precision

| Primitive | Test Case | Precision | Notes |
|-----------|-----------|-----------|-------|
| `rk45_solve` | 4D ODE, t=20 | **machine ε** | Matches hand-rolled RK4 exactly |
| `solve_f64` | Dense linear system | **machine ε** | LU-based, excellent |
| `chi_squared_sf` | LRT p-value | **1e-10** | Matches scipy reference |
| `eigh_f64` | n=8 symmetric | **~1e-3** | Jacobi eigensolver accuracy gap |
| `eigh_f64` | n=16 symmetric | **~0.1** | Degrades with dimension |
| `stats::variance` | Population stats | **machine ε** | Two-pass algorithm, stable |

**Critical gap**: `eigh_f64` uses Jacobi iteration, which has O(n³) cost
and ~1e-3 relative error for n≥8. LAPACK achieves 1e-14. This affects
spectral analysis (Paper 022) and Anderson localization (Paper 023).
ToadStool's NAK eigensolver (GPU) may resolve this.

---

## Evolution Targets

| Target | Current | Proposed | Impact |
|--------|---------|----------|--------|
| Fused dispatch | Local `evolved::fused_pipeline` | ToadStool `StatefulPipeline` | Retire 600 LOC |
| Tiled matmul | Local WGSL shaders | ToadStool `KernelRouter` | Retire 565 LOC |
| Eigensolver | Jacobi (1e-3 at n=8) | Lanczos / divide-and-conquer | 1e-14 precision |
| Batched fitness | CPU loop | GPU parallel eval | EA papers 011–015 |
| HMM chain | Sequential matmul | Batched GEMM | Papers 016–018 |
| Multi-system ODE | Sequential rk45 | GPU parallel integration | Papers 020–021 |

---

## Directory Structure

```
metalForge/
├── README.md              ← this file (dispatch characterization)
└── gpu/
    └── nvidia/
        └── DISPATCH.md    ← RTX 4070 dispatch latency measurements
```

---

## Relationship to hotSpring metalForge

| hotSpring metalForge | neuralSpring metalForge |
|---------------------|------------------------|
| GPU f64 native throughput | ML dispatch overhead profiling |
| NPU (Akida) characterization | Matmul cache tiling analysis |
| Cache line behavior | Workgroup occupancy vs tensor size |
| Register space probing | Shared memory pressure in tiled kernels |

Both feed findings to ToadStool via `wateringHole/handoffs/`.

---

*Hardware characterization for ML dispatch — following the hotSpring metalForge pattern.*
