# RTX 4070 — Hardware Characterization

**Device**: NVIDIA GeForce RTX 4070 (AD104-250-A1)
**Architecture**: Ada Lovelace (TSMC N4)
**Memory**: 12 GB GDDR6X, 192-bit bus, 504 GB/s bandwidth
**Host**: i9-12900K, 32 GB DDR5-4800, PCIe 4.0 x16
**Driver**: Vulkan 1.3 (proprietary 550.x), wgpu v22
**Date**: February 20, 2026

---

## Compute Capabilities

| Spec | Value | Notes |
|------|-------|-------|
| CUDA cores | 5888 | 46 SMs × 128 |
| Tensor cores | 184 (4th gen) | INT8/FP16/BF16/TF32 |
| RT cores | 46 (3rd gen) | Not used by WGSL |
| Base clock | 1920 MHz | |
| Boost clock | 2475 MHz | |
| FP32 TFLOPS | 29.15 | Peak single-precision |
| FP16 TFLOPS | 58.3 | With tensor cores |
| INT8 TOPS | 466.4 | With tensor cores |
| TDP | 200W | |
| SHADER_F64 | **Supported** | Confirmed via wgpu adapter features |
| SHADER_F16 | **Supported** | Confirmed via wgpu adapter features |
| TIMESTAMP_QUERY | **Supported** | Confirmed via wgpu adapter features |

## wgpu Backend Details

| Property | Value |
|----------|-------|
| Backend | Vulkan |
| Device type | `DiscreteGpu` |
| Max buffer size | ~2 GB (limited by Vulkan) |
| Max storage buffers per shader stage | 8 |
| Max workgroup size | 1024 (x: 1024, y: 1024, z: 64) |
| Max compute invocations per workgroup | 1024 |
| Max dispatch x | 65535 |

## Measured Performance (neuralSpring workloads)

| Workload | Time | Throughput |
|----------|------|-----------|
| MatMul 1024×1024 (evolved shader) | 0.8 ms | 2,684 GFLOP/s |
| MLP forward (fused, 9 passes) | 92 µs | — |
| Transformer forward (fused, 18 passes) | 174 µs | — |
| FFT 1024-point (Cooley-Tukey) | ~100 µs | — |
| HMM forward 100 obs × 3 states | ~50 µs | — |
| Batch fitness 512 × 16 | ~30 µs | — |
| RK4 parallel 4 systems × 100 steps | ~40 µs | — |

## CPU Fallback (llvmpipe)

| Property | Value |
|----------|-------|
| Backend | Vulkan (software) |
| Device type | `Cpu` |
| SHADER_F64 | Not supported |
| Max buffer size | ~128 MB |
| Typical performance | 100–1000× slower than discrete GPU |
| Use case | CI, correctness validation |

---

*Hardware characterization — following hotSpring metalForge pattern.*
