# Mixed-Hardware Dispatch — Validation Results

**Date**: February 22, 2026
**Session**: 43
**ToadStool HEAD**: `6ee71f07`

---

## Summary

All mixed-hardware dispatch infrastructure validated:

| Validator | Checks | Status |
|-----------|--------|--------|
| `validate_mixed_dispatch` | 16/16 | PASS |
| `validate_toadstool_dispatch` | 16/16 | PASS |
| `validate_cpu_gpu_parity` | 17/17 | PASS |

---

## Transfer Cost Model

| Path | Estimated (1 MB) | Range | Status |
|------|-------------------|-------|--------|
| GPU→CPU (PCIe 4.0 x16) | 35.3 µs | [30, 50] µs | Validated |
| GPU→NPU P2P (PCIe 4.0 x4) | 134.7 µs | [100, 200] µs | Validated |
| GPU→NPU staged (via CPU) | 139.7 µs | > P2P | Validated |
| GPU→CPU 0 bytes (latency only) | 2.0 µs | < 10 µs | Validated |

## Substrate Routing Validation

| Heuristic | Small → | Large → | Status |
|-----------|---------|---------|--------|
| `pairwise_substrate` | CPU (20×500) | GPU (200×1000) | Validated |
| `batch_fitness_substrate` | CPU (100×100) | GPU (1000×100) | Validated |
| `ode_substrate` | CPU (10×100) | GPU (100×200) | Validated |
| `hmm_substrate` | CPU (3×100) | GPU (10×1000) | Validated |
| `spatial_substrate` | CPU (100) | GPU (10000) | Validated |
| `batch_ipr_substrate` | CPU (100×100) | GPU (1000×100) | Validated |
| `logsumexp_substrate` | CPU (100×100) | GPU (500×100) | Validated |
| `stochastic_substrate` | CPU (10×10×100) | GPU (100×100×20) | Validated |

## Mixed Substrate Selection

| Scenario | Decision | Status |
|----------|----------|--------|
| Small compute (100 µs) | CPU-only | Validated |
| Large compute (100 ms) | GPU-only | Validated |
| Realtime + NPU | GPU→NPU | Validated |
| No GPU | CPU-only | Validated |
| No GPU + NPU + realtime | NPU-only | Validated |

## CPU vs GPU Parity

| Operation | Max Diff | Status |
|-----------|----------|--------|
| MatMul 32×32 (GPU vs Rust) | 2.8e-9 | PASS |
| MatMul 32×32 (CPU vs Rust) | 2.3e-9 | PASS |
| GPU vs CPU cross-hardware | 0.0e0 | PASS (bit-identical) |
| ReLU | 0.0e0 | PASS (bit-identical) |
| Sigmoid | 1.2e-7 | PASS |
| Tanh | 1.2e-7 | PASS |
| Sum reduction | ~4.6e-3 | PASS (f32 ordering) |
| erf(0) | 0.0e0 | PASS (exact) |
| erf(1) | 1.0e-6 | PASS |
| gamma(5) | 0.0e0 | PASS (exact) |
| conv2d identity | exact | PASS |
| max_pool2d 2×2 | exact | PASS |

## PCIe Bridge

| Check | Result |
|-------|--------|
| Default no P2P | Validated |
| Transfer cost positive | Validated |
| P2P detection placeholder | Returns `false` (correct) |
| x16 faster than x4 | Validated |
| Bandwidth constants | 31.5 / 7.9 GB/s |

---

## Next Steps

1. **Hardware benchmarks**: Run actual PCIe transfer timing on RTX 4070
2. **P2P detection**: Implement sysfs IOMMU group check for real P2P capability
3. **NPU integration**: Wire AKD1000 SDK when available
4. **ToadStool absorption**: `mixed.rs` + `pcie_bridge.rs` → `barracuda::unified_hardware`
