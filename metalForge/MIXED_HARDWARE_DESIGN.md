# metalForge — Mixed-Hardware Dispatch Design

**Parent**: ecoPrimals/neuralSpring/metalForge
**License**: AGPL-3.0-or-later
**Date**: February 22, 2026
**Status**: Design — implementation evolving locally for ToadStool absorption

---

## Vision

Enable zero-copy data movement between heterogeneous compute units:
GPU (RTX 4070) ↔ NPU (AKD1000) ↔ CPU (i9-12900K), with automatic
dispatch based on workload characteristics and hardware topology.

---

## Architecture

### Device Topology

```text
┌─────────────────────────────────────┐
│           i9-12900K (CPU)           │
│  DDR5-4800 (31.5 GB/s per channel) │
├──────────┬──────────────────────────┤
│ PCIe 4.0 │        PCIe 4.0         │
│ x16      │        x4               │
│ (31.5    │        (7.9 GB/s)       │
│  GB/s)   │                         │
├──────────┤──────────────────────────┤
│ RTX 4070 │       AKD1000           │
│ 12GB     │       (NPU)             │
│ GDDR6X   │                         │
└──────────┴──────────────────────────┘
```

### Dispatch Substrate Hierarchy

| Substrate | Transfer | Latency | Bandwidth | Use Case |
|-----------|----------|---------|-----------|----------|
| GPU-only | None | 0 | ∞ (local) | Large tensor ops, GEMM, FFT |
| CPU-only | None | 0 | ∞ (local) | Small ops, scalar feedback, fallback |
| GPU→CPU | PCIe DMA | ~2 µs | 31.5 GB/s | Readback for selection, scalar reduce |
| CPU→GPU | PCIe DMA | ~2 µs | 31.5 GB/s | Upload population, parameters |
| GPU→NPU | PCIe P2P? | TBD | ~7.9 GB/s | ESN inference after GPU training |
| NPU→GPU | PCIe P2P? | TBD | ~7.9 GB/s | NPU decision → GPU physics update |
| CPU→NPU | PCIe DMA | ~5 µs | 7.9 GB/s | Model upload, parameter update |

### PCIe Peer-to-Peer (P2P) DMA

PCIe P2P enables direct GPU↔NPU buffer transfer without CPU staging:

1. **Requirement**: Both devices on same PCIe root complex
2. **Detection**: `sysfs` IOMMU group check or wgpu adapter features
3. **Benefit**: Eliminates CPU copy → 2× bandwidth, halved latency
4. **Fallback**: CPU-staged copy (GPU→CPU→NPU) if P2P unavailable

### Transfer Cost Model

```
cost(bytes, src, dst) = latency(src, dst) + bytes / bandwidth(src, dst)

Example: 1 MB GPU→CPU via PCIe 4.0 x16:
  cost = 2 µs + 1,048,576 / 31.5e9 ≈ 35 µs

Example: 1 MB GPU→NPU via CPU staging:
  cost = 2 µs + 1,048,576 / 31.5e9 + 5 µs + 1,048,576 / 7.9e9 ≈ 172 µs

Example: 1 MB GPU→NPU via P2P (if available):
  cost = 2 µs + 1,048,576 / 7.9e9 ≈ 135 µs (23% faster)
```

---

## Dispatch Decision Tree

```
Input: (workload_type, data_size, available_devices)

1. If data_size < CPU_THRESHOLD → CPU-only
2. If data_size > GPU_THRESHOLD:
   a. If workload needs real-time inference AND NPU available:
      → GPU compute → NPU inference (P2P if available, else staged)
   b. Else: GPU-only
3. If CPU_THRESHOLD ≤ data_size ≤ GPU_THRESHOLD:
   → Use empirical crossover from metalForge/forge/dispatch.rs
4. For iterative algorithms (EA, HMM chains):
   → StatefulPipeline on GPU (amortizes dispatch overhead)
```

---

## Implementation Plan

### Phase 1: `mixed.rs` — Substrate Abstraction (current)

- `MixedSubstrate` enum with all dispatch targets
- `TransferCost` estimator based on topology
- `mixed_substrate()` heuristic combining workload size + device availability

### Phase 2: `pcie_bridge.rs` — Transfer Primitives

- `PcieBridge` for device-pair buffer transfer
- `can_p2p()` detection via sysfs/adapter features  
- `transfer_buffer()` async API (P2P or staged fallback)

### Phase 3: Integration

- Wire `PcieBridge` into metalForge dispatch
- Benchmark P2P vs staged on actual hardware
- Evolve for ToadStool absorption into `barracuda::unified_hardware`

---

## Hardware Characterization (from metalForge benchmarks)

| Path | Measured | Expected | Notes |
|------|----------|----------|-------|
| GPU dispatch overhead | 1.5 ms | — | `queue.submit()` + readback |
| GPU compute (5888 cores) | < 0.1 ms | — | Negligible at all tested scales |
| CPU→GPU upload 1MB | ~35 µs | 33 µs | PCIe 4.0 x16 |
| GPU→CPU readback 1MB | ~35 µs | 33 µs | PCIe 4.0 x16 |
| NPU inference (AKD1000) | TBD | ~0.1 ms | 15k neuron ESN |

---

*Mixed-hardware dispatch design — GPU ↔ NPU ↔ CPU via PCIe.
Evolving locally for ToadStool absorption.*
