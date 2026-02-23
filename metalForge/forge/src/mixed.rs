// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(clippy::cast_precision_loss)]

//! Mixed-hardware substrate selection — GPU ↔ NPU ↔ CPU dispatch.
//!
//! Extends `dispatch.rs` with cross-device routing heuristics that account
//! for `PCIe` transfer costs. `ToadStool` can absorb this into
//! `barracuda::unified_hardware` for ecosystem-wide mixed dispatch.

/// Mixed dispatch substrate — extends `Substrate` with cross-device targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedSubstrate {
    GpuOnly,
    CpuOnly,
    NpuOnly,
    GpuToCpu,
    CpuToGpu,
    GpuToNpu,
    NpuToGpu,
}

/// Estimated transfer cost for a cross-device data movement.
#[derive(Debug, Clone, Copy)]
pub struct TransferCost {
    pub bytes: u64,
    pub latency_us: f64,
    pub bandwidth_gbps: f64,
}

impl TransferCost {
    /// Estimate total transfer time in microseconds.
    #[must_use]
    pub fn estimated_us(&self) -> f64 {
        self.latency_us + (self.bytes as f64) / (self.bandwidth_gbps * 1e3)
    }
}

/// `PCIe` 4.0 x16 bandwidth in GB/s.
pub const PCIE4_X16_BANDWIDTH_GBPS: f64 = 31.5;

/// `PCIe` 4.0 x4 bandwidth in GB/s (typical NPU link).
pub const PCIE4_X4_BANDWIDTH_GBPS: f64 = 7.9;

/// Estimated latency for `PCIe` DMA transfer in microseconds.
pub const PCIE_DMA_LATENCY_US: f64 = 2.0;

/// CPU-staged transfer latency (GPU→CPU→NPU) in microseconds.
pub const CPU_STAGED_LATENCY_US: f64 = 7.0;

/// Estimate GPU↔CPU transfer cost.
#[must_use]
pub const fn gpu_cpu_cost(bytes: u64) -> TransferCost {
    TransferCost {
        bytes,
        latency_us: PCIE_DMA_LATENCY_US,
        bandwidth_gbps: PCIE4_X16_BANDWIDTH_GBPS,
    }
}

/// Estimate GPU↔NPU transfer cost (assumes CPU staging; P2P halves latency).
#[must_use]
pub const fn gpu_npu_cost(bytes: u64, p2p_available: bool) -> TransferCost {
    if p2p_available {
        TransferCost {
            bytes,
            latency_us: PCIE_DMA_LATENCY_US,
            bandwidth_gbps: PCIE4_X4_BANDWIDTH_GBPS,
        }
    } else {
        TransferCost {
            bytes,
            latency_us: CPU_STAGED_LATENCY_US,
            bandwidth_gbps: PCIE4_X4_BANDWIDTH_GBPS,
        }
    }
}

/// Select optimal mixed substrate for a workload.
///
/// Uses a simple cost model:
/// - If compute dominates transfer, use GPU
/// - If real-time inference needed and NPU available, use GPU→NPU
/// - If transfer cost exceeds compute benefit, use CPU
#[must_use]
pub fn mixed_substrate(
    compute_us: f64,
    data_bytes: u64,
    gpu_available: bool,
    npu_available: bool,
    needs_realtime_inference: bool,
) -> MixedSubstrate {
    if !gpu_available {
        if npu_available && needs_realtime_inference {
            return MixedSubstrate::NpuOnly;
        }
        return MixedSubstrate::CpuOnly;
    }

    let gpu_transfer = gpu_cpu_cost(data_bytes).estimated_us();

    if needs_realtime_inference && npu_available {
        return MixedSubstrate::GpuToNpu;
    }

    if compute_us > gpu_transfer + super::dispatch::GPU_DISPATCH_OVERHEAD_US as f64 {
        MixedSubstrate::GpuOnly
    } else {
        MixedSubstrate::CpuOnly
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_cpu_cost_1mb() {
        let cost = gpu_cpu_cost(1_048_576);
        let us = cost.estimated_us();
        assert!(
            us > 30.0 && us < 50.0,
            "1MB GPU→CPU should be ~35µs, got {us}"
        );
    }

    #[test]
    fn gpu_npu_p2p_faster_than_staged() {
        let p2p = gpu_npu_cost(1_048_576, true);
        let staged = gpu_npu_cost(1_048_576, false);
        assert!(p2p.estimated_us() < staged.estimated_us());
    }

    #[test]
    fn small_workload_uses_cpu() {
        let sub = mixed_substrate(100.0, 1024, true, false, false);
        assert_eq!(sub, MixedSubstrate::CpuOnly);
    }

    #[test]
    fn large_workload_uses_gpu() {
        let sub = mixed_substrate(100_000.0, 1_048_576, true, false, false);
        assert_eq!(sub, MixedSubstrate::GpuOnly);
    }

    #[test]
    fn realtime_inference_uses_npu() {
        let sub = mixed_substrate(50_000.0, 1_048_576, true, true, true);
        assert_eq!(sub, MixedSubstrate::GpuToNpu);
    }
}
