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
    /// Run compute entirely on the GPU.
    GpuOnly,
    /// Run compute entirely on the CPU.
    CpuOnly,
    /// Run inference entirely on the NPU.
    NpuOnly,
    /// Transfer results from GPU to CPU after compute.
    GpuToCpu,
    /// Stage or upload data from CPU to GPU before compute.
    CpuToGpu,
    /// Hand off from GPU to NPU (e.g. for realtime inference).
    GpuToNpu,
    /// Transfer from NPU to GPU with CPU staging when P2P is unavailable.
    NpuToGpu,
    /// NPU→GPU via `PCIe` P2P DMA (bypasses CPU roundtrip).
    ///
    /// Used when IOMMU group analysis confirms P2P is available between
    /// the NPU and GPU devices. Falls back to `NpuToGpu` (CPU-staged)
    /// when P2P is unavailable.
    NpuToGpuP2P,
}

/// Estimated transfer cost for a cross-device data movement.
#[derive(Debug, Clone, Copy)]
pub struct TransferCost {
    /// Payload size moved across the link in bytes.
    pub bytes: u64,
    /// Fixed DMA or staging latency in microseconds.
    pub latency_us: f64,
    /// Effective link bandwidth in gigabytes per second.
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
/// - If NPU→GPU transfer needed and P2P available, use `NpuToGpuP2P`
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

/// Bandwidth tier for a given `PCIe` link configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthTier {
    /// `PCIe` 4.0 x16 (31.5 GB/s) — discrete GPU
    Pcie4X16,
    /// `PCIe` 4.0 x4 (7.9 GB/s) — NPU / M.2
    Pcie4X4,
    /// `PCIe` 5.0 x16 (63.0 GB/s) — next-gen GPU
    Pcie5X16,
    /// Shared memory / same die (~200 GB/s)
    SharedMemory,
}

impl BandwidthTier {
    /// Bandwidth in GB/s for this tier.
    #[must_use]
    pub const fn bandwidth_gbps(self) -> f64 {
        match self {
            Self::Pcie4X16 => PCIE4_X16_BANDWIDTH_GBPS,
            Self::Pcie4X4 => PCIE4_X4_BANDWIDTH_GBPS,
            Self::Pcie5X16 => 63.0,
            Self::SharedMemory => 200.0,
        }
    }

    /// Latency in microseconds for this tier.
    #[must_use]
    pub const fn latency_us(self) -> f64 {
        match self {
            Self::SharedMemory => 0.1,
            _ => PCIE_DMA_LATENCY_US,
        }
    }
}

/// Transfer cost for a specific bandwidth tier.
#[must_use]
pub const fn transfer_cost_for_tier(bytes: u64, tier: BandwidthTier) -> TransferCost {
    TransferCost {
        bytes,
        latency_us: tier.latency_us(),
        bandwidth_gbps: tier.bandwidth_gbps(),
    }
}

/// Chained substrate transfer: source → intermediate → target.
///
/// Models multi-hop transfers like GPU → CPU → NPU (when P2P is unavailable).
/// Returns the total transfer cost for the two-hop path.
#[must_use]
pub fn chained_transfer_cost(
    bytes: u64,
    hop1_tier: BandwidthTier,
    hop2_tier: BandwidthTier,
) -> TransferCost {
    let cost1 = transfer_cost_for_tier(bytes, hop1_tier);
    let cost2 = transfer_cost_for_tier(bytes, hop2_tier);
    TransferCost {
        bytes,
        latency_us: cost1.latency_us + cost2.latency_us + CPU_STAGED_LATENCY_US,
        bandwidth_gbps: f64::min(hop1_tier.bandwidth_gbps(), hop2_tier.bandwidth_gbps()),
    }
}

/// Compare direct P2P vs CPU-staged transfer for the same byte count.
///
/// Returns `(p2p_cost, staged_cost, p2p_is_faster)`.
#[must_use]
pub fn compare_transfer_paths(
    bytes: u64,
    direct_tier: BandwidthTier,
    staged_hop1: BandwidthTier,
    staged_hop2: BandwidthTier,
) -> (TransferCost, TransferCost, bool) {
    let p2p = transfer_cost_for_tier(bytes, direct_tier);
    let staged = chained_transfer_cost(bytes, staged_hop1, staged_hop2);
    let p2p_faster = p2p.estimated_us() < staged.estimated_us();
    (p2p, staged, p2p_faster)
}

/// Select mixed substrate with P2P awareness for NPU→GPU transfers.
///
/// When `bridge` is provided and P2P is available, uses `NpuToGpuP2P`
/// instead of `NpuToGpu` for transfers from NPU to GPU, bypassing the
/// CPU roundtrip.
#[must_use]
pub fn mixed_substrate_p2p(
    compute_us: f64,
    data_bytes: u64,
    gpu_available: bool,
    npu_available: bool,
    needs_realtime_inference: bool,
    bridge: Option<&super::pcie_bridge::PcieBridge>,
) -> MixedSubstrate {
    let base = mixed_substrate(
        compute_us,
        data_bytes,
        gpu_available,
        npu_available,
        needs_realtime_inference,
    );

    match base {
        MixedSubstrate::NpuToGpu if bridge.is_some_and(super::pcie_bridge::PcieBridge::can_p2p) => {
            MixedSubstrate::NpuToGpuP2P
        }
        MixedSubstrate::GpuToNpu if bridge.is_some_and(super::pcie_bridge::PcieBridge::can_p2p) => {
            MixedSubstrate::GpuToNpu
        }
        other => other,
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

    #[test]
    fn bandwidth_tier_values() {
        assert!((BandwidthTier::Pcie4X16.bandwidth_gbps() - 31.5).abs() < 0.1);
        assert!((BandwidthTier::Pcie4X4.bandwidth_gbps() - 7.9).abs() < 0.1);
        assert!((BandwidthTier::Pcie5X16.bandwidth_gbps() - 63.0).abs() < 0.1);
        assert!(BandwidthTier::SharedMemory.bandwidth_gbps() > 100.0);
    }

    #[test]
    fn chained_slower_than_direct() {
        let direct = transfer_cost_for_tier(1_048_576, BandwidthTier::Pcie4X4);
        let chained =
            chained_transfer_cost(1_048_576, BandwidthTier::Pcie4X16, BandwidthTier::Pcie4X4);
        assert!(
            chained.estimated_us() > direct.estimated_us(),
            "2-hop should be slower: {:.1} vs {:.1}",
            chained.estimated_us(),
            direct.estimated_us()
        );
    }

    #[test]
    fn compare_transfer_paths_p2p_wins() {
        let (p2p, staged, faster) = compare_transfer_paths(
            4_194_304,
            BandwidthTier::Pcie4X4,
            BandwidthTier::Pcie4X16,
            BandwidthTier::Pcie4X4,
        );
        assert!(faster, "P2P should beat staged for 4MB");
        assert!(p2p.estimated_us() < staged.estimated_us());
    }

    #[test]
    fn shared_memory_very_fast() {
        let cost = transfer_cost_for_tier(1_048_576, BandwidthTier::SharedMemory);
        assert!(
            cost.estimated_us() < 10.0,
            "shared memory 1MB should be <10µs, got {:.1}",
            cost.estimated_us()
        );
    }

    #[test]
    fn p2p_routing_without_bridge_uses_base() {
        let sub = mixed_substrate_p2p(100_000.0, 1_048_576, true, false, false, None);
        assert_eq!(sub, MixedSubstrate::GpuOnly);
    }

    #[test]
    fn npu_to_gpu_p2p_variant_exists() {
        assert_ne!(MixedSubstrate::NpuToGpuP2P, MixedSubstrate::NpuToGpu);
    }
}
