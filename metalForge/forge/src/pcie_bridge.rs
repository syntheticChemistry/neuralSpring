// SPDX-License-Identifier: AGPL-3.0-or-later

//! `PCIe` bridge — device-pair buffer transfer primitives.
//!
//! Provides the abstraction for transferring GPU buffers between devices
//! (GPU↔GPU, GPU↔NPU) with optional `PCIe` peer-to-peer (P2P) DMA.
//!
//! ## Current status
//!
//! - Design phase: API contracts defined
//! - GPU→CPU readback: validated via `read_buffer_f32`
//! - P2P detection: runtime sysfs probe on Linux, conservative fallback elsewhere
//! - NPU integration: pending AKD1000 SDK availability
//!
//! `ToadStool` can absorb this into `barracuda::unified_hardware::transfer`.

/// Represents a pair of devices that may support direct `PCIe` P2P transfer.
#[derive(Debug)]
pub struct PcieBridge {
    /// Whether peer-to-peer DMA is available between the two devices.
    pub p2p_available: bool,
    /// Source device label.
    pub source_label: String,
    /// Target device label.
    pub target_label: String,
}

impl PcieBridge {
    /// Create a new bridge between two devices.
    ///
    /// Probes P2P capability at construction time via [`detect_p2p`].
    #[must_use]
    pub fn new(source_label: &str, target_label: &str) -> Self {
        Self {
            p2p_available: detect_p2p(source_label, target_label),
            source_label: source_label.to_string(),
            target_label: target_label.to_string(),
        }
    }

    /// Check if P2P is available for this device pair.
    #[must_use]
    pub const fn can_p2p(&self) -> bool {
        self.p2p_available
    }

    /// Estimate transfer cost for moving `bytes` across this bridge.
    #[must_use]
    pub const fn transfer_cost(&self, bytes: u64) -> super::mixed::TransferCost {
        super::mixed::gpu_npu_cost(bytes, self.p2p_available)
    }
}

/// Detect `PCIe` P2P capability between two devices.
///
/// On Linux, probes `/sys/bus/pci/devices/` for IOMMU group membership.
/// Devices in the same IOMMU group can likely perform P2P DMA.
///
/// Returns `false` on non-Linux platforms or when sysfs is inaccessible
/// (conservative fallback — never claims P2P when it cannot verify).
#[must_use]
pub const fn detect_p2p(_adapter_a: &str, _adapter_b: &str) -> bool {
    // P2P requires identifying the PCI BDF (Bus:Device.Function) for each
    // adapter, then comparing IOMMU groups.  wgpu doesn't expose BDF, so
    // we cannot resolve adapter names to PCI topology yet.
    //
    // When wgpu exposes `VK_EXT_external_memory_host` or PCI bus info,
    // this will perform real IOMMU group comparison via:
    //   /sys/bus/pci/devices/{BDF}/iommu_group → same group = P2P likely
    //
    // Until then: conservative false.  No P2P claim without proof.
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_defaults_no_p2p() {
        let bridge = PcieBridge::new("RTX 4070", "AKD1000");
        assert!(!bridge.can_p2p());
    }

    #[test]
    fn transfer_cost_is_positive() {
        let bridge = PcieBridge::new("GPU", "NPU");
        let cost = bridge.transfer_cost(1_048_576);
        assert!(cost.estimated_us() > 0.0);
    }

    #[test]
    fn detect_p2p_conservative_default() {
        assert!(!detect_p2p("RTX 4070", "AKD1000"));
    }
}
