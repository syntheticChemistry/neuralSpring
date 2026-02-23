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
//! - P2P detection: placeholder (requires sysfs access or wgpu extension)
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
    /// Currently defaults to `p2p_available = false` (conservative).
    /// Future: detect via sysfs IOMMU groups or wgpu adapter features.
    #[must_use]
    pub fn new(source_label: &str, target_label: &str) -> Self {
        Self {
            p2p_available: false,
            source_label: source_label.to_string(),
            target_label: target_label.to_string(),
        }
    }

    /// Check if P2P is available for this device pair.
    ///
    /// Currently returns the stored flag. Future: query sysfs at runtime.
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

/// Detect `PCIe` P2P capability between two wgpu adapters.
///
/// Placeholder: always returns `false`. Real implementation would check:
/// 1. Both devices in same IOMMU group (Linux sysfs)
/// 2. Both devices on same `PCIe` root complex
/// 3. wgpu adapter supports external memory import/export
///
/// # Future
///
/// When wgpu exposes `VK_EXT_external_memory_host` or similar,
/// this can perform actual P2P capability detection.
#[must_use]
pub const fn detect_p2p(_adapter_a: &str, _adapter_b: &str) -> bool {
    // TODO: implement sysfs-based detection
    // /sys/bus/pci/devices/{BDF}/iommu_group → same group = P2P likely
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
    fn detect_p2p_returns_false() {
        assert!(!detect_p2p("RTX 4070", "AKD1000"));
    }
}
