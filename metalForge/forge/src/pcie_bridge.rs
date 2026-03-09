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

    /// Determine the optimal transfer strategy for a buffer of `bytes`.
    ///
    /// Returns `TransferStrategy::P2P` when P2P DMA is available (same IOMMU
    /// group), otherwise `TransferStrategy::CpuStaged`. The actual buffer
    /// transfer is performed by `wgpu`/`barracuda` — this method only
    /// selects the strategy.
    #[must_use]
    pub const fn transfer_buffer_strategy(&self, bytes: u64) -> TransferStrategy {
        if self.p2p_available {
            TransferStrategy::P2P {
                cost: super::mixed::gpu_npu_cost(bytes, true),
            }
        } else {
            TransferStrategy::CpuStaged {
                cost: super::mixed::gpu_npu_cost(bytes, false),
            }
        }
    }
}

/// Strategy for cross-device buffer transfer.
#[derive(Debug)]
pub enum TransferStrategy {
    /// Direct peer-to-peer DMA (same IOMMU group, `PCIe` fabric).
    P2P {
        /// Estimated transfer cost.
        cost: super::mixed::TransferCost,
    },
    /// CPU-staged copy (GPU→host→GPU, higher latency).
    CpuStaged {
        /// Estimated transfer cost.
        cost: super::mixed::TransferCost,
    },
}

/// Detect `PCIe` P2P capability between two devices.
///
/// On Linux, probes `/sys/bus/pci/devices/` for IOMMU group membership.
/// Devices in the same IOMMU group can likely perform P2P DMA.
///
/// Returns `false` on non-Linux platforms or when sysfs is inaccessible
/// (conservative fallback — never claims P2P when it cannot verify).
#[must_use]
pub fn detect_p2p(adapter_a: &str, adapter_b: &str) -> bool {
    detect_p2p_impl(adapter_a, adapter_b)
}

#[cfg(target_os = "linux")]
fn detect_p2p_impl(adapter_a: &str, adapter_b: &str) -> bool {
    // Attempt to match adapter names to PCI devices via sysfs.
    // Each PCI device at /sys/bus/pci/devices/{BDF}/ contains a symlink
    // `iommu_group` → ../../../kernel/iommu_groups/{N}.
    // If two GPU devices share the same IOMMU group, P2P DMA is likely.
    let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") else {
        return false;
    };

    let mut group_a = None;
    let mut group_b = None;

    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(vendor) = std::fs::read_to_string(path.join("vendor")) else {
            continue;
        };
        let Ok(device_id) = std::fs::read_to_string(path.join("device")) else {
            continue;
        };
        let label_hint = format!("{}:{}", vendor.trim(), device_id.trim());

        let Ok(iommu_link) = std::fs::read_link(path.join("iommu_group")) else {
            continue;
        };
        let group_id = iommu_link
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        if adapter_a.contains(&label_hint) || label_hint.contains(adapter_a) {
            group_a = Some(group_id.to_string());
        }
        if adapter_b.contains(&label_hint) || label_hint.contains(adapter_b) {
            group_b = Some(group_id.to_string());
        }
    }

    // Only claim P2P when both devices are found AND share an IOMMU group.
    matches!((group_a, group_b), (Some(a), Some(b)) if !a.is_empty() && a == b)
}

#[cfg(not(target_os = "linux"))]
fn detect_p2p_impl(_adapter_a: &str, _adapter_b: &str) -> bool {
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

    #[test]
    fn transfer_strategy_cpu_staged_without_p2p() {
        let bridge = PcieBridge::new("GPU_A", "GPU_B");
        let strategy = bridge.transfer_buffer_strategy(1_048_576);
        assert!(matches!(strategy, TransferStrategy::CpuStaged { .. }));
    }

    #[test]
    fn transfer_strategy_provides_cost() {
        let bridge = PcieBridge::new("GPU_A", "NPU_A");
        match bridge.transfer_buffer_strategy(4_194_304) {
            TransferStrategy::CpuStaged { cost } => {
                assert!(cost.estimated_us() > 0.0);
                assert_eq!(cost.bytes, 4_194_304);
            }
            TransferStrategy::P2P { cost } => {
                assert!(cost.estimated_us() > 0.0);
                assert_eq!(cost.bytes, 4_194_304);
            }
        }
    }
}
