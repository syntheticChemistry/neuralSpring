// SPDX-License-Identifier: AGPL-3.0-or-later

//! Capability-based GPU/CPU dispatch for science operations.
//!
//! Provides a unified execution context that routes operations to GPU
//! when available, falling back to CPU reference implementations.
//! No hardcoded backend: the adapter is discovered at runtime via
//! `NEURALSPRING_BACKEND` (see [`crate::gpu::Gpu`]).
//!
//! ## Design principles
//!
//! - **Self-knowledge only**: `Dispatcher` discovers its capabilities at init
//! - **No mocks in production**: GPU path is real GPU; CPU path is real CPU math
//! - **Agnostic**: works on RTX 4070, TITAN V (NVK), Raspberry Pi, llvmpipe
//! - **Observable**: every dispatch decision is logged to `execution_log`

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

mod basecamp;
mod cpu_fallback;
mod dispatch_ops;

use barracuda::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use barracuda::device::WgpuDevice;
use barracuda::unified_hardware::BandwidthTier;
use std::sync::Arc;

use crate::gpu::{Gpu, GpuCapabilities};

/// Runtime execution backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// GPU via wgpu/Vulkan (includes NVK, proprietary, etc.)
    Gpu,
    /// CPU reference implementation
    Cpu,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gpu => write!(f, "GPU"),
            Self::Cpu => write!(f, "CPU"),
        }
    }
}

/// Capability-based dispatcher for GPU/CPU execution.
///
/// Created once at startup, shared across all science modules.
/// When GPU is available, operations route through upstream
/// `barracuda::dispatch::domain_ops` (cross-spring evolved from hotSpring
/// precision shaders and wetSpring bio shaders). When unavailable,
/// local CPU references are used.
///
/// The `driver_profile` field (from `barracuda::device::GpuDriverProfile`)
/// provides hardware-adaptive f64 strategy, driver workaround detection,
/// and optimal eigensolve configuration — evolved from hotSpring's
/// core-streaming discovery work.
pub struct Dispatcher {
    gpu: Option<Gpu>,
    prefer_gpu: bool,
    driver_profile: Option<GpuDriverProfile>,
}

impl Dispatcher {
    /// Create a dispatcher by probing the runtime GPU environment.
    ///
    /// If GPU initialization fails, falls back to CPU-only mode silently.
    pub async fn new() -> Self {
        match Gpu::new().await {
            Ok(gpu) => {
                let profile = GpuDriverProfile::from_device(gpu.wgpu_device());
                let tier = BandwidthTier::detect_from_adapter_name(&gpu.adapter_name);
                eprintln!(
                    "[dispatch] GPU available: {} ({:?}, {:?}, f64={:?}, pcie={tier:?})",
                    gpu.adapter_name,
                    gpu.device_type,
                    gpu.backend,
                    profile.fp64_strategy(),
                );
                Self {
                    gpu: Some(gpu),
                    prefer_gpu: true,
                    driver_profile: Some(profile),
                }
            }
            Err(e) => {
                eprintln!("[dispatch] No GPU, CPU-only mode: {e}");
                Self {
                    gpu: None,
                    prefer_gpu: false,
                    driver_profile: None,
                }
            }
        }
    }

    /// Create a CPU-only dispatcher (for testing or CI).
    #[must_use]
    pub const fn cpu_only() -> Self {
        Self {
            gpu: None,
            prefer_gpu: false,
            driver_profile: None,
        }
    }

    /// Create from an existing `Gpu` context.
    #[must_use]
    pub fn from_gpu(gpu: Gpu) -> Self {
        let profile = GpuDriverProfile::from_device(gpu.wgpu_device());
        Self {
            gpu: Some(gpu),
            prefer_gpu: true,
            driver_profile: Some(profile),
        }
    }

    /// Whether GPU execution is available.
    #[must_use]
    pub const fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Current preferred backend.
    #[must_use]
    pub const fn backend(&self) -> Backend {
        if self.prefer_gpu && self.gpu.is_some() {
            Backend::Gpu
        } else {
            Backend::Cpu
        }
    }

    /// GPU capabilities (if available).
    #[must_use]
    pub fn capabilities(&self) -> Option<&GpuCapabilities> {
        self.gpu.as_ref().map(|g| &g.capabilities)
    }

    /// Adapter name (if GPU available).
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        self.gpu
            .as_ref()
            .map_or("(none)", |g| g.adapter_name.as_str())
    }

    /// Get the `WgpuDevice` for direct Tensor API usage.
    #[must_use]
    pub fn wgpu_device(&self) -> Option<&Arc<WgpuDevice>> {
        self.gpu.as_ref().map(Gpu::wgpu_device)
    }

    /// Get the `Gpu` context reference.
    #[must_use]
    pub const fn gpu(&self) -> Option<&Gpu> {
        self.gpu.as_ref()
    }

    /// Upstream driver profile (hotSpring-evolved hardware detection).
    #[must_use]
    pub const fn driver_profile(&self) -> Option<&GpuDriverProfile> {
        self.driver_profile.as_ref()
    }

    /// Hardware-adaptive f64 strategy: `Native` on compute-class GPUs
    /// (Titan V, V100, A100 — 1:2 FP64:FP32), `Hybrid` on consumer
    /// GPUs (RTX 4070 — 1:64 ratio, routes bulk math through df64 f32-pairs).
    #[must_use]
    pub fn fp64_strategy(&self) -> Fp64Strategy {
        self.driver_profile
            .as_ref()
            .map_or(Fp64Strategy::Native, GpuDriverProfile::fp64_strategy)
    }

    /// Whether the GPU driver needs `pow(f64,f64)` polyfill workaround.
    #[must_use]
    pub fn needs_pow_workaround(&self) -> bool {
        self.driver_profile
            .as_ref()
            .is_some_and(GpuDriverProfile::needs_pow_f64_workaround)
    }

    /// `PCIe` bandwidth tier detected from the GPU adapter name.
    ///
    /// Returns `BandwidthTier::Unknown` when no GPU is available.
    /// Evolved from hotSpring's cross-device transfer cost modelling.
    #[must_use]
    pub fn bandwidth_tier(&self) -> BandwidthTier {
        self.gpu.as_ref().map_or(BandwidthTier::Unknown, |g| {
            BandwidthTier::detect_from_adapter_name(&g.adapter_name)
        })
    }

    /// Check whether a combined GPU allocation of `total_bytes` is safe.
    ///
    /// On NVK (nouveau), the kernel driver PTE-faults at ~1.4 GB combined.
    /// Returns `Ok(())` if safe or no GPU, `Err` with a diagnostic message
    /// if the allocation would exceed the driver's safe limit.
    ///
    /// # Errors
    ///
    /// Returns [`barracuda::error::BarracudaError::DeviceLimitExceeded`] when
    /// `total_bytes` exceeds the NVK safe allocation limit (~1.2 GB).
    pub fn check_allocation_safe(&self, total_bytes: u64) -> barracuda::error::Result<()> {
        if let Some(ref profile) = self.driver_profile {
            profile.check_allocation_safe(total_bytes)?;
        }
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════
    // Core dispatch helper
    // ═══════════════════════════════════════════════════════════════

    /// Attempt `gpu_fn` on the GPU device; on failure (or absence) run `cpu_fn`.
    ///
    /// Centralises the "try GPU, log-and-fallback" pattern so each public
    /// method only specifies *what* to compute on each backend.
    fn gpu_or_cpu<T>(
        &self,
        op: &str,
        gpu_fn: impl FnOnce(&Arc<WgpuDevice>) -> Result<T, String>,
        cpu_fn: impl FnOnce() -> T,
    ) -> T {
        if let Some(dev) = self.wgpu_device() {
            match gpu_fn(dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] {op} GPU failed, falling back: {e}"),
            }
        }
        cpu_fn()
    }

    // ═══════════════════════════════════════════════════════════════
    // Mixed-hardware dispatch (metalForge integration)
    // ═══════════════════════════════════════════════════════════════

    /// Route a workload using the `metalForge` mixed-hardware cost model.
    ///
    /// Combines `dispatch.rs` substrate heuristics with `mixed.rs` transfer
    /// cost estimation to select the optimal execution path. This is the
    /// wiring point for `ToadStool` to absorb into `barracuda::unified_hardware`.
    #[allow(clippy::too_many_arguments)]
    pub fn mixed_dispatch<T>(
        &self,
        op: &str,
        compute_us: f64,
        data_bytes: u64,
        npu_available: bool,
        needs_realtime: bool,
        gpu_fn: impl FnOnce(&Arc<WgpuDevice>) -> Result<T, String>,
        cpu_fn: impl FnOnce() -> T,
    ) -> (T, neural_spring_forge::mixed::MixedSubstrate) {
        use neural_spring_forge::mixed::{mixed_substrate, MixedSubstrate};

        let substrate = mixed_substrate(
            compute_us,
            data_bytes,
            self.has_gpu(),
            npu_available,
            needs_realtime,
        );

        match substrate {
            MixedSubstrate::GpuOnly | MixedSubstrate::CpuToGpu => {
                if let Some(dev) = self.wgpu_device() {
                    match gpu_fn(dev) {
                        Ok(result) => return (result, substrate),
                        Err(e) => {
                            eprintln!("[mixed-dispatch] {op} GPU failed, falling back: {e}");
                        }
                    }
                }
                (cpu_fn(), MixedSubstrate::CpuOnly)
            }
            MixedSubstrate::GpuToNpu | MixedSubstrate::NpuToGpu | MixedSubstrate::NpuOnly => {
                eprintln!(
                    "[mixed-dispatch] {op} NPU substrate selected but not available, using GPU"
                );
                if let Some(dev) = self.wgpu_device() {
                    match gpu_fn(dev) {
                        Ok(result) => return (result, substrate),
                        Err(e) => {
                            eprintln!("[mixed-dispatch] {op} GPU fallback failed: {e}");
                        }
                    }
                }
                (cpu_fn(), MixedSubstrate::CpuOnly)
            }
            MixedSubstrate::CpuOnly | MixedSubstrate::GpuToCpu => (cpu_fn(), substrate),
        }
    }
}

#[cfg(test)]
#[path = "tests_cpu.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_gpu.rs"]
mod tests_gpu;
