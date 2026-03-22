// SPDX-License-Identifier: AGPL-3.0-or-later

//! Substrate abstraction — runtime-discovered compute devices.
//!
//! Following the hotSpring/wetSpring `metalForge` pattern: a substrate is a
//! compute device found on this machine. GPUs come from wgpu adapter enumeration
//! (same path `ToadStool`/`BarraCUDA` uses). CPU comes from procfs.
//!
//! Capabilities are what matters for dispatch — code asks "can you do f64?"
//! not "are you an RTX 4070?".
//!
//! ## Absorption target: `barracuda::unified_hardware::types`
//!
//! Once `ToadStool` absorbs a universal substrate model, this module becomes
//! a thin re-export. Until then, we evolve locally and hand off.

use std::fmt;

/// A compute substrate discovered at runtime.
#[derive(Debug, Clone)]
pub struct Substrate {
    /// Substrate type (GPU or CPU).
    pub kind: SubstrateKind,
    /// Device identity (name and driver/backend identifiers).
    pub identity: Identity,
    /// Measured hardware properties (memory, cores, feature flags).
    pub properties: Properties,
    /// Runtime capability set used for shader dispatch and feature checks.
    pub capabilities: Vec<Capability>,
}

/// How we found this device and what to call it.
#[derive(Debug, Clone)]
pub struct Identity {
    /// Human-readable device name.
    pub name: String,
    /// Graphics or compute driver identifier, if known.
    pub driver: Option<String>,
    /// Backend API (e.g. Vulkan, Metal), if known.
    pub backend: Option<String>,
    /// Enumerated adapter index from the runtime, if applicable.
    pub adapter_index: Option<usize>,
    /// PCI bus ID string for the device, if known.
    pub pci_id: Option<String>,
}

/// Measured properties of a substrate.
#[derive(Debug, Clone, Default)]
pub struct Properties {
    /// Total device memory in bytes, if reported.
    pub memory_bytes: Option<u64>,
    /// Physical or logical core count, if known.
    pub core_count: Option<u32>,
    /// Hardware thread count, if known.
    pub thread_count: Option<u32>,
    /// Last-level cache size in kilobytes, if known.
    pub cache_kb: Option<u32>,
    /// Whether double-precision (f64) shader or CPU compute is available.
    pub has_f64: bool,
    /// Whether GPU timestamp queries are supported.
    pub has_timestamps: bool,
}

/// The kind of compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubstrateKind {
    /// GPU compute device (wgpu adapter).
    Gpu,
    /// CPU compute device (host).
    Cpu,
    /// Neural processing unit (e.g., Intel AKD1000, Apple ANE).
    /// Discovered at runtime; routed via `mixed::MixedSubstrate`.
    Npu,
}

/// A capability discovered at runtime on a substrate.
///
/// ML-focused capabilities (neuralSpring domain) alongside the shared
/// compute primitives that all Springs use.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capability {
    /// IEEE 754 f64 compute (GPU `SHADER_F64` or CPU native).
    F64Compute,
    /// f32 compute.
    F32Compute,
    /// WGSL shader dispatch via wgpu.
    ShaderDispatch,
    /// Scalar reduction (GPU reduce pipeline).
    ScalarReduce,
    /// Eigensolve (`BatchedEighGpu`, Lanczos).
    Eigensolve,
    /// Fused map-reduce (entropy, variance, correlation).
    FusedMapReduce,
    /// AVX2/SSE SIMD on CPU.
    SimdVector,
    /// GPU timestamp query support.
    TimestampQuery,
    /// CPU compute (always available).
    CpuCompute,
    /// NPU low-latency inference (INT8/INT4 quantized).
    NpuInference,
    /// NPU batch processing (throughput-optimized).
    NpuBatch,
}

impl fmt::Display for SubstrateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu => write!(f, "GPU"),
            Self::Cpu => write!(f, "CPU"),
            Self::Npu => write!(f, "NPU"),
        }
    }
}

impl fmt::Display for Substrate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}]", self.identity.name, self.kind)?;
        if let Some(ref driver) = self.identity.driver {
            write!(f, " {driver}")?;
        }
        if let Some(mem) = self.properties.memory_bytes {
            let mb = mem / (1024 * 1024);
            write!(f, " {mb}MB")?;
        }
        Ok(())
    }
}

impl Substrate {
    /// Check if this substrate has a specific capability.
    #[must_use]
    pub fn has(&self, cap: &Capability) -> bool {
        self.capabilities.contains(cap)
    }

    /// Return capabilities as a summary string.
    #[must_use]
    pub fn capability_summary(&self) -> String {
        let labels: Vec<&str> = self.capabilities.iter().map(Capability::label).collect();
        labels.join(", ")
    }
}

impl Capability {
    /// Human-readable label for display.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::F64Compute => "f64",
            Self::F32Compute => "f32",
            Self::ShaderDispatch => "shader",
            Self::ScalarReduce => "reduce",
            Self::Eigensolve => "eigen",
            Self::FusedMapReduce => "fmr",
            Self::SimdVector => "simd",
            Self::TimestampQuery => "timestamps",
            Self::CpuCompute => "cpu",
            Self::NpuInference => "npu-infer",
            Self::NpuBatch => "npu-batch",
        }
    }
}

impl Identity {
    /// Build an identity with only a name; other fields unset.
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            driver: None,
            backend: None,
            adapter_index: None,
            pci_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_gpu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity {
                name: String::from("Test GPU"),
                adapter_index: Some(0),
                ..Identity::named("Test GPU")
            },
            properties: Properties {
                has_f64: true,
                ..Properties::default()
            },
            capabilities: vec![Capability::F64Compute, Capability::ShaderDispatch],
        }
    }

    #[test]
    fn has_capability() {
        let gpu = test_gpu();
        assert!(gpu.has(&Capability::F64Compute));
        assert!(gpu.has(&Capability::ShaderDispatch));
        assert!(!gpu.has(&Capability::Eigensolve));
    }

    #[test]
    fn display_shows_kind_and_name() {
        let gpu = test_gpu();
        let s = format!("{gpu}");
        assert!(s.contains("Test GPU"));
        assert!(s.contains("GPU"));
    }

    #[test]
    fn capability_labels() {
        assert_eq!(Capability::F64Compute.label(), "f64");
        assert_eq!(Capability::ShaderDispatch.label(), "shader");
        assert_eq!(Capability::FusedMapReduce.label(), "fmr");
        assert_eq!(Capability::NpuInference.label(), "npu-infer");
        assert_eq!(Capability::NpuBatch.label(), "npu-batch");
    }

    #[test]
    fn npu_substrate_display() {
        let npu = Substrate {
            kind: SubstrateKind::Npu,
            identity: Identity::named("AKD1000"),
            properties: Properties::default(),
            capabilities: vec![Capability::NpuInference, Capability::NpuBatch],
        };
        let s = format!("{npu}");
        assert!(s.contains("NPU"));
        assert!(npu.has(&Capability::NpuInference));
        assert!(!npu.has(&Capability::F64Compute));
    }
}
