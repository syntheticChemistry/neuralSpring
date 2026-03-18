// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hardware probing — GPU via wgpu/`BarraCUDA`, CPU via procfs.
//!
//! GPU discovery leans on wgpu (same path `ToadStool`/`BarraCUDA` uses). We get
//! adapter name, device type, driver, backend, and feature flags (`SHADER_F64`)
//! directly from the Vulkan/wgpu layer.
//!
//! CPU discovery reads `/proc/cpuinfo` for model, core count, and SIMD flags.
//!
//! ## Absorption target: `barracuda::unified_hardware::discovery`
//!
//! Upstream `HardwareDiscovery` exists but doesn't yet expose the substrate
//! model we need. Once it does, this module becomes a thin wrapper.

use crate::substrate::{Capability, Identity, Properties, Substrate, SubstrateKind};
use std::fs;
use std::sync::OnceLock;

/// Cached GPU probe result (groundSpring V116 pattern).
///
/// Creating a `wgpu::Instance` is expensive and can SIGSEGV when multiple
/// threads race. Cache the result so parallel tests and repeated probes
/// share a single discovery pass.
static GPU_PROBE_CACHE: OnceLock<Vec<Substrate>> = OnceLock::new();

/// Probe all GPU adapters via wgpu (cached after first call).
///
/// Uses the same wgpu instance/backend configuration that `BarraCUDA` uses.
/// Each adapter becomes a substrate with capabilities derived from its
/// feature flags (`SHADER_F64` → `F64Compute`, etc.).
#[must_use]
pub fn probe_gpus() -> Vec<Substrate> {
    GPU_PROBE_CACHE.get_or_init(probe_gpus_inner).clone()
}

fn probe_gpus_inner() -> Vec<Substrate> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    let mut gpus = Vec::new();

    for (idx, adapter) in adapters.into_iter().enumerate() {
        let info = adapter.get_info();
        let features = adapter.features();

        if info.device_type == wgpu::DeviceType::Cpu {
            continue;
        }

        let has_f64 = features.contains(wgpu::Features::SHADER_F64);
        let has_timestamps = features.contains(wgpu::Features::TIMESTAMP_QUERY);

        let mut capabilities = vec![Capability::F32Compute, Capability::ShaderDispatch];
        if has_f64 {
            capabilities.push(Capability::F64Compute);
            capabilities.push(Capability::ScalarReduce);
            capabilities.push(Capability::Eigensolve);
            capabilities.push(Capability::FusedMapReduce);
        }
        if has_timestamps {
            capabilities.push(Capability::TimestampQuery);
        }

        let limits = adapter.limits();
        let max_buffer = limits.max_buffer_size;

        gpus.push(Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity {
                name: info.name.clone(),
                driver: Some(format!("{} ({})", info.driver, info.driver_info)),
                backend: Some(format!("{:?}", info.backend)),
                adapter_index: Some(idx),
                pci_id: None,
            },
            properties: Properties {
                memory_bytes: Some(max_buffer),
                has_f64,
                has_timestamps,
                ..Properties::default()
            },
            capabilities,
        });
    }

    gpus
}

/// Probe CPU via `/proc/cpuinfo` and `/proc/meminfo`.
#[must_use]
pub fn probe_cpu() -> Substrate {
    let (model, cores, threads, cache_kb, has_avx2) = parse_cpuinfo();
    let mem_bytes = parse_meminfo();

    let name = model.unwrap_or_else(|| String::from("Unknown CPU"));

    let mut capabilities = vec![
        Capability::F64Compute,
        Capability::F32Compute,
        Capability::Eigensolve,
        Capability::CpuCompute,
    ];
    if has_avx2 {
        capabilities.push(Capability::SimdVector);
    }

    Substrate {
        kind: SubstrateKind::Cpu,
        identity: Identity::named(name),
        properties: Properties {
            memory_bytes: mem_bytes,
            core_count: cores,
            thread_count: threads,
            cache_kb,
            ..Properties::default()
        },
        capabilities,
    }
}

#[cfg(target_os = "linux")]
fn parse_cpuinfo() -> (Option<String>, Option<u32>, Option<u32>, Option<u32>, bool) {
    let Ok(content) = fs::read_to_string("/proc/cpuinfo") else {
        return (None, None, None, None, false);
    };

    let mut model = None;
    let mut cores = None;
    let mut siblings = None;
    let mut cache_kb = None;
    let mut has_avx2 = false;

    for line in content.lines() {
        if let Some((key, val)) = line.split_once(':') {
            let key = key.trim();
            let val = val.trim();
            match key {
                "model name" if model.is_none() => model = Some(val.to_string()),
                "cpu cores" if cores.is_none() => cores = val.parse().ok(),
                "siblings" if siblings.is_none() => siblings = val.parse().ok(),
                "cache size" if cache_kb.is_none() => {
                    cache_kb = val.trim_end_matches(" KB").parse().ok();
                }
                "flags" if !has_avx2 => {
                    has_avx2 = val.split_whitespace().any(|f| f == "avx2");
                }
                _ => {}
            }
        }
    }

    (model, cores, siblings, cache_kb, has_avx2)
}

#[cfg(not(target_os = "linux"))]
fn parse_cpuinfo() -> (Option<String>, Option<u32>, Option<u32>, Option<u32>, bool) {
    (None, None, None, None, false)
}

#[cfg(target_os = "linux")]
fn parse_meminfo() -> Option<u64> {
    let content = fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb_str = rest.trim().trim_end_matches(" kB").trim();
            let kb: u64 = kb_str.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn parse_meminfo() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_always_discovered() {
        let cpu = probe_cpu();
        assert_eq!(cpu.kind, SubstrateKind::Cpu);
        assert!(cpu.has(&Capability::F64Compute));
        assert!(!cpu.identity.name.is_empty());
    }

    #[test]
    fn gpu_probe_uses_wgpu() {
        let gpus = probe_gpus();
        for gpu in &gpus {
            assert_eq!(gpu.kind, SubstrateKind::Gpu);
            assert!(gpu.has(&Capability::ShaderDispatch));
            assert!(gpu.identity.adapter_index.is_some());
            assert!(gpu.identity.driver.is_some());
        }
    }
}
