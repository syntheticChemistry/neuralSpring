// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU compute wrapper for `neuralSpring` validation and benchmarking.
//!
//! Thin config layer over `barracuda::device::WgpuDevice` — imitates the
//! hotSpring `GpuF64` pattern.  No abstraction: exposes raw `wgpu::Device`
//! and `wgpu::Queue` so evolved ops can manage buffers directly.
//!
//! ## Backend selection
//!
//! Set `NEURALSPRING_BACKEND` to control the adapter:
//!
//! | Value | Behaviour |
//! |-------|-----------|
//! | `auto` (default) | Best available (`HighPerformance`) |
//! | `cpu` | Force CPU software rasterizer (llvmpipe) |
//! | `gpu` | Force discrete / integrated GPU |
//! | `list` | Print all adapters and exit |
//! | name/index | Adapter name substring or enumeration index |

use barracuda::device::WgpuDevice;
use bytemuck;
use std::sync::Arc;

/// GPU context for `neuralSpring` workloads.
///
/// Wraps `WgpuDevice` with relaxed limits (llvmpipe caps at 128 MB)
/// and exposes raw `wgpu` handles for direct buffer management.
pub struct Gpu {
    wgpu_device: Arc<WgpuDevice>,
    pub adapter_name: String,
    pub device_type: wgpu::DeviceType,
    pub backend: wgpu::Backend,
}

impl Gpu {
    /// Create from a `WgpuDevice` (already initialised).
    #[must_use]
    pub fn from_device(dev: Arc<WgpuDevice>) -> Self {
        let info = dev.adapter_info();
        Self {
            adapter_name: info.name.clone(),
            device_type: info.device_type,
            backend: info.backend,
            wgpu_device: dev,
        }
    }

    /// Create with the default backend (`NEURALSPRING_BACKEND` env var).
    ///
    /// Uses relaxed limits so CPU software adapters (llvmpipe) work.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested backend is unavailable.
    pub async fn new() -> Result<Self, String> {
        let selector = std::env::var("NEURALSPRING_BACKEND")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();

        if selector == "list" {
            let adapters = WgpuDevice::enumerate_adapters();
            for (i, info) in adapters.iter().enumerate() {
                eprintln!(
                    "  [{i}] {name} ({ty:?}, {backend:?})",
                    name = info.name,
                    ty = info.device_type,
                    backend = info.backend,
                );
            }
            std::process::exit(0);
        }

        match selector.as_str() {
            "gpu" => match WgpuDevice::new_gpu().await {
                Ok(dev) => Ok(Self::from_device(Arc::new(dev))),
                Err(e) => Err(format!("gpu: {e}")),
            },
            "" | "auto" => match WgpuDevice::new().await {
                Ok(dev) => Ok(Self::from_device(Arc::new(dev))),
                Err(e) => Err(format!("auto: {e}")),
            },
            other => Self::create_relaxed(other).await,
        }
    }

    /// Create with the CPU software backend (llvmpipe).
    ///
    /// # Errors
    ///
    /// Returns an error if no CPU adapter is available.
    pub async fn new_cpu() -> Result<Self, String> {
        Self::create_relaxed("cpu").await
    }

    /// Create with a discrete/integrated GPU backend.
    ///
    /// # Errors
    ///
    /// Returns an error if no GPU adapter is available.
    pub async fn new_gpu() -> Result<Self, String> {
        match WgpuDevice::new_gpu().await {
            Ok(dev) => Ok(Self::from_device(Arc::new(dev))),
            Err(e) => Err(format!("gpu: {e}")),
        }
    }

    /// Raw `wgpu::Device` handle — for pipeline creation and buffer ops.
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        self.wgpu_device.device()
    }

    /// Raw `wgpu::Queue` handle — for command submission.
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        self.wgpu_device.queue()
    }

    /// Bridge to barracuda `WgpuDevice` — for Tensor API interop.
    #[must_use]
    pub const fn wgpu_device(&self) -> &Arc<WgpuDevice> {
        &self.wgpu_device
    }

    /// Compile a WGSL shader via barracuda's shader compiler.
    #[must_use]
    pub fn compile_shader(&self, source: &str, label: &str) -> wgpu::ShaderModule {
        self.wgpu_device.compile_shader(source, Some(label))
    }

    /// Allocate a GPU storage buffer for `count` f32 values.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU allocation fails.
    pub fn create_buffer_f32(&self, count: usize) -> Result<wgpu::Buffer, String> {
        self.wgpu_device
            .create_buffer_f32(count)
            .map_err(|e| format!("create_buffer_f32: {e}"))
    }

    /// Upload f32 data to a new GPU storage buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer allocation fails.
    pub fn upload_f32(&self, data: &[f32]) -> Result<wgpu::Buffer, String> {
        let buf = self.create_buffer_f32(data.len())?;
        self.wgpu_device
            .queue()
            .write_buffer(&buf, 0, bytemuck::cast_slice(data));
        Ok(buf)
    }

    /// Read f32 data back from a GPU buffer (blocking).
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU readback fails.
    pub fn read_buffer_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f32>, String> {
        self.wgpu_device
            .read_buffer_f32(buffer, count)
            .map_err(|e| format!("read_buffer_f32: {e}"))
    }

    /// Create a device with relaxed limits for CPU software adapters.
    ///
    /// `barracuda`'s `science_limits()` requests 512 MB which llvmpipe
    /// cannot provide.  We use `downlevel_defaults` instead — our
    /// validation tensors are tiny.
    async fn create_relaxed(selector: &str) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all());

        let adapter = if selector == "cpu" {
            adapters
                .into_iter()
                .find(|a| a.get_info().device_type == wgpu::DeviceType::Cpu)
        } else if let Ok(idx) = selector.parse::<usize>() {
            adapters.into_iter().nth(idx)
        } else {
            let sel = selector.to_ascii_lowercase();
            adapters
                .into_iter()
                .find(|a| a.get_info().name.to_ascii_lowercase().contains(&sel))
        }
        .ok_or_else(|| format!("no adapter matches '{selector}'"))?;

        let info = adapter.get_info();
        let mut features = wgpu::Features::empty();
        let af = adapter.features();
        if af.contains(wgpu::Features::SHADER_F64) {
            features |= wgpu::Features::SHADER_F64;
        }
        if af.contains(wgpu::Features::SHADER_F16) {
            features |= wgpu::Features::SHADER_F16;
        }
        if af.contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("neuralSpring validation (relaxed limits)"),
                    required_features: features,
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("device creation: {e}"))?;

        let dev = WgpuDevice::from_existing(Arc::new(device), Arc::new(queue), info);
        Ok(Self::from_device(Arc::new(dev)))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[tokio::test]
    async fn gpu_new_succeeds() {
        let Ok(gpu) = Gpu::new().await else { return };
        assert!(!gpu.adapter_name.is_empty());
    }

    #[tokio::test]
    async fn gpu_upload_and_readback_roundtrip() {
        let Ok(gpu) = Gpu::new().await else { return };
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let buf = gpu.upload_f32(&data).expect("upload_f32 should succeed");
        let out = gpu
            .read_buffer_f32(&buf, data.len())
            .expect("read_buffer_f32 should succeed");
        for (i, (&got, &want)) in out.iter().zip(data.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-7,
                "roundtrip mismatch at {i}: got {got}, want {want}"
            );
        }
    }

    #[tokio::test]
    async fn gpu_create_empty_buffer() {
        let Ok(gpu) = Gpu::new().await else { return };
        let buf = gpu.create_buffer_f32(64).expect("should create buffer");
        let out = gpu
            .read_buffer_f32(&buf, 64)
            .expect("read_buffer_f32 should succeed");
        assert_eq!(out.len(), 64);
    }

    #[tokio::test]
    async fn gpu_compile_trivial_shader() {
        let Ok(gpu) = Gpu::new().await else { return };
        let _module =
            gpu.compile_shader("@compute @workgroup_size(1) fn main() {}", "test_trivial");
    }

    #[tokio::test]
    async fn gpu_wgpu_device_accessible() {
        let Ok(gpu) = Gpu::new().await else { return };
        let _device = gpu.wgpu_device();
        let _raw_device = gpu.device();
        let _raw_queue = gpu.queue();
    }
}
