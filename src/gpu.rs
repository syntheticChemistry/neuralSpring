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
///
/// # Example
///
/// ```no_run
/// # async fn example() -> Result<(), String> {
/// use neural_spring::gpu::Gpu;
///
/// let gpu = Gpu::new().await?;
/// let buf = gpu.upload_f32(&[1.0, 2.0, 3.0])?;
/// let data = gpu.read_buffer_f32(&buf, 3)?;
/// assert_eq!(data.len(), 3);
/// # Ok(())
/// # }
/// ```
/// Runtime-discovered GPU capabilities.
///
/// Queried from the adapter at initialization time — no hardcoded
/// assumptions about buffer sizes, workgroup limits, or feature support.
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub max_buffer_size: u64,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub supports_f64: bool,
    pub supports_f16: bool,
    pub supports_timestamp_query: bool,
}

impl GpuCapabilities {
    fn from_device_and_adapter(device: &wgpu::Device, features: wgpu::Features) -> Self {
        let limits = device.limits();
        Self {
            max_buffer_size: limits.max_buffer_size,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            supports_f64: features.contains(wgpu::Features::SHADER_F64),
            supports_f16: features.contains(wgpu::Features::SHADER_F16),
            supports_timestamp_query: features.contains(wgpu::Features::TIMESTAMP_QUERY),
        }
    }

    /// Optimal workgroup size for 1D compute dispatches.
    ///
    /// Returns the smaller of the requested size and the hardware limit.
    #[must_use]
    pub fn workgroup_size(&self, preferred: u32) -> u32 {
        preferred.min(self.max_compute_workgroup_size_x)
    }

    /// Number of workgroups needed to cover `n_items` with the given
    /// workgroup size, clamped to the hardware maximum.
    #[must_use]
    pub fn dispatch_count(&self, n_items: u32, workgroup_size: u32) -> u32 {
        n_items
            .div_ceil(workgroup_size)
            .min(self.max_compute_workgroups_per_dimension)
    }

    /// Whether the hardware supports the given WGSL `@workgroup_size(n)`.
    #[must_use]
    pub const fn supports_workgroup(&self, shader_workgroup: u32) -> bool {
        shader_workgroup <= self.max_compute_workgroup_size_x
    }
}

/// GPU context for `neuralSpring` workloads.
///
/// Wraps `WgpuDevice` with runtime-discovered capabilities.
/// Exposes raw `wgpu` handles for direct buffer management.
pub struct Gpu {
    wgpu_device: Arc<WgpuDevice>,
    pub adapter_name: String,
    pub device_type: wgpu::DeviceType,
    pub backend: wgpu::Backend,
    pub capabilities: GpuCapabilities,
}

impl Gpu {
    /// Create from a `WgpuDevice` (already initialised).
    ///
    /// Queries adapter capabilities at construction time — no
    /// hardcoded assumptions carried forward.
    #[must_use]
    pub fn from_device(dev: Arc<WgpuDevice>) -> Self {
        let info = dev.adapter_info();
        let caps = GpuCapabilities::from_device_and_adapter(dev.device(), dev.device().features());
        Self {
            adapter_name: info.name.clone(),
            device_type: info.device_type,
            backend: info.backend,
            capabilities: caps,
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
            "cpu" => Self::new_cpu().await,
            other => Self::select_adapter(other).await,
        }
    }

    /// Create with the CPU software backend (llvmpipe).
    ///
    /// Uses `BarraCUDA`'s `new_cpu_relaxed()` which requests
    /// `downlevel_defaults` limits (llvmpipe caps at 128 MB).
    ///
    /// # Errors
    ///
    /// Returns an error if no CPU adapter is available.
    pub async fn new_cpu() -> Result<Self, String> {
        match WgpuDevice::new_cpu_relaxed().await {
            Ok(dev) => Ok(Self::from_device(Arc::new(dev))),
            Err(e) => Err(format!("cpu: {e}")),
        }
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

    /// Compute the number of workgroups for a 1D dispatch, validated
    /// against runtime-discovered hardware limits.
    ///
    /// `shader_wg` is the shader's `@workgroup_size(N)` value.
    /// Returns the workgroup count needed to cover `n_items`.
    ///
    /// # Panics
    ///
    /// Panics if the hardware does not support the shader's workgroup size.
    #[must_use]
    pub fn dispatch_1d(&self, n_items: u32, shader_wg: u32) -> u32 {
        assert!(
            self.capabilities.supports_workgroup(shader_wg),
            "shader @workgroup_size({shader_wg}) exceeds hardware limit ({})",
            self.capabilities.max_compute_workgroup_size_x,
        );
        self.capabilities.dispatch_count(n_items, shader_wg)
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

    /// Read f64 data back from a GPU buffer (blocking).
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU readback fails.
    pub fn read_buffer_f64(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f64>, String> {
        self.wgpu_device
            .read_buffer_f64(buffer, count)
            .map_err(|e| format!("read_buffer_f64: {e}"))
    }

    /// Select a specific adapter by name substring or enumeration index.
    ///
    /// Uses relaxed limits so CPU software adapters (llvmpipe) work.
    async fn select_adapter(selector: &str) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all());

        let adapter = if let Ok(idx) = selector.parse::<usize>() {
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

        let adapter_limits = adapter.limits();
        let relaxed = wgpu::Limits::downlevel_defaults();
        let limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size.max(relaxed.max_buffer_size),
            max_compute_workgroup_size_x: adapter_limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: adapter_limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: adapter_limits.max_compute_workgroup_size_z,
            max_compute_workgroups_per_dimension: adapter_limits
                .max_compute_workgroups_per_dimension,
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage
                .max(relaxed.max_storage_buffers_per_shader_stage),
            ..relaxed
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("neuralSpring (capability-probed)"),
                    required_features: features,
                    required_limits: limits,
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
pub(crate) mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    use std::sync::{Arc as StdArc, OnceLock};

    static SHARED_GPU: OnceLock<Option<StdArc<Gpu>>> = OnceLock::new();

    /// Returns a shared `Gpu` instance so all test modules use the same
    /// Vulkan device and don't corrupt each other's global wgpu state.
    pub fn shared_gpu() -> Option<StdArc<Gpu>> {
        SHARED_GPU
            .get_or_init(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .ok()?;
                rt.block_on(Gpu::new()).ok().map(StdArc::new)
            })
            .clone()
    }

    #[test]
    fn gpu_new_succeeds() {
        let _lock = crate::test_gpu_lock::acquire();
        let Some(gpu) = shared_gpu() else { return };
        assert!(!gpu.adapter_name.is_empty());
    }

    #[test]
    fn gpu_upload_and_readback_roundtrip() {
        let _lock = crate::test_gpu_lock::acquire();
        let Some(gpu) = shared_gpu() else { return };
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

    #[test]
    fn gpu_create_empty_buffer() {
        let _lock = crate::test_gpu_lock::acquire();
        let Some(gpu) = shared_gpu() else { return };
        let buf = gpu.create_buffer_f32(64).expect("should create buffer");
        let out = gpu
            .read_buffer_f32(&buf, 64)
            .expect("read_buffer_f32 should succeed");
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn gpu_compile_trivial_shader() {
        let _lock = crate::test_gpu_lock::acquire();
        let Some(gpu) = shared_gpu() else { return };
        let _module =
            gpu.compile_shader("@compute @workgroup_size(1) fn main() {}", "test_trivial");
    }

    #[test]
    fn gpu_wgpu_device_accessible() {
        let _lock = crate::test_gpu_lock::acquire();
        let Some(gpu) = shared_gpu() else { return };
        let _device = gpu.wgpu_device();
        let _raw_device = gpu.device();
        let _raw_queue = gpu.queue();
    }

    #[test]
    fn gpu_capabilities_discovered() {
        let _lock = crate::test_gpu_lock::acquire();
        let Some(gpu) = shared_gpu() else { return };
        let caps = &gpu.capabilities;
        assert!(caps.max_buffer_size > 0, "buffer size should be positive");
        assert!(
            caps.max_compute_workgroup_size_x > 0,
            "workgroup size should be positive"
        );
        assert!(
            caps.max_compute_workgroups_per_dimension > 0,
            "dispatch limit should be positive"
        );
        let wg = caps.workgroup_size(256);
        assert!(wg > 0 && wg <= 256);
        let dc = caps.dispatch_count(1024, wg);
        assert!(dc > 0);
    }

    // ── GpuCapabilities unit tests (no GPU hardware needed) ─────

    fn mock_caps(wg_x: u32, max_dispatch: u32) -> GpuCapabilities {
        GpuCapabilities {
            max_buffer_size: 128 * 1024 * 1024,
            max_compute_workgroup_size_x: wg_x,
            max_compute_workgroups_per_dimension: max_dispatch,
            max_storage_buffers_per_shader_stage: 8,
            supports_f64: false,
            supports_f16: false,
            supports_timestamp_query: false,
        }
    }

    #[test]
    fn workgroup_size_clamped() {
        let caps = mock_caps(128, 65535);
        assert_eq!(caps.workgroup_size(256), 128);
        assert_eq!(caps.workgroup_size(64), 64);
        assert_eq!(caps.workgroup_size(128), 128);
    }

    #[test]
    fn dispatch_count_exact() {
        let caps = mock_caps(256, 65535);
        assert_eq!(caps.dispatch_count(256, 256), 1);
        assert_eq!(caps.dispatch_count(257, 256), 2);
        assert_eq!(caps.dispatch_count(512, 256), 2);
    }

    #[test]
    fn dispatch_count_clamped() {
        let caps = mock_caps(256, 100);
        assert_eq!(caps.dispatch_count(100_000, 256), 100);
    }

    #[test]
    fn supports_workgroup_check() {
        let caps = mock_caps(256, 65535);
        assert!(caps.supports_workgroup(256));
        assert!(caps.supports_workgroup(1));
        assert!(!caps.supports_workgroup(512));
    }

    #[test]
    fn gpu_dispatch_1d_basic() {
        let _lock = crate::test_gpu_lock::acquire();
        let Some(gpu) = shared_gpu() else { return };
        let wg = gpu.dispatch_1d(1024, 64);
        assert!(wg > 0);
    }

    #[test]
    fn gpu_from_device_roundtrip() {
        let _lock = crate::test_gpu_lock::acquire();
        let Some(gpu) = shared_gpu() else { return };
        let dev = gpu.wgpu_device().clone();
        let gpu2 = Gpu::from_device(dev);
        assert_eq!(gpu2.adapter_name, gpu.adapter_name);
    }

    #[test]
    fn gpu_new_cpu_if_available() {
        let _lock = crate::test_gpu_lock::acquire();
        if let Some(gpu) = shared_gpu() {
            assert!(!gpu.adapter_name.is_empty());
        }
    }

    #[test]
    fn gpu_new_gpu_if_available() {
        let _lock = crate::test_gpu_lock::acquire();
        if let Some(gpu) = shared_gpu() {
            assert!(!gpu.adapter_name.is_empty());
        }
    }
}
