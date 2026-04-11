// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU compute wrapper for `neuralSpring` validation and benchmarking.
//!
//! Thin config layer over `barracuda::device::WgpuDevice` — imitates the
//! hotSpring `GpuF64` pattern.  No abstraction: exposes raw `wgpu::Device`
//! and `wgpu::Queue` so evolved ops can manage buffers directly.
//!
//! ## Backend selection
//!
//! Set `GPU_BACKEND` to control the adapter (legacy `NEURALSPRING_BACKEND`
//! is accepted as fallback):
//!
//! | Value | Behaviour |
//! |-------|-----------|
//! | `auto` (default) | Best available (`HighPerformance`) |
//! | `cpu` | Force CPU software rasterizer (llvmpipe) |
//! | `gpu` | Force discrete / integrated GPU |
//! | `list` | Print all adapters and exit |
//! | name/index | Adapter name substring or enumeration index |

use barracuda::device::WgpuDevice;
pub use barracuda::shaders::precision::Precision;
use bytemuck;
use std::sync::Arc;

use crate::error::GpuError;

/// GPU context for `neuralSpring` workloads.
///
/// Wraps `WgpuDevice` with relaxed limits (llvmpipe caps at 128 MB)
/// and exposes raw `wgpu` handles for direct buffer management.
///
/// # Example
///
/// ```no_run
/// use neural_spring::error::GpuError;
/// use neural_spring::gpu::Gpu;
///
/// # async fn example() -> Result<(), GpuError> {
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
    /// Maximum storage buffer size in bytes supported by the device.
    pub max_buffer_size: u64,
    /// Maximum X dimension of a compute workgroup.
    pub max_compute_workgroup_size_x: u32,
    /// Maximum number of workgroups along one dispatch dimension.
    pub max_compute_workgroups_per_dimension: u32,
    /// Maximum storage buffers bound per shader stage.
    pub max_storage_buffers_per_shader_stage: u32,
    /// Whether `SHADER_F64` is available.
    pub supports_f64: bool,
    /// Whether `SHADER_F16` is available.
    pub supports_f16: bool,
    /// Whether `TIMESTAMP_QUERY` is available.
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
    /// Human-readable adapter name from `wgpu`.
    pub adapter_name: String,
    /// Discrete vs integrated vs CPU adapter classification.
    pub device_type: wgpu::DeviceType,
    /// Graphics API backend in use (Vulkan, Metal, etc.).
    pub backend: wgpu::Backend,
    /// Runtime limits and feature flags discovered for this device.
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

    /// Create with the default backend (`GPU_BACKEND` env var).
    ///
    /// Falls back to `NEURALSPRING_BACKEND` for backward compatibility.
    /// Uses relaxed limits so CPU software adapters (llvmpipe) work.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested backend is unavailable.
    pub async fn new() -> Result<Self, GpuError> {
        let selector = std::env::var(crate::config::ENV_GPU_BACKEND_LEGACY)
            .or_else(|_| std::env::var(crate::config::ENV_GPU_BACKEND))
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();

        if selector == "list" {
            let adapters = WgpuDevice::enumerate_adapters().await;
            for (i, info) in adapters.iter().enumerate() {
                log::info!(
                    "[{i}] {name} ({ty:?}, {backend:?})",
                    name = info.name,
                    ty = info.device_type,
                    backend = info.backend,
                );
            }
            return Err(GpuError::Device {
                reason: "GPU_BACKEND=list: adapter enumeration complete".into(),
            });
        }

        match selector.as_str() {
            "gpu" => match WgpuDevice::new_gpu().await {
                Ok(dev) => Ok(Self::from_device(Arc::new(dev))),
                Err(e) => Err(GpuError::Device {
                    reason: format!("gpu: {e}"),
                }),
            },
            "" | "auto" => match WgpuDevice::new().await {
                Ok(dev) => Ok(Self::from_device(Arc::new(dev))),
                Err(e) => Err(GpuError::Device {
                    reason: format!("auto: {e}"),
                }),
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
    pub async fn new_cpu() -> Result<Self, GpuError> {
        match WgpuDevice::new_cpu_relaxed().await {
            Ok(dev) => Ok(Self::from_device(Arc::new(dev))),
            Err(e) => Err(GpuError::Device {
                reason: format!("cpu: {e}"),
            }),
        }
    }

    /// Create with a discrete/integrated GPU backend.
    ///
    /// # Errors
    ///
    /// Returns an error if no GPU adapter is available.
    pub async fn new_gpu() -> Result<Self, GpuError> {
        match WgpuDevice::new_gpu().await {
            Ok(dev) => Ok(Self::from_device(Arc::new(dev))),
            Err(e) => Err(GpuError::Device {
                reason: format!("gpu: {e}"),
            }),
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

    /// Compile a hybrid f64/df64 shader via the core streaming path.
    ///
    /// Delegates to `WgpuDevice::compile_shader_df64` which prepends
    /// `df64_core.wgsl` + `df64_transcendentals.wgsl`, runs ILP optimizer
    /// and Sovereign compiler when available. This is the hotSpring/`ToadStool`
    /// three-zone pattern: f64 buffer I/O with df64 compute on FP32 cores.
    #[must_use]
    pub fn compile_shader_f64_hybrid(&self, source: &str, label: &str) -> wgpu::ShaderModule {
        self.wgpu_device.compile_shader_df64(source, Some(label))
    }

    /// Compile a shader at any precision via barraCuda's per-precision methods.
    ///
    /// Routes one f64-canonical shader source through the appropriate
    /// compilation pipeline based on the precision arg:
    ///
    /// - Sub-f32 quantized (Binary..Bf16): `compile_shader` — arithmetic
    ///   runs in f32; quantization is a data-format concern, not a shader
    ///   compilation concern.
    /// - F16/F32: `compile_shader` (auto-downcasts f64 sources)
    /// - F64: `compile_shader_f64` (driver-profile–aware polyfills)
    /// - Df64: `compile_shader_df64` (injects `df64_core` + transcendentals)
    /// - Qf128: `compile_shader_df64` — quad-float builds on f32-pair infra
    /// - Df128: `compile_shader_f64` — double-double on native f64
    #[must_use]
    pub fn compile_shader_universal(
        &self,
        source: &str,
        precision: barracuda::shaders::precision::Precision,
    ) -> wgpu::ShaderModule {
        use barracuda::shaders::precision::Precision;
        match precision {
            Precision::Binary
            | Precision::Int2
            | Precision::Q4
            | Precision::Q8
            | Precision::Fp8E5M2
            | Precision::Fp8E4M3
            | Precision::Bf16
            | Precision::F16
            | Precision::F32 => self.wgpu_device.compile_shader(source, None),
            Precision::Df64 | Precision::Qf128 => {
                self.wgpu_device.compile_shader_df64(source, None)
            }
            Precision::F64 | Precision::Df128 => self.wgpu_device.compile_shader_f64(source, None),
        }
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
    pub fn create_buffer_f32(&self, count: usize) -> Result<wgpu::Buffer, GpuError> {
        self.wgpu_device
            .create_buffer_f32(count)
            .map_err(|e| GpuError::Buffer {
                op: "create_buffer_f32",
                reason: e.to_string(),
            })
    }

    /// Upload f32 data to a new GPU storage buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer allocation fails.
    pub fn upload_f32(&self, data: &[f32]) -> Result<wgpu::Buffer, GpuError> {
        let buf = self.create_buffer_f32(data.len())?;
        self.wgpu_device
            .queue()
            .write_buffer(&buf, 0, bytemuck::cast_slice(data));
        Ok(buf)
    }

    /// Upload f64 data to a new GPU storage buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer allocation fails.
    pub fn upload_f64(&self, data: &[f64]) -> Result<wgpu::Buffer, GpuError> {
        let buf = self.create_buffer_f64(data.len())?;
        self.wgpu_device
            .queue()
            .write_buffer(&buf, 0, bytemuck::cast_slice(data));
        Ok(buf)
    }

    /// Allocate a GPU storage buffer for `count` f64 values.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU allocation fails.
    pub fn create_buffer_f64(&self, count: usize) -> Result<wgpu::Buffer, GpuError> {
        self.wgpu_device
            .create_buffer_f64(count)
            .map_err(|e| GpuError::Buffer {
                op: "create_buffer_f64",
                reason: e.to_string(),
            })
    }

    /// Read f32 data back from a GPU buffer (blocking).
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU readback fails.
    pub fn read_buffer_f32(
        &self,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.wgpu_device
            .read_buffer_f32(buffer, count)
            .map_err(|e| GpuError::Buffer {
                op: "read_buffer_f32",
                reason: e.to_string(),
            })
    }

    /// Read f64 data back from a GPU buffer (blocking).
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU readback fails.
    pub fn read_buffer_f64(
        &self,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f64>, GpuError> {
        self.wgpu_device
            .read_buffer_f64(buffer, count)
            .map_err(|e| GpuError::Buffer {
                op: "read_buffer_f64",
                reason: e.to_string(),
            })
    }

    /// Read u32 data back from a GPU buffer (blocking).
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU readback fails.
    pub fn read_buffer_u32(
        &self,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<u32>, GpuError> {
        self.wgpu_device
            .read_buffer_u32(buffer, count)
            .map_err(|e| GpuError::Buffer {
                op: "read_buffer_u32",
                reason: e.to_string(),
            })
    }

    /// Select a specific adapter by name substring or enumeration index.
    ///
    /// Uses relaxed limits so CPU software adapters (llvmpipe) work.
    async fn select_adapter(selector: &str) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

        let adapter = if let Ok(idx) = selector.parse::<usize>() {
            adapters.into_iter().nth(idx)
        } else {
            let sel = selector.to_ascii_lowercase();
            adapters
                .into_iter()
                .find(|a| a.get_info().name.to_ascii_lowercase().contains(&sel))
        }
        .ok_or_else(|| GpuError::Device {
            reason: format!("no adapter matches '{selector}'"),
        })?;

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
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("neuralSpring (capability-probed)"),
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::default(),
            })
            .await
            .map_err(|e| GpuError::Device {
                reason: format!("device creation: {e}"),
            })?;

        let dev = WgpuDevice::from_existing(device, queue, info);
        Ok(Self::from_device(Arc::new(dev)))
    }
}

#[cfg(test)]
pub(crate) mod tests;
