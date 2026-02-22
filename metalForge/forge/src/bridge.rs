// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bridge between neuralSpring GPU context and `barracuda::device::WgpuDevice`.
//!
//! This formalizes the absorption seam: neuralSpring's `Gpu` struct wraps a
//! `barracuda::device::WgpuDevice` and exposes both raw `wgpu` handles and
//! the `BarraCUDA` Tensor API.
//!
//! `ToadStool` can absorb this bridge pattern into `barracuda::device` to
//! provide a unified device context for all Springs.
//!
//! ## Usage pattern (from neuralSpring `gpu.rs`)
//!
//! ```text
//! let gpu = Gpu::new().await?;          // neuralSpring wrapper
//! let dev = gpu.wgpu_device();          // -> Arc<WgpuDevice> for Tensor API
//! let device = gpu.device();            // -> &wgpu::Device for raw buffer ops
//! let queue = gpu.queue();              // -> &wgpu::Queue for command submission
//! let module = gpu.compile_shader(src); // -> wgpu::ShaderModule via BarraCUDA
//! ```

use barracuda::device::WgpuDevice;
use std::sync::Arc;

/// Backend selection for the neuralSpring GPU context.
///
/// Maps to the `NEURALSPRING_BACKEND` environment variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Best available adapter (`HighPerformance`).
    Auto,
    /// Force CPU software rasterizer (llvmpipe).
    Cpu,
    /// Force discrete/integrated GPU.
    Gpu,
}

/// Create a `WgpuDevice` using the specified backend.
///
/// This is the bridge function: neuralSpring and validation binaries call
/// this to get a device, then use `WgpuDevice` for both raw wgpu ops
/// and the Tensor API.
///
/// # Errors
///
/// Returns an error string if the requested backend is unavailable.
pub async fn create_device(backend: Backend) -> Result<Arc<WgpuDevice>, String> {
    match backend {
        Backend::Auto => WgpuDevice::new()
            .await
            .map(Arc::new)
            .map_err(|e| format!("auto: {e}")),
        Backend::Cpu => WgpuDevice::new_cpu_relaxed()
            .await
            .map(Arc::new)
            .map_err(|e| format!("cpu: {e}")),
        Backend::Gpu => WgpuDevice::new_gpu()
            .await
            .map(Arc::new)
            .map_err(|e| format!("gpu: {e}")),
    }
}

/// Parse a backend selector string (from environment variable).
///
/// Recognized values: `"auto"`, `""`, `"cpu"`, `"gpu"`.
/// Any other value returns `None` (caller should try adapter matching).
#[must_use]
pub fn parse_backend(selector: &str) -> Option<Backend> {
    match selector.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Some(Backend::Auto),
        "cpu" => Some(Backend::Cpu),
        "gpu" => Some(Backend::Gpu),
        _ => None,
    }
}

/// Upload f32 data to a GPU storage buffer via `WgpuDevice`.
///
/// # Errors
///
/// Returns an error if buffer allocation fails.
pub fn upload_f32(device: &WgpuDevice, data: &[f32]) -> Result<wgpu::Buffer, String> {
    let buf = device
        .create_buffer_f32(data.len())
        .map_err(|e| format!("create_buffer_f32: {e}"))?;
    device
        .queue()
        .write_buffer(&buf, 0, bytemuck::cast_slice(data));
    Ok(buf)
}

/// Read f32 data back from a GPU buffer (blocking).
///
/// # Errors
///
/// Returns an error if GPU readback fails.
pub fn read_buffer_f32(
    device: &WgpuDevice,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<f32>, String> {
    device
        .read_buffer_f32(buffer, count)
        .map_err(|e| format!("read_buffer_f32: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_known_backends() {
        assert_eq!(parse_backend("auto"), Some(Backend::Auto));
        assert_eq!(parse_backend(""), Some(Backend::Auto));
        assert_eq!(parse_backend("cpu"), Some(Backend::Cpu));
        assert_eq!(parse_backend("gpu"), Some(Backend::Gpu));
        assert_eq!(parse_backend("CPU"), Some(Backend::Cpu));
        assert_eq!(parse_backend("  GPU  "), Some(Backend::Gpu));
    }

    #[test]
    fn parse_unknown_returns_none() {
        assert_eq!(parse_backend("nvidia"), None);
        assert_eq!(parse_backend("0"), None);
    }

    #[tokio::test]
    async fn create_device_auto() {
        let result = create_device(Backend::Auto).await;
        if let Ok(dev) = result {
            assert!(!dev.adapter_info().name.is_empty());
        }
    }
}
