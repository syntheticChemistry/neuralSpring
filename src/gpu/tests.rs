// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(
    clippy::expect_used,
    reason = "test infrastructure — GPU init is fatal"
)]

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
            f64::from((got - want).abs()) < crate::tolerances::TENSOR_EXACT_F32,
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
    let _module = gpu.compile_shader("@compute @workgroup_size(1) fn main() {}", "test_trivial");
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

#[test]
fn gpu_read_buffer_f64_roundtrip() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(gpu) = shared_gpu() else { return };
    let data: [f64; 3] = [1.0, 2.5, -3.125];
    let buf = gpu
        .wgpu_device()
        .create_buffer_f64(3)
        .expect("create_buffer_f64");
    gpu.queue()
        .write_buffer(&buf, 0, bytemuck::cast_slice(&data));
    let out = gpu
        .read_buffer_f64(&buf, 3)
        .expect("read_buffer_f64 should succeed");
    for (i, (&got, &want)) in out.iter().zip(data.iter()).enumerate() {
        assert!(
            (got - want).abs() < crate::tolerances::ZERO_DETECTION,
            "f64 roundtrip mismatch at {i}: got {got}, want {want}"
        );
    }
}

#[test]
fn gpu_capabilities_clone() {
    let caps = mock_caps(256, 65535);
    #[expect(
        clippy::redundant_clone,
        reason = "verifying Clone impl works correctly"
    )]
    let caps2 = caps.clone();
    assert_eq!(caps2.max_compute_workgroup_size_x, 256);
    assert_eq!(caps2.max_compute_workgroups_per_dimension, 65535);
    assert!(!caps2.supports_f64);
}

#[test]
fn gpu_capabilities_debug() {
    let caps = mock_caps(128, 1024);
    let debug = format!("{caps:?}");
    assert!(debug.contains("128"));
    assert!(debug.contains("1024"));
}

#[test]
fn gpu_dispatch_1d_clamped() {
    let _lock = crate::test_gpu_lock::acquire();
    let Some(gpu) = shared_gpu() else { return };
    let wg = gpu.dispatch_1d(1_000_000, 64);
    assert!(wg <= gpu.capabilities.max_compute_workgroups_per_dimension);
}

#[test]
fn gpu_new_cpu_explicit() {
    let _lock = crate::test_gpu_lock::acquire();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio rt");
    if let Ok(gpu) = rt.block_on(Gpu::new_cpu()) {
        assert!(!gpu.adapter_name.is_empty());
    }
}
