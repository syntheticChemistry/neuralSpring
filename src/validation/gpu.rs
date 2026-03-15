// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU tensor validation helpers (shared across 24+ validation binaries).

use super::{ReductionExpected, ValidationHarness};
use barracuda::device::WgpuDevice;
use barracuda::error::BarracudaError;
use barracuda::tensor::Tensor;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Create a GPU storage buffer initialized from a byte slice.
///
/// Shared across GPU validation binaries to replace the duplicated
/// `storage_buf` helper found in 14+ binaries.
#[must_use]
pub fn storage_buf(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

/// Create a GPU output buffer (storage + copy-src + copy-dst) of the given byte size.
///
/// Shared across GPU validation binaries to replace the duplicated
/// `output_buf` helper found in 14+ binaries.
#[must_use]
pub fn output_buf(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Attempt GPU tensor readback, recording a FAIL check on error.
///
/// Returns `Some(data)` on success, `None` on failure.
/// The caller should early-return on `None`.
pub fn gpu_readback(h: &mut ValidationHarness, tensor: &Tensor) -> Option<Vec<f32>> {
    match tensor.to_vec() {
        Ok(data) => Some(data),
        Err(e) => {
            h.check_bool(&format!("GPU readback: {e}"), false);
            None
        }
    }
}

/// Compute the maximum absolute difference between two f64 slices.
///
/// Used by cross-spring and evolution validators to compare CPU results
/// at full precision.
#[must_use]
pub fn max_abs_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Compute the maximum absolute difference between two f32 slices.
///
/// Used by GPU validation binaries to compare GPU output against
/// GPU output (both f32).
#[must_use]
pub fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0.0_f64, f64::max)
}

/// Compute the maximum absolute difference between GPU f32 output
/// and CPU f64 reference values.
///
/// The promotion to f64 before comparison avoids f32 rounding
/// masking real errors.
#[must_use]
pub fn max_abs_diff_gpu_vs_cpu(gpu: &[f32], cpu: &[f64]) -> f64 {
    gpu.iter()
        .zip(cpu.iter())
        .map(|(&g, &c)| (f64::from(g) - c).abs())
        .fold(0.0_f64, f64::max)
}

/// Batch-check readback values against expected (label, index, expected, tolerance).
///
/// Shared helper that replaces `check_points` duplicated across tensor
/// validation binaries.
pub fn check_gpu_points(
    h: &mut ValidationHarness,
    data: &[f32],
    checks: &[(&str, usize, f64, f64)],
) {
    for &(label, idx, expected, tol) in checks {
        h.check_abs(label, f64::from(data[idx]), expected, tol);
    }
}

/// Create a tensor from data, recording a FAIL check on error.
///
/// Returns `Some(tensor)` on success, `None` on failure.
pub fn gpu_tensor(
    h: &mut ValidationHarness,
    data: &[f32],
    shape: &[usize],
    device: &Arc<WgpuDevice>,
) -> Option<Tensor> {
    match Tensor::from_data(data, shape.to_vec(), device.clone()) {
        Ok(t) => Some(t),
        Err(e) => {
            h.check_bool(&format!("tensor create: {e}"), false);
            None
        }
    }
}

/// Validate a unary tensor operation against expected point checks.
///
/// Handles the common pattern:
///   create tensor → apply op → readback → `check_gpu_points`
/// with graceful error recording on failure at any step.
pub fn validate_tensor_unary(
    h: &mut ValidationHarness,
    device: &Arc<WgpuDevice>,
    data: &[f32],
    shape: &[usize],
    op: impl FnOnce(&Tensor) -> Result<Tensor, BarracudaError>,
    op_name: &str,
    checks: &[(&str, usize, f64, f64)],
) {
    let Some(input) = gpu_tensor(h, data, shape, device) else {
        return;
    };
    match op(&input) {
        Ok(out) => {
            let Some(v) = gpu_readback(h, &out) else {
                return;
            };
            check_gpu_points(h, &v, checks);
        }
        Err(e) => h.check_bool(&format!("{op_name} [ERROR: {e}]"), false),
    }
}

/// Validate a scalar reduction (sum, mean, max, etc.) against an expected value.
///
/// Handles: create tensor → apply reduction op → readback scalar → `check_abs`.
pub fn validate_tensor_reduction(
    h: &mut ValidationHarness,
    device: &Arc<WgpuDevice>,
    data: &[f32],
    shape: &[usize],
    op: impl FnOnce(&Tensor) -> Result<Tensor, BarracudaError>,
    expected: &ReductionExpected<'_>,
) {
    let Some(input) = gpu_tensor(h, data, shape, device) else {
        return;
    };
    match op(&input) {
        Ok(out) => {
            let Some(v) = gpu_readback(h, &out) else {
                return;
            };
            h.check_abs(
                expected.label,
                f64::from(v[0]),
                expected.value,
                expected.tolerance,
            );
        }
        Err(e) => h.check_bool(&format!("{} [ERROR: {e}]", expected.label), false),
    }
}

/// Inputs for a binary tensor validation.
pub struct BinaryTensorInputs<'a> {
    pub data_a: &'a [f32],
    pub data_b: &'a [f32],
    pub shape: &'a [usize],
}

/// Validate a binary tensor operation against expected point checks.
///
/// Handles the common pattern:
///   create `input_a` → create `input_b` → apply op → readback → `check_gpu_points`
/// with graceful error recording on failure at any step.
pub fn validate_tensor_binary(
    h: &mut ValidationHarness,
    device: &Arc<WgpuDevice>,
    inputs: &BinaryTensorInputs<'_>,
    op: impl FnOnce(&Tensor, &Tensor) -> Result<Tensor, BarracudaError>,
    op_name: &str,
    checks: &[(&str, usize, f64, f64)],
) {
    let Some(a) = gpu_tensor(h, inputs.data_a, inputs.shape, device) else {
        return;
    };
    let Some(b) = gpu_tensor(h, inputs.data_b, inputs.shape, device) else {
        return;
    };
    match op(&a, &b) {
        Ok(out) => {
            let Some(v) = gpu_readback(h, &out) else {
                return;
            };
            check_gpu_points(h, &v, checks);
        }
        Err(e) => h.check_bool(&format!("{op_name} [ERROR: {e}]"), false),
    }
}

/// Generate a deterministic f64 test vector of given length.
///
/// Shared across cross-spring and evolution validators to replace
/// duplicated `gen_f64_vec` helpers.
#[must_use]
pub fn gen_test_f64(rng: &mut crate::rng::Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.next_f64()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_abs_diff_f64_exact() {
        assert!((max_abs_diff_f64(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn max_abs_diff_f64_nonzero() {
        let diff = max_abs_diff_f64(&[1.0, 5.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((diff - 3.0).abs() < 1e-15);
    }

    #[test]
    fn max_abs_diff_f64_empty() {
        assert!((max_abs_diff_f64(&[], &[]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn max_abs_diff_f32_exact() {
        assert!((max_abs_diff_f32(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn max_abs_diff_f32_nonzero() {
        let diff = max_abs_diff_f32(&[1.0_f32, 5.0], &[1.0, 2.0]);
        assert!((diff - 3.0).abs() < 1e-6);
    }

    #[test]
    fn max_abs_diff_gpu_vs_cpu_promotion() {
        let gpu = vec![1.0_f32, 2.0];
        let cpu = vec![1.0_f64, 2.001];
        let diff = max_abs_diff_gpu_vs_cpu(&gpu, &cpu);
        assert!((diff - 0.001).abs() < 1e-5);
    }

    #[test]
    fn check_gpu_points_pass_and_fail() {
        let mut h = ValidationHarness::new("test");
        let data = vec![1.0_f32, 2.5, 3.0];
        let checks: Vec<(&str, usize, f64, f64)> = vec![
            ("val0", 0, 1.0, 0.1),
            ("val1", 1, 2.5, 0.01),
            ("val2", 2, 999.0, 0.01),
        ];
        check_gpu_points(&mut h, &data, &checks);
        assert_eq!(h.passed_count(), 2);
        assert_eq!(h.total_count(), 3);
    }

    #[test]
    fn max_abs_diff_f32_empty() {
        let diff = max_abs_diff_f32(&[], &[]);
        assert!((diff - 0.0).abs() < 1e-15, "empty → 0");
    }

    #[test]
    fn max_abs_diff_gpu_vs_cpu_empty() {
        let diff = max_abs_diff_gpu_vs_cpu(&[], &[]);
        assert!((diff - 0.0).abs() < 1e-15, "empty → 0");
    }

    #[test]
    fn max_abs_diff_gpu_vs_cpu_precision() {
        let gpu = vec![0.1_f32];
        let cpu = vec![0.1_f64];
        let diff = max_abs_diff_gpu_vs_cpu(&gpu, &cpu);
        assert!(
            diff < 1e-6,
            "f32 0.1 should be close to f64 0.1, got {diff}"
        );
    }
}
