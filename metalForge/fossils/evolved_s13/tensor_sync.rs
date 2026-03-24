// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU synchronization primitives for safe sequential tensor operations.
//!
//! ## Problem (S-13: `PooledBuffer` drop-before-completion race)
//!
//! `BarraCUDA`'s `BufferPool` returns buffers to the pool in
//! `PooledBuffer::drop` **without waiting for the GPU to finish using them**.
//! When sequential operations (e.g. chained matmuls) produce intermediate
//! tensors that are dropped before readback, the next `acquire_pooled` can
//! reuse a buffer that the GPU is still writing to — causing data corruption
//! or driver hangs.
//!
//! This affects any sequence of GPU tensor operations where intermediate
//! results are not explicitly read back (`to_vec()`), particularly:
//! - Sequential square-matrix `Tensor::matmul` (same bucket size → high reuse)
//! - MLP forward passes (chained matmul → add → activation)
//! - MHA projections (4× square matmul for Q/K/V/O)
//!
//! ## Solution
//!
//! [`gpu_fence`] inserts an explicit GPU synchronization barrier by calling
//! `device.poll(Maintain::Wait)`, ensuring all submitted GPU work completes
//! before the next operation can acquire a pooled buffer.
//!
//! [`materialize`] reads a tensor's data back from the GPU and recreates it,
//! forcing a full sync and preventing the dropped tensor's buffer from
//! racing with subsequent operations.
//!
//! [`fenced_matmul`] wraps a single `Tensor::matmul` with an automatic fence.
//!
//! ## `ToadStool` absorption path
//!
//! The proper fix is in `PooledBuffer::drop` — add `device.poll(Wait)` before
//! returning the buffer to the pool, or track in-flight submissions and only
//! recycle buffers after their associated work completes. This module provides
//! the correctness proof; `ToadStool` should absorb the sync into the pool
//! itself.

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use std::sync::Arc;

/// Force all submitted GPU work to complete before returning.
///
/// Insert between sequential GPU tensor operations to prevent the
/// `PooledBuffer` drop-before-completion race. This is the minimal
/// barrier needed — it ensures any buffer returned to the pool after
/// this call is safe for reuse.
///
/// # Example
///
/// ```ignore
/// let ab = a.matmul(&b)?;
/// gpu_fence(&device);  // Ensure matmul completes before next op
/// let ba = b2.matmul(&a2)?;
/// ```
pub fn gpu_fence(device: &Arc<WgpuDevice>) {
    device.device().poll(wgpu::Maintain::Wait);
}

/// Read tensor data from GPU, recreate the tensor, forcing full sync.
///
/// This ensures the GPU has finished writing to the tensor's buffer
/// before that buffer can be returned to the pool. The cost is one
/// readback + one upload per call — use [`gpu_fence`] instead when you
/// don't need the data materialized.
///
/// # Errors
///
/// Returns an error if the GPU readback or re-upload fails.
pub fn materialize(t: &Tensor, device: &Arc<WgpuDevice>) -> Result<Tensor, String> {
    let data = t
        .to_vec()
        .map_err(|e| format!("materialize readback: {e}"))?;
    let shape = t.shape().to_vec();
    Tensor::from_data(&data, shape, device.clone()).map_err(|e| format!("materialize upload: {e}"))
}

/// Perform a single matmul with a GPU fence after submission.
///
/// Wraps `lhs.matmul(rhs)` with an automatic [`gpu_fence`] to ensure
/// the GPU completes before the caller's next operation can acquire
/// a pooled buffer. Use this for any sequential matmul sequence.
///
/// # Errors
///
/// Returns an error if the matmul fails.
pub fn fenced_matmul(
    lhs: Tensor,
    rhs: &Tensor,
    device: &Arc<WgpuDevice>,
) -> Result<Tensor, String> {
    let result = lhs.matmul(rhs).map_err(|e| format!("fenced_matmul: {e}"))?;
    gpu_fence(device);
    Ok(result)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "fossil test code — GPU setup may fail")]

    use super::*;

    #[tokio::test]
    async fn gpu_fence_completes() {
        let dev = match WgpuDevice::new().await {
            Ok(d) => Arc::new(d),
            Err(_) => return,
        };
        gpu_fence(&dev);
    }

    #[tokio::test]
    async fn materialize_roundtrip() {
        let dev = match WgpuDevice::new().await {
            Ok(d) => Arc::new(d),
            Err(_) => return,
        };
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_data(&data, vec![2, 2], dev.clone()).expect("from_data");
        let t2 = materialize(&t, &dev).expect("materialize");
        let out = t2.to_vec().expect("to_vec");
        assert_eq!(out.len(), 4);
        for (i, (&got, &want)) in out.iter().zip(data.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-7,
                "mismatch at {i}: {got} vs {want}"
            );
        }
    }

    #[tokio::test]
    async fn fenced_matmul_basic() {
        let dev = match WgpuDevice::new().await {
            Ok(d) => Arc::new(d),
            Err(_) => return,
        };
        let a = Tensor::from_data(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], dev.clone())
            .expect("a");
        let b = Tensor::from_data(
            &[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 2],
            dev.clone(),
        )
        .expect("b");
        let result = fenced_matmul(a, &b, &dev).expect("fenced_matmul");
        let out = result.to_vec().expect("to_vec");
        assert!((f64::from(out[0]) - 58.0).abs() < 0.01);
        assert!((f64::from(out[3]) - 154.0).abs() < 0.01);
    }

    // Run alone (--test-threads=1) — concurrent wgpu Instance sharing
    // causes BindGroupLayout invalidation across parallel tests.
    #[tokio::test]
    #[ignore = "wgpu Instance sharing across parallel tests invalidates BindGroupLayouts — run with --test-threads=1"]
    async fn sequential_square_matmul_with_fence() {
        let dev = match WgpuDevice::new().await {
            Ok(d) => Arc::new(d),
            Err(_) => return,
        };

        // Two sequential 4×4 matmuls — would hang without fences
        let a = Tensor::from_data(&[1.0_f32, 2.0, 3.0, 4.0], vec![2, 2], dev.clone()).expect("a");
        let b = Tensor::from_data(&[5.0_f32, 6.0, 7.0, 8.0], vec![2, 2], dev.clone()).expect("b");
        let r1 = fenced_matmul(a, &b, &dev).expect("matmul1");
        let v1 = r1.to_vec().expect("readback1");
        assert!((f64::from(v1[0]) - 19.0).abs() < 0.01, "1*5+2*7=19");

        let c =
            Tensor::from_data(&[9.0_f32, 10.0, 11.0, 12.0], vec![2, 2], dev.clone()).expect("c");
        let d =
            Tensor::from_data(&[13.0_f32, 14.0, 15.0, 16.0], vec![2, 2], dev.clone()).expect("d");
        let r2 = fenced_matmul(c, &d, &dev).expect("matmul2");
        let v2 = r2.to_vec().expect("readback2");
        assert!((f64::from(v2[0]) - 267.0).abs() < 0.01, "9*13+10*15=267");
    }
}
