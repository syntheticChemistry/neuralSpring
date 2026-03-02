// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU Hill activation operations.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use std::sync::Arc;

/// Hill gate kinetic parameters for the two-input AND gate model.
#[derive(Debug, Clone, Copy)]
pub struct HillGateConfig {
    pub vmax: f64,
    pub k_a: f64,
    pub k_b: f64,
    pub n_a: f64,
    pub n_b: f64,
}

/// GPU batch Hill activation: `V_max * x^n / (K^n + x^n)`.
///
/// Genuinely GPU-computed via Tensor log → scale → exp → div pipeline.
/// Replaces `primitives::hill_activation` for batch processing.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn hill_activation_batch_gpu(
    x: &[f64],
    vmax: f64,
    k: f64,
    n_hill: f64,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let len = x.len();
    if len == 0 {
        return Ok(Vec::new());
    }

    let kn = (k.powf(n_hill)) as f32;
    let n_f32 = n_hill as f32;
    let vmax_f32 = vmax as f32;
    let guard = crate::primitives::HILL_EPS as f32;

    let x_f32: Vec<f32> = x
        .iter()
        .map(|&v| (v.max(crate::primitives::LOG_GUARD)) as f32)
        .collect();

    let x_t =
        Tensor::from_data(&x_f32, vec![len], device.clone()).map_err(|e| format!("hill x: {e}"))?;
    let log_x = x_t.log_wgsl().map_err(|e| format!("hill log: {e}"))?;
    let scaled_log = log_x
        .mul_scalar(n_f32)
        .map_err(|e| format!("hill scale: {e}"))?;
    let x_pow_n = scaled_log
        .exp_wgsl()
        .map_err(|e| format!("hill exp: {e}"))?;

    let kn_t = Tensor::from_data(&vec![kn; len], vec![len], device.clone())
        .map_err(|e| format!("hill kn: {e}"))?;
    let eps_t = Tensor::from_data(&vec![guard; len], vec![len], device.clone())
        .map_err(|e| format!("hill eps: {e}"))?;
    let sum1 = x_pow_n.add(&kn_t).map_err(|e| format!("hill add1: {e}"))?;
    let denom = sum1.add(&eps_t).map_err(|e| format!("hill add2: {e}"))?;

    let ratio = x_pow_n.div(&denom).map_err(|e| format!("hill div: {e}"))?;
    let result = ratio
        .mul_scalar(vmax_f32)
        .map_err(|e| format!("hill vmax: {e}"))?;

    let out = result.to_vec().map_err(|e| format!("hill read: {e}"))?;
    Ok(out.into_iter().map(f64::from).collect())
}

/// GPU two-input Hill gate: `f(a,b) = V_max × H(a,K_a,n_a) × H(b,K_b,n_b)`.
///
/// Delegates to upstream `HillGateGpu` — single dispatch replaces the
/// CPU scalar `signal_integration::two_input_hill` loop.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn hill_gate_gpu(
    input_a: &[f64],
    input_b: &[f64],
    cfg: &HillGateConfig,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    use barracuda::ops::bio::hill_gate::{HillGateGpu, HillGateParams};
    use wgpu::util::DeviceExt;

    let len_a = input_a.len();
    let len_b = input_b.len();
    let out_len = len_a * len_b;
    if len_a == 0 || len_b == 0 {
        return Ok(Vec::new());
    }

    let d = device.device();
    let elem_size = std::mem::size_of::<f64>();

    let a_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hill_gate_a"),
        contents: bytemuck::cast_slice(input_a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hill_gate_b"),
        contents: bytemuck::cast_slice(input_b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hill_gate_out"),
        size: (out_len * elem_size) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = HillGateParams {
        n_a: len_a as u32,
        n_b: len_b as u32,
        mode: 1,
        _pad: 0,
        k_a: cfg.k_a,
        k_b: cfg.k_b,
        n_a_exp: cfg.n_a,
        n_b_exp: cfg.n_b,
        vmax: cfg.vmax,
        _pad2: 0.0,
    };

    let op = HillGateGpu::new(device.clone());
    op.dispatch(&a_buf, &b_buf, &out_buf, &params);

    let out_bytes = (out_len * elem_size) as u64;
    let staging = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hill_gate_staging"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
    device.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.device().poll(wgpu::Maintain::Wait);
    let view = slice.get_mapped_range();
    let result: Vec<f64> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_ops::tests_ops::test_device;

    #[test]
    fn gpu_hill_activation_batch_basic() {
        let Some((_guard, dev)) = test_device() else {
            return;
        };
        let result = hill_activation_batch_gpu(&[0.5, 1.0, 2.0], 1.0, 0.5, 2.0, &dev).unwrap();
        assert_eq!(result.len(), 3);
        for &v in &result {
            assert!((0.0..=1.0).contains(&v), "Hill output out of [0,1]: {v}");
        }
    }
}
