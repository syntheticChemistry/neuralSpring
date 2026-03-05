// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated eigensolvers and pangenome statistics.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "eigensolver GPU ops cast matrix dimensions for tensor construction"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use std::sync::Arc;

use super::reduction::chi_squared_gpu;
use crate::tolerances;

/// GPU batched eigenvalue decomposition via `BatchedEighGpu`.
///
/// Replaces `anderson_localization::jacobi_eigh` and `eigh::eigh_householder_qr`
/// for GPU-resident workloads. Uses Jacobi sweeps on GPU (single dispatch for
/// n <= 32, multi-dispatch for larger).
///
/// # Errors
///
/// Returns an error if GPU eigensolve fails.
pub fn eigh_gpu(
    a: &[f64],
    n: usize,
    device: &Arc<WgpuDevice>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    use barracuda::ops::linalg::BatchedEighGpu;
    if n <= 32 {
        BatchedEighGpu::execute_single_dispatch(
            device.clone(),
            a,
            n,
            1,
            30,
            tolerances::JACOBI_GPU_CONVERGENCE,
        )
        .map_err(|e| format!("eigh_gpu single_dispatch: {e}"))
    } else {
        BatchedEighGpu::execute_f64(device.clone(), a, n, 1, 30)
            .map_err(|e| format!("eigh_gpu: {e}"))
    }
}

/// GPU batch disorder sweep: eigensolve multiple Hamiltonians in one dispatch.
///
/// Replaces `anderson_localization::disorder_sweep`. Batches all W values
/// into a single `BatchedEighGpu::execute_single_dispatch` call (n <= 32)
/// and computes mean IPR from eigenvectors via `BatchIprGpu` on GPU.
/// Provenance: eigensolve from hotSpring precision → barracuda absorption,
/// IPR from neuralSpring spectral → barracuda absorption (S52).
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn disorder_sweep_gpu(
    hamiltonians: &[f64],
    n: usize,
    batch_size: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    use barracuda::ops::linalg::BatchedEighGpu;
    use barracuda::spectral::BatchIprGpu;
    use wgpu::util::DeviceExt;

    let (eigenvalues, eigenvectors) = if n <= 32 {
        BatchedEighGpu::execute_single_dispatch(
            device.clone(),
            hamiltonians,
            n,
            batch_size,
            30,
            tolerances::JACOBI_GPU_CONVERGENCE,
        )
        .map_err(|e| format!("disorder_sweep_gpu: {e}"))?
    } else {
        BatchedEighGpu::execute_f64(device.clone(), hamiltonians, n, batch_size, 30)
            .map_err(|e| format!("disorder_sweep_gpu: {e}"))?
    };

    let _ = eigenvalues;

    let total_vectors = batch_size * n;
    let ev_f32: Vec<f32> = eigenvectors.iter().map(|&v| v as f32).collect();
    let d = device.device();

    let ev_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("disorder_sweep_ev"),
        contents: bytemuck::cast_slice(&ev_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let ipr_buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("disorder_sweep_ipr"),
        size: (total_vectors * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let ipr_op = BatchIprGpu::new(device.clone());
    ipr_op.dispatch(&ev_buf, &ipr_buf, n as u32, total_vectors as u32);

    let staging = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("disorder_sweep_staging"),
        size: (total_vectors * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&ipr_buf, 0, &staging, 0, (total_vectors * 4) as u64);
    device.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.device().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let view = slice.get_mapped_range();
    let ipr_f32: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();

    let mut mean_iprs = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let batch_start = b * n;
        let sum: f32 = ipr_f32[batch_start..batch_start + n].iter().sum();
        mean_iprs.push(f64::from(sum / n as f32));
    }
    Ok(mean_iprs)
}

/// GPU spectrum chi-squared: adapts pangenome expected fractions to absolute
/// counts, then computes chi-squared via Tensor elementwise ops.
///
/// Replaces `pangenome_selection::spectrum_chi_squared` for GPU dispatch.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn spectrum_chi_squared_gpu(
    observed: &[f64],
    expected_frac: &[f64],
    device: &Arc<WgpuDevice>,
) -> Result<f64, String> {
    let total: f64 = observed.iter().sum();
    if total == 0.0 {
        return Ok(0.0);
    }
    let expected: Vec<f64> = expected_frac.iter().map(|&f| f * total).collect();
    chi_squared_gpu(observed, &expected, device)
}

/// GPU selection coefficient: L2 deviation of normalized spectrum from neutral.
///
/// Replaces `pangenome_selection::selection_coefficient` for GPU dispatch.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn selection_coefficient_gpu(
    observed: &[f64],
    neutral: &[f64],
    device: &Arc<WgpuDevice>,
) -> Result<f64, String> {
    let total: f64 = observed.iter().sum();
    if total == 0.0 {
        return Ok(0.0);
    }
    let normalized: Vec<f64> = observed.iter().map(|&o| o / total).collect();
    let diff: Vec<f64> = normalized
        .iter()
        .zip(neutral.iter())
        .map(|(&n, &ne)| n - ne)
        .collect();
    let diff_f32: Vec<f32> = diff.iter().map(|&v| v as f32).collect();
    let t = Tensor::from_data(&diff_f32, vec![diff_f32.len()], device.clone())
        .map_err(|e| format!("selection_coeff upload: {e}"))?;
    let norm = t.norm().map_err(|e| format!("selection_coeff norm: {e}"))?;
    let result = norm
        .to_vec()
        .map_err(|e| format!("selection_coeff readback: {e}"))?;
    Ok(f64::from(result[0]))
}
