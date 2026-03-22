// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU evolution and optimization operations.

use barracuda::device::WgpuDevice;
use std::sync::Arc;

/// GPU pairwise distance matrix for n vectors of dimension d.
///
/// Returns flat upper-triangle distances (n*(n-1)/2 elements).
/// Rewired to upstream `PairwiseL2Gpu` — single GPU dispatch replaces O(n²) loop.
/// Provenance: neuralSpring local → barracuda absorption (S52).
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn pairwise_l2_matrix_gpu(
    data: &[f64],
    n: usize,
    dim: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    use crate::safe_cast::{f64_f32, usize_u32, usize_u64};
    use barracuda::ops::bio::PairwiseL2Gpu;
    use wgpu::util::DeviceExt;

    let n_pairs = n * (n - 1) / 2;
    if n < 2 {
        return Ok(Vec::new());
    }

    let input_f32: Vec<f32> = data.iter().map(|&v| f64_f32(v)).collect();
    let d = device.device();

    let input_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pairwise_l2_input"),
        contents: bytemuck::cast_slice(&input_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_bytes = usize_u64(n_pairs * 4);
    let output_buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pairwise_l2_output"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let op = PairwiseL2Gpu::new(device.clone());
    op.dispatch(
        &input_buf,
        &output_buf,
        usize_u32(n, "n_vectors")?,
        usize_u32(dim, "dim")?,
    )
    .map_err(|e| format!("PairwiseL2 dispatch: {e}"))?;

    let staging = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pairwise_l2_staging"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, out_bytes);
    device.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.device().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let view = slice.get_mapped_range();
    let f32_data: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();

    Ok(f32_data.into_iter().map(f64::from).collect())
}

/// GPU multi-objective fitness evaluation.
///
/// Delegates to upstream `MultiObjFitnessGpu` — single dispatch replaces
/// the CPU `directed_evolution::multi_objective_fitness` loop.
///
/// Returns `[pop × n_objectives]` fitness values.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn multi_obj_fitness_gpu(
    genotypes: &[f64],
    pop_size: usize,
    genome_len: usize,
    n_objectives: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    use crate::safe_cast::{usize_u32, usize_u64};
    use barracuda::ops::bio::MultiObjFitnessGpu;
    use wgpu::util::DeviceExt;

    let total_in = pop_size * genome_len;
    let total_out = pop_size * n_objectives;
    if total_in == 0 {
        return Ok(vec![0.0; total_out]);
    }

    let d = device.device();
    let elem_size = std::mem::size_of::<f64>();

    let geno_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("multi_obj_genotypes"),
        contents: bytemuck::cast_slice(genotypes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_bytes = usize_u64(total_out * elem_size);
    let fit_buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("multi_obj_fitness"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let op = MultiObjFitnessGpu::new(device.clone());
    op.dispatch(
        &geno_buf,
        &fit_buf,
        usize_u32(pop_size, "pop_size")?,
        usize_u32(genome_len, "genome_len")?,
        usize_u32(n_objectives, "n_objectives")?,
    );

    let staging = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("multi_obj_staging"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&fit_buf, 0, &staging, 0, out_bytes);
    device.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.device().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let view = slice.get_mapped_range();
    let result: Vec<f64> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();

    Ok(result)
}

/// Dimension parameters for swarm neural-network forward pass.
#[derive(Debug, Clone, Copy)]
pub struct SwarmNnDims {
    /// Number of swarm controllers evaluated in the batch.
    pub n_controllers: usize,
    /// Number of forward evaluations per controller.
    pub n_evals: usize,
    /// Input feature dimension per evaluation.
    pub input_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Output dimension (discrete action space size).
    pub output_dim: usize,
}

/// GPU swarm neural-network forward pass.
///
/// Delegates to upstream `SwarmNnGpu` — single dispatch evaluates all
/// controllers × evaluations in parallel on GPU.
///
/// Returns per-controller per-evaluation action indices.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn swarm_nn_forward_gpu(
    weights: &[f64],
    inputs: &[f64],
    dims: &SwarmNnDims,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<u32>, String> {
    use crate::safe_cast::{usize_u32, usize_u64};
    use barracuda::ops::bio::swarm_nn::{SwarmNnGpu, SwarmNnParams};
    use wgpu::util::DeviceExt;

    let total_actions = dims.n_controllers * dims.n_evals;
    if total_actions == 0 {
        return Ok(Vec::new());
    }

    let d = device.device();

    let w_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("swarm_nn_weights"),
        contents: bytemuck::cast_slice(weights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let i_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("swarm_nn_inputs"),
        contents: bytemuck::cast_slice(inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let action_bytes = usize_u64(total_actions * std::mem::size_of::<u32>());
    let a_buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("swarm_nn_actions"),
        size: action_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = SwarmNnParams {
        n_controllers: usize_u32(dims.n_controllers, "n_controllers")?,
        n_evals: usize_u32(dims.n_evals, "n_evals")?,
        input_dim: usize_u32(dims.input_dim, "input_dim")?,
        hidden_dim: usize_u32(dims.hidden_dim, "hidden_dim")?,
        output_dim: usize_u32(dims.output_dim, "output_dim")?,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let op = SwarmNnGpu::new(device.clone());
    op.dispatch(&w_buf, &i_buf, &a_buf, &params);

    let staging = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("swarm_nn_staging"),
        size: action_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&a_buf, 0, &staging, 0, action_bytes);
    device.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.device().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let view = slice.get_mapped_range();
    let u32_data: Vec<u32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();

    Ok(u32_data)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "GPU test setup uses expect for device creation"
)]
mod tests {
    use super::*;
    use crate::gpu_ops::tests_ops::test_device;
    use crate::tolerances;

    #[test]
    fn gpu_pairwise_l2_matrix_basic() {
        let Some((_guard, dev)) = test_device() else {
            return;
        };
        let vectors = vec![0.0, 0.0, 3.0, 4.0];
        let dist = pairwise_l2_matrix_gpu(&vectors, 2, 2, &dev)
            .expect("pairwise L2 GPU dispatch should succeed on test device");
        assert_eq!(dist.len(), 1, "upper triangle: n*(n-1)/2 = 1 pair");
        assert!(
            (dist[0] - 5.0).abs() < tolerances::GPU_CHI_SQUARED_F32,
            "dist([0,0],[3,4]) ≈ 5"
        );
    }

    #[test]
    fn gpu_pairwise_l2_single_vector_returns_empty() {
        let Some((_guard, dev)) = test_device() else {
            return;
        };
        let result = pairwise_l2_matrix_gpu(&[1.0, 2.0, 3.0], 1, 3, &dev)
            .expect("single vector should succeed");
        assert!(result.is_empty(), "n=1 → 0 pairs");
    }

    #[test]
    fn gpu_pairwise_l2_identity_vectors_zero_distance() {
        let Some((_guard, dev)) = test_device() else {
            return;
        };
        let vectors = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let dist =
            pairwise_l2_matrix_gpu(&vectors, 3, 2, &dev).expect("identical vectors should succeed");
        assert_eq!(dist.len(), 3, "3 vectors → 3 pairs");
        for (i, &d) in dist.iter().enumerate() {
            assert!(
                d < tolerances::GPU_CHI_SQUARED_F32,
                "pair {i}: identical vectors → distance ≈ 0, got {d}"
            );
        }
    }
}
