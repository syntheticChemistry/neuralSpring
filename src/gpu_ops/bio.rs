// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated bio operations: HMM forward/backward/Viterbi,
//! pairwise distance, Hill activation.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use std::sync::Arc;

/// GPU HMM forward step: `alpha[t] = normalize(B[:,o_t] * (A^T @ alpha[t-1]))`.
///
/// Single timestep of the forward algorithm via GPU GEMV + elementwise.
/// The full forward pass calls this in a loop; each step's GPU matmul
/// replaces the CPU double loop.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn hmm_forward_step_gpu(
    alpha_prev: &[f64],
    transition: &[f64],
    emission_col: &[f64],
    n_states: usize,
    device: &Arc<WgpuDevice>,
) -> Result<(Vec<f64>, f64), String> {
    let a_f32: Vec<f32> = alpha_prev.iter().map(|&x| x as f32).collect();
    let t_f32: Vec<f32> = transition.iter().map(|&x| x as f32).collect();
    let e_f32: Vec<f32> = emission_col.iter().map(|&x| x as f32).collect();

    let alpha_t = Tensor::from_data(&a_f32, vec![1, n_states], device.clone())
        .map_err(|e| format!("hmm_fwd alpha: {e}"))?;
    let trans_t = Tensor::from_data(&t_f32, vec![n_states, n_states], device.clone())
        .map_err(|e| format!("hmm_fwd trans: {e}"))?;

    let propagated = alpha_t
        .matmul(&trans_t)
        .map_err(|e| format!("hmm_fwd matmul: {e}"))?;

    let emit_t = Tensor::from_data(&e_f32, vec![1, n_states], device.clone())
        .map_err(|e| format!("hmm_fwd emit: {e}"))?;

    let raw = propagated
        .mul(&emit_t)
        .map_err(|e| format!("hmm_fwd mul: {e}"))?;

    let scale_t = raw.sum().map_err(|e| format!("hmm_fwd sum: {e}"))?;
    let scale_val = scale_t
        .to_vec()
        .map_err(|e| format!("hmm_fwd scale_read: {e}"))?[0];

    let raw_vec = raw.to_vec().map_err(|e| format!("hmm_fwd raw_read: {e}"))?;

    let scale = f64::from(scale_val).max(crate::primitives::LOG_GUARD);
    let alpha_new: Vec<f64> = raw_vec.iter().map(|&x| f64::from(x) / scale).collect();

    Ok((alpha_new, scale))
}

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
    use barracuda::ops::bio::PairwiseL2Gpu;
    use wgpu::util::DeviceExt;

    let n_pairs = n * (n - 1) / 2;
    if n < 2 {
        return Ok(Vec::new());
    }

    let input_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let d = device.device();

    let input_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pairwise_l2_input"),
        contents: bytemuck::cast_slice(&input_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pairwise_l2_output"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let op = PairwiseL2Gpu::new(device.clone());
    op.dispatch(&input_buf, &output_buf, n as u32, dim as u32);

    let staging = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pairwise_l2_staging"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, (n_pairs * 4) as u64);
    device.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.device().poll(wgpu::Maintain::Wait);
    let view = slice.get_mapped_range();
    let f32_data: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();

    Ok(f32_data.into_iter().map(f64::from).collect())
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

/// GPU HMM forward chain: run the full forward algorithm with GPU GEMV per step.
///
/// Composes `hmm_forward_step_gpu` over all observations, returning
/// the log-likelihood. Replaces `Hmm::forward` for GPU execution.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn hmm_forward_chain_gpu(
    initial: &[f64],
    transition: &[f64],
    emission: &[f64],
    observations: &[usize],
    n_states: usize,
    n_obs: usize,
    device: &Arc<WgpuDevice>,
) -> Result<f64, String> {
    let t_len = observations.len();
    if t_len == 0 {
        return Ok(0.0);
    }

    let emit_col: Vec<f64> = (0..n_states)
        .map(|i| emission[i * n_obs + observations[0]])
        .collect();
    let mut alpha: Vec<f64> = initial
        .iter()
        .zip(emit_col.iter())
        .map(|(&pi, &b)| pi * b)
        .collect();
    let sum0: f64 = alpha.iter().sum();
    let scale0 = sum0.max(crate::primitives::LOG_GUARD);
    for v in &mut alpha {
        *v /= scale0;
    }
    let mut log_likelihood = scale0.ln();

    for t in 1..t_len {
        let e_col: Vec<f64> = (0..n_states)
            .map(|i| emission[i * n_obs + observations[t]])
            .collect();
        let (new_alpha, scale) =
            hmm_forward_step_gpu(&alpha, transition, &e_col, n_states, device)?;
        log_likelihood += scale.max(crate::primitives::LOG_GUARD).ln();
        alpha = new_alpha;
    }

    Ok(log_likelihood)
}

/// GPU HMM Viterbi chain: run the full Viterbi algorithm with GPU per step.
///
/// Composes `hmm_viterbi_step_gpu` over all observations, returning
/// the most likely state sequence and its log-probability.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn hmm_viterbi_chain_gpu(
    initial: &[f64],
    transition: &[f64],
    emission: &[f64],
    observations: &[usize],
    n_states: usize,
    n_obs: usize,
    device: &Arc<WgpuDevice>,
) -> Result<(Vec<usize>, f64), String> {
    let t_len = observations.len();
    if t_len == 0 {
        return Ok((Vec::new(), 0.0));
    }

    let log_trans: Vec<f64> = transition
        .iter()
        .map(|&x| x.max(crate::primitives::LOG_GUARD).ln())
        .collect();

    let mut delta: Vec<f64> = initial
        .iter()
        .enumerate()
        .map(|(i, &pi)| {
            pi.max(crate::primitives::LOG_GUARD).ln()
                + emission[i * n_obs + observations[0]]
                    .max(crate::primitives::LOG_GUARD)
                    .ln()
        })
        .collect();

    let mut psi_all = Vec::with_capacity(t_len);

    for t in 1..t_len {
        let log_emit: Vec<f64> = (0..n_states)
            .map(|i| {
                emission[i * n_obs + observations[t]]
                    .max(crate::primitives::LOG_GUARD)
                    .ln()
            })
            .collect();
        let (new_delta, psi) =
            hmm_viterbi_step_gpu(&delta, &log_trans, &log_emit, n_states, device)?;
        psi_all.push(psi);
        delta = new_delta;
    }

    let mut best_state = 0;
    let mut best_val = f64::NEG_INFINITY;
    for (j, &d) in delta.iter().enumerate() {
        if d > best_val {
            best_val = d;
            best_state = j;
        }
    }

    let mut path = vec![0usize; t_len];
    path[t_len - 1] = best_state;
    for t in (0..t_len - 1).rev() {
        path[t] = psi_all[t][path[t + 1]];
    }

    Ok((path, best_val))
}

/// GPU two-input Hill gate: `f(a,b) = V_max × H(a,K_a,n_a) × H(b,K_b,n_b)`.
///
/// Delegates to upstream `HillGateGpu` — single dispatch replaces the
/// CPU scalar `signal_integration::two_input_hill` loop.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
#[allow(clippy::too_many_arguments)]
pub fn hill_gate_gpu(
    input_a: &[f64],
    input_b: &[f64],
    vmax: f64,
    k_a: f64,
    k_b: f64,
    n_a: f64,
    n_b: f64,
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
        k_a,
        k_b,
        n_a_exp: n_a,
        n_b_exp: n_b,
        vmax,
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
    let out_bytes = (total_out * elem_size) as u64;
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
        pop_size as u32,
        genome_len as u32,
        n_objectives as u32,
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
    device.device().poll(wgpu::Maintain::Wait);
    let view = slice.get_mapped_range();
    let result: Vec<f64> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();

    Ok(result)
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
#[allow(clippy::too_many_arguments)]
pub fn swarm_nn_forward_gpu(
    weights: &[f64],
    inputs: &[f64],
    n_controllers: usize,
    n_evals: usize,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<u32>, String> {
    use barracuda::ops::bio::swarm_nn::{SwarmNnGpu, SwarmNnParams};
    use wgpu::util::DeviceExt;

    let total_actions = n_controllers * n_evals;
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
    let action_bytes = (total_actions * std::mem::size_of::<u32>()) as u64;
    let a_buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("swarm_nn_actions"),
        size: action_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = SwarmNnParams {
        n_controllers: n_controllers as u32,
        n_evals: n_evals as u32,
        input_dim: input_dim as u32,
        hidden_dim: hidden_dim as u32,
        output_dim: output_dim as u32,
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
    device.device().poll(wgpu::Maintain::Wait);
    let view = slice.get_mapped_range();
    let u32_data: Vec<u32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();

    Ok(u32_data)
}

/// GPU HMM backward step: `β_t[i] = sum_j(A[i,j] * B[j,o] * β_{t+1}[j]) / scale`.
///
/// Single reverse-timestep via GPU GEMV. The full backward pass calls
/// this in a loop from T-2 down to 0.
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn hmm_backward_step_gpu(
    beta_next: &[f64],
    transition: &[f64],
    emission_col: &[f64],
    scale: f64,
    n_states: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>, String> {
    let b_f32: Vec<f32> = beta_next.iter().map(|&x| x as f32).collect();
    let t_f32: Vec<f32> = transition.iter().map(|&x| x as f32).collect();
    let e_f32: Vec<f32> = emission_col.iter().map(|&x| x as f32).collect();

    let beta_t = Tensor::from_data(&b_f32, vec![1, n_states], device.clone())
        .map_err(|e| format!("hmm_bwd beta: {e}"))?;
    let emit_t = Tensor::from_data(&e_f32, vec![1, n_states], device.clone())
        .map_err(|e| format!("hmm_bwd emit: {e}"))?;
    let weighted = beta_t
        .mul(&emit_t)
        .map_err(|e| format!("hmm_bwd mul: {e}"))?;

    let trans_t = Tensor::from_data(&t_f32, vec![n_states, n_states], device.clone())
        .map_err(|e| format!("hmm_bwd trans: {e}"))?;
    let at = trans_t
        .transpose()
        .map_err(|e| format!("hmm_bwd transpose: {e}"))?;
    let result = weighted
        .matmul(&at)
        .map_err(|e| format!("hmm_bwd matmul: {e}"))?;

    let result_vec = result.to_vec().map_err(|e| format!("hmm_bwd read: {e}"))?;

    let guard = crate::primitives::LOG_GUARD;
    let safe_scale = if scale.abs() < guard { guard } else { scale };
    let beta_new: Vec<f64> = result_vec
        .iter()
        .map(|&x| f64::from(x) / safe_scale)
        .collect();

    Ok(beta_new)
}

/// GPU HMM Viterbi step: `δ_t[j] = max_i(δ_{t-1}[i] + logA[i,j]) + logB[j,o_t]`.
///
/// Returns `(delta_t, psi_t)` where `psi_t[j] = argmax_i(...)`.
/// Score matrix and max-reduction run on GPU. Argmax uses upstream
/// `Tensor::argmax_dim(0)` (rewired S72 — previously CPU loop; upstream
/// `argmax_dim` absorbed from cross-spring evolution: neuralSpring request
/// → `ToadStool` S60 implementation → available since `0c998992`).
///
/// # Errors
///
/// Returns an error if GPU operations fail.
pub fn hmm_viterbi_step_gpu(
    delta_prev: &[f64],
    log_transition: &[f64],
    log_emission_col: &[f64],
    n_states: usize,
    device: &Arc<WgpuDevice>,
) -> Result<(Vec<f64>, Vec<usize>), String> {
    let n = n_states;
    let d_f32: Vec<f32> = delta_prev.iter().map(|&x| x as f32).collect();
    let la_f32: Vec<f32> = log_transition.iter().map(|&x| x as f32).collect();

    let delta_col = Tensor::from_data(&d_f32, vec![n, 1], device.clone())
        .map_err(|e| format!("viterbi delta: {e}"))?;
    let delta_broad = delta_col
        .broadcast(vec![n, n])
        .map_err(|e| format!("viterbi broadcast: {e}"))?;
    let log_a = Tensor::from_data(&la_f32, vec![n, n], device.clone())
        .map_err(|e| format!("viterbi log_a: {e}"))?;
    let scores = delta_broad
        .add(&log_a)
        .map_err(|e| format!("viterbi add: {e}"))?;

    let max_vals = scores
        .max_dim(0, false)
        .map_err(|e| format!("viterbi max: {e}"))?;
    let max_f32 = max_vals
        .to_vec()
        .map_err(|e| format!("viterbi max_read: {e}"))?;

    let argmax_t = scores
        .argmax_dim(0)
        .map_err(|e| format!("viterbi argmax: {e}"))?;
    let argmax_u32 = argmax_t
        .to_vec_u32()
        .map_err(|e| format!("viterbi argmax_read: {e}"))?;

    let mut delta_new = Vec::with_capacity(n);
    let mut psi = Vec::with_capacity(n);
    for j in 0..n {
        delta_new.push(f64::from(max_f32[j]) + log_emission_col[j]);
        psi.push(argmax_u32[j] as usize);
    }

    Ok((delta_new, psi))
}
