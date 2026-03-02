// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU HMM operations: forward, backward, Viterbi.

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_ops::tests_ops::test_device;
    use crate::tolerances;

    #[test]
    fn gpu_hmm_forward_step_normalized() {
        let Some((_guard, dev)) = test_device() else {
            return;
        };
        let alpha = vec![0.5, 0.5];
        let trans = vec![0.7, 0.3, 0.4, 0.6];
        let emit = vec![0.6, 0.4];
        let (new_alpha, scale) = hmm_forward_step_gpu(&alpha, &trans, &emit, 2, &dev).unwrap();
        let sum: f64 = new_alpha.iter().sum();
        assert!(
            (sum - 1.0).abs() < tolerances::GPU_HMM_STEP_F32,
            "HMM fwd step normalized: {sum}"
        );
        assert!(scale > 0.0, "scale must be positive");
    }

    #[test]
    fn gpu_hmm_backward_step_basic() {
        let Some((_guard, dev)) = test_device() else {
            return;
        };
        let beta_next = vec![1.0, 1.0];
        let transition = vec![0.7, 0.3, 0.4, 0.6];
        let emission_col = vec![0.5, 0.5];
        let result =
            hmm_backward_step_gpu(&beta_next, &transition, &emission_col, 1.0, 2, &dev).unwrap();
        assert_eq!(result.len(), 2);
        for &v in &result {
            assert!(v.is_finite(), "backward step should produce finite values");
        }
    }

    #[test]
    fn gpu_hmm_viterbi_step_basic() {
        let Some((_guard, dev)) = test_device() else {
            return;
        };
        let delta_prev = vec![0.0, -1.0];
        let log_trans = vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()];
        let log_emit = vec![0.6_f64.ln(), 0.4_f64.ln()];
        let (delta, psi) =
            hmm_viterbi_step_gpu(&delta_prev, &log_trans, &log_emit, 2, &dev).unwrap();
        assert_eq!(delta.len(), 2);
        assert_eq!(psi.len(), 2);
    }

    #[test]
    fn gpu_hmm_forward_chain_basic() {
        let Some((_guard, dev)) = test_device() else {
            return;
        };
        let trans = vec![0.7, 0.3, 0.4, 0.6];
        let emission = vec![0.1, 0.4, 0.5, 0.6, 0.3, 0.1];
        let initial = vec![0.6, 0.4];
        let obs = vec![0, 1, 2, 0];
        let log_lik = hmm_forward_chain_gpu(&initial, &trans, &emission, &obs, 2, 3, &dev).unwrap();
        assert!(log_lik.is_finite(), "log-likelihood must be finite");
        assert!(log_lik < 0.0, "log-likelihood should be negative");
    }

    #[test]
    fn gpu_hmm_viterbi_chain_basic() {
        let Some((_guard, dev)) = test_device() else {
            return;
        };
        let trans = vec![0.7, 0.3, 0.4, 0.6];
        let emission = vec![0.1, 0.4, 0.5, 0.6, 0.3, 0.1];
        let initial = vec![0.6, 0.4];
        let obs = vec![0, 1, 2, 0];
        let (path, log_prob) =
            hmm_viterbi_chain_gpu(&initial, &trans, &emission, &obs, 2, 3, &dev).unwrap();
        assert_eq!(path.len(), obs.len());
        assert!(log_prob.is_finite());
        for &s in &path {
            assert!(s < 2, "state {s} out of range");
        }
    }
}
