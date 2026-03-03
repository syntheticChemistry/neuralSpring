// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: PhyloNet-HMM introgression (Paper 018).
//!
//! Validates GPU `Tensor::matmul` for HMM forward-pass operations used in
//! PhyloNet-HMM introgression detection: transition probability matrix,
//! emission scoring, state posterior computation.
//!
//! ## S-14 workaround
//!
//! Uses `alpha × A^T` and `obs × B^T` patterns (transpose second operand).
//!
//! ## S-15 workaround
//!
//! All data uses `rng.uniform()` ([0, 1)) to avoid matmul hang on RTX 4070.
//!
//! ## Provenance
//!
//! Python baseline: `control/introgression/introgression.py`

#![expect(clippy::cast_possible_truncation, reason = "validation binary")]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_tensor;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{gpu_readback, max_abs_diff_gpu_vs_cpu, ValidationHarness};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_gpu_introgression");

    validate_transition_matmul(&mut h, &device);
    validate_emission_matmul(&mut h, &device);
    validate_state_posterior(&mut h, &device);
    validate_probabilities_in_unit_interval(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

/// Check 1: HMM transition probability matrix via matmul.
/// `alpha_batch` (`n_seqs` × `n_states`) × A^T (`n_states` × `n_states`) → transition output.
fn validate_transition_matmul(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(42);
    let n_states = 4_usize;
    let n_seqs = 10_usize;

    let alpha: Vec<Vec<f64>> = (0..n_seqs)
        .map(|_| (0..n_states).map(|_| rng.uniform()).collect())
        .collect();
    let trans: Vec<Vec<f64>> = (0..n_states)
        .map(|_| (0..n_states).map(|_| rng.uniform()).collect())
        .collect();

    let mut cpu_out = vec![0.0_f64; n_seqs * n_states];
    for i in 0..n_seqs {
        for j in 0..n_states {
            for k in 0..n_states {
                cpu_out[i * n_states + j] += alpha[i][k] * trans[j][k];
            }
        }
    }

    let alpha_flat: Vec<f32> = alpha
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let trans_flat: Vec<f32> = trans
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let alpha_t = gpu_tensor!(h, &alpha_flat, &[n_seqs, n_states], device);
    let trans_t = gpu_tensor!(h, &trans_flat, &[n_states, n_states], device);
    let trans_t_t = match trans_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let out_t = match alpha_t.matmul(&trans_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("alpha × A^T: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };
    let max_diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_out);
    h.check_upper(
        &format!("transition matmul: max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// Check 2: Emission probability via matmul.
/// `obs_features` (`n_seqs` × `n_obs`) × B^T (`n_states` × `n_obs`)^T → (`n_seqs` × `n_states`).
fn validate_emission_matmul(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(43);
    let n_states = 4_usize;
    let n_obs = 3_usize;
    let n_seqs = 10_usize;

    let obs: Vec<Vec<f64>> = (0..n_seqs)
        .map(|_| (0..n_obs).map(|_| rng.uniform()).collect())
        .collect();
    let emission: Vec<Vec<f64>> = (0..n_states)
        .map(|_| (0..n_obs).map(|_| rng.uniform()).collect())
        .collect();

    let mut cpu_out = vec![0.0_f64; n_seqs * n_states];
    for i in 0..n_seqs {
        for j in 0..n_states {
            for k in 0..n_obs {
                cpu_out[i * n_states + j] += obs[i][k] * emission[j][k];
            }
        }
    }

    let obs_flat: Vec<f32> = obs
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let emission_flat: Vec<f32> = emission
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let obs_t = gpu_tensor!(h, &obs_flat, &[n_seqs, n_obs], device);
    let emission_t = gpu_tensor!(h, &emission_flat, &[n_states, n_obs], device);
    let emission_t_t = match emission_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let out_t = match obs_t.matmul(&emission_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("obs × B^T: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };
    let max_diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_out);
    h.check_upper(
        &format!("emission matmul: max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// Check 3: State posterior computation (element-wise product of transition and emission).
fn validate_state_posterior(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(44);
    let n_states = 4_usize;
    let n_seqs = 10_usize;

    let trans_out: Vec<Vec<f64>> = (0..n_seqs)
        .map(|_| (0..n_states).map(|_| rng.uniform()).collect())
        .collect();
    let emission_out: Vec<Vec<f64>> = (0..n_seqs)
        .map(|_| (0..n_states).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_posterior: Vec<f64> = trans_out
        .iter()
        .zip(emission_out.iter())
        .flat_map(|(t, e)| t.iter().zip(e.iter()).map(|(&a, &b)| a * b))
        .collect();

    let trans_flat: Vec<f32> = trans_out
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let emission_flat: Vec<f32> = emission_out
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let trans_t = gpu_tensor!(h, &trans_flat, &[n_seqs, n_states], device);
    let emission_t = gpu_tensor!(h, &emission_flat, &[n_seqs, n_states], device);

    let posterior_t = match trans_t.mul(&emission_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("element-wise mul: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &posterior_t) else {
        return;
    };
    let max_diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_posterior);
    h.check_upper(
        &format!("state posterior (trans × emit): max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// Check 4: All probabilities in \[0, 1\].
/// Use row-normalized alpha and transition matrix (stochastic) so output stays in \[0,1\].
fn validate_probabilities_in_unit_interval(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(45);
    let n_states = 4_usize;
    let n_seqs = 10_usize;

    let mut alpha: Vec<Vec<f64>> = (0..n_seqs)
        .map(|_| (0..n_states).map(|_| rng.uniform()).collect())
        .collect();
    for row in &mut alpha {
        let s: f64 = row.iter().sum();
        if s > 0.0 {
            for x in row.iter_mut() {
                *x /= s;
            }
        }
    }
    let mut trans: Vec<Vec<f64>> = (0..n_states)
        .map(|_| (0..n_states).map(|_| rng.uniform()).collect())
        .collect();
    for row in &mut trans {
        let s: f64 = row.iter().sum();
        if s > 0.0 {
            for x in row.iter_mut() {
                *x /= s;
            }
        }
    }

    let alpha_flat: Vec<f32> = alpha
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let trans_flat: Vec<f32> = trans
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let alpha_t = gpu_tensor!(h, &alpha_flat, &[n_seqs, n_states], device);
    let trans_t = gpu_tensor!(h, &trans_flat, &[n_states, n_states], device);
    let trans_t_t = match trans_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let out_t = match alpha_t.matmul(&trans_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };

    let in_range = out
        .iter()
        .all(|&x| (0.0_f32..=1.0_f32 + tolerances::GPU_BOUNDS_SLACK_F32 as f32).contains(&x));
    h.check_bool(
        "all transition output values in [0, 1] (or small numerical overflow)",
        in_range,
    );
}

/// Check 5: Determinism.
fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(46);
    let n_states = 4_usize;
    let n_seqs = 10_usize;

    let alpha_flat: Vec<f32> = (0..n_seqs * n_states)
        .map(|_| rng.uniform() as f32)
        .collect();
    let trans_flat: Vec<f32> = (0..n_states * n_states)
        .map(|_| rng.uniform() as f32)
        .collect();

    let run = |_: u32| -> Option<Vec<f32>> {
        let a = Tensor::from_data(&alpha_flat, vec![n_seqs, n_states], device.clone()).ok()?;
        let t = Tensor::from_data(&trans_flat, vec![n_states, n_states], device.clone()).ok()?;
        let tt = t.transpose().ok()?;
        let out = a.matmul(&tt).ok()?;
        out.to_vec().ok()
    };

    let Some(r1) = run(1) else {
        h.check_bool("determinism run1 failed", false);
        return;
    };
    let Some(r2) = run(2) else {
        h.check_bool("determinism run2 failed", false);
        return;
    };

    let identical = r1
        .iter()
        .zip(r2.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    h.check_bool("determinism: two GPU runs bit-identical", identical);
}
