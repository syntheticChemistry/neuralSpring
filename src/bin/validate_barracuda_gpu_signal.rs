// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: Signal integration (Paper 021).
//!
//! Validates GPU `Tensor::matmul` + `Tensor::tanh` for multi-input signal
//! integration (cGMP + QS): `combined_signal` = tanh(signals × `gate_weights^T`).
//!
//! ## S-14 workaround
//!
//! Uses `signals × gate_weights^T` pattern (transpose second operand).
//!
//! ## S-15 workaround
//!
//! All data uses `rng.uniform()` ([0, 1)) to avoid matmul hang on RTX 4070.
//!
//! ## Provenance
//!
//! Python baseline: `control/signal_integration/signal_integration.py`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

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
    let mut h = ValidationHarness::new("barracuda_gpu_signal");

    validate_two_input_regulatory_signal(&mut h, &device);
    validate_signal_combination_weights(&mut h, &device);
    validate_tanh_gate_response(&mut h, &device);
    validate_and_gate_output_structure(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

/// CPU reference: signals (`n_samples` × `n_signals`) × weights^T (`n_outputs` × `n_signals`)^T.
fn cpu_matmul_a_bt(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b.len();
    let depth = a[0].len();
    let mut out = vec![vec![0.0_f64; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            for k in 0..depth {
                out[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    out
}

/// Check 1: Two-input regulatory signal via matmul.
/// signals (`n_samples` × `n_signals`) × `gate_weights^T` (`n_outputs` × `n_signals`)^T.
fn validate_two_input_regulatory_signal(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(42);
    let n_samples = 16_usize;
    let n_signals = 4_usize;
    let n_outputs = 3_usize;

    let signals: Vec<Vec<f64>> = (0..n_samples)
        .map(|_| (0..n_signals).map(|_| rng.uniform()).collect())
        .collect();
    let gate_weights: Vec<Vec<f64>> = (0..n_outputs)
        .map(|_| (0..n_signals).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_out = cpu_matmul_a_bt(&signals, &gate_weights);
    let cpu_flat: Vec<f64> = cpu_out.iter().flat_map(|r| r.iter().copied()).collect();

    let signals_flat: Vec<f32> = signals
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let weights_flat: Vec<f32> = gate_weights
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let signals_t = gpu_tensor!(h, &signals_flat, &[n_samples, n_signals], device);
    let weights_t = gpu_tensor!(h, &weights_flat, &[n_outputs, n_signals], device);
    let weights_t_t = match weights_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let out_t = match signals_t.matmul(&weights_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("signals × gate_weights^T: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };
    let max_diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_flat);
    h.check_upper(
        &format!("two-input regulatory signal matmul: max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// Check 2: Signal combination weights via matmul.
fn validate_signal_combination_weights(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(43);
    let n_samples = 16_usize;
    let n_signals = 4_usize;
    let n_outputs = 3_usize;

    let signals: Vec<Vec<f64>> = (0..n_samples)
        .map(|_| (0..n_signals).map(|_| rng.uniform()).collect())
        .collect();
    let gate_weights: Vec<Vec<f64>> = (0..n_outputs)
        .map(|_| (0..n_signals).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_out = cpu_matmul_a_bt(&signals, &gate_weights);
    let cpu_flat: Vec<f64> = cpu_out.iter().flat_map(|r| r.iter().copied()).collect();

    let signals_flat: Vec<f32> = signals
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let weights_flat: Vec<f32> = gate_weights
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let signals_t = gpu_tensor!(h, &signals_flat, &[n_samples, n_signals], device);
    let weights_t = gpu_tensor!(h, &weights_flat, &[n_outputs, n_signals], device);
    let weights_t_t = match weights_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let out_t = match signals_t.matmul(&weights_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("signal combination matmul: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };
    let max_diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_flat);
    h.check_upper(
        &format!("signal combination weights: max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// Check 3: tanh gate response.
/// `combined_signal` = tanh(signals × `gate_weights^T`).
fn validate_tanh_gate_response(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(44);
    let n_samples = 16_usize;
    let n_signals = 4_usize;
    let n_outputs = 3_usize;

    let signals: Vec<Vec<f64>> = (0..n_samples)
        .map(|_| (0..n_signals).map(|_| rng.uniform()).collect())
        .collect();
    let gate_weights: Vec<Vec<f64>> = (0..n_outputs)
        .map(|_| (0..n_signals).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_linear = cpu_matmul_a_bt(&signals, &gate_weights);
    let cpu_tanh: Vec<f64> = cpu_linear
        .iter()
        .flat_map(|r| r.iter().map(|&x| x.tanh()))
        .collect();

    let signals_flat: Vec<f32> = signals
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let weights_flat: Vec<f32> = gate_weights
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let signals_t = gpu_tensor!(h, &signals_flat, &[n_samples, n_signals], device);
    let weights_t = gpu_tensor!(h, &weights_flat, &[n_outputs, n_signals], device);
    let weights_t_t = match weights_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let linear_t = match signals_t.matmul(&weights_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };
    let act_t = match linear_t.tanh() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("tanh: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &act_t) else {
        return;
    };
    let max_diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_tanh);
    h.check_upper(
        &format!("tanh gate response: max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

/// Check 4: AND gate output structure — shape and bounded output.
fn validate_and_gate_output_structure(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(45);
    let n_samples = 16_usize;
    let n_signals = 4_usize;
    let n_outputs = 3_usize;

    let signals_flat: Vec<f32> = (0..n_samples * n_signals)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights_flat: Vec<f32> = (0..n_outputs * n_signals)
        .map(|_| rng.uniform() as f32)
        .collect();

    let signals_t = gpu_tensor!(h, &signals_flat, &[n_samples, n_signals], device);
    let weights_t = gpu_tensor!(h, &weights_flat, &[n_outputs, n_signals], device);
    let weights_t_t = match weights_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let linear_t = match signals_t.matmul(&weights_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };
    let act_t = match linear_t.tanh() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("tanh: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &act_t) else {
        return;
    };

    let correct_shape = out.len() == n_samples * n_outputs;
    h.check_bool(
        &format!(
            "AND gate output shape n_samples×n_outputs ({} elements)",
            out.len()
        ),
        correct_shape,
    );

    let in_range = out
        .iter()
        .all(|&x| (-1.0_f32 - 1e-5_f32..=1.0_f32 + 1e-5_f32).contains(&x));
    h.check_bool("AND gate output in [-1, 1]", in_range);
}

/// Check 5: Determinism.
fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(46);
    let n_samples = 16_usize;
    let n_signals = 4_usize;
    let n_outputs = 3_usize;

    let signals_flat: Vec<f32> = (0..n_samples * n_signals)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights_flat: Vec<f32> = (0..n_outputs * n_signals)
        .map(|_| rng.uniform() as f32)
        .collect();

    let run = |_: u32| -> Option<Vec<f32>> {
        let s =
            Tensor::from_data(&signals_flat, vec![n_samples, n_signals], device.clone()).ok()?;
        let w =
            Tensor::from_data(&weights_flat, vec![n_outputs, n_signals], device.clone()).ok()?;
        let wt = w.transpose().ok()?;
        let mm = s.matmul(&wt).ok()?;
        let act = mm.tanh().ok()?;
        act.to_vec().ok()
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
