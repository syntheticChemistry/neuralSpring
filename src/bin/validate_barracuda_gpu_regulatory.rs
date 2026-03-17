// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: Gene regulatory network (Paper 020).
//!
//! Validates GPU `Tensor::matmul` + `Tensor::tanh` for regulatory network
//! dynamics: Hill function input transformation, weight application,
//! tanh activation on regulatory output.
//!
//! ## S-14 workaround
//!
//! Uses `input × weights^T` pattern (transpose second operand).
//!
//! ## S-15 workaround
//!
//! All data uses `rng.uniform()` ([0, 1)) to avoid matmul hang on RTX 4070.
//!
//! ## Provenance
//!
//! Python baseline: `control/regulatory_network/regulatory_network.py`

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
            println!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_gpu_regulatory");

    validate_hill_input_matmul(&mut h, &device);
    validate_regulatory_weight_application(&mut h, &device);
    validate_tanh_activation(&mut h, &device);
    validate_output_valid_range(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

/// CPU reference: input (`n_conditions` × `n_inputs`) × weights^T (`n_genes` × `n_inputs`)^T.
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

/// Check 1: Hill function input transformation via matmul.
/// input (`n_conditions` × `n_inputs`) × W^T (`n_genes` × `n_inputs`)^T.
fn validate_hill_input_matmul(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(42);
    let n_genes = 8_usize;
    let n_inputs = 5_usize;
    let n_conditions = 12_usize;

    let input: Vec<Vec<f64>> = (0..n_conditions)
        .map(|_| (0..n_inputs).map(|_| rng.uniform()).collect())
        .collect();
    let weights: Vec<Vec<f64>> = (0..n_genes)
        .map(|_| (0..n_inputs).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_out = cpu_matmul_a_bt(&input, &weights);
    let cpu_flat: Vec<f64> = cpu_out.iter().flat_map(|r| r.iter().copied()).collect();

    let input_flat: Vec<f32> = input
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let weights_flat: Vec<f32> = weights
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let input_t = gpu_tensor!(h, &input_flat, &[n_conditions, n_inputs], device);
    let weights_t = gpu_tensor!(h, &weights_flat, &[n_genes, n_inputs], device);
    let weights_t_t = match weights_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let out_t = match input_t.matmul(&weights_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("input × weights^T: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };
    let max_diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_flat);
    h.check_upper(
        &format!("Hill input matmul: max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// Check 2: Regulatory network weight application.
fn validate_regulatory_weight_application(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(43);
    let n_genes = 8_usize;
    let n_inputs = 5_usize;
    let n_conditions = 12_usize;

    let input: Vec<Vec<f64>> = (0..n_conditions)
        .map(|_| (0..n_inputs).map(|_| rng.uniform()).collect())
        .collect();
    let weights: Vec<Vec<f64>> = (0..n_genes)
        .map(|_| (0..n_inputs).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_out = cpu_matmul_a_bt(&input, &weights);
    let cpu_flat: Vec<f64> = cpu_out.iter().flat_map(|r| r.iter().copied()).collect();

    let input_flat: Vec<f32> = input
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let weights_flat: Vec<f32> = weights
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let input_t = gpu_tensor!(h, &input_flat, &[n_conditions, n_inputs], device);
    let weights_t = gpu_tensor!(h, &weights_flat, &[n_genes, n_inputs], device);
    let weights_t_t = match weights_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let out_t = match input_t.matmul(&weights_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("regulatory weight matmul: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };
    let max_diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_flat);
    h.check_upper(
        &format!("regulatory weight application: max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// Check 3: tanh activation on regulatory output.
/// `regulatory_output` = tanh(input × weights^T).
fn validate_tanh_activation(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(44);
    let n_genes = 8_usize;
    let n_inputs = 5_usize;
    let n_conditions = 12_usize;

    let input: Vec<Vec<f64>> = (0..n_conditions)
        .map(|_| (0..n_inputs).map(|_| rng.uniform()).collect())
        .collect();
    let weights: Vec<Vec<f64>> = (0..n_genes)
        .map(|_| (0..n_inputs).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_linear = cpu_matmul_a_bt(&input, &weights);
    let cpu_tanh: Vec<f64> = cpu_linear
        .iter()
        .flat_map(|r| r.iter().map(|&x| x.tanh()))
        .collect();

    let input_flat: Vec<f32> = input
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let weights_flat: Vec<f32> = weights
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let input_t = gpu_tensor!(h, &input_flat, &[n_conditions, n_inputs], device);
    let weights_t = gpu_tensor!(h, &weights_flat, &[n_genes, n_inputs], device);
    let weights_t_t = match weights_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let linear_t = match input_t.matmul(&weights_t_t) {
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
        &format!("tanh(regulatory output): max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

/// Check 4: Output in valid range [-1, 1] (tanh bounds).
fn validate_output_valid_range(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(45);
    let n_genes = 8_usize;
    let n_inputs = 5_usize;
    let n_conditions = 12_usize;

    let input_flat: Vec<f32> = (0..n_conditions * n_inputs)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights_flat: Vec<f32> = (0..n_genes * n_inputs)
        .map(|_| rng.uniform() as f32)
        .collect();

    let input_t = gpu_tensor!(h, &input_flat, &[n_conditions, n_inputs], device);
    let weights_t = gpu_tensor!(h, &weights_flat, &[n_genes, n_inputs], device);
    let weights_t_t = match weights_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let linear_t = match input_t.matmul(&weights_t_t) {
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

    let in_range = out.iter().all(|&x| {
        (-1.0_f32 - tolerances::GPU_BOUNDS_SLACK_F32 as f32
            ..=1.0_f32 + tolerances::GPU_BOUNDS_SLACK_F32 as f32)
            .contains(&x)
    });
    h.check_bool("tanh output in [-1, 1]", in_range);
}

/// Check 5: Determinism.
fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(46);
    let n_genes = 8_usize;
    let n_inputs = 5_usize;
    let n_conditions = 12_usize;

    let input_flat: Vec<f32> = (0..n_conditions * n_inputs)
        .map(|_| rng.uniform() as f32)
        .collect();
    let weights_flat: Vec<f32> = (0..n_genes * n_inputs)
        .map(|_| rng.uniform() as f32)
        .collect();

    let run = |_: u32| -> Option<Vec<f32>> {
        let i =
            Tensor::from_data(&input_flat, vec![n_conditions, n_inputs], device.clone()).ok()?;
        let w = Tensor::from_data(&weights_flat, vec![n_genes, n_inputs], device.clone()).ok()?;
        let wt = w.transpose().ok()?;
        let mm = i.matmul(&wt).ok()?;
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
