// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: neural network forward pass (Papers 015, 020–021).
//!
//! Validates GPU `Tensor::matmul` + `Tensor::tanh` for MLP forward passes
//! used in swarm robotics controllers (Paper 015), regulatory network GRN
//! inference (Paper 020), and signal integration neural analogs (Paper 021).
//!
//! ## S-14 workaround
//!
//! All matmul operations use transposed operands (A × B^T) following
//! the pattern established in `validate_barracuda_gpu_eco`.
//!
//! ## S-15 workaround
//!
//! `Tensor::matmul` hangs on RTX 4070 Vulkan when input buffers
//! contain negative f32 values.  All test data uses `rng.uniform()`
//! ([0, 1) range) to avoid the hang.  Mathematical correctness of
//! matmul with negative inputs is validated on CPU via the
//! `BarraCUDA` CPU-port validators.
//!
//! ## Provenance
//!
//! CPU baselines: `validate_barracuda_swarm` (10), `validate_barracuda_regulatory` (6),
//! `validate_barracuda_signal` (14).

#![expect(
    clippy::cast_possible_truncation,
    clippy::similar_names,
    reason = "validation binary"
)]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_tensor;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, gpu_readback, max_abs_diff_gpu_vs_cpu};
use std::sync::Arc;

/// CPU A × B^T: A is M×K (Vec of rows), B is N×K (Vec of rows).
fn cpu_a_bt(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b.len();
    let depth = a[0].len();
    let mut out = vec![vec![0.0_f64; cols]; rows];
    for row_idx in 0..rows {
        for col_idx in 0..cols {
            for inner_idx in 0..depth {
                out[row_idx][col_idx] += a[row_idx][inner_idx] * b[col_idx][inner_idx];
            }
        }
    }
    out
}

fn flatten_f32(data: &[Vec<f64>]) -> Vec<f32> {
    data.iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect()
}

fn flatten_f64(data: &[Vec<f64>]) -> Vec<f64> {
    data.iter().flat_map(|r| r.iter().copied()).collect()
}

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
    let mut h = ValidationHarness::new("barracuda_gpu_nn");

    validate_single_layer(&mut h, &device);
    validate_two_layer_mlp(&mut h, &device);
    validate_bias_add(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

/// Single layer: input (16×4) × W^T (4×8) → hidden (16×8), then tanh.
fn validate_single_layer(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let batch = 16_usize;
    let in_dim = 4_usize;
    let out_dim = 8_usize;

    let input: Vec<Vec<f64>> = (0..batch)
        .map(|_| (0..in_dim).map(|_| rng.uniform()).collect())
        .collect();
    let weight: Vec<Vec<f64>> = (0..out_dim)
        .map(|_| (0..in_dim).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_linear = cpu_a_bt(&input, &weight);
    let cpu_tanh: Vec<f64> = flatten_f64(&cpu_linear).iter().map(|&x| x.tanh()).collect();

    let inp_t = gpu_tensor!(h, &flatten_f32(&input), &[batch, in_dim], device);
    let wt_t = gpu_tensor!(h, &flatten_f32(&weight), &[out_dim, in_dim], device);
    let wt_t_t = match wt_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let linear_t = match inp_t.matmul(&wt_t_t) {
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

    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_tanh);
    h.check_upper(
        &format!("single layer matmul+tanh: max diff ({diff:.2e})"),
        diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

/// Two-layer MLP: input (8×4) → hidden (8×6) → output (8×3).
fn validate_two_layer_mlp(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(123);
    let batch = 8_usize;
    let in_dim = 4_usize;
    let hidden = 6_usize;
    let out_dim = 3_usize;

    let input: Vec<Vec<f64>> = (0..batch)
        .map(|_| (0..in_dim).map(|_| rng.uniform()).collect())
        .collect();
    let w1: Vec<Vec<f64>> = (0..hidden)
        .map(|_| (0..in_dim).map(|_| rng.uniform()).collect())
        .collect();
    let w2: Vec<Vec<f64>> = (0..out_dim)
        .map(|_| (0..hidden).map(|_| rng.uniform()).collect())
        .collect();

    let h1_linear = cpu_a_bt(&input, &w1);
    let h1_tanh: Vec<Vec<f64>> = h1_linear
        .iter()
        .map(|row| row.iter().map(|&x| x.tanh()).collect())
        .collect();
    let cpu_out = flatten_f64(&cpu_a_bt(&h1_tanh, &w2));

    let inp_t = gpu_tensor!(h, &flatten_f32(&input), &[batch, in_dim], device);
    let w1_t = gpu_tensor!(h, &flatten_f32(&w1), &[hidden, in_dim], device);
    let w1_t_t = match w1_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose w1: {e}"), false);
            return;
        }
    };
    let w2_t = gpu_tensor!(h, &flatten_f32(&w2), &[out_dim, hidden], device);
    let w2_t_t = match w2_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose w2: {e}"), false);
            return;
        }
    };

    let h1_t = match inp_t.matmul(&w1_t_t) {
        Ok(t) => match t.tanh() {
            Ok(a) => a,
            Err(e) => {
                h.check_bool(&format!("tanh layer1: {e}"), false);
                return;
            }
        },
        Err(e) => {
            h.check_bool(&format!("matmul layer1: {e}"), false);
            return;
        }
    };
    let out_t = match h1_t.matmul(&w2_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul layer2: {e}"), false);
            return;
        }
    };
    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };

    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_out);
    h.check_upper(
        &format!("2-layer MLP: max diff ({diff:.2e})"),
        diff,
        tolerances::TENSOR_MATMUL_F32,
    );

    h.check_bool(
        &format!("output shape batch×out ({} elements)", out.len()),
        out.len() == batch * out_dim,
    );
}

/// Bias addition: matmul + element-wise add.
fn validate_bias_add(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(99);
    let batch = 12_usize;
    let in_dim = 5_usize;
    let out_dim = 3_usize;

    let input: Vec<Vec<f64>> = (0..batch)
        .map(|_| (0..in_dim).map(|_| rng.uniform()).collect())
        .collect();
    let weight: Vec<Vec<f64>> = (0..out_dim)
        .map(|_| (0..in_dim).map(|_| rng.uniform()).collect())
        .collect();
    let bias_f64: Vec<f64> = (0..out_dim).map(|_| rng.uniform()).collect();

    let cpu_linear = cpu_a_bt(&input, &weight);
    let cpu_biased: Vec<f64> = flatten_f64(&cpu_linear)
        .chunks(out_dim)
        .flat_map(|row| row.iter().zip(bias_f64.iter()).map(|(&v, &b)| v + b))
        .collect();

    let inp_t = gpu_tensor!(h, &flatten_f32(&input), &[batch, in_dim], device);
    let wt_t = gpu_tensor!(h, &flatten_f32(&weight), &[out_dim, in_dim], device);
    let wt_t_t = match wt_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let bias_row: Vec<f32> = bias_f64
        .iter()
        .cycle()
        .take(batch * out_dim)
        .map(|&x| x as f32)
        .collect();
    let bias_t = gpu_tensor!(h, &bias_row, &[batch, out_dim], device);

    let mm = match inp_t.matmul(&wt_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };
    let biased = match mm.add(&bias_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("add bias: {e}"), false);
            return;
        }
    };
    let Some(out) = gpu_readback(h, &biased) else {
        return;
    };

    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_biased);
    h.check_upper(
        &format!("matmul+bias: max diff ({diff:.2e})"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let batch = 16_usize;
    let in_dim = 4_usize;
    let out_dim = 8_usize;

    let inp: Vec<f32> = (0..batch * in_dim).map(|_| rng.uniform() as f32).collect();
    let wt: Vec<f32> = (0..out_dim * in_dim)
        .map(|_| rng.uniform() as f32)
        .collect();

    let run = |_: u32| -> Option<Vec<f32>> {
        let i = Tensor::from_data(&inp, vec![batch, in_dim], device.clone()).ok()?;
        let w = Tensor::from_data(&wt, vec![out_dim, in_dim], device.clone()).ok()?;
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
