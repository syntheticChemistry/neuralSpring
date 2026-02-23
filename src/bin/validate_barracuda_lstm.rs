// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` Tensor validation: LSTM weather forecast (Study 004).
//!
//! Tests LSTM gate computations using barracuda Tensor. Validates forget, input,
//! cell candidate, output gates, cell state update, and hidden state against
//! CPU f64 reference. Same gate math as Exp 003; multi-step sequencing is
//! covered by `validate_barracuda_sequence`.
//!
//! ## S-14 workaround
//!
//! Uses A × B^T pattern: transpose weights before matmul.
//!
//! ## S-15 workaround
//!
//! All data uses `rng.uniform()` (positive-only).
//!
//! ## Provenance
//!
//! Python baseline: `control/lstm_weather/lstm_era5.py`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::manual_let_else,
    clippy::similar_names,
    clippy::single_match_else,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::too_many_lines
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::require;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{max_abs_diff_gpu_vs_cpu, ValidationHarness};
use std::sync::Arc;

const BATCH: usize = 4;
const INPUT_DIM: usize = 5;
const HIDDEN_DIM: usize = 8;

fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn tensor(
    data: &[f32],
    shape: Vec<usize>,
    device: &Arc<WgpuDevice>,
) -> Result<Tensor, barracuda::error::BarracudaError> {
    Tensor::from_data(data, shape, device.clone())
}

/// CPU A × B^T
fn cpu_matmul_a_bt(
    a: &[f64],
    shape_a: (usize, usize),
    b: &[f64],
    shape_b: (usize, usize),
) -> Vec<f64> {
    let (m, k) = shape_a;
    let (n, _) = shape_b;
    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for d in 0..k {
                sum += a[i * k + d] * b[j * k + d];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

#[tokio::main]
async fn main() {
    let Ok(gpu) = Gpu::new().await else {
        eprintln!("  0/0 checks — skipping gracefully");
        std::process::exit(0);
    };
    eprintln!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let device = gpu.wgpu_device().clone();
    let harness_name = format!("barracuda_lstm[{}]", gpu.adapter_name);
    let mut h = ValidationHarness::new(&harness_name);

    validate_lstm_pipeline(&mut h, &device);

    h.finish();
}

fn validate_lstm_pipeline(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = Rng::new(42);
    let gate_dim = INPUT_DIM + HIDDEN_DIM;

    let w_f: Vec<f64> = (0..HIDDEN_DIM * gate_dim).map(|_| rng.uniform()).collect();
    let b_f: Vec<f64> = (0..HIDDEN_DIM).map(|_| rng.uniform()).collect();
    let w_i: Vec<f64> = (0..HIDDEN_DIM * gate_dim).map(|_| rng.uniform()).collect();
    let b_i: Vec<f64> = (0..HIDDEN_DIM).map(|_| rng.uniform()).collect();
    let w_c: Vec<f64> = (0..HIDDEN_DIM * gate_dim).map(|_| rng.uniform()).collect();
    let b_c: Vec<f64> = (0..HIDDEN_DIM).map(|_| rng.uniform()).collect();
    let w_o: Vec<f64> = (0..HIDDEN_DIM * gate_dim).map(|_| rng.uniform()).collect();
    let b_o: Vec<f64> = (0..HIDDEN_DIM).map(|_| rng.uniform()).collect();

    let c_prev: Vec<f64> = (0..BATCH * HIDDEN_DIM).map(|_| rng.uniform()).collect();

    let x0: Vec<f64> = (0..BATCH * INPUT_DIM).map(|_| rng.uniform()).collect();
    let h0: Vec<f64> = (0..BATCH * HIDDEN_DIM).map(|_| rng.uniform()).collect();
    let concat0: Vec<f64> = (0..BATCH)
        .flat_map(|i| {
            let x_row = &x0[i * INPUT_DIM..(i + 1) * INPUT_DIM];
            let h_row = &h0[i * HIDDEN_DIM..(i + 1) * HIDDEN_DIM];
            x_row.iter().chain(h_row).copied().collect::<Vec<_>>()
        })
        .collect();

    let raw_f0 = cpu_matmul_a_bt(&concat0, (BATCH, gate_dim), &w_f, (HIDDEN_DIM, gate_dim));
    let forget0_cpu: Vec<f64> = raw_f0
        .iter()
        .enumerate()
        .map(|(idx, &v)| sigmoid_f64(v + b_f[idx % HIDDEN_DIM]))
        .collect();
    let raw_i0 = cpu_matmul_a_bt(&concat0, (BATCH, gate_dim), &w_i, (HIDDEN_DIM, gate_dim));
    let input0_cpu: Vec<f64> = raw_i0
        .iter()
        .enumerate()
        .map(|(idx, &v)| sigmoid_f64(v + b_i[idx % HIDDEN_DIM]))
        .collect();
    let raw_c0 = cpu_matmul_a_bt(&concat0, (BATCH, gate_dim), &w_c, (HIDDEN_DIM, gate_dim));
    let candidate0_cpu: Vec<f64> = raw_c0
        .iter()
        .enumerate()
        .map(|(idx, &v)| (v + b_c[idx % HIDDEN_DIM]).tanh())
        .collect();
    let raw_o0 = cpu_matmul_a_bt(&concat0, (BATCH, gate_dim), &w_o, (HIDDEN_DIM, gate_dim));
    let output0_cpu: Vec<f64> = raw_o0
        .iter()
        .enumerate()
        .map(|(idx, &v)| sigmoid_f64(v + b_o[idx % HIDDEN_DIM]))
        .collect();

    let forget_mul_c: Vec<f64> = forget0_cpu
        .iter()
        .zip(c_prev.iter())
        .map(|(&f, &c)| f * c)
        .collect();
    let input_mul_cand: Vec<f64> = input0_cpu
        .iter()
        .zip(candidate0_cpu.iter())
        .map(|(&i, &c)| i * c)
        .collect();
    let cell0_cpu: Vec<f64> = forget_mul_c
        .iter()
        .zip(input_mul_cand.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    let tanh_cell: Vec<f64> = cell0_cpu.iter().map(|&c| c.tanh()).collect();
    let hidden0_cpu: Vec<f64> = output0_cpu
        .iter()
        .zip(tanh_cell.iter())
        .map(|(&o, &t)| o * t)
        .collect();

    let concat0_f32: Vec<f32> = concat0.iter().map(|&x| x as f32).collect();
    let w_f_f32: Vec<f32> = w_f.iter().map(|&x| x as f32).collect();
    let b_f_f32: Vec<f32> = b_f.iter().map(|&x| x as f32).collect();
    let bias_broadcast: Vec<f32> = (0..BATCH).flat_map(|_| b_f_f32.iter().copied()).collect();

    let concat_t = require!(
        h,
        tensor(&concat0_f32, vec![BATCH, gate_dim], device),
        "Tensor::from_data concat"
    );
    let w_f_t = require!(
        h,
        tensor(&w_f_f32, vec![HIDDEN_DIM, gate_dim], device),
        "Tensor::from_data W_f"
    );
    let w_f_t_t = require!(h, w_f_t.transpose(), "W_f transpose");
    let raw_f_t = require!(h, concat_t.matmul(&w_f_t_t), "matmul forget");
    let bias_t = require!(
        h,
        tensor(&bias_broadcast, vec![BATCH, HIDDEN_DIM], device),
        "Tensor::from_data bias"
    );
    let forget_t = require!(h, raw_f_t.add(&bias_t), "add bias");
    let forget_act = require!(h, forget_t.sigmoid(), "sigmoid");
    let forget_out = require!(h, forget_act.to_vec(), "readback forget");

    let diff_single = max_abs_diff_gpu_vs_cpu(&forget_out, &forget0_cpu);
    h.check_upper(
        &format!("single-step forget gate (diff={diff_single:.2e})"),
        diff_single,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let w_i_f32: Vec<f32> = w_i.iter().map(|&x| x as f32).collect();
    let b_i_f32: Vec<f32> = b_i.iter().map(|&x| x as f32).collect();
    let bias_i_bc: Vec<f32> = (0..BATCH).flat_map(|_| b_i_f32.iter().copied()).collect();
    let concat_t = require!(
        h,
        tensor(&concat0_f32, vec![BATCH, gate_dim], device),
        "Tensor concat for input gate"
    );
    let w_i_t = require!(
        h,
        tensor(&w_i_f32, vec![HIDDEN_DIM, gate_dim], device),
        "W_i"
    );
    let w_i_tt = require!(h, w_i_t.transpose(), "W_i^T");
    let raw_i_t = require!(h, concat_t.matmul(&w_i_tt), "matmul input");
    let bias_i_t = require!(
        h,
        tensor(&bias_i_bc, vec![BATCH, HIDDEN_DIM], device),
        "bias_i"
    );
    let input_t = require!(h, raw_i_t.add(&bias_i_t), "add bias input");
    let input_act = require!(h, input_t.sigmoid(), "sigmoid input");
    let input_out = require!(h, input_act.to_vec(), "readback input");
    let diff_i = max_abs_diff_gpu_vs_cpu(&input_out, &input0_cpu);
    h.check_upper(
        &format!("single-step input gate (diff={diff_i:.2e})"),
        diff_i,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let w_c_f32: Vec<f32> = w_c.iter().map(|&x| x as f32).collect();
    let b_c_f32: Vec<f32> = b_c.iter().map(|&x| x as f32).collect();
    let bias_c_bc: Vec<f32> = (0..BATCH).flat_map(|_| b_c_f32.iter().copied()).collect();
    let concat_t2 = require!(
        h,
        tensor(&concat0_f32, vec![BATCH, gate_dim], device),
        "Tensor concat for candidate"
    );
    let w_c_t = require!(
        h,
        tensor(&w_c_f32, vec![HIDDEN_DIM, gate_dim], device),
        "W_c"
    );
    let w_c_tt = require!(h, w_c_t.transpose(), "W_c^T");
    let raw_c_t = require!(h, concat_t2.matmul(&w_c_tt), "matmul candidate");
    let bias_c_t = require!(
        h,
        tensor(&bias_c_bc, vec![BATCH, HIDDEN_DIM], device),
        "bias_c"
    );
    let cand_t = require!(h, raw_c_t.add(&bias_c_t), "add bias cand");
    let cand_act = require!(h, cand_t.tanh(), "tanh candidate");
    let cand_out = require!(h, cand_act.to_vec(), "readback candidate");
    let diff_c = max_abs_diff_gpu_vs_cpu(&cand_out, &candidate0_cpu);
    h.check_upper(
        &format!("single-step cell candidate (diff={diff_c:.2e})"),
        diff_c,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let w_o_f32: Vec<f32> = w_o.iter().map(|&x| x as f32).collect();
    let b_o_f32: Vec<f32> = b_o.iter().map(|&x| x as f32).collect();
    let bias_o_bc: Vec<f32> = (0..BATCH).flat_map(|_| b_o_f32.iter().copied()).collect();
    let concat_t3 = require!(
        h,
        tensor(&concat0_f32, vec![BATCH, gate_dim], device),
        "Tensor concat for output gate"
    );
    let w_o_t = require!(
        h,
        tensor(&w_o_f32, vec![HIDDEN_DIM, gate_dim], device),
        "W_o"
    );
    let w_o_tt = require!(h, w_o_t.transpose(), "W_o^T");
    let raw_o_t = require!(h, concat_t3.matmul(&w_o_tt), "matmul output");
    let bias_o_t = require!(
        h,
        tensor(&bias_o_bc, vec![BATCH, HIDDEN_DIM], device),
        "bias_o"
    );
    let output_t = require!(h, raw_o_t.add(&bias_o_t), "add bias output");
    let output_act = require!(h, output_t.sigmoid(), "sigmoid output");
    let output_out = require!(h, output_act.to_vec(), "readback output");
    let diff_o = max_abs_diff_gpu_vs_cpu(&output_out, &output0_cpu);
    h.check_upper(
        &format!("single-step output gate (diff={diff_o:.2e})"),
        diff_o,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let c_prev_f32: Vec<f32> = c_prev.iter().map(|&x| x as f32).collect();
    let forget_vals_t = require!(
        h,
        tensor(&forget_out, vec![BATCH, HIDDEN_DIM], device),
        "Tensor forget vals"
    );
    let c_prev_t = require!(
        h,
        tensor(&c_prev_f32, vec![BATCH, HIDDEN_DIM], device),
        "Tensor c_prev"
    );
    let input_vals_t = require!(
        h,
        tensor(&input_out, vec![BATCH, HIDDEN_DIM], device),
        "Tensor input vals"
    );
    let cand_vals_t = require!(
        h,
        tensor(&cand_out, vec![BATCH, HIDDEN_DIM], device),
        "Tensor cand vals"
    );
    let forget_mul_c_t = require!(h, forget_vals_t.mul(&c_prev_t), "forget*c");
    let input_mul_cand_t = require!(h, input_vals_t.mul(&cand_vals_t), "input*cand");
    let cell_t = require!(h, forget_mul_c_t.add(&input_mul_cand_t), "cell");
    let cell_out = require!(h, cell_t.to_vec(), "readback cell");
    let diff_cell = max_abs_diff_gpu_vs_cpu(&cell_out, &cell0_cpu);
    h.check_upper(
        &format!("cell state update (diff={diff_cell:.2e})"),
        diff_cell,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );

    let output_vals_t = require!(
        h,
        tensor(&output_out, vec![BATCH, HIDDEN_DIM], device),
        "Tensor output vals"
    );
    let cell_tanh_t = require!(h, cell_t.tanh(), "tanh(cell)");
    let hidden_t = require!(h, output_vals_t.mul(&cell_tanh_t), "hidden");
    let hidden_out = require!(h, hidden_t.to_vec(), "readback hidden");
    let diff_h = max_abs_diff_gpu_vs_cpu(&hidden_out, &hidden0_cpu);
    h.check_upper(
        &format!("hidden state (diff={diff_h:.2e})"),
        diff_h,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    validate_determinism(h, device);
}

fn validate_determinism(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = Rng::new(123);
    let gate_dim = INPUT_DIM + HIDDEN_DIM;

    let concat: Vec<f32> = (0..BATCH * gate_dim)
        .map(|_| rng.uniform() as f32)
        .collect();
    let w: Vec<f32> = (0..HIDDEN_DIM * gate_dim)
        .map(|_| rng.uniform() as f32)
        .collect();

    let run = || -> Option<Vec<f32>> {
        let c = Tensor::from_data(&concat, vec![BATCH, gate_dim], device.clone()).ok()?;
        let wt = Tensor::from_data(&w, vec![HIDDEN_DIM, gate_dim], device.clone()).ok()?;
        let wt_t = wt.transpose().ok()?;
        let mm = c.matmul(&wt_t).ok()?;
        let act = mm.sigmoid().ok()?;
        act.to_vec().ok()
    };

    let Some(r1) = run() else {
        h.check_bool("determinism run 1 failed", false);
        return;
    };
    let Some(r2) = run() else {
        h.check_bool("determinism run 2 failed", false);
        return;
    };

    let identical = r1
        .iter()
        .zip(r2.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    h.check_bool("determinism: two runs bit-identical", identical);
}
