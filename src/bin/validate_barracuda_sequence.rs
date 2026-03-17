// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` Tensor validation: LSTM sequence model (Exp 003).
//!
//! Tests LSTM gate computations using barracuda Tensor matmul + sigmoid + tanh.
//! LSTM gates: forget = `σ(W_f×x+b_f)`, input = `σ(W_i×x+b_i)`, candidate = `tanh(W_c×x+b_c)`,
//! output = `σ(W_o×x+b_o)`. Cell: `c_t` = forget×c_{t-1} + input×candidate. Hidden: `h_t` = `output×tanh(c_t)`.
//!
//! ## S-14 workaround
//!
//! Uses A × B^T pattern: transpose weights before matmul.
//!
//! ## S-15 workaround
//!
//! All data uses `rng.uniform()` (positive-only) to avoid matmul hangs.
//!
//! ## Provenance
//!
//! CPU baseline: `neural_spring::sequence::lstm_cell`.

#![expect(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "validation binary"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::require;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{max_abs_diff_gpu_vs_cpu, ValidationHarness};
use std::sync::Arc;

const BATCH: usize = 8;
const INPUT_DIM: usize = 6;
const HIDDEN_DIM: usize = 4;

use neural_spring::primitives::sigmoid as sigmoid_f64;

#[tokio::main]
async fn main() {
    let Ok(gpu) = Gpu::new().await else {
        neural_spring::validation::exit_no_gpu();
    };
    println!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let device = gpu.wgpu_device().clone();
    let harness_name = format!("barracuda_sequence[{}]", gpu.adapter_name);
    let mut h = ValidationHarness::new(&harness_name);

    validate_lstm_gates(&mut h, &device);

    h.finish();
}

fn tensor(
    data: &[f32],
    shape: Vec<usize>,
    device: &Arc<WgpuDevice>,
) -> Result<Tensor, barracuda::error::BarracudaError> {
    Tensor::from_data(data, shape, device.clone())
}

/// CPU A × B^T: A (`rows_a` × depth), B (`rows_b` × depth) → output (`rows_a` × `rows_b`)
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

fn validate_lstm_gates(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = Rng::new(42);
    let gate_dim = INPUT_DIM + HIDDEN_DIM;

    let x: Vec<f64> = (0..BATCH * INPUT_DIM).map(|_| rng.uniform()).collect();
    let h_prev: Vec<f64> = (0..BATCH * HIDDEN_DIM).map(|_| rng.uniform()).collect();
    let c_prev: Vec<f64> = (0..BATCH * HIDDEN_DIM).map(|_| rng.uniform()).collect();

    let concat: Vec<f64> = (0..BATCH)
        .flat_map(|i| {
            let x_row = &x[i * INPUT_DIM..(i + 1) * INPUT_DIM];
            let h_row = &h_prev[i * HIDDEN_DIM..(i + 1) * HIDDEN_DIM];
            x_row.iter().chain(h_row).copied().collect::<Vec<_>>()
        })
        .collect();

    let w_f: Vec<f64> = (0..HIDDEN_DIM * gate_dim).map(|_| rng.uniform()).collect();
    let b_f: Vec<f64> = (0..HIDDEN_DIM).map(|_| rng.uniform()).collect();
    let w_i: Vec<f64> = (0..HIDDEN_DIM * gate_dim).map(|_| rng.uniform()).collect();
    let b_i: Vec<f64> = (0..HIDDEN_DIM).map(|_| rng.uniform()).collect();
    let w_c: Vec<f64> = (0..HIDDEN_DIM * gate_dim).map(|_| rng.uniform()).collect();
    let b_c: Vec<f64> = (0..HIDDEN_DIM).map(|_| rng.uniform()).collect();
    let w_o: Vec<f64> = (0..HIDDEN_DIM * gate_dim).map(|_| rng.uniform()).collect();
    let b_o: Vec<f64> = (0..HIDDEN_DIM).map(|_| rng.uniform()).collect();

    let raw_f = cpu_matmul_a_bt(&concat, (BATCH, gate_dim), &w_f, (HIDDEN_DIM, gate_dim));
    let forget_cpu: Vec<f64> = raw_f
        .iter()
        .enumerate()
        .map(|(idx, &v)| sigmoid_f64(v + b_f[idx % HIDDEN_DIM]))
        .collect();

    let raw_i = cpu_matmul_a_bt(&concat, (BATCH, gate_dim), &w_i, (HIDDEN_DIM, gate_dim));
    let input_cpu: Vec<f64> = raw_i
        .iter()
        .enumerate()
        .map(|(idx, &v)| sigmoid_f64(v + b_i[idx % HIDDEN_DIM]))
        .collect();

    let raw_c = cpu_matmul_a_bt(&concat, (BATCH, gate_dim), &w_c, (HIDDEN_DIM, gate_dim));
    let candidate_cpu: Vec<f64> = raw_c
        .iter()
        .enumerate()
        .map(|(idx, &v)| (v + b_c[idx % HIDDEN_DIM]).tanh())
        .collect();

    let raw_o = cpu_matmul_a_bt(&concat, (BATCH, gate_dim), &w_o, (HIDDEN_DIM, gate_dim));
    let output_cpu: Vec<f64> = raw_o
        .iter()
        .enumerate()
        .map(|(idx, &v)| sigmoid_f64(v + b_o[idx % HIDDEN_DIM]))
        .collect();

    let forget_mul_c: Vec<f64> = forget_cpu
        .iter()
        .zip(c_prev.iter())
        .map(|(&f, &c)| f * c)
        .collect();
    let input_mul_cand: Vec<f64> = input_cpu
        .iter()
        .zip(candidate_cpu.iter())
        .map(|(&i, &c)| i * c)
        .collect();
    let cell_cpu: Vec<f64> = forget_mul_c
        .iter()
        .zip(input_mul_cand.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    let tanh_cell: Vec<f64> = cell_cpu.iter().map(|&c| c.tanh()).collect();
    let hidden_cpu: Vec<f64> = output_cpu
        .iter()
        .zip(tanh_cell.iter())
        .map(|(&o, &t)| o * t)
        .collect();

    let concat_f32: Vec<f32> = concat.iter().map(|&x| x as f32).collect();
    let w_f_f32: Vec<f32> = w_f.iter().map(|&x| x as f32).collect();
    let b_f_f32: Vec<f32> = b_f.iter().map(|&x| x as f32).collect();
    let bias_broadcast_f: Vec<f32> = (0..BATCH).flat_map(|_| b_f_f32.iter().copied()).collect();

    let concat_t = require!(
        h,
        tensor(&concat_f32, vec![BATCH, gate_dim], device),
        "Tensor::from_data concat"
    );
    let w_f_t = require!(
        h,
        tensor(&w_f_f32, vec![HIDDEN_DIM, gate_dim], device),
        "Tensor::from_data W_f"
    );
    let w_f_t_t = require!(h, w_f_t.transpose(), "W_f transpose");
    let raw_f_t = require!(h, concat_t.matmul(&w_f_t_t), "matmul forget");
    let bias_f_t = require!(
        h,
        tensor(&bias_broadcast_f, vec![BATCH, HIDDEN_DIM], device),
        "Tensor::from_data bias_f"
    );
    let forget_t = require!(h, raw_f_t.add(&bias_f_t), "add bias forget");
    let forget_act = require!(h, forget_t.sigmoid(), "sigmoid forget");
    let forget_out = require!(h, forget_act.to_vec(), "readback forget");

    let diff_f = max_abs_diff_gpu_vs_cpu(&forget_out, &forget_cpu);
    h.check_upper(
        &format!("forget gate accuracy (diff={diff_f:.2e})"),
        diff_f,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let w_i_f32: Vec<f32> = w_i.iter().map(|&x| x as f32).collect();
    let b_i_f32: Vec<f32> = b_i.iter().map(|&x| x as f32).collect();
    let bias_broadcast_i: Vec<f32> = (0..BATCH).flat_map(|_| b_i_f32.iter().copied()).collect();

    let concat_t2 = require!(
        h,
        tensor(&concat_f32, vec![BATCH, gate_dim], device),
        "Tensor::from_data concat"
    );
    let w_i_t = require!(
        h,
        tensor(&w_i_f32, vec![HIDDEN_DIM, gate_dim], device),
        "Tensor::from_data W_i"
    );
    let w_i_t_t = require!(h, w_i_t.transpose(), "W_i transpose");
    let raw_i_t = require!(h, concat_t2.matmul(&w_i_t_t), "matmul input gate");
    let bias_i_t = require!(
        h,
        tensor(&bias_broadcast_i, vec![BATCH, HIDDEN_DIM], device),
        "Tensor::from_data bias_i"
    );
    let input_t = require!(h, raw_i_t.add(&bias_i_t), "add bias input");
    let input_act = require!(h, input_t.sigmoid(), "sigmoid input");
    let input_out = require!(h, input_act.to_vec(), "readback input");

    let diff_i = max_abs_diff_gpu_vs_cpu(&input_out, &input_cpu);
    h.check_upper(
        &format!("input gate accuracy (diff={diff_i:.2e})"),
        diff_i,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let w_c_f32: Vec<f32> = w_c.iter().map(|&x| x as f32).collect();
    let b_c_f32: Vec<f32> = b_c.iter().map(|&x| x as f32).collect();
    let bias_broadcast_c: Vec<f32> = (0..BATCH).flat_map(|_| b_c_f32.iter().copied()).collect();

    let concat_t3 = require!(
        h,
        tensor(&concat_f32, vec![BATCH, gate_dim], device),
        "Tensor::from_data concat"
    );
    let w_c_t = require!(
        h,
        tensor(&w_c_f32, vec![HIDDEN_DIM, gate_dim], device),
        "Tensor::from_data W_c"
    );
    let w_c_t_t = require!(h, w_c_t.transpose(), "W_c transpose");
    let raw_c_t = require!(h, concat_t3.matmul(&w_c_t_t), "matmul candidate");
    let bias_c_t = require!(
        h,
        tensor(&bias_broadcast_c, vec![BATCH, HIDDEN_DIM], device),
        "Tensor::from_data bias_c"
    );
    let cand_t = require!(h, raw_c_t.add(&bias_c_t), "add bias candidate");
    let cand_act = require!(h, cand_t.tanh(), "tanh candidate");
    let cand_out = require!(h, cand_act.to_vec(), "readback candidate");

    let diff_c = max_abs_diff_gpu_vs_cpu(&cand_out, &candidate_cpu);
    h.check_upper(
        &format!("cell candidate accuracy (diff={diff_c:.2e})"),
        diff_c,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let w_o_f32: Vec<f32> = w_o.iter().map(|&x| x as f32).collect();
    let b_o_f32: Vec<f32> = b_o.iter().map(|&x| x as f32).collect();
    let bias_broadcast_o: Vec<f32> = (0..BATCH).flat_map(|_| b_o_f32.iter().copied()).collect();

    let concat_t4 = require!(
        h,
        tensor(&concat_f32, vec![BATCH, gate_dim], device),
        "Tensor::from_data concat"
    );
    let w_o_t = require!(
        h,
        tensor(&w_o_f32, vec![HIDDEN_DIM, gate_dim], device),
        "Tensor::from_data W_o"
    );
    let w_o_t_t = require!(h, w_o_t.transpose(), "W_o transpose");
    let raw_o_t = require!(h, concat_t4.matmul(&w_o_t_t), "matmul output gate");
    let bias_o_t = require!(
        h,
        tensor(&bias_broadcast_o, vec![BATCH, HIDDEN_DIM], device),
        "Tensor::from_data bias_o"
    );
    let output_t = require!(h, raw_o_t.add(&bias_o_t), "add bias output");
    let output_act = require!(h, output_t.sigmoid(), "sigmoid output");
    let output_out = require!(h, output_act.to_vec(), "readback output");

    let diff_o = max_abs_diff_gpu_vs_cpu(&output_out, &output_cpu);
    h.check_upper(
        &format!("output gate accuracy (diff={diff_o:.2e})"),
        diff_o,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let c_prev_f32: Vec<f32> = c_prev.iter().map(|&x| x as f32).collect();

    let forget_vals_t = require!(
        h,
        tensor(&forget_out, vec![BATCH, HIDDEN_DIM], device),
        "Tensor forget"
    );
    let c_prev_t = require!(
        h,
        tensor(&c_prev_f32, vec![BATCH, HIDDEN_DIM], device),
        "Tensor c_prev"
    );
    let input_vals_t = require!(
        h,
        tensor(&input_out, vec![BATCH, HIDDEN_DIM], device),
        "Tensor input"
    );
    let cand_vals_t = require!(
        h,
        tensor(&cand_out, vec![BATCH, HIDDEN_DIM], device),
        "Tensor candidate"
    );
    let forget_mul_c_t = require!(h, forget_vals_t.mul(&c_prev_t), "forget * c_prev");
    let input_mul_cand_t = require!(h, input_vals_t.mul(&cand_vals_t), "input * candidate");
    let cell_t = require!(h, forget_mul_c_t.add(&input_mul_cand_t), "cell update");
    let cell_out = require!(h, cell_t.to_vec(), "readback cell");
    let diff_cell = max_abs_diff_gpu_vs_cpu(&cell_out, &cell_cpu);
    h.check_upper(
        &format!("cell state update (diff={diff_cell:.2e})"),
        diff_cell,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );

    let output_vals_t = require!(
        h,
        tensor(&output_out, vec![BATCH, HIDDEN_DIM], device),
        "Tensor output"
    );
    let cell_tanh = require!(h, cell_t.tanh(), "tanh(cell)");
    let hidden_t = require!(
        h,
        output_vals_t.mul(&cell_tanh),
        "hidden = output * tanh(cell)"
    );
    let hidden_out = require!(h, hidden_t.to_vec(), "readback hidden");
    let diff_h = max_abs_diff_gpu_vs_cpu(&hidden_out, &hidden_cpu);
    h.check_upper(
        &format!("hidden state (diff={diff_h:.2e})"),
        diff_h,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    validate_determinism(h, device);
}

fn validate_determinism(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let mut rng = Rng::new(99);
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
