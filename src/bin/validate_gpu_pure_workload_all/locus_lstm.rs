// SPDX-License-Identifier: AGPL-3.0-or-later

//! Locus variance (meta-population, paper 025) and LSTM glucose path via `Tensor::matmul` (026).

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::LocusVarianceGpu;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::primitives::sigmoid_f32;
use neural_spring::rng::Rng;
use neural_spring::sequence::{LstmWeights, lstm_cell};
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, output_buf, storage_buf};
use std::sync::Arc;

pub fn validate_locus_variance(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 4_usize;
    let n_loci = 16_usize;
    let mut rng = Rng::new(111);
    let freqs: Vec<f64> = (0..n_pops * n_loci).map(|_| rng.uniform()).collect();

    let cpu_var: Vec<f64> = (0..n_loci)
        .map(|l| {
            let vals: Vec<f64> = (0..n_pops).map(|p| freqs[p * n_loci + l]).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64
        })
        .collect();
    let cpu_mean = cpu_var.iter().sum::<f64>() / cpu_var.len() as f64;

    let op = LocusVarianceGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let freqs_buf = storage_buf(device, "lv_f", bytemuck::cast_slice(&freqs));
    let out_buf = output_buf(device, "lv_out", (n_loci * 8) as u64);

    op.dispatch(&freqs_buf, &out_buf, n_pops as u32, n_loci as u32);

    match gpu.read_buffer_f64(&out_buf, n_loci) {
        Ok(gpu_v) => {
            let gpu_mean = gpu_v.iter().sum::<f64>() / gpu_v.len() as f64;
            h.check_abs(
                "locus_var 4×16",
                gpu_mean,
                cpu_mean,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => h.check_bool(&format!("locus_var: {e}"), false),
    }
}

pub fn validate_lstm_glucose(h: &mut ValidationHarness, gpu: &Gpu) {
    let hs = 8_usize;
    let seq_len = 12_usize;
    let mut rng = Rng::new(42);

    let w_input: Vec<f64> = (0..4 * hs).map(|_| rng.normal() * 0.5).collect();
    let w_hidden: Vec<f64> = (0..4 * hs * hs).map(|_| rng.normal() * 0.1).collect();
    let mut b_input = vec![0.0_f64; 4 * hs];
    let b_hidden = vec![0.0_f64; 4 * hs];
    for b in &mut b_input[hs..2 * hs] {
        *b = 1.0;
    }

    let window: Vec<f64> = (0..seq_len).map(|_| rng.normal() * 0.5).collect();

    let lstm_w = LstmWeights {
        w_input: &w_input,
        w_hidden: &w_hidden,
        b_input: &b_input,
        b_hidden: &b_hidden,
        hidden_size: hs,
    };
    let mut h_state = vec![0.0_f64; hs];
    let mut c_state = vec![0.0_f64; hs];
    for val in &window {
        let (hn, cn) = lstm_cell(&[*val], &h_state, &c_state, &lstm_w);
        h_state = hn;
        c_state = cn;
    }
    let cpu_mean = h_state.iter().sum::<f64>() / h_state.len() as f64;

    let device = Arc::clone(gpu.wgpu_device());
    let gpu_result = gpu_lstm_forward(
        &window, &w_input, &w_hidden, &b_input, &b_hidden, hs, &device,
    );
    match gpu_result {
        Ok(gpu_hidden) => {
            let gpu_mean =
                gpu_hidden.iter().map(|&v| f64::from(v)).sum::<f64>() / gpu_hidden.len() as f64;
            h.check_abs(
                &format!("LSTM glucose {seq_len}×{hs}: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_LSTM_GLUCOSE_F32,
            );
        }
        Err(e) => h.check_bool(&format!("LSTM glucose: {e}"), false),
    }
}

fn gpu_lstm_forward(
    window: &[f64],
    w_input: &[f64],
    w_hidden: &[f64],
    b_input: &[f64],
    b_hidden: &[f64],
    hs: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f32>, String> {
    let wi_f32: Vec<f32> = w_input.iter().map(|&v| v as f32).collect();
    let wh_f32: Vec<f32> = w_hidden.iter().map(|&v| v as f32).collect();
    let bi_f32: Vec<f32> = b_input.iter().map(|&v| v as f32).collect();
    let bh_f32: Vec<f32> = b_hidden.iter().map(|&v| v as f32).collect();

    let wi_t = Tensor::from_data(&wi_f32, vec![4 * hs, 1], device.clone())
        .map_err(|e| format!("Wi: {e}"))?
        .transpose()
        .map_err(|e| format!("Wi^T: {e}"))?;
    let wh_t = Tensor::from_data(&wh_f32, vec![4 * hs, hs], device.clone())
        .map_err(|e| format!("Wh: {e}"))?
        .transpose()
        .map_err(|e| format!("Wh^T: {e}"))?;
    let bi_t = Tensor::from_data(&bi_f32, vec![1, 4 * hs], device.clone())
        .map_err(|e| format!("bi: {e}"))?;
    let bh_t = Tensor::from_data(&bh_f32, vec![1, 4 * hs], device.clone())
        .map_err(|e| format!("bh: {e}"))?;

    let mut h_vec = vec![0.0_f32; hs];
    let mut c_vec = vec![0.0_f32; hs];

    for &val in window {
        let x_t = Tensor::from_data(&[val as f32], vec![1, 1], device.clone())
            .map_err(|e| format!("x: {e}"))?;
        let h_t = Tensor::from_data(&h_vec, vec![1, hs], device.clone())
            .map_err(|e| format!("h: {e}"))?;

        let input_proj = x_t.matmul(&wi_t).map_err(|e| format!("x@Wi: {e}"))?;
        let hidden_proj = h_t.matmul(&wh_t).map_err(|e| format!("h@Wh: {e}"))?;
        let gates = input_proj
            .add(&hidden_proj)
            .map_err(|e| format!("add: {e}"))?
            .add(&bi_t)
            .map_err(|e| format!("add_bi: {e}"))?
            .add(&bh_t)
            .map_err(|e| format!("add_bh: {e}"))?;

        let g = gates.to_vec().map_err(|e| format!("readback: {e}"))?;

        let f_gate: Vec<f32> = g[..hs].iter().map(|v| sigmoid_f32(*v)).collect();
        let i_gate: Vec<f32> = g[hs..2 * hs].iter().map(|v| sigmoid_f32(*v)).collect();
        let g_gate: Vec<f32> = g[2 * hs..3 * hs].iter().map(|v| v.tanh()).collect();
        let o_gate: Vec<f32> = g[3 * hs..].iter().map(|v| sigmoid_f32(*v)).collect();

        c_vec = (0..hs)
            .map(|j| f_gate[j].mul_add(c_vec[j], i_gate[j] * g_gate[j]))
            .collect();
        h_vec = (0..hs).map(|j| o_gate[j] * c_vec[j].tanh()).collect();
    }

    Ok(h_vec)
}
