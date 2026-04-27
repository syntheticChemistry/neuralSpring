// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU workload validation for WDM surrogates + `coralForge` domains.
//!
//! Proves the evolution chain for WDM + structural biology:
//! ```text
//! Python baseline → Rust CPU → BarraCUDA GPU Tensor → Pure GPU (scalar readback)
//!                                                    → `ToadStool` streaming
//! ```
//!
//! Each domain dispatches its ML workload entirely through the `BarraCUDA`
//! Tensor API on GPU, reads back only scalar summaries, and compares
//! against CPU references. This is the "pure GPU" proof that the math
//! is truly portable to any hardware `ToadStool` manages.
//!
//! ## Structure
//!
//! WDM surrogates (transport MLP, EOS MLP, S(q,ω) LSTM, ESN reservoir)
//! live here. coralForge + `AlphaFold3` domains live in `coral_af3.rs`.
//!
//! ## Provenance
//!
//! CPU reference: neuralSpring library modules (Rust CPU).
//! GPU dispatch: `BarraCUDA` Tensor API via WGSL shaders.
//! Validated on: llvmpipe (software Vulkan) and RTX 4070 (hardware Vulkan).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    reason = "validation binary"
)]

mod coral_af3;

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::primitives::sigmoid_f32;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, exit_no_gpu};
use std::sync::Arc;
use std::time::Instant;

type Dev = Arc<WgpuDevice>;

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
        Err(_) => exit_no_gpu(),
    };

    let device: Dev = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("gpu_pure_wdm_coral");
    let t0 = Instant::now();

    // WDM surrogates
    validate_wdm_transport_mlp(&mut h, &device);
    validate_wdm_eos_mlp(&mut h, &device);
    validate_wdm_sqw_lstm(&mut h, &device);
    validate_wdm_esn_reservoir(&mut h, &device);

    // coralForge + AlphaFold3
    coral_af3::validate_coral_attention(&mut h, &device);
    coral_af3::validate_coral_trimul(&mut h, &device);
    coral_af3::validate_af3_pldt(&mut h, &device);
    coral_af3::validate_af3_pae(&mut h, &device);
    coral_af3::validate_af3_diffusion_forward(&mut h, &device);
    coral_af3::validate_af3_pairformer_ffn(&mut h, &device);
    coral_af3::validate_af3_pairformer_trimul(&mut h, &device);
    coral_af3::validate_determinism(&mut h, &device);

    let elapsed = t0.elapsed();
    println!(
        "\n  total pure-GPU WDM+coralForge time: {:.1}ms (11 domains + determinism)",
        elapsed.as_secs_f64() * 1000.0,
    );

    h.finish();
}

// ── Shared MLP helpers ──────────────────────────────────────────────────

pub(crate) fn gpu_mlp_forward(
    x: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    in_dim: usize,
    hid_dim: usize,
    out_dim: usize,
    device: &Dev,
) -> Result<Vec<f32>, String> {
    let x_t =
        Tensor::from_data(x, vec![1, in_dim], device.clone()).map_err(|e| format!("x: {e}"))?;
    let w1_t = Tensor::from_data(w1, vec![hid_dim, in_dim], device.clone())
        .map_err(|e| format!("w1: {e}"))?;
    let w1_t_t = w1_t.transpose().map_err(|e| format!("w1T: {e}"))?;
    let b1_t =
        Tensor::from_data(b1, vec![1, hid_dim], device.clone()).map_err(|e| format!("b1: {e}"))?;
    let w2_t = Tensor::from_data(w2, vec![out_dim, hid_dim], device.clone())
        .map_err(|e| format!("w2: {e}"))?;
    let w2_t_t = w2_t.transpose().map_err(|e| format!("w2T: {e}"))?;
    let b2_t =
        Tensor::from_data(b2, vec![1, out_dim], device.clone()).map_err(|e| format!("b2: {e}"))?;

    let h1 = x_t.matmul(&w1_t_t).map_err(|e| format!("mm1: {e}"))?;
    let h1b = h1.add(&b1_t).map_err(|e| format!("add1: {e}"))?;
    let h1a = h1b.relu().map_err(|e| format!("relu: {e}"))?;
    let h2 = h1a.matmul(&w2_t_t).map_err(|e| format!("mm2: {e}"))?;
    let out = h2.add(&b2_t).map_err(|e| format!("add2: {e}"))?;
    out.to_vec().map_err(|e| format!("readback: {e}"))
}

fn cpu_mlp_forward(
    x: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    in_dim: usize,
    hid_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut hidden = vec![0.0_f32; hid_dim];
    for i in 0..hid_dim {
        let mut sum = b1[i];
        for j in 0..in_dim {
            sum += w1[i * in_dim + j] * x[j];
        }
        hidden[i] = sum.max(0.0);
    }
    let mut output = vec![0.0_f32; out_dim];
    for i in 0..out_dim {
        let mut sum = b2[i];
        for j in 0..hid_dim {
            sum += w2[i * hid_dim + j] * hidden[j];
        }
        output[i] = sum;
    }
    output
}

// ═══════════════════════════════════════════════════════════════════
// 1. WDM Transport MLP (nW-01)
// ═══════════════════════════════════════════════════════════════════

fn validate_wdm_transport_mlp(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(101);
    let (in_d, hid_d, out_d) = (4, 16, 3);
    let w1: Vec<f32> = (0..hid_d * in_d)
        .map(|_| rng.normal() as f32 * 0.3)
        .collect();
    let b1: Vec<f32> = (0..hid_d).map(|_| rng.normal() as f32 * 0.1).collect();
    let w2: Vec<f32> = (0..out_d * hid_d)
        .map(|_| rng.normal() as f32 * 0.3)
        .collect();
    let b2: Vec<f32> = (0..out_d).map(|_| rng.normal() as f32 * 0.1).collect();
    let x: Vec<f32> = (0..in_d).map(|_| rng.normal() as f32).collect();

    let cpu = cpu_mlp_forward(&x, &w1, &b1, &w2, &b2, in_d, hid_d, out_d);
    let cpu_mean = cpu.iter().map(|v| f64::from(*v)).sum::<f64>() / out_d as f64;

    match gpu_mlp_forward(&x, &w1, &b1, &w2, &b2, in_d, hid_d, out_d, device) {
        Ok(gpu_out) => {
            let gpu_mean = gpu_out.iter().map(|v| f64::from(*v)).sum::<f64>() / out_d as f64;
            h.check_abs(
                "wdm_transport MLP: GPU mean vs CPU mean",
                gpu_mean,
                cpu_mean,
                tolerances::ML_MLP_F32,
            );
            h.check_bool(
                "wdm_transport MLP: all finite",
                gpu_out.iter().all(|v| v.is_finite()),
            );
        }
        Err(e) => h.check_bool(&format!("wdm_transport MLP: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. WDM EOS MLP (nW-02)
// ═══════════════════════════════════════════════════════════════════

fn validate_wdm_eos_mlp(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(202);
    let (in_d, hid_d, out_d) = (3, 32, 2);
    let w1: Vec<f32> = (0..hid_d * in_d)
        .map(|_| rng.normal() as f32 * 0.3)
        .collect();
    let b1: Vec<f32> = (0..hid_d).map(|_| rng.normal() as f32 * 0.1).collect();
    let w2: Vec<f32> = (0..out_d * hid_d)
        .map(|_| rng.normal() as f32 * 0.3)
        .collect();
    let b2: Vec<f32> = (0..out_d).map(|_| rng.normal() as f32 * 0.1).collect();
    let x: Vec<f32> = (0..in_d).map(|_| rng.normal() as f32).collect();

    let cpu = cpu_mlp_forward(&x, &w1, &b1, &w2, &b2, in_d, hid_d, out_d);

    match gpu_mlp_forward(&x, &w1, &b1, &w2, &b2, in_d, hid_d, out_d, device) {
        Ok(gpu_out) => {
            for i in 0..out_d {
                h.check_abs(
                    &format!("wdm_eos MLP out[{i}]: GPU vs CPU"),
                    f64::from(gpu_out[i]),
                    f64::from(cpu[i]),
                    tolerances::ML_MLP_F32,
                );
            }
        }
        Err(e) => h.check_bool(&format!("wdm_eos MLP: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. WDM S(q,ω) LSTM gates (nW-03) — gate projection on GPU
// ═══════════════════════════════════════════════════════════════════

fn validate_wdm_sqw_lstm(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(303);
    let hs = 8;
    let input_size = 1;

    let w_i: Vec<f32> = (0..4 * hs * input_size)
        .map(|_| rng.normal() as f32 * 0.2)
        .collect();
    let w_h: Vec<f32> = (0..4 * hs * hs)
        .map(|_| rng.normal() as f32 * 0.1)
        .collect();
    let b: Vec<f32> = (0..4 * hs).map(|_| rng.normal() as f32 * 0.05).collect();
    let x_val = 0.7_f32;
    let h_prev = vec![0.0_f32; hs];

    let cpu_gates: Vec<f32> = {
        let mut g = vec![0.0_f32; 4 * hs];
        for idx in 0..4 * hs {
            g[idx] = w_i[idx * input_size].mul_add(x_val, b[idx]);
            for j in 0..hs {
                g[idx] += w_h[idx * hs + j] * h_prev[j];
            }
        }
        g
    };

    let gpu_gates = (|| -> Result<Vec<f32>, String> {
        let x_t = Tensor::from_data(&[x_val], vec![1, 1], device.clone())
            .map_err(|e| format!("x: {e}"))?;
        let wi_t = Tensor::from_data(&w_i, vec![4 * hs, input_size], device.clone())
            .map_err(|e| format!("wi: {e}"))?;
        let wi_tt = wi_t.transpose().map_err(|e| format!("wiT: {e}"))?;
        let h_t = Tensor::from_data(&h_prev, vec![1, hs], device.clone())
            .map_err(|e| format!("h: {e}"))?;
        let wh_t = Tensor::from_data(&w_h, vec![4 * hs, hs], device.clone())
            .map_err(|e| format!("wh: {e}"))?;
        let wh_tt = wh_t.transpose().map_err(|e| format!("whT: {e}"))?;
        let b_t = Tensor::from_data(&b, vec![1, 4 * hs], device.clone())
            .map_err(|e| format!("b: {e}"))?;

        let ip = x_t.matmul(&wi_tt).map_err(|e| format!("xW: {e}"))?;
        let hp = h_t.matmul(&wh_tt).map_err(|e| format!("hW: {e}"))?;
        let s1 = ip.add(&hp).map_err(|e| format!("add1: {e}"))?;
        let gates = s1.add(&b_t).map_err(|e| format!("add_b: {e}"))?;
        gates.to_vec().map_err(|e| format!("readback: {e}"))
    })();

    match gpu_gates {
        Ok(gg) => {
            let max_diff = gg
                .iter()
                .zip(cpu_gates.iter())
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                .fold(0.0_f64, f64::max);
            h.check_bool(
                &format!("wdm_sqw LSTM gates: max_diff={max_diff:.2e}"),
                max_diff < tolerances::ML_MLP_F32,
            );

            let _f_gate: Vec<f32> = gg[..hs].iter().map(|v| sigmoid_f32(*v)).collect();
            let i_gate: Vec<f32> = gg[hs..2 * hs].iter().map(|v| sigmoid_f32(*v)).collect();
            let g_gate: Vec<f32> = gg[2 * hs..3 * hs].iter().map(|v| v.tanh()).collect();
            let o_gate: Vec<f32> = gg[3 * hs..].iter().map(|v| sigmoid_f32(*v)).collect();
            let c_new: Vec<f32> = (0..hs).map(|j| i_gate[j] * g_gate[j]).collect();
            let h_new: Vec<f32> = (0..hs).map(|j| o_gate[j] * c_new[j].tanh()).collect();
            h.check_bool(
                "wdm_sqw LSTM h_new bounded",
                h_new
                    .iter()
                    .all(|v| f64::from(v.abs()) <= 1.0 + tolerances::TENSOR_EXACT_F32),
            );
        }
        Err(e) => h.check_bool(&format!("wdm_sqw LSTM: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. WDM ESN reservoir (nW-05)
// ═══════════════════════════════════════════════════════════════════

fn validate_wdm_esn_reservoir(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(505);
    let rs = 16;

    let w_res: Vec<f32> = (0..rs * rs).map(|_| rng.normal() as f32 * 0.1).collect();
    let w_in: Vec<f32> = (0..rs).map(|_| rng.normal() as f32 * 0.3).collect();
    let state: Vec<f32> = (0..rs).map(|_| rng.normal() as f32 * 0.5).collect();
    let x_val = 0.42_f32;

    let cpu_new: Vec<f32> = {
        let mut s = vec![0.0_f32; rs];
        for i in 0..rs {
            let mut sum = w_in[i] * x_val;
            for j in 0..rs {
                sum += w_res[i * rs + j] * state[j];
            }
            s[i] = sum.tanh();
        }
        s
    };

    let gpu_new = (|| -> Result<Vec<f32>, String> {
        let st = Tensor::from_data(&state, vec![1, rs], device.clone())
            .map_err(|e| format!("state: {e}"))?;
        let wr = Tensor::from_data(&w_res, vec![rs, rs], device.clone())
            .map_err(|e| format!("W_res: {e}"))?;
        let wr_t = wr.transpose().map_err(|e| format!("W_resT: {e}"))?;
        let rp = st.matmul(&wr_t).map_err(|e| format!("mm: {e}"))?;
        let wi_data: Vec<f32> = w_in.iter().map(|&w| w * x_val).collect();
        let inp = Tensor::from_data(&wi_data, vec![1, rs], device.clone())
            .map_err(|e| format!("inp: {e}"))?;
        let combined = rp.add(&inp).map_err(|e| format!("add: {e}"))?;
        let activated = combined.tanh().map_err(|e| format!("tanh: {e}"))?;
        activated.to_vec().map_err(|e| format!("readback: {e}"))
    })();

    match gpu_new {
        Ok(gn) => {
            let max_diff = gn
                .iter()
                .zip(cpu_new.iter())
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                .fold(0.0_f64, f64::max);
            h.check_bool(
                &format!("wdm_esn reservoir: max_diff={max_diff:.2e}"),
                max_diff < tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_bool(
                "wdm_esn reservoir: all bounded",
                gn.iter()
                    .all(|v| f64::from(v.abs()) <= 1.0 + tolerances::TENSOR_EXACT_F32),
            );
        }
        Err(e) => h.check_bool(&format!("wdm_esn reservoir: {e}"), false),
    }
}
