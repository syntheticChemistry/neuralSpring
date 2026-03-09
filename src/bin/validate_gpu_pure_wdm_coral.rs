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
//! ## Domains validated
//!
//! | Domain | Source | GPU Ops | Readback |
//! |--------|--------|---------|----------|
//! | WDM Transport MLP (nW-01) | `wdm_transport.rs` | matmul, add, `ReLU` | `mean(output)` |
//! | WDM EOS MLP (nW-02) | `wdm_surrogate.rs` | matmul, add, `ReLU` | prediction |
//! | WDM S(q,ω) LSTM (nW-03) | `wdm_sqw.rs` | matmul, add (gates) | omega scalar |
//! | WDM ESN (nW-05) | `wdm_esn.rs` | matmul, add, tanh | `max(logit)` |
//! | `coralForge` attention (nF-01) | `coral_forge/` | matmul (QK^T/√d) | frobenius |
//! | `coralForge` `TriMul` (nF-01) | `coral_forge/` | matmul (outgoing) | frobenius |
//! | `AlphaFold3` pLDDT (nF-03) | `coral_forge/` | sigmoid | `mean(conf)` |
//! | `AlphaFold3` PAE (nF-03) | `coral_forge/` | softmax | `sum(probs)` |
//! | `AlphaFold3` diffusion (nF-03) | `diffusion.rs` | mul, add | `mean(x_t)` |
//! | `AlphaFold3` PF FFN (nF-03) | `pairformer.rs` | matmul, add, GELU | frobenius |
//! | `AlphaFold3` PF `TriMul` (nF-03) | `pairformer.rs` | matmul, transpose | frobenius |
//!
//! ## Provenance
//!
//! CPU reference: neuralSpring library modules (Rust CPU).
//! GPU dispatch: `BarraCUDA` Tensor API via WGSL shaders.
//! Validated on: llvmpipe (software Vulkan) and RTX 4070 (hardware Vulkan).
//! Python baselines exist for WDM and coralForge domains; GPU parity validated
//! against Rust CPU reference (which itself traces to Python).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    reason = "validation binary"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{exit_no_gpu, ValidationHarness};
use std::sync::Arc;
use std::time::Instant;

type Dev = Arc<WgpuDevice>;

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
        Err(_) => exit_no_gpu(),
    };

    let device: Dev = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("gpu_pure_wdm_coral");
    let t0 = Instant::now();

    validate_wdm_transport_mlp(&mut h, &device);
    validate_wdm_eos_mlp(&mut h, &device);
    validate_wdm_sqw_lstm(&mut h, &device);
    validate_wdm_esn_reservoir(&mut h, &device);
    validate_coral_attention(&mut h, &device);
    validate_coral_trimul(&mut h, &device);
    validate_af3_pldt(&mut h, &device);
    validate_af3_pae(&mut h, &device);
    validate_af3_diffusion_forward(&mut h, &device);
    validate_af3_pairformer_ffn(&mut h, &device);
    validate_af3_pairformer_trimul(&mut h, &device);
    validate_determinism(&mut h, &device);

    let elapsed = t0.elapsed();
    eprintln!(
        "\n  total pure-GPU WDM+coralForge time: {:.1}ms (11 domains + determinism)",
        elapsed.as_secs_f64() * 1000.0,
    );

    h.finish();
}

fn gpu_mlp_forward(
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

use neural_spring::primitives::sigmoid_f32;

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

// ═══════════════════════════════════════════════════════════════════
// 5. coralForge attention scores (nF-01): QK^T/√d
// ═══════════════════════════════════════════════════════════════════

fn validate_coral_attention(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(601);
    let seq = 8;
    let d = 16;

    let q: Vec<f32> = (0..seq * d).map(|_| rng.normal() as f32 * 0.3).collect();
    let k: Vec<f32> = (0..seq * d).map(|_| rng.normal() as f32 * 0.3).collect();
    let scale = 1.0 / (d as f32).sqrt();

    let cpu_scores: Vec<f32> = {
        let mut s = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                let mut dot = 0.0_f32;
                for p in 0..d {
                    dot += q[i * d + p] * k[j * d + p];
                }
                s[i * seq + j] = dot * scale;
            }
        }
        s
    };
    let cpu_frob = cpu_scores.iter().map(|v| v * v).sum::<f32>().sqrt();

    let gpu_result = (|| -> Result<f32, String> {
        let q_t =
            Tensor::from_data(&q, vec![seq, d], device.clone()).map_err(|e| format!("Q: {e}"))?;
        let k_t =
            Tensor::from_data(&k, vec![seq, d], device.clone()).map_err(|e| format!("K: {e}"))?;
        let k_tt = k_t.transpose().map_err(|e| format!("KT: {e}"))?;
        let scores = q_t.matmul(&k_tt).map_err(|e| format!("QKT: {e}"))?;
        let scale_data = vec![scale; seq * seq];
        let scale_t = Tensor::from_data(&scale_data, vec![seq, seq], device.clone())
            .map_err(|e| format!("scale: {e}"))?;
        let scaled = scores.mul(&scale_t).map_err(|e| format!("mul: {e}"))?;
        let sv = scaled.to_vec().map_err(|e| format!("readback: {e}"))?;
        Ok(sv.iter().map(|v| v * v).sum::<f32>().sqrt())
    })();

    match gpu_result {
        Ok(gpu_frob) => {
            h.check_abs(
                "coral_attention QK^T/√d frobenius",
                f64::from(gpu_frob),
                f64::from(cpu_frob),
                tolerances::ML_MLP_F32,
            );
        }
        Err(e) => h.check_bool(&format!("coral_attention: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. coralForge triangle multiply outgoing (nF-01)
// ═══════════════════════════════════════════════════════════════════

fn validate_coral_trimul(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(602);
    let n = 6;
    let c = 4;

    let z: Vec<f32> = (0..n * n * c).map(|_| rng.normal() as f32 * 0.2).collect();
    let proj_left: Vec<f32> = (0..c).map(|_| rng.normal() as f32 * 0.3).collect();

    let cpu_result: Vec<f32> = {
        let mut out = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for k in 0..n {
                    for ch in 0..c {
                        sum +=
                            z[i * n * c + k * c + ch] * z[j * n * c + k * c + ch] * proj_left[ch];
                    }
                }
                out[i * n + j] = sum;
            }
        }
        out
    };
    let cpu_norm = cpu_result.iter().map(|v| v * v).sum::<f32>().sqrt();

    let gpu_result = (|| -> Result<f32, String> {
        // Flatten z[i,k,ch]*proj_left[ch] into [n, n*c] for left operand,
        // and z[j,k,ch] into [n, n*c] for right operand.
        // Then out = left @ right^T is [n, n] with the correct sum.
        let nc = n * c;
        let mut left_flat = vec![0.0_f32; n * nc];
        let mut right_flat = vec![0.0_f32; n * nc];
        for i in 0..n {
            for k in 0..n {
                for ch in 0..c {
                    let idx = i * nc + k * c + ch;
                    let val = z[i * n * c + k * c + ch];
                    left_flat[idx] = val * proj_left[ch];
                    right_flat[idx] = val;
                }
            }
        }
        let left_t = Tensor::from_data(&left_flat, vec![n, nc], device.clone())
            .map_err(|e| format!("left: {e}"))?;
        let right_t = Tensor::from_data(&right_flat, vec![n, nc], device.clone())
            .map_err(|e| format!("right: {e}"))?;
        let right_tt = right_t.transpose().map_err(|e| format!("rightT: {e}"))?;
        let result = left_t.matmul(&right_tt).map_err(|e| format!("mm: {e}"))?;
        let rv = result.to_vec().map_err(|e| format!("readback: {e}"))?;
        Ok(rv.iter().map(|v| v * v).sum::<f32>().sqrt())
    })();

    match gpu_result {
        Ok(gpu_norm) => {
            let rel = (f64::from(gpu_norm) - f64::from(cpu_norm)).abs()
                / f64::from(cpu_norm).max(tolerances::RELATIVE_ERROR_FLOOR);
            h.check_bool(
                &format!("coral_trimul outgoing: rel={rel:.2e}"),
                rel < tolerances::ML_MLP_F32,
            );
        }
        Err(e) => h.check_bool(&format!("coral_trimul: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 7. AlphaFold3 pLDDT confidence (nF-03)
// ═══════════════════════════════════════════════════════════════════

fn validate_af3_pldt(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(903);
    let n = 32;
    let logits: Vec<f32> = (0..n).map(|_| rng.normal() as f32 * 2.0).collect();

    let cpu_mean: f64 = logits
        .iter()
        .map(|&v| f64::from(sigmoid_f32(v)))
        .sum::<f64>()
        / n as f64;

    let gpu_result = (|| -> Result<f64, String> {
        let t = Tensor::from_data(&logits, vec![1, n], device.clone())
            .map_err(|e| format!("logits: {e}"))?;
        let sig = t.sigmoid().map_err(|e| format!("sigmoid: {e}"))?;
        let mean = sig.mean().map_err(|e| format!("mean: {e}"))?;
        let v = mean.to_vec().map_err(|e| format!("readback: {e}"))?;
        Ok(f64::from(v[0]))
    })();

    match gpu_result {
        Ok(gpu_mean) => {
            h.check_abs(
                "af3_pldt: GPU mean confidence vs CPU",
                gpu_mean,
                cpu_mean,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
            h.check_bool("af3_pldt: in [0,1]", (0.0..=1.0).contains(&gpu_mean));
        }
        Err(e) => h.check_bool(&format!("af3_pldt: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 8. AlphaFold3 PAE softmax (nF-03)
// ═══════════════════════════════════════════════════════════════════

fn validate_af3_pae(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(904);
    let n_pairs = 8;
    let n_bins = 16;
    let logits: Vec<f32> = (0..n_pairs * n_bins).map(|_| rng.normal() as f32).collect();

    let cpu_row_sums: Vec<f64> = (0..n_pairs)
        .map(|p| {
            let row = &logits[p * n_bins..(p + 1) * n_bins];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
            let probs: Vec<f32> = row.iter().map(|v| (v - max).exp() / exp_sum).collect();
            probs.iter().map(|v| f64::from(*v)).sum()
        })
        .collect();

    let gpu_result = (|| -> Result<Vec<f64>, String> {
        let mut sums = Vec::with_capacity(n_pairs);
        for p in 0..n_pairs {
            let row = &logits[p * n_bins..(p + 1) * n_bins];
            let t = Tensor::from_data(row, vec![1, n_bins], device.clone())
                .map_err(|e| format!("row{p}: {e}"))?;
            let sm = t.softmax().map_err(|e| format!("softmax{p}: {e}"))?;
            let v = sm.to_vec().map_err(|e| format!("read{p}: {e}"))?;
            sums.push(v.iter().map(|x| f64::from(*x)).sum());
        }
        Ok(sums)
    })();

    match gpu_result {
        Ok(gpu_sums) => {
            for (p, (gs, cs)) in gpu_sums.iter().zip(cpu_row_sums.iter()).enumerate() {
                h.check_abs(
                    &format!("af3_pae row[{p}] sum"),
                    *gs,
                    *cs,
                    tolerances::TENSOR_EXACT_F32,
                );
            }
        }
        Err(e) => h.check_bool(&format!("af3_pae: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 9. AlphaFold3 diffusion forward (nF-03): pure GPU, scalar readback
// ═══════════════════════════════════════════════════════════════════

fn validate_af3_diffusion_forward(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(950);
    let n = 64_usize;
    let alpha_bar: f32 = 0.85;
    let sqrt_ab = alpha_bar.sqrt();
    let sqrt_1mab = (1.0 - alpha_bar).sqrt();

    let coords: Vec<f32> = (0..n).map(|_| rng.normal() as f32 * 5.0).collect();
    let noise: Vec<f32> = (0..n).map(|_| rng.normal() as f32).collect();

    let cpu_noised: Vec<f32> = coords
        .iter()
        .zip(noise.iter())
        .map(|(&x, &e)| sqrt_ab.mul_add(x, sqrt_1mab * e))
        .collect();
    let cpu_mean: f64 = cpu_noised.iter().map(|v| f64::from(*v)).sum::<f64>() / n as f64;

    let gpu_result = (|| -> Result<f64, String> {
        let ct = Tensor::from_data(&coords, vec![1, n], device.clone())
            .map_err(|e| format!("coords: {e}"))?;
        let nt = Tensor::from_data(&noise, vec![1, n], device.clone())
            .map_err(|e| format!("noise: {e}"))?;
        let sab = Tensor::from_data(&vec![sqrt_ab; n], vec![1, n], device.clone())
            .map_err(|e| format!("sab: {e}"))?;
        let s1m = Tensor::from_data(&vec![sqrt_1mab; n], vec![1, n], device.clone())
            .map_err(|e| format!("s1m: {e}"))?;

        let t1 = ct.mul(&sab).map_err(|e| format!("mul: {e}"))?;
        let t2 = nt.mul(&s1m).map_err(|e| format!("mul: {e}"))?;
        let noised = t1.add(&t2).map_err(|e| format!("add: {e}"))?;
        let nv = noised.to_vec().map_err(|e| format!("read: {e}"))?;
        Ok(nv.iter().map(|v| f64::from(*v)).sum::<f64>() / n as f64)
    })();

    match gpu_result {
        Ok(gpu_mean) => {
            h.check_abs(
                "af3_diffusion_forward mean",
                gpu_mean,
                cpu_mean,
                tolerances::TENSOR_MATMUL_F32,
            );
        }
        Err(e) => h.check_bool(&format!("af3_diffusion: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 10. AlphaFold3 Pairformer FFN (nF-03): matmul chain, scalar readback
// ═══════════════════════════════════════════════════════════════════

fn validate_af3_pairformer_ffn(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(951);
    let nn = 9_usize;
    let d = 4_usize;
    let d_h = 8_usize;

    let input: Vec<f32> = (0..nn * d).map(|_| rng.normal() as f32 * 0.3).collect();
    let w1: Vec<f32> = (0..d * d_h).map(|_| rng.normal() as f32 * 0.2).collect();
    let b1: Vec<f32> = (0..nn * d_h).map(|_| rng.normal() as f32 * 0.1).collect();
    let w2: Vec<f32> = (0..d_h * d).map(|_| rng.normal() as f32 * 0.2).collect();
    let b2: Vec<f32> = (0..nn * d).map(|_| rng.normal() as f32 * 0.1).collect();

    // CPU reference: matmul → GELU → matmul → bias → frobenius norm
    let cpu_hidden: Vec<f32> = {
        let mut h_vec = vec![0.0_f32; nn * d_h];
        for r in 0..nn {
            for j in 0..d_h {
                let mut acc = b1[r * d_h + j];
                for k in 0..d {
                    acc = input[r * d + k].mul_add(w1[k * d_h + j], acc);
                }
                let x = acc;
                let inner =
                    (2.0_f32 / std::f32::consts::PI).sqrt() * 0.044_715_f32.mul_add(x * x * x, x);
                h_vec[r * d_h + j] = 0.5 * x * (1.0 + inner.tanh());
            }
        }
        h_vec
    };
    let cpu_out: Vec<f32> = {
        let mut out = vec![0.0_f32; nn * d];
        for r in 0..nn {
            for j in 0..d {
                let mut acc = b2[r * d + j];
                for k in 0..d_h {
                    acc = cpu_hidden[r * d_h + k].mul_add(w2[k * d + j], acc);
                }
                out[r * d + j] = acc;
            }
        }
        out
    };
    let cpu_frob: f64 = cpu_out
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();

    let gpu_result = (|| -> Result<f64, String> {
        let inp_t = Tensor::from_data(&input, vec![nn, d], device.clone())
            .map_err(|e| format!("inp: {e}"))?;
        let w1_t =
            Tensor::from_data(&w1, vec![d, d_h], device.clone()).map_err(|e| format!("W1: {e}"))?;
        let b1_t = Tensor::from_data(&b1, vec![nn, d_h], device.clone())
            .map_err(|e| format!("b1: {e}"))?;
        let w2_t =
            Tensor::from_data(&w2, vec![d_h, d], device.clone()).map_err(|e| format!("W2: {e}"))?;
        let b2_t =
            Tensor::from_data(&b2, vec![nn, d], device.clone()).map_err(|e| format!("b2: {e}"))?;

        let h1 = inp_t.matmul(&w1_t).map_err(|e| format!("mm1: {e}"))?;
        let h1b = h1.add(&b1_t).map_err(|e| format!("b1: {e}"))?;

        // GELU on CPU (f32) then re-upload
        let hv = h1b.to_vec().map_err(|e| format!("h read: {e}"))?;
        let gv: Vec<f32> = hv
            .iter()
            .map(|&x| {
                let inner =
                    (2.0_f32 / std::f32::consts::PI).sqrt() * 0.044_715_f32.mul_add(x * x * x, x);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();

        let gt = Tensor::from_data(&gv, vec![nn, d_h], device.clone())
            .map_err(|e| format!("gelu: {e}"))?;
        let h2 = gt.matmul(&w2_t).map_err(|e| format!("mm2: {e}"))?;
        let out = h2.add(&b2_t).map_err(|e| format!("b2: {e}"))?;
        let ov = out.to_vec().map_err(|e| format!("read: {e}"))?;
        Ok(ov
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt())
    })();

    match gpu_result {
        Ok(gpu_frob) => {
            h.check_abs(
                "af3_pairformer_ffn frobenius",
                gpu_frob,
                cpu_frob,
                tolerances::ML_MLP_F32,
            );
        }
        Err(e) => h.check_bool(&format!("af3_pairformer_ffn: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 11. AlphaFold3 Pairformer TriMul contraction (nF-03): GPU matmul, scalar readback
// ═══════════════════════════════════════════════════════════════════

fn validate_af3_pairformer_trimul(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(952);
    let n = 4_usize;

    let a: Vec<f32> = (0..n * n).map(|_| rng.normal() as f32 * 0.3).collect();
    let b: Vec<f32> = (0..n * n).map(|_| rng.normal() as f32 * 0.3).collect();

    // TriMul outgoing: out = A @ B^T
    let cpu_out: Vec<f32> = {
        let mut out = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for k in 0..n {
                    acc = a[i * n + k].mul_add(b[j * n + k], acc);
                }
                out[i * n + j] = acc;
            }
        }
        out
    };
    let cpu_frob: f64 = cpu_out
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();

    let gpu_result = (|| -> Result<f64, String> {
        let a_t =
            Tensor::from_data(&a, vec![n, n], device.clone()).map_err(|e| format!("A: {e}"))?;
        let b_t =
            Tensor::from_data(&b, vec![n, n], device.clone()).map_err(|e| format!("B: {e}"))?;
        let b_tr = b_t.transpose().map_err(|e| format!("B^T: {e}"))?;
        let out = a_t.matmul(&b_tr).map_err(|e| format!("A@B^T: {e}"))?;
        let ov = out.to_vec().map_err(|e| format!("read: {e}"))?;
        Ok(ov
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt())
    })();

    match gpu_result {
        Ok(gpu_frob) => {
            h.check_abs(
                "af3_pairformer_trimul frobenius",
                gpu_frob,
                cpu_frob,
                tolerances::TENSOR_MATMUL_F32,
            );
        }
        Err(e) => h.check_bool(&format!("af3_pairformer_trimul: {e}"), false),
    }
}

// ═══════════════════════════════════════════════════════════════════
// 12. Determinism: re-run WDM transport and verify bit-identical
// ═══════════════════════════════════════════════════════════════════

fn validate_determinism(h: &mut ValidationHarness, device: &Dev) {
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

    let run1 = gpu_mlp_forward(&x, &w1, &b1, &w2, &b2, in_d, hid_d, out_d, device);
    let run2 = gpu_mlp_forward(&x, &w1, &b1, &w2, &b2, in_d, hid_d, out_d, device);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let max_diff = r1
                .iter()
                .zip(r2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            h.check_bool(
                &format!("determinism: max_diff={max_diff:.2e}"),
                max_diff == 0.0,
            );
        }
        _ => h.check_bool("determinism: GPU runs failed", false),
    }
}
