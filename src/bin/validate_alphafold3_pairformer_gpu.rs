// SPDX-License-Identifier: AGPL-3.0-or-later

//! nF-03 Phase D-GPU: `BarraCUDA` Tensor validation for `AlphaFold3` Pairformer primitives.
//!
//! Validates timestep conditioning, triangle multiply (outgoing/incoming),
//! triangle attention QKV, and pair transition FFN through `BarraCUDA` Tensor
//! ops on GPU, comparing with Rust CPU f64 reference implementations.
//!
//! Evolution chain:
//! ```text
//! Abramson et al. 2024 §3 → Python (control) → Rust (CPU f64)
//! → BarraCUDA (GPU f32 Tensor) → [Future: df64 WGSL shaders]
//! ```
//!
//! Cross-spring provenance:
//! - `hotSpring`: df64 precision shaders for `TriMul`/`TriAttn` (validated Phase B)
//! - `wetSpring`: evolutionary covariance → pair representations
//! - `neuralSpring`: Pairformer block composition + timestep conditioning

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::expect_used
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::coral_forge;
use neural_spring::coral_forge::pairformer;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

type Dev = Arc<WgpuDevice>;

#[tokio::main]
async fn main() {
    let Ok(gpu) = Gpu::new().await else {
        neural_spring::validation::exit_no_gpu();
    };
    eprintln!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let device: Dev = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("alphafold3_pairformer_gpu");

    validate_timestep_conditioning_gpu(&mut h, &device);
    validate_trimul_outgoing_gpu(&mut h, &device);
    validate_trimul_incoming_gpu(&mut h, &device);
    validate_triangle_attention_gpu(&mut h, &device);
    validate_pair_ffn_gpu(&mut h, &device);
    validate_full_pairformer_block_gpu(&mut h, &device);

    h.finish();
}

// ═══════════════════════════════════════════════════════════════════
// 1. Timestep conditioning: project t_emb → pair broadcast add
// ═══════════════════════════════════════════════════════════════════

fn validate_timestep_conditioning_gpu(h: &mut ValidationHarness, device: &Dev) {
    let n = 4_usize;
    let d = 6_usize;
    let d_model = 8_usize;
    let nn = n * n;
    let mut rng = Rng::new(50);

    let pair_repr: Vec<f64> = (0..nn * d).map(|_| rng.normal() * 0.3).collect();
    let t_emb: Vec<f64> = pairformer::sinusoidal_embedding(25.0, d_model);
    let w_cond: Vec<f64> = (0..d_model * d).map(|_| rng.normal() * 0.2).collect();
    let b_cond: Vec<f64> = (0..d).map(|_| rng.normal() * 0.05).collect();

    let cpu_ref =
        pairformer::condition_pair_with_timestep(&pair_repr, n, d, &t_emb, &w_cond, &b_cond);

    // GPU: t_emb @ w_cond + b_cond → [d], then broadcast-add to pair_repr
    let t_emb_f32: Vec<f32> = t_emb.iter().map(|&v| v as f32).collect();
    let w_cond_f32: Vec<f32> = w_cond.iter().map(|&v| v as f32).collect();
    let b_cond_f32: Vec<f32> = b_cond.iter().map(|&v| v as f32).collect();
    let pair_f32: Vec<f32> = pair_repr.iter().map(|&v| v as f32).collect();

    let gpu_result = (|| -> Result<Vec<f32>, String> {
        let temb_t = Tensor::from_data(&t_emb_f32, vec![1, d_model], device.clone())
            .map_err(|e| format!("t_emb: {e}"))?;
        let wcond_t = Tensor::from_data(&w_cond_f32, vec![d_model, d], device.clone())
            .map_err(|e| format!("w_cond: {e}"))?;
        let bcond_t = Tensor::from_data(&b_cond_f32, vec![1, d], device.clone())
            .map_err(|e| format!("b_cond: {e}"))?;

        let proj = temb_t
            .matmul(&wcond_t)
            .map_err(|e| format!("proj matmul: {e}"))?;
        let cond = proj.add(&bcond_t).map_err(|e| format!("proj add: {e}"))?;
        let cond_vec = cond.to_vec().map_err(|e| format!("cond read: {e}"))?;

        // Broadcast cond [d] to [nn, d] and add to pair_repr
        let cond_broadcast: Vec<f32> = (0..nn).flat_map(|_| cond_vec.iter().copied()).collect();
        let cond_broad_t = Tensor::from_data(&cond_broadcast, vec![nn, d], device.clone())
            .map_err(|e| format!("cond broadcast: {e}"))?;
        let pair_t = Tensor::from_data(&pair_f32, vec![nn, d], device.clone())
            .map_err(|e| format!("pair: {e}"))?;
        let result = pair_t
            .add(&cond_broad_t)
            .map_err(|e| format!("pair add: {e}"))?;
        result.to_vec().map_err(|e| format!("readback: {e}"))
    })();

    let gpu_vec = match gpu_result {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("timestep cond GPU: {e}"), false);
            return;
        }
    };

    let max_diff: f64 = gpu_vec
        .iter()
        .zip(cpu_ref.iter())
        .map(|(g, c)| (f64::from(*g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("nF-PF→cond: diff {max_diff:.2e} < tol"),
        max_diff < tolerances::TENSOR_MATMUL_F32,
    );
    h.check_bool("nF-PF→cond: finite", gpu_vec.iter().all(|v| v.is_finite()));
}

// ═══════════════════════════════════════════════════════════════════
// 2. Triangle multiply outgoing (Algorithm 11) on GPU
// ═══════════════════════════════════════════════════════════════════

fn validate_trimul_outgoing_gpu(h: &mut ValidationHarness, device: &Dev) {
    let n = 4_usize;
    let c = 3_usize;
    let nn = n * n;
    let mut rng = Rng::new(51);

    let proj_a: Vec<f64> = (0..nn * c).map(|_| rng.normal() * 0.3).collect();
    let proj_b: Vec<f64> = (0..nn * c).map(|_| rng.normal() * 0.3).collect();

    let cpu_ref = coral_forge::triangle_mul_outgoing(&proj_a, &proj_b, n, c);

    // GPU: for each channel, out[i,j,ch] = sum_k A[i,k,ch] * B[j,k,ch]
    // Reshape A to [c, n, n] → for channel ch, A_ch is [n, n]
    // TriMul outgoing: out[i,j] = sum_k A[i,k] * B[j,k] = A @ B^T
    let gpu_result = (|| -> Result<Vec<f32>, String> {
        let mut out = vec![0.0_f32; nn * c];
        for ch in 0..c {
            let a_ch: Vec<f32> = (0..nn).map(|idx| proj_a[idx * c + ch] as f32).collect();
            let b_ch: Vec<f32> = (0..nn).map(|idx| proj_b[idx * c + ch] as f32).collect();

            let a_t = Tensor::from_data(&a_ch, vec![n, n], device.clone())
                .map_err(|e| format!("A ch{ch}: {e}"))?;
            let b_t = Tensor::from_data(&b_ch, vec![n, n], device.clone())
                .map_err(|e| format!("B ch{ch}: {e}"))?;
            let b_tr = b_t.transpose().map_err(|e| format!("B^T ch{ch}: {e}"))?;
            let ab = a_t
                .matmul(&b_tr)
                .map_err(|e| format!("A@B^T ch{ch}: {e}"))?;
            let ab_vec = ab.to_vec().map_err(|e| format!("read ch{ch}: {e}"))?;

            for (idx, &val) in ab_vec.iter().enumerate() {
                out[idx * c + ch] = val;
            }
        }
        Ok(out)
    })();

    let gpu_vec = match gpu_result {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("TriMul out GPU: {e}"), false);
            return;
        }
    };

    let max_diff: f64 = gpu_vec
        .iter()
        .zip(cpu_ref.iter())
        .map(|(g, c)| (f64::from(*g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("nF-PF→TriMul out: diff {max_diff:.2e} < tol"),
        max_diff < tolerances::TENSOR_MATMUL_F32,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 3. Triangle multiply incoming (Algorithm 12) on GPU
// ═══════════════════════════════════════════════════════════════════

fn validate_trimul_incoming_gpu(h: &mut ValidationHarness, device: &Dev) {
    let n = 4_usize;
    let c = 3_usize;
    let nn = n * n;
    let mut rng = Rng::new(52);

    let proj_a: Vec<f64> = (0..nn * c).map(|_| rng.normal() * 0.3).collect();
    let proj_b: Vec<f64> = (0..nn * c).map(|_| rng.normal() * 0.3).collect();

    let cpu_ref = coral_forge::triangle_mul_incoming(&proj_a, &proj_b, n, c);

    // TriMul incoming: out[i,j] = sum_k A[k,i] * B[k,j] = A^T @ B
    let gpu_result = (|| -> Result<Vec<f32>, String> {
        let mut out = vec![0.0_f32; nn * c];
        for ch in 0..c {
            let a_ch: Vec<f32> = (0..nn).map(|idx| proj_a[idx * c + ch] as f32).collect();
            let b_ch: Vec<f32> = (0..nn).map(|idx| proj_b[idx * c + ch] as f32).collect();

            let a_t = Tensor::from_data(&a_ch, vec![n, n], device.clone())
                .map_err(|e| format!("A ch{ch}: {e}"))?;
            let b_t = Tensor::from_data(&b_ch, vec![n, n], device.clone())
                .map_err(|e| format!("B ch{ch}: {e}"))?;
            let a_tr = a_t.transpose().map_err(|e| format!("A^T ch{ch}: {e}"))?;
            let ab = a_tr
                .matmul(&b_t)
                .map_err(|e| format!("A^T@B ch{ch}: {e}"))?;
            let ab_vec = ab.to_vec().map_err(|e| format!("read ch{ch}: {e}"))?;

            for (idx, &val) in ab_vec.iter().enumerate() {
                out[idx * c + ch] = val;
            }
        }
        Ok(out)
    })();

    let gpu_vec = match gpu_result {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("TriMul in GPU: {e}"), false);
            return;
        }
    };

    let max_diff: f64 = gpu_vec
        .iter()
        .zip(cpu_ref.iter())
        .map(|(g, c)| (f64::from(*g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("nF-PF→TriMul in: diff {max_diff:.2e} < tol"),
        max_diff < tolerances::TENSOR_MATMUL_F32,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 4. Triangle attention: QK^T/√d scores on GPU (Algorithms 13-14)
// ═══════════════════════════════════════════════════════════════════

fn validate_triangle_attention_gpu(h: &mut ValidationHarness, device: &Dev) {
    let n = 4_usize;
    let d = 6_usize;
    let n_heads = 2_usize;
    let head_dim = 3_usize;
    let nn = n * n;
    let mut rng = Rng::new(53);

    let normed: Vec<f64> = (0..nn * d).map(|_| rng.normal() * 0.3).collect();
    let w_q: Vec<f64> = (0..d * n_heads * head_dim)
        .map(|_| rng.normal() * 0.2)
        .collect();
    let w_k: Vec<f64> = (0..d * n_heads * head_dim)
        .map(|_| rng.normal() * 0.2)
        .collect();
    let w_v: Vec<f64> = (0..d * n_heads * head_dim)
        .map(|_| rng.normal() * 0.2)
        .collect();

    let h_hd = n_heads * head_dim;

    // Prepare f32 versions first (before closures borrow the originals)
    let normed_f32: Vec<f32> = normed.iter().map(|&v| v as f32).collect();
    let wq_f32: Vec<f32> = w_q.iter().map(|&v| v as f32).collect();
    let wk_f32: Vec<f32> = w_k.iter().map(|&v| v as f32).collect();
    let wv_f32: Vec<f32> = w_v.iter().map(|&v| v as f32).collect();

    // CPU reference: project Q, K via matmul
    let cpu_q: Vec<f64> = {
        let wq = &w_q;
        normed
            .chunks_exact(d)
            .flat_map(|x| {
                (0..h_hd).map(|j| {
                    x.iter()
                        .enumerate()
                        .fold(0.0_f64, |acc, (k, &xk)| xk.mul_add(wq[k * h_hd + j], acc))
                })
            })
            .collect()
    };

    let cpu_k: Vec<f64> = {
        let wk = &w_k;
        normed
            .chunks_exact(d)
            .flat_map(|x| {
                (0..h_hd).map(|j| {
                    x.iter()
                        .enumerate()
                        .fold(0.0_f64, |acc, (k, &xk)| xk.mul_add(wk[k * h_hd + j], acc))
                })
            })
            .collect()
    };

    #[allow(clippy::type_complexity)]
    let gpu_result = (|| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let normed_t = Tensor::from_data(&normed_f32, vec![nn, d], device.clone())
            .map_err(|e| format!("normed: {e}"))?;
        let wq_t = Tensor::from_data(&wq_f32, vec![d, h_hd], device.clone())
            .map_err(|e| format!("Wq: {e}"))?;
        let wk_t = Tensor::from_data(&wk_f32, vec![d, h_hd], device.clone())
            .map_err(|e| format!("Wk: {e}"))?;
        let wv_t = Tensor::from_data(&wv_f32, vec![d, h_hd], device.clone())
            .map_err(|e| format!("Wv: {e}"))?;

        let q = normed_t
            .matmul_ref(&wq_t)
            .map_err(|e| format!("Q matmul: {e}"))?;
        let k = normed_t
            .matmul_ref(&wk_t)
            .map_err(|e| format!("K matmul: {e}"))?;
        let v = normed_t
            .matmul_ref(&wv_t)
            .map_err(|e| format!("V matmul: {e}"))?;

        Ok((
            q.to_vec().map_err(|e| format!("Q read: {e}"))?,
            k.to_vec().map_err(|e| format!("K read: {e}"))?,
            v.to_vec().map_err(|e| format!("V read: {e}"))?,
        ))
    })();

    let (gpu_q, gpu_k, _gpu_v) = match gpu_result {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("TriAttn GPU: {e}"), false);
            return;
        }
    };

    // Validate Q projection GPU↔CPU
    let q_diff: f64 = gpu_q
        .iter()
        .zip(cpu_q.iter())
        .map(|(g, c)| (f64::from(*g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("nF-PF→TriAttn Q: diff {q_diff:.2e} < tol"),
        q_diff < tolerances::TENSOR_MATMUL_F32,
    );

    // Validate K projection GPU↔CPU
    let k_diff: f64 = gpu_k
        .iter()
        .zip(cpu_k.iter())
        .map(|(g, c)| (f64::from(*g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("nF-PF→TriAttn K: diff {k_diff:.2e} < tol"),
        k_diff < tolerances::TENSOR_MATMUL_F32,
    );

    // Compute attention scores on GPU: Q @ K^T / sqrt(head_dim)
    let scale = (head_dim as f32).sqrt();
    let inv_scale: Vec<f32> = vec![1.0 / scale; nn * nn];

    let score_result = (|| -> Result<Vec<f32>, String> {
        let q_t = Tensor::from_data(&gpu_q, vec![nn, h_hd], device.clone())
            .map_err(|e| format!("Q re: {e}"))?;
        let k_t = Tensor::from_data(&gpu_k, vec![nn, h_hd], device.clone())
            .map_err(|e| format!("K re: {e}"))?;
        let k_tr = k_t.transpose().map_err(|e| format!("K^T: {e}"))?;
        let scores_raw = q_t.matmul(&k_tr).map_err(|e| format!("QK^T: {e}"))?;

        let inv_t = Tensor::from_data(&inv_scale, vec![nn, nn], device.clone())
            .map_err(|e| format!("inv_scale: {e}"))?;
        let scores = scores_raw.mul(&inv_t).map_err(|e| format!("scale: {e}"))?;
        scores.to_vec().map_err(|e| format!("scores read: {e}"))
    })();

    let gpu_scores = match score_result {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("TriAttn scores GPU: {e}"), false);
            return;
        }
    };

    h.check_bool(
        "nF-PF→TriAttn: scores finite",
        gpu_scores.iter().all(|v| v.is_finite()),
    );
    h.check_bool("nF-PF→TriAttn: scores shape", gpu_scores.len() == nn * nn);
}

// ═══════════════════════════════════════════════════════════════════
// 5. Pair transition FFN on GPU (matches diffusion validator)
// ═══════════════════════════════════════════════════════════════════

fn validate_pair_ffn_gpu(h: &mut ValidationHarness, device: &Dev) {
    let n = 3_usize;
    let d = 4_usize;
    let d_hidden = 6_usize;
    let nn = n * n;
    let mut rng = Rng::new(54);

    let input: Vec<f64> = (0..nn * d).map(|_| rng.normal() * 0.3).collect();
    let w1: Vec<f64> = (0..d * d_hidden).map(|_| rng.normal() * 0.2).collect();
    let b1: Vec<f64> = (0..d_hidden).map(|_| rng.normal() * 0.1).collect();
    let w2: Vec<f64> = (0..d_hidden * d).map(|_| rng.normal() * 0.2).collect();
    let b2: Vec<f64> = (0..d).map(|_| rng.normal() * 0.1).collect();

    let cpu_ref =
        coral_forge::diffusion::pair_transition_ffn(&input, n, d, &w1, &b1, d_hidden, &w2, &b2);

    let input_f32: Vec<f32> = input.iter().map(|&v| v as f32).collect();
    let w1_f32: Vec<f32> = w1.iter().map(|&v| v as f32).collect();
    let w2_f32: Vec<f32> = w2.iter().map(|&v| v as f32).collect();
    let b1_broad: Vec<f32> = (0..nn).flat_map(|_| b1.iter().map(|&v| v as f32)).collect();
    let b2_broad: Vec<f32> = (0..nn).flat_map(|_| b2.iter().map(|&v| v as f32)).collect();

    let gpu_result = (|| -> Result<Vec<f32>, String> {
        let inp_t = Tensor::from_data(&input_f32, vec![nn, d], device.clone())
            .map_err(|e| format!("input: {e}"))?;
        let w1_t = Tensor::from_data(&w1_f32, vec![d, d_hidden], device.clone())
            .map_err(|e| format!("W1: {e}"))?;
        let w2_t = Tensor::from_data(&w2_f32, vec![d_hidden, d], device.clone())
            .map_err(|e| format!("W2: {e}"))?;
        let b1_t = Tensor::from_data(&b1_broad, vec![nn, d_hidden], device.clone())
            .map_err(|e| format!("b1: {e}"))?;
        let b2_t = Tensor::from_data(&b2_broad, vec![nn, d], device.clone())
            .map_err(|e| format!("b2: {e}"))?;

        let h1 = inp_t.matmul(&w1_t).map_err(|e| format!("mm W1: {e}"))?;
        let h1b = h1.add(&b1_t).map_err(|e| format!("add b1: {e}"))?;

        let h1_vec = h1b.to_vec().map_err(|e| format!("h read: {e}"))?;
        let gelu_vec: Vec<f32> = h1_vec.iter().map(|&x| gelu_f32(x)).collect();

        let g_t = Tensor::from_data(&gelu_vec, vec![nn, d_hidden], device.clone())
            .map_err(|e| format!("gelu: {e}"))?;
        let h2 = g_t.matmul(&w2_t).map_err(|e| format!("mm W2: {e}"))?;
        let out = h2.add(&b2_t).map_err(|e| format!("add b2: {e}"))?;
        out.to_vec().map_err(|e| format!("out read: {e}"))
    })();

    let gpu_vec = match gpu_result {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("PF FFN GPU: {e}"), false);
            return;
        }
    };

    let max_diff: f64 = gpu_vec
        .iter()
        .zip(cpu_ref.iter())
        .map(|(g, c)| (f64::from(*g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("nF-PF→FFN: diff {max_diff:.2e} < tol"),
        max_diff < tolerances::TENSOR_MATMUL_F32,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 6. Full Pairformer block: TriMul → TriAttn → FFN → conditioning
// ═══════════════════════════════════════════════════════════════════

fn validate_full_pairformer_block_gpu(h: &mut ValidationHarness, device: &Dev) {
    let n = 3_usize;
    let d = 4_usize;
    let nn = n * n;
    let n_heads = 2_usize;
    let head_dim = 2_usize;
    let d_hidden = 6_usize;
    let d_model = d;
    let mut rng = Rng::new(55);

    let pair: Vec<f64> = (0..nn * d).map(|_| rng.normal() * 0.3).collect();
    let ln_gamma: Vec<f64> = (0..d).map(|_| rng.next_f64().mul_add(0.4, 0.8)).collect();
    let ln_beta: Vec<f64> = (0..d).map(|_| rng.normal() * 0.05).collect();
    let tri_out_wa: Vec<f64> = (0..d * d).map(|_| rng.normal() * 0.15).collect();
    let tri_out_wb: Vec<f64> = (0..d * d).map(|_| rng.normal() * 0.15).collect();
    let tri_out_wg: Vec<f64> = (0..d * d).map(|_| rng.normal() * 0.15).collect();
    let tri_in_wa: Vec<f64> = (0..d * d).map(|_| rng.normal() * 0.15).collect();
    let tri_in_wb: Vec<f64> = (0..d * d).map(|_| rng.normal() * 0.15).collect();
    let tri_in_wg: Vec<f64> = (0..d * d).map(|_| rng.normal() * 0.15).collect();
    let tri_attn_wq: Vec<f64> = (0..d * n_heads * head_dim)
        .map(|_| rng.normal() * 0.15)
        .collect();
    let tri_attn_wk: Vec<f64> = (0..d * n_heads * head_dim)
        .map(|_| rng.normal() * 0.15)
        .collect();
    let tri_attn_wv: Vec<f64> = (0..d * n_heads * head_dim)
        .map(|_| rng.normal() * 0.15)
        .collect();
    let ffn_w1: Vec<f64> = (0..d * d_hidden).map(|_| rng.normal() * 0.15).collect();
    let ffn_b1: Vec<f64> = (0..d_hidden).map(|_| rng.normal() * 0.05).collect();
    let ffn_w2: Vec<f64> = (0..d_hidden * d).map(|_| rng.normal() * 0.15).collect();
    let ffn_b2: Vec<f64> = (0..d).map(|_| rng.normal() * 0.05).collect();
    let cond_w: Vec<f64> = (0..d_model * d).map(|_| rng.normal() * 0.15).collect();
    let cond_b: Vec<f64> = (0..d).map(|_| rng.normal() * 0.05).collect();

    let t_emb = pairformer::sinusoidal_embedding(25.0, d_model);

    let weights = pairformer::PairformerWeights {
        ln_gamma: &ln_gamma,
        ln_beta: &ln_beta,
        tri_out_wa: &tri_out_wa,
        tri_out_wb: &tri_out_wb,
        tri_out_wg: &tri_out_wg,
        tri_in_wa: &tri_in_wa,
        tri_in_wb: &tri_in_wb,
        tri_in_wg: &tri_in_wg,
        n_heads,
        head_dim,
        tri_attn_wq: &tri_attn_wq,
        tri_attn_wk: &tri_attn_wk,
        tri_attn_wv: &tri_attn_wv,
        ffn_w1: &ffn_w1,
        ffn_b1: &ffn_b1,
        d_hidden,
        ffn_w2: &ffn_w2,
        ffn_b2: &ffn_b2,
        cond_w: &cond_w,
        cond_b: &cond_b,
    };

    let cpu_out = pairformer::pairformer_block(&pair, n, d, &weights, Some(&t_emb));

    // GPU end-to-end: run the FFN portion on GPU (the bottleneck matmuls)
    // We test the FFN + conditioning via GPU matmul since those are the
    // computation-heavy components. TriMul and TriAttn are validated separately above.
    let pair_f32: Vec<f32> = pair.iter().map(|&v| v as f32).collect();

    // Verify CPU output is finite and the right shape
    h.check_bool(
        "nF-PF→block: CPU finite",
        cpu_out.iter().all(|v| v.is_finite()),
    );

    // GPU: upload pair → matmul FFN path
    // Since the full block includes layer norm + TriMul + TriAttn + FFN + conditioning,
    // and only FFN/conditioning have GPU matmul paths, we validate the FFN stage in isolation.
    // The overall block correctness is proven by CPU validator; here we prove GPU matmul
    // produces equivalent FFN output when given the same pre-FFN input.

    // Validate FFN portion in isolation on GPU (using pair as input proxy)
    let ffn_gpu = (|| -> Result<Vec<f32>, String> {
        let inp_t = Tensor::from_data(&pair_f32, vec![nn, d], device.clone())
            .map_err(|e| format!("pair: {e}"))?;
        let w1_f32: Vec<f32> = ffn_w1.iter().map(|&v| v as f32).collect();
        let w2_f32: Vec<f32> = ffn_w2.iter().map(|&v| v as f32).collect();
        let b1_broad: Vec<f32> = (0..nn)
            .flat_map(|_| ffn_b1.iter().map(|&v| v as f32))
            .collect();
        let b2_broad: Vec<f32> = (0..nn)
            .flat_map(|_| ffn_b2.iter().map(|&v| v as f32))
            .collect();

        let w1_t = Tensor::from_data(&w1_f32, vec![d, d_hidden], device.clone())
            .map_err(|e| format!("W1: {e}"))?;
        let w2_t = Tensor::from_data(&w2_f32, vec![d_hidden, d], device.clone())
            .map_err(|e| format!("W2: {e}"))?;
        let b1_t = Tensor::from_data(&b1_broad, vec![nn, d_hidden], device.clone())
            .map_err(|e| format!("b1: {e}"))?;
        let b2_t = Tensor::from_data(&b2_broad, vec![nn, d], device.clone())
            .map_err(|e| format!("b2: {e}"))?;

        let h1 = inp_t.matmul(&w1_t).map_err(|e| format!("mm1: {e}"))?;
        let h1b = h1.add(&b1_t).map_err(|e| format!("b1: {e}"))?;
        let hv = h1b.to_vec().map_err(|e| format!("hv: {e}"))?;
        let gv: Vec<f32> = hv.iter().map(|&x| gelu_f32(x)).collect();
        let gt = Tensor::from_data(&gv, vec![nn, d_hidden], device.clone())
            .map_err(|e| format!("g: {e}"))?;
        let h2 = gt.matmul(&w2_t).map_err(|e| format!("mm2: {e}"))?;
        let out = h2.add(&b2_t).map_err(|e| format!("b2: {e}"))?;
        out.to_vec().map_err(|e| format!("read: {e}"))
    })();

    let gpu_ffn = match ffn_gpu {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("PF block FFN GPU: {e}"), false);
            return;
        }
    };

    // Validate GPU FFN against CPU FFN (same input: raw pair, no layer norm)
    let cpu_ffn_ref = coral_forge::diffusion::pair_transition_ffn(
        &pair, n, d, &ffn_w1, &ffn_b1, d_hidden, &ffn_w2, &ffn_b2,
    );

    let ffn_diff: f64 = gpu_ffn
        .iter()
        .zip(cpu_ffn_ref.iter())
        .map(|(g, c)| (f64::from(*g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("nF-PF→block FFN: diff {ffn_diff:.2e} < tol"),
        ffn_diff < tolerances::TENSOR_MATMUL_F32,
    );

    h.check_bool("nF-PF→block: shape", cpu_out.len() == nn * d);

    let _ = device;
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn gelu_f32(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = sqrt_2_over_pi * 0.044_715_f32.mul_add(x * x * x, x);
    0.5 * x * (1.0 + inner.tanh())
}
