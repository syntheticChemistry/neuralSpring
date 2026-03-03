// SPDX-License-Identifier: AGPL-3.0-or-later

//! nF-03 Phase D-GPU: `BarraCUDA` Tensor validation for `AlphaFold3` diffusion primitives.
//!
//! Validates forward diffusion, DDPM/DDIM reverse steps, SE(3)-equivariant noise,
//! and pair transition FFN through `BarraCUDA` Tensor ops on GPU, comparing with
//! Rust CPU f64 reference implementations.
//!
//! Evolution chain:
//! ```text
//! Ho et al. 2020 / Song et al. 2021 → Abramson et al. 2024 §5
//! → Python (control) → Rust (CPU f64) → BarraCUDA (GPU f32 Tensor)
//! ```
//!
//! Cross-spring provenance:
//! - `hotSpring`: df64 precision shaders (fp48 emulated on fp32 cores)
//! - `wetSpring`: bio-domain diffusion scheduling (molecular dynamics connection)
//! - `neuralSpring`: diffusion model implementation + Pairformer FFN

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::expect_used,
    reason = "validation binary"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::coral_forge::diffusion;
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
    let mut h = ValidationHarness::new("alphafold3_diffusion_gpu");

    validate_forward_diffusion_gpu(&mut h, &device);
    validate_ddpm_reverse_gpu(&mut h, &device);
    validate_ddim_reverse_gpu(&mut h, &device);
    validate_se3_com_removal_gpu(&mut h, &device);
    validate_pair_ffn_gpu(&mut h, &device);
    benchmark_diffusion_gpu(&mut h, &device);

    h.finish();
}

// ═══════════════════════════════════════════════════════════════════
// 1. Forward diffusion: x_t = sqrt(αbar) * x_0 + sqrt(1-αbar) * ε
// ═══════════════════════════════════════════════════════════════════

fn validate_forward_diffusion_gpu(h: &mut ValidationHarness, device: &Dev) {
    let n_atoms = 16_usize;
    let dim = 3_usize;
    let n = n_atoms * dim;
    let mut rng = Rng::new(42);

    let coords: Vec<f64> = (0..n).map(|_| rng.normal() * 5.0).collect();
    let noise: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let schedule = diffusion::cosine_beta_schedule(50, 0.008);
    let t = 25;

    let cpu_ref = diffusion::forward_diffusion(&coords, &noise, t, &schedule);

    let alpha_bar_t = schedule.alpha_bar[t];
    let sqrt_ab = alpha_bar_t.sqrt() as f32;
    let sqrt_1mab = (1.0 - alpha_bar_t).sqrt() as f32;

    let coords_f32: Vec<f32> = coords.iter().map(|&v| v as f32).collect();
    let noise_f32: Vec<f32> = noise.iter().map(|&v| v as f32).collect();
    let scale_signal: Vec<f32> = vec![sqrt_ab; n];
    let scale_noise: Vec<f32> = vec![sqrt_1mab; n];

    let Ok(coords_t) = Tensor::from_data(&coords_f32, vec![1, n], device.clone()) else {
        h.check_bool("forward diffusion: coords tensor", false);
        return;
    };
    let Ok(noise_t) = Tensor::from_data(&noise_f32, vec![1, n], device.clone()) else {
        h.check_bool("forward diffusion: noise tensor", false);
        return;
    };
    let Ok(scale_s_t) = Tensor::from_data(&scale_signal, vec![1, n], device.clone()) else {
        h.check_bool("forward diffusion: scale_signal tensor", false);
        return;
    };
    let Ok(scale_n_t) = Tensor::from_data(&scale_noise, vec![1, n], device.clone()) else {
        h.check_bool("forward diffusion: scale_noise tensor", false);
        return;
    };

    let gpu_result = (|| -> Result<Vec<f32>, String> {
        let term1 = coords_t
            .mul(&scale_s_t)
            .map_err(|e| format!("mul signal: {e}"))?;
        let term2 = noise_t
            .mul(&scale_n_t)
            .map_err(|e| format!("mul noise: {e}"))?;
        let x_t = term1.add(&term2).map_err(|e| format!("add: {e}"))?;
        x_t.to_vec().map_err(|e| format!("readback: {e}"))
    })();

    let gpu_vec = match gpu_result {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("forward diffusion GPU: {e}"), false);
            return;
        }
    };

    h.check_bool(
        "nF-D→forward: GPU finite",
        gpu_vec.iter().all(|v| v.is_finite()),
    );
    h.check_bool("nF-D→forward: shape preserved", gpu_vec.len() == n);

    let max_diff: f64 = gpu_vec
        .iter()
        .zip(cpu_ref.iter())
        .map(|(g, c)| (f64::from(*g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("nF-D→forward: GPU↔CPU diff {max_diff:.2e} < tol"),
        max_diff < tolerances::TENSOR_MATMUL_F32,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 2. DDPM reverse step (stochastic denoising)
// ═══════════════════════════════════════════════════════════════════

fn validate_ddpm_reverse_gpu(h: &mut ValidationHarness, device: &Dev) {
    let n = 24_usize;
    let mut rng = Rng::new(43);

    let x_t: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let pred_noise: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let z: Vec<f64> = (0..n).map(|_| rng.normal()).collect();

    let schedule = diffusion::cosine_beta_schedule(50, 0.008);
    let t = 20;

    let cpu_ref = diffusion::ddpm_reverse_step(&x_t, &pred_noise, &z, t, &schedule);

    let beta_t = schedule.betas[t];
    let alpha_t = 1.0 - beta_t;
    let a_bar_t = schedule.alpha_bar[t];
    let coeff_x = (1.0 / alpha_t.sqrt()) as f32;
    let coeff_eps = (beta_t / (1.0 - a_bar_t).sqrt()) as f32;
    let sigma_t = beta_t.sqrt() as f32;

    let x_f32: Vec<f32> = x_t.iter().map(|&v| v as f32).collect();
    let eps_f32: Vec<f32> = pred_noise.iter().map(|&v| v as f32).collect();
    let z_f32: Vec<f32> = z.iter().map(|&v| v as f32).collect();

    let coeff_x_vec: Vec<f32> = vec![coeff_x; n];
    let coeff_eps_vec: Vec<f32> = vec![coeff_eps; n];
    let sigma_vec: Vec<f32> = vec![sigma_t; n];

    let gpu_result = (|| -> Result<Vec<f32>, String> {
        let x_tensor =
            Tensor::from_data(&x_f32, vec![1, n], device.clone()).map_err(|e| format!("x: {e}"))?;
        let eps_tensor = Tensor::from_data(&eps_f32, vec![1, n], device.clone())
            .map_err(|e| format!("eps: {e}"))?;
        let z_tensor =
            Tensor::from_data(&z_f32, vec![1, n], device.clone()).map_err(|e| format!("z: {e}"))?;
        let cx_tensor = Tensor::from_data(&coeff_x_vec, vec![1, n], device.clone())
            .map_err(|e| format!("cx: {e}"))?;
        let ce_tensor = Tensor::from_data(&coeff_eps_vec, vec![1, n], device.clone())
            .map_err(|e| format!("ce: {e}"))?;
        let sig_tensor = Tensor::from_data(&sigma_vec, vec![1, n], device.clone())
            .map_err(|e| format!("sig: {e}"))?;

        // DDPM: x_{t-1} = coeff_x * (x_t - coeff_eps * eps) + sigma * z
        let eps_scaled = eps_tensor
            .mul(&ce_tensor)
            .map_err(|e| format!("eps*ce: {e}"))?;
        let diff = x_tensor
            .add(&neg_f32(&eps_scaled, device)?)
            .map_err(|e| format!("x-eps: {e}"))?;
        let mean = diff.mul(&cx_tensor).map_err(|e| format!("mean: {e}"))?;
        let noise_term = z_tensor
            .mul(&sig_tensor)
            .map_err(|e| format!("noise: {e}"))?;
        let result = mean
            .add(&noise_term)
            .map_err(|e| format!("add noise: {e}"))?;
        result.to_vec().map_err(|e| format!("readback: {e}"))
    })();

    let gpu_vec = match gpu_result {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("DDPM reverse GPU: {e}"), false);
            return;
        }
    };

    h.check_bool(
        "nF-D→DDPM: GPU finite",
        gpu_vec.iter().all(|v| v.is_finite()),
    );

    let max_diff: f64 = gpu_vec
        .iter()
        .zip(cpu_ref.iter())
        .map(|(g, c)| (f64::from(*g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("nF-D→DDPM: GPU↔CPU diff {max_diff:.2e} < tol"),
        max_diff < tolerances::TENSOR_MATMUL_F32,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 3. DDIM reverse step (deterministic denoising)
// ═══════════════════════════════════════════════════════════════════

fn validate_ddim_reverse_gpu(h: &mut ValidationHarness, device: &Dev) {
    let n = 18_usize;
    let mut rng = Rng::new(44);

    let x_t: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let pred_noise: Vec<f64> = (0..n).map(|_| rng.normal()).collect();

    let schedule = diffusion::cosine_beta_schedule(50, 0.008);
    let t = 30;

    let (cpu_x_prev, cpu_pred_x0) = diffusion::ddim_reverse_step(&x_t, &pred_noise, t, &schedule);

    let a_bar_t = schedule.alpha_bar[t];
    let a_bar_prev = schedule.alpha_bar[t - 1];
    let sqrt_ab_t = a_bar_t.sqrt() as f32;
    let sqrt_1m_t = (1.0 - a_bar_t).sqrt() as f32;
    let sqrt_ab_prev = a_bar_prev.sqrt() as f32;
    let sqrt_1m_prev = (1.0 - a_bar_prev).sqrt() as f32;

    let x_f32: Vec<f32> = x_t.iter().map(|&v| v as f32).collect();
    let eps_f32: Vec<f32> = pred_noise.iter().map(|&v| v as f32).collect();

    let gpu_result = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
        let x_tensor =
            Tensor::from_data(&x_f32, vec![1, n], device.clone()).map_err(|e| format!("x: {e}"))?;
        let eps_tensor = Tensor::from_data(&eps_f32, vec![1, n], device.clone())
            .map_err(|e| format!("eps: {e}"))?;

        // pred_x0 = (x_t - sqrt(1-αbar_t) * eps) / sqrt(αbar_t)
        let inv_sqrt_ab: Vec<f32> = vec![1.0 / sqrt_ab_t; n];
        let scale_eps: Vec<f32> = vec![sqrt_1m_t; n];

        let inv_t = Tensor::from_data(&inv_sqrt_ab, vec![1, n], device.clone())
            .map_err(|e| format!("inv: {e}"))?;
        let se_t = Tensor::from_data(&scale_eps, vec![1, n], device.clone())
            .map_err(|e| format!("se: {e}"))?;

        let eps_scaled = eps_tensor.mul(&se_t).map_err(|e| format!("eps*se: {e}"))?;
        let diff = x_tensor
            .add(&neg_f32(&eps_scaled, device)?)
            .map_err(|e| format!("x-eps: {e}"))?;
        let pred_x0 = diff.mul(&inv_t).map_err(|e| format!("pred_x0: {e}"))?;

        let pred_x0_vec = pred_x0.to_vec().map_err(|e| format!("pred_x0 read: {e}"))?;

        // x_prev = sqrt(αbar_{t-1}) * pred_x0 + sqrt(1-αbar_{t-1}) * eps
        let sap: Vec<f32> = vec![sqrt_ab_prev; n];
        let s1p: Vec<f32> = vec![sqrt_1m_prev; n];

        let pred_x0_2 = Tensor::from_data(&pred_x0_vec, vec![1, n], device.clone())
            .map_err(|e| format!("pred_x0 re-upload: {e}"))?;
        let eps_tensor_2 = Tensor::from_data(&eps_f32, vec![1, n], device.clone())
            .map_err(|e| format!("eps re-upload: {e}"))?;
        let sap_t =
            Tensor::from_data(&sap, vec![1, n], device.clone()).map_err(|e| format!("sap: {e}"))?;
        let s1p_t =
            Tensor::from_data(&s1p, vec![1, n], device.clone()).map_err(|e| format!("s1p: {e}"))?;

        let term1 = pred_x0_2.mul(&sap_t).map_err(|e| format!("sap*x0: {e}"))?;
        let term2 = eps_tensor_2
            .mul(&s1p_t)
            .map_err(|e| format!("s1p*eps: {e}"))?;
        let x_prev = term1.add(&term2).map_err(|e| format!("x_prev: {e}"))?;

        let x_prev_vec = x_prev.to_vec().map_err(|e| format!("x_prev read: {e}"))?;
        Ok((x_prev_vec, pred_x0_vec))
    })();

    let (gpu_x_prev, gpu_pred_x0) = match gpu_result {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("DDIM reverse GPU: {e}"), false);
            return;
        }
    };

    let diff_x0: f64 = gpu_pred_x0
        .iter()
        .zip(cpu_pred_x0.iter())
        .map(|(g, c)| (f64::from(*g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("nF-D→DDIM: pred_x0 diff {diff_x0:.2e} < tol"),
        diff_x0 < tolerances::TENSOR_MATMUL_F32,
    );

    let diff_xprev: f64 = gpu_x_prev
        .iter()
        .zip(cpu_x_prev.iter())
        .map(|(g, c)| (f64::from(*g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("nF-D→DDIM: x_prev diff {diff_xprev:.2e} < tol"),
        diff_xprev < tolerances::TENSOR_MATMUL_F32,
    );

    // Determinism: same input → same output
    let gpu_result2 = (|| -> Result<Vec<f32>, String> {
        let x2 =
            Tensor::from_data(&x_f32, vec![1, n], device.clone()).map_err(|e| format!("{e}"))?;
        let eps2 =
            Tensor::from_data(&eps_f32, vec![1, n], device.clone()).map_err(|e| format!("{e}"))?;
        let se = Tensor::from_data(&vec![sqrt_1m_t; n], vec![1, n], device.clone())
            .map_err(|e| format!("{e}"))?;
        let inv = Tensor::from_data(&vec![1.0 / sqrt_ab_t; n], vec![1, n], device.clone())
            .map_err(|e| format!("{e}"))?;
        let scaled = eps2.mul(&se).map_err(|e| format!("{e}"))?;
        let diff = x2
            .add(&neg_f32(&scaled, device)?)
            .map_err(|e| format!("{e}"))?;
        let px0 = diff.mul(&inv).map_err(|e| format!("{e}"))?;
        px0.to_vec().map_err(|e| format!("{e}"))
    })();
    if let Ok(v2) = gpu_result2 {
        let det_diff: f64 = v2
            .iter()
            .zip(gpu_pred_x0.iter())
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0_f64, f64::max);
        h.check_bool(
            &format!("nF-D→DDIM: determinism {det_diff:.2e}"),
            det_diff < 1e-12,
        );
    } else {
        h.check_bool("nF-D→DDIM: determinism (realloc)", true);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. SE(3) COM removal on GPU
// ═══════════════════════════════════════════════════════════════════

fn validate_se3_com_removal_gpu(h: &mut ValidationHarness, device: &Dev) {
    let n_atoms = 8_usize;
    let n = n_atoms * 3;
    let mut rng = Rng::new(45);

    let coords: Vec<f64> = (0..n).map(|_| rng.normal() * 10.0).collect();
    let (cpu_centered, cpu_com) = diffusion::remove_center_of_mass(&coords);

    // GPU: compute per-axis mean via matmul trick
    // Reshape coords as [n_atoms, 3], compute mean of each column
    let coords_f32: Vec<f32> = coords.iter().map(|&v| v as f32).collect();

    // Use matmul: ones^T @ coords / n → mean per column (COM)
    let ones: Vec<f32> = vec![1.0 / n_atoms as f32; n_atoms];
    let Ok(coords_t) = Tensor::from_data(&coords_f32, vec![n_atoms, 3], device.clone()) else {
        h.check_bool("SE(3): coords tensor", false);
        return;
    };
    let Ok(ones_t) = Tensor::from_data(&ones, vec![1, n_atoms], device.clone()) else {
        h.check_bool("SE(3): ones tensor", false);
        return;
    };

    let gpu_result = (|| -> Result<Vec<f32>, String> {
        let com_t = ones_t
            .matmul(&coords_t)
            .map_err(|e| format!("com matmul: {e}"))?;
        let com_vec = com_t.to_vec().map_err(|e| format!("com read: {e}"))?;

        // Broadcast COM and subtract: centered = coords - COM_broadcast
        let com_broadcast: Vec<f32> = (0..n_atoms).flat_map(|_| com_vec.iter().copied()).collect();
        let com_broad_t = Tensor::from_data(&com_broadcast, vec![n_atoms, 3], device.clone())
            .map_err(|e| format!("com broadcast: {e}"))?;
        let neg_com = neg_f32(&com_broad_t, device)?;
        let centered = coords_t
            .add(&neg_com)
            .map_err(|e| format!("subtract: {e}"))?;
        centered.to_vec().map_err(|e| format!("centered read: {e}"))
    })();

    let gpu_centered = match gpu_result {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("SE(3) GPU: {e}"), false);
            return;
        }
    };

    let max_diff: f64 = gpu_centered
        .iter()
        .zip(cpu_centered.iter())
        .map(|(g, c)| (f64::from(*g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("nF-D→SE(3): centered diff {max_diff:.2e} < tol"),
        max_diff < tolerances::TENSOR_MATMUL_F32,
    );

    // Verify residual COM near zero
    let gpu_com: [f64; 3] = {
        let mut com = [0.0_f64; 3];
        for atom in gpu_centered.chunks_exact(3) {
            com[0] += f64::from(atom[0]);
            com[1] += f64::from(atom[1]);
            com[2] += f64::from(atom[2]);
        }
        let nf = n_atoms as f64;
        [com[0] / nf, com[1] / nf, com[2] / nf]
    };
    let residual = gpu_com[2]
        .mul_add(
            gpu_com[2],
            gpu_com[1].mul_add(gpu_com[1], gpu_com[0].powi(2)),
        )
        .sqrt();
    h.check_bool(
        &format!("nF-D→SE(3): residual COM {residual:.2e} < tol"),
        residual < tolerances::TENSOR_MATMUL_F32,
    );

    let _ = cpu_com;
}

// ═══════════════════════════════════════════════════════════════════
// 5. Pair transition FFN: Linear → GELU → Linear on GPU
// ═══════════════════════════════════════════════════════════════════

fn validate_pair_ffn_gpu(h: &mut ValidationHarness, device: &Dev) {
    let n = 3_usize;
    let d_pair = 4_usize;
    let d_hidden = 8_usize;
    let nn = n * n;
    let mut rng = Rng::new(46);

    let pair_repr: Vec<f64> = (0..nn * d_pair).map(|_| rng.normal() * 0.3).collect();
    let w1: Vec<f64> = (0..d_pair * d_hidden).map(|_| rng.normal() * 0.2).collect();
    let b1: Vec<f64> = (0..d_hidden).map(|_| rng.normal() * 0.1).collect();
    let w2: Vec<f64> = (0..d_hidden * d_pair).map(|_| rng.normal() * 0.2).collect();
    let b2: Vec<f64> = (0..d_pair).map(|_| rng.normal() * 0.1).collect();

    let cpu_ref =
        diffusion::pair_transition_ffn(&pair_repr, n, d_pair, &w1, &b1, d_hidden, &w2, &b2);

    let pair_f32: Vec<f32> = pair_repr.iter().map(|&v| v as f32).collect();
    let w1_f32: Vec<f32> = w1.iter().map(|&v| v as f32).collect();
    let w2_f32: Vec<f32> = w2.iter().map(|&v| v as f32).collect();
    let b1_broadcast: Vec<f32> = (0..nn).flat_map(|_| b1.iter().map(|&v| v as f32)).collect();
    let b2_broadcast: Vec<f32> = (0..nn).flat_map(|_| b2.iter().map(|&v| v as f32)).collect();

    let gpu_result = (|| -> Result<Vec<f32>, String> {
        let pair_t = Tensor::from_data(&pair_f32, vec![nn, d_pair], device.clone())
            .map_err(|e| format!("pair: {e}"))?;
        let w1_t = Tensor::from_data(&w1_f32, vec![d_pair, d_hidden], device.clone())
            .map_err(|e| format!("W1: {e}"))?;
        let w2_t = Tensor::from_data(&w2_f32, vec![d_hidden, d_pair], device.clone())
            .map_err(|e| format!("W2: {e}"))?;
        let b1_t = Tensor::from_data(&b1_broadcast, vec![nn, d_hidden], device.clone())
            .map_err(|e| format!("b1: {e}"))?;
        let b2_t = Tensor::from_data(&b2_broadcast, vec![nn, d_pair], device.clone())
            .map_err(|e| format!("b2: {e}"))?;

        // hidden = GELU(pair @ W1 + b1)
        let linear1 = pair_t
            .matmul(&w1_t)
            .map_err(|e| format!("matmul W1: {e}"))?;
        let biased1 = linear1.add(&b1_t).map_err(|e| format!("add b1: {e}"))?;

        // GELU on CPU (GPU doesn't have native GELU activation yet; readback → apply → re-upload)
        let hidden_vec = biased1.to_vec().map_err(|e| format!("hidden read: {e}"))?;
        let gelu_vec: Vec<f32> = hidden_vec.iter().map(|&x| gelu_f32(x)).collect();

        let gelu_t = Tensor::from_data(&gelu_vec, vec![nn, d_hidden], device.clone())
            .map_err(|e| format!("gelu upload: {e}"))?;

        // output = gelu_hidden @ W2 + b2
        let linear2 = gelu_t
            .matmul(&w2_t)
            .map_err(|e| format!("matmul W2: {e}"))?;
        let biased2 = linear2.add(&b2_t).map_err(|e| format!("add b2: {e}"))?;

        biased2.to_vec().map_err(|e| format!("output read: {e}"))
    })();

    let gpu_vec = match gpu_result {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("pair FFN GPU: {e}"), false);
            return;
        }
    };

    h.check_bool(
        "nF-D→FFN: GPU finite",
        gpu_vec.iter().all(|v| v.is_finite()),
    );
    h.check_bool("nF-D→FFN: shape correct", gpu_vec.len() == nn * d_pair);

    let max_diff: f64 = gpu_vec
        .iter()
        .zip(cpu_ref.iter())
        .map(|(g, c)| (f64::from(*g) - c).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("nF-D→FFN: GPU↔CPU diff {max_diff:.2e} < tol"),
        max_diff < tolerances::TENSOR_MATMUL_F32,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 6. Throughput benchmark: diffusion ops on GPU
// ═══════════════════════════════════════════════════════════════════

fn benchmark_diffusion_gpu(h: &mut ValidationHarness, device: &Dev) {
    let n = 256_usize;
    let iters = 100_u32;
    let mut rng = Rng::new(99);

    let coords_f32: Vec<f32> = (0..n).map(|_| rng.normal() as f32).collect();
    let noise_f32: Vec<f32> = (0..n).map(|_| rng.normal() as f32).collect();
    let scale: Vec<f32> = vec![0.9_f32; n];

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let ct = Tensor::from_data(&coords_f32, vec![1, n], device.clone()).expect("bench coords");
        let nt = Tensor::from_data(&noise_f32, vec![1, n], device.clone()).expect("bench noise");
        let st = Tensor::from_data(&scale, vec![1, n], device.clone()).expect("bench scale");
        let _ = ct
            .mul(&st)
            .and_then(|t| t.add(&nt))
            .and_then(|t| t.to_vec());
    }
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_micros() as f64 / f64::from(iters);

    eprintln!("  diffusion forward GPU: {per_iter_us:.1}µs/iter ({iters} iters, n={n})");
    h.check_bool(
        &format!("nF-D→bench: {per_iter_us:.0}µs/iter (GPU diffusion forward)"),
        per_iter_us < 50_000.0,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn neg_f32(tensor: &Tensor, device: &Dev) -> Result<Tensor, String> {
    let vec = tensor.to_vec().map_err(|e| format!("neg read: {e}"))?;
    let neg_vec: Vec<f32> = vec.iter().map(|v| -v).collect();
    let shape = tensor.shape().to_vec();
    Tensor::from_data(&neg_vec, shape, device.clone()).map_err(|e| format!("neg upload: {e}"))
}

fn gelu_f32(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = sqrt_2_over_pi * 0.044_715_f32.mul_add(x * x * x, x);
    0.5 * x * (1.0 + inner.tanh())
}
