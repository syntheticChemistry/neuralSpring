// SPDX-License-Identifier: AGPL-3.0-or-later

//! coralForge + `AlphaFold3` pure-GPU validations: attention, triangle multiply,
//! confidence, PAE, diffusion, pairformer FFN/TriMul, and determinism.

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::primitives::sigmoid_f32;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

type Dev = Arc<WgpuDevice>;

// ═══════════════════════════════════════════════════════════════════
// 5. coralForge attention scores (nF-01): QK^T/√d
// ═══════════════════════════════════════════════════════════════════

pub fn validate_coral_attention(h: &mut ValidationHarness, device: &Dev) {
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

pub fn validate_coral_trimul(h: &mut ValidationHarness, device: &Dev) {
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

pub fn validate_af3_pldt(h: &mut ValidationHarness, device: &Dev) {
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

pub fn validate_af3_pae(h: &mut ValidationHarness, device: &Dev) {
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

pub fn validate_af3_diffusion_forward(h: &mut ValidationHarness, device: &Dev) {
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

pub fn validate_af3_pairformer_ffn(h: &mut ValidationHarness, device: &Dev) {
    let mut rng = Rng::new(951);
    let nn = 9_usize;
    let d = 4_usize;
    let d_h = 8_usize;

    let input: Vec<f32> = (0..nn * d).map(|_| rng.normal() as f32 * 0.3).collect();
    let w1: Vec<f32> = (0..d * d_h).map(|_| rng.normal() as f32 * 0.2).collect();
    let b1: Vec<f32> = (0..nn * d_h).map(|_| rng.normal() as f32 * 0.1).collect();
    let w2: Vec<f32> = (0..d_h * d).map(|_| rng.normal() as f32 * 0.2).collect();
    let b2: Vec<f32> = (0..nn * d).map(|_| rng.normal() as f32 * 0.1).collect();

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
// 11. AlphaFold3 Pairformer TriMul contraction (nF-03)
// ═══════════════════════════════════════════════════════════════════

pub fn validate_af3_pairformer_trimul(h: &mut ValidationHarness, device: &Dev) {
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

pub fn validate_determinism(h: &mut ValidationHarness, device: &Dev) {
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

    let run1 = super::gpu_mlp_forward(&x, &w1, &b1, &w2, &b2, in_d, hid_d, out_d, device);
    let run2 = super::gpu_mlp_forward(&x, &w1, &b1, &w2, &b2, in_d, hid_d, out_d, device);

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
