// SPDX-License-Identifier: AGPL-3.0-or-later

//! Phase 4 metalForge WGSL shader validation — direct dispatch.
//!
//! Validates four Phase 4 WGSL shaders against CPU reference implementations
//! via direct `gpu_shader_validation::dispatch_shader` + `read_buffer_f32`:
//!
//! | Shader | Algorithm | Paper |
//! |--------|-----------|-------|
//! | `hmm_backward_log.wgsl` | HMM backward pass (log-domain) | 016–018 (Liu) |
//! | `hmm_viterbi.wgsl` | HMM Viterbi decoding (log-domain) | 016–018 (Liu) |
//! | `matrix_correlation.wgsl` | Pearson correlation of N×N matrices | 024–025 (Anderson) |
//! | `linear_regression.wgsl` | OLS via normal equations | baseCamp Sub-03 |
//!
//! These shaders use f32 buffers. CPU references use f64 — tolerance allows
//! for f32 precision loss (~1e-3 to 1e-5 depending on accumulation depth).
//!
//! `ToadStool` absorption targets:
//! - `hmm_backward_log` → `barracuda::ops::bio::hmm_backward`
//! - `hmm_viterbi` → `barracuda::ops::bio::hmm_viterbi`
//! - `matrix_correlation` → `barracuda::stats::matrix_correlation_gpu`
//! - `linear_regression` → `barracuda::stats::linear_regression_gpu`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HmmParams {
    n_states: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ViterbiParams {
    n_states: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CorrParams {
    n: u32,
    total_pairs: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RegParams {
    n: u32,
    _pad: [u32; 3],
}

use neural_spring::gpu::Gpu;
use neural_spring::gpu_shader_validation::{dispatch_shader, wg1d, ShaderBinding};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("gpu_shader_phase4");

    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "GPU: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(e) => {
            eprintln!("No GPU available ({e}), skipping");
            h.finish();
        }
    };

    let mut rng = Rng::new(42);

    validate_hmm_backward_log(&mut h, &gpu, &mut rng);
    validate_hmm_viterbi(&mut h, &gpu, &mut rng);
    validate_matrix_correlation(&mut h, &gpu, &mut rng);
    validate_linear_regression(&mut h, &gpu, &mut rng);

    h.finish();
}

// ─────────────────────────────────────────────────────────────────────
// HMM backward pass in log-domain
// ─────────────────────────────────────────────────────────────────────

fn validate_hmm_backward_log(h: &mut ValidationHarness, gpu: &Gpu, rng: &mut Rng) {
    let n = 4_u32;
    let nn = n as usize;

    let log_a: Vec<f32> = {
        let mut m = vec![0.0f32; nn * nn];
        for i in 0..nn {
            let raw: Vec<f64> = (0..nn).map(|_| rng.uniform().abs() + 0.01).collect();
            let s: f64 = raw.iter().sum();
            for (j, &r) in raw.iter().enumerate() {
                m[i * nn + j] = (r / s).ln() as f32;
            }
        }
        m
    };
    let log_b_col: Vec<f32> = (0..nn)
        .map(|_| (rng.uniform().abs() + 0.01).ln() as f32)
        .collect();
    let log_beta_next: Vec<f32> = (0..nn)
        .map(|_| (rng.uniform().abs() + 0.01).ln() as f32)
        .collect();

    let cpu_result: Vec<f32> = (0..nn)
        .map(|i| {
            let mut max_val: f32 = -1e30;
            for j in 0..nn {
                let val = log_a[i * nn + j] + log_b_col[j] + log_beta_next[j];
                max_val = max_val.max(val);
            }
            let mut sum_exp: f32 = 0.0;
            for j in 0..nn {
                let val = log_a[i * nn + j] + log_b_col[j] + log_beta_next[j];
                sum_exp += (val - max_val).exp();
            }
            max_val + sum_exp.ln()
        })
        .collect();

    let device = gpu.device();
    let shader_src = neural_spring_forge::shaders::HMM_BACKWARD_LOG;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hmm_backward_log"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let log_a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("log_a"),
        contents: bytemuck::cast_slice(&log_a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let log_b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("log_b_col"),
        contents: bytemuck::cast_slice(&log_b_col),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let beta_next_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("log_beta_next"),
        contents: bytemuck::cast_slice(&log_beta_next),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let beta_cur_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("log_beta_cur"),
        size: (nn * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = HmmParams {
        n_states: n,
        _pad: [0; 3],
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    dispatch_shader(
        gpu,
        &shader,
        "hmm_backward_log",
        &[
            ShaderBinding::StorageRo(&log_a_buf),
            ShaderBinding::StorageRo(&log_b_buf),
            ShaderBinding::StorageRo(&beta_next_buf),
            ShaderBinding::StorageRw(&beta_cur_buf),
            ShaderBinding::Uniform(&params_buf),
        ],
        wg1d(n),
    );

    match gpu.read_buffer_f32(&beta_cur_buf, nn) {
        Ok(gpu_result) => {
            let max_diff = cpu_result
                .iter()
                .zip(gpu_result.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0_f32, f32::max);

            h.check_abs(
                "HMM backward: GPU ↔ CPU max diff",
                f64::from(max_diff),
                0.0,
                tolerances::GPU_FITNESS_F32,
            );

            for (i, (&c, &g)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
                h.check_bool(
                    &format!("HMM backward: state {i} finite"),
                    g.is_finite() && c.is_finite(),
                );
            }
        }
        Err(e) => h.check_bool(&format!("HMM backward readback: {e}"), false),
    }
}

// ─────────────────────────────────────────────────────────────────────
// HMM Viterbi decoding in log-domain
// ─────────────────────────────────────────────────────────────────────

fn validate_hmm_viterbi(h: &mut ValidationHarness, gpu: &Gpu, rng: &mut Rng) {
    let n = 4_u32;
    let nn = n as usize;

    let log_a: Vec<f32> = {
        let mut m = vec![0.0f32; nn * nn];
        for i in 0..nn {
            let raw: Vec<f64> = (0..nn).map(|_| rng.uniform().abs() + 0.01).collect();
            let s: f64 = raw.iter().sum();
            for (j, &r) in raw.iter().enumerate() {
                m[i * nn + j] = (r / s).ln() as f32;
            }
        }
        m
    };
    let log_b_col: Vec<f32> = (0..nn)
        .map(|_| (rng.uniform().abs() + 0.01).ln() as f32)
        .collect();
    let delta_prev: Vec<f32> = (0..nn)
        .map(|_| (rng.uniform().abs() + 0.01).ln() as f32)
        .collect();

    let (cpu_delta, cpu_psi): (Vec<f32>, Vec<u32>) = {
        let mut deltas = Vec::with_capacity(nn);
        let mut psis = Vec::with_capacity(nn);
        for j in 0..nn {
            let mut best_val: f32 = -1e30;
            let mut best_i: u32 = 0;
            for i in 0..nn {
                let val = delta_prev[i] + log_a[i * nn + j];
                if val > best_val {
                    best_val = val;
                    best_i = i as u32;
                }
            }
            deltas.push(best_val + log_b_col[j]);
            psis.push(best_i);
        }
        (deltas, psis)
    };

    let device = gpu.device();
    let shader_src = neural_spring_forge::shaders::HMM_VITERBI;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hmm_viterbi"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let log_a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("log_a"),
        contents: bytemuck::cast_slice(&log_a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let log_b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("log_b_col"),
        contents: bytemuck::cast_slice(&log_b_col),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let delta_prev_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("delta_prev"),
        contents: bytemuck::cast_slice(&delta_prev),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let delta_cur_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("delta_cur"),
        size: (nn * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let psi_cur_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("psi_cur"),
        size: (nn * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = ViterbiParams {
        n_states: n,
        _pad: [0; 3],
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    dispatch_shader(
        gpu,
        &shader,
        "hmm_viterbi",
        &[
            ShaderBinding::StorageRo(&log_a_buf),
            ShaderBinding::StorageRo(&log_b_buf),
            ShaderBinding::StorageRo(&delta_prev_buf),
            ShaderBinding::StorageRw(&delta_cur_buf),
            ShaderBinding::StorageRw(&psi_cur_buf),
            ShaderBinding::Uniform(&params_buf),
        ],
        wg1d(n),
    );

    match gpu.read_buffer_f32(&delta_cur_buf, nn) {
        Ok(gpu_delta) => {
            let max_diff = cpu_delta
                .iter()
                .zip(gpu_delta.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0_f32, f32::max);

            h.check_abs(
                "Viterbi: delta GPU ↔ CPU max diff",
                f64::from(max_diff),
                0.0,
                tolerances::DISPATCH_VITERBI_F32,
            );

            for (j, (&c, &g)) in cpu_delta.iter().zip(gpu_delta.iter()).enumerate() {
                h.check_bool(
                    &format!("Viterbi: delta[{j}] finite"),
                    g.is_finite() && c.is_finite(),
                );
            }
        }
        Err(e) => h.check_bool(&format!("Viterbi delta readback: {e}"), false),
    }

    // psi buffer contains u32 backpointers — read raw bytes and interpret
    match gpu.read_buffer_f32(&psi_cur_buf, nn) {
        Ok(psi_raw) => {
            for (j, (&c, &raw)) in cpu_psi.iter().zip(psi_raw.iter()).enumerate() {
                let g = u32::from_le_bytes(raw.to_le_bytes());
                h.check_abs(
                    &format!("Viterbi: psi[{j}] match"),
                    f64::from(g),
                    f64::from(c),
                    tolerances::BOOLEAN_VALIDATION_SLACK,
                );
            }
        }
        Err(e) => h.check_bool(&format!("Viterbi psi readback: {e}"), false),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Matrix correlation (Pearson of upper triangle)
// ─────────────────────────────────────────────────────────────────────

fn validate_matrix_correlation(h: &mut ValidationHarness, gpu: &Gpu, rng: &mut Rng) {
    let n = 8_u32;
    let nn = n as usize;
    let total_pairs = nn * (nn - 1) / 2;

    let mat_a: Vec<f32> = (0..nn * nn).map(|_| rng.normal() as f32).collect();
    let mat_b: Vec<f32> = (0..nn * nn).map(|_| rng.normal() as f32).collect();

    let cpu_corr = {
        let mut a_vals = Vec::with_capacity(total_pairs);
        let mut b_vals = Vec::with_capacity(total_pairs);
        for i in 0..nn {
            for j in (i + 1)..nn {
                a_vals.push(f64::from(mat_a[i * nn + j]));
                b_vals.push(f64::from(mat_b[i * nn + j]));
            }
        }
        let n_f = a_vals.len() as f64;
        let mean_a = a_vals.iter().sum::<f64>() / n_f;
        let mean_b = b_vals.iter().sum::<f64>() / n_f;
        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;
        for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
            cov += (a - mean_a) * (b - mean_b);
            var_a += (a - mean_a).powi(2);
            var_b += (b - mean_b).powi(2);
        }
        let denom = (var_a * var_b).sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            cov / denom
        }
    };

    let device = gpu.device();
    let shader_src = neural_spring_forge::shaders::MATRIX_CORRELATION;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matrix_correlation"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let n_workgroups = (total_pairs as u32).div_ceil(256);

    let mat_a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mat_a"),
        contents: bytemuck::cast_slice(&mat_a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mat_b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mat_b"),
        contents: bytemuck::cast_slice(&mat_b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let partials_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("partials"),
        size: (n_workgroups as usize * 6 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = CorrParams {
        n,
        total_pairs: total_pairs as u32,
        _pad: [0; 2],
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    dispatch_shader(
        gpu,
        &shader,
        "matrix_correlation",
        &[
            ShaderBinding::StorageRo(&mat_a_buf),
            ShaderBinding::StorageRo(&mat_b_buf),
            ShaderBinding::StorageRw(&partials_buf),
            ShaderBinding::Uniform(&params_buf),
        ],
        (n_workgroups, 1, 1),
    );

    match gpu.read_buffer_f32(&partials_buf, n_workgroups as usize * 6) {
        Ok(partials) => {
            let mut sum_a = 0.0_f64;
            let mut sum_b = 0.0_f64;
            let mut sum_ab = 0.0_f64;
            let mut sum_a2 = 0.0_f64;
            let mut sum_b2 = 0.0_f64;
            let mut count = 0.0_f64;

            for wg in 0..n_workgroups as usize {
                let base = wg * 6;
                sum_a += f64::from(partials[base]);
                sum_b += f64::from(partials[base + 1]);
                sum_ab += f64::from(partials[base + 2]);
                sum_a2 += f64::from(partials[base + 3]);
                sum_b2 += f64::from(partials[base + 4]);
                count += f64::from(partials[base + 5]);
            }

            let gpu_corr = if count > 0.0 {
                let n_f = count;
                let cov = sum_ab - sum_a * sum_b / n_f;
                let va = sum_a2 - sum_a * sum_a / n_f;
                let vb = sum_b2 - sum_b * sum_b / n_f;
                let denom = (va * vb).sqrt();
                if denom < 1e-12 {
                    0.0
                } else {
                    cov / denom
                }
            } else {
                0.0
            };

            h.check_abs(
                "Matrix corr: GPU ↔ CPU Pearson",
                gpu_corr,
                cpu_corr,
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );

            h.check_abs(
                "Matrix corr: count matches total_pairs",
                count,
                total_pairs as f64,
                tolerances::BOOLEAN_VALIDATION_SLACK,
            );

            h.check_bool(
                "Matrix corr: r in [-1, 1]",
                (-1.01..=1.01).contains(&gpu_corr),
            );
        }
        Err(e) => h.check_bool(&format!("Matrix corr readback: {e}"), false),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Linear regression via normal equations
// ─────────────────────────────────────────────────────────────────────

fn validate_linear_regression(h: &mut ValidationHarness, gpu: &Gpu, rng: &mut Rng) {
    let n = 128_u32;
    let nn = n as usize;

    let true_a: f64 = 2.5;
    let true_b: f64 = -1.0;
    let x: Vec<f32> = (0..nn).map(|i| (i as f32) / (nn as f32)).collect();
    let y: Vec<f32> = x
        .iter()
        .map(|&xi| {
            rng.normal()
                .mul_add(0.01, true_a.mul_add(f64::from(xi), true_b)) as f32
        })
        .collect();

    let cpu_ab = {
        let n_f = nn as f64;
        let sx: f64 = x.iter().map(|&v| f64::from(v)).sum();
        let sy: f64 = y.iter().map(|&v| f64::from(v)).sum();
        let sxx: f64 = x.iter().map(|&v| f64::from(v).powi(2)).sum();
        let sxy: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| f64::from(xi) * f64::from(yi))
            .sum();
        #[allow(clippy::suspicious_operation_groupings)]
        let denom = n_f.mul_add(sxx, -(sx * sx));
        if denom.abs() < 1e-15 {
            (0.0, 0.0)
        } else {
            let a = n_f.mul_add(sxy, -(sx * sy)) / denom;
            let b = sxx.mul_add(sy, -(sx * sxy)) / denom;
            (a, b)
        }
    };

    let device = gpu.device();
    let shader_src = neural_spring_forge::shaders::LINEAR_REGRESSION;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("linear_regression"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let n_workgroups = n.div_ceil(256);

    let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("x"),
        contents: bytemuck::cast_slice(&x),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let y_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("y"),
        contents: bytemuck::cast_slice(&y),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let partials_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("partials"),
        size: (n_workgroups as usize * 5 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = RegParams { n, _pad: [0; 3] };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    dispatch_shader(
        gpu,
        &shader,
        "linear_regression",
        &[
            ShaderBinding::StorageRo(&x_buf),
            ShaderBinding::StorageRo(&y_buf),
            ShaderBinding::StorageRw(&partials_buf),
            ShaderBinding::Uniform(&params_buf),
        ],
        (n_workgroups, 1, 1),
    );

    match gpu.read_buffer_f32(&partials_buf, n_workgroups as usize * 5) {
        Ok(partials) => {
            let mut sx = 0.0_f64;
            let mut sy = 0.0_f64;
            let mut sxx = 0.0_f64;
            let mut sxy = 0.0_f64;
            let mut count = 0.0_f64;

            for wg in 0..n_workgroups as usize {
                let base = wg * 5;
                sx += f64::from(partials[base]);
                sy += f64::from(partials[base + 1]);
                sxx += f64::from(partials[base + 2]);
                sxy += f64::from(partials[base + 3]);
                count += f64::from(partials[base + 4]);
            }

            #[allow(clippy::suspicious_operation_groupings)]
            let denom = count.mul_add(sxx, -(sx * sx));
            let (gpu_a, gpu_b) = if denom.abs() < 1e-15 {
                (0.0, 0.0)
            } else {
                (
                    count.mul_add(sxy, -(sx * sy)) / denom,
                    sxx.mul_add(sy, -(sx * sxy)) / denom,
                )
            };

            h.check_abs(
                "Linear reg: slope GPU ↔ CPU",
                gpu_a,
                cpu_ab.0,
                tolerances::GPU_MEAN_DISPATCH_F32,
            );
            h.check_abs(
                "Linear reg: intercept GPU ↔ CPU",
                gpu_b,
                cpu_ab.1,
                tolerances::GPU_MEAN_DISPATCH_F32,
            );
            h.check_abs(
                "Linear reg: slope near true",
                gpu_a,
                true_a,
                tolerances::GPU_VARIANCE_DISPATCH_F32,
            );
            h.check_abs(
                "Linear reg: intercept near true",
                gpu_b,
                true_b,
                tolerances::GPU_VARIANCE_DISPATCH_F32,
            );
            h.check_abs(
                "Linear reg: count matches n",
                count,
                nn as f64,
                tolerances::BOOLEAN_VALIDATION_SLACK,
            );
        }
        Err(e) => h.check_bool(&format!("Linear reg readback: {e}"), false),
    }
}
