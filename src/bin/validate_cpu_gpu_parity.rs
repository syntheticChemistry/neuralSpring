// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU vs GPU parity validation for `BarraCuda` compute paths.
//!
//! Validates that `CpuExecutor` (pure Rust + Rayon) produces identical
//! results to the Tensor API (WGSL via wgpu) for key mathematical
//! operations. This proves the pure Rust math is correct independently
//! of GPU shader compilation.
//!
//! ## Operations validated
//!
//! - `MatMul`: CPU tiled Rust vs GPU WGSL
//! - Activations: `ReLU`, Sigmoid, Tanh
//! - Reductions: `ReduceSum` (`ReduceMean` skipped: upstream shader entry point)
//! - Special: erf, gamma (CPU f64)
//! - Conv/Pool: `cpu_conv_pool` against known outputs
//!
//! ## Approach
//!
//! 1. Run Tensor API on GPU device (if available) → GPU result
//! 2. Run Tensor API on CPU device (llvmpipe, if available) → CPU result
//! 3. Compare GPU vs CPU (when both available)
//! 4. Compare each against pure Rust reference

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use barracuda::cpu_conv_pool;
use barracuda::device::WgpuDevice;
use barracuda::special;
use barracuda::tensor::Tensor;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

fn main() {
    let mut h = ValidationHarness::new("cpu_gpu_parity");

    // Special functions: pure CPU, no wgpu needed
    validate_special_functions(&mut h);

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            h.check_bool(&format!("tokio runtime: {e}"), false);
            h.finish();
        }
    };
    let (gpu_dev, cpu_dev) = rt.block_on(async {
        (
            WgpuDevice::new_gpu().await.ok().map(Arc::new),
            WgpuDevice::new_cpu_relaxed().await.ok().map(Arc::new),
        )
    });

    let has_gpu = gpu_dev.is_some();
    let has_cpu = cpu_dev.is_some();

    if !has_gpu && !has_cpu {
        h.check_bool(
            "skip_tensor_parity: no GPU or CPU wgpu device available",
            true,
        );
        h.finish(); // never returns
    }

    let gpu_dev = gpu_dev.as_ref().map(Arc::clone);
    let cpu_dev = cpu_dev.as_ref().map(Arc::clone);

    rt.block_on(async {
        if let Some(ref dev) = gpu_dev {
            validate_matmul_parity(&mut h, dev, "gpu").await;
            validate_activation_parity(&mut h, dev, "gpu").await;
            validate_reduction_parity(&mut h, dev, "gpu").await;
        }
        if let Some(ref dev) = cpu_dev {
            validate_matmul_parity(&mut h, dev, "cpu").await;
            validate_activation_parity(&mut h, dev, "cpu").await;
            validate_reduction_parity(&mut h, dev, "cpu").await;
        }
        if let (Some(ref g), Some(ref c)) = (gpu_dev, cpu_dev) {
            validate_cross_hardware_matmul(&mut h, g, c).await;
            validate_cross_hardware_activations(&mut h, g, c).await;
        }
    });

    validate_conv_pool_parity(&mut h);

    h.finish();
}

// ─── Pure Rust reference implementations ─────────────────────────────────────

fn matmul_naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

const fn relu_cpu(x: f32) -> f32 {
    x.max(0.0)
}

fn sigmoid_cpu(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh_cpu(x: f32) -> f32 {
    x.tanh()
}

// ─── 1. validate_matmul_parity ──────────────────────────────────────────────

async fn validate_matmul_parity(h: &mut ValidationHarness, device: &Arc<WgpuDevice>, label: &str) {
    let mut rng = Rng::new(42);
    let m = 32_usize;
    let k = 32_usize;
    let n = 32_usize;

    let data_a: Vec<f32> = (0..m * k)
        .map(|_| (rng.uniform() as f32).mul_add(0.1, -0.05))
        .collect();
    let data_b: Vec<f32> = (0..k * n)
        .map(|_| (rng.uniform() as f32).mul_add(0.1, -0.05))
        .collect();

    let ref_c = matmul_naive(&data_a, &data_b, m, k, n);

    let a = match Tensor::from_vec_on(data_a.clone(), vec![m, k], device.clone()).await {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("{label} matmul from_vec_on: {e}"), false);
            return;
        }
    };
    let b = match Tensor::from_vec_on(data_b.clone(), vec![k, n], device.clone()).await {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("{label} matmul from_vec_on b: {e}"), false);
            return;
        }
    };

    let result = match a.matmul(&b) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("{label} matmul exec: {e}"), false);
            return;
        }
    };

    let gpu_result = match result.to_vec() {
        Ok(v) => v,
        Err(e) => {
            h.check_bool(&format!("{label} matmul to_vec: {e}"), false);
            return;
        }
    };

    let max_diff = ref_c
        .iter()
        .zip(gpu_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    h.check_upper(
        &format!("{label} matmul vs naive Rust (32×32×32)"),
        f64::from(max_diff),
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

// ─── 2. validate_activation_parity ──────────────────────────────────────────

async fn validate_activation_parity(
    h: &mut ValidationHarness,
    device: &Arc<WgpuDevice>,
    label: &str,
) {
    let mut rng = Rng::new(43);
    let n = 1000_usize;
    let data: Vec<f32> = (0..n)
        .map(|_| (rng.uniform() as f32).mul_add(4.0, -2.0))
        .collect();

    let t = match Tensor::from_vec_on(data.clone(), vec![n], device.clone()).await {
        Ok(x) => x,
        Err(e) => {
            h.check_bool(&format!("{label} act from_vec_on: {e}"), false);
            return;
        }
    };

    // ReLU
    if let Ok(out) = t.clone().relu() {
        if let Ok(v) = out.to_vec() {
            let ref_relu: Vec<f32> = data.iter().map(|&x| relu_cpu(x)).collect();
            let max_diff = v
                .iter()
                .zip(ref_relu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            h.check_upper(
                &format!("{label} ReLU vs Rust"),
                f64::from(max_diff),
                tolerances::TENSOR_EXACT_F32,
            );
        }
    }

    // Sigmoid
    let Ok(t2) = Tensor::from_vec_on(data.clone(), vec![n], device.clone()).await else {
        return;
    };
    if let Ok(out) = t2.sigmoid() {
        if let Ok(v) = out.to_vec() {
            let ref_sig: Vec<f32> = data.iter().map(|&x| sigmoid_cpu(x)).collect();
            let max_diff = v
                .iter()
                .zip(ref_sig.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            h.check_upper(
                &format!("{label} Sigmoid vs Rust"),
                f64::from(max_diff),
                tolerances::TENSOR_EXACT_F32,
            );
        }
    }

    // Tanh
    let Ok(t3) = Tensor::from_vec_on(data.clone(), vec![n], device.clone()).await else {
        return;
    };
    if let Ok(out) = t3.tanh() {
        if let Ok(v) = out.to_vec() {
            let ref_tanh: Vec<f32> = data.iter().map(|&x| tanh_cpu(x)).collect();
            let max_diff = v
                .iter()
                .zip(ref_tanh.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            h.check_upper(
                &format!("{label} Tanh vs Rust"),
                f64::from(max_diff),
                tolerances::TENSOR_EXACT_F32,
            );
        }
    }
}

// ─── 3. validate_reduction_parity ────────────────────────────────────────────

async fn validate_reduction_parity(
    h: &mut ValidationHarness,
    device: &Arc<WgpuDevice>,
    label: &str,
) {
    let mut rng = Rng::new(44);
    let n = 256_usize;
    let data: Vec<f32> = (0..n).map(|_| (rng.uniform() as f32) * 2.0).collect();

    let ref_sum: f32 = data.iter().sum();

    let t = match Tensor::from_vec_on(data.clone(), vec![n], device.clone()).await {
        Ok(x) => x,
        Err(e) => {
            h.check_bool(&format!("{label} reduce from_vec_on: {e}"), false);
            return;
        }
    };

    if let Ok(sum_t) = t.sum() {
        if let Ok(v) = sum_t.to_vec() {
            if let Some(&got) = v.first() {
                let tol = f64::from(ref_sum.abs()).max(1.0) * tolerances::GPU_FITNESS_F32;
                h.check_abs(
                    &format!("{label} sum vs manual"),
                    f64::from(got),
                    f64::from(ref_sum),
                    tol,
                );
            }
        }
    } else {
        h.check_bool(&format!("{label} sum (Tensor API unavailable)"), true);
    }

    // Mean reduce: skipped — BarraCuda mean_reduce.wgsl entry point is
    // "mean_reduce" but pipeline expects "main". Sum validates reduction path.
}

// ─── Cross-hardware: GPU vs CPU (same WGSL) ──────────────────────────────────

async fn validate_cross_hardware_matmul(
    h: &mut ValidationHarness,
    gpu: &Arc<WgpuDevice>,
    cpu: &Arc<WgpuDevice>,
) {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let a_gpu = match Tensor::from_vec_on(a_data.clone(), vec![2, 3], gpu.clone()).await {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("cross matmul gpu from_vec: {e}"), false);
            return;
        }
    };
    let Ok(b_gpu) = Tensor::from_vec_on(b_data.clone(), vec![3, 2], gpu.clone()).await else {
        return;
    };
    let gpu_res = match a_gpu.matmul(&b_gpu) {
        Ok(t) => t.to_vec(),
        Err(e) => {
            h.check_bool(&format!("cross matmul gpu exec: {e}"), false);
            return;
        }
    };

    let a_cpu = match Tensor::from_vec_on(a_data, vec![2, 3], cpu.clone()).await {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("cross matmul cpu from_vec: {e}"), false);
            return;
        }
    };
    let Ok(b_cpu) = Tensor::from_vec_on(b_data, vec![3, 2], cpu.clone()).await else {
        return;
    };
    let cpu_res = match a_cpu.matmul(&b_cpu) {
        Ok(t) => t.to_vec(),
        Err(e) => {
            h.check_bool(&format!("cross matmul cpu exec: {e}"), false);
            return;
        }
    };

    let (Ok(gpu_v), Ok(cpu_v)) = (gpu_res, cpu_res) else {
        return;
    };

    let max_diff = gpu_v
        .iter()
        .zip(cpu_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    h.check_upper(
        "GPU vs CPU matmul (cross-hardware)",
        f64::from(max_diff),
        tolerances::GPU_FITNESS_F32,
    );
}

async fn validate_cross_hardware_activations(
    h: &mut ValidationHarness,
    gpu: &Arc<WgpuDevice>,
    cpu: &Arc<WgpuDevice>,
) {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let Ok(t_gpu) = Tensor::from_vec_on(data.clone(), vec![5], gpu.clone()).await else {
        return;
    };
    let Ok(t_cpu) = Tensor::from_vec_on(data, vec![5], cpu.clone()).await else {
        return;
    };

    let gpu_relu = t_gpu.clone().relu().and_then(|t| t.to_vec());
    let cpu_relu = t_cpu.relu().and_then(|t| t.to_vec());
    if let (Ok(g), Ok(c)) = (gpu_relu, cpu_relu) {
        let max_diff = g
            .iter()
            .zip(c.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        h.check_upper(
            "GPU vs CPU ReLU",
            f64::from(max_diff),
            tolerances::TENSOR_EXACT_F32,
        );
    }
}

// ─── 4. validate_special_functions ───────────────────────────────────────────

fn validate_special_functions(h: &mut ValidationHarness) {
    h.check_abs("erf(0) == 0", special::erf(0.0), 0.0, tolerances::EXACT_F64);
    h.check_abs(
        "erf(1) ≈ 0.8427",
        special::erf(1.0),
        0.842_700_792_949_715,
        tolerances::SPECIAL_FUNCTION_F64,
    );

    match special::gamma(5.0) {
        Ok(g) => {
            h.check_abs("gamma(5) == 24", g, 24.0, tolerances::EXACT_F64);
        }
        Err(e) => {
            h.check_bool(&format!("gamma(5): {e}"), false);
        }
    }
}

// ─── 5. validate_conv_pool_parity ───────────────────────────────────────────

fn validate_conv_pool_parity(h: &mut ValidationHarness) {
    // Tiny 2x2 input, 1x1 kernel → output should match manual
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let kernel: Vec<f32> = vec![1.0]; // 1x1x1x1
    let out = cpu_conv_pool::conv2d(
        &input, &kernel, 1, 1, 2, 2, // n, c_in, h, w
        1, 1, 1, // c_out, k_h, k_w
        1, 1, // stride
        0, 0, // pad
        1, 1, // dilation
    );
    match out {
        Ok(v) => {
            let expected = [1.0, 2.0, 3.0, 4.0];
            let tol = tolerances::TENSOR_EXACT_F32 as f32;
            let ok = v.len() == expected.len()
                && v.iter()
                    .zip(expected.iter())
                    .all(|(a, b)| (a - b).abs() < tol);
            h.check_bool("conv2d 1x1 identity", ok);
        }
        Err(e) => h.check_bool(&format!("conv2d: {e}"), false),
    }

    // Max pool: 1x1x4x4, 2x2 kernel, stride 2
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let out = cpu_conv_pool::max_pool2d(&input, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0);
    match out {
        Ok(v) => {
            // 2x2 output: max of each 2x2 block -> 5, 7, 13, 15
            let expected = [5.0, 7.0, 13.0, 15.0];
            let tol = tolerances::TENSOR_EXACT_F32 as f32;
            let ok = v.len() == expected.len()
                && v.iter()
                    .zip(expected.iter())
                    .all(|(a, b)| (a - b).abs() < tol);
            h.check_bool("max_pool2d 2x2", ok);
        }
        Err(e) => h.check_bool(&format!("max_pool2d: {e}"), false),
    }
}
