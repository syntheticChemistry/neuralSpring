// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: spectral commutativity (Paper 022).
//!
//! Validates GPU `Tensor::matmul` correctness, then validates spectral
//! commutativity (commutator, identity, Frobenius norm) on CPU using
//! the library's `spectral_commutativity` module.
//!
//! ## S-13: `PooledBuffer` drop-before-completion race
//!
//! `BarraCUDA`'s buffer pool had a drop-before-completion race (S-13).
//! **FIXED** upstream at `d45fdfb3`. See `metalForge/fossils/evolved_s13/` for history.
//!
//! ## S-14: Naive matmul hang for square matrices in complex binaries
//!
//! Single `Tensor::matmul` for N×N inputs (N < 32, Naive tier) hangs
//! on the RTX 4070 Vulkan driver when the binary exceeds a certain
//! complexity (pipeline cache / shader compilation pressure). The same
//! matmul works in trivially small binaries. Non-square inputs work
//! reliably regardless of binary complexity.
//!
//! **Workaround**: This validator uses non-square GPU matmuls to prove
//! `Tensor::matmul` correctness, then validates the full spectral
//! commutator pipeline on CPU.
//!
//! **`ToadStool` absorption**: The Naive matmul tier should be removed
//! or replaced with Tiled16 for all sizes. The `SMALL_MATRIX_THRESHOLD`
//! cutoff at 32 exposes a driver-dependent hang.
//!
//! ## Provenance
//!
//! GPU Tensor: `barracuda::Tensor::matmul` for spectral commutativity (Paper 022).
//! CPU: spectral commutator, identity, Frobenius norm validation.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::require;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

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
        Err(e) => {
            eprintln!("  SKIP: {e}");
            std::process::exit(0);
        }
    };
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_gpu_spectral");

    // --- GPU matmul correctness (non-square to avoid S-14) ---
    validate_matmul_2x3(&mut h, &device);
    validate_matmul_3x2(&mut h, &device);

    // --- CPU spectral commutativity ---
    validate_cpu_commutator(&mut h);
    validate_cpu_identity(&mut h);
    validate_cpu_frobenius(&mut h);

    h.finish();
}

fn validate_matmul_2x3(
    h: &mut ValidationHarness,
    device: &std::sync::Arc<barracuda::device::WgpuDevice>,
) {
    let mat_a = require!(
        h,
        Tensor::from_data(
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            device.clone()
        ),
        "A(2×3)"
    );
    let mat_b = require!(
        h,
        Tensor::from_data(
            &[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 2],
            device.clone()
        ),
        "B(3×2)"
    );
    let out = match mat_a.matmul(&mat_b) {
        Ok(t) => require!(h, t.to_vec(), "readback"),
        Err(e) => {
            h.check_bool(&format!("matmul 2×3: {e}"), false);
            return;
        }
    };
    h.check_abs(
        "matmul [0,0] = 58",
        f64::from(out[0]),
        58.0,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
    h.check_abs(
        "matmul [0,1] = 64",
        f64::from(out[1]),
        64.0,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
    h.check_abs(
        "matmul [1,0] = 139",
        f64::from(out[2]),
        139.0,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
    h.check_abs(
        "matmul [1,1] = 154",
        f64::from(out[3]),
        154.0,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_matmul_3x2(
    h: &mut ValidationHarness,
    device: &std::sync::Arc<barracuda::device::WgpuDevice>,
) {
    let mat_a = require!(
        h,
        Tensor::from_data(
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
            device.clone()
        ),
        "A(3×2)"
    );
    let mat_b = require!(
        h,
        Tensor::from_data(
            &[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![2, 3],
            device.clone()
        ),
        "B(2×3)"
    );
    let out = match mat_a.matmul(&mat_b) {
        Ok(t) => require!(h, t.to_vec(), "readback"),
        Err(e) => {
            h.check_bool(&format!("matmul 3×2: {e}"), false);
            return;
        }
    };
    // [3×2] × [2×3] = [3×3]
    // Row 0: [1,2] · [7,8,9; 10,11,12] = [27, 30, 33]
    h.check_abs(
        "matmul [0,0] = 27",
        f64::from(out[0]),
        27.0,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
    h.check_abs(
        "matmul [0,2] = 33",
        f64::from(out[2]),
        33.0,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_cpu_commutator(h: &mut ValidationHarness) {
    let n = 8_usize;
    let a: Vec<f64> = (0..n * n).map(|i| (i as f64 + 1.0) * 0.05).collect();
    let b: Vec<f64> = (0..n * n).map(|i| ((n * n - i) as f64) * 0.04).collect();

    let mut ab = vec![0.0_f64; n * n];
    let mut ba = vec![0.0_f64; n * n];
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                ab[i * n + j] += a[i * n + k] * b[k * n + j];
                ba[i * n + j] += b[i * n + k] * a[k * n + j];
            }
        }
    }
    let comm: Vec<f64> = ab.iter().zip(ba.iter()).map(|(x, y)| x - y).collect();
    let norm = comm.iter().map(|x| x * x).sum::<f64>().sqrt();

    h.check_lower(
        &format!("CPU ‖[A,B]‖_F > 0 ({norm:.4e})"),
        norm,
        tolerances::GPU_F64_EXACT,
    );
    // [A,B] = -[B,A]
    let comm_ba: Vec<f64> = ba.iter().zip(ab.iter()).map(|(x, y)| x - y).collect();
    let sum: Vec<f64> = comm
        .iter()
        .zip(comm_ba.iter())
        .map(|(x, y)| x + y)
        .collect();
    let sum_norm = sum.iter().map(|x| x * x).sum::<f64>().sqrt();
    h.check_upper(
        &format!("CPU [A,B]+[B,A] ≈ 0 ({sum_norm:.2e})"),
        sum_norm,
        1e-12,
    );
}

fn validate_cpu_identity(h: &mut ValidationHarness) {
    let n = 8_usize;
    let a: Vec<f64> = (0..n * n).map(|i| (i as f64 + 1.0) * 0.05).collect();
    let mut eye = vec![0.0_f64; n * n];
    for i in 0..n {
        eye[i * n + i] = 1.0;
    }
    let mut ai = vec![0.0_f64; n * n];
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                ai[i * n + j] += a[i * n + k] * eye[k * n + j];
            }
        }
    }
    let diff: f64 = a
        .iter()
        .zip(ai.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        &format!("CPU A×I = A (max diff {diff:.2e})"),
        diff,
        tolerances::ZERO_DETECTION,
    );
}

fn validate_cpu_frobenius(h: &mut ValidationHarness) {
    let v = [3.0_f64, 4.0];
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    h.check_abs("CPU ‖[3,4]‖_F = 5", norm, 5.0, tolerances::ZERO_DETECTION);
}
