// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: Anderson Hamiltonian × wavefunction (Paper 023).
//!
//! Validates GPU `Tensor::matmul` for tight-binding Hamiltonian application,
//! eigenvalue bounds via Gershgorin discs, and Gram orthogonality checks for
//! Anderson localization (Paper 023) and spectral analysis (Paper 022).
//!
//! ## S-14 workaround
//!
//! Uses Ψ^T × Ψ (wavefunction Gram) via transpose to avoid the Naive
//! matmul hang.  H × Ψ is validated through the Gram + Rayleigh quotient.
//!
//! ## S-15 workaround
//!
//! `Tensor::matmul` hangs on RTX 4070 Vulkan when input buffers
//! have many elements with small magnitude (≤ 0.1), including sparse
//! matrices with exact zeros.  The Hamiltonian is made dense by adding
//! a uniform background coupling `BG = 0.5` to all off-tridiagonal
//! elements.  CPU reference uses the same dense Hamiltonian, so
//! correctness is preserved.  The true sparse Hamiltonian is validated
//! on CPU via `validate_barracuda_anderson`.
//!
//! ## Provenance
//!
//! CPU baseline: `validate_barracuda_anderson` (7 checks, jacobi vs `eigh_f64`).

#![expect(
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    reason = "validation binary"
)]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_tensor;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{gpu_readback, max_abs_diff_gpu_vs_cpu, ValidationHarness};
use std::sync::Arc;

fn cpu_gram(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = rows.len();
    let d = rows[0].len();
    let mut g = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..d {
                g[i][j] += rows[i][k] * rows[j][k];
            }
        }
    }
    g
}

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
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_gpu_anderson");

    validate_wavefunction_gram(&mut h, &device);
    validate_hamiltonian_product(&mut h, &device);
    validate_gershgorin_bounds(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

/// Wavefunction Gram matrix: Ψ (M×N) × Ψ^T (N×M) → G (M×M).
fn validate_wavefunction_gram(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(42);
    let m = 8_usize;
    let n = 16_usize;

    let psi: Vec<Vec<f64>> = (0..m)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    let cpu = cpu_gram(&psi);

    let flat: Vec<f32> = psi
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let p_t1 = gpu_tensor!(h, &flat, &[m, n], device);
    let p_t2 = gpu_tensor!(h, &flat, &[m, n], device);
    let p_t2_t = match p_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };
    let gram_t = match p_t1.matmul(&p_t2_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul Ψ×Ψ^T: {e}"), false);
            return;
        }
    };
    let Some(gram) = gpu_readback(h, &gram_t) else {
        return;
    };

    let cpu_flat: Vec<f64> = cpu.iter().flat_map(|r| r.iter().copied()).collect();
    let diff = max_abs_diff_gpu_vs_cpu(&gram, &cpu_flat);
    h.check_upper(
        &format!("Ψ Gram: max diff ({diff:.2e})"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );

    let min_diag = (0..m)
        .map(|i| f64::from(gram[i * m + i]))
        .fold(f64::INFINITY, f64::min);
    h.check_lower(
        &format!("Ψ Gram diag non-negative ({min_diag:.4})"),
        min_diag,
        -tolerances::TENSOR_EXACT_F32,
    );
}

/// H × Ψ via GPU: construct H (N×N) and Ψ batch (N×M), compute
/// product using H rows as "features" of a transpose-based matmul.
fn validate_hamiltonian_product(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(123);
    let n = 16_usize;
    let m = 4_usize;
    let hopping = 1.0_f64;

    let disorder: Vec<f64> = (0..n).map(|_| rng.uniform() * 4.0).collect();
    let bg = 0.5_f64;
    let mut h_rows: Vec<Vec<f64>> = vec![vec![bg; n]; n];
    for i in 0..n {
        h_rows[i][i] = disorder[i] + bg;
        if i + 1 < n {
            h_rows[i][i + 1] = hopping + bg;
            h_rows[i + 1][i] = hopping + bg;
        }
    }

    let psi_rows: Vec<Vec<f64>> = (0..m)
        .map(|_| (0..n).map(|_| rng.uniform()).collect())
        .collect();
    let cpu_psi_h: Vec<Vec<f64>> = psi_rows
        .iter()
        .map(|psi_row| {
            let mut out = vec![0.0_f64; n];
            for j in 0..n {
                for k in 0..n {
                    out[j] += psi_row[k] * h_rows[j][k];
                }
            }
            out
        })
        .collect();

    let psi_flat: Vec<f32> = psi_rows
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let h_flat: Vec<f32> = h_rows
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let psi_t = gpu_tensor!(h, &psi_flat, &[m, n], device);
    let h_t = gpu_tensor!(h, &h_flat, &[n, n], device);
    let h_t_t = match h_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose H: {e}"), false);
            return;
        }
    };

    let out_t = match psi_t.matmul(&h_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul Ψ×H^T: {e}"), false);
            return;
        }
    };
    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };

    let cpu_psi_h_flat: Vec<f64> = cpu_psi_h.iter().flat_map(|r| r.iter().copied()).collect();
    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_psi_h_flat);
    h.check_upper(
        &format!("Ψ×H^T: max diff ({diff:.2e})"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );

    h.check_bool("H×Ψ all finite", out.iter().all(|x| x.is_finite()));
}

/// Gershgorin: eigenvalue bounds from Hamiltonian diagonal and off-diagonal.
fn validate_gershgorin_bounds(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(99);
    let n = 16_usize;
    let m = 4_usize;

    let disorder: Vec<f64> = (0..n).map(|_| rng.uniform() * 6.0).collect();
    let bg = 0.5_f64;
    let mut h_rows: Vec<Vec<f64>> = vec![vec![bg; n]; n];
    for i in 0..n {
        h_rows[i][i] = disorder[i] + bg;
        if i + 1 < n {
            h_rows[i][i + 1] = 1.0 + bg;
            h_rows[i + 1][i] = 1.0 + bg;
        }
    }

    let mut gersh_min = f64::INFINITY;
    let mut gersh_max = f64::NEG_INFINITY;
    for i in 0..n {
        let diag = h_rows[i][i];
        let radius: f64 = (0..n).filter(|&j| j != i).map(|j| h_rows[i][j].abs()).sum();
        gersh_min = gersh_min.min(diag - radius);
        gersh_max = gersh_max.max(diag + radius);
    }

    let psi: Vec<Vec<f64>> = (0..m)
        .map(|_| {
            let mut v: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect();

    let cpu_psi_h: Vec<Vec<f64>> = psi
        .iter()
        .map(|psi_row| {
            let mut out = vec![0.0_f64; n];
            for j in 0..n {
                for k in 0..n {
                    out[j] += psi_row[k] * h_rows[j][k];
                }
            }
            out
        })
        .collect();

    let rayleigh: Vec<f64> = psi
        .iter()
        .zip(cpu_psi_h.iter())
        .map(|(p, hp)| p.iter().zip(hp.iter()).map(|(a, b)| a * b).sum::<f64>())
        .collect();

    let all_in_range = rayleigh
        .iter()
        .all(|&r| r >= gersh_min - 1.0 && r <= gersh_max + 1.0);

    h.check_bool(
        &format!("Rayleigh quotients in Gershgorin [{gersh_min:.1}, {gersh_max:.1}] ± 1.0"),
        all_in_range,
    );

    let psi_flat: Vec<f32> = psi
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let h_flat: Vec<f32> = h_rows
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let psi_t = gpu_tensor!(h, &psi_flat, &[m, n], device);
    let h_t = gpu_tensor!(h, &h_flat, &[n, n], device);
    let h_t_t = match h_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let out_t = match psi_t.matmul(&h_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };
    let Some(gpu_out) = gpu_readback(h, &out_t) else {
        return;
    };

    let cpu_psi_h_flat: Vec<f64> = cpu_psi_h.iter().flat_map(|r| r.iter().copied()).collect();
    let diff = max_abs_diff_gpu_vs_cpu(&gpu_out, &cpu_psi_h_flat);
    h.check_upper(
        &format!("Gershgorin Ψ×H^T: max diff ({diff:.2e})"),
        diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let m = 8_usize;
    let n = 16_usize;
    let x: Vec<f32> = (0..m * n).map(|_| rng.uniform() as f32).collect();

    let run = || -> Option<Vec<f32>> {
        let t1 = Tensor::from_data(&x, vec![m, n], device.clone()).ok()?;
        let t2 = Tensor::from_data(&x, vec![m, n], device.clone()).ok()?;
        let t2t = t2.transpose().ok()?;
        let out = t1.matmul(&t2t).ok()?;
        out.to_vec().ok()
    };

    let Some(r1) = run() else {
        h.check_bool("determinism run1 failed", false);
        return;
    };
    let Some(r2) = run() else {
        h.check_bool("determinism run2 failed", false);
        return;
    };

    let identical = r1
        .iter()
        .zip(r2.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    h.check_bool("determinism: two GPU runs bit-identical", identical);
}
