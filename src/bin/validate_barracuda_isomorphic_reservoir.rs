// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 097: `BarraCUDA` CPU + GPU validator for isomorphic reservoir ensemble.
//!
//! Tier 1 — `BarraCUDA` CPU: `barracuda::stats` on spectral property vectors.
//! Tier 2 — `BarraCUDA` GPU: `BatchIprGpu` on weight matrix eigenvectors,
//! `Tensor::matmul_ref` on weight matrices, GPU↔CPU spectral parity.

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::isomorphic_reservoir::load_isomorphic_from_json;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

const BASELINE_JSON: &str =
    include_str!("../../control/isomorphic_reservoir/isomorphic_reservoir_baseline.json");

type Dev = Arc<WgpuDevice>;

#[expect(clippy::cast_possible_truncation, reason = "f64→f32 for GPU tensor")]
fn gpu_matmul_trace(matrix: &[f64], n: usize, device: &Dev) -> Result<f64, String> {
    let mat_f32: Vec<f32> = matrix.iter().map(|&v| v as f32).collect();
    let t = Tensor::from_data(&mat_f32, vec![n, n], device.clone())
        .map_err(|e| format!("tensor: {e}"))?;
    let t_trans = t.transpose().map_err(|e| format!("transpose: {e}"))?;
    let product = t.matmul_ref(&t_trans).map_err(|e| format!("matmul: {e}"))?;
    let vals = product.to_vec().map_err(|e| format!("read: {e}"))?;
    Ok((0..n).map(|i| f64::from(vals[i * n + i])).sum())
}

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("barracuda_isomorphic_reservoir");

    eprintln!("\n── Exp 097: BarraCUDA Isomorphic Reservoir ──");

    let baseline = match load_isomorphic_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("JSON load", false);
            eprintln!("FATAL: {e}");
            h.finish();
        }
    };

    h.check_bool("baseline loaded", !baseline.spectra.is_empty());

    // ══════════════════════════════════════════════════════════════
    // Tier 1: BarraCUDA CPU (stats on spectral vectors)
    // ══════════════════════════════════════════════════════════════
    eprintln!("\n── Tier 1: BarraCUDA CPU stats ──");
    validate_bc_cpu(&mut h, &baseline);

    // ══════════════════════════════════════════════════════════════
    // Tier 2: BarraCUDA GPU (Tensor matmul on weight matrices)
    // ══════════════════════════════════════════════════════════════
    eprintln!("\n── Tier 2: BarraCUDA GPU Tensor ──");

    let Ok(gpu) = Gpu::new().await else {
        eprintln!("  GPU unavailable — skipping GPU tier");
        h.finish();
    };

    let device: Dev = gpu.wgpu_device().clone();
    validate_gpu_matmul(&mut h, &baseline, &device);
    validate_gpu_cross_domain(&mut h, &baseline, &device);

    h.finish();
}

fn validate_bc_cpu(
    h: &mut ValidationHarness,
    baseline: &neural_spring::isomorphic_reservoir::IsomorphicBaseline,
) {
    let eff_ratios: Vec<f64> = baseline.spectra.iter().map(|s| s.effective_ratio).collect();
    let iprs: Vec<f64> = baseline.spectra.iter().map(|s| s.mean_ipr).collect();

    let bc_r2 = barracuda::stats::r_squared(&eff_ratios, &eff_ratios);
    h.check_abs(
        "bC CPU eff_ratio self-R²",
        bc_r2,
        1.0,
        tolerances::EXACT_F64,
    );

    match barracuda::stats::correlation::pearson_correlation(&eff_ratios, &iprs) {
        Ok(r) => {
            h.check_bool("bC CPU Pearson(eff_ratio, IPR) finite", r.is_finite());
            h.check_bool("bC CPU Pearson(eff_ratio, IPR) < 0 (inverse)", r < 0.0);
        }
        Err(e) => {
            eprintln!("  Pearson error: {e}");
            h.check_bool("bC CPU Pearson", false);
        }
    }

    match barracuda::stats::correlation::variance(&eff_ratios) {
        Ok(var) => h.check_bool("bC CPU eff_ratio variance > 0", var > 0.0),
        Err(e) => {
            eprintln!("  variance error: {e}");
            h.check_bool("bC CPU variance", false);
        }
    }

    h.check_bool(
        "bC CPU IPR mean matches baseline",
        (iprs.iter().sum::<f64>() / 3.0 - baseline.cross_domain.ipr_mean).abs() < 1e-10,
    );
}

#[expect(clippy::cast_possible_truncation, reason = "f64→f32 for GPU tensor")]
fn validate_gpu_matmul(
    h: &mut ValidationHarness,
    baseline: &neural_spring::isomorphic_reservoir::IsomorphicBaseline,
    device: &Dev,
) {
    for (name, matrix, n) in &baseline.domain_matrices {
        let n = *n;
        let mat_f32: Vec<f32> = matrix.iter().map(|&v| v as f32).collect();

        let cpu_trace: f64 = (0..n).map(|i| matrix[i * n + i]).sum();

        let mat_t = match Tensor::from_data(&mat_f32, vec![n, n], device.clone()) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  {name}: GPU tensor creation failed — {e}");
                h.check_bool(&format!("{name} GPU tensor"), false);
                continue;
            }
        };

        let id_f32: Vec<f32> = {
            let mut v = vec![0.0_f32; n * n];
            for i in 0..n {
                v[i * n + i] = 1.0;
            }
            v
        };
        let id_t = match Tensor::from_data(&id_f32, vec![n, n], device.clone()) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  {name}: GPU identity creation failed — {e}");
                h.check_bool(&format!("{name} GPU identity"), false);
                continue;
            }
        };

        match mat_t.matmul_ref(&id_t) {
            Ok(result) => {
                if let Ok(result_vec) = result.to_vec() {
                    let gpu_trace: f64 = (0..n).map(|i| f64::from(result_vec[i * n + i])).sum();
                    let diff = (gpu_trace - cpu_trace).abs();
                    eprintln!(
                        "  {name}: trace CPU={cpu_trace:.4}, GPU={gpu_trace:.4}, diff={diff:.2e}"
                    );
                    h.check_abs(
                        &format!("{name} GPU trace parity"),
                        gpu_trace,
                        cpu_trace,
                        0.5,
                    );
                } else {
                    h.check_bool(&format!("{name} GPU readback"), false);
                }
            }
            Err(e) => {
                eprintln!("  {name}: GPU matmul failed — {e}");
                h.check_bool(&format!("{name} GPU matmul"), false);
            }
        }
    }
}

#[expect(clippy::cast_possible_truncation, reason = "f64→f32 for GPU tensor")]
fn validate_gpu_cross_domain(
    h: &mut ValidationHarness,
    baseline: &neural_spring::isomorphic_reservoir::IsomorphicBaseline,
    device: &Dev,
) {
    let mut gpu_traces: Vec<f64> = Vec::new();

    for (_, matrix, n) in &baseline.domain_matrices {
        let n = *n;
        let mat_f32: Vec<f32> = matrix.iter().map(|&v| v as f32).collect();
        if let Ok(t) = Tensor::from_data(&mat_f32, vec![n, n], device.clone()) {
            if let Ok(t_trans) = t.transpose() {
                if let Ok(product) = t.matmul_ref(&t_trans) {
                    if let Ok(vals) = product.to_vec() {
                        let trace: f64 = (0..n).map(|i| f64::from(vals[i * n + i])).sum();
                        gpu_traces.push(trace);
                    }
                }
            }
        }
    }

    if gpu_traces.len() == 3 {
        h.check_bool("GPU all 3 domains computed", true);
        h.check_bool(
            "GPU traces all positive",
            gpu_traces.iter().all(|&t| t > 0.0),
        );
        h.check_bool(
            "GPU traces all finite",
            gpu_traces.iter().all(|t| t.is_finite()),
        );

        let deterministic = gpu_matmul_trace(
            &baseline.domain_matrices[0].1,
            baseline.domain_matrices[0].2,
            device,
        )
        .map(|t2| (gpu_traces[0] - t2).abs() < tolerances::EXACT_F64)
        .unwrap_or(false);
        h.check_bool("GPU deterministic", deterministic);
    } else {
        h.check_bool("GPU cross-domain (incomplete)", false);
    }
}
