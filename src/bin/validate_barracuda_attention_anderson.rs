// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 100: `BarraCUDA` CPU + GPU validator for attention Anderson spectral.

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::attention_anderson::load_attention_anderson_from_json;
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

const BASELINE_JSON: &str =
    include_str!("../../control/attention_anderson/attention_anderson_baseline.json");

type Dev = Arc<WgpuDevice>;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("barracuda_attention_anderson");

    println!("\n── Exp 100: BarraCUDA Attention Anderson ──");

    let baseline = match load_attention_anderson_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("JSON load", false);
            println!("FATAL: {e}");
            h.finish();
        }
    };

    h.check_bool("baseline loaded", !baseline.results.is_empty());

    // Tier 1: BarraCUDA CPU stats
    println!("\n── Tier 1: BarraCUDA CPU ──");

    let iprs: Vec<f64> = baseline.results.iter().map(|r| r.mean_ipr).collect();
    let xis: Vec<f64> = baseline.results.iter().map(|r| r.xi).collect();

    let r2 = barracuda::stats::r_squared(&iprs, &iprs);
    h.check_abs("bC CPU self-R²", r2, 1.0, tolerances::EXACT_F64);

    match barracuda::stats::correlation::pearson_correlation(&iprs, &xis) {
        Ok(r) => {
            h.check_bool("bC CPU Pearson(IPR, ξ) finite", r.is_finite());
            h.check_bool("bC CPU Pearson(IPR, ξ) < 0", r < 0.0);
        }
        Err(e) => {
            println!("  Pearson error: {e}");
            h.check_bool("bC CPU Pearson", false);
        }
    }

    // Tier 2: GPU on reference matrix
    println!("\n── Tier 2: BarraCUDA GPU ──");

    let Ok(gpu) = Gpu::new().await else {
        println!("  GPU unavailable");
        h.finish();
    };

    let device: Dev = gpu.wgpu_device().clone();
    validate_gpu(&mut h, &baseline, &device);

    h.finish();
}

#[expect(clippy::cast_possible_truncation, reason = "f64→f32 for GPU tensor")]
fn validate_gpu(
    h: &mut ValidationHarness,
    baseline: &neural_spring::attention_anderson::AttentionAndersonBaseline,
    device: &Dev,
) {
    let n = baseline.reference_n;
    let mat_f32: Vec<f32> = baseline
        .reference_matrix
        .iter()
        .map(|&v| v as f32)
        .collect();

    let cpu_trace: f64 = (0..n).map(|i| baseline.reference_matrix[i * n + i]).sum();

    if let Ok(t) = Tensor::from_data(&mat_f32, vec![n, n], device.clone()) {
        let id_f32: Vec<f32> = {
            let mut v = vec![0.0_f32; n * n];
            for i in 0..n {
                v[i * n + i] = 1.0;
            }
            v
        };
        if let Ok(id_t) = Tensor::from_data(&id_f32, vec![n, n], device.clone()) {
            if let Ok(result) = t.matmul_ref(&id_t) {
                if let Ok(vals) = result.to_vec() {
                    let gpu_trace: f64 = (0..n).map(|i| f64::from(vals[i * n + i])).sum();
                    let diff = (gpu_trace - cpu_trace).abs();
                    println!("  trace CPU={cpu_trace:.6}, GPU={gpu_trace:.6}, diff={diff:.2e}");
                    h.check_abs("GPU trace parity", gpu_trace, cpu_trace, 0.01);
                } else {
                    h.check_bool("GPU readback", false);
                }
            } else {
                h.check_bool("GPU matmul", false);
            }
        } else {
            h.check_bool("GPU identity", false);
        }
    } else {
        h.check_bool("GPU tensor", false);
    }

    // GPU determinism
    if let Ok(t1) = Tensor::from_data(&mat_f32, vec![n, n], device.clone())
        && let Ok(t2) = Tensor::from_data(&mat_f32, vec![n, n], device.clone())
    {
        if let (Ok(v1), Ok(v2)) = (t1.to_vec(), t2.to_vec()) {
            let same = v1
                .iter()
                .zip(v2.iter())
                .all(|(a, b)| (a - b).abs() < f32::EPSILON);
            h.check_bool("GPU deterministic", same);
        } else {
            h.check_bool("GPU deterministic", false);
        }
    }
}
