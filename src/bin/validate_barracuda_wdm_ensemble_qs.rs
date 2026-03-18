// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 098: `BarraCUDA` CPU + GPU validator for WDM ensemble QS.
//!
//! Tier 1 — `BarraCUDA` CPU: `barracuda::stats` on coupling vectors.
//! Tier 2 — `BarraCUDA` GPU: `Tensor` operations on disorder fields.

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::wdm_ensemble_qs::load_ensemble_from_json;
use std::sync::Arc;

const BASELINE_JSON: &str =
    include_str!("../../control/wdm_ensemble_qs/wdm_ensemble_qs_baseline.json");

type Dev = Arc<WgpuDevice>;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("barracuda_wdm_ensemble_qs");

    println!("\n── Exp 098: BarraCUDA WDM Ensemble QS ──");

    let baseline = match load_ensemble_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("JSON load", false);
            println!("FATAL: {e}");
            h.finish();
        }
    };

    h.check_bool("baseline loaded", !baseline.slices.is_empty());

    // Tier 1: BarraCUDA CPU stats
    println!("\n── Tier 1: BarraCUDA CPU ──");
    validate_bc_cpu(&mut h, &baseline);

    // Tier 2: BarraCUDA GPU Tensor
    println!("\n── Tier 2: BarraCUDA GPU ──");
    let Ok(gpu) = Gpu::new().await else {
        println!("  GPU unavailable");
        h.finish();
    };
    let device: Dev = gpu.wgpu_device().clone();
    validate_gpu(&mut h, &baseline, &device);

    h.finish();
}

fn validate_bc_cpu(
    h: &mut ValidationHarness,
    baseline: &neural_spring::wdm_ensemble_qs::EnsembleBaseline,
) {
    let ws: Vec<f64> = baseline.slices.iter().map(|s| s.mean_w).collect();
    let xis: Vec<f64> = baseline.slices.iter().map(|s| s.xi).collect();

    match barracuda::stats::correlation::pearson_correlation(&ws, &xis) {
        Ok(r) => {
            h.check_abs(
                "bC CPU r(W,ξ)",
                r,
                baseline.r_w_xi,
                tolerances::CORRELATION_CROSS_VALIDATION,
            );
            h.check_bool("bC CPU r(W,ξ) < 0", r < 0.0);
        }
        Err(e) => {
            println!("  Pearson error: {e}");
            h.check_bool("bC CPU Pearson", false);
        }
    }

    match barracuda::stats::correlation::variance(&ws) {
        Ok(v) => h.check_bool("bC CPU W variance > 0", v > 0.0),
        Err(e) => {
            println!("  variance error: {e}");
            h.check_bool("bC CPU variance", false);
        }
    }

    let r2 = barracuda::stats::r_squared(&ws, &ws);
    h.check_abs("bC CPU self-R²", r2, 1.0, tolerances::EXACT_F64);
}

#[expect(clippy::cast_possible_truncation, reason = "f64→f32 for GPU tensor")]
fn validate_gpu(
    h: &mut ValidationHarness,
    baseline: &neural_spring::wdm_ensemble_qs::EnsembleBaseline,
    device: &Dev,
) {
    let ref_d = &baseline.reference_disorder;
    let n = ref_d.len();
    let ref_f32: Vec<f32> = ref_d.iter().map(|&v| v as f32).collect();

    let cpu_sum: f64 = ref_d.iter().sum();

    if let Ok(t) = Tensor::from_data(&ref_f32, vec![1, n], device.clone()) {
        if let Ok(ones) = Tensor::from_data(&vec![1.0_f32; n], vec![n, 1], device.clone()) {
            if let Ok(result) = t.matmul_ref(&ones) {
                if let Ok(vals) = result.to_vec() {
                    let gpu_sum = f64::from(vals[0]);
                    let diff = (gpu_sum - cpu_sum).abs();
                    println!("  disorder sum: CPU={cpu_sum:.4}, GPU={gpu_sum:.4}, diff={diff:.2e}");
                    h.check_abs(
                        "GPU disorder sum",
                        gpu_sum,
                        cpu_sum,
                        tolerances::GPU_ACCUMULATION_F32,
                    );
                } else {
                    h.check_bool("GPU readback", false);
                }
            } else {
                h.check_bool("GPU matmul", false);
            }
        } else {
            h.check_bool("GPU ones", false);
        }
    } else {
        h.check_bool("GPU tensor", false);
    }

    // GPU determinism
    if let Ok(t1) = Tensor::from_data(&ref_f32, vec![1, n], device.clone())
        && let Ok(t2) = Tensor::from_data(&ref_f32, vec![1, n], device.clone())
    {
        if let (Ok(r1), Ok(r2)) = (t1.to_vec(), t2.to_vec()) {
            let same = r1
                .iter()
                .zip(r2.iter())
                .all(|(a, b)| (a - b).abs() < f32::EPSILON);
            h.check_bool("GPU deterministic", same);
        } else {
            h.check_bool("GPU deterministic", false);
        }
    }
}
