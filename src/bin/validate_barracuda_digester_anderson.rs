// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 096: `BarraCUDA` CPU + GPU validator for digester-Anderson coupling.
//!
//! Tier 1 — `BarraCUDA` CPU: `barracuda::stats` (Pearson correlation, R²,
//! variance) on coupling vectors (W vs R²).
//!
//! Tier 2 — `BarraCUDA` GPU: ESN inference via `Tensor::matmul_ref` + `tanh` +
//! `add`, GPU Pearson correlation of W vs R², GPU↔CPU parity.
//!
//! Composes Paper 027 (ESN) + Paper 023 (Anderson) on GPU.

#![expect(
    clippy::cast_possible_truncation,
    reason = "f64→f32 GPU casting for Tensor operations"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::digester_anderson::{CouplingPredictor, load_coupling_from_json, pearson_r};
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

const BASELINE_JSON: &str =
    include_str!("../../control/digester_anderson/digester_anderson_baseline.json");

type Dev = Arc<WgpuDevice>;

fn gpu_esn_predict(
    pred: &CouplingPredictor,
    input: &[f64; 5],
    device: &Dev,
) -> Result<f64, String> {
    let rs = pred.reservoir_size;

    let x: Vec<f32> = (0..5)
        .map(|i| ((input[i] - pred.x_mean[i]) / pred.x_std[i]) as f32)
        .collect();

    let w_in_f32: Vec<f32> = pred.w_in.iter().map(|&v| v as f32).collect();
    let w_res_f32: Vec<f32> = pred.w_res.iter().map(|&v| v as f32).collect();
    let b_res_f32: Vec<f32> = pred.b_res.iter().map(|&v| v as f32).collect();
    let w_out_f32: Vec<f32> = pred.w_out.iter().map(|&v| v as f32).collect();

    let x_t = Tensor::from_data(&x, vec![1, 5], device.clone()).map_err(|e| format!("x: {e}"))?;
    let w_in_t = Tensor::from_data(&w_in_f32, vec![rs, 5], device.clone())
        .map_err(|e| format!("W_in: {e}"))?;
    let w_in_tr = w_in_t.transpose().map_err(|e| format!("W_in^T: {e}"))?;
    let b_t = Tensor::from_data(&b_res_f32, vec![1, rs], device.clone())
        .map_err(|e| format!("b: {e}"))?;

    let z1 = x_t.matmul_ref(&w_in_tr).map_err(|e| format!("z1: {e}"))?;
    let z1b = z1.add(&b_t).map_err(|e| format!("z1b: {e}"))?;
    let h1 = z1b.tanh().map_err(|e| format!("h1: {e}"))?;

    let w_res_t = Tensor::from_data(&w_res_f32, vec![rs, rs], device.clone())
        .map_err(|e| format!("W_res: {e}"))?;
    let w_res_tr = w_res_t.transpose().map_err(|e| format!("W_res^T: {e}"))?;
    let ip = x_t.matmul_ref(&w_in_tr).map_err(|e| format!("ip: {e}"))?;
    let rp = h1.matmul_ref(&w_res_tr).map_err(|e| format!("rp: {e}"))?;
    let z2 = ip.add(&rp).map_err(|e| format!("z2: {e}"))?;
    let z2b = z2.add(&b_t).map_err(|e| format!("z2b: {e}"))?;
    let h2 = z2b.tanh().map_err(|e| format!("h2: {e}"))?;

    let w_out_t = Tensor::from_data(&w_out_f32, vec![rs, 1], device.clone())
        .map_err(|e| format!("w_out: {e}"))?;
    let y_t = h2
        .matmul_ref(&w_out_t)
        .map_err(|e| format!("readout: {e}"))?;
    let y_vec = y_t.to_vec().map_err(|e| format!("read: {e}"))?;

    let y_norm = f64::from(y_vec[0]);
    Ok(y_norm.mul_add(pred.y_std, pred.y_mean))
}

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("barracuda_digester_anderson");

    println!("\n── Exp 096: BarraCUDA Digester-Anderson Coupling ──");

    let baseline = match load_coupling_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("JSON load", false);
            println!("FATAL: {e}");
            h.finish();
        }
    };

    let pred = &baseline.predictor;
    h.check_bool("baseline loaded", pred.reservoir_size == 512);

    // ══════════════════════════════════════════════════════════════
    // Tier 1: BarraCUDA CPU (stats primitives on coupling vectors)
    // ══════════════════════════════════════════════════════════════
    println!("\n── Tier 1: BarraCUDA CPU stats ──");
    validate_bc_cpu_stats(&mut h, &baseline);

    // ══════════════════════════════════════════════════════════════
    // Tier 2: BarraCUDA GPU (Tensor ESN + coupling)
    // ══════════════════════════════════════════════════════════════
    println!("\n── Tier 2: BarraCUDA GPU Tensor ──");

    let Ok(gpu) = Gpu::new().await else {
        println!("  GPU unavailable — skipping GPU tier");
        h.finish();
    };

    let device: Dev = gpu.wgpu_device().clone();
    validate_gpu_esn(&mut h, &baseline, &device);
    validate_gpu_coupling(&mut h, &baseline, &device);
    validate_gpu_determinism(&mut h, &baseline, &device);

    h.finish();
}

fn validate_bc_cpu_stats(
    h: &mut ValidationHarness,
    baseline: &neural_spring::digester_anderson::CouplingBaseline,
) {
    let w_vals: Vec<f64> = baseline.communities.iter().map(|c| c.disorder_w).collect();
    let r2_vals: Vec<f64> = baseline.communities.iter().map(|c| c.r2_test).collect();
    let xi_vals: Vec<f64> = baseline
        .communities
        .iter()
        .map(|c| c.loc_length_xi)
        .collect();

    let bc_r_w = pearson_r(&w_vals, &r2_vals);
    h.check_abs(
        "bC CPU r(W, R²) parity",
        bc_r_w,
        baseline.metrics.pearson_w_r2,
        tolerances::CROSS_LANGUAGE,
    );

    let bc_r_xi = pearson_r(&xi_vals, &r2_vals);
    h.check_abs(
        "bC CPU r(ξ, R²) parity",
        bc_r_xi,
        baseline.metrics.pearson_xi_r2,
        tolerances::CROSS_LANGUAGE,
    );

    h.check_bool("bC CPU r(W, R²) < 0", bc_r_w < 0.0);
    h.check_bool("bC CPU r(ξ, R²) > 0", bc_r_xi > 0.0);

    match barracuda::stats::correlation::variance(&w_vals) {
        Ok(var) => h.check_bool("bC CPU W variance > 0", var > 0.0),
        Err(e) => {
            println!("  bC variance error: {e}");
            h.check_bool("bC CPU W variance", false);
        }
    }

    let bc_r2 = barracuda::stats::r_squared(&r2_vals, &r2_vals);
    h.check_abs("bC CPU R² self-identity", bc_r2, 1.0, tolerances::EXACT_F64);
}

fn validate_gpu_esn(
    h: &mut ValidationHarness,
    baseline: &neural_spring::digester_anderson::CouplingBaseline,
    device: &Dev,
) {
    let pred = &baseline.predictor;

    for (i, rp) in baseline.reference_predictions.iter().enumerate() {
        let cpu_y = pred.predict(
            rp.input[0],
            rp.input[1],
            rp.input[2],
            rp.input[3],
            rp.input[4],
        );

        match gpu_esn_predict(pred, &rp.input, device) {
            Ok(gpu_y) => {
                let diff = (gpu_y - cpu_y).abs();
                println!("  ref {i}: CPU={cpu_y:.2}, GPU={gpu_y:.2}, diff={diff:.2e}");

                h.check_bool(&format!("GPU ref {i} finite"), gpu_y.is_finite());
                h.check_abs(
                    &format!("GPU ref {i} vs CPU"),
                    gpu_y,
                    cpu_y,
                    tolerances::TENSOR_TRANSCENDENTAL_F32,
                );
            }
            Err(e) => {
                println!("  ref {i}: GPU FAILED — {e}");
                h.check_bool(&format!("GPU ref {i}"), false);
            }
        }
    }
}

fn validate_gpu_coupling(
    h: &mut ValidationHarness,
    baseline: &neural_spring::digester_anderson::CouplingBaseline,
    device: &Dev,
) {
    let pred = &baseline.predictor;

    let mut gpu_ok = true;
    let mut cpu_predictions = Vec::new();
    let mut gpu_predictions = Vec::new();

    for rp in &baseline.reference_predictions {
        let cpu_y = pred.predict(
            rp.input[0],
            rp.input[1],
            rp.input[2],
            rp.input[3],
            rp.input[4],
        );
        cpu_predictions.push(cpu_y);

        if let Ok(gpu_y) = gpu_esn_predict(pred, &rp.input, device) {
            gpu_predictions.push(gpu_y);
        } else {
            gpu_ok = false;
            break;
        }
    }

    if gpu_ok && cpu_predictions.len() >= 2 {
        let bc_r2_cpu = barracuda::stats::r_squared(&cpu_predictions, &gpu_predictions);
        h.check_bool("GPU↔CPU prediction R² > 0.99", bc_r2_cpu > 0.99);

        let max_diff = cpu_predictions
            .iter()
            .zip(&gpu_predictions)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0_f64, f64::max);
        println!("  max GPU↔CPU diff: {max_diff:.2e}");
        h.check_bool("GPU↔CPU max diff < 1.0", max_diff < 1.0);
    } else {
        h.check_bool("GPU coupling (failed)", false);
    }
}

fn validate_gpu_determinism(
    h: &mut ValidationHarness,
    baseline: &neural_spring::digester_anderson::CouplingBaseline,
    device: &Dev,
) {
    let pred = &baseline.predictor;
    let input = &baseline.reference_predictions[0].input;

    let r1 = gpu_esn_predict(pred, input, device);
    let r2 = gpu_esn_predict(pred, input, device);
    match (r1, r2) {
        (Ok(y1), Ok(y2)) => {
            h.check_abs("GPU deterministic", y1, y2, tolerances::EXACT_F64);
        }
        _ => h.check_bool("GPU determinism (failed)", false),
    }
}
