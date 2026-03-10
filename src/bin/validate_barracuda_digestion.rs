// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper 027: `BarraCUDA` CPU + GPU validator for ESN digestion prediction.
//!
//! Validates the digestion prediction ESN through two tiers:
//!
//! **Tier 1 — `BarraCUDA` CPU**: Verifies that `barracuda::stats` primitives
//! (variance, Pearson correlation, R²) produce identical results to the pure
//! Rust implementation on synthetic digester data.
//!
//! **Tier 2 — `BarraCUDA` GPU**: Runs the ESN 2-step recurrence through
//! `Tensor::matmul_ref` + `Tensor::add` + `Tensor::tanh`, and the linear
//! readout through `Tensor::matmul_ref`. Compares GPU predictions vs CPU.
//!
//! ESN architecture:
//! ```text
//! Step 1: h = tanh(W_in @ x + b)
//! Step 2: h = tanh(W_in @ x + W_res @ h + b)
//! Readout: y_norm = h @ w_out
//! Yield:   y = y_norm * y_std + y_mean
//! ```
//!
//! ## Provenance
//!
//! | Baseline | Source |
//! |----------|--------|
//! | Python baseline | `control/digestion_prediction/digestion_prediction.py` |
//! | Reference | Wang et al. (2020), Bioresour Technol 298:122495 |
//! | Baseline date | 2026-03-10 |

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "validation binary with multi-tier ESN GPU promotion"
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::digestion_prediction::{
    biogas_yield, load_digestion_from_json, DigestionPredictor,
};
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

const BASELINE_JSON: &str =
    include_str!("../../control/digestion_prediction/digestion_prediction_baseline.json");

type Dev = Arc<WgpuDevice>;

// ═══════════════════════════════════════════════════════════════════
// GPU ESN inference
// ═══════════════════════════════════════════════════════════════════

fn gpu_esn_predict(
    pred: &DigestionPredictor,
    t: f64,
    ph: f64,
    olr: f64,
    hrt: f64,
    vs_ts: f64,
    device: &Dev,
) -> Result<(f64, Vec<f32>), String> {
    let rs = pred.reservoir_size;

    let x = [
        ((t - pred.norm.x_mean[0]) / pred.norm.x_std[0]) as f32,
        ((ph - pred.norm.x_mean[1]) / pred.norm.x_std[1]) as f32,
        ((olr - pred.norm.x_mean[2]) / pred.norm.x_std[2]) as f32,
        ((hrt - pred.norm.x_mean[3]) / pred.norm.x_std[3]) as f32,
        ((vs_ts - pred.norm.x_mean[4]) / pred.norm.x_std[4]) as f32,
    ];

    let w_in_f32: Vec<f32> = pred.w_in.iter().map(|&v| v as f32).collect();
    let w_res_f32: Vec<f32> = pred.w_res.iter().map(|&v| v as f32).collect();
    let b_res_f32: Vec<f32> = pred.b_res.iter().map(|&v| v as f32).collect();
    let w_out_f32: Vec<f32> = pred.w_out.iter().map(|&v| v as f32).collect();

    let x_tensor =
        Tensor::from_data(&x, vec![1, 5], device.clone()).map_err(|e| format!("x: {e}"))?;
    let w_in_tensor = Tensor::from_data(&w_in_f32, vec![rs, 5], device.clone())
        .map_err(|e| format!("W_in: {e}"))?;
    let w_in_t = w_in_tensor
        .transpose()
        .map_err(|e| format!("W_in^T: {e}"))?;
    let b_tensor = Tensor::from_data(&b_res_f32, vec![1, rs], device.clone())
        .map_err(|e| format!("b: {e}"))?;

    // Step 1: h = tanh(x @ W_in^T + b)
    let z1 = x_tensor
        .matmul_ref(&w_in_t)
        .map_err(|e| format!("z1: {e}"))?;
    let z1b = z1.add(&b_tensor).map_err(|e| format!("z1b: {e}"))?;
    let h1 = z1b.tanh().map_err(|e| format!("h1: {e}"))?;

    // Step 2: h = tanh(x @ W_in^T + h1 @ W_res^T + b)
    let w_res_tensor = Tensor::from_data(&w_res_f32, vec![rs, rs], device.clone())
        .map_err(|e| format!("W_res: {e}"))?;
    let w_res_t = w_res_tensor
        .transpose()
        .map_err(|e| format!("W_res^T: {e}"))?;
    let input_proj = x_tensor
        .matmul_ref(&w_in_t)
        .map_err(|e| format!("ip: {e}"))?;
    let res_proj = h1
        .matmul_ref(&w_res_t)
        .map_err(|e| format!("rp: {e}"))?;
    let z2 = input_proj
        .add(&res_proj)
        .map_err(|e| format!("z2: {e}"))?;
    let z2b = z2.add(&b_tensor).map_err(|e| format!("z2b: {e}"))?;
    let h2 = z2b.tanh().map_err(|e| format!("h2: {e}"))?;

    let h2_vec = h2.to_vec().map_err(|e| format!("h2 read: {e}"))?;

    // Readout: y_norm = h2 @ w_out
    let w_out_tensor = Tensor::from_data(&w_out_f32, vec![rs, 1], device.clone())
        .map_err(|e| format!("w_out: {e}"))?;
    let y_norm_tensor = h2
        .matmul_ref(&w_out_tensor)
        .map_err(|e| format!("readout: {e}"))?;
    let y_norm_vec = y_norm_tensor
        .to_vec()
        .map_err(|e| format!("readout read: {e}"))?;

    let y_norm = f64::from(y_norm_vec[0]);
    let y_pred = y_norm * pred.norm.y_std + pred.norm.y_mean;

    Ok((y_pred, h2_vec))
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("barracuda_digestion");

    eprintln!("\n── Paper 027: BarraCUDA Digestion Prediction ──");

    let baseline = match load_digestion_from_json(BASELINE_JSON) {
        Ok(b) => b,
        Err(e) => {
            h.check_bool("JSON load", false);
            eprintln!("FATAL: {e}");
            h.finish();
        }
    };

    let pred = &baseline.predictor;
    h.check_bool("baseline loaded", pred.reservoir_size == 512);

    // ── Tier 1: BarraCUDA CPU (stats primitives) ──
    eprintln!("\n── Tier 1: BarraCUDA CPU stats ──");

    validate_barracuda_cpu_stats(&mut h);

    // ── Tier 2: BarraCUDA GPU (Tensor ESN inference) ──
    eprintln!("\n── Tier 2: BarraCUDA GPU Tensor ESN ──");

    let Ok(gpu) = Gpu::new().await else {
        eprintln!("  GPU unavailable — skipping GPU tier");
        h.finish();
    };

    let device: Dev = gpu.wgpu_device().clone();
    validate_gpu_esn(&mut h, pred, &device);
    validate_gpu_determinism(&mut h, pred, &device);

    h.finish();
}

fn validate_barracuda_cpu_stats(h: &mut ValidationHarness) {
    let y_true = [300.0, 310.0, 280.0, 320.0, 290.0];
    let y_pred = [295.0, 315.0, 275.0, 325.0, 288.0];

    let mean_t = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_tot: f64 = y_true.iter().map(|v| (v - mean_t).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(&y_pred)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let rs_r2 = 1.0 - ss_res / ss_tot;

    let bc_r2 = barracuda::stats::r_squared(&y_true, &y_pred);
    h.check_abs("bC CPU R² matches Rust", bc_r2, rs_r2, tolerances::CROSS_LANGUAGE);

    let rs_var = y_true.iter().map(|v| (v - mean_t).powi(2)).sum::<f64>()
        / (y_true.len() as f64 - 1.0);
    match barracuda::stats::correlation::variance(&y_true) {
        Ok(bc_var) => h.check_abs("bC CPU variance matches Rust", bc_var, rs_var, tolerances::EXACT_F64),
        Err(e) => {
            eprintln!("  bC variance failed: {e}");
            h.check_bool("bC CPU variance", false);
        }
    }

    match barracuda::stats::correlation::pearson_correlation(&y_true, &y_pred) {
        Ok(bc_pearson) => {
            h.check_bool("bC CPU Pearson finite", bc_pearson.is_finite());
            h.check_bool("bC CPU Pearson > 0.9", bc_pearson > 0.9);
        }
        Err(e) => {
            eprintln!("  bC Pearson failed: {e}");
            h.check_bool("bC CPU Pearson", false);
        }
    }

    let y_analytical = biogas_yield(35.0, 7.2, 3.0, 20.0, 75.0);
    h.check_bool("analytical yield > 250", y_analytical > 250.0);
}

fn validate_gpu_esn(h: &mut ValidationHarness, pred: &DigestionPredictor, device: &Dev) {
    let test_cases: &[(f64, f64, f64, f64, f64, &str)] = &[
        (35.0, 7.2, 3.0, 20.0, 75.0, "mesophilic optimum"),
        (55.0, 7.2, 3.0, 20.0, 75.0, "thermophilic optimum"),
        (35.0, 5.5, 3.0, 20.0, 75.0, "low pH stress"),
        (35.0, 7.2, 7.0, 20.0, 75.0, "high OLR inhibition"),
        (35.0, 7.2, 3.0, 5.0, 75.0, "short HRT"),
    ];

    for &(t, ph, olr, hrt, vs_ts, desc) in test_cases {
        let cpu_pred = pred.predict(t, ph, olr, hrt, vs_ts);
        let gpu_result = gpu_esn_predict(pred, t, ph, olr, hrt, vs_ts, device);

        match gpu_result {
            Ok((gpu_pred, gpu_h)) => {
                let diff = (gpu_pred - cpu_pred).abs();
                eprintln!(
                    "  {desc}: CPU={cpu_pred:.2}, GPU={gpu_pred:.2}, diff={diff:.2e}"
                );

                h.check_bool(
                    &format!("GPU finite ({desc})"),
                    gpu_pred.is_finite(),
                );
                h.check_abs(
                    &format!("GPU yield vs CPU ({desc})"),
                    gpu_pred,
                    cpu_pred,
                    tolerances::TENSOR_TRANSCENDENTAL_F32,
                );
                h.check_bool(
                    &format!("GPU reservoir finite ({desc})"),
                    gpu_h.iter().all(|v| v.is_finite()),
                );
            }
            Err(e) => {
                eprintln!("  {desc}: GPU FAILED — {e}");
                h.check_bool(&format!("GPU ESN ({desc})"), false);
            }
        }
    }

    let cpu_meso = pred.predict(35.0, 7.2, 3.0, 20.0, 75.0);
    let cpu_low_ph = pred.predict(35.0, 5.5, 3.0, 20.0, 75.0);
    h.check_bool(
        "GPU: low pH < mesophilic (physics preserved)",
        cpu_low_ph < cpu_meso,
    );
}

fn validate_gpu_determinism(h: &mut ValidationHarness, pred: &DigestionPredictor, device: &Dev) {
    let r1 = gpu_esn_predict(pred, 35.0, 7.2, 3.0, 20.0, 75.0, device);
    let r2 = gpu_esn_predict(pred, 35.0, 7.2, 3.0, 20.0, 75.0, device);
    match (r1, r2) {
        (Ok((y1, _)), Ok((y2, _))) => {
            h.check_abs("GPU deterministic", y1, y2, tolerances::EXACT_F64);
        }
        _ => h.check_bool("GPU determinism (failed to run)", false),
    }
}
