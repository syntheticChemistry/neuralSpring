// SPDX-License-Identifier: AGPL-3.0-or-later

//! nW-05: `BarraCUDA` GPU validator for WDM ESN regime classifier.
//!
//! Runs ESN 2-step recurrence + readout through `BarraCUDA` Tensor ops
//! on GPU, comparing with Rust CPU reference.
//!
//! ESN architecture:
//! ```text
//! Step 1: h = tanh(W_in · x + b)
//! Step 2: h = tanh(W_in · x + W_res · h + b)
//! Readout: scores = h · W_out + b_out
//! Label:   argmax(scores)
//! ```
//!
//! Evolution chain:
//! ```text
//! Jaeger ESN → Python (scikit-learn ridge) → Rust (CPU) → BarraCUDA (GPU)
//! ```

#![expect(clippy::cast_possible_truncation, reason = "validation binary")]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::wdm_esn::{self, EsnClassifier};
use std::sync::Arc;

const BASELINE_JSON: &str = include_str!("../../control/wdm/esn_regime_baseline.json");

type Dev = Arc<WgpuDevice>;

#[tokio::main]
async fn main() {
    let Ok(gpu) = Gpu::new().await else {
        neural_spring::validation::exit_no_gpu();
    };
    eprintln!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let device: Dev = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_wdm_esn");

    let esn = match wdm_esn::load_esn_from_json(BASELINE_JSON) {
        Ok(e) => e,
        Err(e) => {
            h.check_bool("JSON load", false);
            eprintln!("FATAL: failed to load ESN: {e}");
            h.finish();
        }
    };

    validate_gpu_esn(&mut h, &esn, &device);

    h.finish();
}

fn gpu_esn_classify(
    esn: &EsnClassifier,
    log_rho: f64,
    log_t: f64,
    device: &Dev,
) -> Result<(usize, Vec<f32>), String> {
    let rs = esn.reservoir_size;
    let nc = esn.n_classes;

    let x0 = ((log_rho - esn.norm.x_mean[0]) / esn.norm.x_std[0]) as f32;
    let x1 = ((log_t - esn.norm.x_mean[1]) / esn.norm.x_std[1]) as f32;

    let x_tensor = Tensor::from_data(&[x0, x1], vec![1, 2], device.clone())
        .map_err(|e| format!("x tensor: {e}"))?;

    let w_in_f32: Vec<f32> = esn.w_in.iter().map(|&v| v as f32).collect();
    let b_f32: Vec<f32> = esn.b_res.iter().map(|&v| v as f32).collect();

    let w_in_tensor = Tensor::from_data(&w_in_f32, vec![rs, 2], device.clone())
        .map_err(|e| format!("W_in: {e}"))?;
    let w_in_t = w_in_tensor
        .transpose()
        .map_err(|e| format!("W_in^T: {e}"))?;

    let b_tensor = Tensor::from_data(&b_f32, vec![1, rs], device.clone())
        .map_err(|e| format!("b_res: {e}"))?;

    // Step 1: h = tanh(x @ W_in^T + b)
    let z1 = x_tensor
        .matmul_ref(&w_in_t)
        .map_err(|e| format!("step1 matmul: {e}"))?;
    let z1b = z1.add(&b_tensor).map_err(|e| format!("step1 add: {e}"))?;
    let h1 = z1b.tanh().map_err(|e| format!("step1 tanh: {e}"))?;

    // Step 2: h = tanh(x @ W_in^T + h1 @ W_res^T + b)
    let w_res_f32: Vec<f32> = esn.w_res.iter().map(|&v| v as f32).collect();
    let w_res_tensor = Tensor::from_data(&w_res_f32, vec![rs, rs], device.clone())
        .map_err(|e| format!("W_res: {e}"))?;
    let w_res_t = w_res_tensor
        .transpose()
        .map_err(|e| format!("W_res^T: {e}"))?;

    let input_proj = x_tensor
        .matmul(&w_in_t)
        .map_err(|e| format!("step2 input: {e}"))?;
    let res_proj = h1.matmul(&w_res_t).map_err(|e| format!("step2 res: {e}"))?;
    let z2 = input_proj
        .add(&res_proj)
        .map_err(|e| format!("step2 add1: {e}"))?;
    let z2b = z2.add(&b_tensor).map_err(|e| format!("step2 add2: {e}"))?;
    let h2 = z2b.tanh().map_err(|e| format!("step2 tanh: {e}"))?;

    // Readout: scores = h2 @ W_out + b_out
    let w_out_f32: Vec<f32> = esn.w_out.iter().map(|&v| v as f32).collect();
    let b_out_f32: Vec<f32> = esn.b_out.iter().map(|&v| v as f32).collect();

    let w_out_tensor = Tensor::from_data(&w_out_f32, vec![rs, nc], device.clone())
        .map_err(|e| format!("W_out: {e}"))?;
    let b_out_tensor = Tensor::from_data(&b_out_f32, vec![1, nc], device.clone())
        .map_err(|e| format!("b_out: {e}"))?;

    let scores_raw = h2
        .matmul(&w_out_tensor)
        .map_err(|e| format!("readout matmul: {e}"))?;
    let scores_biased = scores_raw
        .add(&b_out_tensor)
        .map_err(|e| format!("readout add: {e}"))?;

    let scores_vec = scores_biased
        .to_vec()
        .map_err(|e| format!("readback: {e}"))?;

    let label = scores_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    Ok((label, scores_vec))
}

fn validate_gpu_esn(h: &mut ValidationHarness, esn: &EsnClassifier, device: &Dev) {
    let test_cases: &[(f64, f64, &str)] = &[
        (-1.0, 8.0, "hot-sparse (Classical)"),
        (0.5, 5.5, "WDM regime"),
        (2.0, 4.0, "cold-dense (Degenerate)"),
    ];

    for &(log_rho, log_t, desc) in test_cases {
        let (cpu_label, cpu_scores) = esn.classify(log_rho, log_t);

        let (gpu_label, gpu_scores) = match gpu_esn_classify(esn, log_rho, log_t, device) {
            Ok(r) => r,
            Err(e) => {
                h.check_bool(&format!("GPU classify {desc}"), false);
                eprintln!("  GPU classify failed for {desc}: {e}");
                continue;
            }
        };

        h.check_bool(
            &format!("GPU scores finite ({desc})"),
            gpu_scores.iter().all(|s| s.is_finite()),
        );

        h.check_bool(
            &format!("GPU label matches CPU ({desc})"),
            gpu_label == cpu_label,
        );

        let max_score_diff: f64 = gpu_scores
            .iter()
            .zip(cpu_scores.iter())
            .map(|(g, c)| (f64::from(*g) - c).abs())
            .fold(0.0_f64, f64::max);

        h.check_bool(
            &format!("GPU scores within f32 tol ({desc}), diff={max_score_diff:.2e}"),
            max_score_diff < tolerances::TENSOR_TRANSCENDENTAL_F32,
        );
    }

    // Determinism
    let Ok((_, s1)) = gpu_esn_classify(esn, 0.5, 5.5, device) else {
        h.check_bool("GPU determinism (run 1 failed)", false);
        return;
    };
    let Ok((_, s2)) = gpu_esn_classify(esn, 0.5, 5.5, device) else {
        h.check_bool("GPU determinism (run 2 failed)", false);
        return;
    };
    let det = s1
        .iter()
        .zip(s2.iter())
        .all(|(a, b)| (f64::from(*a) - f64::from(*b)).abs() < f64::EPSILON);
    h.check_bool("GPU determinism", det);
}
