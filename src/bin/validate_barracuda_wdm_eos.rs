// SPDX-License-Identifier: AGPL-3.0-or-later

//! nW-02: `BarraCUDA` GPU validator for WDM EOS surrogates.
//!
//! Runs MLP inference through `BarraCUDA` Tensor ops on GPU,
//! comparing with Rust CPU reference path.
//!
//! Evolution chain:
//! ```text
//! FPEOS tables → Python MLP → Rust MLP (CPU) → BarraCUDA (GPU) → Pure GPU
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::many_single_char_names
)]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::wdm_surrogate::{self, EosSurrogate};
use std::sync::Arc;

const BASELINE_JSON: &str = include_str!("../../control/wdm/eos_surrogate_baseline.json");

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
    let mut h = ValidationHarness::new("barracuda_wdm_eos");

    for element in ["H", "He", "C"] {
        let surrogate = match wdm_surrogate::load_surrogate_from_json(BASELINE_JSON, element) {
            Ok(s) => s,
            Err(e) => {
                h.check_bool(&format!("{element}: JSON load"), false);
                eprintln!("FATAL: failed to load {element}: {e}");
                h.finish();
            }
        };
        validate_gpu_mlp(&mut h, &surrogate, &device);
    }

    h.finish();
}

fn gpu_mlp_forward(
    surr: &EosSurrogate,
    x0: f32,
    x1: f32,
    device: &Dev,
) -> Result<Vec<f32>, String> {
    let mut current = Tensor::from_data(&[x0, x1], vec![1, 2], device.clone())
        .map_err(|e| format!("input tensor: {e}"))?;

    for (i, layer) in surr.layers.iter().enumerate() {
        let w_f32: Vec<f32> = layer.weights.iter().map(|&v| v as f32).collect();
        let b_f32: Vec<f32> = layer.bias.iter().map(|&v| v as f32).collect();

        let w_tensor = Tensor::from_data(
            &w_f32,
            vec![layer.out_features, layer.in_features],
            device.clone(),
        )
        .map_err(|e| format!("layer {i} W: {e}"))?;

        let w_t = w_tensor
            .transpose()
            .map_err(|e| format!("layer {i} W^T: {e}"))?;

        let z = current
            .matmul(&w_t)
            .map_err(|e| format!("layer {i} matmul: {e}"))?;

        let b_tensor = Tensor::from_data(&b_f32, vec![1, layer.out_features], device.clone())
            .map_err(|e| format!("layer {i} b: {e}"))?;

        let z_biased = z
            .add(&b_tensor)
            .map_err(|e| format!("layer {i} add: {e}"))?;

        current = if i < surr.layers.len() - 1 {
            z_biased
                .relu()
                .map_err(|e| format!("layer {i} relu: {e}"))?
        } else {
            z_biased
        };
    }

    current.to_vec().map_err(|e| format!("readback: {e}"))
}

fn validate_gpu_mlp(h: &mut ValidationHarness, surr: &EosSurrogate, device: &Dev) {
    let elem = &surr.element;
    let guard = neural_spring::tolerances::LOG_ZERO_GUARD;

    let test_rho = 1.0_f64;
    let test_t = 100_000.0_f64;
    let (cpu_p, cpu_e) = surr.predict(test_rho, test_t);

    let log_rho = (test_rho + guard).log10();
    let log_t = (test_t + guard).log10();
    let x0 = ((log_rho - surr.norm.x_mean[0]) / surr.norm.x_std[0]) as f32;
    let x1 = ((log_t - surr.norm.x_mean[1]) / surr.norm.x_std[1]) as f32;

    let gpu_output = match gpu_mlp_forward(surr, x0, x1, device) {
        Ok(out) => out,
        Err(e) => {
            h.check_bool(&format!("{elem}: GPU forward pass"), false);
            eprintln!("  GPU forward failed for {elem}: {e}");
            return;
        }
    };

    let gpu_log_p_norm = f64::from(gpu_output[0]);
    let gpu_log_e_norm = f64::from(gpu_output[1]);

    let gpu_log_p = gpu_log_p_norm.mul_add(surr.norm.y_std[0], surr.norm.y_mean[0]);
    let gpu_log_e = gpu_log_e_norm.mul_add(surr.norm.y_std[1], surr.norm.y_mean[1]);

    let gpu_p = gpu_log_p.signum() * 10.0_f64.powf(gpu_log_p.abs());
    let gpu_e = gpu_log_e.signum() * 10.0_f64.powf(gpu_log_e.abs());

    h.check_bool(&format!("{elem}: GPU P is finite"), gpu_p.is_finite());
    h.check_bool(&format!("{elem}: GPU E is finite"), gpu_e.is_finite());

    let p_rel = if cpu_p.abs() > 1e-10 {
        ((gpu_p - cpu_p) / cpu_p).abs()
    } else {
        (gpu_p - cpu_p).abs()
    };
    let e_rel = if cpu_e.abs() > 1e-10 {
        ((gpu_e - cpu_e) / cpu_e).abs()
    } else {
        (gpu_e - cpu_e).abs()
    };

    h.check_bool(
        &format!("{elem}: GPU P within f32 tolerance of CPU"),
        p_rel < tolerances::ML_MLP_F32,
    );
    h.check_bool(
        &format!("{elem}: GPU E within f32 tolerance of CPU"),
        e_rel < tolerances::ML_MLP_F32,
    );

    let out2 = match gpu_mlp_forward(surr, x0, x1, device) {
        Ok(out) => out,
        Err(e) => {
            h.check_bool(&format!("{elem}: GPU determinism (re-run)"), false);
            eprintln!("  GPU determinism re-run failed for {elem}: {e}");
            return;
        }
    };
    h.check_bool(
        &format!("{elem}: GPU determinism"),
        (f64::from(gpu_output[0]) - f64::from(out2[0])).abs() < f64::EPSILON
            && (f64::from(gpu_output[1]) - f64::from(out2[1])).abs() < f64::EPSILON,
    );
}
