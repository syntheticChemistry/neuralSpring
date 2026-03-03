// SPDX-License-Identifier: AGPL-3.0-or-later

//! nW-01: `BarraCUDA` GPU validator for WDM transport surrogates.
//!
//! Runs MLP inference through `BarraCUDA` Tensor ops on GPU,
//! comparing with Rust CPU reference path.
//!
//! Evolution chain:
//! ```text
//! Stanton-Murillo model → Python MLP → Rust MLP (CPU) → BarraCUDA (GPU)
//! ```

#![expect(clippy::cast_possible_truncation, reason = "validation binary")]

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::wdm_transport::{self, TransportSurrogate};
use std::sync::Arc;

const BASELINE_JSON: &str = include_str!("../../control/wdm/transport_surrogate_baseline.json");

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
    let mut h = ValidationHarness::new("barracuda_wdm_transport");

    let surr = match wdm_transport::load_transport_from_json(BASELINE_JSON) {
        Ok(s) => s,
        Err(e) => {
            h.check_bool("JSON load", false);
            eprintln!("FATAL: failed to load transport surrogate: {e}");
            h.finish();
        }
    };

    validate_gpu_mlp(&mut h, &surr, &device);

    h.finish();
}

fn gpu_mlp_forward(
    surr: &TransportSurrogate,
    x0: f32,
    x1: f32,
    x2: f32,
    device: &Dev,
) -> Result<Vec<f32>, String> {
    let mut current = Tensor::from_data(&[x0, x1, x2], vec![1, 3], device.clone())
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

fn validate_gpu_mlp(h: &mut ValidationHarness, surr: &TransportSurrogate, device: &Dev) {
    let test_cases: &[(f64, f64, f64)] = &[(0.5, 5.0, 6.0), (-1.0, 8.0, 13.0), (1.7, 4.0, 1.0)];

    for &(log_rho, log_t, z_star) in test_cases {
        let (cpu_d, cpu_eta, cpu_lam) = surr.predict(log_rho, log_t, z_star);

        let x0 = ((log_rho - surr.norm.x_mean[0]) / surr.norm.x_std[0]) as f32;
        let x1 = ((log_t - surr.norm.x_mean[1]) / surr.norm.x_std[1]) as f32;
        let x2 = ((z_star - surr.norm.x_mean[2]) / surr.norm.x_std[2]) as f32;

        let gpu_output = match gpu_mlp_forward(surr, x0, x1, x2, device) {
            Ok(out) => out,
            Err(e) => {
                h.check_bool(&format!("GPU forward (ρ={log_rho})"), false);
                eprintln!("  GPU forward failed: {e}");
                continue;
            }
        };

        let gpu_d_norm = f64::from(gpu_output[0]);
        let gpu_eta_norm = f64::from(gpu_output[1]);
        let gpu_lam_norm = f64::from(gpu_output[2]);

        let gpu_d_log = gpu_d_norm.mul_add(surr.norm.y_std[0], surr.norm.y_mean[0]);
        let gpu_eta_log = gpu_eta_norm.mul_add(surr.norm.y_std[1], surr.norm.y_mean[1]);
        let gpu_lam_log = gpu_lam_norm.mul_add(surr.norm.y_std[2], surr.norm.y_mean[2]);

        let gpu_d = 10.0_f64.powf(gpu_d_log);
        let gpu_eta = 10.0_f64.powf(gpu_eta_log);
        let gpu_lam = 10.0_f64.powf(gpu_lam_log);

        h.check_bool(&format!("GPU D* finite (ρ={log_rho})"), gpu_d.is_finite());
        h.check_bool(&format!("GPU η* finite (ρ={log_rho})"), gpu_eta.is_finite());
        h.check_bool(&format!("GPU λ* finite (ρ={log_rho})"), gpu_lam.is_finite());

        let d_rel = rel_err(gpu_d, cpu_d);
        let eta_rel = rel_err(gpu_eta, cpu_eta);
        let lam_rel = rel_err(gpu_lam, cpu_lam);

        h.check_bool(
            &format!("GPU D* within f32 tol (ρ={log_rho})"),
            d_rel < tolerances::ML_MLP_F32,
        );
        h.check_bool(
            &format!("GPU η* within f32 tol (ρ={log_rho})"),
            eta_rel < tolerances::ML_MLP_F32,
        );
        h.check_bool(
            &format!("GPU λ* within f32 tol (ρ={log_rho})"),
            lam_rel < tolerances::ML_MLP_F32,
        );
    }

    let x0 = ((0.5 - surr.norm.x_mean[0]) / surr.norm.x_std[0]) as f32;
    let x1 = ((5.0 - surr.norm.x_mean[1]) / surr.norm.x_std[1]) as f32;
    let x2 = ((6.0 - surr.norm.x_mean[2]) / surr.norm.x_std[2]) as f32;

    let run1 = gpu_mlp_forward(surr, x0, x1, x2, device);
    let run2 = gpu_mlp_forward(surr, x0, x1, x2, device);
    match (run1, run2) {
        (Ok(a), Ok(b)) => {
            let det = a
                .iter()
                .zip(b.iter())
                .all(|(x, y)| (f64::from(*x) - f64::from(*y)).abs() < f64::EPSILON);
            h.check_bool("GPU determinism", det);
        }
        _ => {
            h.check_bool("GPU determinism (re-run failed)", false);
        }
    }
}

fn rel_err(gpu: f64, cpu: f64) -> f64 {
    if cpu.abs() > tolerances::RELATIVE_ERROR_FLOOR {
        ((gpu - cpu) / cpu).abs()
    } else {
        (gpu - cpu).abs()
    }
}
