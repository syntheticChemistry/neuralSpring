// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validates `barracuda::ops::logsumexp::LogSumExp` against analytical values.
//!
//! This proves that `BarraCUDA`'s native `LogSumExp` op is correct and could
//! replace the manual logsumexp in `metalForge/shaders/hmm_forward_log.wgsl`.
//! Once `StatefulPipeline` supports chained logsumexp dispatches, the local
//! HMM shader can be retired.
//!
//! ## Provenance
//!
//! All expected values are analytical (logsumexp definition).
//! `logsumexp(x) = log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))`
//!
//! ## Backend selection
//!
//! Set `NEURALSPRING_BACKEND=cpu|gpu|auto`.

#![allow(clippy::cast_precision_loss)]

use barracuda::device::WgpuDevice;
use barracuda::ops::logsumexp::LogSumExp;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::require;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

fn cpu_logsumexp(values: &[f32]) -> f32 {
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        return max_val;
    }
    max_val
        + values
            .iter()
            .map(|&v| (v - max_val).exp())
            .sum::<f32>()
            .ln()
}

#[tokio::main]
async fn main() {
    let Ok(gpu) = Gpu::new().await else {
        eprintln!("  0/0 checks — no adapter");
        std::process::exit(0);
    };
    eprintln!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let device = gpu.wgpu_device().clone();

    let mut h = ValidationHarness::new("barracuda_logsumexp");

    validate_basic(&mut h, &device);
    validate_hmm_like(&mut h, &device);
    validate_negative_values(&mut h, &device);
    validate_single_element(&mut h, &device);
    validate_equal_values(&mut h, &device);

    h.finish();
}

fn validate_basic(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let expected = cpu_logsumexp(&data);

    let tensor = require!(
        h,
        Tensor::from_data(&data, vec![data.len()], device.clone()),
        "tensor creation"
    );

    match LogSumExp::new(tensor).execute() {
        Ok(result) => {
            let result_data = require!(h, result.to_vec(), "readback");
            let gpu_val = result_data[0];
            h.check_abs(
                &format!("basic [1,2,3,4]: {gpu_val:.6} vs {expected:.6}"),
                f64::from(gpu_val),
                f64::from(expected),
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("basic: execute failed — {e}"), false);
        }
    }
}

fn validate_hmm_like(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let data: Vec<f32> = vec![-5.0, -3.0, -8.0, -2.5, -6.0];
    let expected = cpu_logsumexp(&data);

    let tensor = require!(
        h,
        Tensor::from_data(&data, vec![data.len()], device.clone()),
        "tensor creation"
    );

    match LogSumExp::new(tensor).execute() {
        Ok(result) => {
            let result_data = require!(h, result.to_vec(), "readback");
            let gpu_val = result_data[0];
            h.check_abs(
                &format!("HMM-like [-5,-3,-8,-2.5,-6]: {gpu_val:.6} vs {expected:.6}"),
                f64::from(gpu_val),
                f64::from(expected),
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("HMM-like: execute failed — {e}"), false);
        }
    }
}

fn validate_negative_values(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let data: Vec<f32> = vec![-100.0, -99.0, -101.0];
    let expected = cpu_logsumexp(&data);

    let tensor = require!(
        h,
        Tensor::from_data(&data, vec![data.len()], device.clone()),
        "tensor creation"
    );

    match LogSumExp::new(tensor).execute() {
        Ok(result) => {
            let result_data = require!(h, result.to_vec(), "readback");
            let gpu_val = result_data[0];
            h.check_abs(
                &format!("large negative: {gpu_val:.4} vs {expected:.4}"),
                f64::from(gpu_val),
                f64::from(expected),
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("large negative: failed — {e}"), false);
        }
    }
}

fn validate_single_element(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let data = vec![42.0_f32];
    let expected = 42.0_f32;

    let tensor = require!(
        h,
        Tensor::from_data(&data, vec![1], device.clone()),
        "tensor creation"
    );

    match LogSumExp::new(tensor).execute() {
        Ok(result) => {
            let result_data = require!(h, result.to_vec(), "readback");
            let gpu_val = result_data[0];
            h.check_abs(
                &format!("single [42]: {gpu_val:.6} vs {expected:.6}"),
                f64::from(gpu_val),
                f64::from(expected),
                tolerances::TENSOR_EXACT_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("single: failed — {e}"), false);
        }
    }
}

fn validate_equal_values(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 8;
    let val = 1.0_f32;
    let data = vec![val; n];
    #[allow(clippy::cast_precision_loss)]
    let expected = val + (n as f32).ln();

    let tensor = require!(
        h,
        Tensor::from_data(&data, vec![n], device.clone()),
        "tensor creation"
    );

    match LogSumExp::new(tensor).execute() {
        Ok(result) => {
            let result_data = require!(h, result.to_vec(), "readback");
            let gpu_val = result_data[0];
            h.check_abs(
                &format!("equal [1.0; 8]: {gpu_val:.6} vs {expected:.6} (1 + ln8)"),
                f64::from(gpu_val),
                f64::from(expected),
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("equal: failed — {e}"), false);
        }
    }
}
