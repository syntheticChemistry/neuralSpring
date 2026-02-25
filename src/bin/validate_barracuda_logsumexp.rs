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
//! ## Known upstream issue
//!
//! barracuda `LogSumExp` currently binds a 4-byte output buffer where the
//! shader expects 8 bytes.  wgpu converts this into a validation-error panic
//! that corrupts GPU state for the process lifetime.  We probe once and skip
//! gracefully until upstream fixes the binding layout.
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

/// Probe `LogSumExp` with a trivial tensor; returns `true` if the op works.
///
/// A wgpu validation-error panic corrupts GPU state for the rest of the
/// process, so we probe once and skip all checks if the op is broken.
fn logsumexp_probe(device: &Arc<WgpuDevice>) -> bool {
    let data = vec![1.0_f32];
    let Ok(tensor) = Tensor::from_data(&data, vec![1], device.clone()) else {
        return false;
    };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        LogSumExp::new(tensor).execute()
    }))
    .is_ok_and(|r| r.is_ok())
}

fn validate_case(
    h: &mut ValidationHarness,
    device: &Arc<WgpuDevice>,
    label: &str,
    data: &[f32],
    expected: f32,
    tolerance: f64,
) {
    let tensor = require!(
        h,
        Tensor::from_data(data, vec![data.len()], device.clone()),
        &format!("{label}: tensor creation")
    );
    match LogSumExp::new(tensor).execute() {
        Ok(result) => {
            let result_data = require!(h, result.to_vec(), &format!("{label}: readback"));
            let gpu_val = result_data[0];
            h.check_abs(
                &format!("{label}: {gpu_val:.6} vs {expected:.6}"),
                f64::from(gpu_val),
                f64::from(expected),
                tolerance,
            );
        }
        Err(e) => {
            h.check_bool(&format!("{label}: execute failed — {e}"), false);
        }
    }
}

#[tokio::main]
async fn main() {
    let Ok(gpu) = Gpu::new().await else {
        neural_spring::validation::exit_no_gpu();
    };
    eprintln!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let device = gpu.wgpu_device().clone();

    let mut h = ValidationHarness::new("barracuda_logsumexp");

    if !logsumexp_probe(&device) {
        eprintln!(
            "  [skip] LogSumExp op panics (upstream buffer-size mismatch) — \
             skipping all checks until barracuda fixes binding layout"
        );
        h.check_bool(
            "LogSumExp probe: upstream buffer-size mismatch (barracuda issue, not neuralSpring)",
            false,
        );
        h.finish();
    }

    let basic = [1.0_f32, 2.0, 3.0, 4.0];
    validate_case(
        &mut h,
        &device,
        "basic [1,2,3,4]",
        &basic,
        cpu_logsumexp(&basic),
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let hmm = [-5.0_f32, -3.0, -8.0, -2.5, -6.0];
    validate_case(
        &mut h,
        &device,
        "HMM-like",
        &hmm,
        cpu_logsumexp(&hmm),
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let neg = [-100.0_f32, -99.0, -101.0];
    validate_case(
        &mut h,
        &device,
        "large negative",
        &neg,
        cpu_logsumexp(&neg),
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    validate_case(
        &mut h,
        &device,
        "single [42]",
        &[42.0_f32],
        42.0,
        tolerances::TENSOR_EXACT_F32,
    );

    let n = 8;
    let eq_data = vec![1.0_f32; n];
    #[allow(clippy::cast_precision_loss)]
    let eq_expected = 1.0 + (n as f32).ln();
    validate_case(
        &mut h,
        &device,
        "equal [1.0; 8]",
        &eq_data,
        eq_expected,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    h.finish();
}
