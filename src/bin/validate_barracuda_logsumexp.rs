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
//! ## Upstream evolution (`ToadStool` S60–S65)
//!
//! `LogSumExp::execute()` evolved to f64-only (uses `compile_shader_f64`,
//! `create_buffer_f64`).  This validator feeds f64 tensors and reads back
//! with `to_f64_vec()` to match the current upstream contract.
//!
//! ## Backend selection
//!
//! Set `GPU_BACKEND=cpu|gpu|auto`.

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use barracuda::device::WgpuDevice;
use barracuda::ops::logsumexp::LogSumExp;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::require;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

fn cpu_logsumexp(values: &[f64]) -> f64 {
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_val.is_finite() {
        return max_val;
    }
    max_val
        + values
            .iter()
            .map(|&v| (v - max_val).exp())
            .sum::<f64>()
            .ln()
}

fn logsumexp_probe(device: &Arc<WgpuDevice>) -> bool {
    let data = vec![1.0_f64];
    let Ok(tensor) = Tensor::from_f64_data(&data, vec![1], device.clone()) else {
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
    data: &[f64],
    expected: f64,
    tolerance: f64,
) {
    let tensor = require!(
        h,
        Tensor::from_f64_data(data, vec![data.len()], device.clone()),
        &format!("{label}: tensor creation")
    );
    match LogSumExp::new(tensor).execute() {
        Ok(result) => {
            let result_data = require!(h, result.to_f64_vec(), &format!("{label}: readback"));
            let gpu_val = result_data[0];
            h.check_abs(
                &format!("{label}: {gpu_val:.6} vs {expected:.6}"),
                gpu_val,
                expected,
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
        eprintln!("  [skip] LogSumExp op not functional — skipping all checks");
        h.check_bool("LogSumExp probe: op not functional", false);
        h.finish();
    }

    let basic = [1.0_f64, 2.0, 3.0, 4.0];
    validate_case(
        &mut h,
        &device,
        "basic [1,2,3,4]",
        &basic,
        cpu_logsumexp(&basic),
        tolerances::GPU_F64_TRANSCENDENTAL,
    );

    let hmm = [-5.0_f64, -3.0, -8.0, -2.5, -6.0];
    validate_case(
        &mut h,
        &device,
        "HMM-like",
        &hmm,
        cpu_logsumexp(&hmm),
        tolerances::GPU_F64_TRANSCENDENTAL,
    );

    let neg = [-100.0_f64, -99.0, -101.0];
    validate_case(
        &mut h,
        &device,
        "large negative",
        &neg,
        cpu_logsumexp(&neg),
        tolerances::GPU_F64_TRANSCENDENTAL,
    );

    validate_case(
        &mut h,
        &device,
        "single [42]",
        &[42.0_f64],
        42.0,
        tolerances::GPU_F64_EXACT,
    );

    let n = 8;
    let eq_data = vec![1.0_f64; n];
    let eq_expected = 1.0 + (n as f64).ln();
    validate_case(
        &mut h,
        &device,
        "equal [1.0; 8]",
        &eq_data,
        eq_expected,
        tolerances::GPU_F64_TRANSCENDENTAL,
    );

    h.finish();
}
