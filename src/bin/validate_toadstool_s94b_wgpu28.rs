// SPDX-License-Identifier: AGPL-3.0-or-later

//! `ToadStool` S94b + wgpu 28 + `BarraCUDA` v0.3.3 validation.
//!
//! Validates the S125 upstream sync: wgpu 22→28 API migration, `BarraCUDA`
//! v0.3.3 fused op absorption, and `ToadStool` S94b pin.
//!
//! ## What changed (S87 → S94b / v0.3.1 → v0.3.3)
//!
//! - `BarraCUDA` extracted to standalone primal (S89)
//! - wgpu 22 → 28: `PollType::Wait`, `immediate_size`, `entry_point: Option<&str>`,
//!   `set_bind_group` `Option<&BindGroup>`, `Instance::new(&ref)`,
//!   `enumerate_adapters` async, `DeviceDescriptor` new fields
//! - `VarianceF64::mean_variance()` fused Welford shader (single dispatch)
//! - `CorrelationF64::correlation_full()` → `CorrelationResult` (single dispatch)
//! - `matrix_correlation()` → p×p Pearson matrix (single dispatch)
//! - `GuardedDeviceHandle` RAII encoder barriers
//! - `NpuDispatch`, `GpuAdapterInfo` added to `ToadStool`
//! - D-SOV resolved (capability-based discovery)
//!
//! ## Cross-spring provenance
//!
//! | Op | Primary origin | Secondary |
//! |----|---------------|-----------|
//! | Fused mean+variance | hotSpring Welford | neuralSpring validation |
//! | Fused correlation | wetSpring bio + hotSpring precision | neuralSpring ML |
//! | Matrix correlation | airSpring sensors + groundSpring stats | all springs |
//! | Shannon entropy | wetSpring diversity | hotSpring fused map-reduce |
//! | Chi-squared | neuralSpring → `BarraCUDA` S76 | hotSpring f64 pipeline |
//! | KL divergence | neuralSpring → `BarraCUDA` S76 | hotSpring f64 pipeline |
//!
//! ## Provenance
//!
//! Cross-spring origin: hotSpring, wetSpring, airSpring, groundSpring, neuralSpring → `ToadStool` S94b/`BarraCUDA` v0.3.3 → neuralSpring.
//! Absorption: S125 sync — wgpu 22→28, VarianceF64/CorrelationF64 fused ops, `GuardedDeviceHandle`, `NpuDispatch`.
//! Validation: wgpu 28 API, `BarraCUDA` v0.3.3 fused ops vs CPU reference.
//!
//! ```text
//! cargo run --release --bin validate_toadstool_s94b_wgpu28
//! ```

#![expect(
    clippy::expect_used,
    reason = "validation binary — direct assertions on known-good data"
)]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_ops;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn validate_fused_mean_variance(h: &mut ValidationHarness, gpu: &Gpu) {
    eprintln!("\n── VarianceF64::mean_variance() (hotSpring Welford → BarraCUDA v0.3.3) ──");
    let dev = gpu.wgpu_device();
    let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    match gpu_ops::mean_variance_gpu(&data, dev) {
        Ok([m, v]) => {
            h.check_abs("fused_mean", m, 5.0, tolerances::CROSS_LANGUAGE);
            h.check_abs("fused_variance", v, 4.0, tolerances::GPU_CHI_SQUARED_F32);
            eprintln!("    [PASS] mean={m:.6}, variance={v:.6} (single Welford dispatch)");
        }
        Err(e) => {
            eprintln!("    [SKIP] mean_variance_gpu failed: {e}");
            h.check_bool("fused_mean_variance_available", false);
        }
    }
}

fn validate_fused_correlation(h: &mut ValidationHarness, gpu: &Gpu) {
    eprintln!(
        "\n── CorrelationF64::correlation_full() (wetSpring bio + hotSpring precision → BarraCUDA v0.3.3) ──"
    );
    let dev = gpu.wgpu_device();
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.0, 4.0, 6.0, 8.0, 10.0];

    match gpu_ops::correlation_full_gpu(&x, &y, dev) {
        Ok(result) => {
            h.check_abs(
                "corr_full_pearson_r",
                result.pearson_r,
                1.0,
                tolerances::GPU_PEARSON_F32,
            );
            h.check_abs(
                "corr_full_mean_x",
                result.mean_x,
                3.0,
                tolerances::CROSS_LANGUAGE,
            );
            h.check_abs(
                "corr_full_mean_y",
                result.mean_y,
                6.0,
                tolerances::CROSS_LANGUAGE,
            );
            h.check_bool("corr_full_var_x_positive", result.var_x > 0.0);
            h.check_bool("corr_full_var_y_positive", result.var_y > 0.0);
            eprintln!(
                "    [PASS] r={:.6}, mean_x={:.6}, mean_y={:.6}, var_x={:.6}, var_y={:.6}",
                result.pearson_r, result.mean_x, result.mean_y, result.var_x, result.var_y
            );
        }
        Err(e) => {
            eprintln!("    [SKIP] correlation_full_gpu failed: {e}");
            h.check_bool("correlation_full_available", false);
        }
    }
}

fn validate_correlation_matrix(h: &mut ValidationHarness, gpu: &Gpu) {
    eprintln!(
        "\n── matrix_correlation() (airSpring sensors + groundSpring stats → BarraCUDA v0.3.3) ──"
    );
    let dev = gpu.wgpu_device();
    // 4 samples, 3 features: col0=[1,2,3,4], col1=[2,4,6,8], col2=[4,3,2,1]
    let data = [
        1.0, 2.0, 4.0, //
        2.0, 4.0, 3.0, //
        3.0, 6.0, 2.0, //
        4.0, 8.0, 1.0, //
    ];

    match gpu_ops::correlation_matrix_gpu(&data, 4, 3, dev) {
        Ok(corr) => {
            h.check_bool("corr_matrix_size", corr.len() == 9);
            h.check_abs(
                "corr_matrix_diag_00",
                corr[0],
                1.0,
                tolerances::GPU_PEARSON_F32,
            );
            h.check_abs(
                "corr_matrix_diag_11",
                corr[4],
                1.0,
                tolerances::GPU_PEARSON_F32,
            );
            h.check_abs(
                "corr_matrix_diag_22",
                corr[8],
                1.0,
                tolerances::GPU_PEARSON_F32,
            );
            h.check_abs(
                "corr_matrix_01_perfect",
                corr[1],
                1.0,
                tolerances::GPU_PEARSON_F32,
            );
            h.check_abs(
                "corr_matrix_02_inverse",
                corr[2],
                -1.0,
                tolerances::GPU_PEARSON_F32,
            );
            eprintln!("    [PASS] 3×3 matrix: diag all 1.0, [0,1]=+1.0, [0,2]=-1.0");
        }
        Err(e) => {
            eprintln!("    [SKIP] correlation_matrix_gpu failed: {e}");
            h.check_bool("correlation_matrix_available", false);
        }
    }
}

fn validate_existing_fused_ops(h: &mut ValidationHarness, gpu: &Gpu) {
    eprintln!("\n── Existing fused ops (neuralSpring → BarraCUDA, cross-spring) ──");
    let dev = gpu.wgpu_device();

    // Shannon entropy (wetSpring diversity → hotSpring fused map-reduce → BarraCUDA)
    match gpu_ops::shannon_entropy_gpu(&[0.25, 0.25, 0.25, 0.25], dev) {
        Ok(h_val) => {
            let expected = 4.0_f64.ln();
            h.check_abs(
                "shannon_uniform4",
                h_val,
                expected,
                tolerances::GPU_ENTROPY_F32,
            );
            eprintln!("    [PASS] Shannon H(uniform4) = {h_val:.6} (expected {expected:.6})");
        }
        Err(e) => {
            eprintln!("    [SKIP] shannon_entropy_gpu: {e}");
            h.check_bool("shannon_available", false);
        }
    }

    // Chi-squared (neuralSpring → BarraCUDA S76 → hotSpring f64 pipeline)
    match gpu_ops::chi_squared_gpu(&[10.0, 20.0, 30.0, 40.0], &[25.0, 25.0, 25.0, 25.0], dev) {
        Ok(chi2) => {
            h.check_abs("chi2_known", chi2, 20.0, tolerances::GPU_CHI_SQUARED_F32);
            eprintln!("    [PASS] χ²([10,20,30,40], [25,25,25,25]) = {chi2:.6} (expected 20)");
        }
        Err(e) => {
            eprintln!("    [SKIP] chi_squared_gpu: {e}");
            h.check_bool("chi2_available", false);
        }
    }

    // KL divergence (neuralSpring → BarraCUDA S76)
    match gpu_ops::kl_divergence_gpu(&[0.25, 0.25, 0.25, 0.25], &[0.25, 0.25, 0.25, 0.25], dev) {
        Ok(kl) => {
            h.check_abs("kl_identical", kl, 0.0, tolerances::GPU_KL_DISPATCH_F32);
            eprintln!("    [PASS] KL(p,p) = {kl:.6} (expected 0)");
        }
        Err(e) => {
            eprintln!("    [SKIP] kl_divergence_gpu: {e}");
            h.check_bool("kl_available", false);
        }
    }

    // Variance via fused Welford (hotSpring → BarraCUDA v0.3.3)
    match gpu_ops::variance_gpu(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], dev) {
        Ok(v) => {
            h.check_abs("variance_welford", v, 4.0, tolerances::GPU_CHI_SQUARED_F32);
            eprintln!("    [PASS] Var([2,4,4,4,5,5,7,9]) = {v:.6} (expected 4)");
        }
        Err(e) => {
            eprintln!("    [SKIP] variance_gpu: {e}");
            h.check_bool("variance_available", false);
        }
    }

    // Pearson correlation (wetSpring bio + hotSpring precision → BarraCUDA)
    match gpu_ops::pearson_correlation_gpu(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], dev) {
        Ok(r) => {
            h.check_abs("pearson_perfect", r, 1.0, tolerances::GPU_PEARSON_F32);
            eprintln!("    [PASS] Pearson r(x, 2x) = {r:.6} (expected 1)");
        }
        Err(e) => {
            eprintln!("    [SKIP] pearson_correlation_gpu: {e}");
            h.check_bool("pearson_available", false);
        }
    }
}

fn validate_wgpu28_api(h: &mut ValidationHarness, gpu: &Gpu) {
    eprintln!("\n── wgpu 28 API surface (PollType::Wait, immediate_size, etc.) ──");

    h.check_bool("gpu_adapter_available", !gpu.adapter_name.is_empty());
    eprintln!("    adapter: {}", gpu.adapter_name);
    eprintln!("    device_type: {:?}", gpu.device_type);
    eprintln!("    backend: {:?}", gpu.backend);

    // wgpu 28 PollType::Wait exercised via Tensor creation + readback
    let dev = gpu.wgpu_device();
    match barracuda::tensor::Tensor::from_data(&[1.0_f32, 2.0, 3.0], vec![3], dev.clone()) {
        Ok(t) => match t.to_vec() {
            Ok(v) => {
                h.check_bool("wgpu28_tensor_roundtrip", v == [1.0, 2.0, 3.0]);
                eprintln!("    [PASS] Tensor f32 roundtrip via wgpu 28 pipeline");
            }
            Err(e) => {
                eprintln!("    [SKIP] Tensor readback failed (upstream SIGSEGV?): {e}");
                h.check_bool("wgpu28_tensor_readback", false);
            }
        },
        Err(e) => {
            eprintln!("    [SKIP] Tensor creation failed: {e}");
            h.check_bool("wgpu28_tensor_creation", false);
        }
    }
}

fn main() {
    eprintln!("╔════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  ToadStool S94b + wgpu 28 + BarraCUDA v0.3.3 Validation              ║");
    eprintln!("║  Session 126 — Cross-spring fused op absorption + API migration       ║");
    eprintln!("╚════════════════════════════════════════════════════════════════════════╝");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu_result = rt.block_on(async { Gpu::new().await });

    let mut h = ValidationHarness::new("toadstool_s94b_wgpu28");

    match gpu_result {
        Ok(gpu) => {
            validate_wgpu28_api(&mut h, &gpu);
            validate_fused_mean_variance(&mut h, &gpu);
            validate_fused_correlation(&mut h, &gpu);
            validate_correlation_matrix(&mut h, &gpu);
            validate_existing_fused_ops(&mut h, &gpu);
        }
        Err(e) => {
            eprintln!("\n  [SKIP] No GPU available: {e}");
            eprintln!("  GPU tests cannot run without a GPU adapter.");
            h.check_bool("gpu_available", false);
        }
    }

    h.finish();
}
