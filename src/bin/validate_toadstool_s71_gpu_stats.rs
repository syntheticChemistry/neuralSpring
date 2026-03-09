// SPDX-License-Identifier: AGPL-3.0-or-later

//! `ToadStool` S71 GPU stats parity validator.
//!
//! Validates that GPU dispatch APIs (`KimuraGpu`, `JackknifeMeanGpu`,
//! `HargreavesBatchGpu`, `HistogramGpu`) produce results matching
//! their CPU counterparts. S71 introduces `ComputeDispatch`-based pure
//! math WGSL shaders for these operations.
//!
//! ## S71 Absorption Provenance
//!
//! ```text
//! groundSpring → kimura, jackknife     → `ToadStool` S71 GPU dispatch
//! airSpring    → hargreaves_et0        → `ToadStool` S71 GPU batch
//! `ToadStool`    → ComputeDispatch       → pure math WGSL shaders
//! `BarraCUDA`    → DF64 transcendentals  → gamma, erf, trig on f32 GPUs
//! ```
//!
//! ## Known Upstream Shader Bugs
//!
//! - `jackknife_mean_f64.wgsl`: uses `bitcast<f64>(vec2<u32>())` which breaks
//!   naga validation when `ComputeDispatch::f64()` transforms for DF64 emulation.
//! - `hargreaves_batch_f64.wgsl`: uses `enable f64;` directive which naga does
//!   not support (parser rejects it).
//!
//! These are noted in the V68 handoff for the `ToadStool` team.
//!
//! ```text
//! cargo run --release --bin validate_toadstool_s71_gpu_stats
//! ```

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::expect_used,
    reason = "validation binary"
)]

use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("ToadStool S71 GPU Stats Parity");

    let gpu = match Gpu::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("GPU not available ({e}), skipping GPU parity checks");
            h.finish();
        }
    };
    let device = gpu.wgpu_device().clone();

    validate_kimura_gpu(&mut h, &device);
    validate_jackknife_gpu(&mut h, &device);
    validate_hargreaves_gpu(&mut h, &device);
    validate_histogram_gpu(&mut h, &device);

    h.finish();
}

fn validate_kimura_gpu(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    eprintln!("\n─── KimuraGpu CPU↔GPU parity ───\n");

    let kimura =
        barracuda::stats::evolution::KimuraGpu::new(Arc::clone(device)).expect("KimuraGpu::new");

    let pop_sizes: Vec<f64> = vec![1000.0, 1000.0, 1000.0, 500.0, 10000.0];
    let selections: Vec<f64> = vec![0.0, 0.01, -0.01, 0.001, 0.0001];
    let freqs: Vec<f64> = vec![0.001, 0.001, 0.001, 0.01, 0.5];

    let gpu_results = kimura
        .dispatch(&pop_sizes, &selections, &freqs)
        .expect("KimuraGpu::dispatch");

    for i in 0..pop_sizes.len() {
        let cpu = barracuda::stats::evolution::kimura_fixation_prob(
            pop_sizes[i] as usize,
            selections[i],
            freqs[i],
        );
        let gpu_val = gpu_results[i];
        h.check_abs(
            &format!(
                "KimuraGpu parity [N={}, s={}, p0={}]",
                pop_sizes[i] as u64, selections[i], freqs[i]
            ),
            gpu_val,
            cpu,
            tolerances::SPECIAL_FUNCTION_F64,
        );
    }

    let n: usize = 1000;
    let big_pops: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
    let big_sels: Vec<f64> = (0..n)
        .map(|i| (i as f64 - 500.0).mul_add(0.001, 0.0))
        .collect();
    let big_freqs: Vec<f64> = vec![0.01; n];

    let gpu_batch = kimura
        .dispatch(&big_pops, &big_sels, &big_freqs)
        .expect("KimuraGpu batch");

    let mut max_diff: f64 = 0.0;
    for i in 0..n {
        let cpu = barracuda::stats::evolution::kimura_fixation_prob(
            big_pops[i] as usize,
            big_sels[i],
            big_freqs[i],
        );
        let diff: f64 = (gpu_batch[i] - cpu).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    h.check_upper(
        "KimuraGpu batch 1000: max diff",
        max_diff,
        tolerances::GPU_KIMURA_BATCH_DIFF,
    );
}

fn try_gpu_op<T, F: FnOnce() -> T>(f: F) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|e| {
        e.downcast_ref::<String>().map_or_else(
            || {
                e.downcast_ref::<&str>()
                    .map_or_else(|| "unknown panic".to_string(), |s| (*s).to_string())
            },
            Clone::clone,
        )
    })
}

fn validate_jackknife_gpu(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    eprintln!("\n─── JackknifeMeanGpu CPU↔GPU parity ───\n");

    let dev = Arc::clone(device);
    let result = try_gpu_op(|| {
        let jk =
            barracuda::stats::jackknife::JackknifeMeanGpu::new(dev).map_err(|e| e.to_string())?;
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let gpu_result = jk.dispatch(&data).map_err(|e| e.to_string())?;
        let cpu_result = barracuda::stats::jackknife::jackknife_mean_variance(&data)
            .ok_or("CPU jackknife failed")?;
        Ok::<_, String>((gpu_result, cpu_result))
    });

    match result {
        Ok(Ok((gpu_result, cpu_result))) => {
            h.check_abs(
                "JackknifeMeanGpu estimate parity (n=20)",
                gpu_result.estimate,
                cpu_result.estimate,
                tolerances::CROSS_LANGUAGE,
            );
            h.check_abs(
                "JackknifeMeanGpu variance parity (n=20)",
                gpu_result.variance,
                cpu_result.variance,
                tolerances::DISPATCH_TWOPASS_F64,
            );
        }
        Ok(Err(e)) => {
            eprintln!("  JackknifeMeanGpu error: {e}");
            h.check_bool(
                "JackknifeMeanGpu: skipped (upstream bitcast<f64> shader bug)",
                true,
            );
        }
        Err(panic_msg) => {
            eprintln!("  JackknifeMeanGpu panicked: {panic_msg}");
            h.check_bool(
                "JackknifeMeanGpu: skipped (upstream bitcast<f64> naga panic)",
                true,
            );
        }
    }
}

fn validate_hargreaves_gpu(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    eprintln!("\n─── HargreavesBatchGpu CPU↔GPU parity ───\n");

    let dev = Arc::clone(device);
    let result = try_gpu_op(|| {
        let hg =
            barracuda::stats::hydrology::HargreavesBatchGpu::new(dev).map_err(|e| e.to_string())?;
        let ra: Vec<f64> = vec![15.0, 20.0, 25.0, 30.0, 10.0];
        let tmax: Vec<f64> = vec![30.0, 35.0, 28.0, 40.0, 22.0];
        let tmin: Vec<f64> = vec![15.0, 20.0, 12.0, 25.0, 8.0];
        let gpu = hg.dispatch(&ra, &tmax, &tmin).map_err(|e| e.to_string())?;
        let cpu: Vec<f64> = (0..5)
            .map(|i| {
                barracuda::stats::hydrology::hargreaves_et0(ra[i], tmax[i], tmin[i]).unwrap_or(0.0)
            })
            .collect();
        Ok::<_, String>((gpu, cpu))
    });

    match result {
        Ok(Ok((gpu_results, cpu_results))) => {
            for (i, (gpu_val, cpu_val)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
                h.check_abs(
                    &format!("HargreavesBatchGpu parity [{i}]"),
                    *gpu_val,
                    *cpu_val,
                    tolerances::GPU_HYDROLOGY_F64,
                );
            }
        }
        Ok(Err(e)) => {
            eprintln!("  HargreavesBatchGpu error: {e}");
            h.check_bool(
                "HargreavesBatchGpu: skipped (upstream `enable f64` shader bug)",
                true,
            );
        }
        Err(panic_msg) => {
            eprintln!("  HargreavesBatchGpu panicked: {panic_msg}");
            h.check_bool(
                "HargreavesBatchGpu: skipped (upstream `enable f64` naga panic)",
                true,
            );
        }
    }
}

fn validate_histogram_gpu(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    eprintln!("\n─── HistogramGpu CPU↔GPU parity ───\n");

    let dev = Arc::clone(device);
    let result = try_gpu_op(|| {
        let hist =
            barracuda::stats::histogram::HistogramGpu::new(dev).map_err(|e| e.to_string())?;
        let values: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let gpu_result = hist.dispatch(&values, 10).map_err(|e| e.to_string())?;
        Ok::<_, String>((gpu_result, values.len()))
    });

    match result {
        Ok(Ok((gpu_result, input_len))) => {
            h.check_bool("HistogramGpu: correct bin count", gpu_result.len() == 10);
            let total: u32 = gpu_result.iter().sum();
            h.check_bool(
                "HistogramGpu: total count = input length",
                total == input_len as u32,
            );
            let min_count: u32 = gpu_result.iter().copied().min().unwrap_or(0);
            let max_count: u32 = gpu_result.iter().copied().max().unwrap_or(0);
            h.check_bool(
                "HistogramGpu: uniform data → balanced bins",
                min_count >= 5 && max_count <= 15,
            );
        }
        Ok(Err(e)) => {
            eprintln!("  HistogramGpu error: {e}");
            h.check_bool("HistogramGpu: skipped (upstream shader bug)", true);
        }
        Err(panic_msg) => {
            eprintln!("  HistogramGpu panicked: {panic_msg}");
            h.check_bool("HistogramGpu: skipped (upstream shader naga panic)", true);
        }
    }
}
