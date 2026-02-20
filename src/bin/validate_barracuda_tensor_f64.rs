// SPDX-License-Identifier: AGPL-3.0-only

//! Validation binary: `BarraCUDA` f64 GPU operations (`SHADER_F64` path).
//!
//! Proves that GPU f64 WGSL shaders produce precision math matching
//! analytical/CPU expectations.  These ops require `SHADER_F64` support
//! (NVIDIA discrete GPUs, not llvmpipe).
//!
//! Discovery: hotSpring found that `BarraCUDA` provides `compile_shader_f64`
//! and a full f64 WGSL math library.  `neuralSpring` validates these here so
//! the `ToadStool` team can absorb our proofs for other Springs.
//!
//! ## Backend selection
//!
//! Uses `NEURALSPRING_BACKEND` (same as tensor validator).  The f64 path
//! requires a GPU with `SHADER_F64`; on llvmpipe this will gracefully skip.
//!
//! ## Provenance
//!
//! Expected values: analytical formulas computed in f64 Rust.

#![allow(clippy::cast_precision_loss)]

use barracuda::device::WgpuDevice;
use barracuda::ops::cosine_similarity_f64::CosineSimilarityF64;
use barracuda::ops::fused_map_reduce_f64::{FusedMapReduceF64, MapOp, ReduceOp};
use barracuda::ops::max_abs_diff_f64::MaxAbsDiffF64;
use barracuda::ops::norm_reduce_f64::NormReduceF64;
use barracuda::ops::sum_reduce_f64::SumReduceF64;
use barracuda::ops::variance_reduce_f64::VarianceReduceF64;
use barracuda::ops::weighted_dot_f64::WeightedDotF64;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: {e}");
            eprintln!("  0/0 checks — skipping gracefully");
            std::process::exit(0);
        }
    };

    eprintln!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );

    let device = gpu.wgpu_device().clone();

    if !device.has_f64_shaders() {
        eprintln!("SKIP: adapter does not support SHADER_F64");
        eprintln!("  0/0 checks — skipping gracefully");
        std::process::exit(0);
    }

    let mut h = ValidationHarness::new("barracuda_tensor_f64");

    validate_f64_roundtrip(&mut h, &device);
    validate_sum_reduce(&mut h, &device);
    validate_fused_map_reduce(&mut h, &device);
    validate_norm_reduce(&mut h, &device);
    validate_variance_reduce(&mut h, &device);
    validate_weighted_dot(&mut h, &device);
    validate_max_abs_diff(&mut h, &device);
    validate_cosine_similarity(&mut h, &device);

    h.finish();
}

// ── f64 Tensor round-trip ──────────────────────────────────────────────

fn validate_f64_roundtrip(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let data: Vec<f64> = vec![
        std::f64::consts::PI,
        std::f64::consts::E,
        -1.234_567_890_123_456_7,
        0.0,
        1e15,
    ];

    match Tensor::from_f64_data(&data, vec![5], device.clone()) {
        Ok(tensor) => match tensor.to_f64_vec() {
            Ok(out) => {
                for (i, (&expected, &observed)) in data.iter().zip(out.iter()).enumerate() {
                    h.check_abs(
                        &format!("f64 roundtrip [{i}]"),
                        observed,
                        expected,
                        tolerances::EXACT_F64,
                    );
                }
            }
            Err(e) => h.check_bool(&format!("f64 roundtrip readback [ERROR: {e}]"), false),
        },
        Err(e) => h.check_bool(&format!("f64 roundtrip upload [ERROR: {e}]"), false),
    }
}

// ── SumReduceF64 ───────────────────────────────────────────────────────

fn validate_sum_reduce(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let data: Vec<f64> = (1..=100).map(f64::from).collect();
    let expected_sum = 5050.0;
    let expected_max = 100.0;
    let expected_min = 1.0;
    let expected_mean = 50.5;

    match SumReduceF64::sum(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "SumReduceF64::sum",
            v,
            expected_sum,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("SumReduceF64::sum [ERROR: {e}]"), false),
    }
    match SumReduceF64::max(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "SumReduceF64::max",
            v,
            expected_max,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("SumReduceF64::max [ERROR: {e}]"), false),
    }
    match SumReduceF64::min(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "SumReduceF64::min",
            v,
            expected_min,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("SumReduceF64::min [ERROR: {e}]"), false),
    }
    match SumReduceF64::mean(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "SumReduceF64::mean",
            v,
            expected_mean,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("SumReduceF64::mean [ERROR: {e}]"), false),
    }
}

// ── FusedMapReduceF64 ──────────────────────────────────────────────────

fn validate_fused_map_reduce(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let fmr = match FusedMapReduceF64::new(device.clone()) {
        Ok(f) => f,
        Err(e) => {
            h.check_bool(&format!("FusedMapReduceF64::new [ERROR: {e}]"), false);
            return;
        }
    };

    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    match fmr.sum(&data) {
        Ok(v) => h.check_abs("FusedMapReduce::sum", v, 15.0, tolerances::GPU_F64_EXACT),
        Err(e) => h.check_bool(&format!("FusedMapReduce::sum [ERROR: {e}]"), false),
    }

    match fmr.sum_of_squares(&data) {
        Ok(v) => h.check_abs(
            "FusedMapReduce::sum_of_squares",
            v,
            55.0,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(
            &format!("FusedMapReduce::sum_of_squares [ERROR: {e}]"),
            false,
        ),
    }

    match fmr.l1_norm(&data) {
        Ok(v) => h.check_abs(
            "FusedMapReduce::l1_norm",
            v,
            15.0,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("FusedMapReduce::l1_norm [ERROR: {e}]"), false),
    }

    match fmr.max(&data) {
        Ok(v) => h.check_abs("FusedMapReduce::max", v, 5.0, tolerances::GPU_F64_EXACT),
        Err(e) => h.check_bool(&format!("FusedMapReduce::max [ERROR: {e}]"), false),
    }

    match fmr.min(&data) {
        Ok(v) => h.check_abs("FusedMapReduce::min", v, 1.0, tolerances::GPU_F64_EXACT),
        Err(e) => h.check_bool(&format!("FusedMapReduce::min [ERROR: {e}]"), false),
    }

    // Shannon entropy: H = -sum(p * ln(p)), p = counts / total
    let counts: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];
    let total: f64 = counts.iter().sum();
    let expected_shannon: f64 = counts
        .iter()
        .map(|&c| {
            let p = c / total;
            -p * p.ln()
        })
        .sum();

    match fmr.shannon_entropy(&counts) {
        Ok(v) => h.check_abs(
            "FusedMapReduce::shannon_entropy",
            v,
            expected_shannon,
            tolerances::GPU_F64_TRANSCENDENTAL,
        ),
        Err(e) => h.check_bool(&format!("FusedMapReduce::shannon [ERROR: {e}]"), false),
    }

    // Generic fused: Square + Sum = sum_of_squares
    match fmr.execute(&data, 0.0, MapOp::Square, ReduceOp::Sum) {
        Ok(v) => h.check_abs(
            "FusedMapReduce::execute(Square,Sum)",
            v,
            55.0,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("FusedMapReduce::execute [ERROR: {e}]"), false),
    }
}

// ── NormReduceF64 ──────────────────────────────────────────────────────

fn validate_norm_reduce(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let data: Vec<f64> = vec![3.0, -4.0];
    let expected_l1 = 7.0;
    let expected_l2 = 5.0;
    let expected_l2_sq = 25.0;
    let expected_linf = 4.0;

    match NormReduceF64::l1(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "NormReduceF64::l1",
            v,
            expected_l1,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("NormReduceF64::l1 [ERROR: {e}]"), false),
    }
    match NormReduceF64::l2(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "NormReduceF64::l2",
            v,
            expected_l2,
            tolerances::GPU_F64_TRANSCENDENTAL,
        ),
        Err(e) => h.check_bool(&format!("NormReduceF64::l2 [ERROR: {e}]"), false),
    }
    match NormReduceF64::l2_squared(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "NormReduceF64::l2_squared",
            v,
            expected_l2_sq,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("NormReduceF64::l2_squared [ERROR: {e}]"), false),
    }
    match NormReduceF64::linf(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "NormReduceF64::linf",
            v,
            expected_linf,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("NormReduceF64::linf [ERROR: {e}]"), false),
    }
    match NormReduceF64::frobenius(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "NormReduceF64::frobenius",
            v,
            expected_l2,
            tolerances::GPU_F64_TRANSCENDENTAL,
        ),
        Err(e) => h.check_bool(&format!("NormReduceF64::frobenius [ERROR: {e}]"), false),
    }
}

// ── VarianceReduceF64 ──────────────────────────────────────────────────

fn validate_variance_reduce(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let data: Vec<f64> = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let n = data.len() as f64;
    let mean_val: f64 = data.iter().sum::<f64>() / n;
    let pop_var: f64 = data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / n;
    let sample_var: f64 = data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (n - 1.0);
    let pop_std = pop_var.sqrt();
    let sample_std = sample_var.sqrt();

    match VarianceReduceF64::mean(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "VarianceReduceF64::mean",
            v,
            mean_val,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("VarianceReduceF64::mean [ERROR: {e}]"), false),
    }
    match VarianceReduceF64::population_variance(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "VarianceReduceF64::pop_variance",
            v,
            pop_var,
            tolerances::GPU_F64_STATS,
        ),
        Err(e) => h.check_bool(&format!("VarianceReduceF64::pop_var [ERROR: {e}]"), false),
    }
    match VarianceReduceF64::variance(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "VarianceReduceF64::sample_variance",
            v,
            sample_var,
            tolerances::GPU_F64_STATS,
        ),
        Err(e) => h.check_bool(&format!("VarianceReduceF64::var [ERROR: {e}]"), false),
    }
    match VarianceReduceF64::population_std(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "VarianceReduceF64::pop_std",
            v,
            pop_std,
            tolerances::GPU_F64_TRANSCENDENTAL,
        ),
        Err(e) => h.check_bool(&format!("VarianceReduceF64::pop_std [ERROR: {e}]"), false),
    }
    match VarianceReduceF64::std(device.clone(), &data) {
        Ok(v) => h.check_abs(
            "VarianceReduceF64::sample_std",
            v,
            sample_std,
            tolerances::GPU_F64_TRANSCENDENTAL,
        ),
        Err(e) => h.check_bool(&format!("VarianceReduceF64::std [ERROR: {e}]"), false),
    }
}

// ── WeightedDotF64 ─────────────────────────────────────────────────────

fn validate_weighted_dot(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let wd = match WeightedDotF64::new(device.clone()) {
        Ok(w) => w,
        Err(e) => {
            h.check_bool(&format!("WeightedDotF64::new [ERROR: {e}]"), false);
            return;
        }
    };

    let vec_a: Vec<f64> = vec![1.0, 2.0, 3.0];
    let vec_b: Vec<f64> = vec![4.0, 5.0, 6.0];
    let weights: Vec<f64> = vec![1.0, 2.0, 3.0];
    let expected_dot = 3.0f64.mul_add(6.0, 1.0f64.mul_add(4.0, 2.0 * 5.0));
    let expected_weighted =
        (3.0 * 3.0f64).mul_add(6.0, (1.0 * 1.0f64).mul_add(4.0, 2.0 * 2.0 * 5.0));
    let expected_norm_sq = 1.0 + 4.0 + 9.0;

    match wd.dot(&vec_a, &vec_b) {
        Ok(v) => h.check_abs(
            "WeightedDotF64::dot",
            v,
            expected_dot,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("WeightedDotF64::dot [ERROR: {e}]"), false),
    }
    match wd.weighted_dot(&weights, &vec_a, &vec_b) {
        Ok(v) => h.check_abs(
            "WeightedDotF64::weighted_dot",
            v,
            expected_weighted,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("WeightedDotF64::weighted [ERROR: {e}]"), false),
    }
    match wd.norm_squared(&vec_a) {
        Ok(v) => h.check_abs(
            "WeightedDotF64::norm_squared",
            v,
            expected_norm_sq,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("WeightedDotF64::norm_sq [ERROR: {e}]"), false),
    }
}

// ── MaxAbsDiffF64 ──────────────────────────────────────────────────────

fn validate_max_abs_diff(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let vec_a: Vec<f64> = vec![1.0, 5.0, 3.0, 10.0];
    let vec_b: Vec<f64> = vec![1.0, 2.0, 3.0, 3.0];
    let expected = 7.0; // |10 - 3|

    match MaxAbsDiffF64::compute(device.clone(), &vec_a, &vec_b) {
        Ok(v) => h.check_abs(
            "MaxAbsDiffF64::compute",
            v,
            expected,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("MaxAbsDiffF64::compute [ERROR: {e}]"), false),
    }

    let same: Vec<f64> = vec![1.0, 2.0, 3.0];
    match MaxAbsDiffF64::compute(device.clone(), &same, &same) {
        Ok(v) => h.check_abs(
            "MaxAbsDiffF64::compute(same) == 0",
            v,
            0.0,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("MaxAbsDiffF64::same [ERROR: {e}]"), false),
    }
}

// ── CosineSimilarityF64 ────────────────────────────────────────────────

fn validate_cosine_similarity(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let cs = match CosineSimilarityF64::new(device.clone()) {
        Ok(c) => c,
        Err(e) => {
            h.check_bool(&format!("CosineSimilarityF64::new [ERROR: {e}]"), false);
            return;
        }
    };

    let vec_a: Vec<f64> = vec![1.0, 0.0, 0.0];
    let vec_b: Vec<f64> = vec![0.0, 1.0, 0.0];
    match cs.similarity(&vec_a, &vec_b) {
        Ok(v) => h.check_abs(
            "CosineSimilarity orthogonal == 0",
            v,
            0.0,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("CosineSimilarity::orth [ERROR: {e}]"), false),
    }

    match cs.similarity(&vec_a, &vec_a) {
        Ok(v) => h.check_abs(
            "CosineSimilarity identical == 1",
            v,
            1.0,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("CosineSimilarity::same [ERROR: {e}]"), false),
    }

    let neg_a: Vec<f64> = vec![-1.0, 0.0, 0.0];
    match cs.similarity(&vec_a, &neg_a) {
        Ok(v) => h.check_abs(
            "CosineSimilarity opposite == -1",
            v,
            -1.0,
            tolerances::GPU_F64_EXACT,
        ),
        Err(e) => h.check_bool(&format!("CosineSimilarity::opp [ERROR: {e}]"), false),
    }

    let vec_c: Vec<f64> = vec![1.0, 1.0, 0.0];
    let vec_d: Vec<f64> = vec![1.0, 0.0, 0.0];
    let expected = 1.0 / 2.0_f64.sqrt();
    match cs.similarity(&vec_c, &vec_d) {
        Ok(v) => h.check_abs(
            "CosineSimilarity 45° ≈ 0.7071",
            v,
            expected,
            tolerances::GPU_F64_TRANSCENDENTAL,
        ),
        Err(e) => h.check_bool(&format!("CosineSimilarity::45 [ERROR: {e}]"), false),
    }
}
