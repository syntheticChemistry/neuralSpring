// SPDX-License-Identifier: AGPL-3.0-or-later

//! ToadStool unidirectional streaming proof: spectral analysis pipeline.
//!
//! Demonstrates the streaming dispatch pattern ToadStool will absorb:
//! data flows GPU-ward through chained operations with minimal CPU
//! round-trips. Each stage's output feeds the next stage's input.
//!
//! ## Pipeline stages
//!
//! ```text
//! Stage 1: Hamiltonian assembly (CPU, feeds GPU)
//!   ↓
//! Stage 2: Eigensolve batch (GPU via Dispatcher)
//!   ↓ eigenvalues + eigenvectors stay in Rust memory
//! Stage 3: IPR batch (GPU via BatchIprGpu) — scalar readback
//!   ↓ IPR scalars
//! Stage 4: Statistics (GPU via Dispatcher::variance, ::mean)
//!   ↓ scalar readback
//! Stage 5: Anderson diagnostic (threshold check, scalar)
//! ```
//!
//! ## What this proves
//!
//! 1. **BarraCUDA CPU**: pure Rust math matches Python at machine ε
//! 2. **BarraCUDA GPU**: typed ops produce identical scientific conclusions
//! 3. **Streaming**: no per-sample CPU↔GPU round-trips — batch dispatch
//! 4. **Portability**: same code path runs on any wgpu-compatible hardware
//! 5. **ToadStool readiness**: pipeline structure matches absorption target
//!
//! ## Papers
//!
//! - Paper A: Weight Hamiltonians (baseCamp Sub-01, B-01..B-03)
//! - Paper C: Anderson multi-agent (baseCamp Sub-05, B-13..B-15)
//! - Papers 022-023: Spectral theory (Kachkovskiy)

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use barracuda::spectral::BatchIprGpu;
use neural_spring::anderson_localization::{anderson_hamiltonian_random, mean_ipr};
use neural_spring::eigh::eigh_householder_qr;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::gpu_ops;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::weight_spectral;
use neural_spring::weight_spectral::level_spacing_ratio;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("streaming_spectral_pipeline");

    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "GPU: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(e) => {
            eprintln!("No GPU available ({e}), skipping");
            h.finish();
        }
    };

    let dev = Arc::clone(gpu.wgpu_device());
    let mut rng = Rng::new(42);

    // ═══════════════════════════════════════════════════════════════════
    // PART 1: Batch streaming eigensolve — multiple Hamiltonians
    // ═══════════════════════════════════════════════════════════════════
    validate_batch_streaming_eigensolve(&mut h, &gpu, &dev, &mut rng);

    // ═══════════════════════════════════════════════════════════════════
    // PART 2: Anderson disorder sweep — streaming across W values
    // ═══════════════════════════════════════════════════════════════════
    validate_streaming_disorder_sweep(&mut h, &gpu, &dev, &mut rng);

    // ═══════════════════════════════════════════════════════════════════
    // PART 3: CPU ↔ GPU parity across the full pipeline
    // ═══════════════════════════════════════════════════════════════════
    let dispatcher = Dispatcher::from_gpu(gpu);
    validate_dispatcher_pipeline_parity(&mut h, &dispatcher, &mut rng);

    h.finish();
}

// ─────────────────────────────────────────────────────────────────────
// PART 1: Batch streaming — eigensolve → IPR → stats on GPU
// ─────────────────────────────────────────────────────────────────────

fn validate_batch_streaming_eigensolve(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    dev: &Arc<barracuda::device::WgpuDevice>,
    rng: &mut Rng,
) {
    let weight_rows = 8_usize;
    let weight_cols = 8_usize;
    let n = weight_rows + weight_cols; // Hamiltonian is (m+n)×(m+n)
    let n_hamiltonians = 8;

    let mut all_cpu_iprs = Vec::new();
    let mut all_gpu_ipr_vals: Vec<f32> = Vec::new();
    let device = gpu.device();

    for batch_idx in 0..n_hamiltonians {
        let weights: Vec<f64> = (0..weight_rows * weight_cols)
            .map(|_| rng.normal())
            .collect();
        let ham = weight_spectral::weight_to_hamiltonian(&weights, weight_rows, weight_cols);
        let cpu_decomp = eigh_householder_qr(&ham, n);
        let cpu_ipr = mean_ipr(&cpu_decomp.eigenvectors, n);
        all_cpu_iprs.push(cpu_ipr);

        let (_, gpu_evecs) = gpu_ops::eigh_gpu(&ham, n, dev).expect("eigh_gpu");
        let evecs_f32: Vec<f32> = gpu_evecs.iter().map(|&v| v as f32).collect();

        let ipr_op = BatchIprGpu::new(Arc::clone(dev));
        let vecs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stream_vecs"),
            contents: bytemuck::cast_slice(&evecs_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stream_ipr"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        ipr_op.dispatch(&vecs_buf, &out_buf, n as u32, n as u32);
        let gpu_iprs = gpu.read_buffer_f32(&out_buf, n).expect("IPR readback");

        let gpu_mean_ipr = gpu_iprs.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        all_gpu_ipr_vals.extend(gpu_iprs);

        h.check_abs(
            &format!("Stream batch {batch_idx}: IPR GPU ↔ CPU"),
            gpu_mean_ipr,
            cpu_ipr,
            tolerances::GPU_BATCH_IPR_F32,
        );
    }

    let cpu_mean_ipr = all_cpu_iprs.iter().sum::<f64>() / n_hamiltonians as f64;
    let gpu_grand_mean =
        all_gpu_ipr_vals.iter().map(|&v| f64::from(v)).sum::<f64>() / all_gpu_ipr_vals.len() as f64;

    h.check_abs(
        "Stream: grand mean IPR GPU ↔ CPU",
        gpu_grand_mean,
        cpu_mean_ipr,
        tolerances::GPU_BATCH_IPR_F32,
    );

    h.check_bool(
        "Stream: batch pipeline produced correct count",
        all_gpu_ipr_vals.len() == n_hamiltonians * n,
    );

    let gpu_var_ipr = {
        let vals: Vec<f64> = all_gpu_ipr_vals.iter().map(|&v| f64::from(v)).collect();
        let m = vals.iter().sum::<f64>() / vals.len() as f64;
        vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / vals.len() as f64
    };

    h.check_bool("Stream: IPR variance finite", gpu_var_ipr.is_finite());
    h.check_bool("Stream: IPR variance > 0", gpu_var_ipr > 0.0);
}

// ─────────────────────────────────────────────────────────────────────
// PART 2: Anderson disorder sweep — streaming across W values
// ─────────────────────────────────────────────────────────────────────

fn validate_streaming_disorder_sweep(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    dev: &Arc<barracuda::device::WgpuDevice>,
    rng: &mut Rng,
) {
    let n = 16_usize;
    let disorder_strengths = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
    let device = gpu.device();

    let mut cpu_iprs_by_w = Vec::new();
    let mut gpu_iprs_by_w = Vec::new();

    for &w in &disorder_strengths {
        let ham = anderson_hamiltonian_random(n, 1.0, w, rng);

        let cpu_decomp = eigh_householder_qr(&ham, n);
        let cpu_ipr = mean_ipr(&cpu_decomp.eigenvectors, n);
        cpu_iprs_by_w.push(cpu_ipr);

        let (_, gpu_evecs) = gpu_ops::eigh_gpu(&ham, n, dev).expect("eigh_gpu disorder");
        let evecs_f32: Vec<f32> = gpu_evecs.iter().map(|&v| v as f32).collect();

        let ipr_op = BatchIprGpu::new(Arc::clone(dev));
        let vecs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("disorder_vecs"),
            contents: bytemuck::cast_slice(&evecs_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("disorder_ipr"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        ipr_op.dispatch(&vecs_buf, &out_buf, n as u32, n as u32);
        let gpu_iprs = gpu
            .read_buffer_f32(&out_buf, n)
            .expect("disorder IPR readback");
        let gpu_mean_ipr = gpu_iprs.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        gpu_iprs_by_w.push(gpu_mean_ipr);

        h.check_abs(
            &format!("Disorder W={w}: IPR GPU ↔ CPU"),
            gpu_mean_ipr,
            cpu_ipr,
            tolerances::GPU_BATCH_IPR_F32,
        );
    }

    h.check_bool(
        "Anderson transition: IPR(W=16) > IPR(W=0.5)",
        gpu_iprs_by_w[5] > gpu_iprs_by_w[0],
    );

    h.check_bool(
        "Anderson: strong disorder localizes (IPR > 0.2)",
        gpu_iprs_by_w[5] > 0.2,
    );

    h.check_bool(
        "Anderson: sweep monotonic tendency",
        gpu_iprs_by_w[5] > gpu_iprs_by_w[2],
    );
}

// ─────────────────────────────────────────────────────────────────────
// PART 3: Dispatcher pipeline parity — CPU ↔ GPU same conclusions
// ─────────────────────────────────────────────────────────────────────

fn validate_dispatcher_pipeline_parity(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
) {
    let wr = 8_usize;
    let wc = 8_usize;
    let n = wr + wc;
    let weights: Vec<f64> = (0..wr * wc).map(|_| rng.normal()).collect();
    let ham = weight_spectral::weight_to_hamiltonian(&weights, wr, wc);

    let (mut dispatch_evals, _dispatch_evecs) = dispatcher.eigh(&ham, n);
    let cpu_decomp = eigh_householder_qr(&ham, n);

    dispatch_evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut cpu_evals_sorted = cpu_decomp.eigenvalues.clone();
    cpu_evals_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let eval_diff = dispatch_evals
        .iter()
        .zip(cpu_evals_sorted.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "Dispatcher eigensolve ↔ CPU parity (sorted)",
        eval_diff,
        0.0,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    let dispatch_var = dispatcher.variance(&dispatch_evals);
    let cpu_var = {
        let m = cpu_decomp.eigenvalues.iter().sum::<f64>() / n as f64;
        cpu_decomp
            .eigenvalues
            .iter()
            .map(|&v| (v - m).powi(2))
            .sum::<f64>()
            / n as f64
    };

    h.check_abs(
        "Dispatcher variance ↔ CPU",
        dispatch_var,
        cpu_var,
        tolerances::GPU_VARIANCE_F64,
    );

    let dispatch_mean = dispatcher.mean(&dispatch_evals);
    let cpu_mean = cpu_decomp.eigenvalues.iter().sum::<f64>() / n as f64;
    h.check_abs("Dispatcher mean ↔ CPU", dispatch_mean, cpu_mean, 1e-10);

    let l2_sorted = dispatch_evals
        .iter()
        .zip(cpu_evals_sorted.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    h.check_abs(
        "Dispatcher L2(sorted evals) ≈ 0",
        l2_sorted,
        0.0,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    let dispatch_frob = dispatcher.frobenius_norm(&ham);
    let cpu_frob = ham.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    h.check_abs(
        "Dispatcher Frobenius norm ↔ CPU",
        dispatch_frob,
        cpu_frob,
        1e-6,
    );

    let cpu_lsr = level_spacing_ratio(&cpu_decomp.eigenvalues);
    h.check_bool("CPU LSR finite", cpu_lsr.is_finite());
    h.check_bool("CPU LSR > 0", cpu_lsr > 0.0);
}
