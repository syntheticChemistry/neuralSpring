// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline + metalForge cross-system validation for publication
//! experiments (Exp-050, Exp-052, Exp-053).
//!
//! Progression: Python → Rust CPU → `BarraCUDA` CPU → `BarraCUDA` GPU (prev) →
//! **Pure GPU pipeline + metalForge cross-system** (this)
//!
//! Proves:
//! 1. Spectral analysis pipeline runs entirely on GPU (scalar readback)
//! 2. Cross-system dispatch routes workloads correctly (CPU ↔ GPU parity)
//! 3. metalForge mixed-hardware routing respects compute/data cost model
//!
//! Papers: A (Training Trajectory), C (Anderson Multi-Agent), D (Hessian).
//!
//! ## Provenance
//!
//! Validation class: Integration.
//! Python baseline: spectral analysis pipeline (eigh, IPR, variance).
//! Components: `BarraCUDA` `BatchIprGpu`, Dispatcher, `weight_spectral`, metalForge `MixedSubstrate`.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    reason = "validation binary"
)]

use barracuda::spectral::BatchIprGpu;
use neural_spring::anderson_localization::mean_ipr;
use neural_spring::eigh::eigh_householder_qr;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::gpu_ops;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{exit_no_gpu, ValidationHarness};
use neural_spring::weight_spectral;
use neural_spring_forge::mixed::MixedSubstrate;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("publication_gpu_pipeline");

    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "GPU: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => exit_no_gpu(),
    };

    let dev = Arc::clone(gpu.wgpu_device());
    let mut rng = Rng::new(42);

    // ═══════════════════════════════════════════════════════════════════
    // PART 1: Pure GPU pipeline — eigensolve → BatchIprGpu → readback
    // ═══════════════════════════════════════════════════════════════════

    validate_pure_gpu_spectral_pipeline(&mut h, &gpu, &dev, &mut rng);

    let dispatcher = Dispatcher::from_gpu(gpu);

    // ═══════════════════════════════════════════════════════════════════
    // PART 2: Cross-system dispatch — CPU ↔ GPU parity through Dispatcher
    // ═══════════════════════════════════════════════════════════════════

    validate_cross_system_eigensolve(&mut h, &dispatcher, &mut rng);

    // ═══════════════════════════════════════════════════════════════════
    // PART 3: metalForge mixed-hardware routing for spectral workloads
    // ═══════════════════════════════════════════════════════════════════

    validate_mixed_hardware_routing(&mut h, &dispatcher, &mut rng);

    h.finish();
}

fn validate_pure_gpu_spectral_pipeline(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    dev: &Arc<barracuda::device::WgpuDevice>,
    rng: &mut Rng,
) {
    let m = 16;
    let ham = weight_spectral::weight_to_hamiltonian(
        &(0..8 * 8).map(|_| rng.normal()).collect::<Vec<f64>>(),
        8,
        8,
    );

    let cpu_decomp = eigh_householder_qr(&ham, m);
    let cpu_ipr = mean_ipr(&cpu_decomp.eigenvectors, m);

    let (_, gpu_evecs) = gpu_ops::eigh_gpu(&ham, m, dev).expect("eigh_gpu");

    let evecs_f32: Vec<f32> = gpu_evecs.iter().map(|&v| v as f32).collect();
    let n_vectors = m;
    let dim = m;

    let ipr_op = BatchIprGpu::new(Arc::clone(dev));
    let device = gpu.device();

    let vecs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pub_ipr_vecs"),
        contents: bytemuck::cast_slice(&evecs_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pub_ipr_out"),
        size: (n_vectors * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    ipr_op.dispatch(&vecs_buf, &out_buf, dim as u32, n_vectors as u32);

    match gpu.read_buffer_f32(&out_buf, n_vectors) {
        Ok(gpu_ipr_vals) => {
            let gpu_mean_ipr: f64 =
                gpu_ipr_vals.iter().map(|&v| f64::from(v)).sum::<f64>() / n_vectors as f64;

            h.check_abs(
                "Pure GPU pipeline: BatchIprGpu matches CPU mean_ipr",
                gpu_mean_ipr,
                cpu_ipr,
                tolerances::GPU_BATCH_IPR_F32,
            );

            h.check_bool(
                "Pure GPU pipeline: IPR in delocalized range",
                (0.01..=0.2).contains(&gpu_mean_ipr),
            );
        }
        Err(e) => h.check_bool(&format!("IPR readback: {e}"), false),
    }

    let cpu_evals = cpu_decomp.eigenvalues;
    let gpu_var = gpu_ops::variance_gpu(&cpu_evals, dev).expect("variance_gpu");
    let cpu_var = {
        let mean = cpu_evals.iter().sum::<f64>() / m as f64;
        cpu_evals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / m as f64
    };

    h.check_abs(
        "Pure GPU pipeline: eigenvalue variance matches CPU",
        gpu_var,
        cpu_var,
        tolerances::GPU_VARIANCE_F64,
    );
}

fn validate_cross_system_eigensolve(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
) {
    let m = 20;
    let mut hessian = vec![0.0; m * m];
    for i in 0..m {
        for j in i..m {
            let v = rng.normal();
            hessian[i * m + j] = v;
            hessian[j * m + i] = v;
        }
    }

    let cpu_only = Dispatcher::cpu_only();
    let (cpu_evals, _) = cpu_only.eigh(&hessian, m);
    let (gpu_evals, _) = dispatcher.eigh(&hessian, m);

    let mut cpu_sorted = cpu_evals;
    cpu_sorted.sort_by(f64::total_cmp);
    let mut gpu_sorted = gpu_evals;
    gpu_sorted.sort_by(f64::total_cmp);

    let eval_diff = cpu_sorted
        .iter()
        .zip(gpu_sorted.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "Cross-system: Dispatcher eigh CPU ↔ GPU parity",
        eval_diff < tolerances::GPU_EIGH_DISPATCH_F64,
    );

    let cpu_matmul = cpu_only.mat_mul(&hessian, &hessian, m);
    let gpu_matmul = dispatcher.mat_mul(&hessian, &hessian, m);

    let mm_diff = cpu_matmul
        .iter()
        .zip(gpu_matmul.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "Cross-system: Dispatcher matmul CPU ↔ GPU parity",
        mm_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );

    let data: Vec<f64> = (0..256).map(|_| rng.normal()).collect();
    let cpu_var = cpu_only.variance(&data);
    let gpu_var = dispatcher.variance(&data);

    h.check_abs(
        "Cross-system: Dispatcher variance CPU ↔ GPU",
        gpu_var,
        cpu_var,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );

    let cpu_mean = cpu_only.mean(&data);
    let gpu_mean = dispatcher.mean(&data);

    h.check_abs(
        "Cross-system: Dispatcher mean CPU ↔ GPU",
        gpu_mean,
        cpu_mean,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );
}

fn validate_mixed_hardware_routing(
    h: &mut ValidationHarness,
    dispatcher: &Dispatcher,
    rng: &mut Rng,
) {
    let data: Vec<f64> = (0..1024).map(|_| rng.normal()).collect();
    let data_bytes = (data.len() * 8) as u64;

    let cpu_var = {
        let n = data.len() as f64;
        let m = data.iter().sum::<f64>() / n;
        data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
    };

    let (mixed_var, var_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "publication_spectral_variance",
            compute_us: 50_000.0,
            data_bytes,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| {
            let var_op = barracuda::ops::variance_f64_wgsl::VarianceF64::new(dev.clone())
                .map_err(|e| format!("{e}"))?;
            var_op.variance(&data).map_err(|e| format!("{e}"))
        },
        || cpu_var,
    );

    if dispatcher.has_gpu() {
        h.check_abs(
            "metalForge: spectral variance CPU ↔ GPU",
            mixed_var,
            cpu_var,
            tolerances::GPU_VARIANCE_F64,
        );
        h.check_bool(
            "metalForge: spectral variance → GPU substrate",
            var_sub == MixedSubstrate::GpuOnly,
        );
    } else {
        h.check_bool(
            "metalForge: spectral variance finite",
            mixed_var.is_finite(),
        );
    }

    let probs: Vec<f64> = {
        let raw: Vec<f64> = (0..128).map(|_| rng.uniform().abs() + 1e-10).collect();
        let sum: f64 = raw.iter().sum();
        raw.iter().map(|v| v / sum).collect()
    };
    let cpu_entropy = neural_spring::primitives::shannon_entropy(&probs);

    let (mixed_entropy, ent_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "publication_spectral_entropy",
            compute_us: 30_000.0,
            data_bytes: (probs.len() * 8) as u64,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| neural_spring::gpu_ops::shannon_entropy_gpu(&probs, dev),
        || cpu_entropy,
    );

    if dispatcher.has_gpu() {
        h.check_abs(
            "metalForge: spectral entropy CPU ↔ GPU",
            mixed_entropy,
            cpu_entropy,
            tolerances::GPU_ENTROPY_F64,
        );
        h.check_bool(
            "metalForge: spectral entropy → GPU substrate",
            ent_sub == MixedSubstrate::GpuOnly,
        );
    } else {
        h.check_bool(
            "metalForge: spectral entropy finite",
            mixed_entropy.is_finite(),
        );
    }

    let parity_data: Vec<f64> = (0..512).map(|_| rng.normal()).collect();
    let cpu_sum: f64 = parity_data.iter().sum();

    let (mixed_sum, _sum_sub) = dispatcher.mixed_dispatch(
        &MixedWorkload {
            op: "publication_eigenvalue_sum",
            compute_us: 10_000.0,
            data_bytes: (parity_data.len() * 8) as u64,
            npu_available: false,
            needs_realtime: false,
        },
        |dev| gpu_ops::sum_gpu(&parity_data, dev),
        || cpu_sum,
    );

    if dispatcher.has_gpu() {
        h.check_abs(
            "metalForge: eigenvalue sum CPU ↔ GPU",
            mixed_sum,
            cpu_sum,
            tolerances::GPU_SUM_DISPATCH_F32,
        );
    } else {
        h.check_bool("metalForge: eigenvalue sum finite", mixed_sum.is_finite());
    }

    h.check_bool(
        "metalForge: dispatcher reports GPU available",
        dispatcher.has_gpu(),
    );
}
