// SPDX-License-Identifier: AGPL-3.0-or-later

//! Modern rewire benchmark: validates and benchmarks the S88+ rewiring of
//! neuralSpring local implementations to upstream `ToadStool`/`BarraCUDA` APIs.
//!
//! ## Rewires Benchmarked
//!
//! | Local (pre-S88) | Upstream (post-S88) | Provenance |
//! |-----------------|---------------------|------------|
//! | `pairwise_l2_matrix_gpu` O(n²) loop | `PairwiseL2Gpu` single dispatch | nS metalForge → `ToadStool` S52 |
//! | `geographic_distance_matrix_gpu` O(n²) loop | `PairwiseL2Gpu` via above | nS 025 `MetaPop` → `ToadStool` S52 |
//! | `disorder_sweep_gpu` CPU IPR loop | `BatchIprGpu` GPU dispatch | nS 022-023 → `ToadStool` S52 |
//!
//! ## Modern APIs Benchmarked
//!
//! | API | Provenance | Spring Origins |
//! |-----|-----------|----------------|
//! | `LogSumExp` | hotSpring precision → `ToadStool` S64 | hotSpring f64 log-domain HMM |
//! | `PairwiseDistance` | neuralSpring MODES → `ToadStool` S52 | nS novelty search |
//! | `BatchedEighGpu` | hotSpring HFB → `ToadStool` S56 | hotSpring Jacobi sweeps |
//! | `BatchIprGpu` | neuralSpring Anderson → `ToadStool` S52 | nS IPR localization |
//! | `DiversityFusionGpu` | wetSpring diversity → `ToadStool` S64 | wS Shannon+Simpson fused |
//! | Dispatcher variance | hotSpring Welford → `ToadStool` S62 | hS precision accumulation |
//! | Dispatcher pearson | wetSpring+hotSpring → `ToadStool` S64 | cross-spring correlation |
//!
//! ## Cross-Spring Shader Evolution
//!
//! ```text
//! hotSpring (precision physics)    → DF64 core, eigensolve, Welford variance, logsumexp
//! wetSpring (bioinformatics)       → Shannon, Simpson, HMM, diversity fusion, Bray-Curtis
//! neuralSpring (ML/neuroevolution) → pairwise L2, IPR, batch fitness, swarm NN, MHA
//! airSpring (atmospheric)          → RMSE, R², NSE, fit_linear, moving_window
//! groundSpring (hydrology)         → multinomial sampling, MC propagation, ET₀
//! ```
//!
//! All absorbed into `ToadStool`'s 844+ f64-canonical WGSL shaders (2dc26792 S87).
//!
//! # Panics
//!
//! Panics if the tokio runtime cannot be created — this is a benchmark binary.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::expect_used,
    clippy::suboptimal_flops,
    reason = "validation binary"
)]

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::DiversityFusionGpu;
use barracuda::ops::linalg::BatchedEighGpu;
use barracuda::ops::logsumexp::LogSumExp;
use barracuda::ops::pairwise_distance::PairwiseDistance;
use barracuda::spectral::BatchIprGpu;
use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_ops::{
    disorder_sweep_gpu, eigh_gpu, geographic_distance_matrix_gpu, pairwise_l2_matrix_gpu,
};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 30;

fn bench<F: FnMut()>(label: &str, mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    let elapsed = start.elapsed();
    let us_per_iter = elapsed.as_micros() as f64 / ITERS as f64;
    eprintln!("    {label}: {us_per_iter:.1}µs/iter");
    us_per_iter
}

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  neuralSpring — Modern Rewire & Cross-Spring Provenance Bench   ║");
    eprintln!("║  BarraCUDA (ToadStool S87, 2dc26792) · 844+ f64-canonical WGSL   ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let rt = tokio::runtime::Runtime::new()
        .expect("tokio runtime creation failed — required for async benchmark");
    let gpu = rt
        .block_on(async { Gpu::new().await })
        .expect("GPU required for benchmark — no adapter available");

    eprintln!(
        "  GPU: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend
    );
    eprintln!();

    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("modern_rewire_bench");

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 1: Rewired GPU Ops — pairwise_l2_matrix_gpu
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ Rewire #1: pairwise_l2_matrix_gpu → PairwiseL2Gpu ═══");
    eprintln!("  OLD: O(n²) loop calling l2_distance_gpu per pair");
    eprintln!("  NEW: single PairwiseL2Gpu::dispatch (1 GPU dispatch)");
    eprintln!("  Provenance: neuralSpring metalForge pairwise_l2.wgsl → BarraCUDA");
    eprintln!("              neuralSpring absorbed from MODES (Exp-012) novelty search");
    eprintln!();

    let n_vecs = 200;
    let dim = 50;
    let mut rng = Rng::new(42);
    let data: Vec<f64> = (0..n_vecs * dim).map(|_| rng.next_f64() * 10.0).collect();

    let rewire1_us = bench(
        &format!("pairwise_l2_matrix_gpu (PairwiseL2Gpu) {n_vecs}×{dim}"),
        || {
            let _ = std::hint::black_box(pairwise_l2_matrix_gpu(&data, n_vecs, dim, &device));
        },
    );
    let result =
        pairwise_l2_matrix_gpu(&data, n_vecs, dim, &device).expect("pairwise_l2_matrix_gpu");
    let n_pairs = n_vecs * (n_vecs - 1) / 2;
    h.check_bool(
        &format!("pairwise_l2_matrix output length = {n_pairs}"),
        result.len() == n_pairs,
    );
    h.check_bool(
        "pairwise_l2_matrix values > 0",
        result.iter().all(|&v| v >= 0.0),
    );
    h.check_bool(
        &format!("pairwise_l2_matrix {rewire1_us:.0}µs (PairwiseL2Gpu)"),
        rewire1_us.is_finite(),
    );
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 2: Rewired GPU Ops — geographic_distance_matrix_gpu
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ Rewire #2: geographic_distance_matrix_gpu → PairwiseL2Gpu ═══");
    eprintln!("  OLD: O(n²) loop calling l2_distance_gpu per pair");
    eprintln!("  NEW: delegates to pairwise_l2_matrix_gpu (PairwiseL2Gpu), expands to full matrix");
    eprintln!("  Provenance: neuralSpring 025 MetaPop → PairwiseL2Gpu (BarraCUDA)");
    eprintln!();

    let n_coords = 100;
    let coords: Vec<(f64, f64)> = (0..n_coords)
        .map(|_| (rng.next_f64() * 100.0, rng.next_f64() * 100.0))
        .collect();

    let rewire2_us = bench(
        &format!("geographic_distance_matrix_gpu (PairwiseL2Gpu) {n_coords} coords"),
        || {
            let _ = std::hint::black_box(geographic_distance_matrix_gpu(&coords, &device));
        },
    );
    let geo_result =
        geographic_distance_matrix_gpu(&coords, &device).expect("geographic_distance_matrix_gpu");
    h.check_bool(
        &format!("geo_distance output = {n_coords}×{n_coords}"),
        geo_result.len() == n_coords * n_coords,
    );
    let symmetric = (0..n_coords).all(|i| {
        (0..n_coords).all(|j| {
            (geo_result[i * n_coords + j] - geo_result[j * n_coords + i]).abs()
                < tolerances::TENSOR_EXACT_F32
        })
    });
    h.check_bool("geo_distance symmetric", symmetric);
    h.check_bool(
        "geo_distance diagonal = 0",
        (0..n_coords).all(|i| geo_result[i * n_coords + i] == 0.0),
    );
    h.check_bool(
        &format!("geo_distance {rewire2_us:.0}µs (PairwiseL2Gpu)"),
        rewire2_us.is_finite(),
    );
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 3: Rewired GPU Ops — disorder_sweep_gpu IPR
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ Rewire #3: disorder_sweep_gpu CPU IPR → BatchIprGpu ═══");
    eprintln!("  OLD: CPU loop computing Σ|ψ|⁴ per eigenvector after eigensolve");
    eprintln!("  NEW: BatchIprGpu::dispatch on GPU after BatchedEighGpu");
    eprintln!("  Provenance: neuralSpring 022-023 Anderson → BatchIprGpu (BarraCUDA)");
    eprintln!("              Eigensolve: hotSpring Jacobi → BatchedEighGpu (BarraCUDA)");
    eprintln!();

    let n_dim = 16;
    let batch_size = 20;
    let mut hamiltonians = vec![0.0_f64; n_dim * n_dim * batch_size];
    for b in 0..batch_size {
        let base = b * n_dim * n_dim;
        for i in 0..n_dim {
            let w = rng.next_f64() * 4.0 - 2.0;
            hamiltonians[base + i * n_dim + i] = w;
            if i + 1 < n_dim {
                hamiltonians[base + i * n_dim + (i + 1)] = -1.0;
                hamiltonians[base + (i + 1) * n_dim + i] = -1.0;
            }
        }
    }

    let rewire3_us = bench(
        &format!("disorder_sweep_gpu (BatchIprGpu) {batch_size}×{n_dim}"),
        || {
            let _ = std::hint::black_box(disorder_sweep_gpu(
                &hamiltonians,
                n_dim,
                batch_size,
                &device,
            ));
        },
    );
    let ipr_result =
        disorder_sweep_gpu(&hamiltonians, n_dim, batch_size, &device).expect("disorder_sweep_gpu");
    h.check_bool(
        &format!("disorder_sweep output length = {batch_size}"),
        ipr_result.len() == batch_size,
    );
    h.check_bool(
        "disorder_sweep IPR values in (0, 1]",
        ipr_result
            .iter()
            .all(|&v| v > 0.0 && v <= 1.0 + tolerances::GPU_BATCH_IPR_F32),
    );
    h.check_bool(
        &format!("disorder_sweep {rewire3_us:.0}µs (BatchIprGpu)"),
        rewire3_us.is_finite(),
    );
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 4: Modern API — LogSumExp (hotSpring precision)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ Modern API: LogSumExp ═══");
    eprintln!("  Provenance: hotSpring log-domain HMM precision → BarraCUDA logsumexp_f64.wgsl");
    eprintln!("  Used by: HMM forward, softmax, log-likelihood computation");
    eprintln!("  Cross-spring: hotSpring needed log-domain stability for lattice QCD;");
    eprintln!(
        "                wetSpring needed it for HMM Baum-Welch; neuralSpring for log-likelihood"
    );
    eprintln!();

    let lse_n = 10_000_usize;
    let lse_data_f64: Vec<f64> = (0..lse_n).map(|_| rng.next_f64() * 20.0 - 10.0).collect();

    let lse_us = bench(&format!("LogSumExp GPU (f64) {lse_n}"), || {
        let t = Tensor::from_data_pod(&lse_data_f64, vec![lse_n], device.clone())
            .expect("LogSumExp tensor");
        let _ = std::hint::black_box(LogSumExp::new(t).execute().expect("LogSumExp execute"));
    });

    let t = Tensor::from_data_pod(&lse_data_f64, vec![lse_n], device.clone())
        .expect("LogSumExp tensor");
    let lse_result = LogSumExp::new(t).execute().expect("LogSumExp execute");
    h.check_bool(
        "LogSumExp f64 executes (correctness validated in validate_gpu_logsumexp)",
        true,
    );
    let _ = lse_result;
    h.check_bool(&format!("LogSumExp {lse_us:.0}µs"), lse_us.is_finite());
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 5: Modern API — PairwiseDistance (universal)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ Modern API: PairwiseDistance ═══");
    eprintln!("  Provenance: neuralSpring MODES pairwise_l2 → BarraCUDA pairwise_distance");
    eprintln!("  Universal distance metric: L1, L2, Lp norms");
    eprintln!("  Cross-spring: neuralSpring needed L2 for novelty search;");
    eprintln!("                wetSpring needed Hamming/Jaccard for genomics;");
    eprintln!("                BarraCUDA generalized to PairwiseDistance(p)");
    eprintln!();

    let pd_n = 5000_usize;
    let pd_dim = 32_usize;
    let pd_a: Vec<f32> = (0..pd_n * pd_dim).map(|_| rng.next_f64() as f32).collect();
    let pd_b: Vec<f32> = (0..pd_n * pd_dim).map(|_| rng.next_f64() as f32).collect();

    let pd_us = bench(&format!("PairwiseDistance L2 {pd_n}×{pd_dim}"), || {
        let t_a = Tensor::from_data(&pd_a, vec![pd_n, pd_dim], device.clone())
            .expect("PairwiseDistance tensor A");
        let t_b = Tensor::from_data(&pd_b, vec![pd_n, pd_dim], device.clone())
            .expect("PairwiseDistance tensor B");
        let op = PairwiseDistance::new(t_a, t_b, Some(2.0), None).expect("PairwiseDistance::new");
        let _ = std::hint::black_box(op.execute().expect("PairwiseDistance execute"));
    });
    h.check_bool(&format!("PairwiseDistance {pd_us:.0}µs"), pd_us.is_finite());
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 6: Modern API — BatchedEighGpu (hotSpring eigensolve)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ Modern API: BatchedEighGpu ═══");
    eprintln!("  Provenance: hotSpring HFB diagonalization → BarraCUDA Jacobi GPU sweeps");
    eprintln!("  neuralSpring uses it for Anderson localization, weight spectral analysis");
    eprintln!("  Cross-spring: hotSpring needed batched eigensolve for HFB nuclear structure;");
    eprintln!("                neuralSpring adopted for weight spectral analysis (Exp-040+)");
    eprintln!();

    let eigh_n = 24;
    let eigh_batch = 50;
    let mut eigh_data = vec![0.0_f64; eigh_n * eigh_n * eigh_batch];
    for b in 0..eigh_batch {
        let base = b * eigh_n * eigh_n;
        for i in 0..eigh_n {
            eigh_data[base + i * eigh_n + i] = rng.next_f64() * 6.0 - 3.0;
            if i + 1 < eigh_n {
                let off = rng.next_f64() * 0.5;
                eigh_data[base + i * eigh_n + (i + 1)] = off;
                eigh_data[base + (i + 1) * eigh_n + i] = off;
            }
        }
    }

    let eigh_us = bench(&format!("BatchedEighGpu {eigh_batch}×{eigh_n}"), || {
        let _ = std::hint::black_box(
            BatchedEighGpu::execute_single_dispatch(
                device.clone(),
                &eigh_data,
                eigh_n,
                eigh_batch,
                30,
                1e-12,
            )
            .expect("BatchedEighGpu execute"),
        );
    });

    let (evals, _evecs) = BatchedEighGpu::execute_single_dispatch(
        device.clone(),
        &eigh_data,
        eigh_n,
        eigh_batch,
        30,
        1e-12,
    )
    .expect("BatchedEighGpu execute");
    h.check_bool(
        &format!("BatchedEighGpu output: {} eigenvalues", evals.len()),
        evals.len() == eigh_n * eigh_batch,
    );

    let (cpu_evals, _) = eigh_gpu(&eigh_data[..eigh_n * eigh_n], eigh_n, &device)
        .expect("eigh_gpu single-batch reference");
    let mut batch0_evals: Vec<f64> = evals[..eigh_n].to_vec();
    let mut cpu_sorted = cpu_evals;
    batch0_evals.sort_by(f64::total_cmp);
    cpu_sorted.sort_by(f64::total_cmp);
    let eigh_ok = batch0_evals
        .iter()
        .zip(cpu_sorted.iter())
        .all(|(&a, &b)| (a - b).abs() < 0.05);
    h.check_bool("BatchedEighGpu batch[0] ≈ eigh_gpu single", eigh_ok);
    h.check_bool(
        &format!("BatchedEighGpu {eigh_us:.0}µs"),
        eigh_us.is_finite(),
    );
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 7: Modern API — BatchIprGpu (neuralSpring spectral)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ Modern API: BatchIprGpu ═══");
    eprintln!("  Provenance: neuralSpring Anderson localization 022-023 → BarraCUDA");
    eprintln!("  IPR = Σ|ψ_i|⁴ measures eigenvector localization");
    eprintln!("  Cross-spring: neuralSpring needed for disorder sweep;");
    eprintln!("                hotSpring uses for nuclear wavefunction localization");
    eprintln!();

    bench_ipr_gpu(&mut h, &device);
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 8: Cross-spring GPU diversity (wetSpring)
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ Modern API: DiversityFusionGpu ═══");
    eprintln!("  Provenance: wetSpring diversity.rs → diversity_fusion_f64.wgsl → BarraCUDA");
    eprintln!("  Fused Shannon + Simpson + Pielou in one GPU dispatch");
    eprintln!("  Cross-spring: wetSpring metagenomics → neuralSpring eco_dynamics;");
    eprintln!("                neuralSpring added GPU dispatch path (metalForge);");
    eprintln!("                BarraCUDA absorbed and f64-canonicalized");
    eprintln!();

    bench_diversity_fusion(&mut h, &device);
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 9: Dispatcher cross-spring f64 ops
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ Dispatcher: Cross-Spring f64 Precision Ops ═══");
    eprintln!("  Dispatcher auto-routes CPU→GPU based on size threshold");
    eprintln!("  f64 ops evolved from: hotSpring (Welford variance), wetSpring (correlation),");
    eprintln!("  neuralSpring (matmul), airSpring (regression) → BarraCUDA universal precision");
    eprintln!();

    bench_dispatcher_f64(&mut h);
    eprintln!();

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("═══ Cross-Spring Shader Provenance Summary ═══");
    eprintln!();
    eprintln!("  ┌─────────────┬──────────────────────────────────────────────────┐");
    eprintln!("  │ Spring      │ Key Contributions to BarraCUDA Shaders          │");
    eprintln!("  ├─────────────┼──────────────────────────────────────────────────┤");
    eprintln!("  │ hotSpring   │ DF64 core, eigensolve, Welford var, logsumexp   │");
    eprintln!("  │             │ → precision shaders, Jacobi sweeps, HFB         │");
    eprintln!("  ├─────────────┼──────────────────────────────────────────────────┤");
    eprintln!("  │ wetSpring   │ Shannon, Simpson, HMM, diversity fusion         │");
    eprintln!("  │             │ → bio shaders, Bray-Curtis, Felsenstein         │");
    eprintln!("  ├─────────────┼──────────────────────────────────────────────────┤");
    eprintln!("  │ neuralSpring│ Pairwise L2, IPR, batch fitness, swarm NN       │");
    eprintln!("  │             │ → ML shaders, novelty search, Anderson disorder │");
    eprintln!("  ├─────────────┼──────────────────────────────────────────────────┤");
    eprintln!("  │ airSpring   │ RMSE, R², NSE, fit_linear, moving_window        │");
    eprintln!("  │             │ → stats shaders, regression, ET₀                │");
    eprintln!("  ├─────────────┼──────────────────────────────────────────────────┤");
    eprintln!("  │ groundSpring│ multinomial sampling, MC propagation, ET₀       │");
    eprintln!("  │             │ → stochastic shaders, bootstrap                 │");
    eprintln!("  └─────────────┴──────────────────────────────────────────────────┘");
    eprintln!();
    eprintln!("  844+ f64-canonical WGSL shaders in BarraCUDA (ToadStool S87, 2dc26792)");
    eprintln!("  All springs benefit via path-dependent evolution:");
    eprintln!("    hotSpring precision → neuralSpring eigensolve");
    eprintln!("    wetSpring bio → neuralSpring eco_dynamics");
    eprintln!("    neuralSpring ML → hotSpring batch fitness");
    eprintln!("  Each spring pushes domain expertise → BarraCUDA absorbs → all benefit");
    eprintln!();

    h.finish();
}

fn bench_ipr_gpu(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    use wgpu::util::DeviceExt;

    let dim = 128_u32;
    let n_vectors = 500_u32;
    let mut rng = Rng::new(99);
    let ev: Vec<f32> = (0..dim * n_vectors)
        .map(|_| rng.next_f64() as f32)
        .collect();

    let d = device.device();
    let ev_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bench_ev"),
        contents: bytemuck::cast_slice(&ev),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let op = BatchIprGpu::new(device.clone());

    let ipr_us = bench(&format!("BatchIprGpu {n_vectors}×{dim}"), || {
        let ipr_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench_ipr"),
            size: u64::from(n_vectors) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        op.dispatch(&ev_buf, &ipr_buf, dim, n_vectors);

        let staging = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(n_vectors) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&ipr_buf, 0, &staging, 0, u64::from(n_vectors) * 4);
        device.queue().submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let view = slice.get_mapped_range();
        let _: Vec<f32> = std::hint::black_box(bytemuck::cast_slice(&view).to_vec());
    });

    h.check_bool(&format!("BatchIprGpu {ipr_us:.0}µs"), ipr_us.is_finite());
}

fn bench_diversity_fusion(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n_samples = 64;
    let n_species = 200;
    let mut rng = Rng::new(400);
    let abundances: Vec<f64> = (0..n_samples * n_species)
        .map(|_| (rng.next_f64() * 50.0).max(0.0))
        .collect();

    let rt = tokio::runtime::Runtime::new()
        .expect("tokio runtime creation failed — required for async benchmark");

    let cpu_us = bench("diversity_fusion_cpu (wetSpring→BarraCUDA)", || {
        std::hint::black_box(barracuda::ops::bio::diversity_fusion_cpu(
            &abundances,
            n_species,
        ));
    });

    let gpu_us = bench("DiversityFusionGpu (wetSpring→BarraCUDA→GPU)", || {
        rt.block_on(async {
            let op = DiversityFusionGpu::new(device.clone()).expect("DiversityFusionGpu");
            let _ = op
                .compute(&abundances, n_samples, n_species)
                .expect("DiversityFusionGpu compute failed — check GPU memory and input shape");
        });
    });

    if gpu_us > 0.0 && cpu_us > 0.0 {
        let ratio = cpu_us / gpu_us;
        eprintln!("    → GPU/CPU ratio: {ratio:.2}×");
    }

    h.check_bool(
        &format!("DiversityFusion CPU {cpu_us:.0}µs"),
        cpu_us.is_finite(),
    );
    h.check_bool(
        &format!("DiversityFusion GPU {gpu_us:.0}µs"),
        gpu_us.is_finite(),
    );
}

fn bench_dispatcher_f64(h: &mut ValidationHarness) {
    use neural_spring::gpu_dispatch::Dispatcher;

    let rt = tokio::runtime::Runtime::new()
        .expect("tokio runtime creation failed — required for async benchmark");
    let dispatcher = rt.block_on(async { Dispatcher::new().await });

    let n = 50_000_usize;
    let mut rng = Rng::new(700);
    let big_a: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0 - 5.0).collect();
    let big_b: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0 - 5.0).collect();

    let var_us = bench("Dispatcher::variance 50k (hotSpring Welford→GPU)", || {
        let _ = std::hint::black_box(dispatcher.variance(&big_a));
    });
    h.check_bool(
        &format!("Dispatcher variance {var_us:.0}µs"),
        var_us.is_finite(),
    );

    let pearson_us = bench(
        "Dispatcher::pearson 50k (wetSpring+hotSpring→GPU)",
        || {
            let _ = std::hint::black_box(dispatcher.pearson_correlation(&big_a, &big_b));
        },
    );
    h.check_bool(
        &format!("Dispatcher pearson {pearson_us:.0}µs"),
        pearson_us.is_finite(),
    );

    let probs: Vec<f64> = big_a.iter().map(|x| x.abs() / 1000.0 + 1e-10).collect();
    let shannon_us = bench("Dispatcher::shannon 50k (wetSpring fused→GPU)", || {
        let _ = std::hint::black_box(dispatcher.shannon_entropy(&probs));
    });
    h.check_bool(
        &format!("Dispatcher shannon {shannon_us:.0}µs"),
        shannon_us.is_finite(),
    );

    let side = 200_usize;
    let mat: Vec<f64> = (0..side * side).map(|_| rng.next_f64()).collect();
    let matmul_us = bench("Dispatcher::mat_mul 200×200 (neuralSpring→GPU)", || {
        let _ = std::hint::black_box(dispatcher.mat_mul(&mat, &mat, side));
    });
    h.check_bool(
        &format!("Dispatcher matmul {matmul_us:.0}µs"),
        matmul_us.is_finite(),
    );
}
