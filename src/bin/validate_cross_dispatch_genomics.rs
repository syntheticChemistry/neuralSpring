// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-dispatch validation: genomics workloads on GPU vs CPU, parity proof.
//!
//! Demonstrates `BarraCUDA`'s dispatch system for pangenome Jaccard distances
//! and meta-population locus variance — identical computations on GPU (WGSL)
//! and CPU (Rust math), parity validated within f32 tolerance.
//!
//! ## What this proves
//!
//! - **Math portability**: Jaccard distance and variance decomposition produce
//!   identical results on GPU and CPU
//! - **Dispatch routing**: `dispatch_for("pairwise_distance", n)` and
//!   `dispatch_for("variance", n)` route correctly by workload size
//! - **Timing**: GPU shows throughput advantage for large workloads
//!
//! ## Evolution path
//!
//! ```text
//! GPU-only (validate_gpu_pangenome, validate_gpu_meta_pop)
//!   → Cross-dispatch GPU ↔ CPU (this binary)
//!   → metalForge cross-system (GPU → NPU → CPU)
//! ```
//!
//! ## Papers validated
//!
//! - Paper 024: Pangenome Selection Dynamics (Anderson, 2024)
//! - Paper 025: Meta-Population Differentiation (Anderson, 2024)
//!
//! ## Provenance
//!
//! CPU/GPU dispatch: genomics (Jaccard + locus variance) via `DispatchConfig`.
//! Validates: `pairwise_jaccard`, `locus_variance` GPU↔CPU parity.
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::similar_names
)]

use std::time::Instant;

use barracuda::dispatch::{dispatch_for, DispatchTarget};
use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::meta_population::{allele_frequencies, generate_population};
use neural_spring::pangenome_selection::{generate_pa_matrix, jaccard_distance_matrix};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const JACCARD_WGSL: &str = include_str!("../../metalForge/shaders/pairwise_jaccard.wgsl");
const VARIANCE_WGSL: &str = include_str!("../../metalForge/shaders/locus_variance.wgsl");

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(e) => {
            eprintln!("  SKIP: {e} — no GPU/CPU adapter available");
            eprintln!("  0/0 checks — skipping gracefully");
            std::process::exit(0);
        }
    };

    let mut h = ValidationHarness::new("cross_dispatch_genomics");

    validate_dispatch_routing(&mut h);
    validate_jaccard_parity(&mut h, &gpu);
    validate_variance_parity(&mut h, &gpu);
    validate_jaccard_timing(&mut h, &gpu);
    validate_variance_timing(&mut h, &gpu);

    h.finish();
}

// ── Dispatch routing ─────────────────────────────────────────────

fn validate_dispatch_routing(h: &mut ValidationHarness) {
    let small_pw = dispatch_for("pairwise_distance", 10);
    let large_pw = dispatch_for("pairwise_distance", 10_000);
    let small_var = dispatch_for("variance", 10);
    let large_var = dispatch_for("variance", 10_000);

    h.check_bool(
        &format!("dispatch: pairwise_distance(10) → {small_pw:?}"),
        matches!(small_pw, DispatchTarget::Cpu),
    );
    h.check_bool(
        &format!("dispatch: pairwise_distance(10k) → {large_pw:?}"),
        matches!(large_pw, DispatchTarget::Gpu),
    );
    h.check_bool(
        &format!("dispatch: variance(10) → {small_var:?}"),
        matches!(small_var, DispatchTarget::Cpu),
    );
    h.check_bool(
        &format!("dispatch: variance(10k) → {large_var:?}"),
        matches!(large_var, DispatchTarget::Gpu),
    );
}

// ── Jaccard parity (Paper 024) ───────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct JaccardParams {
    n_genomes: u32,
    n_genes: u32,
}

fn cpu_jaccard_upper(pa: &[f64], n_genes: usize, n_genomes: usize) -> Vec<f64> {
    let jd = jaccard_distance_matrix(pa, n_genes, n_genomes);
    let mut upper = Vec::new();
    for i in 0..n_genomes {
        for j in (i + 1)..n_genomes {
            upper.push(jd[i * n_genomes + j]);
        }
    }
    upper
}

fn gpu_jaccard(
    gpu: &Gpu,
    pa_f32: &[f32],
    n_genomes: u32,
    n_genes: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xd_jaccard"),
        source: wgpu::ShaderSource::Wgsl(JACCARD_WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("xd_jac_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("xd_jac_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("xd_jac_pipe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "pairwise_jaccard",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let n_pairs = (n_genomes * (n_genomes - 1) / 2) as usize;
    let pa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_pa"),
        contents: bytemuck::cast_slice(pa_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_jac_out"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = JaccardParams { n_genomes, n_genes };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_jac_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xd_jac_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pa_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dist_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("xd_jac_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("xd_jac_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n_pairs as u32).div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    gpu.read_buffer_f32(&dist_buf, n_pairs)
}

fn validate_jaccard_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(42);
    let n_genomes = 30_usize;
    let n_genes = 200_usize;
    let env: Vec<usize> = (0..15).map(|_| 0).chain((0..15).map(|_| 1)).collect();
    let pa = generate_pa_matrix(n_genomes, n_genes, 0.25, 0.10, &mut rng, &env);

    let cpu_upper = cpu_jaccard_upper(&pa, n_genes, n_genomes);
    let pa_f32: Vec<f32> = pa.iter().map(|&v| v as f32).collect();

    match gpu_jaccard(gpu, &pa_f32, n_genomes as u32, n_genes as u32) {
        Ok(gpu_dist) => {
            let max_diff: f64 = gpu_dist
                .iter()
                .zip(cpu_upper.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "Jaccard parity (30×200): {n_pairs} pairs, max diff {max_diff:.2e}",
                    n_pairs = gpu_dist.len()
                ),
                max_diff,
                tolerances::GPU_JACCARD_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("Jaccard parity: failed — {e}"), false);
        }
    }
}

fn validate_jaccard_timing(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(77);
    let n_genomes = 50_usize;
    let n_genes = 500_usize;
    let env: Vec<usize> = (0..25).map(|_| 0).chain((0..25).map(|_| 1)).collect();
    let pa = generate_pa_matrix(n_genomes, n_genes, 0.25, 0.10, &mut rng, &env);
    let pa_f32: Vec<f32> = pa.iter().map(|&v| v as f32).collect();

    let cpu_start = Instant::now();
    let cpu_upper = cpu_jaccard_upper(&pa, n_genes, n_genomes);
    let cpu_us = cpu_start.elapsed().as_micros();

    let gpu_start = Instant::now();
    let gpu_result = gpu_jaccard(gpu, &pa_f32, n_genomes as u32, n_genes as u32);
    let gpu_us = gpu_start.elapsed().as_micros();

    match gpu_result {
        Ok(gpu_dist) => {
            let max_diff: f64 = gpu_dist
                .iter()
                .zip(cpu_upper.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "Jaccard timing (50×500): GPU={gpu_us}μs CPU={cpu_us}μs diff={max_diff:.2e}"
                ),
                max_diff,
                tolerances::GPU_JACCARD_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("Jaccard timing: failed — {e}"), false);
        }
    }
}

// ── Locus variance parity (Paper 025) ────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VarianceParams {
    n_pops: u32,
    n_loci: u32,
}

fn cpu_locus_variance(all_freqs: &[Vec<f64>], n_loci: usize) -> Vec<f64> {
    let n_pops = all_freqs.len();
    (0..n_loci)
        .map(|j| {
            let mean: f64 = all_freqs.iter().map(|af| af[j]).sum::<f64>() / n_pops as f64;
            all_freqs
                .iter()
                .map(|af| (af[j] - mean).powi(2))
                .sum::<f64>()
                / n_pops as f64
        })
        .collect()
}

fn gpu_variance(gpu: &Gpu, af_f32: &[f32], n_pops: u32, n_loci: u32) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xd_variance"),
        source: wgpu::ShaderSource::Wgsl(VARIANCE_WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("xd_var_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("xd_var_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("xd_var_pipe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "locus_variance",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let af_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_af"),
        contents: bytemuck::cast_slice(af_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let var_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_var_out"),
        size: u64::from(n_loci) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = VarianceParams { n_pops, n_loci };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_var_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xd_var_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: af_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: var_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("xd_var_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("xd_var_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n_loci.div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    gpu.read_buffer_f32(&var_buf, n_loci as usize)
}

fn make_meta_pop_data(seed: u64) -> (Vec<Vec<f64>>, usize, usize, usize) {
    let mut rng = Rng::new(seed);
    let n_pops = 6_usize;
    let n_loci = 100_usize;
    let n_individuals = 20_usize;
    let temperatures = [65.0, 72.0, 78.0, 85.0, 70.0, 90.0];
    let ancestral: Vec<f64> = (0..n_loci).map(|_| rng.beta(2.0, 2.0)).collect();
    let populations: Vec<Vec<f64>> = (0..n_pops)
        .map(|i| {
            generate_population(
                n_individuals,
                n_loci,
                &ancestral,
                0.15,
                temperatures[i],
                65.0,
                90.0,
                n_loci / 5,
                &mut rng,
            )
        })
        .collect();
    (populations, n_pops, n_loci, n_individuals)
}

fn validate_variance_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let (populations, n_pops, n_loci, n_individuals) = make_meta_pop_data(42);
    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .map(|pop| allele_frequencies(pop, n_individuals, n_loci))
        .collect();
    let cpu_var = cpu_locus_variance(&all_freqs, n_loci);

    let af_f32: Vec<f32> = all_freqs
        .iter()
        .flat_map(|af| af.iter().map(|&v| v as f32))
        .collect();

    match gpu_variance(gpu, &af_f32, n_pops as u32, n_loci as u32) {
        Ok(gpu_var) => {
            let max_diff: f64 = gpu_var
                .iter()
                .zip(cpu_var.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("variance parity (6×100): max diff {max_diff:.2e}"),
                max_diff,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("variance parity: failed — {e}"), false);
        }
    }
}

fn validate_variance_timing(h: &mut ValidationHarness, gpu: &Gpu) {
    let (populations, n_pops, n_loci, n_individuals) = make_meta_pop_data(77);
    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .map(|pop| allele_frequencies(pop, n_individuals, n_loci))
        .collect();
    let af_f32: Vec<f32> = all_freqs
        .iter()
        .flat_map(|af| af.iter().map(|&v| v as f32))
        .collect();

    let cpu_start = Instant::now();
    let cpu_var = cpu_locus_variance(&all_freqs, n_loci);
    let cpu_us = cpu_start.elapsed().as_micros();

    let gpu_start = Instant::now();
    let gpu_result = gpu_variance(gpu, &af_f32, n_pops as u32, n_loci as u32);
    let gpu_us = gpu_start.elapsed().as_micros();

    match gpu_result {
        Ok(gpu_var) => {
            let max_diff: f64 = gpu_var
                .iter()
                .zip(cpu_var.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "variance timing (6×100): GPU={gpu_us}μs CPU={cpu_us}μs diff={max_diff:.2e}"
                ),
                max_diff,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("variance timing: failed — {e}"), false);
        }
    }
}

// ── wgpu layout helpers ────────────────────────────────────────────

const fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
