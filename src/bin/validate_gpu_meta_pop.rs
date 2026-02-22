// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: per-locus allele frequency variance via `metalForge` WGSL shader.
//!
//! Validates `metalForge/shaders/locus_variance.wgsl` against CPU
//! variance computation from `meta_population.rs`.  The GPU shader
//! computes per-locus variance in a single dispatch (one thread per locus).
//!
//! Evolution path:
//! ```text
//! Python (numpy.var) → Rust CPU (loop) → BarraCUDA CPU (stats::variance)
//!   → GPU WGSL shader (locus_variance.wgsl) → ToadStool absorption
//! ```
//!
//! ## Papers validated
//!
//! - Paper 025: Meta-Population Differentiation (Anderson, 2024)
//!
//! ## Provenance
//!
//! CPU reference: `meta_population::inter_population_af_variance` (per-locus variance).
//! WGSL shader: `metalForge/shaders/locus_variance.wgsl`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::similar_names
)]

use barracuda::ops::bio::LocusVarianceGpu;
use neural_spring::gpu::Gpu;
use neural_spring::meta_population::{allele_frequencies, generate_population};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("../../metalForge/shaders/locus_variance.wgsl");

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

    let mut h = ValidationHarness::new("gpu_meta_pop");

    validate_small_variance(&mut h, &gpu);
    validate_larger_variance(&mut h, &gpu);
    validate_uniform_pops(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);
    validate_upstream_parity(&mut h, &gpu);

    h.finish();
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VarianceParams {
    n_pops: u32,
    n_loci: u32,
}

fn gpu_locus_variance(
    gpu: &Gpu,
    allele_freqs: &[f32],
    n_pops: u32,
    n_loci: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("locus_variance"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("variance_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("variance_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("variance_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "locus_variance",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let af_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("allele_freqs"),
        contents: bytemuck::cast_slice(allele_freqs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let var_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("per_locus_var"),
        size: u64::from(n_loci) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = VarianceParams { n_pops, n_loci };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("variance_bg"),
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
        label: Some("variance_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("variance_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n_loci.div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&var_buf, n_loci as usize)
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

fn make_test_data(seed: u64) -> (Vec<Vec<f64>>, usize, usize, usize) {
    let mut rng = Rng::new(seed);
    let n_pops = 6_usize;
    let n_loci = 100_usize;
    let n_individuals = 20_usize;
    let fst_target = 0.15;
    let temperatures = [65.0, 72.0, 78.0, 85.0, 70.0, 90.0];
    let temp_min = 65.0;
    let temp_max = 90.0;
    let n_thermal = n_loci / 5;

    let ancestral: Vec<f64> = (0..n_loci).map(|_| rng.beta(2.0, 2.0)).collect();
    let populations: Vec<Vec<f64>> = (0..n_pops)
        .map(|i| {
            generate_population(
                n_individuals,
                n_loci,
                &ancestral,
                fst_target,
                temperatures[i],
                temp_min,
                temp_max,
                n_thermal,
                &mut rng,
            )
        })
        .collect();
    (populations, n_pops, n_loci, n_individuals)
}

fn validate_small_variance(h: &mut ValidationHarness, gpu: &Gpu) {
    let (populations, n_pops, n_loci, n_individuals) = make_test_data(42);

    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .map(|pop| allele_frequencies(pop, n_individuals, n_loci))
        .collect();
    let cpu_var = cpu_locus_variance(&all_freqs, n_loci);

    // Flatten to row-major f32: af[pop * n_loci + locus]
    let af_f32: Vec<f32> = all_freqs
        .iter()
        .flat_map(|af| af.iter().map(|&v| v as f32))
        .collect();

    match gpu_locus_variance(gpu, &af_f32, n_pops as u32, n_loci as u32) {
        Ok(gpu_var) => {
            h.check_bool(
                &format!("6×100: correct count ({})", gpu_var.len()),
                gpu_var.len() == n_loci,
            );

            let max_diff: f64 = gpu_var
                .iter()
                .zip(cpu_var.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("6×100: max GPU-CPU var diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );

            let gpu_mean: f64 =
                gpu_var.iter().map(|&v| f64::from(v)).sum::<f64>() / gpu_var.len() as f64;
            h.check_lower(
                &format!("6×100: mean locus variance > 0 ({gpu_mean:.6})"),
                gpu_mean,
                0.0,
            );
        }
        Err(e) => {
            h.check_bool(&format!("6×100: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger_variance(h: &mut ValidationHarness, gpu: &Gpu) {
    let (populations, n_pops, n_loci, n_individuals) = make_test_data(77);

    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .map(|pop| allele_frequencies(pop, n_individuals, n_loci))
        .collect();
    let cpu_var = cpu_locus_variance(&all_freqs, n_loci);

    let af_f32: Vec<f32> = all_freqs
        .iter()
        .flat_map(|af| af.iter().map(|&v| v as f32))
        .collect();

    match gpu_locus_variance(gpu, &af_f32, n_pops as u32, n_loci as u32) {
        Ok(gpu_var) => {
            let max_diff: f64 = gpu_var
                .iter()
                .zip(cpu_var.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("seed=77: max GPU-CPU var diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );

            let cpu_mean: f64 = cpu_var.iter().sum::<f64>() / cpu_var.len() as f64;
            let gpu_mean: f64 =
                gpu_var.iter().map(|&v| f64::from(v)).sum::<f64>() / gpu_var.len() as f64;
            h.check_abs(
                &format!("seed=77: mean var GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("seed=77: dispatch failed — {e}"), false);
        }
    }
}

fn validate_uniform_pops(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 4_u32;
    let n_loci = 16_u32;
    let af_f32: Vec<f32> = vec![0.5; (n_pops * n_loci) as usize];

    match gpu_locus_variance(gpu, &af_f32, n_pops, n_loci) {
        Ok(gpu_var) => {
            let all_zero = gpu_var
                .iter()
                .all(|&v| v.abs() < tolerances::GPU_LOCUS_VARIANCE_F32 as f32);
            h.check_bool(
                &format!(
                    "uniform AF=0.5: all variance≈0 (max={:.2e})",
                    gpu_var.iter().map(|v| v.abs()).fold(0.0_f32, f32::max)
                ),
                all_zero,
            );
        }
        Err(e) => {
            h.check_bool(&format!("uniform: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let (populations, n_pops, n_loci, n_individuals) = make_test_data(42);

    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .map(|pop| allele_frequencies(pop, n_individuals, n_loci))
        .collect();
    let af_f32: Vec<f32> = all_freqs
        .iter()
        .flat_map(|af| af.iter().map(|&v| v as f32))
        .collect();

    let run1 = gpu_locus_variance(gpu, &af_f32, n_pops as u32, n_loci as u32);
    let run2 = gpu_locus_variance(gpu, &af_f32, n_pops as u32, n_loci as u32);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(&a, &b)| (a - b).abs() < f32::EPSILON);
            h.check_bool("determinism: two variance runs identical", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}

fn validate_upstream_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let (populations, n_pops, n_loci, n_individuals) = make_test_data(42);
    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .map(|pop| allele_frequencies(pop, n_individuals, n_loci))
        .collect();
    let af_f32: Vec<f32> = all_freqs
        .iter()
        .flat_map(|af| af.iter().map(|&v| v as f32))
        .collect();

    let local = gpu_locus_variance(gpu, &af_f32, n_pops as u32, n_loci as u32);

    let dev = Arc::clone(gpu.wgpu_device());
    let device = gpu.device();
    let op = LocusVarianceGpu::new(dev);
    let freq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("freqs"), contents: bytemuck::cast_slice(&af_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let var_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("var"), size: (n_loci * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    op.dispatch(&freq_buf, &var_buf, n_pops as u32, n_loci as u32);
    let upstream = gpu.read_buffer_f32(&var_buf, n_loci);

    match (local, upstream) {
        (Ok(l), Ok(u)) => {
            let max_diff: f64 = l.iter().zip(u.iter())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("upstream parity: local vs LocusVarianceGpu diff {max_diff:.2e}"),
                max_diff, tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        _ => h.check_bool("upstream parity: dispatch failed", false),
    }
}

const fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
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
