// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline: locus_variance → `mean_reduce` → scalar readback (Paper 025).
//!
//! Chains two shader stages in a single `CommandEncoder` with zero CPU round-trips.
//! Stage 1: `locus_variance` — per-locus allele frequency variance across populations.
//! Stage 2: `mean_reduce` — variance array to scalar mean.
//!
//! ## Pipeline
//!
//! ```text
//! Upload allele frequencies (once)
//!   ↓
//! ┌─────────────────────────────────────────────────────┐
//! │  Stage 1: locus_variance.wgsl                        │
//! │    allele_freqs → per_locus_var[n_loci]              │
//! │                                                     │
//! │  Stage 2: mean_reduce.wgsl                           │
//! │    per_locus_var[] → mean_variance (scalar)           │
//! └─────────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — NO CPU round-trip)
//! Readback: 4 bytes (one f32 scalar)
//! ```
//!
//! ## Provenance
//!
//! GPU pipeline: locus_variance → mean_reduce.
//! Validates: meta-population mean per-locus variance (Campbell, Anderson et al., 2017).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::needless_range_loop
)]

use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const VARIANCE_WGSL: &str = include_str!("../../metalForge/shaders/locus_variance.wgsl");
const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VarianceParams {
    n_pops: u32,
    n_loci: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ReduceParams {
    n: u32,
}

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

    let mut h = ValidationHarness::new("gpu_pipeline_meta_pop");

    validate_meta_pop_small(&mut h, &gpu);
    validate_meta_pop_larger(&mut h, &gpu);
    validate_meta_pop_uniform(&mut h, &gpu);
    validate_meta_pop_differentiated(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_locus_variance(allele_freqs: &[f32], n_pops: usize, n_loci: usize) -> f32 {
    let mut total = 0.0_f32;
    for locus in 0..n_loci {
        let mut sum = 0.0_f32;
        for pop in 0..n_pops {
            sum += allele_freqs[pop * n_loci + locus];
        }
        let mean = sum / n_pops as f32;
        let mut var_sum = 0.0_f32;
        for pop in 0..n_pops {
            let diff = allele_freqs[pop * n_loci + locus] - mean;
            var_sum = diff.mul_add(diff, var_sum);
        }
        total += var_sum / n_pops as f32;
    }
    total / n_loci as f32
}

// ── GPU chained pipeline ───────────────────────────────────────────

fn gpu_mean_locus_variance(
    gpu: &Gpu,
    allele_freqs: &[f32],
    n_pops: u32,
    n_loci: u32,
) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let variance_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_variance"),
        source: wgpu::ShaderSource::Wgsl(VARIANCE_WGSL.into()),
    });

    let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_meta_reduce"),
        source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
    });

    let variance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_variance_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });

    let variance_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_variance_pl"),
        bind_group_layouts: &[&variance_bgl],
        push_constant_ranges: &[],
    });

    let variance_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_variance_pipeline"),
        layout: Some(&variance_pl),
        module: &variance_shader,
        entry_point: "locus_variance",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_meta_reduce_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });

    let reduce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_meta_reduce_pl"),
        bind_group_layouts: &[&reduce_bgl],
        push_constant_ranges: &[],
    });

    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_meta_reduce_pipeline"),
        layout: Some(&reduce_pl),
        module: &reduce_shader,
        entry_point: "mean_reduce",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let af_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_meta_allele_freqs"),
        contents: bytemuck::cast_slice(allele_freqs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let var_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_meta_variances"),
        size: u64::from(n_loci) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let variance_params = VarianceParams {
        n_pops,
        n_loci,
        _pad0: 0,
        _pad1: 0,
    };
    let variance_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_meta_variance_params"),
        contents: bytemuck::bytes_of(&variance_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_meta_mean_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let reduce_params = ReduceParams { n: n_loci };
    let reduce_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_meta_reduce_params"),
        contents: bytemuck::bytes_of(&reduce_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let variance_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_variance_bg"),
        layout: &variance_bgl,
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
                resource: variance_params_buf.as_entire_binding(),
            },
        ],
    });

    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_meta_reduce_bg"),
        layout: &reduce_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: var_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: reduce_params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("chain_meta_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_variance_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&variance_pipeline);
        pass.set_bind_group(0, &variance_bg, &[]);
        pass.dispatch_workgroups(n_loci.div_ceil(256), 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_meta_reduce_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&reduce_pipeline);
        pass.set_bind_group(0, &reduce_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));

    let result = gpu.read_buffer_f32(&result_buf, 1)?;
    Ok(result[0])
}

// ── Validation functions ───────────────────────────────────────────

fn validate_meta_pop_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 4_usize;
    let n_loci = 12_usize;
    let mut rng = Rng::new(42);
    let allele_freqs: Vec<f32> = (0..n_pops * n_loci).map(|_| rng.uniform() as f32).collect();

    let cpu_mean = cpu_mean_locus_variance(&allele_freqs, n_pops, n_loci);

    match gpu_mean_locus_variance(gpu, &allele_freqs, n_pops as u32, n_loci as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("meta_pop small 4×12: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("meta_pop small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_meta_pop_larger(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 8_usize;
    let n_loci = 32_usize;
    let mut rng = Rng::new(777);
    let allele_freqs: Vec<f32> = (0..n_pops * n_loci).map(|_| rng.uniform() as f32).collect();

    let cpu_mean = cpu_mean_locus_variance(&allele_freqs, n_pops, n_loci);

    match gpu_mean_locus_variance(gpu, &allele_freqs, n_pops as u32, n_loci as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("meta_pop larger 8×32: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("meta_pop larger: dispatch failed — {e}"), false);
        }
    }
}

fn validate_meta_pop_uniform(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 4_usize;
    let n_loci = 8_usize;
    let allele_freqs: Vec<f32> = vec![0.5; n_pops * n_loci];

    match gpu_mean_locus_variance(gpu, &allele_freqs, n_pops as u32, n_loci as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("meta_pop uniform: mean variance={gpu_mean:.6} vs 0"),
                f64::from(gpu_mean),
                0.0,
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("meta_pop uniform: dispatch failed — {e}"), false);
        }
    }
}

fn validate_meta_pop_differentiated(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 4_usize;
    let n_loci = 6_usize;
    let mut allele_freqs = vec![0.0_f32; n_pops * n_loci];
    for pop in 0..n_pops {
        for locus in 0..n_loci {
            allele_freqs[pop * n_loci + locus] = (pop as f32)
                .mul_add(0.2, locus as f32 * 0.05)
                .clamp(0.01, 0.99);
        }
    }

    let cpu_mean = cpu_mean_locus_variance(&allele_freqs, n_pops, n_loci);

    match gpu_mean_locus_variance(gpu, &allele_freqs, n_pops as u32, n_loci as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("meta_pop differentiated: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_LOCUS_VARIANCE_F32,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("meta_pop differentiated: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 6_usize;
    let n_loci = 16_usize;
    let mut rng = Rng::new(99);
    let allele_freqs: Vec<f32> = (0..n_pops * n_loci).map(|_| rng.uniform() as f32).collect();

    let r1 = gpu_mean_locus_variance(gpu, &allele_freqs, n_pops as u32, n_loci as u32);
    let r2 = gpu_mean_locus_variance(gpu, &allele_freqs, n_pops as u32, n_loci as u32);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("meta_pop determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("meta_pop determinism: dispatch failed", false);
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
