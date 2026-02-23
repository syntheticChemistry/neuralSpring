// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline: Wright-Fisher drift → `mean_reduce` → scalar readback.
//!
//! Chains two shader stages in a single `CommandEncoder` with zero CPU round-trips.
//! Stage 1: `wright_fisher_step.wgsl` — one generation of drift + selection.
//! Stage 2: `mean_reduce.wgsl` — frequency array to scalar mean.
//!
//! ## Pipeline
//!
//! ```text
//! Upload freq_in + selection + PRNG state (once)
//!   ↓
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Stage 1: wright_fisher_step.wgsl                               │
//! │    freq_in × selection → freq_out\[n_pops × n_loci\]           │
//! │                                                                 │
//! │  Stage 2: mean_reduce.wgsl                                      │
//! │    freq_out → mean_frequency (scalar)                           │
//! └─────────────────────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — NO CPU round-trip)
//! Readback: 4 bytes (one f32 scalar)
//! ```
//!
//! ## Papers validated
//!
//! - Paper 024: Pangenome Selection (Moulana, Anderson et al., 2020)
//! - Paper 025: Meta-Population Dynamics (Campbell, Anderson et al., 2017)
//!
//! ## Provenance
//!
//! GPU pipeline: `wright_fisher_step` → `mean_reduce`.
//! Validates: end-to-end GPU-resident stochastic population genetics.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const WF_WGSL: &str = include_str!("../../metalForge/shaders/wright_fisher_step.wgsl");
const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WfParams {
    n_pops: u32,
    n_loci: u32,
    two_n: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ReduceParams {
    n: u32,
}

const fn splitmix32(state: &mut u32) -> u32 {
    *state = state.wrapping_add(0x9e37_79b9);
    let mut z = *state;
    z = (z ^ (z >> 15)).wrapping_mul(0x85eb_ca6b);
    z = (z ^ (z >> 13)).wrapping_mul(0xc2b2_ae35);
    z ^ (z >> 16)
}

fn seed_prng(n_threads: usize, base_seed: u32) -> Vec<u32> {
    let mut result = Vec::with_capacity(n_threads * 4);
    for t in 0..n_threads {
        let mut sm = base_seed.wrapping_add(t as u32 * 1_000_003);
        for _ in 0..4 {
            result.push(splitmix32(&mut sm));
        }
    }
    result
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

    let mut h = ValidationHarness::new("gpu_pipeline_wright_fisher");

    validate_neutral_mean(&mut h, &gpu);
    validate_selection_shifts_mean(&mut h, &gpu);
    validate_boundary_frequencies(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

/// Chains `wright_fisher_step` → `mean_reduce` in a single encoder submission.
fn gpu_wf_mean(
    gpu: &Gpu,
    freq_in: &[f32],
    selection: &[f32],
    prng_state: &[u32],
    n_pops: u32,
    n_loci: u32,
    two_n: u32,
) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();
    let n_total = n_pops * n_loci;

    let wf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pipe_wf"),
        source: wgpu::ShaderSource::Wgsl(WF_WGSL.into()),
    });
    let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pipe_reduce"),
        source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
    });

    // Stage 1 layout: freq_in(ro), selection(ro), freq_out(rw), prng_state(rw), params(uniform)
    let wf_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pipe_wf_bgl"),
        entries: &[
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            storage_rw_entry(3),
            uniform_entry(4),
        ],
    });
    let wf_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipe_wf_pl"),
        bind_group_layouts: &[&wf_bgl],
        push_constant_ranges: &[],
    });
    let wf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipe_wf_pipeline"),
        layout: Some(&wf_pl),
        module: &wf_shader,
        entry_point: "wright_fisher",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Stage 2 layout: values(ro), result(rw), params(uniform)
    let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pipe_reduce_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });
    let reduce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipe_reduce_pl"),
        bind_group_layouts: &[&reduce_bgl],
        push_constant_ranges: &[],
    });
    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipe_reduce_pipeline"),
        layout: Some(&reduce_pl),
        module: &reduce_shader,
        entry_point: "mean_reduce",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Buffers
    let freq_in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pipe_freq_in"),
        contents: bytemuck::cast_slice(freq_in),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let selection_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pipe_selection"),
        contents: bytemuck::cast_slice(selection),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let freq_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pipe_freq_out"),
        size: u64::from(n_total) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let prng_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pipe_prng"),
        contents: bytemuck::cast_slice(prng_state),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let wf_params = WfParams {
        n_pops,
        n_loci,
        two_n,
        _pad: 0,
    };
    let wf_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pipe_wf_params"),
        contents: bytemuck::bytes_of(&wf_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pipe_mean_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let reduce_params = ReduceParams { n: n_total };
    let reduce_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pipe_reduce_params"),
        contents: bytemuck::bytes_of(&reduce_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Bind groups
    let wf_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pipe_wf_bg"),
        layout: &wf_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: freq_in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: selection_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: freq_out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: prng_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wf_params_buf.as_entire_binding(),
            },
        ],
    });
    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pipe_reduce_bg"),
        layout: &reduce_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: freq_out_buf.as_entire_binding(),
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

    // Single encoder: WF → reduce (no CPU round-trip)
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pipe_wf_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pipe_wf_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&wf_pipeline);
        pass.set_bind_group(0, &wf_bg, &[]);
        pass.dispatch_workgroups(gpu.dispatch_1d(n_total, 256), 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pipe_reduce_pass"),
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

/// Neutral drift (s=0): mean frequency ≈ 0.5 ± stochastic noise.
fn validate_neutral_mean(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 1_u32;
    let n_loci = 500_u32;
    let two_n = 100_u32;
    let n_total = (n_pops * n_loci) as usize;

    let freq_in: Vec<f32> = vec![0.5; n_total];
    let selection: Vec<f32> = vec![0.0; n_loci as usize];
    let prng_state = seed_prng(n_total, 42);

    match gpu_wf_mean(
        gpu,
        &freq_in,
        &selection,
        &prng_state,
        n_pops,
        n_loci,
        two_n,
    ) {
        Ok(mean) => {
            let diff = (f64::from(mean) - 0.5).abs();
            h.check_upper(
                &format!("neutral pipeline: |mean - 0.5| = {diff:.4} within QS_VARIANCE_MAX"),
                diff,
                tolerances::QS_VARIANCE_MAX,
            );
        }
        Err(e) => {
            h.check_bool(&format!("neutral pipeline: dispatch failed — {e}"), false);
        }
    }
}

/// Positive selection (s=0.1): pipeline mean should exceed neutral expectation.
fn validate_selection_shifts_mean(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 1_u32;
    let n_loci = 500_u32;
    let two_n = 200_u32;
    let n_total = (n_pops * n_loci) as usize;

    let freq_in: Vec<f32> = vec![0.5; n_total];
    let selection: Vec<f32> = vec![0.1; n_loci as usize];
    let prng_state = seed_prng(n_total, 123);

    match gpu_wf_mean(
        gpu,
        &freq_in,
        &selection,
        &prng_state,
        n_pops,
        n_loci,
        two_n,
    ) {
        Ok(mean) => {
            h.check_bool(
                &format!("selection pipeline: mean={mean:.4} > 0.5 after positive selection"),
                f64::from(mean) > 0.5,
            );
        }
        Err(e) => {
            h.check_bool(&format!("selection pipeline: dispatch failed — {e}"), false);
        }
    }
}

/// Boundary: `freq_in` = 0 or 1 stays fixed regardless of selection.
fn validate_boundary_frequencies(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 1_u32;
    let n_loci = 100_u32;
    let two_n = 50_u32;
    let n_total = (n_pops * n_loci) as usize;

    // Half at fixation (1.0), half at loss (0.0)
    let freq_in: Vec<f32> = (0..n_total)
        .map(|i| if i < n_total / 2 { 0.0 } else { 1.0 })
        .collect();
    let selection: Vec<f32> = vec![0.05; n_loci as usize];
    let prng_state = seed_prng(n_total, 777);

    match gpu_wf_mean(
        gpu,
        &freq_in,
        &selection,
        &prng_state,
        n_pops,
        n_loci,
        two_n,
    ) {
        Ok(mean) => {
            // Mean of half-0 half-1 population should stay near 0.5
            let diff = (f64::from(mean) - 0.5).abs();
            h.check_upper(
                &format!("boundary pipeline: |mean - 0.5| = {diff:.4} (fixed alleles)"),
                diff,
                tolerances::QS_VARIANCE_MAX,
            );
        }
        Err(e) => {
            h.check_bool(&format!("boundary pipeline: dispatch failed — {e}"), false);
        }
    }
}

/// Same PRNG seed → identical pipeline scalar.
fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_pops = 1_u32;
    let n_loci = 200_u32;
    let two_n = 50_u32;
    let n_total = (n_pops * n_loci) as usize;

    let freq_in: Vec<f32> = vec![0.5; n_total];
    let selection: Vec<f32> = vec![0.02; n_loci as usize];
    let prng1 = seed_prng(n_total, 9999);
    let prng2 = seed_prng(n_total, 9999);

    let r1 = gpu_wf_mean(gpu, &freq_in, &selection, &prng1, n_pops, n_loci, two_n);
    let r2 = gpu_wf_mean(gpu, &freq_in, &selection, &prng2, n_pops, n_loci, two_n);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("pipeline determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("pipeline determinism: dispatch failed", false);
        }
    }
}

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
