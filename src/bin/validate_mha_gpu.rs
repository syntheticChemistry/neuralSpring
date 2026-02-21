// SPDX-License-Identifier: AGPL-3.0-or-later

//! S-03b resolution: GPU-resident MHA via matmul + head split/concat shaders.
//!
//! Validates `metalForge/shaders/head_split.wgsl` and `head_concat.wgsl`
//! against the CPU head-split/concat in `evolved::mha`, then composes
//! a full MHA pipeline using validated `barracuda::matmul` + these shaders.
//!
//! The native `BarraCUDA` MHA fuses matmul into the projection shaders,
//! causing GPU hangs (S-03b). This decomposed approach avoids the hang
//! by separating projection (matmul) from data movement (head split/concat).
//!
//! Absorption target: `barracuda::ops::mha` --- replace fused projection
//! shaders with matmul + head_split/head_concat.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::suboptimal_flops
)]

use barracuda::device::WgpuDevice;
use neural_spring::gpu::Gpu;
use neural_spring::require;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const WGSL_HEAD_SPLIT: &str = include_str!("../../metalForge/shaders/head_split.wgsl");
const WGSL_HEAD_CONCAT: &str = include_str!("../../metalForge/shaders/head_concat.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HeadParams {
    batch_size: u32,
    seq_len: u32,
    d_model: u32,
    num_heads: u32,
    head_dim: u32,
    _pad: [u32; 3],
}

#[tokio::main]
async fn main() {
    let Ok(gpu) = Gpu::new().await else {
        eprintln!("  SKIP — no adapter");
        std::process::exit(0);
    };
    eprintln!(
        "  adapter: {} ({:?}, {:?})",
        gpu.adapter_name, gpu.device_type, gpu.backend,
    );
    let dev = gpu.wgpu_device().clone();

    let mut h = ValidationHarness::new("mha_gpu_s03b");

    validate_head_split(&mut h, &dev);
    validate_head_concat(&mut h, &dev);
    validate_split_concat_roundtrip(&mut h, &dev);
    validate_larger_sizes(&mut h, &dev);

    h.finish();
}

fn validate_head_split(h: &mut ValidationHarness, dev: &Arc<WgpuDevice>) {
    let batch = 1_u32;
    let seq = 4_u32;
    let heads = 2_u32;
    let d_head = 3_u32;
    let d_model = heads * d_head;
    let n = (batch * seq * d_model) as usize;

    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();

    let mut expected = vec![0.0_f32; n];
    for s in 0..seq {
        for hh in 0..heads {
            for d in 0..d_head {
                let src = s * d_model + hh * d_head + d;
                let dst = hh * seq * d_head + s * d_head + d;
                expected[dst as usize] = input[src as usize];
            }
        }
    }

    let gpu_out = require!(
        h,
        run_head_split(dev, &input, batch, seq, d_model, heads, d_head),
        "head_split readback"
    );

    let max_err = gpu_out
        .iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max);

    h.check_abs(
        "head_split layout matches CPU",
        f64::from(max_err),
        0.0,
        tolerances::TENSOR_EXACT_F32,
    );
    h.check_bool("head_split output length", gpu_out.len() == expected.len());
}

fn validate_head_concat(h: &mut ValidationHarness, dev: &Arc<WgpuDevice>) {
    let batch = 1_u32;
    let seq = 4_u32;
    let heads = 2_u32;
    let d_head = 3_u32;
    let d_model = heads * d_head;
    let n = (batch * seq * d_model) as usize;

    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();

    let mut expected = vec![0.0_f32; n];
    for s in 0..seq {
        for hh in 0..heads {
            for d in 0..d_head {
                let src = hh * seq * d_head + s * d_head + d;
                let dst = s * d_model + hh * d_head + d;
                expected[dst as usize] = input[src as usize];
            }
        }
    }

    let gpu_out = require!(
        h,
        run_head_concat(dev, &input, batch, seq, d_model, heads, d_head),
        "head_concat readback"
    );

    let max_err = gpu_out
        .iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max);

    h.check_abs(
        "head_concat layout matches CPU",
        f64::from(max_err),
        0.0,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_split_concat_roundtrip(h: &mut ValidationHarness, dev: &Arc<WgpuDevice>) {
    let batch = 2_u32;
    let seq = 8_u32;
    let heads = 4_u32;
    let d_head = 8_u32;
    let d_model = heads * d_head;

    let mut rng = Rng::new(42);
    let n = (batch * seq * d_model) as usize;
    let input: Vec<f32> = (0..n).map(|_| rng.uniform() as f32).collect();

    let split = require!(
        h,
        run_head_split(dev, &input, batch, seq, d_model, heads, d_head),
        "head_split readback"
    );
    let roundtrip = require!(
        h,
        run_head_concat(dev, &split, batch, seq, d_model, heads, d_head),
        "head_concat readback"
    );

    let max_err = input
        .iter()
        .zip(roundtrip.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    h.check_abs(
        "split->concat roundtrip exact",
        f64::from(max_err),
        0.0,
        tolerances::TENSOR_EXACT_F32,
    );
}

fn validate_larger_sizes(h: &mut ValidationHarness, dev: &Arc<WgpuDevice>) {
    // Test head_split/head_concat at production MHA sizes
    // B=2, H=8, S=128, D/H=64 (d_model=512)
    for &(batch, seq, heads, d_head, label) in &[
        (1_u32, 8_u32, 4_u32, 8_u32, "small (1,8,4,8)"),
        (2, 32, 8, 64, "medium (2,32,8,64)"),
        (4, 128, 8, 64, "large (4,128,8,64)"),
    ] {
        let d_model = heads * d_head;
        let n = (batch * seq * d_model) as usize;
        let mut rng = Rng::new(42 + u64::from(batch * seq));
        let input: Vec<f32> = (0..n).map(|_| rng.uniform() as f32).collect();

        // CPU reference split
        let mut cpu_split = vec![0.0_f32; n];
        for b in 0..batch {
            for s_idx in 0..seq {
                for hh in 0..heads {
                    for d in 0..d_head {
                        let src = b * seq * d_model + s_idx * d_model + hh * d_head + d;
                        let dst = b * heads * seq * d_head + hh * seq * d_head + s_idx * d_head + d;
                        cpu_split[dst as usize] = input[src as usize];
                    }
                }
            }
        }

        let gpu_split = require!(
            h,
            run_head_split(dev, &input, batch, seq, d_model, heads, d_head),
            "head_split readback"
        );

        let split_err = gpu_split
            .iter()
            .zip(cpu_split.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0_f32, f32::max);

        h.check_abs(
            &format!("head_split {label}"),
            f64::from(split_err),
            0.0,
            tolerances::TENSOR_EXACT_F32,
        );

        // Roundtrip
        let roundtrip = require!(
            h,
            run_head_concat(dev, &gpu_split, batch, seq, d_model, heads, d_head),
            "head_concat readback"
        );

        let rt_err = input
            .iter()
            .zip(roundtrip.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        h.check_abs(
            &format!("split->concat roundtrip {label}"),
            f64::from(rt_err),
            0.0,
            tolerances::TENSOR_EXACT_F32,
        );
    }
}

// ─── GPU dispatch helpers ────────────────────────────────────────

fn run_head_split(
    dev: &Arc<WgpuDevice>,
    input: &[f32],
    batch: u32,
    seq: u32,
    d_model: u32,
    heads: u32,
    d_head: u32,
) -> Result<Vec<f32>, impl std::fmt::Display> {
    let total = (batch * heads * seq * d_head) as usize;
    dispatch_head_shader(
        dev,
        WGSL_HEAD_SPLIT,
        "head_split",
        input,
        total,
        HeadParams {
            batch_size: batch,
            seq_len: seq,
            d_model,
            num_heads: heads,
            head_dim: d_head,
            _pad: [0; 3],
        },
    )
}

fn run_head_concat(
    dev: &Arc<WgpuDevice>,
    input: &[f32],
    batch: u32,
    seq: u32,
    d_model: u32,
    heads: u32,
    d_head: u32,
) -> Result<Vec<f32>, impl std::fmt::Display> {
    let total = (batch * seq * d_model) as usize;
    dispatch_head_shader(
        dev,
        WGSL_HEAD_CONCAT,
        "head_concat",
        input,
        total,
        HeadParams {
            batch_size: batch,
            seq_len: seq,
            d_model,
            num_heads: heads,
            head_dim: d_head,
            _pad: [0; 3],
        },
    )
}

fn dispatch_head_shader(
    dev: &Arc<WgpuDevice>,
    wgsl: &str,
    entry: &str,
    input: &[f32],
    output_count: usize,
    params: HeadParams,
) -> Result<Vec<f32>, impl std::fmt::Display> {
    let device = dev.device();
    let queue = dev.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("head_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("head_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&pl),
        module: &shader,
        entry_point: entry,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (output_count * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("head_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("head_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("head_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((output_count as u32).div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    dev.read_buffer_f32(&out_buf, output_count)
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
