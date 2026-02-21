// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline: `hill_gate` → `mean_reduce` → scalar readback (Paper 021).
//!
//! Chains two shader stages in a single `CommandEncoder` with zero CPU round-trips.
//! Stage 1: `hill_gate` — two-input Hill AND gate over 2D grid.
//! Stage 2: `mean_reduce` — output array to scalar mean.
//!
//! ## Pipeline
//!
//! ```text
//! Upload cdg_grid[nx], ai_grid[ny] (once)
//!   ↓
//! ┌──────────────────────────────────────────────────┐
//! │  Stage 1: hill_gate.wgsl                         │
//! │    cdg_grid[nx], ai_grid[ny] → output[nx*ny]    │
//! │                                                  │
//! │  Stage 2: mean_reduce.wgsl                       │
//! │    output[nx*ny] → mean_hill (scalar)            │
//! └──────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — NO CPU round-trip)
//! Readback: 4 bytes
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::signal_integration::two_input_hill;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const HILL_WGSL: &str = include_str!("../../metalForge/shaders/hill_gate.wgsl");
const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HillParams {
    nx: u32,
    ny: u32,
    vmax: f32,
    k1: f32,
    k2: f32,
    n1: f32,
    n2: f32,
    _pad: u32,
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

    let mut h = ValidationHarness::new("gpu_pipeline_signal");

    validate_small_grid(&mut h, &gpu);
    validate_larger_grid(&mut h, &gpu);
    validate_high_params(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_hill_grid(
    cdg_grid: &[f64],
    ai_grid: &[f64],
    vmax: f64,
    k1: f64,
    k2: f64,
    n1: f64,
    n2: f64,
) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for cdg in cdg_grid {
        for ai in ai_grid {
            sum += two_input_hill(*cdg, *ai, vmax, k1, k2, n1, n2);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn make_linear_grid(n: usize, low: f64, high: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![low];
    }
    (0..n)
        .map(|i| low + (high - low) * (i as f64) / ((n - 1) as f64))
        .collect()
}

// ── GPU chained pipeline ───────────────────────────────────────────

fn gpu_hill_mean(gpu: &Gpu, cdg: &[f32], ai: &[f32], params: &HillParams) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();
    let n_total = (params.nx * params.ny) as usize;

    let hill_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_hill"),
        source: wgpu::ShaderSource::Wgsl(HILL_WGSL.into()),
    });

    let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_reduce"),
        source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
    });

    let hill_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_hill_bgl"),
        entries: &[
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            uniform_entry(3),
        ],
    });

    let hill_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_hill_pl"),
        bind_group_layouts: &[&hill_bgl],
        push_constant_ranges: &[],
    });

    let hill_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_hill_pipeline"),
        layout: Some(&hill_pl),
        module: &hill_shader,
        entry_point: "hill_gate",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_reduce_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });

    let reduce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_reduce_pl"),
        bind_group_layouts: &[&reduce_bgl],
        push_constant_ranges: &[],
    });

    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_reduce_pipeline"),
        layout: Some(&reduce_pl),
        module: &reduce_shader,
        entry_point: "mean_reduce",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let cdg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_cdg"),
        contents: bytemuck::cast_slice(cdg),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let ai_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_ai"),
        contents: bytemuck::cast_slice(ai),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_hill_output"),
        size: (n_total * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let hill_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_hill_params"),
        contents: bytemuck::bytes_of(params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_mean_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let reduce_params = ReduceParams { n: n_total as u32 };
    let reduce_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_reduce_params"),
        contents: bytemuck::bytes_of(&reduce_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let hill_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_hill_bg"),
        layout: &hill_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cdg_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ai_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: hill_params_buf.as_entire_binding(),
            },
        ],
    });

    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_reduce_bg"),
        layout: &reduce_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buf.as_entire_binding(),
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
        label: Some("chain_signal_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_hill_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&hill_pipeline);
        pass.set_bind_group(0, &hill_bg, &[]);
        pass.dispatch_workgroups((n_total as u32).div_ceil(256), 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_reduce_pass"),
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

fn validate_small_grid(h: &mut ValidationHarness, gpu: &Gpu) {
    let nx = 10_usize;
    let ny = 10_usize;
    let vmax = 1.0_f64;
    let k1 = 1.0_f64;
    let k2 = 1.0_f64;
    let n1 = 2.0_f64;
    let n2 = 2.0_f64;

    let cdg_cpu = make_linear_grid(nx, 0.01, 5.0);
    let ai_cpu = make_linear_grid(ny, 0.01, 5.0);
    let cpu_mean = cpu_mean_hill_grid(&cdg_cpu, &ai_cpu, vmax, k1, k2, n1, n2);

    let cdg_f32: Vec<f32> = cdg_cpu.iter().map(|x| *x as f32).collect();
    let ai_f32: Vec<f32> = ai_cpu.iter().map(|x| *x as f32).collect();
    let params = HillParams {
        nx: nx as u32,
        ny: ny as u32,
        vmax: vmax as f32,
        k1: k1 as f32,
        k2: k2 as f32,
        n1: n1 as f32,
        n2: n2 as f32,
        _pad: 0,
    };

    match gpu_hill_mean(gpu, &cdg_f32, &ai_f32, &params) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("signal small 10×10: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                cpu_mean,
                tolerances::GPU_HILL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("signal small grid: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger_grid(h: &mut ValidationHarness, gpu: &Gpu) {
    let nx = 32_usize;
    let ny = 32_usize;
    let vmax = 1.0_f64;
    let k1 = 1.0_f64;
    let k2 = 1.0_f64;
    let n1 = 2.0_f64;
    let n2 = 2.0_f64;

    let cdg_cpu = make_linear_grid(nx, 0.01, 5.0);
    let ai_cpu = make_linear_grid(ny, 0.01, 5.0);
    let cpu_mean = cpu_mean_hill_grid(&cdg_cpu, &ai_cpu, vmax, k1, k2, n1, n2);

    let cdg_f32: Vec<f32> = cdg_cpu.iter().map(|x| *x as f32).collect();
    let ai_f32: Vec<f32> = ai_cpu.iter().map(|x| *x as f32).collect();
    let params = HillParams {
        nx: nx as u32,
        ny: ny as u32,
        vmax: vmax as f32,
        k1: k1 as f32,
        k2: k2 as f32,
        n1: n1 as f32,
        n2: n2 as f32,
        _pad: 0,
    };

    match gpu_hill_mean(gpu, &cdg_f32, &ai_f32, &params) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("signal larger 32×32: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                cpu_mean,
                tolerances::GPU_HILL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("signal larger grid: dispatch failed — {e}"), false);
        }
    }
}

fn validate_high_params(h: &mut ValidationHarness, gpu: &Gpu) {
    let nx = 10_usize;
    let ny = 10_usize;
    let vmax = 2.0_f64;
    let k1 = 1.0_f64;
    let k2 = 1.0_f64;
    let n1 = 3.0_f64;
    let n2 = 3.0_f64;

    let cdg_cpu = make_linear_grid(nx, 0.5, 3.0);
    let ai_cpu = make_linear_grid(ny, 0.5, 3.0);
    let cpu_mean = cpu_mean_hill_grid(&cdg_cpu, &ai_cpu, vmax, k1, k2, n1, n2);

    let cdg_f32: Vec<f32> = cdg_cpu.iter().map(|x| *x as f32).collect();
    let ai_f32: Vec<f32> = ai_cpu.iter().map(|x| *x as f32).collect();
    let params = HillParams {
        nx: nx as u32,
        ny: ny as u32,
        vmax: vmax as f32,
        k1: k1 as f32,
        k2: k2 as f32,
        n1: n1 as f32,
        n2: n2 as f32,
        _pad: 0,
    };

    match gpu_hill_mean(gpu, &cdg_f32, &ai_f32, &params) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("signal high params vmax=2: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                cpu_mean,
                tolerances::GPU_HILL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("signal high params: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let nx = 10_usize;
    let ny = 10_usize;
    let cdg_cpu = make_linear_grid(nx, 0.01, 5.0);
    let ai_cpu = make_linear_grid(ny, 0.01, 5.0);
    let cdg_f32: Vec<f32> = cdg_cpu.iter().map(|x| *x as f32).collect();
    let ai_f32: Vec<f32> = ai_cpu.iter().map(|x| *x as f32).collect();
    let params = HillParams {
        nx: nx as u32,
        ny: ny as u32,
        vmax: 1.0,
        k1: 1.0,
        k2: 1.0,
        n1: 2.0,
        n2: 2.0,
        _pad: 0,
    };

    let r1 = gpu_hill_mean(gpu, &cdg_f32, &ai_f32, &params);
    let r2 = gpu_hill_mean(gpu, &cdg_f32, &ai_f32, &params);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("signal determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("signal determinism: dispatch failed", false);
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
