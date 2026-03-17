// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU pipeline validation: `HillGate` f64 polyfill + CPU mean (Paper 021).
//!
//! Stage 1: Polyfilled `HillGate` f64 dispatch → `output[nx*ny]` (f64).
//! Stage 2: CPU mean over output.
//!
//! Uses S-17 `pow()` → `pow_f64()` polyfill to avoid NVVM/NAK compilation
//! failure on RTX 4070 (Ada Lovelace) and TITAN V (NVK).
//!
//! ## Pipeline
//!
//! ```text
//! Upload cdg_grid[nx], ai_grid[ny] (once)
//!   ↓
//! HillGate f64 dispatch (polyfill) → output[nx*ny]
//!   ↓
//! CPU mean(output) → scalar
//! ```
//!
//! ## Provenance
//!
//! Shader: `barracuda::ops::bio::hill_gate::WGSL_HILL_GATE_F64` (patched).
//! Validated on: RTX 4070 (Vulkan), TITAN V (NVK).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "validation binary"
)]

use barracuda::ops::bio::hill_gate::{HillGateParams, WGSL_HILL_GATE_F64};
use neural_spring::gpu::Gpu;
use neural_spring::signal_integration::two_input_hill;
use neural_spring::tolerances;
use neural_spring::validation::{patch_pow_to_polyfill, ValidationHarness};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            println!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let dev = Arc::clone(gpu.wgpu_device());
    let patched = patch_pow_to_polyfill(WGSL_HILL_GATE_F64);
    let module = dev.compile_shader_f64(&patched, Some("hill_gate_f64_polyfill"));

    let device = gpu.device();
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("HillGate BGL"),
        entries: &[
            bgl_entry(0, true),
            bgl_entry(1, true),
            bgl_entry(2, false),
            bgl_entry(3, true),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("HillGate Layout"),
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("HillGate Pipeline (polyfill)"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let mut h = ValidationHarness::new("gpu_pipeline_signal");

    validate_small_grid(&mut h, &gpu, &pipeline, &bgl);
    validate_larger_grid(&mut h, &gpu, &pipeline, &bgl);
    validate_high_params(&mut h, &gpu, &pipeline, &bgl);
    validate_determinism(&mut h, &gpu, &pipeline, &bgl);

    h.finish();
}

const fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    let ty = if binding == 3 {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        }
    } else {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        }
    };
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn dispatch_hill_f64(
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    cdg: &[f64],
    ai: &[f64],
    params: &HillGateParams,
) -> Result<Vec<f64>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let cdg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cdg_f64"),
        contents: bytemuck::cast_slice(cdg),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let ai_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ai_f64"),
        contents: bytemuck::cast_slice(ai),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_total = (params.n_a * params.n_b) as usize;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hill_out_f64"),
        size: (n_total * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hill_params"),
        contents: bytemuck::bytes_of(params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let workgroups = (params.n_a * params.n_b).div_ceil(256);

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: bgl,
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
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("HillGate Dispatch"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("HillGate Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f64(&output, n_total)
}

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

fn validate_small_grid(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
) {
    let (nx, ny) = (10_usize, 10_usize);
    let (vmax, k1, k2, n1, n2) = (1.0, 1.0, 1.0, 2.0, 2.0);
    let cdg = make_linear_grid(nx, 0.5, 5.0);
    let ai = make_linear_grid(ny, 0.5, 5.0);
    let cpu_mean = cpu_mean_hill_grid(&cdg, &ai, vmax, k1, k2, n1, n2);

    let params = HillGateParams {
        n_a: nx as u32,
        n_b: ny as u32,
        mode: 1,
        _pad: 0,
        k_a: k1,
        k_b: k2,
        n_a_exp: n1,
        n_b_exp: n2,
        vmax,
        _pad2: 0.0,
    };

    match dispatch_hill_f64(gpu, pipeline, bgl, &cdg, &ai, &params) {
        Ok(out) => {
            let gpu_mean = out.iter().sum::<f64>() / out.len() as f64;
            h.check_abs(
                &format!("signal small 10×10: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_F64_TRANSCENDENTAL,
            );
        }
        Err(e) => h.check_bool(&format!("signal small grid: dispatch failed — {e}"), false),
    }
}

fn validate_larger_grid(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
) {
    let (nx, ny) = (32_usize, 32_usize);
    let (vmax, k1, k2, n1, n2) = (1.0, 1.0, 1.0, 2.0, 2.0);
    let cdg = make_linear_grid(nx, 0.5, 5.0);
    let ai = make_linear_grid(ny, 0.5, 5.0);
    let cpu_mean = cpu_mean_hill_grid(&cdg, &ai, vmax, k1, k2, n1, n2);

    let params = HillGateParams {
        n_a: nx as u32,
        n_b: ny as u32,
        mode: 1,
        _pad: 0,
        k_a: k1,
        k_b: k2,
        n_a_exp: n1,
        n_b_exp: n2,
        vmax,
        _pad2: 0.0,
    };

    match dispatch_hill_f64(gpu, pipeline, bgl, &cdg, &ai, &params) {
        Ok(out) => {
            let gpu_mean = out.iter().sum::<f64>() / out.len() as f64;
            h.check_abs(
                &format!("signal larger 32×32: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_F64_TRANSCENDENTAL,
            );
        }
        Err(e) => h.check_bool(&format!("signal larger grid: dispatch failed — {e}"), false),
    }
}

fn validate_high_params(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
) {
    let (nx, ny) = (10_usize, 10_usize);
    let (vmax, k1, k2, n1, n2) = (2.0, 1.0, 1.0, 3.0, 3.0);
    let cdg = make_linear_grid(nx, 0.5, 3.0);
    let ai = make_linear_grid(ny, 0.5, 3.0);
    let cpu_mean = cpu_mean_hill_grid(&cdg, &ai, vmax, k1, k2, n1, n2);

    let params = HillGateParams {
        n_a: nx as u32,
        n_b: ny as u32,
        mode: 1,
        _pad: 0,
        k_a: k1,
        k_b: k2,
        n_a_exp: n1,
        n_b_exp: n2,
        vmax,
        _pad2: 0.0,
    };

    match dispatch_hill_f64(gpu, pipeline, bgl, &cdg, &ai, &params) {
        Ok(out) => {
            let gpu_mean = out.iter().sum::<f64>() / out.len() as f64;
            h.check_abs(
                &format!("signal high params vmax=2: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_F64_TRANSCENDENTAL,
            );
        }
        Err(e) => h.check_bool(&format!("signal high params: dispatch failed — {e}"), false),
    }
}

fn validate_determinism(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
) {
    let (nx, ny) = (10_usize, 10_usize);
    let cdg = make_linear_grid(nx, 0.5, 5.0);
    let ai = make_linear_grid(ny, 0.5, 5.0);
    let params = HillGateParams {
        n_a: nx as u32,
        n_b: ny as u32,
        mode: 1,
        _pad: 0,
        k_a: 1.0,
        k_b: 1.0,
        n_a_exp: 2.0,
        n_b_exp: 2.0,
        vmax: 1.0,
        _pad2: 0.0,
    };

    let r1 = dispatch_hill_f64(gpu, pipeline, bgl, &cdg, &ai, &params);
    let r2 = dispatch_hill_f64(gpu, pipeline, bgl, &cdg, &ai, &params);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            let a_mean = a.iter().sum::<f64>() / a.len() as f64;
            let b_mean = b.iter().sum::<f64>() / b.len() as f64;
            h.check_bool(
                &format!("signal determinism: run1={a_mean:.6} == run2={b_mean:.6}"),
                (a_mean - b_mean).abs() < f64::EPSILON,
            );
        }
        _ => h.check_bool("signal determinism: dispatch failed", false),
    }
}
