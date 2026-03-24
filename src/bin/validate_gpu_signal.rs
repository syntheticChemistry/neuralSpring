// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: two-input Hill function via polyfilled f64 shader.
//!
//! Validates the `HillGate` f64 shader against CPU
//! `signal_integration::two_input_hill`. The GPU op evaluates
//! the Hill function over a 2D (cdg, ai) grid in a single dispatch.
//!
//! ## S-17 Fix
//!
//! `HillGateGpu` uses native WGSL `pow(f64, f64)` which causes NVVM/NAK
//! compilation failure on both RTX 4070 (Ada Lovelace, proprietary) and
//! TITAN V (NVK open-source). The fix replaces `pow(` with `pow_f64(` in the
//! shader source; `compile_shader_f64` auto-injects the polyfill. See
//! `validate_hillgate_f64_fix.rs` for the full proof-of-concept.
//!
//! ## Papers validated
//!
//! - Paper 021: Signal Integration (two-input Hill function / AND gate)
//!
//! ## Provenance
//!
//! CPU reference: `signal_integration::two_input_hill` (seed=0, 10×10 grid).
//! GPU shader: `barracuda::ops::bio::hill_gate::WGSL_HILL_GATE_F64` (patched)
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
use neural_spring::validation::{ValidationHarness, patch_pow_to_polyfill};
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

    let mut h = ValidationHarness::new("gpu_signal");
    let dev = Arc::clone(gpu.wgpu_device());

    let patched_source = patch_pow_to_polyfill(WGSL_HILL_GATE_F64);
    let module = dev.compile_shader_f64(&patched_source, Some("hill_gate_f64_polyfill"));

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

    validate_small_grid(&mut h, &gpu, &pipeline, &bgl);
    validate_and_gate_corners(&mut h, &gpu, &pipeline, &bgl);
    validate_determinism(&mut h, &gpu, &pipeline, &bgl);
    validate_larger_grid(&mut h, &gpu, &pipeline, &bgl);

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

    let input_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cdg_f64"),
        contents: bytemuck::cast_slice(cdg),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let input_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
                resource: input_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_b.as_entire_binding(),
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

    Ok(gpu.read_buffer_f64(&output, n_total)?)
}

fn cpu_hill_grid(
    cdg_grid: &[f64],
    ai_grid: &[f64],
    vmax: f64,
    k1: f64,
    k2: f64,
    n1: f64,
    n2: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(cdg_grid.len() * ai_grid.len());
    for cdg in cdg_grid {
        for ai in ai_grid {
            out.push(two_input_hill(*cdg, *ai, vmax, k1, k2, n1, n2));
        }
    }
    out
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
    let nx = 10_usize;
    let ny = 10_usize;
    let (vmax, k1, k2, n1, n2) = (1.0, 1.0, 1.0, 2.0, 2.0);

    let cdg = make_linear_grid(nx, 0.5, 5.0);
    let ai = make_linear_grid(ny, 0.5, 5.0);
    let cpu_out = cpu_hill_grid(&cdg, &ai, vmax, k1, k2, n1, n2);

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
        Ok(gpu_out) => {
            h.check_bool(
                &format!("small grid: correct cell count ({})", gpu_out.len()),
                gpu_out.len() == cpu_out.len(),
            );

            let max_diff: f64 = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("small grid: max GPU-CPU diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_F64_TRANSCENDENTAL,
            );
        }
        Err(e) => {
            h.check_bool(&format!("small grid: dispatch failed — {e}"), false);
        }
    }
}

fn validate_and_gate_corners(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
) {
    let cdg = [0.01_f64, 5.0];
    let ai = [0.01_f64, 5.0];
    let (vmax, k1, k2, n1, n2) = (1.0, 1.0, 1.0, 2.0, 2.0);
    let cpu_out = cpu_hill_grid(&cdg, &ai, vmax, k1, k2, n1, n2);

    let params = HillGateParams {
        n_a: 2,
        n_b: 2,
        mode: 1,
        _pad: 0,
        k_a: 1.0,
        k_b: 1.0,
        n_a_exp: 2.0,
        n_b_exp: 2.0,
        vmax: 1.0,
        _pad2: 0.0,
    };

    match dispatch_hill_f64(gpu, pipeline, bgl, &cdg, &ai, &params) {
        Ok(gpu_out) => {
            let max_diff: f64 = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!("AND gate corners: max diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_F64_TRANSCENDENTAL,
            );

            let threshold = 0.5;
            let (off_off, on_off, off_on, on_on) = (gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3]);
            h.check_bool("AND corners: off/off low", off_off < threshold);
            h.check_bool("AND corners: on/off low", on_off < threshold);
            h.check_bool("AND corners: off/on low", off_on < threshold);
            h.check_bool("AND corners: on/on high", on_on > threshold);
        }
        Err(e) => {
            h.check_bool(&format!("AND gate corners: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
) {
    let nx = 10_usize;
    let ny = 10_usize;
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

    let run1 = dispatch_hill_f64(gpu, pipeline, bgl, &cdg, &ai, &params);
    let run2 = dispatch_hill_f64(gpu, pipeline, bgl, &cdg, &ai, &params);

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1
                .iter()
                .zip(r2.iter())
                .all(|(a, b)| (a - b).abs() < f64::EPSILON);
            h.check_bool("determinism: two hill_gate runs identical", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}

fn validate_larger_grid(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
) {
    let nx = 32_usize;
    let ny = 32_usize;
    let (vmax, k1, k2, n1, n2) = (1.0, 1.0, 1.0, 2.0, 2.0);

    let cdg = make_linear_grid(nx, 0.5, 5.0);
    let ai = make_linear_grid(ny, 0.5, 5.0);
    let cpu_out = cpu_hill_grid(&cdg, &ai, vmax, k1, k2, n1, n2);

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
        Ok(gpu_out) => {
            let max_diff: f64 = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "32×32 grid: max GPU-CPU diff ({max_diff:.2e}), {} cells",
                    gpu_out.len()
                ),
                max_diff,
                tolerances::GPU_F64_TRANSCENDENTAL,
            );
        }
        Err(e) => {
            h.check_bool(&format!("32×32 grid: dispatch failed — {e}"), false);
        }
    }
}
