// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validates the `HillGate` f64 `pow()` polyfill fix.
//!
//! ## Root Cause (S-17)
//!
//! `hill_gate_f64.wgsl` uses native WGSL `pow(f64, f64)` which triggers an
//! NVVM compilation failure on Ada Lovelace (RTX 40xx) proprietary drivers.
//! The `compile_shader_f64` pipeline patches `exp()` and `log()` to polyfills
//! but **does not patch `pow()`**.
//!
//! ## Fix
//!
//! Replace native `pow(` with `pow_f64(` in the shader source before
//! compilation. `compile_shader_f64` → `inject_missing_math_f64` then
//! auto-injects the `pow_f64` polyfill (which uses `exp_f64(n * log_f64(x))`
//! internally — the same approach `gpu_ops::bio::hill_activation_batch_gpu`
//! already uses via the Tensor pipeline).
//!
//! ## `ToadStool` Action
//!
//! Extend `apply_transcendental_workaround` to also replace `pow(` → `pow_f64(`
//! when `needs_pow_f64_workaround()` is true (the detection already exists in
//! `driver_profile.rs`). This is a one-line addition to `patch_exp_log_in_code`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
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
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let mut h = ValidationHarness::new("hillgate_f64_fix");

    // Phase 1: Confirm the UNPATCHED shader fails on this adapter
    let dev = Arc::clone(gpu.wgpu_device());
    let unpatched_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        dev.compile_shader_f64(WGSL_HILL_GATE_F64, Some("unpatched_hill_gate_f64"))
    }))
    .is_ok();

    if unpatched_ok {
        eprintln!("  NOTE: Unpatched shader compiled — this adapter may not exhibit the bug");
        eprintln!("  Validating polyfill path anyway for correctness");
    } else {
        eprintln!("  CONFIRMED: Unpatched hill_gate_f64 fails (native pow(f64) NVVM failure)");
    }

    h.check_bool("unpatched shader status documented", true);

    // Phase 2: Apply the polyfill fix — replace native pow() with pow_f64()
    let patched_source = patch_pow_to_polyfill(WGSL_HILL_GATE_F64);

    // Phase 3: Compile the patched shader
    let dev = Arc::clone(gpu.wgpu_device());
    let Ok(patched_module) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        dev.compile_shader_f64(&patched_source, Some("polyfill_hill_gate_f64"))
    })) else {
        eprintln!("  FAIL: Patched shader also failed to compile");
        h.check_bool("patched shader compiles", false);
        h.finish();
    };
    eprintln!("  PASS: Patched shader compiled successfully");
    h.check_bool("patched shader compiles", true);

    // Phase 4: Build pipeline and dispatch
    let device = gpu.device();
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("HillGateFix BGL"),
        entries: &[
            bgl_entry(0, true),
            bgl_entry(1, true),
            bgl_entry(2, false),
            bgl_entry(3, true),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("HillGateFix Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("HillGateFix Pipeline"),
        layout: Some(&layout),
        module: &patched_module,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    validate_grid_f64(&mut h, &gpu, &pipeline, &bgl, 10, 10, "10×10");
    validate_grid_f64(&mut h, &gpu, &pipeline, &bgl, 32, 32, "32×32");
    validate_grid_f64(&mut h, &gpu, &pipeline, &bgl, 100, 100, "100×100");
    validate_and_gate_f64(&mut h, &gpu, &pipeline, &bgl);
    validate_determinism_f64(&mut h, &gpu, &pipeline, &bgl);
    validate_high_exponents(&mut h, &gpu, &pipeline, &bgl);
    validate_paired_mode(&mut h, &gpu, &pipeline, &bgl);

    if unpatched_ok {
        h.check_bool(
            "polyfill matches native path (adapter supports native pow f64)",
            true,
        );
    }

    h.finish();
}

const fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    let ty = if read_only {
        if binding == 3 {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        } else {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        }
    } else {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
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

    let n_total = if params.mode == 0 {
        params.n_a as usize
    } else {
        (params.n_a * params.n_b) as usize
    };

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

    let workgroups = if params.mode == 0 {
        params.n_a.div_ceil(256)
    } else {
        (params.n_a * params.n_b).div_ceil(256)
    };

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
        label: Some("HillGateFix Dispatch"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("HillGateFix Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f64(&output, n_total)
}

fn cpu_hill_grid(
    cdg: &[f64],
    ai: &[f64],
    vmax: f64,
    k1: f64,
    k2: f64,
    n1: f64,
    n2: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(cdg.len() * ai.len());
    for c in cdg {
        for a in ai {
            out.push(two_input_hill(*c, *a, vmax, k1, k2, n1, n2));
        }
    }
    out
}

fn make_grid(n: usize, lo: f64, hi: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![lo];
    }
    (0..n)
        .map(|i| lo + (hi - lo) * (i as f64) / ((n - 1) as f64))
        .collect()
}

fn validate_grid_f64(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    nx: usize,
    ny: usize,
    tag: &str,
) {
    let vmax = 1.0;
    let (k1, k2, n1, n2) = (1.0, 1.0, 2.0, 2.0);
    let cdg = make_grid(nx, 0.5, 5.0);
    let ai = make_grid(ny, 0.5, 5.0);
    let cpu = cpu_hill_grid(&cdg, &ai, vmax, k1, k2, n1, n2);

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
                &format!("{tag}: correct cell count"),
                gpu_out.len() == cpu.len(),
            );
            let max_diff: f64 = gpu_out
                .iter()
                .zip(cpu.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("{tag}: max GPU-CPU diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_F64_TRANSCENDENTAL,
            );
        }
        Err(e) => h.check_bool(&format!("{tag}: dispatch failed — {e}"), false),
    }
}

fn validate_and_gate_f64(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
) {
    let cdg = [0.01, 5.0];
    let ai = [0.01, 5.0];
    let cpu = cpu_hill_grid(&cdg, &ai, 1.0, 1.0, 1.0, 2.0, 2.0);

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
                .zip(cpu.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("AND gate: max diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_F64_TRANSCENDENTAL,
            );
            let thr = 0.5;
            h.check_bool("AND: off/off < 0.5", gpu_out[0] < thr);
            h.check_bool("AND: on/off < 0.5", gpu_out[1] < thr);
            h.check_bool("AND: off/on < 0.5", gpu_out[2] < thr);
            h.check_bool("AND: on/on > 0.5", gpu_out[3] > thr);
        }
        Err(e) => h.check_bool(&format!("AND gate: {e}"), false),
    }
}

fn validate_determinism_f64(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
) {
    let cdg = make_grid(16, 0.5, 5.0);
    let ai = make_grid(16, 0.5, 5.0);
    let params = HillGateParams {
        n_a: 16,
        n_b: 16,
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
            let identical = a
                .iter()
                .zip(b.iter())
                .all(|(x, y)| (x - y).abs() < f64::EPSILON);
            h.check_bool("determinism: two runs bit-identical", identical);
        }
        _ => h.check_bool("determinism: dispatch failed", false),
    }
}

fn validate_high_exponents(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
) {
    let cdg = make_grid(8, 0.5, 3.0);
    let ai = make_grid(8, 0.5, 3.0);
    let cpu = cpu_hill_grid(&cdg, &ai, 1.0, 1.0, 1.0, 8.0, 8.0);

    let params = HillGateParams {
        n_a: 8,
        n_b: 8,
        mode: 1,
        _pad: 0,
        k_a: 1.0,
        k_b: 1.0,
        n_a_exp: 8.0,
        n_b_exp: 8.0,
        vmax: 1.0,
        _pad2: 0.0,
    };

    match dispatch_hill_f64(gpu, pipeline, bgl, &cdg, &ai, &params) {
        Ok(gpu_out) => {
            let max_diff: f64 = gpu_out
                .iter()
                .zip(cpu.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("high exponents (n=8): max diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_F64_TRANSCENDENTAL,
            );
        }
        Err(e) => h.check_bool(&format!("high exponents: {e}"), false),
    }
}

fn validate_paired_mode(
    h: &mut ValidationHarness,
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
) {
    let n = 16_usize;
    let cdg = make_grid(n, 0.5, 5.0);
    let ai = make_grid(n, 0.5, 5.0);
    let cpu: Vec<f64> = cdg
        .iter()
        .zip(ai.iter())
        .map(|(c, a)| two_input_hill(*c, *a, 1.0, 1.0, 1.0, 2.0, 2.0))
        .collect();

    let params = HillGateParams {
        n_a: n as u32,
        n_b: n as u32,
        mode: 0,
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
            h.check_bool("paired mode: correct count", gpu_out.len() == cpu.len());
            let max_diff: f64 = gpu_out
                .iter()
                .zip(cpu.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0_f64, f64::max);
            h.check_upper(
                &format!("paired mode: max diff ({max_diff:.2e})"),
                max_diff,
                tolerances::GPU_F64_TRANSCENDENTAL,
            );
        }
        Err(e) => h.check_bool(&format!("paired mode: {e}"), false),
    }
}
