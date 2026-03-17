// SPDX-License-Identifier: AGPL-3.0-or-later
//! Diagnostic binary: traces the fused f64 GPU regression on Ada Lovelace.
//!
//! Tests compilation paths to identify whether `enable f64;` in the shader
//! source vs. stripping it causes the failure.
//!
//! ## Root Cause (Mar 2026)
//!
//! In wgpu 28, naga resolves f64 support from device capability flags, not
//! WGSL `enable f64;` directives.  When the directive is left in the source,
//! NVIDIA PTXAS on Ada Lovelace (SM89, RTX 40xx) silently produces broken
//! shaders that return zeros for all outputs.  Stripping the directive
//! before compilation fixes the issue.
//!
//! Fix: `barraCuda` `pipeline_cache.rs` → `get_or_compile_shader_f64_native`
//! now strips `enable f64;` before compilation, matching the behavior of
//! `compile_shader_f64` and `compile_shader_df64`.

#![expect(
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::cast_possible_truncation,
    clippy::default_trait_access,
    reason = "diagnostic binary"
)]

use neural_spring::validation::OrExit;
use std::sync::Arc;

fn main() {
    let rt = tokio::runtime::Runtime::new().or_exit("tokio runtime");
    let gpu = rt
        .block_on(neural_spring::gpu::Gpu::new())
        .or_exit("GPU init");
    let dev: Arc<barracuda::device::WgpuDevice> = gpu.wgpu_device().clone();

    println!("=== GPU: {} ===", dev.adapter_info().name);

    let profile = barracuda::device::driver_profile::GpuDriverProfile::from_device(&dev);
    println!("  fp64_strategy: {:?}", profile.fp64_strategy());
    println!("  precision_routing: {:?}", profile.precision_routing());
    println!("  f64_zeros_risk: {}", profile.f64_zeros_risk());

    // Run probes
    let caps = rt.block_on(barracuda::device::probe::probe_f64_builtins(&dev));
    println!("  Probes: {caps}");

    let healthy = barracuda::device::test_harness::fused_ops_healthy(&dev);
    println!("  fused_ops_healthy: {healthy}");

    // === Minimal f64 echo shader: read f64, write f64 ===
    println!("\n=== Test 1: Minimal f64 echo shader ===");
    test_echo_shader(&dev, true);
    test_echo_shader(&dev, false);

    // === Minimal DF64 shader: same as variance but ultra-simple ===
    println!("\n=== Test 2: Minimal DF64 sum shader ===");
    test_df64_sum_shader(&dev, true);
    test_df64_sum_shader(&dev, false);

    // === VarianceF64 via compile_shader_f64 ===
    println!("\n=== Test 3: DF64 variance shader via compile_shader_f64 ===");
    test_variance_via_compile_shader_f64(&dev);

    println!("\n=== Diagnosis Complete ===");
}

fn test_echo_shader(dev: &Arc<barracuda::device::WgpuDevice>, with_enable_f64: bool) {
    let enable = if with_enable_f64 { "enable f64;\n" } else { "" };
    let src = format!(
        "{enable}\
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(1)
fn main() {{
    output[0] = input[0] + input[1];
    output[1] = input[2] * input[3];
}}"
    );

    let label = if with_enable_f64 {
        "echo WITH enable f64"
    } else {
        "echo WITHOUT enable f64"
    };

    match run_f64_shader(dev, &src, &[1.0_f64, 2.0, 3.0, 4.0], 2, label) {
        Ok(result) => {
            println!(
                "  {label}: [{:.6}, {:.6}] (expected [3.0, 12.0])",
                result[0], result[1]
            );
        }
        Err(e) => println!("  {label}: ERROR {e}"),
    }
}

fn test_df64_sum_shader(dev: &Arc<barracuda::device::WgpuDevice>, with_enable_f64: bool) {
    let enable = if with_enable_f64 { "enable f64;\n" } else { "" };
    let df64_core =
        include_str!("../../../barraCuda/crates/barracuda/src/shaders/math/df64_core.wgsl");

    let src = format!(
        "{enable}\
{df64_core}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(1)
fn main() {{
    // Sum 4 f64 values via DF64 arithmetic
    var acc = df64_zero();
    for (var i = 0u; i < 4u; i++) {{
        acc = df64_add(acc, df64_from_f64(input[i]));
    }}
    output[0] = df64_to_f64(acc);
}}"
    );

    let label = if with_enable_f64 {
        "df64_sum WITH enable f64"
    } else {
        "df64_sum WITHOUT enable f64"
    };

    match run_f64_shader(dev, &src, &[1.0_f64, 2.0, 3.0, 4.0], 1, label) {
        Ok(result) => {
            println!("  {label}: {:.6} (expected 10.0)", result[0]);
        }
        Err(e) => println!("  {label}: ERROR {e}"),
    }
}

fn test_variance_via_compile_shader_f64(dev: &Arc<barracuda::device::WgpuDevice>) {
    let df64_core =
        include_str!("../../../barraCuda/crates/barracuda/src/shaders/math/df64_core.wgsl");
    let df64_shader = include_str!(
        "../../../barraCuda/crates/barracuda/src/shaders/reduce/mean_variance_df64.wgsl"
    );

    let combined = format!("enable f64;\n{df64_core}\n{df64_shader}");

    // Path A: raw compilation (what create_f64_data_pipeline does)
    let module_raw = dev
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raw"),
            source: wgpu::ShaderSource::Wgsl(combined.as_str().into()),
        });

    // Path B: compile_shader_f64 (strips enable f64, applies driver patches)
    let module_patched = dev.compile_shader_f64(&combined, Some("patched"));

    println!("  Raw module created: OK");
    println!("  Patched module created: OK");

    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];

    let result_raw = dispatch_variance(dev, &module_raw, &data, "raw");
    match &result_raw {
        Ok([m, v]) => println!("  Raw:     mean={m:.6}, var={v:.6} (expected 3.0, 2.0)"),
        Err(e) => println!("  Raw:     ERROR {e}"),
    }

    let result_patched = dispatch_variance(dev, &module_patched, &data, "patched");
    match &result_patched {
        Ok([m, v]) => println!("  Patched: mean={m:.6}, var={v:.6} (expected 3.0, 2.0)"),
        Err(e) => println!("  Patched: ERROR {e}"),
    }

    // Path C: strip enable f64 manually, raw compile (no driver patches)
    let stripped = combined
        .lines()
        .filter(|l| l.trim() != "enable f64;")
        .collect::<Vec<_>>()
        .join("\n");
    let module_stripped = dev
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stripped_raw"),
            source: wgpu::ShaderSource::Wgsl(stripped.as_str().into()),
        });
    let result_stripped = dispatch_variance(dev, &module_stripped, &data, "stripped_raw");
    match &result_stripped {
        Ok([m, v]) => println!("  Stripped: mean={m:.6}, var={v:.6} (expected 3.0, 2.0)"),
        Err(e) => println!("  Stripped: ERROR {e}"),
    }
}

fn dispatch_variance(
    dev: &Arc<barracuda::device::WgpuDevice>,
    module: &wgpu::ShaderModule,
    data: &[f64],
    label: &str,
) -> Result<[f64; 2], String> {
    use wgpu::util::DeviceExt;

    let input_buf = dev
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let output_buf = dev
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&[0.0_f64; 2]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n: u32,
        ddof: u32,
        _pad0: u32,
        _pad1: u32,
    }

    let params = Params {
        n: data.len() as u32,
        ddof: 0,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = dev
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = dev
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let bg = dev.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let pl = dev
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

    let pipeline = dev
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pl),
            module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

    let mut encoder = dev
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    let staging = dev.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, 16);

    let cmd = encoder.finish();
    dev.queue().submit(Some(cmd));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    let _ = dev.device().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(10)),
    });
    rx.recv()
        .map_err(|e| format!("channel: {e}"))?
        .map_err(|e| format!("map: {e}"))?;

    let data = slice.get_mapped_range();
    let vals: &[f64] = bytemuck::cast_slice(&data);
    Ok([vals[0], vals[1]])
}

fn run_f64_shader(
    dev: &Arc<barracuda::device::WgpuDevice>,
    src: &str,
    input: &[f64],
    output_count: usize,
    label: &str,
) -> Result<Vec<f64>, String> {
    use wgpu::util::DeviceExt;

    let module = dev
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });

    let input_buf = dev
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let output_bytes = (output_count * std::mem::size_of::<f64>()) as u64;
    let zeros = vec![0.0_f64; output_count];
    let output_buf = dev
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    let bgl = dev
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let bg = dev.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    let pl = dev
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

    let pipeline = dev
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

    let mut encoder = dev
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    let staging = dev.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, output_bytes);

    let cmd = encoder.finish();
    dev.queue().submit(Some(cmd));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    let _ = dev.device().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(10)),
    });
    rx.recv()
        .map_err(|e| format!("channel: {e}"))?
        .map_err(|e| format!("map: {e}"))?;

    let data = slice.get_mapped_range();
    Ok(bytemuck::cast_slice::<u8, f64>(&data).to_vec())
}
