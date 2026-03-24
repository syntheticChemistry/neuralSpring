// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: Xoshiro128** PRNG shader.
//!
//! Validates `metalForge/shaders/xoshiro128ss.wgsl` for:
//! - Uniformity (mean ≈ 0.5)
//! - Range [0, 1)
//! - Determinism (same seed → same output)
//! - Independence (different threads → different sequences)
//! - Multi-call (state advance produces different output)
//!
//! ## Provenance
//!
//! WGSL shader: `metalForge/shaders/xoshiro128ss.wgsl`
//! Validates: uniformity, determinism, independence (analytical).
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::rng::WGSL_XOSHIRO128SS;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const fn splitmix32(state: &mut u32) -> u32 {
    *state = state.wrapping_add(0x9E37_79B9);
    let mut z = *state;
    z = (z ^ (z >> 16)).wrapping_mul(0x85EB_CA6B);
    z = (z ^ (z >> 13)).wrapping_mul(0xC2B2_AE35);
    z ^ (z >> 16)
}

const fn seed_state(base_seed: u32, thread_id: u32) -> [u32; 4] {
    let mut sm = base_seed.wrapping_add(thread_id.wrapping_mul(2_654_435_761));
    [
        splitmix32(&mut sm),
        splitmix32(&mut sm),
        splitmix32(&mut sm),
        splitmix32(&mut sm),
    ]
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    n_threads: u32,
    n_samples: u32,
}

#[expect(clippy::too_many_lines, reason = "validation binary")]
fn gpu_generate(
    gpu: &Gpu,
    state: &[u32],
    n_threads: u32,
    n_samples: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xoshiro128ss"),
        source: wgpu::ShaderSource::Wgsl(WGSL_XOSHIRO128SS.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("prng_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("prng_pl"),
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("prng_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("generate"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("prng_state"),
        contents: bytemuck::cast_slice(state),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_len = (n_threads * n_samples) as usize;
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prng_output"),
        size: (output_len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = Params {
        n_threads,
        n_samples,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("prng_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("prng_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_buf.as_entire_binding(),
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("prng_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("prng_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(n_threads.div_ceil(256), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));

    Ok(gpu.read_buffer_f32(&output_buf, output_len)?)
}

#[tokio::main]
async fn main() {
    const N_THREADS: u32 = 1024;
    const N_SAMPLES: u32 = 100;
    const BASE_SEED: u32 = 42;

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

    let mut h = ValidationHarness::new("gpu_prng");

    // Build initial state
    let state: Vec<u32> = (0..N_THREADS)
        .flat_map(|tid| seed_state(BASE_SEED, tid))
        .collect();

    match gpu_generate(&gpu, &state, N_THREADS, N_SAMPLES) {
        Ok(output) => {
            // 1. Uniformity: mean ∈ [0.48, 0.52]
            let mean: f64 = output.iter().map(|&x| f64::from(x)).sum::<f64>() / output.len() as f64;
            h.check_abs(
                &format!("uniformity: mean={mean:.6} ∈ [0.48, 0.52]"),
                mean,
                0.5,
                tolerances::GPU_PRNG_UNIFORMITY_MEAN,
            );

            // 2. Range: all ∈ [0, 1)
            let all_in_range = output.iter().all(|&x| (0.0_f32..1.0).contains(&x));
            h.check_bool("range: all values in [0, 1)", all_in_range);

            // 3. Determinism: same seed → same output (run twice)
            let out1 = output.clone();
            let state_copy: Vec<u32> = state.clone();
            match gpu_generate(&gpu, &state_copy, N_THREADS, N_SAMPLES) {
                Ok(out2) => {
                    let identical = out1
                        .iter()
                        .zip(out2.iter())
                        .all(|(a, b)| (a - b).abs() < f32::EPSILON);
                    h.check_bool("determinism: same seed produces same output", identical);
                }
                Err(e) => {
                    h.check_bool(&format!("determinism: dispatch failed — {e}"), false);
                }
            }

            // 4. Independence: thread 0 and thread 1 have different sequences
            let thread0: Vec<f32> = output[0..N_SAMPLES as usize].to_vec();
            let thread1: Vec<f32> = output[N_SAMPLES as usize..(2 * N_SAMPLES) as usize].to_vec();
            let different = thread0
                .iter()
                .zip(thread1.iter())
                .any(|(a, b)| (a - b).abs() >= f32::EPSILON);
            h.check_bool(
                "independence: thread 0 and thread 1 produce different sequences",
                different,
            );

            // 5. Multi-call: run again with advanced state produces different output
            match gpu_generate_multi_call(&gpu, &state, N_THREADS, N_SAMPLES) {
                Ok((out_run1, out_run2)) => {
                    let different = out_run1
                        .iter()
                        .zip(out_run2.iter())
                        .any(|(a, b)| (a - b).abs() >= f32::EPSILON);
                    h.check_bool(
                        "multi-call: second call with advanced state produces different output",
                        different,
                    );
                }
                Err(e) => {
                    h.check_bool(&format!("multi-call: dispatch failed — {e}"), false);
                }
            }
        }
        Err(e) => {
            h.check_bool(&format!("gpu_generate failed — {e}"), false);
        }
    }

    h.finish();
}

/// Run the PRNG shader twice in sequence. State is updated in-place between
/// runs. Returns (output of run 1, output of run 2).
#[expect(clippy::too_many_lines, reason = "validation binary")]
fn gpu_generate_multi_call(
    gpu: &Gpu,
    initial_state: &[u32],
    n_threads: u32,
    n_samples: u32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xoshiro128ss_multi"),
        source: wgpu::ShaderSource::Wgsl(WGSL_XOSHIRO128SS.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("prng_multi_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("prng_multi_pl"),
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("prng_multi_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("generate"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let output_len = (n_threads * n_samples) as usize;

    let state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("prng_multi_state"),
        contents: bytemuck::cast_slice(initial_state),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prng_multi_output"),
        size: (output_len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = Params {
        n_threads,
        n_samples,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("prng_multi_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("prng_multi_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_buf.as_entire_binding(),
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

    // Run 1: dispatch, then readback (blocks until done)
    {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prng_enc1"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prng_pass1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_threads.div_ceil(256), 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
    }

    let out_run1 = gpu
        .read_buffer_f32(&output_buf, output_len)
        .map_err(|e| e.to_string())?;

    // Run 2: state was updated in-place; dispatch again
    {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prng_enc2"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prng_pass2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_threads.div_ceil(256), 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
    }

    let out_run2 = gpu
        .read_buffer_f32(&output_buf, output_len)
        .map_err(|e| e.to_string())?;

    Ok((out_run1, out_run2))
}
