// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: batch neural-net controller forward pass via `metalForge` WGSL shader.
//!
//! Validates `metalForge/shaders/swarm_nn_forward.wgsl` against CPU
//! `swarm_robotics::neural_forward`. The GPU shader evaluates many
//! controllers × sense inputs in a single dispatch.
//!
//! ## Papers validated
//!
//! - Paper 015: Swarm Robotics (neural net controller inference)
//!
//! ## Provenance
//!
//! CPU reference: `swarm_robotics::neural_forward` (seed=42, batch 10×8).
//! WGSL shader: `metalForge/shaders/swarm_nn_forward.wgsl`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]

use barracuda::ops::bio::swarm_nn::SwarmNnParams;
use barracuda::ops::bio::SwarmNnGpu;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::swarm_robotics::{create_controller, neural_forward, ControllerType};
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("../../metalForge/shaders/swarm_nn_forward.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Config {
    n_controllers: u32,
    n_evals: u32,
}

fn read_buffer_u32(gpu: &Gpu, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<u32>, String> {
    let staging = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (count * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 4) as u64);
    gpu.queue().submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    gpu.device().poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("{e:?}"))?;
    let data = slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    Ok(result)
}

fn gpu_nn_forward(
    gpu: &Gpu,
    params: &[f32],
    inputs: &[f32],
    n_controllers: u32,
    n_evals: u32,
) -> Result<Vec<u32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("swarm_nn_forward"),
        source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("swarm_nn_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("swarm_nn_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("swarm_nn_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "swarm_nn_forward",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(params),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let inputs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("inputs"),
        contents: bytemuck::cast_slice(inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_actions = (n_controllers * n_evals) as usize;
    let actions_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("actions"),
        size: (n_actions * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let config = Config {
        n_controllers,
        n_evals,
    };
    let config_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("config"),
        contents: bytemuck::bytes_of(&config),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("swarm_nn_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: inputs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: actions_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: config_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("swarm_nn_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("swarm_nn_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let workgroup_count = gpu.dispatch_1d(n_actions as u32, 256);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    read_buffer_u32(gpu, &actions_buf, n_actions)
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

    let mut h = ValidationHarness::new("gpu_swarm_nn");

    validate_single_controller(&mut h, &gpu);
    validate_batch(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);
    validate_upstream_parity(&mut h, &gpu);

    h.finish();
}

fn validate_single_controller(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(42);
    let ctrl = create_controller(ControllerType::NeuralNet, &mut rng);

    let sense_values: Vec<f32> = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let n_evals = sense_values.len();

    let params_f32: Vec<f32> = ctrl.params.iter().map(|&p| p as f32).collect();

    match gpu_nn_forward(gpu, &params_f32, &sense_values, 1, n_evals as u32) {
        Ok(gpu_actions) => {
            for (eval_idx, &sense) in sense_values.iter().enumerate() {
                let cpu_action = neural_forward(&ctrl.params, f64::from(sense));
                let gpu_action = gpu_actions[eval_idx];
                h.check_bool(
                    &format!(
                        "single controller sense={sense:.2}: GPU={gpu_action} vs CPU={cpu_action}"
                    ),
                    gpu_action == cpu_action as u32,
                );
            }
        }
        Err(e) => {
            h.check_bool(&format!("single controller: dispatch failed — {e}"), false);
        }
    }
}

fn validate_batch(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_controllers = 10_usize;
    let n_evals = 8_usize;

    let mut rng = Rng::new(123);
    let controllers: Vec<_> = (0..n_controllers)
        .map(|_| create_controller(ControllerType::NeuralNet, &mut rng))
        .collect();

    let sense_values: Vec<f32> = (0..n_evals)
        .map(|i| (i as f32 + 0.5) / (n_evals as f32))
        .collect();

    let params_flat: Vec<f32> = controllers
        .iter()
        .flat_map(|c| c.params.iter().map(|&p| p as f32))
        .collect();

    match gpu_nn_forward(
        gpu,
        &params_flat,
        &sense_values,
        n_controllers as u32,
        n_evals as u32,
    ) {
        Ok(gpu_actions) => {
            let expected = n_controllers * n_evals;
            h.check_bool(
                &format!(
                    "batch: correct action count ({} vs expected {expected})",
                    gpu_actions.len()
                ),
                gpu_actions.len() == expected,
            );

            let mut mismatches = 0_usize;
            for (ctrl_idx, ctrl) in controllers.iter().enumerate() {
                for (eval_idx, &sense) in sense_values.iter().enumerate() {
                    let cpu_action = neural_forward(&ctrl.params, f64::from(sense));
                    let gpu_action = gpu_actions[ctrl_idx * n_evals + eval_idx];
                    if gpu_action != cpu_action as u32 {
                        mismatches += 1;
                    }
                }
            }

            h.check_bool(
                &format!("batch 10×8: all actions match CPU ({mismatches} mismatches)"),
                mismatches == 0,
            );

            let all_valid = gpu_actions.iter().all(|&a| a < 5);
            h.check_bool("batch: all action indices in 0..5", all_valid);
        }
        Err(e) => {
            h.check_bool(&format!("batch: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_controllers = 3_usize;
    let n_evals = 4_usize;

    let mut rng = Rng::new(99);
    let controllers: Vec<_> = (0..n_controllers)
        .map(|_| create_controller(ControllerType::NeuralNet, &mut rng))
        .collect();

    let sense_values: Vec<f32> = vec![0.1, 0.3, 0.7, 0.9];
    let params_flat: Vec<f32> = controllers
        .iter()
        .flat_map(|c| c.params.iter().map(|&p| p as f32))
        .collect();

    let run1 = gpu_nn_forward(
        gpu,
        &params_flat,
        &sense_values,
        n_controllers as u32,
        n_evals as u32,
    );
    let run2 = gpu_nn_forward(
        gpu,
        &params_flat,
        &sense_values,
        n_controllers as u32,
        n_evals as u32,
    );

    match (run1, run2) {
        (Ok(r1), Ok(r2)) => {
            let identical = r1.iter().zip(r2.iter()).all(|(a, b)| *a == *b);
            h.check_bool(
                "determinism: two swarm_nn_forward runs identical",
                identical,
            );
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}

fn validate_upstream_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_controllers = 10_u32;
    let n_evals = 5_u32;
    let n_actions = (n_controllers * n_evals) as usize;
    let weights_per_ctrl = 33_u32;

    let mut rng = Rng::new(42);
    let weights: Vec<f32> = (0..n_controllers * weights_per_ctrl)
        .map(|_| (rng.uniform() as f32).mul_add(2.0, -1.0))
        .collect();
    let inputs: Vec<f32> = (0..n_controllers * n_evals)
        .map(|_| rng.uniform() as f32)
        .collect();

    let local = gpu_nn_forward(gpu, &weights, &inputs, n_controllers, n_evals);

    let dev = Arc::clone(gpu.wgpu_device());
    let device = gpu.device();
    let op = SwarmNnGpu::new(dev);
    let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"),
        contents: bytemuck::cast_slice(&weights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let inputs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("inputs"),
        contents: bytemuck::cast_slice(&inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let actions_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("actions"),
        size: (n_actions * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    op.dispatch(
        &weights_buf,
        &inputs_buf,
        &actions_buf,
        &SwarmNnParams {
            n_controllers,
            n_evals,
            input_dim: 1,
            hidden_dim: 4,
            output_dim: 5,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        },
    );

    let upstream_raw = gpu.read_buffer_f32(&actions_buf, n_actions);
    let upstream: Result<Vec<u32>, _> =
        upstream_raw.map(|f32_vec| bytemuck::cast_slice::<f32, u32>(&f32_vec).to_vec());

    match (local, upstream) {
        (Ok(l), Ok(u)) => {
            let bit_exact = l.iter().zip(u.iter()).all(|(&a, &b)| a == b);
            h.check_bool(
                &format!(
                    "upstream parity: local vs SwarmNnGpu bit-exact u32 ({} actions)",
                    l.len()
                ),
                bit_exact,
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
