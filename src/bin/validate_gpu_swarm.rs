// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: batch neural-net controller forward pass via `BarraCUDA` `SwarmNnGpu`.
//!
//! Validates `barracuda::ops::bio::SwarmNnGpu` against CPU
//! `swarm_robotics::neural_forward`. The GPU op evaluates many
//! controllers × sense inputs in a single dispatch.
//!
//! ## Papers validated
//!
//! - Paper 015: Swarm Robotics (neural net controller inference)
//!
//! ## Provenance
//!
//! CPU reference: `swarm_robotics::neural_forward` (seed=42, batch 10×8).
//! GPU op: `barracuda::ops::bio::SwarmNnGpu`
//! Validated on: RTX 4070 (Vulkan), llvmpipe (CPU fallback).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "validation binary"
)]

use barracuda::ops::bio::swarm_nn::SwarmNnParams;
use barracuda::ops::bio::SwarmNnGpu;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::swarm_robotics::{create_controller, neural_forward, ControllerType};
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;
use wgpu::util::DeviceExt;

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
    let _ = gpu.device().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("{e:?}"))?;
    let data = slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    Ok(result)
}

fn gpu_nn_forward(
    gpu: &Gpu,
    op: &SwarmNnGpu,
    weights: &[f64],
    sense_values: &[f64],
    n_controllers: u32,
    n_evals: u32,
) -> Result<Vec<u32>, String> {
    let device = gpu.device();

    // BarraCUDA expects inputs [ctrl, eval, dim] = (ctrl * n_evals + eval) * input_dim.
    let inputs_f64: Vec<f64> = (0..n_controllers)
        .flat_map(|_| sense_values.iter().copied())
        .collect();

    let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"),
        contents: bytemuck::cast_slice(weights),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let inputs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("inputs"),
        contents: bytemuck::cast_slice(&inputs_f64),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_actions = (n_controllers * n_evals) as usize;
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
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let op = SwarmNnGpu::new(Arc::clone(gpu.wgpu_device()));
    let mut h = ValidationHarness::new("gpu_swarm_nn");

    validate_single_controller(&mut h, &gpu, &op);
    validate_batch(&mut h, &gpu, &op);
    validate_determinism(&mut h, &gpu, &op);

    h.finish();
}

fn validate_single_controller(h: &mut ValidationHarness, gpu: &Gpu, op: &SwarmNnGpu) {
    let mut rng = Rng::new(42);
    let ctrl = create_controller(ControllerType::NeuralNet, &mut rng);

    let sense_values: Vec<f64> = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let n_evals = sense_values.len();

    match gpu_nn_forward(gpu, op, &ctrl.params, &sense_values, 1, n_evals as u32) {
        Ok(gpu_actions) => {
            for (eval_idx, &sense) in sense_values.iter().enumerate() {
                let cpu_action = neural_forward(&ctrl.params, sense);
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

fn validate_batch(h: &mut ValidationHarness, gpu: &Gpu, op: &SwarmNnGpu) {
    let n_controllers = 10_usize;
    let n_evals = 8_usize;

    let mut rng = Rng::new(123);
    let controllers: Vec<_> = (0..n_controllers)
        .map(|_| create_controller(ControllerType::NeuralNet, &mut rng))
        .collect();

    let sense_values: Vec<f64> = (0..n_evals)
        .map(|i| (i as f64 + 0.5) / (n_evals as f64))
        .collect();

    let weights_flat: Vec<f64> = controllers
        .iter()
        .flat_map(|c| c.params.iter().copied())
        .collect();

    match gpu_nn_forward(
        gpu,
        op,
        &weights_flat,
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
                    let cpu_action = neural_forward(&ctrl.params, sense);
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

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu, op: &SwarmNnGpu) {
    let n_controllers = 3_usize;
    let n_evals = 4_usize;

    let mut rng = Rng::new(99);
    let controllers: Vec<_> = (0..n_controllers)
        .map(|_| create_controller(ControllerType::NeuralNet, &mut rng))
        .collect();

    let sense_values: Vec<f64> = vec![0.1, 0.3, 0.7, 0.9];
    let weights_flat: Vec<f64> = controllers
        .iter()
        .flat_map(|c| c.params.iter().copied())
        .collect();

    let run1 = gpu_nn_forward(
        gpu,
        op,
        &weights_flat,
        &sense_values,
        n_controllers as u32,
        n_evals as u32,
    );
    let run2 = gpu_nn_forward(
        gpu,
        op,
        &weights_flat,
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
