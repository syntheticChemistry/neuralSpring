// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline: swarm_nn_scores → `mean_reduce` → scalar readback (Paper 015).
//!
//! Chains two shader stages in a single `CommandEncoder` with zero CPU round-trips.
//! Stage 1: `swarm_nn_forward_scores` — max output activation per (controller, eval).
//! Stage 2: `mean_reduce` — scores array to scalar mean.
//!
//! ## Pipeline
//!
//! ```text
//! Upload params + inputs (once)
//!   ↓
//! ┌─────────────────────────────────────────────────────┐
//! │  Stage 1: swarm_nn_scores.wgsl                      │
//! │    NN forward → scores[n_controllers × n_evals]      │
//! │                                                     │
//! │  Stage 2: mean_reduce.wgsl                           │
//! │    scores[] → mean_score (scalar)                     │
//! └─────────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — NO CPU round-trip)
//! Readback: 4 bytes (one f32 scalar)
//! ```
//!
//! ## Provenance
//!
//! GPU pipeline: swarm_nn_scores → mean_reduce.
//! Validates: mean of tanh-like (sigmoid) output activations across swarm controllers.

#![allow(
    clippy::cast_precision_loss,
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
use neural_spring::rng::Rng;
use neural_spring::swarm_robotics::{create_controller, neural_forward_max_score, ControllerType};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const SWARM_WGSL: &str = include_str!("../../metalForge/shaders/swarm_nn_scores.wgsl");
const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SwarmConfig {
    n_controllers: u32,
    n_evals: u32,
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

    let mut h = ValidationHarness::new("gpu_pipeline_swarm");

    validate_swarm_small(&mut h, &gpu);
    validate_swarm_larger(&mut h, &gpu);
    validate_swarm_single_controller(&mut h, &gpu);
    validate_swarm_random_inputs(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_swarm_scores(
    params: &[f32],
    inputs: &[f32],
    n_controllers: usize,
    n_evals: usize,
) -> f32 {
    let mut total = 0.0_f32;
    for ctrl in 0..n_controllers {
        let ctrl_params: Vec<f64> = params[ctrl * 33..(ctrl + 1) * 33]
            .iter()
            .map(|&x| f64::from(x))
            .collect();
        for input in inputs.iter().take(n_evals) {
            total += neural_forward_max_score(&ctrl_params, f64::from(*input)) as f32;
        }
    }
    total / (n_controllers * n_evals) as f32
}

// ── GPU chained pipeline ───────────────────────────────────────────

fn gpu_mean_swarm_scores(
    gpu: &Gpu,
    params: &[f32],
    inputs: &[f32],
    n_controllers: u32,
    n_evals: u32,
) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();
    let n_total = (n_controllers * n_evals) as usize;

    let swarm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_swarm"),
        source: wgpu::ShaderSource::Wgsl(SWARM_WGSL.into()),
    });

    let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_swarm_reduce"),
        source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
    });

    let swarm_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_swarm_bgl"),
        entries: &[
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            uniform_entry(3),
        ],
    });

    let swarm_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_swarm_pl"),
        bind_group_layouts: &[&swarm_bgl],
        push_constant_ranges: &[],
    });

    let swarm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_swarm_pipeline"),
        layout: Some(&swarm_pl),
        module: &swarm_shader,
        entry_point: "swarm_nn_forward_scores",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_swarm_reduce_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });

    let reduce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_swarm_reduce_pl"),
        bind_group_layouts: &[&reduce_bgl],
        push_constant_ranges: &[],
    });

    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_swarm_reduce_pipeline"),
        layout: Some(&reduce_pl),
        module: &reduce_shader,
        entry_point: "mean_reduce",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_swarm_params"),
        contents: bytemuck::cast_slice(params),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let inputs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_swarm_inputs"),
        contents: bytemuck::cast_slice(inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let scores_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_swarm_scores"),
        size: (n_total * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let config = SwarmConfig {
        n_controllers,
        n_evals,
    };
    let config_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_swarm_config"),
        contents: bytemuck::bytes_of(&config),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_swarm_mean_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let reduce_params = ReduceParams { n: n_total as u32 };
    let reduce_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_swarm_reduce_params"),
        contents: bytemuck::bytes_of(&reduce_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let swarm_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_swarm_bg"),
        layout: &swarm_bgl,
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
                resource: scores_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: config_buf.as_entire_binding(),
            },
        ],
    });

    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_swarm_reduce_bg"),
        layout: &reduce_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scores_buf.as_entire_binding(),
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
        label: Some("chain_swarm_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_swarm_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&swarm_pipeline);
        pass.set_bind_group(0, &swarm_bg, &[]);
        pass.dispatch_workgroups(n_total.div_ceil(256) as u32, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_swarm_reduce_pass"),
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

fn validate_swarm_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_controllers = 4_usize;
    let n_evals = 5_usize;
    let mut rng = Rng::new(42);
    let controllers: Vec<_> = (0..n_controllers)
        .map(|_| create_controller(ControllerType::NeuralNet, &mut rng))
        .collect();
    let params: Vec<f32> = controllers
        .iter()
        .flat_map(|c| c.params.iter().map(|&p| p as f32))
        .collect();
    let inputs: Vec<f32> = (0..n_evals)
        .map(|i| (i as f32 + 0.5) / n_evals as f32)
        .collect();

    let cpu_mean = cpu_mean_swarm_scores(&params, &inputs, n_controllers, n_evals);

    match gpu_mean_swarm_scores(gpu, &params, &inputs, n_controllers as u32, n_evals as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("swarm small 4×5: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("swarm small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_swarm_larger(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_controllers = 10_usize;
    let n_evals = 8_usize;
    let mut rng = Rng::new(777);
    let controllers: Vec<_> = (0..n_controllers)
        .map(|_| create_controller(ControllerType::NeuralNet, &mut rng))
        .collect();
    let params: Vec<f32> = controllers
        .iter()
        .flat_map(|c| c.params.iter().map(|&p| p as f32))
        .collect();
    let inputs: Vec<f32> = (0..n_evals)
        .map(|i| (i as f32 + 0.5) / n_evals as f32)
        .collect();

    let cpu_mean = cpu_mean_swarm_scores(&params, &inputs, n_controllers, n_evals);

    match gpu_mean_swarm_scores(gpu, &params, &inputs, n_controllers as u32, n_evals as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("swarm larger 10×8: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("swarm larger: dispatch failed — {e}"), false);
        }
    }
}

fn validate_swarm_single_controller(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_controllers = 1_usize;
    let n_evals = 6_usize;
    let mut rng = Rng::new(123);
    let ctrl = create_controller(ControllerType::NeuralNet, &mut rng);
    let params: Vec<f32> = ctrl.params.iter().map(|&p| p as f32).collect();
    let inputs: Vec<f32> = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

    let cpu_mean = cpu_mean_swarm_scores(&params, &inputs, n_controllers, n_evals);

    match gpu_mean_swarm_scores(gpu, &params, &inputs, n_controllers as u32, n_evals as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("swarm single ctrl 1×6: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("swarm single ctrl: dispatch failed — {e}"), false);
        }
    }
}

fn validate_swarm_random_inputs(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_controllers = 3_usize;
    let n_evals = 4_usize;
    let mut rng = Rng::new(555);
    let controllers: Vec<_> = (0..n_controllers)
        .map(|_| create_controller(ControllerType::NeuralNet, &mut rng))
        .collect();
    let params: Vec<f32> = controllers
        .iter()
        .flat_map(|c| c.params.iter().map(|&p| p as f32))
        .collect();
    let inputs: Vec<f32> = (0..n_evals).map(|_| rng.uniform() as f32).collect();

    let cpu_mean = cpu_mean_swarm_scores(&params, &inputs, n_controllers, n_evals);

    match gpu_mean_swarm_scores(gpu, &params, &inputs, n_controllers as u32, n_evals as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("swarm random inputs 3×4: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::TENSOR_TRANSCENDENTAL_F32,
            );
        }
        Err(e) => {
            h.check_bool(
                &format!("swarm random inputs: dispatch failed — {e}"),
                false,
            );
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_controllers = 5_usize;
    let n_evals = 4_usize;
    let mut rng = Rng::new(99);
    let controllers: Vec<_> = (0..n_controllers)
        .map(|_| create_controller(ControllerType::NeuralNet, &mut rng))
        .collect();
    let params: Vec<f32> = controllers
        .iter()
        .flat_map(|c| c.params.iter().map(|&p| p as f32))
        .collect();
    let inputs: Vec<f32> = vec![0.1, 0.3, 0.7, 0.9];

    let r1 = gpu_mean_swarm_scores(gpu, &params, &inputs, n_controllers as u32, n_evals as u32);
    let r2 = gpu_mean_swarm_scores(gpu, &params, &inputs, n_controllers as u32, n_evals as u32);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("swarm determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("swarm determinism: dispatch failed", false);
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
