// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-dispatch validation: Phase 4e domain shaders (GPU ↔ CPU parity).
//!
//! Validates GPU ↔ CPU parity for pairwise L2 (MODES Paper 012),
//! `multi_obj_fitness` (Directed Evolution Paper 014), `swarm_nn_forward`
//! (Swarm Robotics Paper 015), and `hill_gate` (Signal Integration Paper 021).
//!
//! ## Evolution path
//!
//! ```text
//! GPU-only (validate_gpu_modes, validate_gpu_directed, ...)
//!   → Cross-dispatch GPU ↔ CPU (this binary)
//!   → metalForge cross-system (GPU → NPU → CPU)
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

use barracuda::dispatch::{dispatch_for, DispatchTarget};
use bytemuck::{Pod, Zeroable};
use neural_spring::directed_evolution::multi_objective_fitness;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::signal_integration::two_input_hill;
use neural_spring::swarm_robotics::neural_forward;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const PAIRWISE_L2_WGSL: &str = include_str!("../../metalForge/shaders/pairwise_l2.wgsl");
const MULTI_OBJ_FITNESS_WGSL: &str =
    include_str!("../../metalForge/shaders/multi_obj_fitness.wgsl");
const SWARM_NN_WGSL: &str = include_str!("../../metalForge/shaders/swarm_nn_forward.wgsl");
const HILL_GATE_WGSL: &str = include_str!("../../metalForge/shaders/hill_gate.wgsl");

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

    let mut h = ValidationHarness::new("cross_dispatch_phase4e");

    validate_dispatch_routing(&mut h);
    validate_pairwise_l2_parity(&mut h, &gpu);
    validate_multi_obj_fitness_parity(&mut h, &gpu);
    validate_swarm_nn_parity(&mut h, &gpu);
    validate_hill_gate_parity(&mut h, &gpu);

    h.finish();
}

// ── Dispatch routing ─────────────────────────────────────────────

fn validate_dispatch_routing(h: &mut ValidationHarness) {
    let small_pairwise = dispatch_for("pairwise_l2", 20);
    let large_pairwise = dispatch_for("pairwise_l2", 10_000);
    let small_multi_obj = dispatch_for("multi_obj_fitness", 50);
    let large_multi_obj = dispatch_for("multi_obj_fitness", 10_000);
    let small_swarm = dispatch_for("swarm_nn", 20);
    let large_swarm = dispatch_for("swarm_nn", 10_000);
    let small_hill = dispatch_for("hill_gate", 100);
    let large_hill = dispatch_for("hill_gate", 10_000);

    h.check_bool(
        &format!("dispatch: pairwise_l2(20) → {small_pairwise:?}"),
        matches!(small_pairwise, DispatchTarget::Cpu),
    );
    h.check_bool(
        &format!("dispatch: pairwise_l2(10k) → {large_pairwise:?}"),
        matches!(large_pairwise, DispatchTarget::Gpu),
    );
    h.check_bool(
        &format!("dispatch: multi_obj_fitness(50) → {small_multi_obj:?}"),
        matches!(small_multi_obj, DispatchTarget::Cpu),
    );
    h.check_bool(
        &format!("dispatch: multi_obj_fitness(10k) → {large_multi_obj:?}"),
        matches!(large_multi_obj, DispatchTarget::Gpu),
    );
    h.check_bool(
        &format!("dispatch: swarm_nn(20) → {small_swarm:?}"),
        matches!(small_swarm, DispatchTarget::Cpu),
    );
    h.check_bool(
        &format!("dispatch: swarm_nn(10k) → {large_swarm:?}"),
        matches!(large_swarm, DispatchTarget::Gpu),
    );
    h.check_bool(
        &format!("dispatch: hill_gate(100) → {small_hill:?}"),
        matches!(small_hill, DispatchTarget::Cpu),
    );
    h.check_bool(
        &format!("dispatch: hill_gate(10k) → {large_hill:?}"),
        matches!(large_hill, DispatchTarget::Gpu),
    );
}

// ── Pairwise L2 parity (MODES Paper 012) ─────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PairwiseL2Params {
    n: u32,
    dim: u32,
}

fn cpu_pairwise_l2(features: &[f32], n: usize, dim: usize) -> Vec<f64> {
    let n_pairs = n * (n - 1) / 2;
    let mut distances = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sum_sq = 0.0_f64;
            for d in 0..dim {
                let a = f64::from(features[i * dim + d]);
                let b = f64::from(features[j * dim + d]);
                let diff = a - b;
                sum_sq += diff * diff;
            }
            distances.push(sum_sq.sqrt());
        }
    }
    distances
}

fn gpu_pairwise_l2(gpu: &Gpu, features: &[f32], n: u32, dim: u32) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xd_pairwise_l2"),
        source: wgpu::ShaderSource::Wgsl(PAIRWISE_L2_WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("xd_pairwise_l2_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("xd_pairwise_l2_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("xd_pairwise_l2_pipe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "pairwise_l2",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let n_pairs = (n * (n - 1) / 2) as usize;
    let features_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_features"),
        contents: bytemuck::cast_slice(features),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = PairwiseL2Params { n, dim };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_pairwise_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xd_pairwise_l2_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: features_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dist_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("xd_pairwise_l2_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("xd_pairwise_l2_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n_pairs as u32).div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&dist_buf, n_pairs)
}

fn validate_pairwise_l2_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(42);
    let n = 10_usize;
    let dim = 8_usize;
    let features: Vec<f32> = (0..n * dim).map(|_| rng.uniform() as f32).collect();

    let cpu_dist = cpu_pairwise_l2(&features, n, dim);

    match gpu_pairwise_l2(gpu, &features, n as u32, dim as u32) {
        Ok(gpu_dist) => {
            let max_diff: f64 = gpu_dist
                .iter()
                .zip(cpu_dist.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "pairwise_l2 parity (n=10, dim=8): max diff {max_diff:.2e}, {} pairs",
                    gpu_dist.len()
                ),
                max_diff,
                tolerances::GPU_MODES_L2_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("pairwise_l2 parity: failed — {e}"), false);
        }
    }
}

// ── Multi-objective fitness parity (Directed Evolution Paper 014) ─

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MultiObjParams {
    pop_size: u32,
    genome_len: u32,
    n_objectives: u32,
    _pad: u32,
}

fn cpu_multi_obj_fitness(
    population: &[f64],
    pop_size: usize,
    genome_len: usize,
    n_objectives: usize,
) -> Vec<f64> {
    let mut fitness = Vec::with_capacity(pop_size * n_objectives);
    for i in 0..pop_size {
        let genotype = &population[i * genome_len..(i + 1) * genome_len];
        fitness.extend(multi_objective_fitness(genotype, n_objectives));
    }
    fitness
}

fn gpu_multi_obj_fitness(
    gpu: &Gpu,
    genotypes: &[f32],
    pop_size: u32,
    genome_len: u32,
    n_objectives: u32,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xd_multi_obj"),
        source: wgpu::ShaderSource::Wgsl(MULTI_OBJ_FITNESS_WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("xd_multi_obj_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("xd_multi_obj_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("xd_multi_obj_pipe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "multi_obj_fitness",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let n_fitness = (pop_size * n_objectives) as usize;
    let genotypes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_genotypes"),
        contents: bytemuck::cast_slice(genotypes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let fitness_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_fitness"),
        size: (n_fitness * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = MultiObjParams {
        pop_size,
        genome_len,
        n_objectives,
        _pad: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_multi_obj_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xd_multi_obj_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: genotypes_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: fitness_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("xd_multi_obj_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("xd_multi_obj_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n_fitness as u32).div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&fitness_buf, n_fitness)
}

fn validate_multi_obj_fitness_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(42);
    let pop_size = 8_usize;
    let genome_len = 12_usize;
    let n_objectives = 3_usize;

    let population_f64: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let genotypes_f32: Vec<f32> = population_f64.iter().map(|&x| x as f32).collect();

    let cpu_fitness = cpu_multi_obj_fitness(&population_f64, pop_size, genome_len, n_objectives);

    match gpu_multi_obj_fitness(
        gpu,
        &genotypes_f32,
        pop_size as u32,
        genome_len as u32,
        n_objectives as u32,
    ) {
        Ok(gpu_fitness) => {
            let max_diff: f64 = gpu_fitness
                .iter()
                .zip(cpu_fitness.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "multi_obj_fitness parity (8×12×3): max diff {max_diff:.2e}, {} outputs",
                    gpu_fitness.len()
                ),
                max_diff,
                tolerances::GPU_MULTI_OBJ_FITNESS_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("multi_obj_fitness parity: failed — {e}"), false);
        }
    }
}

// ── Swarm NN forward parity (Swarm Robotics Paper 015) ───────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SwarmConfig {
    n_controllers: u32,
    n_evals: u32,
}

fn read_buffer_u32(gpu: &Gpu, buf: &wgpu::Buffer, count: usize) -> Result<Vec<u32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_u32"),
        size: (count * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, (count * 4) as u64);
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("{e:?}"))?;
    let data = slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(result)
}

fn gpu_swarm_nn_forward(
    gpu: &Gpu,
    params: &[f32],
    inputs: &[f32],
    n_controllers: u32,
    n_evals: u32,
) -> Result<Vec<u32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xd_swarm_nn"),
        source: wgpu::ShaderSource::Wgsl(SWARM_NN_WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("xd_swarm_nn_bgl"),
        entries: &[
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            uniform_entry(3),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("xd_swarm_nn_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("xd_swarm_nn_pipe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "swarm_nn_forward",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let n_actions = (n_controllers * n_evals) as usize;
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_swarm_params"),
        contents: bytemuck::cast_slice(params),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let inputs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_swarm_inputs"),
        contents: bytemuck::cast_slice(inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let actions_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_swarm_actions"),
        size: (n_actions * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let config = SwarmConfig {
        n_controllers,
        n_evals,
    };
    let config_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_swarm_config"),
        contents: bytemuck::bytes_of(&config),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xd_swarm_nn_bg"),
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
        label: Some("xd_swarm_nn_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("xd_swarm_nn_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n_actions as u32).div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    read_buffer_u32(gpu, &actions_buf, n_actions)
}

fn validate_swarm_nn_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    const PARAMS_PER_CTRL: usize = 33;
    let mut rng = Rng::new(42);
    let n_controllers = 4_usize;
    let n_evals = 5_usize;

    let params_f64: Vec<f64> = (0..n_controllers * PARAMS_PER_CTRL)
        .map(|_| rng.uniform())
        .collect();
    let params_f32: Vec<f32> = params_f64.iter().map(|&x| x as f32).collect();
    let inputs_f32: Vec<f32> = (0..n_evals)
        .map(|i| (i as f32 + 0.5) / (n_evals as f32))
        .collect();

    let mut cpu_actions = Vec::with_capacity(n_controllers * n_evals);
    for ctrl in 0..n_controllers {
        let ctrl_params = &params_f64[ctrl * PARAMS_PER_CTRL..(ctrl + 1) * PARAMS_PER_CTRL];
        for eval in 0..n_evals {
            let sense = f64::from(inputs_f32[eval]);
            cpu_actions.push(neural_forward(ctrl_params, sense) as u32);
        }
    }

    match gpu_swarm_nn_forward(
        gpu,
        &params_f32,
        &inputs_f32,
        n_controllers as u32,
        n_evals as u32,
    ) {
        Ok(gpu_actions) => {
            let mismatches = gpu_actions
                .iter()
                .zip(cpu_actions.iter())
                .filter(|(&g, &c)| g != c)
                .count();

            h.check_bool(
                &format!(
                    "swarm_nn parity (4×5): {} mismatches / {} (exact match)",
                    mismatches,
                    gpu_actions.len()
                ),
                mismatches == 0,
            );

            let total = gpu_actions.len();
            h.check_bool(
                &format!("swarm_nn: correct output count ({total})"),
                total == n_controllers * n_evals,
            );
        }
        Err(e) => {
            h.check_bool(&format!("swarm_nn parity: failed — {e}"), false);
        }
    }
}

// ── Hill gate parity (Signal Integration Paper 021) ────────────────

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

fn gpu_hill_gate(
    gpu: &Gpu,
    cdg_grid: &[f32],
    ai_grid: &[f32],
    params: &HillParams,
) -> Result<Vec<f32>, String> {
    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xd_hill_gate"),
        source: wgpu::ShaderSource::Wgsl(HILL_GATE_WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("xd_hill_gate_bgl"),
        entries: &[
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            uniform_entry(3),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("xd_hill_gate_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("xd_hill_gate_pipe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "hill_gate",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let n_total = (params.nx * params.ny) as usize;
    let cdg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_cdg_grid"),
        contents: bytemuck::cast_slice(cdg_grid),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let ai_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_ai_grid"),
        contents: bytemuck::cast_slice(ai_grid),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_hill_output"),
        size: (n_total * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_hill_params"),
        contents: bytemuck::bytes_of(params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xd_hill_gate_bg"),
        layout: &bgl,
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
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("xd_hill_gate_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("xd_hill_gate_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n_total as u32).div_ceil(256), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    gpu.read_buffer_f32(&output_buf, n_total)
}

fn validate_hill_gate_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(42);
    let nx = 8_usize;
    let ny = 6_usize;
    let vmax = 1.0_f64;
    let k1 = 0.5_f64;
    let k2 = 0.3_f64;
    let n1 = 2.0_f64;
    let n2 = 2.0_f64;

    let cdg_cpu: Vec<f64> = (0..nx).map(|_| rng.uniform().mul_add(2.0, 0.01)).collect();
    let ai_cpu: Vec<f64> = (0..ny).map(|_| rng.uniform().mul_add(2.0, 0.01)).collect();

    let cpu_out = cpu_hill_grid(&cdg_cpu, &ai_cpu, vmax, k1, k2, n1, n2);

    let cdg_f32: Vec<f32> = cdg_cpu.iter().map(|&x| x as f32).collect();
    let ai_f32: Vec<f32> = ai_cpu.iter().map(|&x| x as f32).collect();
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

    match gpu_hill_gate(gpu, &cdg_f32, &ai_f32, &params) {
        Ok(gpu_out) => {
            let max_diff: f64 = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(&g, &c)| (f64::from(g) - c).abs())
                .fold(0.0_f64, f64::max);

            h.check_upper(
                &format!(
                    "hill_gate parity (8×6): max diff {max_diff:.2e}, {} cells",
                    gpu_out.len()
                ),
                max_diff,
                tolerances::GPU_HILL_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("hill_gate parity: failed — {e}"), false);
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
