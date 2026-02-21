// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline: HMM forward → `mean_reduce` → scalar readback (Papers 016-018).
//!
//! Chains two shader stages in a single `CommandEncoder` with zero CPU round-trips.
//! Stage 1: T dispatches of `hmm_forward_log` (ping-pong alpha buffers).
//! Stage 2: `mean_reduce` on final alpha.
//!
//! ## Pipeline
//!
//! ```text
//! Upload log_trans, log_emit[T], initial alpha (once)
//!   ↓
//! ┌─────────────────────────────────────────────────────┐
//! │  Stage 1: hmm_forward_log × T (ping-pong)          │
//! │    alpha_prev, log_trans, log_emit_t → alpha_curr    │
//! │                                                     │
//! │  Stage 2: mean_reduce.wgsl                           │
//! │    final_alpha[N] → mean_log_prob (scalar)          │
//! └─────────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — NO CPU round-trip)
//! Readback: 4 bytes (one f32 scalar)
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::explicit_iter_loop
)]

use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const HMM_WGSL: &str = include_str!("../../metalForge/shaders/hmm_forward_log.wgsl");
const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HmmParams {
    n_states: u32,
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

    let mut h = ValidationHarness::new("gpu_pipeline_hmm");

    validate_small(&mut h, &gpu);
    validate_larger(&mut h, &gpu);
    validate_single_state(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_hmm_forward_log_mean(
    log_trans: &[f32],
    log_emit: &[Vec<f32>],
    initial_log_alpha: &[f32],
    n_states: usize,
) -> f32 {
    let mut alpha = initial_log_alpha.to_vec();
    for t_emit in log_emit {
        let mut new_alpha = vec![0.0_f32; n_states];
        for j in 0..n_states {
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..n_states {
                let v = alpha[i] + log_trans[i * n_states + j];
                if v > max_val {
                    max_val = v;
                }
            }
            let mut sum_exp = 0.0_f32;
            for i in 0..n_states {
                let v = alpha[i] + log_trans[i * n_states + j];
                sum_exp += (v - max_val).exp();
            }
            new_alpha[j] = max_val + sum_exp.ln() + t_emit[j];
        }
        alpha = new_alpha;
    }
    alpha.iter().sum::<f32>() / n_states as f32
}

// ── GPU chained pipeline ───────────────────────────────────────────

fn gpu_hmm_forward_log_mean(
    gpu: &Gpu,
    log_trans: &[f32],
    log_emit: &[Vec<f32>],
    initial_log_alpha: &[f32],
    n_states: u32,
) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();
    let t_steps = log_emit.len();

    // Shader modules
    let hmm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_hmm"),
        source: wgpu::ShaderSource::Wgsl(HMM_WGSL.into()),
    });

    let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_reduce"),
        source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
    });

    // HMM bind group layout
    let hmm_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_hmm_bgl"),
        entries: &[
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_ro_entry(2),
            storage_rw_entry(3),
            uniform_entry(4),
        ],
    });

    let hmm_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_hmm_pl"),
        bind_group_layouts: &[&hmm_bgl],
        push_constant_ranges: &[],
    });

    let hmm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_hmm_pipeline"),
        layout: Some(&hmm_pl),
        module: &hmm_shader,
        entry_point: "hmm_forward_log",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Reduce bind group layout
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

    // Buffers: alpha ping-pong
    let n = n_states as usize;
    let mut alpha_a = initial_log_alpha.to_vec();
    alpha_a.resize(n, f32::NEG_INFINITY);
    let alpha_a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_alpha_a"),
        contents: bytemuck::cast_slice(&alpha_a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let alpha_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_alpha_b"),
        size: u64::from(n_states) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let log_trans_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_log_trans"),
        contents: bytemuck::cast_slice(log_trans),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let hmm_params = HmmParams { n_states };
    let hmm_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_hmm_params"),
        contents: bytemuck::bytes_of(&hmm_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Per-timestep log_emit buffers
    let log_emit_bufs: Vec<wgpu::Buffer> = log_emit
        .iter()
        .map(|emit| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chain_log_emit"),
                contents: bytemuck::cast_slice(emit),
                usage: wgpu::BufferUsages::STORAGE,
            })
        })
        .collect();

    // Bind groups for each HMM step
    let mut hmm_bind_groups = Vec::with_capacity(t_steps);
    for t in 0..t_steps {
        let (alpha_prev, alpha_curr) = if t % 2 == 0 {
            (&alpha_a_buf, &alpha_buf_b)
        } else {
            (&alpha_buf_b, &alpha_a_buf)
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chain_hmm_bg"),
            layout: &hmm_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: alpha_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: log_trans_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: log_emit_bufs[t].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: alpha_curr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: hmm_params_buf.as_entire_binding(),
                },
            ],
        });
        hmm_bind_groups.push(bg);
    }

    // Result and reduce buffers
    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_mean_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let reduce_params = ReduceParams { n: n_states };
    let reduce_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_reduce_params"),
        contents: bytemuck::bytes_of(&reduce_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let final_alpha = if t_steps.is_multiple_of(2) {
        &alpha_a_buf
    } else {
        &alpha_buf_b
    };

    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_reduce_bg"),
        layout: &reduce_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: final_alpha.as_entire_binding(),
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

    // Single CommandEncoder: T HMM passes + 1 reduce pass
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("chain_encoder"),
    });

    for t in 0..t_steps {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_hmm_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&hmm_pipeline);
        pass.set_bind_group(0, &hmm_bind_groups[t], &[]);
        pass.dispatch_workgroups(n_states.div_ceil(256), 1, 1);
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

fn make_log_trans(n: usize, rng: &mut Rng) -> Vec<f32> {
    let mut trans = vec![0.0_f32; n * n];
    for i in 0..n {
        let mut row_sum = 0.0_f32;
        for j in 0..n {
            let v = rng.uniform() as f32 + 0.1;
            trans[i * n + j] = v;
            row_sum += v;
        }
        for j in 0..n {
            trans[i * n + j] = (trans[i * n + j] / row_sum).ln();
        }
    }
    trans
}

fn validate_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_states = 3_usize;
    let t_steps = 5_usize;
    let mut rng = Rng::new(42);

    let log_trans = make_log_trans(n_states, &mut rng);

    let mut log_emit = Vec::with_capacity(t_steps);
    for _ in 0..t_steps {
        let mut row = vec![0.0_f32; n_states];
        let mut sum = 0.0_f32;
        for v in &mut row {
            *v = rng.uniform() as f32 + 0.1;
            sum += *v;
        }
        for v in &mut row {
            *v = (*v / sum).ln();
        }
        log_emit.push(row);
    }

    let initial: Vec<f32> = (0..n_states)
        .map(|_| (1.0 / n_states as f32).ln())
        .collect();

    let cpu_mean = cpu_hmm_forward_log_mean(&log_trans, &log_emit, &initial, n_states);

    match gpu_hmm_forward_log_mean(gpu, &log_trans, &log_emit, &initial, n_states as u32) {
        Ok(gpu_mean) => {
            h.check_bool(
                &format!("HMM small: GPU mean finite ({gpu_mean:.6})"),
                gpu_mean.is_finite(),
            );
            h.check_abs(
                &format!("HMM small: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_HMM_ALPHA_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("HMM small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_states = 4_usize;
    let t_steps = 20_usize;
    let mut rng = Rng::new(777);

    let log_trans = make_log_trans(n_states, &mut rng);

    let mut log_emit = Vec::with_capacity(t_steps);
    for _ in 0..t_steps {
        let mut row = vec![0.0_f32; n_states];
        let mut sum = 0.0_f32;
        for v in &mut row {
            *v = rng.uniform() as f32 + 0.1;
            sum += *v;
        }
        for v in &mut row {
            *v = (*v / sum).ln();
        }
        log_emit.push(row);
    }

    let initial: Vec<f32> = (0..n_states)
        .map(|_| (1.0 / n_states as f32).ln())
        .collect();

    let cpu_mean = cpu_hmm_forward_log_mean(&log_trans, &log_emit, &initial, n_states);

    match gpu_hmm_forward_log_mean(gpu, &log_trans, &log_emit, &initial, n_states as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("HMM larger: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_HMM_ALPHA_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("HMM larger: dispatch failed — {e}"), false);
        }
    }
}

fn validate_single_state(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_states = 1_usize;
    let t_steps = 10_usize;

    let log_trans = vec![0.0_f32];
    let log_emit: Vec<Vec<f32>> = (0..t_steps).map(|_| vec![0.0_f32]).collect();
    let initial = vec![0.0_f32];

    let cpu_mean = cpu_hmm_forward_log_mean(&log_trans, &log_emit, &initial, n_states);

    match gpu_hmm_forward_log_mean(gpu, &log_trans, &log_emit, &initial, n_states as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("HMM single state: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_HMM_ALPHA_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("HMM single state: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_states = 3_usize;
    let t_steps = 5_usize;
    let mut rng = Rng::new(123);

    let log_trans = make_log_trans(n_states, &mut rng);

    let mut log_emit = Vec::with_capacity(t_steps);
    for _ in 0..t_steps {
        let mut row = vec![0.0_f32; n_states];
        let mut sum = 0.0_f32;
        for v in &mut row {
            *v = rng.uniform() as f32 + 0.1;
            sum += *v;
        }
        for v in &mut row {
            *v = (*v / sum).ln();
        }
        log_emit.push(row);
    }

    let initial: Vec<f32> = (0..n_states)
        .map(|_| (1.0 / n_states as f32).ln())
        .collect();

    let r1 = gpu_hmm_forward_log_mean(gpu, &log_trans, &log_emit, &initial, n_states as u32);
    let r2 = gpu_hmm_forward_log_mean(gpu, &log_trans, &log_emit, &initial, n_states as u32);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("HMM determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("HMM determinism: dispatch failed", false);
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
