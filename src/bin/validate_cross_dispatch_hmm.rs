// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-dispatch validation: HMM forward log-likelihood (Papers 016, 018).
//!
//! Validates GPU ↔ CPU parity for the HMM forward algorithm using
//! `hmm_forward_log.wgsl` shader vs the `neural_spring::hmm::Hmm` CPU reference.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::similar_names,
    clippy::too_many_lines
)]

use barracuda::dispatch::{dispatch_for, DispatchTarget};
use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::hmm::Hmm;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const HMM_WGSL: &str = include_str!("../../metalForge/shaders/hmm_forward_log.wgsl");

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
            eprintln!("  SKIP: {e}");
            std::process::exit(0);
        }
    };

    let mut h = ValidationHarness::new("cross_dispatch_hmm");
    validate_dispatch_routing(&mut h);
    validate_hmm_parity(&mut h, &gpu);
    h.finish();
}

// ── Dispatch routing ─────────────────────────────────────────────

fn validate_dispatch_routing(h: &mut ValidationHarness) {
    let small = dispatch_for("hmm_forward", 100);
    let large = dispatch_for("hmm_forward", 10_000);

    h.check_bool(
        &format!("dispatch: hmm_forward(100) → {small:?}"),
        matches!(small, DispatchTarget::Cpu),
    );
    h.check_bool(
        &format!("dispatch: hmm_forward(10k) → {large:?}"),
        matches!(large, DispatchTarget::Gpu),
    );
}

// ── HMM parity: GPU vs CPU ───────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HmmParams {
    n_states: u32,
}

fn gpu_hmm_forward(
    gpu: &Gpu,
    log_initial: &[f32],
    log_trans: &[f32],
    log_emissions: &[f32],
) -> Result<f32, String> {
    let n_states = log_initial.len();
    let n_obs = log_emissions.len() / n_states;
    assert_eq!(log_trans.len(), n_states * n_states);
    assert_eq!(log_emissions.len(), n_obs * n_states);

    let device = gpu.device();
    let queue = gpu.queue();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xd_hmm"),
        source: wgpu::ShaderSource::Wgsl(HMM_WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("xd_hmm_bgl"),
        entries: &[
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_ro_entry(2),
            storage_rw_entry(3),
            uniform_entry(4),
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("xd_hmm_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("xd_hmm_pipe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "hmm_forward_log",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let log_trans_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_log_trans"),
        contents: bytemuck::cast_slice(log_trans),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let params = HmmParams {
        #[allow(clippy::cast_possible_truncation)]
        n_states: n_states as u32,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_hmm_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let buf_size = (n_states * 4) as u64;
    let mut alpha_init: Vec<f32> = log_initial.to_vec();
    for (j, a) in alpha_init.iter_mut().enumerate() {
        *a += log_emissions[j];
    }

    let alpha_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xd_alpha_a"),
        contents: bytemuck::cast_slice(&alpha_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let alpha_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xd_alpha_b"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    #[allow(clippy::cast_possible_truncation)]
    let workgroup_count = (n_states as u32).div_ceil(256);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("xd_hmm_enc"),
    });

    let mut src_is_a = true;
    for t in 1..n_obs {
        let emit_offset = t * n_states;
        let log_emit_t = &log_emissions[emit_offset..emit_offset + n_states];

        let log_emit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xd_log_emit"),
            contents: bytemuck::cast_slice(log_emit_t),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let (src, dst) = if src_is_a {
            (&alpha_a, &alpha_b)
        } else {
            (&alpha_b, &alpha_a)
        };

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xd_hmm_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: log_trans_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: log_emit_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("xd_hmm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        src_is_a = !src_is_a;
    }

    queue.submit(std::iter::once(encoder.finish()));

    let final_buf = if src_is_a { alpha_a } else { alpha_b };
    let gpu_alpha = gpu.read_buffer_f32(&final_buf, n_states)?;

    let max_a = gpu_alpha.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_lik: f32 = max_a + gpu_alpha.iter().map(|&a| (a - max_a).exp()).sum::<f32>().ln();
    Ok(log_lik)
}

fn hmm_to_log_f32(hmm: &Hmm) -> (Vec<f32>, Vec<f32>) {
    let log_initial: Vec<f32> = hmm.initial.iter().map(|&p| (p as f32).ln()).collect();
    let log_trans: Vec<f32> = hmm.transition.iter().map(|&p| (p as f32).ln()).collect();
    (log_initial, log_trans)
}

fn obs_to_log_emissions(hmm: &Hmm, obs: &[usize]) -> Vec<f32> {
    let n = hmm.num_states();
    let m = hmm.num_symbols();
    obs.iter()
        .flat_map(|&o| {
            let oi = o.min(m - 1);
            (0..n).map(move |j| (hmm.emission[j * m + oi] as f32).ln())
        })
        .collect()
}

fn validate_hmm_parity(h: &mut ValidationHarness, gpu: &Gpu) {
    let mut rng = Rng::new(42);
    let n_states = 3_usize;
    let n_obs = 4_usize;
    let seq_len = 20_usize;

    let transition: Vec<Vec<f64>> = (0..n_states)
        .map(|_| {
            let row: Vec<f64> = (0..n_states).map(|_| rng.uniform()).collect();
            let sum: f64 = row.iter().sum();
            row.into_iter().map(|x| x / sum).collect()
        })
        .collect();

    let emission: Vec<Vec<f64>> = (0..n_states)
        .map(|_| {
            let row: Vec<f64> = (0..n_obs).map(|_| rng.uniform()).collect();
            let sum: f64 = row.iter().sum();
            row.into_iter().map(|x| x / sum).collect()
        })
        .collect();

    let mut initial: Vec<f64> = (0..n_states).map(|_| rng.uniform()).collect();
    let sum_init: f64 = initial.iter().sum();
    for x in &mut initial {
        *x /= sum_init;
    }

    let hmm = Hmm::new(transition, emission, initial);
    let obs: Vec<usize> = (0..seq_len).map(|_| rng.usize(n_obs)).collect();

    let (_, cpu_ll) = hmm.forward(&obs);
    let (log_initial, log_trans) = hmm_to_log_f32(&hmm);
    let log_emissions = obs_to_log_emissions(&hmm, &obs);

    match gpu_hmm_forward(gpu, &log_initial, &log_trans, &log_emissions) {
        Ok(gpu_ll) => {
            let diff = (f64::from(gpu_ll) - cpu_ll).abs();
            h.check_upper(
                &format!(
                    "HMM log-lik parity: GPU={gpu_ll:.4} vs CPU={cpu_ll:.4}, diff={diff:.2e}"
                ),
                diff,
                tolerances::GPU_HMM_LOG_LIKELIHOOD_F32,
            );
            h.check_bool(
                "HMM log-lik negative",
                cpu_ll < 0.0,
            );
        }
        Err(e) => {
            h.check_bool(&format!("HMM parity: GPU failed — {e}"), false);
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
