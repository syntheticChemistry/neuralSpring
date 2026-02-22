// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure GPU pipeline: pairwise_hamming → `mean_reduce` → scalar readback (Paper 017).
//!
//! Chains two shader stages in a single `CommandEncoder` with zero CPU round-trips.
//! Stage 1: `pairwise_hamming` — proportional Hamming for all n*(n-1)/2 pairs.
//! Stage 2: `mean_reduce` — distance array to scalar mean.
//!
//! ## Pipeline
//!
//! ```text
//! Upload sequences (once)
//!   ↓
//! ┌─────────────────────────────────────────────────────┐
//! │  Stage 1: pairwise_hamming.wgsl                     │
//! │    sequences → distances[n_pairs]                    │
//! │                                                     │
//! │  Stage 2: mean_reduce.wgsl                           │
//! │    distances[] → mean_distance (scalar)               │
//! └─────────────────────────────────────────────────────┘
//!   ↓  (single queue.submit — NO CPU round-trip)
//! Readback: 4 bytes (one f32 scalar)
//! ```
//!
//! ## Provenance
//!
//! GPU pipeline: pairwise_hamming → mean_reduce.
//! Validates: SATé alignment mean pairwise distance (Liu et al., 2009).

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::needless_range_loop
)]

use bytemuck::{Pod, Zeroable};
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

const HAMMING_WGSL: &str = include_str!("../../metalForge/shaders/pairwise_hamming.wgsl");
const REDUCE_WGSL: &str = include_str!("../../metalForge/shaders/mean_reduce.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HammingParams {
    n_seqs: u32,
    seq_len: u32,
    _pad0: u32,
    _pad1: u32,
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

    let mut h = ValidationHarness::new("gpu_pipeline_sate");

    validate_sate_small(&mut h, &gpu);
    validate_sate_larger(&mut h, &gpu);
    validate_sate_identical(&mut h, &gpu);
    validate_sate_all_differ(&mut h, &gpu);
    validate_determinism(&mut h, &gpu);

    h.finish();
}

// ── CPU reference ──────────────────────────────────────────────────

fn cpu_mean_pairwise_hamming(sequences: &[u32], n_seqs: usize, seq_len: usize) -> f32 {
    let n_pairs = n_seqs * (n_seqs - 1) / 2;
    if n_pairs == 0 {
        return 0.0;
    }
    let mut total = 0.0_f32;
    for i in 0..n_seqs {
        for j in (i + 1)..n_seqs {
            let mut diff = 0_u32;
            for s in 0..seq_len {
                if sequences[i * seq_len + s] != sequences[j * seq_len + s] {
                    diff += 1;
                }
            }
            total += diff as f32 / seq_len as f32;
        }
    }
    total / n_pairs as f32
}

// ── GPU chained pipeline ───────────────────────────────────────────

fn gpu_mean_pairwise_hamming(
    gpu: &Gpu,
    sequences: &[u32],
    n_seqs: u32,
    seq_len: u32,
) -> Result<f32, String> {
    let device = gpu.device();
    let queue = gpu.queue();
    let n_pairs = (n_seqs * (n_seqs - 1) / 2) as usize;

    let hamming_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_hamming"),
        source: wgpu::ShaderSource::Wgsl(HAMMING_WGSL.into()),
    });

    let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chain_sate_reduce"),
        source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
    });

    let hamming_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_hamming_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });

    let hamming_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_hamming_pl"),
        bind_group_layouts: &[&hamming_bgl],
        push_constant_ranges: &[],
    });

    let hamming_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_hamming_pipeline"),
        layout: Some(&hamming_pl),
        module: &hamming_shader,
        entry_point: "pairwise_hamming",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chain_sate_reduce_bgl"),
        entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
    });

    let reduce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chain_sate_reduce_pl"),
        bind_group_layouts: &[&reduce_bgl],
        push_constant_ranges: &[],
    });

    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chain_sate_reduce_pipeline"),
        layout: Some(&reduce_pl),
        module: &reduce_shader,
        entry_point: "mean_reduce",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let seq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_sate_sequences"),
        contents: bytemuck::cast_slice(sequences),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_sate_distances"),
        size: (n_pairs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let hamming_params = HammingParams {
        n_seqs,
        seq_len,
        _pad0: 0,
        _pad1: 0,
    };
    let hamming_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_sate_hamming_params"),
        contents: bytemuck::bytes_of(&hamming_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chain_sate_mean_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let reduce_params = ReduceParams { n: n_pairs as u32 };
    let reduce_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chain_sate_reduce_params"),
        contents: bytemuck::bytes_of(&reduce_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let hamming_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_hamming_bg"),
        layout: &hamming_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: seq_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dist_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: hamming_params_buf.as_entire_binding(),
            },
        ],
    });

    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chain_sate_reduce_bg"),
        layout: &reduce_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dist_buf.as_entire_binding(),
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
        label: Some("chain_sate_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_hamming_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&hamming_pipeline);
        pass.set_bind_group(0, &hamming_bg, &[]);
        pass.dispatch_workgroups(n_pairs.div_ceil(256) as u32, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chain_sate_reduce_pass"),
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

fn generate_sequences(n_seqs: usize, seq_len: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let mut flat = Vec::with_capacity(n_seqs * seq_len);
    for _ in 0..(n_seqs * seq_len) {
        flat.push(rng.usize(4) as u32);
    }
    flat
}

fn validate_sate_small(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 8_usize;
    let seq_len = 20_usize;
    let sequences = generate_sequences(n_seqs, seq_len, 42);

    let cpu_mean = cpu_mean_pairwise_hamming(&sequences, n_seqs, seq_len);

    match gpu_mean_pairwise_hamming(gpu, &sequences, n_seqs as u32, seq_len as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("sate small 8×20: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_HAMMING_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("sate small: dispatch failed — {e}"), false);
        }
    }
}

fn validate_sate_larger(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 16_usize;
    let seq_len = 50_usize;
    let sequences = generate_sequences(n_seqs, seq_len, 777);

    let cpu_mean = cpu_mean_pairwise_hamming(&sequences, n_seqs, seq_len);

    match gpu_mean_pairwise_hamming(gpu, &sequences, n_seqs as u32, seq_len as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("sate larger 16×50: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_HAMMING_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("sate larger: dispatch failed — {e}"), false);
        }
    }
}

fn validate_sate_identical(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 4_usize;
    let seq_len = 10_usize;
    let base_seq: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
    let mut sequences = vec![0_u32; n_seqs * seq_len];
    for i in 0..n_seqs {
        sequences[i * seq_len..(i + 1) * seq_len].copy_from_slice(&base_seq);
    }

    match gpu_mean_pairwise_hamming(gpu, &sequences, n_seqs as u32, seq_len as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("sate identical: mean distance={gpu_mean:.6} vs 0"),
                f64::from(gpu_mean),
                0.0,
                tolerances::GPU_HAMMING_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("sate identical: dispatch failed — {e}"), false);
        }
    }
}

fn validate_sate_all_differ(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 4_usize;
    let seq_len = 8_usize;
    let mut sequences = vec![0_u32; n_seqs * seq_len];
    for i in 0..n_seqs {
        for s in 0..seq_len {
            sequences[i * seq_len + s] = ((i + s) % 4) as u32;
        }
    }

    let cpu_mean = cpu_mean_pairwise_hamming(&sequences, n_seqs, seq_len);

    match gpu_mean_pairwise_hamming(gpu, &sequences, n_seqs as u32, seq_len as u32) {
        Ok(gpu_mean) => {
            h.check_abs(
                &format!("sate all differ: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_HAMMING_F32,
            );
        }
        Err(e) => {
            h.check_bool(&format!("sate all differ: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_seqs = 6_usize;
    let seq_len = 12_usize;
    let sequences = generate_sequences(n_seqs, seq_len, 99);

    let r1 = gpu_mean_pairwise_hamming(gpu, &sequences, n_seqs as u32, seq_len as u32);
    let r2 = gpu_mean_pairwise_hamming(gpu, &sequences, n_seqs as u32, seq_len as u32);

    match (r1, r2) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("sate determinism: run1={a:.6} == run2={b:.6}"),
                (a - b).abs() < f32::EPSILON,
            );
        }
        _ => {
            h.check_bool("sate determinism: dispatch failed", false);
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
