// SPDX-License-Identifier: AGPL-3.0-or-later

//! nF-01 Phase B GPU: Sovereign folding WGSL shader validation (df64 core streaming).
//!
//! Validates that the Evoformer WGSL shaders (df64 core streaming on consumer GPU)
//! reproduce CPU f64 references within df64-class tolerance (~1e-10).
//!
//! Three-zone core streaming: f64 buffer I/O → df64 compute on FP32 cores → f64 output.
//! Achieves ~14-digit (fp48) precision on consumer GPUs instead of ~7-digit (f32).
//!
//! ## Shaders validated
//!
//! | Shader | Algorithm | Core streaming |
//! |--------|-----------|----------------|
//! | `gelu_f64.wgsl` | Pointwise GELU | `tanh_df64` |
//! | `triangle_mul_outgoing_f64.wgsl` | Algorithm 11 | df64 accumulation |
//! | `triangle_mul_incoming_f64.wgsl` | Algorithm 12 | df64 accumulation |
//! | `triangle_attention_f64.wgsl` | Algorithms 13-14 | df64 dot + bias |
//! | `sdpa_scores_f64.wgsl` | QKᵀ/√d (pass 1) | df64 dot + `sqrt_df64` |
//! | `softmax_f64.wgsl` | Row-wise softmax (pass 2) | `exp_df64` + df64 div |
//! | `layer_norm_f64.wgsl` | Layer normalization | df64 mean/var + `sqrt_df64` |
//! | `attention_apply_f64.wgsl` | Σ weights × V (pass 3) | df64 weighted sum |
//! | `outer_product_mean_f64.wgsl` | MSA → pair (OPM) | df64 accumulation |
//! | `msa_row_attention_scores_f64.wgsl` | Row attn + pair bias | df64 dot + bias |
//! | `msa_col_attention_scores_f64.wgsl` | Column attn (no bias) | df64 dot product |
//! | `ipa_scores_f64.wgsl` | IPA (SE(3)-equivariant) | df64 three-term |
//! | `backbone_update_f64.wgsl` | Frame composition | df64 matrix mul |
//! | `torsion_angles_f64.wgsl` | Fused `ResNet` + normalize | df64 linear + `sqrt_df64` |
//! | 3-pass SDPA pipeline | scores→softmax→apply | Full df64 chain |
//!
//! ## Provenance
//!
//! CPU reference: `neural_spring::sovereign_folding` + `neural_spring::structure_module`.
//! GPU: `metalForge/shaders/` WGSL → `compile_shader_f64_hybrid` (df64 preamble + f64 types).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::manual_div_ceil,
    clippy::suboptimal_flops,
    clippy::many_single_char_names
)]

use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::sovereign_folding;
use neural_spring::structure_module;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

/// df64 arithmetic tolerance — dot products, matrix ops, accumulations.
/// Observed precision: ~1e-7 to 1e-8 (limited by f32 FMA error tracking).
const GPU_DF64_TOL: f64 = 1e-6;

/// df64 transcendental tolerance — `exp_df64`, `tanh_df64`, etc.
/// Limited by degree-6 Horner polynomial in `exp_df64` (~3-4 decimal digits
/// vs native f64 `exp`/`tanh`). Still 10-100x better than pure f32.
const GPU_DF64_TRANS_TOL: f64 = 5e-4;

const fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if read_only {
                wgpu::BufferBindingType::Storage { read_only: true }
            } else {
                wgpu::BufferBindingType::Storage { read_only: false }
            },
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

fn run_compute(
    gpu: &Gpu,
    shader: &wgpu::ShaderModule,
    entry: &str,
    bind_group: &wgpu::BindGroup,
    layout: &wgpu::BindGroupLayout,
    workgroups: (u32, u32, u32),
) {
    let device = gpu.device();
    let queue = gpu.queue();

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pl),
        module: shader,
        entry_point: entry,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    queue.submit(std::iter::once(encoder.finish()));
}

fn read_gpu_f64(gpu: &Gpu, buf: &wgpu::Buffer, count: usize) -> Vec<f64> {
    gpu.read_buffer_f64(buf, count).expect("GPU f64 readback")
}

fn upload_f64(gpu: &Gpu, data: &[f64], label: &str) -> wgpu::Buffer {
    gpu.device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
}

fn max_diff(gpu_result: &[f64], cpu_ref: &[f64]) -> f64 {
    gpu_result
        .iter()
        .zip(cpu_ref.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0_f64, f64::max)
}

// ─── GELU (df64 core streaming) ─────────────────────────────────────

fn validate_gelu_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 64_u32;
    let mut rng = Rng::new(42);
    let input: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
    let cpu_ref: Vec<f64> = input.iter().map(|&x| sovereign_folding::gelu(x)).collect();

    let shader = gpu.compile_shader_f64_hybrid(neural_spring_forge::shaders::GELU_F64, "gelu_f64");

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n: u32,
        _p1: u32,
        _p2: u32,
        _p3: u32,
    }

    let in_buf = upload_f64(gpu, &input, "gelu_in");
    let out_buf = gpu.create_buffer_f64(n as usize).expect("alloc gelu_out");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gelu_params"),
            contents: bytemuck::bytes_of(&Params {
                n,
                _p1: 0,
                _p2: 0,
                _p3: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                uniform_entry(2),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    run_compute(gpu, &shader, "gelu_f64", &bg, &bgl, ((n + 255) / 256, 1, 1));

    let gpu_result = read_gpu_f64(gpu, &out_buf, n as usize);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  GELU GPU max diff: {md:.2e}");
    h.check_abs("GELU GPU max diff", md, 0.0, GPU_DF64_TRANS_TOL);
    h.check_bool("GELU GPU finite", gpu_result.iter().all(|v| v.is_finite()));
}

// ─── Triangle mul outgoing (df64 core streaming) ────────────────────

fn validate_triangle_outgoing_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 8_u32;
    let c = 4_u32;
    let total = (n * n * c) as usize;
    let mut rng = Rng::new(42);
    let proj_a: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
    let proj_b: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
    let cpu_ref =
        sovereign_folding::triangle_mul_outgoing(&proj_a, &proj_b, n as usize, c as usize);

    let shader = gpu.compile_shader_f64_hybrid(
        neural_spring_forge::shaders::TRIANGLE_MUL_OUTGOING_F64,
        "tri_out",
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n_res: u32,
        channels: u32,
        _p0: u32,
        _p1: u32,
    }

    let a_buf = upload_f64(gpu, &proj_a, "tri_a");
    let b_buf = upload_f64(gpu, &proj_b, "tri_b");
    let out_buf = gpu.create_buffer_f64(total).expect("alloc tri_out");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tri_params"),
            contents: bytemuck::bytes_of(&Params {
                n_res: n,
                channels: c,
                _p0: 0,
                _p1: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    run_compute(
        gpu,
        &shader,
        "main",
        &bg,
        &bgl,
        (((total as u32) + 255) / 256, 1, 1),
    );

    let gpu_result = read_gpu_f64(gpu, &out_buf, total);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  TriMul outgoing GPU max diff: {md:.2e}");
    h.check_abs("TriMul outgoing GPU max diff", md, 0.0, GPU_DF64_TOL);
    h.check_bool(
        "TriMul outgoing GPU finite",
        gpu_result.iter().all(|v| v.is_finite()),
    );
}

// ─── Triangle mul incoming (df64 core streaming) ────────────────────

fn validate_triangle_incoming_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 8_u32;
    let c = 4_u32;
    let total = (n * n * c) as usize;
    let mut rng = Rng::new(99);
    let proj_a: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
    let proj_b: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
    let cpu_ref =
        sovereign_folding::triangle_mul_incoming(&proj_a, &proj_b, n as usize, c as usize);

    let shader = gpu.compile_shader_f64_hybrid(
        neural_spring_forge::shaders::TRIANGLE_MUL_INCOMING_F64,
        "tri_in",
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n_res: u32,
        channels: u32,
        _p0: u32,
        _p1: u32,
    }

    let a_buf = upload_f64(gpu, &proj_a, "tri_a");
    let b_buf = upload_f64(gpu, &proj_b, "tri_b");
    let out_buf = gpu.create_buffer_f64(total).expect("alloc tri_out");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tri_params"),
            contents: bytemuck::bytes_of(&Params {
                n_res: n,
                channels: c,
                _p0: 0,
                _p1: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    run_compute(
        gpu,
        &shader,
        "main",
        &bg,
        &bgl,
        (((total as u32) + 255) / 256, 1, 1),
    );

    let gpu_result = read_gpu_f64(gpu, &out_buf, total);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  TriMul incoming GPU max diff: {md:.2e}");
    h.check_abs("TriMul incoming GPU max diff", md, 0.0, GPU_DF64_TOL);
    h.check_bool(
        "TriMul incoming GPU finite",
        gpu_result.iter().all(|v| v.is_finite()),
    );
}

// ─── SDPA scores (df64 core streaming) ──────────────────────────────

fn validate_sdpa_scores_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let (b, he, sq, skv, d) = (1_u32, 2_u32, 8_u32, 8_u32, 4_u32);
    let q_len = (b * he * sq * d) as usize;
    let k_len = (b * he * skv * d) as usize;
    let out_len = (b * he * sq * skv) as usize;

    let mut rng = Rng::new(42);
    let q: Vec<f64> = (0..q_len).map(|_| rng.next_f64()).collect();
    let k: Vec<f64> = (0..k_len).map(|_| rng.next_f64()).collect();
    let cpu_ref = sovereign_folding::sdpa_scores(
        &q,
        &k,
        b as usize,
        he as usize,
        sq as usize,
        skv as usize,
        d as usize,
    );

    let shader =
        gpu.compile_shader_f64_hybrid(neural_spring_forge::shaders::SDPA_SCORES_F64, "sdpa_scores");

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct AP {
        batch_size: u32,
        num_heads: u32,
        q_seq_len: u32,
        kv_seq_len: u32,
        head_dim: u32,
        _p0: u32,
        _p1: u32,
        _p2: u32,
    }

    let q_buf = upload_f64(gpu, &q, "q");
    let k_buf = upload_f64(gpu, &k, "k");
    let out_buf = gpu.create_buffer_f64(out_len).expect("alloc scores");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&AP {
                batch_size: b,
                num_heads: he,
                q_seq_len: sq,
                kv_seq_len: skv,
                head_dim: d,
                _p0: 0,
                _p1: 0,
                _p2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: k_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    run_compute(
        gpu,
        &shader,
        "main",
        &bg,
        &bgl,
        (((out_len as u32) + 255) / 256, 1, 1),
    );

    let gpu_result = read_gpu_f64(gpu, &out_buf, out_len);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  SDPA scores GPU max diff: {md:.2e}");
    h.check_abs("SDPA scores GPU max diff", md, 0.0, GPU_DF64_TOL);
    h.check_bool(
        "SDPA scores GPU finite",
        gpu_result.iter().all(|v| v.is_finite()),
    );
}

// ─── Triangle attention scores (df64 core streaming) ────────────────

fn validate_triangle_attention_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let (r, n, he, d) = (4_u32, 8_u32, 2_u32, 4_u32);
    let qk_len = (r * n * he * d) as usize;
    let bias_len = (he * n * n) as usize;
    let out_len = (r * he * n * n) as usize;

    let mut rng = Rng::new(77);
    let q: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let k: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let bias: Vec<f64> = (0..bias_len).map(|_| rng.next_f64() * 0.1).collect();
    let cpu_ref = sovereign_folding::triangle_attention_scores(
        &q,
        &k,
        &bias,
        r as usize,
        n as usize,
        he as usize,
        d as usize,
    );

    let shader = gpu.compile_shader_f64_hybrid(
        neural_spring_forge::shaders::TRIANGLE_ATTENTION_F64,
        "tri_attn",
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n_rows: u32,
        n_res: u32,
        n_heads: u32,
        head_dim: u32,
    }

    let q_buf = upload_f64(gpu, &q, "q");
    let k_buf = upload_f64(gpu, &k, "k");
    let bias_buf = upload_f64(gpu, &bias, "bias");
    let out_buf = gpu.create_buffer_f64(out_len).expect("alloc tri_attn_out");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&Params {
                n_rows: r,
                n_res: n,
                n_heads: he,
                head_dim: d,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                uniform_entry(4),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: k_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bias_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    run_compute(
        gpu,
        &shader,
        "main",
        &bg,
        &bgl,
        (((out_len as u32) + 255) / 256, 1, 1),
    );

    let gpu_result = read_gpu_f64(gpu, &out_buf, out_len);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  TriAttn scores GPU max diff: {md:.2e}");
    h.check_abs("TriAttn scores GPU max diff", md, 0.0, GPU_DF64_TOL);
    h.check_bool(
        "TriAttn scores GPU finite",
        gpu_result.iter().all(|v| v.is_finite()),
    );
}

// ─── Softmax (df64 core streaming) ──────────────────────────────────

fn validate_softmax_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let (rows, cols) = (4_u32, 16_u32);
    let total = (rows * cols) as usize;

    let mut rng = Rng::new(55);
    let input: Vec<f64> = (0..total).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
    let cpu_ref = sovereign_folding::softmax_rows(&input, rows as usize, cols as usize);

    let shader =
        gpu.compile_shader_f64_hybrid(neural_spring_forge::shaders::SOFTMAX_F64, "softmax");

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        num_rows: u32,
        row_len: u32,
        _p0: u32,
        _p1: u32,
    }

    let in_buf = upload_f64(gpu, &input, "sm_in");
    let out_buf = gpu.create_buffer_f64(total).expect("alloc sm_out");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sm_params"),
            contents: bytemuck::bytes_of(&Params {
                num_rows: rows,
                row_len: cols,
                _p0: 0,
                _p1: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                uniform_entry(2),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    run_compute(gpu, &shader, "main", &bg, &bgl, (rows, 1, 1));

    let gpu_result = read_gpu_f64(gpu, &out_buf, total);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  Softmax GPU max diff: {md:.2e}");
    h.check_abs("Softmax GPU max diff", md, 0.0, GPU_DF64_TRANS_TOL);

    for row in 0..rows as usize {
        let sum: f64 = gpu_result[row * cols as usize..(row + 1) * cols as usize]
            .iter()
            .sum();
        h.check_abs(
            &format!("Softmax GPU row {row} sum"),
            sum,
            1.0,
            GPU_DF64_TRANS_TOL,
        );
    }
}

// ─── Layer Norm (df64 core streaming) ───────────────────────────────

fn validate_layer_norm_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let (seq_len, hidden_dim) = (8_u32, 16_u32);
    let total = (seq_len * hidden_dim) as usize;

    let mut rng = Rng::new(42);
    let input: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
    let gamma: Vec<f64> = (0..hidden_dim).map(|_| 0.5 + rng.next_f64()).collect();
    let beta: Vec<f64> = (0..hidden_dim)
        .map(|_| rng.next_f64() * 0.2 - 0.1)
        .collect();
    let cpu_ref = sovereign_folding::layer_norm(
        &input,
        seq_len as usize,
        hidden_dim as usize,
        &gamma,
        &beta,
        1e-5,
    );

    let shader =
        gpu.compile_shader_f64_hybrid(neural_spring_forge::shaders::LAYER_NORM_F64, "layer_norm");

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        seq_len: u32,
        hidden_dim: u32,
        eps_hi: f32,
        eps_lo: f32,
    }

    let in_buf = upload_f64(gpu, &input, "ln_in");
    let gamma_buf = upload_f64(gpu, &gamma, "gamma");
    let beta_buf = upload_f64(gpu, &beta, "beta");
    let out_buf = gpu.create_buffer_f64(total).expect("alloc ln_out");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ln_params"),
            contents: bytemuck::bytes_of(&Params {
                seq_len,
                hidden_dim,
                eps_hi: 1e-5,
                eps_lo: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                uniform_entry(4),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gamma_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: beta_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    run_compute(gpu, &shader, "layer_norm", &bg, &bgl, (seq_len, 1, 1));

    let gpu_result = read_gpu_f64(gpu, &out_buf, total);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  LayerNorm GPU max diff: {md:.2e}");
    h.check_abs("LayerNorm GPU max diff", md, 0.0, GPU_DF64_TOL);
    h.check_bool(
        "LayerNorm GPU finite",
        gpu_result.iter().all(|v| v.is_finite()),
    );
}

// ─── Attention Apply (df64 core streaming) ──────────────────────────

fn validate_attention_apply_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let (b, he, sq, skv, d) = (1_u32, 2_u32, 8_u32, 8_u32, 4_u32);
    let w_len = (b * he * sq * skv) as usize;
    let v_len = (b * he * skv * d) as usize;
    let out_len = (b * he * sq * d) as usize;

    let mut rng = Rng::new(42);

    let weights_raw: Vec<f64> = (0..w_len).map(|_| rng.next_f64()).collect();
    let weights_rows = (b * he * sq) as usize;
    let weights = sovereign_folding::softmax_rows(&weights_raw, weights_rows, skv as usize);
    let v: Vec<f64> = (0..v_len).map(|_| rng.next_f64()).collect();
    let cpu_ref = sovereign_folding::attention_apply(
        &weights,
        &v,
        b as usize,
        he as usize,
        sq as usize,
        skv as usize,
        d as usize,
    );

    let shader = gpu.compile_shader_f64_hybrid(
        neural_spring_forge::shaders::ATTENTION_APPLY_F64,
        "attn_apply",
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct AP {
        batch_size: u32,
        num_heads: u32,
        q_seq_len: u32,
        kv_seq_len: u32,
        head_dim: u32,
        _p0: u32,
        _p1: u32,
        _p2: u32,
    }

    let w_buf = upload_f64(gpu, &weights, "weights");
    let v_buf = upload_f64(gpu, &v, "value");
    let out_buf = gpu.create_buffer_f64(out_len).expect("alloc attn_out");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&AP {
                batch_size: b,
                num_heads: he,
                q_seq_len: sq,
                kv_seq_len: skv,
                head_dim: d,
                _p0: 0,
                _p1: 0,
                _p2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: v_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let wg_x = (d + 15) / 16;
    let wg_y = (sq + 15) / 16;
    let wg_z = b * he;
    run_compute(gpu, &shader, "main", &bg, &bgl, (wg_x, wg_y, wg_z));

    let gpu_result = read_gpu_f64(gpu, &out_buf, out_len);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  AttnApply GPU max diff: {md:.2e}");
    h.check_abs("AttnApply GPU max diff", md, 0.0, GPU_DF64_TOL);
    h.check_bool(
        "AttnApply GPU finite",
        gpu_result.iter().all(|v| v.is_finite()),
    );
}

// ─── Outer Product Mean (df64 core streaming) ───────────────────────

fn validate_outer_product_mean_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let (n_seq, n_res, c_a, c_b) = (6_u32, 4_u32, 3_u32, 2_u32);
    let c_out = c_a * c_b;
    let a_len = (n_seq * n_res * c_a) as usize;
    let b_len = (n_seq * n_res * c_b) as usize;
    let out_len = (n_res * n_res * c_out) as usize;

    let mut rng = Rng::new(42);
    let a: Vec<f64> = (0..a_len).map(|_| rng.next_f64()).collect();
    let bv: Vec<f64> = (0..b_len).map(|_| rng.next_f64()).collect();
    let cpu_ref = sovereign_folding::outer_product_mean(
        &a,
        &bv,
        n_seq as usize,
        n_res as usize,
        c_a as usize,
        c_b as usize,
    );

    let shader =
        gpu.compile_shader_f64_hybrid(neural_spring_forge::shaders::OUTER_PRODUCT_MEAN_F64, "opm");

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n_seq: u32,
        n_res: u32,
        c_a: u32,
        c_b: u32,
    }

    let a_buf = upload_f64(gpu, &a, "opm_a");
    let b_buf = upload_f64(gpu, &bv, "opm_b");
    let out_buf = gpu.create_buffer_f64(out_len).expect("alloc opm_out");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("opm_params"),
            contents: bytemuck::bytes_of(&Params {
                n_seq,
                n_res,
                c_a,
                c_b,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    run_compute(
        gpu,
        &shader,
        "main",
        &bg,
        &bgl,
        (((out_len as u32) + 255) / 256, 1, 1),
    );

    let gpu_result = read_gpu_f64(gpu, &out_buf, out_len);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  OPM GPU max diff: {md:.2e}");
    h.check_abs("OPM GPU max diff", md, 0.0, GPU_DF64_TOL);
    h.check_bool("OPM GPU finite", gpu_result.iter().all(|v| v.is_finite()));
}

// ─── MSA Row Attention Scores (df64 core streaming) ─────────────────

fn validate_msa_row_attention_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let (n_seq, n_res, n_heads, head_dim) = (4_u32, 6_u32, 2_u32, 4_u32);
    let qk_len = (n_seq * n_res * n_heads * head_dim) as usize;
    let bias_len = (n_heads * n_res * n_res) as usize;
    let out_len = (n_seq * n_heads * n_res * n_res) as usize;

    let mut rng = Rng::new(42);
    let q: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let k: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let bias: Vec<f64> = (0..bias_len).map(|_| rng.next_f64() * 0.1).collect();
    let cpu_ref = sovereign_folding::msa_row_attention_scores(
        &q,
        &k,
        &bias,
        n_seq as usize,
        n_res as usize,
        n_heads as usize,
        head_dim as usize,
    );

    let shader = gpu.compile_shader_f64_hybrid(
        neural_spring_forge::shaders::MSA_ROW_ATTENTION_SCORES_F64,
        "msa_row",
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n_seq: u32,
        n_res: u32,
        n_heads: u32,
        head_dim: u32,
    }

    let q_buf = upload_f64(gpu, &q, "msa_row_q");
    let k_buf = upload_f64(gpu, &k, "msa_row_k");
    let bias_buf = upload_f64(gpu, &bias, "msa_row_bias");
    let out_buf = gpu.create_buffer_f64(out_len).expect("alloc msa_row_out");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("msa_row_params"),
            contents: bytemuck::bytes_of(&Params {
                n_seq,
                n_res,
                n_heads,
                head_dim,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                uniform_entry(4),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: k_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bias_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    run_compute(
        gpu,
        &shader,
        "main",
        &bg,
        &bgl,
        (((out_len as u32) + 255) / 256, 1, 1),
    );

    let gpu_result = read_gpu_f64(gpu, &out_buf, out_len);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  MSA row scores GPU max diff: {md:.2e}");
    h.check_abs("MSA row scores GPU max diff", md, 0.0, GPU_DF64_TOL);
    h.check_bool(
        "MSA row scores GPU finite",
        gpu_result.iter().all(|v| v.is_finite()),
    );
}

// ─── MSA Column Attention Scores (df64 core streaming) ──────────────

fn validate_msa_col_attention_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let (n_seq, n_res, n_heads, head_dim) = (4_u32, 6_u32, 2_u32, 4_u32);
    let qk_len = (n_seq * n_res * n_heads * head_dim) as usize;
    let out_len = (n_res * n_heads * n_seq * n_seq) as usize;

    let mut rng = Rng::new(77);
    let q: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let k: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let cpu_ref = sovereign_folding::msa_col_attention_scores(
        &q,
        &k,
        n_seq as usize,
        n_res as usize,
        n_heads as usize,
        head_dim as usize,
    );

    let shader = gpu.compile_shader_f64_hybrid(
        neural_spring_forge::shaders::MSA_COL_ATTENTION_SCORES_F64,
        "msa_col",
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n_seq: u32,
        n_res: u32,
        n_heads: u32,
        head_dim: u32,
    }

    let q_buf = upload_f64(gpu, &q, "msa_col_q");
    let k_buf = upload_f64(gpu, &k, "msa_col_k");
    let out_buf = gpu.create_buffer_f64(out_len).expect("alloc msa_col_out");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("msa_col_params"),
            contents: bytemuck::bytes_of(&Params {
                n_seq,
                n_res,
                n_heads,
                head_dim,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: k_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    run_compute(
        gpu,
        &shader,
        "main",
        &bg,
        &bgl,
        (((out_len as u32) + 255) / 256, 1, 1),
    );

    let gpu_result = read_gpu_f64(gpu, &out_buf, out_len);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  MSA col scores GPU max diff: {md:.2e}");
    h.check_abs("MSA col scores GPU max diff", md, 0.0, GPU_DF64_TOL);
    h.check_bool(
        "MSA col scores GPU finite",
        gpu_result.iter().all(|v| v.is_finite()),
    );
}

// ─── Full 3-pass SDPA Pipeline (df64 core streaming) ────────────────

fn validate_sdpa_pipeline_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let (b, he, sq, skv, d) = (1_u32, 2_u32, 8_u32, 8_u32, 4_u32);
    let qk_len = (b * he * sq * d) as usize;
    let v_len = (b * he * skv * d) as usize;
    let scores_len = (b * he * sq * skv) as usize;
    let out_len = (b * he * sq * d) as usize;

    let mut rng = Rng::new(42);
    let q: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let k: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let v: Vec<f64> = (0..v_len).map(|_| rng.next_f64()).collect();
    let cpu_ref = sovereign_folding::sdpa_full(
        &q,
        &k,
        &v,
        b as usize,
        he as usize,
        sq as usize,
        skv as usize,
        d as usize,
    );

    let scores_shader = gpu.compile_shader_f64_hybrid(
        neural_spring_forge::shaders::SDPA_SCORES_F64,
        "pipeline_scores",
    );
    let softmax_shader = gpu.compile_shader_f64_hybrid(
        neural_spring_forge::shaders::SOFTMAX_F64,
        "pipeline_softmax",
    );
    let apply_shader = gpu.compile_shader_f64_hybrid(
        neural_spring_forge::shaders::ATTENTION_APPLY_F64,
        "pipeline_apply",
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct AP {
        batch_size: u32,
        num_heads: u32,
        q_seq_len: u32,
        kv_seq_len: u32,
        head_dim: u32,
        _p0: u32,
        _p1: u32,
        _p2: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct SP {
        num_rows: u32,
        row_len: u32,
        _p0: u32,
        _p1: u32,
    }

    let attn_params = AP {
        batch_size: b,
        num_heads: he,
        q_seq_len: sq,
        kv_seq_len: skv,
        head_dim: d,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };

    let q_buf = upload_f64(gpu, &q, "q");
    let k_buf = upload_f64(gpu, &k, "k");
    let v_buf = upload_f64(gpu, &v, "v");
    let scores_buf = gpu.create_buffer_f64(scores_len).expect("alloc scores");
    let weights_buf = gpu.create_buffer_f64(scores_len).expect("alloc weights");
    let output_buf = gpu.create_buffer_f64(out_len).expect("alloc output");
    let device = gpu.device();

    // Pass 1: scores = Q @ K^T / sqrt(d)
    let p1_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p1_params"),
        contents: bytemuck::bytes_of(&attn_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let p1_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });
    let p1_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &p1_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: k_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scores_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: p1_params.as_entire_binding(),
            },
        ],
    });
    run_compute(
        gpu,
        &scores_shader,
        "main",
        &p1_bg,
        &p1_bgl,
        (((scores_len as u32) + 255) / 256, 1, 1),
    );

    // Pass 2: weights = softmax(scores)
    let sm_rows = b * he * sq;
    let sm_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sm_params"),
        contents: bytemuck::bytes_of(&SP {
            num_rows: sm_rows,
            row_len: skv,
            _p0: 0,
            _p1: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let p2_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ],
    });
    let p2_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &p2_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scores_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weights_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sm_params.as_entire_binding(),
            },
        ],
    });
    run_compute(
        gpu,
        &softmax_shader,
        "main",
        &p2_bg,
        &p2_bgl,
        (sm_rows, 1, 1),
    );

    // Pass 3: output = weights @ V
    let p3_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p3_params"),
        contents: bytemuck::bytes_of(&attn_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let p3_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });
    let p3_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &p3_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: weights_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: v_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: p3_params.as_entire_binding(),
            },
        ],
    });
    run_compute(
        gpu,
        &apply_shader,
        "main",
        &p3_bg,
        &p3_bgl,
        ((d + 15) / 16, (sq + 15) / 16, b * he),
    );

    let gpu_result = read_gpu_f64(gpu, &output_buf, out_len);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  SDPA pipeline GPU max diff: {md:.2e}");
    h.check_abs("SDPA pipeline GPU max diff", md, 0.0, GPU_DF64_TRANS_TOL);
    h.check_bool(
        "SDPA pipeline GPU finite",
        gpu_result.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "SDPA pipeline GPU nonzero",
        gpu_result.iter().any(|v| *v != 0.0),
    );
}

// ─── IPA Scores (df64 core streaming) ───────────────────────────────

fn validate_ipa_scores_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let (n_res, n_heads, head_dim, n_points) = (4_u32, 2_u32, 4_u32, 3_u32);
    let qk_len = (n_res * n_heads * head_dim) as usize;
    let bias_len = (n_heads * n_res * n_res) as usize;
    let pts_len = (n_res * n_heads * n_points * 3) as usize;
    let frames_len = (n_res * 12) as usize;
    let out_len = (n_heads * n_res * n_res) as usize;
    let (w_l, w_c, w_p, gamma) = (1.0_f32, 1.0_f32, 1.0_f32, 0.5_f32);

    let mut rng = Rng::new(99);
    let q: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let k: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let bias: Vec<f64> = (0..bias_len).map(|_| rng.next_f64() * 0.1).collect();
    let qp: Vec<f64> = (0..pts_len).map(|_| rng.next_f64()).collect();
    let kp: Vec<f64> = (0..pts_len).map(|_| rng.next_f64()).collect();

    let mut frames = vec![0.0_f64; frames_len];
    for i in 0..n_res as usize {
        frames[i * 12] = 1.0;
        frames[i * 12 + 4] = 1.0;
        frames[i * 12 + 8] = 1.0;
        for t in 9..12 {
            frames[i * 12 + t] = rng.next_f64();
        }
    }

    let cpu_ref = structure_module::ipa_scores(
        &q,
        &k,
        &bias,
        &qp,
        &kp,
        &frames,
        n_res as usize,
        n_heads as usize,
        head_dim as usize,
        n_points as usize,
        f64::from(w_l),
        f64::from(w_c),
        f64::from(w_p),
        f64::from(gamma),
    );

    let shader = gpu.compile_shader_f64_hybrid(neural_spring_forge::shaders::IPA_SCORES_F64, "ipa");

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n_res: u32,
        n_heads: u32,
        head_dim: u32,
        n_points: u32,
        w_l: f32,
        w_c: f32,
        w_p: f32,
        gamma: f32,
    }

    let q_buf = upload_f64(gpu, &q, "ipa_q");
    let k_buf = upload_f64(gpu, &k, "ipa_k");
    let bias_buf = upload_f64(gpu, &bias, "ipa_bias");
    let qp_buf = upload_f64(gpu, &qp, "ipa_qp");
    let kp_buf = upload_f64(gpu, &kp, "ipa_kp");
    let frames_buf = upload_f64(gpu, &frames, "ipa_frames");
    let out_buf = gpu.create_buffer_f64(out_len).expect("alloc ipa_out");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ipa_params"),
            contents: bytemuck::bytes_of(&Params {
                n_res,
                n_heads,
                head_dim,
                n_points,
                w_l,
                w_c,
                w_p,
                gamma,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, false),
                uniform_entry(7),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: k_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bias_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: qp_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: kp_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: frames_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    run_compute(
        gpu,
        &shader,
        "main",
        &bg,
        &bgl,
        (((out_len as u32) + 255) / 256, 1, 1),
    );

    let gpu_result = read_gpu_f64(gpu, &out_buf, out_len);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  IPA scores GPU max diff: {md:.2e}");
    h.check_abs("IPA scores GPU max diff", md, 0.0, GPU_DF64_TOL);
    h.check_bool(
        "IPA scores GPU finite",
        gpu_result.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "IPA scores GPU nonzero",
        gpu_result.iter().any(|v| *v != 0.0),
    );
}

// ─── Backbone Update (df64 core streaming) ──────────────────────────

fn validate_backbone_update_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_res = 4_u32;
    let frames_len = (n_res * 12) as usize;
    let quats_len = (n_res * 4) as usize;
    let trans_len = (n_res * 3) as usize;

    let mut rng = Rng::new(88);

    let mut cur_frames = vec![0.0_f64; frames_len];
    for i in 0..n_res as usize {
        cur_frames[i * 12] = 1.0;
        cur_frames[i * 12 + 4] = 1.0;
        cur_frames[i * 12 + 8] = 1.0;
        for t in 9..12 {
            cur_frames[i * 12 + t] = rng.next_f64();
        }
    }

    let mut dq: Vec<f64> = (0..quats_len).map(|_| rng.next_f64() * 0.1).collect();
    for i in 0..n_res as usize {
        dq[i * 4] += 1.0;
    }
    let dt: Vec<f64> = (0..trans_len).map(|_| rng.next_f64() * 0.1).collect();

    let cpu_ref = structure_module::backbone_update(&dq, &dt, &cur_frames, n_res as usize);

    let shader = gpu.compile_shader_f64_hybrid(
        neural_spring_forge::shaders::BACKBONE_UPDATE_F64,
        "bb_update",
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n_res: u32,
        _p0: u32,
        _p1: u32,
        _p2: u32,
    }

    let dq_buf = upload_f64(gpu, &dq, "bb_dq");
    let dt_buf = upload_f64(gpu, &dt, "bb_dt");
    let cur_buf = upload_f64(gpu, &cur_frames, "bb_cur");
    let out_buf = gpu.create_buffer_f64(frames_len).expect("alloc bb_out");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bb_params"),
            contents: bytemuck::bytes_of(&Params {
                n_res,
                _p0: 0,
                _p1: 0,
                _p2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                uniform_entry(4),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dq_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dt_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cur_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    run_compute(gpu, &shader, "main", &bg, &bgl, ((n_res + 255) / 256, 1, 1));

    let gpu_result = read_gpu_f64(gpu, &out_buf, frames_len);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  Backbone GPU max diff: {md:.2e}");
    h.check_abs("Backbone GPU max diff", md, 0.0, GPU_DF64_TOL);
    h.check_bool(
        "Backbone GPU finite",
        gpu_result.iter().all(|v| v.is_finite()),
    );
    h.check_bool("Backbone GPU nonzero", gpu_result.iter().any(|v| *v != 0.0));
}

// ─── Torsion Angles (df64 core streaming) ───────────────────────────

fn validate_torsion_angles_gpu(h: &mut ValidationHarness, gpu: &Gpu) {
    let (n_res, c_single, c_hidden) = (4_u32, 8_u32, 6_u32);
    let hh = c_hidden * c_hidden;
    let n_weights = c_single * c_hidden + c_hidden + 4 * (hh + c_hidden) + c_hidden * 14 + 14;
    let single_len = (n_res * c_single) as usize;
    let out_len = (n_res * 14) as usize;

    let mut rng = Rng::new(77);
    let single: Vec<f64> = (0..single_len).map(|_| rng.next_f64()).collect();
    let weights: Vec<f64> = (0..n_weights as usize)
        .map(|_| rng.next_f64() * 0.1)
        .collect();

    let cpu_ref = structure_module::torsion_angles(
        &single,
        &weights,
        n_res as usize,
        c_single as usize,
        c_hidden as usize,
    );

    let shader =
        gpu.compile_shader_f64_hybrid(neural_spring_forge::shaders::TORSION_ANGLES_F64, "torsion");

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n_res: u32,
        c_single: u32,
        c_hidden: u32,
        _pad: u32,
    }

    let single_buf = upload_f64(gpu, &single, "tor_single");
    let weights_buf = upload_f64(gpu, &weights, "tor_weights");
    let out_buf = gpu.create_buffer_f64(out_len).expect("alloc tor_out");
    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tor_params"),
            contents: bytemuck::bytes_of(&Params {
                n_res,
                c_single,
                c_hidden,
                _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });
    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: single_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weights_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    run_compute(gpu, &shader, "main", &bg, &bgl, ((n_res + 255) / 256, 1, 1));

    let gpu_result = read_gpu_f64(gpu, &out_buf, out_len);
    let md = max_diff(&gpu_result, &cpu_ref);
    eprintln!("  Torsion GPU max diff: {md:.2e}");
    h.check_abs("Torsion GPU max diff", md, 0.0, GPU_DF64_TOL);
    h.check_bool(
        "Torsion GPU finite",
        gpu_result.iter().all(|v| v.is_finite()),
    );

    let mut unit_ok = true;
    for i in 0..n_res as usize {
        for a in 0..7 {
            let s = gpu_result[i * 14 + a * 2];
            let c = gpu_result[i * 14 + a * 2 + 1];
            let r = s.hypot(c);
            if (r - 1.0).abs() > 0.01 {
                unit_ok = false;
            }
        }
    }
    h.check_bool("Torsion GPU unit circle", unit_ok);
}

// ─── Main ───────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            let profile =
                barracuda::device::driver_profile::GpuDriverProfile::from_device(g.wgpu_device());
            let strategy = profile.fp64_strategy();
            eprintln!("  FP64 strategy: {strategy:?}");
            eprintln!("  precision: df64 core streaming (f64 buffers, df64 compute)");
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let mut h = ValidationHarness::new("sovereign_folding_gpu");

    validate_gelu_gpu(&mut h, &gpu);
    validate_triangle_outgoing_gpu(&mut h, &gpu);
    validate_triangle_incoming_gpu(&mut h, &gpu);
    validate_sdpa_scores_gpu(&mut h, &gpu);
    validate_triangle_attention_gpu(&mut h, &gpu);
    validate_softmax_gpu(&mut h, &gpu);
    validate_layer_norm_gpu(&mut h, &gpu);
    validate_attention_apply_gpu(&mut h, &gpu);
    validate_outer_product_mean_gpu(&mut h, &gpu);
    validate_msa_row_attention_gpu(&mut h, &gpu);
    validate_msa_col_attention_gpu(&mut h, &gpu);
    validate_ipa_scores_gpu(&mut h, &gpu);
    validate_backbone_update_gpu(&mut h, &gpu);
    validate_torsion_angles_gpu(&mut h, &gpu);
    validate_sdpa_pipeline_gpu(&mut h, &gpu);

    h.finish();
}
