// SPDX-License-Identifier: AGPL-3.0-or-later

//! nF-01 Phase B GPU: coralForge Evoformer shader validation (df64 core streaming).
//!
//! Validates Evoformer building-block WGSL shaders reproduce CPU f64 references
//! within df64-class tolerance.  Structure module shaders and the 3-pass SDPA
//! pipeline are validated in `validate_coral_forge_gpu_pipeline`.
//!
//! Three-zone core streaming: f64 buffer I/O → df64 compute on FP32 cores → f64 output.
//!
//! ## Shaders validated
//!
//! | Shader | Algorithm |
//! |--------|-----------|
//! | `gelu_f64.wgsl` | Pointwise GELU |
//! | `triangle_mul_outgoing_f64.wgsl` | Algorithm 11 |
//! | `triangle_mul_incoming_f64.wgsl` | Algorithm 12 |
//! | `triangle_attention_f64.wgsl` | Algorithms 13-14 |
//! | `softmax_f64.wgsl` | Row-wise softmax |
//! | `layer_norm_f64.wgsl` | Layer normalization |
//! | `outer_product_mean_f64.wgsl` | MSA → pair (OPM) |
//! | `msa_row_attention_scores_f64.wgsl` | Row attn + pair bias |
//! | `msa_col_attention_scores_f64.wgsl` | Column attn (no bias) |
//!
//! ## Provenance
//!
//! CPU reference: `neural_spring::coral_forge`.
//! GPU: `metalForge/shaders/` WGSL → `compile_shader_f64_hybrid`.

#![expect(
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    reason = "validation binary"
)]

use neural_spring::coral_forge;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_shader_validation::{
    dispatch_and_read, max_diff, upload_f64, upload_params, wg1d, ShaderBinding,
};
use neural_spring::require;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

// ─── GELU ────────────────────────────────────────────────────────────

fn validate_gelu(h: &mut ValidationHarness, gpu: &Gpu) {
    let n = 64_u32;
    let mut rng = Rng::new(42);
    let input: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
    let cpu_ref: Vec<f64> = input.iter().map(|&x| coral_forge::gelu(x)).collect();

    let shader = gpu.compile_shader_f64_hybrid(neural_spring_forge::shaders::GELU_F64, "gelu_f64");

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        n: u32,
        _pad: [u32; 3],
    }

    let in_buf = upload_f64(gpu, &input, "gelu_in");
    let out_buf = require!(h, gpu.create_buffer_f64(n as usize), "alloc gelu_out");
    let params_buf = upload_params(gpu, &P { n, _pad: [0; 3] }, "gelu_params");

    let result = dispatch_and_read(
        gpu,
        &shader,
        "main",
        &[
            ShaderBinding::StorageRo(&in_buf),
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params_buf),
        ],
        wg1d(n),
        n as usize,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    eprintln!("  GELU GPU max diff: {md:.2e}");
    h.check_abs(
        "GELU GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_TRANSCENDENTAL,
    );
    h.check_bool("GELU GPU finite", result.iter().all(|v| v.is_finite()));
}

// ─── Triangle mul outgoing ───────────────────────────────────────────

fn validate_triangle_outgoing(h: &mut ValidationHarness, gpu: &Gpu) {
    let (n, c) = (8_u32, 4_u32);
    let total = (n * n * c) as usize;
    let mut rng = Rng::new(42);
    let proj_a: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
    let proj_b: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
    let cpu_ref = coral_forge::triangle_mul_outgoing(&proj_a, &proj_b, n as usize, c as usize);

    let shader = gpu.compile_shader_f64_hybrid(
        neural_spring_forge::shaders::TRIANGLE_MUL_OUTGOING_F64,
        "tri_out",
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        n_res: u32,
        channels: u32,
        _pad: [u32; 2],
    }

    let a_buf = upload_f64(gpu, &proj_a, "tri_a");
    let b_buf = upload_f64(gpu, &proj_b, "tri_b");
    let out_buf = require!(h, gpu.create_buffer_f64(total), "alloc tri_out");
    let params = upload_params(
        gpu,
        &P {
            n_res: n,
            channels: c,
            _pad: [0; 2],
        },
        "tri_params",
    );

    let result = dispatch_and_read(
        gpu,
        &shader,
        "main",
        &[
            ShaderBinding::StorageRo(&a_buf),
            ShaderBinding::StorageRo(&b_buf),
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params),
        ],
        wg1d(total as u32),
        total,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    eprintln!("  TriMul outgoing GPU max diff: {md:.2e}");
    h.check_abs(
        "TriMul outgoing GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_ARITHMETIC,
    );
    h.check_bool(
        "TriMul outgoing GPU finite",
        result.iter().all(|v| v.is_finite()),
    );
}

// ─── Triangle mul incoming ───────────────────────────────────────────

fn validate_triangle_incoming(h: &mut ValidationHarness, gpu: &Gpu) {
    let (n, c) = (8_u32, 4_u32);
    let total = (n * n * c) as usize;
    let mut rng = Rng::new(99);
    let proj_a: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
    let proj_b: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
    let cpu_ref = coral_forge::triangle_mul_incoming(&proj_a, &proj_b, n as usize, c as usize);

    let shader = gpu.compile_shader_f64_hybrid(
        neural_spring_forge::shaders::TRIANGLE_MUL_INCOMING_F64,
        "tri_in",
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        n_res: u32,
        channels: u32,
        _pad: [u32; 2],
    }

    let a_buf = upload_f64(gpu, &proj_a, "tri_a");
    let b_buf = upload_f64(gpu, &proj_b, "tri_b");
    let out_buf = require!(h, gpu.create_buffer_f64(total), "alloc tri_out");
    let params = upload_params(
        gpu,
        &P {
            n_res: n,
            channels: c,
            _pad: [0; 2],
        },
        "tri_params",
    );

    let result = dispatch_and_read(
        gpu,
        &shader,
        "main",
        &[
            ShaderBinding::StorageRo(&a_buf),
            ShaderBinding::StorageRo(&b_buf),
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params),
        ],
        wg1d(total as u32),
        total,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    eprintln!("  TriMul incoming GPU max diff: {md:.2e}");
    h.check_abs(
        "TriMul incoming GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_ARITHMETIC,
    );
    h.check_bool(
        "TriMul incoming GPU finite",
        result.iter().all(|v| v.is_finite()),
    );
}

// ─── Triangle attention scores ───────────────────────────────────────

fn validate_triangle_attention(h: &mut ValidationHarness, gpu: &Gpu) {
    let (r, n, he, d) = (4_u32, 8_u32, 2_u32, 4_u32);
    let qk_len = (r * n * he * d) as usize;
    let bias_len = (he * n * n) as usize;
    let out_len = (r * he * n * n) as usize;

    let mut rng = Rng::new(77);
    let q: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let k: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let bias: Vec<f64> = (0..bias_len).map(|_| rng.next_f64() * 0.1).collect();
    let cpu_ref = coral_forge::triangle_attention_scores(
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
    struct P {
        n_rows: u32,
        n_res: u32,
        n_heads: u32,
        head_dim: u32,
    }

    let q_buf = upload_f64(gpu, &q, "q");
    let k_buf = upload_f64(gpu, &k, "k");
    let bias_buf = upload_f64(gpu, &bias, "bias");
    let out_buf = require!(h, gpu.create_buffer_f64(out_len), "alloc tri_attn_out");
    let params = upload_params(
        gpu,
        &P {
            n_rows: r,
            n_res: n,
            n_heads: he,
            head_dim: d,
        },
        "params",
    );

    let result = dispatch_and_read(
        gpu,
        &shader,
        "main",
        &[
            ShaderBinding::StorageRo(&q_buf),
            ShaderBinding::StorageRo(&k_buf),
            ShaderBinding::StorageRo(&bias_buf),
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params),
        ],
        wg1d(out_len as u32),
        out_len,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    eprintln!("  TriAttn scores GPU max diff: {md:.2e}");
    h.check_abs(
        "TriAttn scores GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_ARITHMETIC,
    );
    h.check_bool(
        "TriAttn scores GPU finite",
        result.iter().all(|v| v.is_finite()),
    );
}

// ─── Softmax ─────────────────────────────────────────────────────────

fn validate_softmax(h: &mut ValidationHarness, gpu: &Gpu) {
    let (rows, cols) = (4_u32, 16_u32);
    let total = (rows * cols) as usize;

    let mut rng = Rng::new(55);
    let input: Vec<f64> = (0..total).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
    let cpu_ref = coral_forge::softmax_rows(&input, rows as usize, cols as usize);

    let shader =
        gpu.compile_shader_f64_hybrid(neural_spring_forge::shaders::SOFTMAX_F64, "softmax");

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        num_rows: u32,
        row_len: u32,
        _pad: [u32; 2],
    }

    let in_buf = upload_f64(gpu, &input, "sm_in");
    let out_buf = require!(h, gpu.create_buffer_f64(total), "alloc sm_out");
    let params = upload_params(
        gpu,
        &P {
            num_rows: rows,
            row_len: cols,
            _pad: [0; 2],
        },
        "sm_params",
    );

    let result = dispatch_and_read(
        gpu,
        &shader,
        "main",
        &[
            ShaderBinding::StorageRo(&in_buf),
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params),
        ],
        (rows, 1, 1),
        total,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    eprintln!("  Softmax GPU max diff: {md:.2e}");
    h.check_abs(
        "Softmax GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_TRANSCENDENTAL,
    );

    for row in 0..rows as usize {
        let sum: f64 = result[row * cols as usize..(row + 1) * cols as usize]
            .iter()
            .sum();
        h.check_abs(
            &format!("Softmax GPU row {row} sum"),
            sum,
            1.0,
            tolerances::GPU_DF64_TRANSCENDENTAL,
        );
    }
}

// ─── Layer Norm ──────────────────────────────────────────────────────

fn validate_layer_norm(h: &mut ValidationHarness, gpu: &Gpu) {
    let (seq_len, hidden_dim) = (8_u32, 16_u32);
    let total = (seq_len * hidden_dim) as usize;

    let mut rng = Rng::new(42);
    let input: Vec<f64> = (0..total).map(|_| rng.next_f64()).collect();
    let gamma: Vec<f64> = (0..hidden_dim).map(|_| 0.5 + rng.next_f64()).collect();
    let beta: Vec<f64> = (0..hidden_dim)
        .map(|_| rng.next_f64() * 0.2 - 0.1)
        .collect();
    let cpu_ref = coral_forge::layer_norm(
        &input,
        seq_len as usize,
        hidden_dim as usize,
        &gamma,
        &beta,
        tolerances::LAYER_NORM_EPS,
    );

    let shader =
        gpu.compile_shader_f64_hybrid(neural_spring_forge::shaders::LAYER_NORM_F64, "layer_norm");

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        seq_len: u32,
        hidden_dim: u32,
        eps_hi: f32,
        eps_lo: f32,
    }

    #[expect(clippy::cast_possible_truncation, reason = "validation binary")]
    let eps_hi = tolerances::LAYER_NORM_EPS as f32;
    let in_buf = upload_f64(gpu, &input, "ln_in");
    let gamma_buf = upload_f64(gpu, &gamma, "gamma");
    let beta_buf = upload_f64(gpu, &beta, "beta");
    let out_buf = require!(h, gpu.create_buffer_f64(total), "alloc ln_out");
    let params = upload_params(
        gpu,
        &P {
            seq_len,
            hidden_dim,
            eps_hi,
            eps_lo: 0.0,
        },
        "ln_params",
    );

    let result = dispatch_and_read(
        gpu,
        &shader,
        "layer_norm",
        &[
            ShaderBinding::StorageRo(&in_buf),
            ShaderBinding::StorageRo(&gamma_buf),
            ShaderBinding::StorageRo(&beta_buf),
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params),
        ],
        (seq_len, 1, 1),
        total,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    eprintln!("  LayerNorm GPU max diff: {md:.2e}");
    h.check_abs(
        "LayerNorm GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_ARITHMETIC,
    );
    h.check_bool("LayerNorm GPU finite", result.iter().all(|v| v.is_finite()));
}

// ─── Outer Product Mean ──────────────────────────────────────────────

fn validate_outer_product_mean(h: &mut ValidationHarness, gpu: &Gpu) {
    let (n_seq, n_res, c_a, c_b) = (6_u32, 4_u32, 3_u32, 2_u32);
    let a_len = (n_seq * n_res * c_a) as usize;
    let b_len = (n_seq * n_res * c_b) as usize;
    let out_len = (n_res * n_res * c_a * c_b) as usize;

    let mut rng = Rng::new(42);
    let a: Vec<f64> = (0..a_len).map(|_| rng.next_f64()).collect();
    let bv: Vec<f64> = (0..b_len).map(|_| rng.next_f64()).collect();
    let cpu_ref = coral_forge::outer_product_mean(
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
    struct P {
        n_seq: u32,
        n_res: u32,
        c_a: u32,
        c_b: u32,
    }

    let a_buf = upload_f64(gpu, &a, "opm_a");
    let b_buf = upload_f64(gpu, &bv, "opm_b");
    let out_buf = require!(h, gpu.create_buffer_f64(out_len), "alloc opm_out");
    let params = upload_params(
        gpu,
        &P {
            n_seq,
            n_res,
            c_a,
            c_b,
        },
        "opm_params",
    );

    let result = dispatch_and_read(
        gpu,
        &shader,
        "main",
        &[
            ShaderBinding::StorageRo(&a_buf),
            ShaderBinding::StorageRo(&b_buf),
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params),
        ],
        wg1d(out_len as u32),
        out_len,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    eprintln!("  OPM GPU max diff: {md:.2e}");
    h.check_abs("OPM GPU max diff", md, 0.0, tolerances::GPU_DF64_ARITHMETIC);
    h.check_bool("OPM GPU finite", result.iter().all(|v| v.is_finite()));
}

// ─── MSA Row Attention Scores ────────────────────────────────────────

fn validate_msa_row_attention(h: &mut ValidationHarness, gpu: &Gpu) {
    let (n_seq, n_res, n_heads, head_dim) = (4_u32, 6_u32, 2_u32, 4_u32);
    let qk_len = (n_seq * n_res * n_heads * head_dim) as usize;
    let bias_len = (n_heads * n_res * n_res) as usize;
    let out_len = (n_seq * n_heads * n_res * n_res) as usize;

    let mut rng = Rng::new(42);
    let q: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let k: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let bias: Vec<f64> = (0..bias_len).map(|_| rng.next_f64() * 0.1).collect();
    let cpu_ref = coral_forge::msa_row_attention_scores(
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
    struct P {
        n_seq: u32,
        n_res: u32,
        n_heads: u32,
        head_dim: u32,
    }

    let q_buf = upload_f64(gpu, &q, "msa_row_q");
    let k_buf = upload_f64(gpu, &k, "msa_row_k");
    let bias_buf = upload_f64(gpu, &bias, "msa_row_bias");
    let out_buf = require!(h, gpu.create_buffer_f64(out_len), "alloc msa_row_out");
    let params = upload_params(
        gpu,
        &P {
            n_seq,
            n_res,
            n_heads,
            head_dim,
        },
        "msa_row_params",
    );

    let result = dispatch_and_read(
        gpu,
        &shader,
        "main",
        &[
            ShaderBinding::StorageRo(&q_buf),
            ShaderBinding::StorageRo(&k_buf),
            ShaderBinding::StorageRo(&bias_buf),
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params),
        ],
        wg1d(out_len as u32),
        out_len,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    eprintln!("  MSA row scores GPU max diff: {md:.2e}");
    h.check_abs(
        "MSA row scores GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_ARITHMETIC,
    );
    h.check_bool(
        "MSA row scores GPU finite",
        result.iter().all(|v| v.is_finite()),
    );
}

// ─── MSA Column Attention Scores ─────────────────────────────────────

fn validate_msa_col_attention(h: &mut ValidationHarness, gpu: &Gpu) {
    let (n_seq, n_res, n_heads, head_dim) = (4_u32, 6_u32, 2_u32, 4_u32);
    let qk_len = (n_seq * n_res * n_heads * head_dim) as usize;
    let out_len = (n_res * n_heads * n_seq * n_seq) as usize;

    let mut rng = Rng::new(77);
    let q: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let k: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let cpu_ref = coral_forge::msa_col_attention_scores(
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
    struct P {
        n_seq: u32,
        n_res: u32,
        n_heads: u32,
        head_dim: u32,
    }

    let q_buf = upload_f64(gpu, &q, "msa_col_q");
    let k_buf = upload_f64(gpu, &k, "msa_col_k");
    let out_buf = require!(h, gpu.create_buffer_f64(out_len), "alloc msa_col_out");
    let params = upload_params(
        gpu,
        &P {
            n_seq,
            n_res,
            n_heads,
            head_dim,
        },
        "msa_col_params",
    );

    let result = dispatch_and_read(
        gpu,
        &shader,
        "main",
        &[
            ShaderBinding::StorageRo(&q_buf),
            ShaderBinding::StorageRo(&k_buf),
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params),
        ],
        wg1d(out_len as u32),
        out_len,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    eprintln!("  MSA col scores GPU max diff: {md:.2e}");
    h.check_abs(
        "MSA col scores GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_ARITHMETIC,
    );
    h.check_bool(
        "MSA col scores GPU finite",
        result.iter().all(|v| v.is_finite()),
    );
}

// ─── Main ────────────────────────────────────────────────────────────

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
            eprintln!("  FP64 strategy: {:?}", profile.fp64_strategy());
            eprintln!("  precision: df64 core streaming (f64 buffers, df64 compute)");
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let mut h = ValidationHarness::new("coral_forge_gpu");

    validate_gelu(&mut h, &gpu);
    validate_triangle_outgoing(&mut h, &gpu);
    validate_triangle_incoming(&mut h, &gpu);
    validate_triangle_attention(&mut h, &gpu);
    validate_softmax(&mut h, &gpu);
    validate_layer_norm(&mut h, &gpu);
    validate_outer_product_mean(&mut h, &gpu);
    validate_msa_row_attention(&mut h, &gpu);
    validate_msa_col_attention(&mut h, &gpu);

    h.finish();
}
