// SPDX-License-Identifier: AGPL-3.0-or-later

//! nF-01 Phase B GPU: coralForge attention pipelines + structure module
//! shader validation (df64 core streaming).
//!
//! Validates the multi-pass SDPA pipeline and structure module WGSL shaders
//! reproduce CPU f64 references within df64-class tolerance.  Evoformer
//! building-block shaders are validated in `validate_coral_forge_gpu`.
//!
//! ## Shaders validated
//!
//! | Shader | Algorithm |
//! |--------|-----------|
//! | `sdpa_scores_f64.wgsl` | QKᵀ/√d (pass 1) |
//! | `attention_apply_f64.wgsl` | Σ weights × V (pass 3) |
//! | 3-pass SDPA pipeline | scores→softmax→apply (full chain) |
//! | `ipa_scores_f64.wgsl` | IPA (SE(3)-equivariant) |
//! | `backbone_update_f64.wgsl` | Frame composition |
//! | `torsion_angles_f64.wgsl` | Fused `ResNet` + normalize |
//!
//! ## Provenance
//!
//! CPU reference: `neural_spring::coral_forge` + `neural_spring::coral_forge::structure`.
//! GPU: `metalForge/shaders/` WGSL → `compile_shader_f64_hybrid`.

#![expect(
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    reason = "validation binary"
)]

use neural_spring::coral_forge;
use neural_spring::coral_forge::structure;
use neural_spring::coral_forge::structure::IpaConfig;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_shader_validation::{
    ShaderBinding, dispatch_and_read, dispatch_shader, max_diff, upload_f64, upload_params, wg1d,
};
use neural_spring::require;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

// ─── Attention params (shared by SDPA scores, apply, and pipeline) ───

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnParams {
    batch_size: u32,
    num_heads: u32,
    q_seq_len: u32,
    kv_seq_len: u32,
    head_dim: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxParams {
    num_rows: u32,
    row_len: u32,
    _pad: [u32; 2],
}

// ─── SDPA scores ─────────────────────────────────────────────────────

fn validate_sdpa_scores(h: &mut ValidationHarness, gpu: &Gpu) {
    let (b, he, sq, skv, d) = (1_u32, 2_u32, 8_u32, 8_u32, 4_u32);
    let q_len = (b * he * sq * d) as usize;
    let k_len = (b * he * skv * d) as usize;
    let out_len = (b * he * sq * skv) as usize;

    let mut rng = Rng::new(42);
    let q: Vec<f64> = (0..q_len).map(|_| rng.next_f64()).collect();
    let k: Vec<f64> = (0..k_len).map(|_| rng.next_f64()).collect();
    let cpu_ref = coral_forge::sdpa_scores(
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

    let q_buf = upload_f64(gpu, &q, "q");
    let k_buf = upload_f64(gpu, &k, "k");
    let out_buf = require!(h, gpu.create_buffer_f64(out_len), "alloc scores");
    let params = upload_params(
        gpu,
        &AttnParams {
            batch_size: b,
            num_heads: he,
            q_seq_len: sq,
            kv_seq_len: skv,
            head_dim: d,
            _pad: [0; 3],
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
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params),
        ],
        wg1d(out_len as u32),
        out_len,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    println!("  SDPA scores GPU max diff: {md:.2e}");
    h.check_abs(
        "SDPA scores GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_ARITHMETIC,
    );
    h.check_bool(
        "SDPA scores GPU finite",
        result.iter().all(|v| v.is_finite()),
    );
}

// ─── Attention Apply ─────────────────────────────────────────────────

fn validate_attention_apply(h: &mut ValidationHarness, gpu: &Gpu) {
    let (b, he, sq, skv, d) = (1_u32, 2_u32, 8_u32, 8_u32, 4_u32);
    let w_len = (b * he * sq * skv) as usize;
    let v_len = (b * he * skv * d) as usize;
    let out_len = (b * he * sq * d) as usize;

    let mut rng = Rng::new(42);
    let weights_raw: Vec<f64> = (0..w_len).map(|_| rng.next_f64()).collect();
    let weights_rows = (b * he * sq) as usize;
    let weights = coral_forge::softmax_rows(&weights_raw, weights_rows, skv as usize);
    let v: Vec<f64> = (0..v_len).map(|_| rng.next_f64()).collect();
    let cpu_ref = coral_forge::attention_apply(
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

    let w_buf = upload_f64(gpu, &weights, "weights");
    let v_buf = upload_f64(gpu, &v, "value");
    let out_buf = require!(h, gpu.create_buffer_f64(out_len), "alloc attn_out");
    let params = upload_params(
        gpu,
        &AttnParams {
            batch_size: b,
            num_heads: he,
            q_seq_len: sq,
            kv_seq_len: skv,
            head_dim: d,
            _pad: [0; 3],
        },
        "params",
    );

    let result = dispatch_and_read(
        gpu,
        &shader,
        "main",
        &[
            ShaderBinding::StorageRo(&w_buf),
            ShaderBinding::StorageRo(&v_buf),
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params),
        ],
        (d.div_ceil(16), sq.div_ceil(16), b * he),
        out_len,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    println!("  AttnApply GPU max diff: {md:.2e}");
    h.check_abs(
        "AttnApply GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_ARITHMETIC,
    );
    h.check_bool("AttnApply GPU finite", result.iter().all(|v| v.is_finite()));
}

// ─── Full 3-pass SDPA Pipeline ───────────────────────────────────────

#[expect(clippy::too_many_lines, reason = "validation binary")]
fn validate_sdpa_pipeline(h: &mut ValidationHarness, gpu: &Gpu) {
    let (b, he, sq, skv, d) = (1_u32, 2_u32, 8_u32, 8_u32, 4_u32);
    let qk_len = (b * he * sq * d) as usize;
    let v_len = (b * he * skv * d) as usize;
    let scores_len = (b * he * sq * skv) as usize;
    let out_len = (b * he * sq * d) as usize;

    let mut rng = Rng::new(42);
    let q: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let k: Vec<f64> = (0..qk_len).map(|_| rng.next_f64()).collect();
    let v: Vec<f64> = (0..v_len).map(|_| rng.next_f64()).collect();
    let cpu_ref = coral_forge::sdpa_full(
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

    let attn_params = AttnParams {
        batch_size: b,
        num_heads: he,
        q_seq_len: sq,
        kv_seq_len: skv,
        head_dim: d,
        _pad: [0; 3],
    };

    let q_buf = upload_f64(gpu, &q, "q");
    let k_buf = upload_f64(gpu, &k, "k");
    let v_buf = upload_f64(gpu, &v, "v");
    let scores_buf = require!(h, gpu.create_buffer_f64(scores_len), "alloc scores");
    let weights_buf = require!(h, gpu.create_buffer_f64(scores_len), "alloc weights");
    let output_buf = require!(h, gpu.create_buffer_f64(out_len), "alloc output");

    // Pass 1: scores = Q @ K^T / sqrt(d)
    let p1_params = upload_params(gpu, &attn_params, "p1_params");
    dispatch_shader(
        gpu,
        &scores_shader,
        "main",
        &[
            ShaderBinding::StorageRo(&q_buf),
            ShaderBinding::StorageRo(&k_buf),
            ShaderBinding::StorageRw(&scores_buf),
            ShaderBinding::Uniform(&p1_params),
        ],
        wg1d(scores_len as u32),
    );

    // Pass 2: weights = softmax(scores)
    let sm_rows = b * he * sq;
    let sm_params = upload_params(
        gpu,
        &SoftmaxParams {
            num_rows: sm_rows,
            row_len: skv,
            _pad: [0; 2],
        },
        "sm_params",
    );
    dispatch_shader(
        gpu,
        &softmax_shader,
        "main",
        &[
            ShaderBinding::StorageRo(&scores_buf),
            ShaderBinding::StorageRw(&weights_buf),
            ShaderBinding::Uniform(&sm_params),
        ],
        (sm_rows, 1, 1),
    );

    // Pass 3: output = weights @ V
    let p3_params = upload_params(gpu, &attn_params, "p3_params");
    let result = dispatch_and_read(
        gpu,
        &apply_shader,
        "main",
        &[
            ShaderBinding::StorageRo(&weights_buf),
            ShaderBinding::StorageRo(&v_buf),
            ShaderBinding::StorageRw(&output_buf),
            ShaderBinding::Uniform(&p3_params),
        ],
        (d.div_ceil(16), sq.div_ceil(16), b * he),
        out_len,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    println!("  SDPA pipeline GPU max diff: {md:.2e}");
    h.check_abs(
        "SDPA pipeline GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_TRANSCENDENTAL,
    );
    h.check_bool(
        "SDPA pipeline GPU finite",
        result.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "SDPA pipeline GPU nonzero",
        result.iter().any(|v| *v != 0.0),
    );
}

// ─── IPA Scores ──────────────────────────────────────────────────────

fn validate_ipa_scores(h: &mut ValidationHarness, gpu: &Gpu) {
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

    let cfg = IpaConfig {
        n_res: n_res as usize,
        n_heads: n_heads as usize,
        head_dim: head_dim as usize,
        n_points: n_points as usize,
        w_l: f64::from(w_l),
        w_c: f64::from(w_c),
        w_p: f64::from(w_p),
        gamma: f64::from(gamma),
    };
    let cpu_ref = structure::ipa_scores(&q, &k, &bias, &qp, &kp, &frames, &cfg);

    let shader = gpu.compile_shader_f64_hybrid(neural_spring_forge::shaders::IPA_SCORES_F64, "ipa");

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct IpaParams {
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
    let out_buf = require!(h, gpu.create_buffer_f64(out_len), "alloc ipa_out");
    let params = upload_params(
        gpu,
        &IpaParams {
            n_res,
            n_heads,
            head_dim,
            n_points,
            w_l,
            w_c,
            w_p,
            gamma,
        },
        "ipa_params",
    );

    let result = dispatch_and_read(
        gpu,
        &shader,
        "main",
        &[
            ShaderBinding::StorageRo(&q_buf),
            ShaderBinding::StorageRo(&k_buf),
            ShaderBinding::StorageRo(&bias_buf),
            ShaderBinding::StorageRo(&qp_buf),
            ShaderBinding::StorageRo(&kp_buf),
            ShaderBinding::StorageRo(&frames_buf),
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params),
        ],
        wg1d(out_len as u32),
        out_len,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    println!("  IPA scores GPU max diff: {md:.2e}");
    h.check_abs(
        "IPA scores GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_ARITHMETIC,
    );
    h.check_bool(
        "IPA scores GPU finite",
        result.iter().all(|v| v.is_finite()),
    );
    h.check_bool("IPA scores GPU nonzero", result.iter().any(|v| *v != 0.0));
}

// ─── Backbone Update ─────────────────────────────────────────────────

fn validate_backbone_update(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_res = 4_u32;
    let frames_len = (n_res * 12) as usize;

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

    let quats_len = (n_res * 4) as usize;
    let trans_len = (n_res * 3) as usize;
    let mut dq: Vec<f64> = (0..quats_len).map(|_| rng.next_f64() * 0.1).collect();
    for i in 0..n_res as usize {
        dq[i * 4] += 1.0;
    }
    let dt: Vec<f64> = (0..trans_len).map(|_| rng.next_f64() * 0.1).collect();

    let cpu_ref = structure::backbone_update(&dq, &dt, &cur_frames, n_res as usize);

    let shader = gpu.compile_shader_f64_hybrid(
        neural_spring_forge::shaders::BACKBONE_UPDATE_F64,
        "bb_update",
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        n_res: u32,
        _pad: [u32; 3],
    }

    let dq_buf = upload_f64(gpu, &dq, "bb_dq");
    let dt_buf = upload_f64(gpu, &dt, "bb_dt");
    let cur_buf = upload_f64(gpu, &cur_frames, "bb_cur");
    let out_buf = require!(h, gpu.create_buffer_f64(frames_len), "alloc bb_out");
    let params = upload_params(
        gpu,
        &P {
            n_res,
            _pad: [0; 3],
        },
        "bb_params",
    );

    let result = dispatch_and_read(
        gpu,
        &shader,
        "main",
        &[
            ShaderBinding::StorageRo(&dq_buf),
            ShaderBinding::StorageRo(&dt_buf),
            ShaderBinding::StorageRo(&cur_buf),
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params),
        ],
        wg1d(n_res),
        frames_len,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    println!("  Backbone GPU max diff: {md:.2e}");
    h.check_abs(
        "Backbone GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_ARITHMETIC,
    );
    h.check_bool("Backbone GPU finite", result.iter().all(|v| v.is_finite()));
    h.check_bool("Backbone GPU nonzero", result.iter().any(|v| *v != 0.0));
}

// ─── Torsion Angles ──────────────────────────────────────────────────

fn validate_torsion_angles(h: &mut ValidationHarness, gpu: &Gpu) {
    let (n_res, c_single, c_hidden) = (4_u32, 8_u32, 6_u32);
    let n_weights =
        c_single * c_hidden + c_hidden + 4 * (c_hidden * c_hidden + c_hidden) + c_hidden * 14 + 14;
    let single_len = (n_res * c_single) as usize;
    let out_len = (n_res * 14) as usize;

    let mut rng = Rng::new(77);
    let single: Vec<f64> = (0..single_len).map(|_| rng.next_f64()).collect();
    let weights: Vec<f64> = (0..n_weights as usize)
        .map(|_| rng.next_f64() * 0.1)
        .collect();

    let cpu_ref = structure::torsion_angles(
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
    struct P {
        n_res: u32,
        c_single: u32,
        c_hidden: u32,
        _pad: u32,
    }

    let single_buf = upload_f64(gpu, &single, "tor_single");
    let weights_buf = upload_f64(gpu, &weights, "tor_weights");
    let out_buf = require!(h, gpu.create_buffer_f64(out_len), "alloc tor_out");
    let params = upload_params(
        gpu,
        &P {
            n_res,
            c_single,
            c_hidden,
            _pad: 0,
        },
        "tor_params",
    );

    let result = dispatch_and_read(
        gpu,
        &shader,
        "main",
        &[
            ShaderBinding::StorageRo(&single_buf),
            ShaderBinding::StorageRo(&weights_buf),
            ShaderBinding::StorageRw(&out_buf),
            ShaderBinding::Uniform(&params),
        ],
        wg1d(n_res),
        out_len,
    )
    .expect("dispatch_and_read");

    let md = max_diff(&result, &cpu_ref);
    println!("  Torsion GPU max diff: {md:.2e}");
    h.check_abs(
        "Torsion GPU max diff",
        md,
        0.0,
        tolerances::GPU_DF64_ARITHMETIC,
    );
    h.check_bool("Torsion GPU finite", result.iter().all(|v| v.is_finite()));

    let mut unit_ok = true;
    for i in 0..n_res as usize {
        for a in 0..7 {
            let s = result[i * 14 + a * 2];
            let c = result[i * 14 + a * 2 + 1];
            if (s.hypot(c) - 1.0).abs() > 0.01 {
                unit_ok = false;
            }
        }
    }
    h.check_bool("Torsion GPU unit circle", unit_ok);
}

// ─── Main ────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            println!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            let caps =
                barracuda::device::capabilities::DeviceCapabilities::from_device(g.wgpu_device());
            println!("  FP64 strategy: {:?}", caps.fp64_strategy());
            println!("  precision: df64 core streaming (f64 buffers, df64 compute)");
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let mut h = ValidationHarness::new("coral_forge_gpu_pipeline");

    validate_sdpa_scores(&mut h, &gpu);
    validate_attention_apply(&mut h, &gpu);
    validate_sdpa_pipeline(&mut h, &gpu);
    validate_ipa_scores(&mut h, &gpu);
    validate_backbone_update(&mut h, &gpu);
    validate_torsion_angles(&mut h, &gpu);

    h.finish();
}
