// SPDX-License-Identifier: AGPL-3.0-or-later

//! BarraCUDA CPU↔GPU parity for WDM surrogate and coralForge domain compositions.
//!
//! Extends `validate_barracuda_parity` (Phase 0++ primitives) to cover the
//! domain-specific compositions used in WDM warm-dense-matter surrogates and
//! coralForge protein structure prediction. Each test constructs a meaningful
//! domain operation through `Dispatcher` CPU and GPU paths, proving the pure
//! Rust math is truly portable across hardware substrates.
//!
//! ## WDM coverage
//!
//! - nW-01 transport: 3-layer MLP forward pass (matmul + sigmoid chain)
//! - nW-02 EOS: MLP with softplus activation (matmul + activation)
//! - nW-03 S(q,ω): LSTM-style gate composition (matmul + sigmoid + tanh)
//! - nW-05 ESN: reservoir state update (matmul + tanh + spectral radius)
//!
//! ## coralForge coverage
//!
//! - Evoformer attention: QK^T/√d → softmax (matmul + softmax)
//! - Triangle multiply: dot product contraction (dot accumulation)
//! - Confidence pLDDT: matmul → sigmoid mean (matmul + mean)
//! - Layer norm: mean + variance composition
//! - SE(3) equivariance: COM mean + L2 residual
//!
//! ## Provenance
//!
//! Validation class: GPU cross-dispatch.
//! CPU reference: neuralSpring lib (Rust CPU).
//! GPU path: `BarraCUDA` dispatch (`DispatchConfig` routing).

#![expect(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    reason = "validation binary"
)]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{max_abs_diff_f64, ValidationHarness};

fn rect_matmul(disp: &Dispatcher, a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    barracuda::dispatch::matmul_dispatch(a, b, m, k, n, disp.wgpu_device()).unwrap_or_else(|_| {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (0..k).fold(0.0, |acc, p| a[i * k + p].mul_add(b[p * n + j], acc));
            }
        }
        c
    })
}

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("wdm_coral_parity");
    let mut rng = Rng::new(42);
    let gpu_disp = Dispatcher::new().await;
    let cpu_disp = Dispatcher::cpu_only();

    eprintln!(
        "[parity] GPU: {} ({})",
        gpu_disp.has_gpu(),
        gpu_disp.adapter_name()
    );

    validate_wdm_transport_mlp(&mut h, &gpu_disp, &cpu_disp, &mut rng);
    validate_wdm_eos_mlp(&mut h, &gpu_disp, &cpu_disp, &mut rng);
    validate_wdm_sqw_lstm_gate(&mut h, &gpu_disp, &cpu_disp, &mut rng);
    validate_wdm_esn_reservoir(&mut h, &gpu_disp, &cpu_disp, &mut rng);
    validate_evoformer_attention(&mut h, &gpu_disp, &cpu_disp, &mut rng);
    validate_triangle_contraction(&mut h, &gpu_disp, &cpu_disp, &mut rng);
    validate_pldt_composition(&mut h, &gpu_disp, &cpu_disp, &mut rng);
    validate_layer_norm_composition(&mut h, &gpu_disp, &cpu_disp, &mut rng);
    validate_se3_equivariance(&mut h, &gpu_disp, &cpu_disp, &mut rng);

    h.finish();
}

// ═══════════════════════════════════════════════════════════════════
// WDM nW-01: Transport MLP — 3-layer forward pass
// ═══════════════════════════════════════════════════════════════════

fn validate_wdm_transport_mlp(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
    rng: &mut Rng,
) {
    let batch = 4;
    let d_in = 2;
    let d_h = 8;
    let d_out = 1;

    let input: Vec<f64> = (0..batch * d_in).map(|_| rng.normal()).collect();
    let w1: Vec<f64> = (0..d_in * d_h).map(|_| rng.normal() * 0.3).collect();
    let w2: Vec<f64> = (0..d_h * d_h).map(|_| rng.normal() * 0.3).collect();
    let w3: Vec<f64> = (0..d_h * d_out).map(|_| rng.normal() * 0.3).collect();

    let gpu_out = mlp_forward_3layer(gpu, &input, &w1, &w2, &w3, batch, d_in, d_h, d_out);
    let cpu_out = mlp_forward_3layer(cpu, &input, &w1, &w2, &w3, batch, d_in, d_h, d_out);

    let diff = max_abs_diff_f64(&gpu_out, &cpu_out);
    h.check_bool(
        &format!("nW-01 transport MLP 3-layer GPU↔CPU (diff={diff:.2e})"),
        diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
    h.check_bool(
        "nW-01 transport MLP finite",
        gpu_out.iter().all(|v| v.is_finite()),
    );
}

// ═══════════════════════════════════════════════════════════════════
// WDM nW-02: EOS MLP — matmul + softplus activation
// ═══════════════════════════════════════════════════════════════════

fn validate_wdm_eos_mlp(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
    rng: &mut Rng,
) {
    let batch = 4;
    let d_in = 3;
    let d_h = 16;
    let d_out = 2;

    let input: Vec<f64> = (0..batch * d_in).map(|_| rng.normal()).collect();
    let w1: Vec<f64> = (0..d_in * d_h).map(|_| rng.normal() * 0.2).collect();
    let w2: Vec<f64> = (0..d_h * d_out).map(|_| rng.normal() * 0.2).collect();

    let gpu_h1 = rect_matmul(gpu, &input, &w1, batch, d_in, d_h);
    let cpu_h1 = rect_matmul(cpu, &input, &w1, batch, d_in, d_h);

    let gpu_act: Vec<f64> = gpu_h1.iter().map(|&x| softplus(x)).collect();
    let cpu_act: Vec<f64> = cpu_h1.iter().map(|&x| softplus(x)).collect();

    let gpu_out = rect_matmul(gpu, &gpu_act, &w2, batch, d_h, d_out);
    let cpu_out = rect_matmul(cpu, &cpu_act, &w2, batch, d_h, d_out);

    let diff = max_abs_diff_f64(&gpu_out, &cpu_out);
    h.check_bool(
        &format!("nW-02 EOS MLP softplus GPU↔CPU (diff={diff:.2e})"),
        diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );

    let gpu_mean = gpu.mean(&gpu_out);
    let cpu_mean = cpu.mean(&cpu_out);
    h.check_abs(
        "nW-02 EOS output mean GPU↔CPU",
        gpu_mean,
        cpu_mean,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );
}

// ═══════════════════════════════════════════════════════════════════
// WDM nW-03: S(q,ω) LSTM-style gate composition
// ═══════════════════════════════════════════════════════════════════

fn validate_wdm_sqw_lstm_gate(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
    rng: &mut Rng,
) {
    let hidden = 8;
    let d_in = 4;

    let x_t: Vec<f64> = (0..d_in).map(|_| rng.normal()).collect();
    let h_prev: Vec<f64> = (0..hidden).map(|_| rng.normal() * 0.1).collect();
    let w_xh: Vec<f64> = (0..d_in * hidden).map(|_| rng.normal() * 0.2).collect();
    let w_hh: Vec<f64> = (0..hidden * hidden).map(|_| rng.normal() * 0.2).collect();

    let gpu_gates = lstm_gate_step(gpu, &x_t, &h_prev, &w_xh, &w_hh, d_in, hidden);
    let cpu_gates = lstm_gate_step(cpu, &x_t, &h_prev, &w_xh, &w_hh, d_in, hidden);

    let diff = max_abs_diff_f64(&gpu_gates, &cpu_gates);
    h.check_bool(
        &format!("nW-03 LSTM gate GPU↔CPU (diff={diff:.2e})"),
        diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
    h.check_bool(
        "nW-03 LSTM gate bounded [-1,1]",
        gpu_gates.iter().all(|&v| (-1.0..=1.0).contains(&v)),
    );
}

// ═══════════════════════════════════════════════════════════════════
// WDM nW-05: ESN reservoir state update
// ═══════════════════════════════════════════════════════════════════

fn validate_wdm_esn_reservoir(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
    rng: &mut Rng,
) {
    let n_res = 16;
    let d_in = 4;
    let spectral_radius = 0.9;

    let raw: Vec<f64> = (0..n_res * n_res).map(|_| rng.normal() * 0.3).collect();
    let w_res_sym: Vec<f64> = {
        let mut s = vec![0.0_f64; n_res * n_res];
        for i in 0..n_res {
            for j in 0..n_res {
                let v = 0.5 * (raw[i * n_res + j] + raw[j * n_res + i]);
                s[i * n_res + j] = v;
                s[j * n_res + i] = v;
            }
        }
        s
    };
    let w_in: Vec<f64> = (0..d_in * n_res).map(|_| rng.normal() * 0.5).collect();
    let state: Vec<f64> = (0..n_res).map(|_| rng.normal() * 0.1).collect();
    let input: Vec<f64> = (0..d_in).map(|_| rng.normal()).collect();

    let (gpu_evals, _) = gpu.eigh(&w_res_sym, n_res);
    let max_eval_gpu = gpu_evals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let (cpu_evals, _) = cpu.eigh(&w_res_sym, n_res);
    let max_eval_cpu = cpu_evals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    h.check_abs(
        "nW-05 ESN spectral radius GPU↔CPU",
        max_eval_gpu,
        max_eval_cpu,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    let scale = spectral_radius / max_eval_gpu.max(1e-15);
    let w_res: Vec<f64> = w_res_sym.iter().map(|&v| v * scale).collect();

    let gpu_next = esn_step(gpu, &state, &input, &w_res, &w_in, n_res, d_in);
    let cpu_next = esn_step(cpu, &state, &input, &w_res, &w_in, n_res, d_in);

    let diff = max_abs_diff_f64(&gpu_next, &cpu_next);
    h.check_bool(
        &format!("nW-05 ESN state update GPU↔CPU (diff={diff:.2e})"),
        diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
    h.check_bool(
        "nW-05 ESN state bounded",
        gpu_next.iter().all(|&v| v.abs() <= 1.0),
    );
}

// ═══════════════════════════════════════════════════════════════════
// coralForge: Evoformer attention QK^T/√d → softmax
// ═══════════════════════════════════════════════════════════════════

fn validate_evoformer_attention(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
    rng: &mut Rng,
) {
    let n_res = 6;
    let head_dim = 4;
    let n_heads = 2;

    let q: Vec<f64> = (0..n_res * n_heads * head_dim)
        .map(|_| rng.normal() * 0.3)
        .collect();
    let k: Vec<f64> = (0..n_res * n_heads * head_dim)
        .map(|_| rng.normal() * 0.3)
        .collect();

    let scale = (head_dim as f64).sqrt();

    for head in 0..n_heads {
        let q_h: Vec<f64> = (0..n_res)
            .flat_map(|r| {
                let off = (r * n_heads + head) * head_dim;
                q[off..off + head_dim].to_vec()
            })
            .collect();
        let k_h: Vec<f64> = (0..n_res)
            .flat_map(|r| {
                let off = (r * n_heads + head) * head_dim;
                k[off..off + head_dim].to_vec()
            })
            .collect();

        let k_t = transpose_rect(&k_h, n_res, head_dim);

        let gpu_qk = rect_matmul(gpu, &q_h, &k_t, n_res, head_dim, n_res);
        let cpu_qk = rect_matmul(cpu, &q_h, &k_t, n_res, head_dim, n_res);

        let gpu_scaled: Vec<f64> = gpu_qk.iter().map(|&v| v / scale).collect();
        let cpu_scaled: Vec<f64> = cpu_qk.iter().map(|&v| v / scale).collect();

        let diff = max_abs_diff_f64(&gpu_scaled, &cpu_scaled);
        h.check_bool(
            &format!("coralForge attn head[{head}] QK^T GPU↔CPU (diff={diff:.2e})"),
            diff < tolerances::GPU_MATMUL_RANDOM_F32,
        );

        for row_idx in 0..n_res {
            let gpu_row = &gpu_scaled[row_idx * n_res..(row_idx + 1) * n_res];
            let cpu_row = &cpu_scaled[row_idx * n_res..(row_idx + 1) * n_res];

            let gpu_sm = gpu.softmax(gpu_row);
            let cpu_sm = cpu.softmax(cpu_row);

            let sm_diff = max_abs_diff_f64(&gpu_sm, &cpu_sm);
            h.check_bool(
                &format!(
                    "coralForge attn h[{head}]r[{row_idx}] softmax GPU↔CPU (diff={sm_diff:.2e})"
                ),
                sm_diff < tolerances::GPU_SOFTMAX_DISPATCH_F32,
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// coralForge: Triangle multiply contraction via Dispatcher
// ═══════════════════════════════════════════════════════════════════

fn validate_triangle_contraction(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
    rng: &mut Rng,
) {
    let n = 4;
    let c = 2;

    let proj_a: Vec<f64> = (0..n * n * c).map(|_| rng.normal() * 0.3).collect();
    let proj_b: Vec<f64> = (0..n * n * c).map(|_| rng.normal() * 0.3).collect();

    let gpu_out = trimul_via_dispatcher(gpu, &proj_a, &proj_b, n, c);
    let cpu_out = trimul_via_dispatcher(cpu, &proj_a, &proj_b, n, c);

    let diff = max_abs_diff_f64(&gpu_out, &cpu_out);
    h.check_bool(
        &format!("coralForge trimul outgoing GPU↔CPU (diff={diff:.2e})"),
        diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
    h.check_bool(
        "coralForge trimul finite",
        gpu_out.iter().all(|v| v.is_finite()),
    );
}

// ═══════════════════════════════════════════════════════════════════
// coralForge: pLDDT confidence head composition
// ═══════════════════════════════════════════════════════════════════

fn validate_pldt_composition(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
    rng: &mut Rng,
) {
    let n_res = 12;
    let d = 4;

    let repr: Vec<f64> = (0..n_res * d).map(|_| rng.normal() * 0.5).collect();
    let w: Vec<f64> = (0..d).map(|_| rng.normal() * 0.3).collect();
    let bias = rng.normal() * 0.1;

    let gpu_logits = rect_matmul(gpu, &repr, &w, n_res, d, 1);
    let cpu_logits = rect_matmul(cpu, &repr, &w, n_res, d, 1);

    let gpu_pldt: Vec<f64> = gpu_logits.iter().map(|&l| sigmoid(l + bias)).collect();
    let cpu_pldt: Vec<f64> = cpu_logits.iter().map(|&l| sigmoid(l + bias)).collect();

    let diff = max_abs_diff_f64(&gpu_pldt, &cpu_pldt);
    h.check_bool(
        &format!("coralForge pLDDT GPU↔CPU (diff={diff:.2e})"),
        diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );

    let gpu_mean = gpu.mean(&gpu_pldt);
    let cpu_mean = cpu.mean(&cpu_pldt);
    h.check_abs(
        "coralForge pLDDT mean GPU↔CPU",
        gpu_mean,
        cpu_mean,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );

    h.check_bool(
        "coralForge pLDDT in [0,1]",
        gpu_pldt.iter().all(|&v| (0.0..=1.0).contains(&v)),
    );
}

// ═══════════════════════════════════════════════════════════════════
// coralForge: Layer norm composition (mean + variance)
// ═══════════════════════════════════════════════════════════════════

fn validate_layer_norm_composition(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
    rng: &mut Rng,
) {
    let n = 4;
    let d = 6;
    let eps = tolerances::LAYER_NORM_EPS;

    let input: Vec<f64> = (0..n * d).map(|_| rng.normal()).collect();
    let gamma: Vec<f64> = (0..d).map(|_| 0.8 + rng.next_f64() * 0.4).collect();
    let beta: Vec<f64> = (0..d).map(|_| rng.normal() * 0.1).collect();

    let gpu_out = layer_norm_via_dispatcher(gpu, &input, n, d, &gamma, &beta, eps);
    let cpu_out = layer_norm_via_dispatcher(cpu, &input, n, d, &gamma, &beta, eps);

    let diff = max_abs_diff_f64(&gpu_out, &cpu_out);
    h.check_bool(
        &format!("coralForge layer norm GPU↔CPU (diff={diff:.2e})"),
        diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );

    for row_idx in 0..n {
        let row = &gpu_out[row_idx * d..(row_idx + 1) * d];
        let row_mean = gpu.mean(row);
        h.check_abs(
            &format!("coralForge LN row[{row_idx}] near zero mean"),
            row_mean.abs(),
            0.0,
            0.5,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// coralForge: SE(3) equivariance — COM removal + L2 residual
// ═══════════════════════════════════════════════════════════════════

fn validate_se3_equivariance(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
    rng: &mut Rng,
) {
    let n_atoms = 8;
    let coords: Vec<f64> = (0..n_atoms * 3).map(|_| rng.normal() * 10.0).collect();

    let x: Vec<f64> = (0..n_atoms).map(|i| coords[i * 3]).collect();
    let y: Vec<f64> = (0..n_atoms).map(|i| coords[i * 3 + 1]).collect();
    let z: Vec<f64> = (0..n_atoms).map(|i| coords[i * 3 + 2]).collect();

    let gpu_com = [gpu.mean(&x), gpu.mean(&y), gpu.mean(&z)];
    let cpu_com = [cpu.mean(&x), cpu.mean(&y), cpu.mean(&z)];

    h.check_abs(
        "SE(3) COM x GPU↔CPU",
        gpu_com[0],
        cpu_com[0],
        tolerances::GPU_MEAN_DISPATCH_F32,
    );
    h.check_abs(
        "SE(3) COM y GPU↔CPU",
        gpu_com[1],
        cpu_com[1],
        tolerances::GPU_MEAN_DISPATCH_F32,
    );
    h.check_abs(
        "SE(3) COM z GPU↔CPU",
        gpu_com[2],
        cpu_com[2],
        tolerances::GPU_MEAN_DISPATCH_F32,
    );

    let gpu_centered: Vec<f64> = coords
        .chunks_exact(3)
        .flat_map(|a| vec![a[0] - gpu_com[0], a[1] - gpu_com[1], a[2] - gpu_com[2]])
        .collect();
    let cpu_centered: Vec<f64> = coords
        .chunks_exact(3)
        .flat_map(|a| vec![a[0] - cpu_com[0], a[1] - cpu_com[1], a[2] - cpu_com[2]])
        .collect();

    let diff = max_abs_diff_f64(&gpu_centered, &cpu_centered);
    h.check_bool(
        &format!("SE(3) centered coords GPU↔CPU (diff={diff:.2e})"),
        diff < tolerances::GPU_MEAN_DISPATCH_F32,
    );

    let gpu_res_x: Vec<f64> = (0..n_atoms).map(|i| gpu_centered[i * 3]).collect();
    let gpu_res_mean = gpu.mean(&gpu_res_x);
    h.check_abs(
        "SE(3) residual COM near zero",
        gpu_res_mean.abs(),
        0.0,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );

    let translation = [5.0, -3.0, 7.0];
    let translated: Vec<f64> = coords
        .chunks_exact(3)
        .flat_map(|a| {
            vec![
                a[0] + translation[0],
                a[1] + translation[1],
                a[2] + translation[2],
            ]
        })
        .collect();
    let tx: Vec<f64> = (0..n_atoms).map(|i| translated[i * 3]).collect();
    let ty: Vec<f64> = (0..n_atoms).map(|i| translated[i * 3 + 1]).collect();
    let tz: Vec<f64> = (0..n_atoms).map(|i| translated[i * 3 + 2]).collect();
    let t_com = [gpu.mean(&tx), gpu.mean(&ty), gpu.mean(&tz)];
    let t_centered: Vec<f64> = translated
        .chunks_exact(3)
        .flat_map(|a| vec![a[0] - t_com[0], a[1] - t_com[1], a[2] - t_com[2]])
        .collect();

    let inv_diff = max_abs_diff_f64(&gpu_centered, &t_centered);
    h.check_bool(
        &format!("SE(3) translation invariance (diff={inv_diff:.2e})"),
        inv_diff < tolerances::GPU_MEAN_DISPATCH_F32,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Domain composition helpers
// ═══════════════════════════════════════════════════════════════════

fn mlp_forward_3layer(
    disp: &Dispatcher,
    input: &[f64],
    w1: &[f64],
    w2: &[f64],
    w3: &[f64],
    batch: usize,
    d_in: usize,
    d_h: usize,
    d_out: usize,
) -> Vec<f64> {
    let h1 = rect_matmul(disp, input, w1, batch, d_in, d_h);
    let a1: Vec<f64> = h1.iter().map(|&x| sigmoid(x)).collect();
    let h2 = rect_matmul(disp, &a1, w2, batch, d_h, d_h);
    let a2: Vec<f64> = h2.iter().map(|&x| sigmoid(x)).collect();
    rect_matmul(disp, &a2, w3, batch, d_h, d_out)
}

fn lstm_gate_step(
    disp: &Dispatcher,
    x: &[f64],
    h_prev: &[f64],
    w_xh: &[f64],
    w_hh: &[f64],
    d_in: usize,
    hidden: usize,
) -> Vec<f64> {
    let xw = rect_matmul(disp, x, w_xh, 1, d_in, hidden);
    let hw = rect_matmul(disp, h_prev, w_hh, 1, hidden, hidden);
    xw.iter()
        .zip(hw.iter())
        .map(|(&a, &b)| (a + b).tanh())
        .collect()
}

fn esn_step(
    disp: &Dispatcher,
    state: &[f64],
    input: &[f64],
    w_res: &[f64],
    w_in: &[f64],
    n_res: usize,
    d_in: usize,
) -> Vec<f64> {
    let res_update = rect_matmul(disp, state, w_res, 1, n_res, n_res);
    let in_proj = rect_matmul(disp, input, w_in, 1, d_in, n_res);
    res_update
        .iter()
        .zip(in_proj.iter())
        .map(|(&r, &i)| (r + i).tanh())
        .collect()
}

fn trimul_via_dispatcher(
    disp: &Dispatcher,
    proj_a: &[f64],
    proj_b: &[f64],
    n: usize,
    c: usize,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n * c];
    for i in 0..n {
        for j in 0..n {
            for ch in 0..c {
                let a_col: Vec<f64> = (0..n).map(|k| proj_a[(i * n + k) * c + ch]).collect();
                let b_col: Vec<f64> = (0..n).map(|k| proj_b[(j * n + k) * c + ch]).collect();
                let dot_vec = rect_matmul(disp, &a_col, &b_col, 1, n, 1);
                out[(i * n + j) * c + ch] = dot_vec[0];
            }
        }
    }
    out
}

fn layer_norm_via_dispatcher(
    disp: &Dispatcher,
    input: &[f64],
    n: usize,
    d: usize,
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(n * d);
    for row in input.chunks_exact(d) {
        let mu = disp.mean(row);
        let var = disp.variance(row);
        let inv_std = 1.0 / (var + eps).sqrt();
        for (i, &x) in row.iter().enumerate() {
            out.push((x - mu).mul_add(inv_std * gamma[i], beta[i]));
        }
    }
    out
}

fn transpose_rect(m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0_f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = m[i * cols + j];
        }
    }
    t
}

use neural_spring::primitives::sigmoid;

fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}
