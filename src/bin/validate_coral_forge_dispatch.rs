// SPDX-License-Identifier: AGPL-3.0-or-later

//! coralForge Evoformer dispatch parity — proves `AlphaFold2`/OpenFold primitives
//! route correctly through `BarraCUDA` CPU↔GPU dispatch and `metalForge`
//! mixed-hardware substrate routing.
//!
//! Closes the dispatch + metalForge tiers for nF-01 (OpenFold) and nF-02
//! (`AlphaFold2`), complementing `validate_wdm_alphafold_dispatch` which
//! covers nF-03 (`AlphaFold3` confidence heads).
//!
//! ## Coverage
//!
//! - **Triangle multiply outgoing**: A/B projection → matmul (Algorithm 11)
//! - **Triangle multiply incoming**: same, transposed accumulation (Algorithm 12)
//! - **Attention scores**: QKᵀ/√d (via Dispatcher matmul + softmax)
//! - **Outer product mean**: MSA accumulation (via Dispatcher matmul)
//! - **IPA distance scores**: multi-term attention (SE(3)-equivariant) (nF-02)
//! - **Mixed-hardware routing**: metalForge substrate selection for folding
//! - **NUCLEUS coordination**: tower (eigensolve) + node (folding state)

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{exit_no_gpu, ValidationHarness};
use neural_spring_forge::mixed::MixedSubstrate;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("coral_forge_dispatch");

    let Ok(gpu) = Gpu::new().await else {
        exit_no_gpu();
    };
    let gpu_disp = Dispatcher::from_gpu(gpu);
    let cpu_disp = Dispatcher::cpu_only();

    validate_triangle_mul_outgoing(&mut h, &gpu_disp, &cpu_disp);
    validate_triangle_mul_incoming(&mut h, &gpu_disp, &cpu_disp);
    validate_attention_scores(&mut h, &gpu_disp, &cpu_disp);
    validate_outer_product_mean(&mut h, &gpu_disp, &cpu_disp);
    validate_ipa_distance(&mut h, &gpu_disp, &cpu_disp);
    validate_mixed_routing_folding(&mut h, &gpu_disp);
    validate_nucleus_folding(&mut h, &gpu_disp);

    h.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Triangle multiply outgoing (Algorithm 11): proj_a ⊗ proj_b → sum_k
// ═══════════════════════════════════════════════════════════════════

fn validate_triangle_mul_outgoing(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(1101);
    let n = 4;
    let c = 3;

    let proj_a: Vec<f64> = (0..n * n * c).map(|_| rng.normal() * 0.3).collect();
    let proj_b: Vec<f64> = (0..n * n * c).map(|_| rng.normal() * 0.3).collect();

    let g_out = triangle_mul_outgoing_dispatch(gpu, &proj_a, &proj_b, n, c);
    let c_out = triangle_mul_outgoing_dispatch(cpu, &proj_a, &proj_b, n, c);

    let max_diff = g_out
        .iter()
        .zip(c_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("trimul outgoing dispatch parity (max_diff={max_diff:.2e})"),
        max_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
    h.check_bool(
        "trimul outgoing finite",
        g_out.iter().all(|v| v.is_finite()),
    );
    h.check_bool("trimul outgoing correct shape", g_out.len() == n * n * c);
}

// ═══════════════════════════════════════════════════════════════════
// Triangle multiply incoming (Algorithm 12): same, transposed
// ═══════════════════════════════════════════════════════════════════

fn validate_triangle_mul_incoming(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(1102);
    let n = 4;
    let c = 3;

    let proj_a: Vec<f64> = (0..n * n * c).map(|_| rng.normal() * 0.3).collect();
    let proj_b: Vec<f64> = (0..n * n * c).map(|_| rng.normal() * 0.3).collect();

    let g_out = triangle_mul_incoming_dispatch(gpu, &proj_a, &proj_b, n, c);
    let c_out = triangle_mul_incoming_dispatch(cpu, &proj_a, &proj_b, n, c);

    let max_diff = g_out
        .iter()
        .zip(c_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("trimul incoming dispatch parity (max_diff={max_diff:.2e})"),
        max_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
    h.check_bool(
        "trimul incoming finite",
        g_out.iter().all(|v| v.is_finite()),
    );
}

// ═══════════════════════════════════════════════════════════════════
// Attention scores: QKᵀ/√d (Evoformer self-attention)
// ═══════════════════════════════════════════════════════════════════

fn validate_attention_scores(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(1103);
    let n_res = 6;
    let n_heads = 2;
    let head_dim = 4;
    let total = n_res * n_heads * head_dim;

    let q: Vec<f64> = (0..total).map(|_| rng.normal() * 0.3).collect();
    let k: Vec<f64> = (0..total).map(|_| rng.normal() * 0.3).collect();

    let g_scores = attention_scores_dispatch(gpu, &q, &k, n_res, n_heads, head_dim);
    let c_scores = attention_scores_dispatch(cpu, &q, &k, n_res, n_heads, head_dim);

    let max_diff = g_scores
        .iter()
        .zip(c_scores.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("attention scores dispatch parity (max_diff={max_diff:.2e})"),
        max_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );

    let g_weights = apply_softmax_rows(gpu, &g_scores, n_heads * n_res, n_res);
    let c_weights = apply_softmax_rows(cpu, &c_scores, n_heads * n_res, n_res);

    for (i, (gw, cw)) in g_weights
        .chunks(n_res)
        .zip(c_weights.chunks(n_res))
        .enumerate()
    {
        let sum_g: f64 = gw.iter().sum();
        h.check_abs(
            &format!("attention softmax row[{i}] sums to 1"),
            sum_g,
            1.0,
            tolerances::TENSOR_EXACT_F32,
        );
        let row_diff: f64 = gw
            .iter()
            .zip(cw.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        h.check_bool(
            &format!("attention softmax row[{i}] GPU↔CPU (diff={row_diff:.2e})"),
            row_diff < tolerances::GPU_MATMUL_RANDOM_F32,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// Outer product mean: MSA → pair representation (Algorithm 10)
// ═══════════════════════════════════════════════════════════════════

fn validate_outer_product_mean(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(1104);
    let n_seq = 3;
    let n_res = 4;
    let c_a = 2;
    let c_b = 2;

    let a: Vec<f64> = (0..n_seq * n_res * c_a)
        .map(|_| rng.normal() * 0.3)
        .collect();
    let b: Vec<f64> = (0..n_seq * n_res * c_b)
        .map(|_| rng.normal() * 0.3)
        .collect();

    let g_opm = opm_dispatch(gpu, &a, &b, n_seq, n_res, c_a, c_b);
    let c_opm = opm_dispatch(cpu, &a, &b, n_seq, n_res, c_a, c_b);

    let expected_len = n_res * n_res * c_a * c_b;
    h.check_bool("OPM correct shape", g_opm.len() == expected_len);

    let max_diff = g_opm
        .iter()
        .zip(c_opm.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("OPM dispatch parity (max_diff={max_diff:.2e})"),
        max_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
    h.check_bool("OPM all finite", g_opm.iter().all(|v| v.is_finite()));
}

// ═══════════════════════════════════════════════════════════════════
// IPA distance scores: SE(3)-equivariant multi-term attention (nF-02)
// ═══════════════════════════════════════════════════════════════════

fn validate_ipa_distance(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(1105);
    let n_res = 4;
    let n_qp = 2;

    let points_q: Vec<f64> = (0..n_res * n_qp * 3).map(|_| rng.normal() * 2.0).collect();
    let points_k: Vec<f64> = (0..n_res * n_qp * 3).map(|_| rng.normal() * 2.0).collect();

    let g_dist = ipa_distance_dispatch(gpu, &points_q, &points_k, n_res, n_qp);
    let c_dist = ipa_distance_dispatch(cpu, &points_q, &points_k, n_res, n_qp);

    let max_diff = g_dist
        .iter()
        .zip(c_dist.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("IPA distance dispatch parity (max_diff={max_diff:.2e})"),
        max_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
    h.check_bool(
        "IPA distances non-negative",
        g_dist.iter().all(|&v| v >= 0.0),
    );

    let self_dist = ipa_distance_dispatch(gpu, &points_q, &points_q, n_res, n_qp);
    h.check_bool("IPA self-distance diagonal zero", {
        (0..n_res).all(|i| self_dist[i * n_res + i] < 1e-10)
    });
}

// ═══════════════════════════════════════════════════════════════════
// Mixed-hardware routing: metalForge substrate for folding workloads
// ═══════════════════════════════════════════════════════════════════

fn validate_mixed_routing_folding(h: &mut ValidationHarness, disp: &Dispatcher) {
    let small_attn = MixedWorkload {
        op: "evoformer_attention_small",
        compute_us: 80.0,
        data_bytes: 2048,
        npu_available: false,
        needs_realtime: false,
    };

    let data = [0.5, 1.0, 1.5, 2.0];
    let (result, substrate) = disp.mixed_dispatch(
        &small_attn,
        |dev| {
            let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let t = barracuda::tensor::Tensor::from_data(&f32_data, vec![4], dev.clone())
                .map_err(|e| format!("{e}"))?;
            let mean = t.mean().map_err(|e| format!("{e}"))?;
            let v = mean.to_vec().map_err(|e| format!("{e}"))?;
            Ok(f64::from(v[0]))
        },
        || data.iter().sum::<f64>() / data.len() as f64,
    );
    h.check_abs("mixed small evoformer → correct result", result, 1.25, 0.01);
    h.check_bool(
        "mixed small evoformer → CPU (dispatch overhead dominates)",
        substrate == MixedSubstrate::CpuOnly,
    );

    let large_trimul = MixedWorkload {
        op: "evoformer_triangle_multiply_large",
        compute_us: 500_000.0,
        data_bytes: 8_388_608,
        npu_available: false,
        needs_realtime: false,
    };
    let (result_lg, substrate_lg) = disp.mixed_dispatch(
        &large_trimul,
        |dev| {
            let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let t = barracuda::tensor::Tensor::from_data(&f32_data, vec![4], dev.clone())
                .map_err(|e| format!("{e}"))?;
            let mean = t.mean().map_err(|e| format!("{e}"))?;
            let v = mean.to_vec().map_err(|e| format!("{e}"))?;
            Ok(f64::from(v[0]))
        },
        || data.iter().sum::<f64>() / data.len() as f64,
    );
    h.check_abs("mixed large trimul → correct result", result_lg, 1.25, 0.01);
    h.check_bool(
        "mixed large trimul → GPU (compute dominates)",
        substrate_lg == MixedSubstrate::GpuOnly,
    );

    let npu_folding = MixedWorkload {
        op: "evoformer_realtime_folding",
        compute_us: 300_000.0,
        data_bytes: 4_194_304,
        npu_available: true,
        needs_realtime: true,
    };
    let (_, substrate_npu) = disp.mixed_dispatch(
        &npu_folding,
        |dev| {
            let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let t = barracuda::tensor::Tensor::from_data(&f32_data, vec![4], dev.clone())
                .map_err(|e| format!("{e}"))?;
            let v = t
                .mean()
                .map_err(|e| format!("{e}"))?
                .to_vec()
                .map_err(|e| format!("{e}"))?;
            Ok(f64::from(v[0]))
        },
        || data.iter().sum::<f64>() / data.len() as f64,
    );
    h.check_bool(
        "mixed NPU realtime folding → GpuToNpu routing",
        substrate_npu == MixedSubstrate::GpuToNpu,
    );
}

// ═══════════════════════════════════════════════════════════════════
// NUCLEUS coordination: tower (eigensolve) + node (folding state)
// ═══════════════════════════════════════════════════════════════════

fn validate_nucleus_folding(h: &mut ValidationHarness, disp: &Dispatcher) {
    let contact_map = vec![1.0, 0.8, 0.2, 0.8, 1.0, 0.5, 0.2, 0.5, 1.0];
    let (eigenvalues, _) = disp.eigh(&contact_map, 3);
    let mut sorted = eigenvalues;
    sorted.sort_by(f64::total_cmp);
    h.check_bool("nucleus tower: contact map eigensolve", sorted.len() == 3);
    h.check_bool(
        "nucleus tower: positive semi-definite (λ_min ≥ -ε)",
        sorted[0] > -0.1,
    );

    let residue_confidences = vec![0.85, 0.92, 0.78, 0.95];
    let probs = disp.softmax(&residue_confidences);
    let sum: f64 = probs.iter().sum();
    h.check_abs(
        "nucleus node: folding confidence softmax sums to 1",
        sum,
        1.0,
        1e-10,
    );
    h.check_bool(
        "nucleus node: all confidences positive",
        probs.iter().all(|&p| p > 0.0),
    );

    let entropy = disp.shannon_entropy(&probs);
    h.check_bool("nucleus nest: folding entropy finite", entropy.is_finite());
    h.check_bool("nucleus nest: entropy > 0", entropy > 0.0);
}

// ═══════════════════════════════════════════════════════════════════
// Domain-specific dispatch compositions
// ═══════════════════════════════════════════════════════════════════

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

fn triangle_mul_outgoing_dispatch(
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
                let b_col: Vec<f64> = (0..n).map(|k| proj_b[(k * n + j) * c + ch]).collect();
                let dot = rect_matmul(disp, &a_col, &b_col, 1, n, 1);
                out[(i * n + j) * c + ch] = dot[0];
            }
        }
    }
    out
}

fn triangle_mul_incoming_dispatch(
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
                let a_col: Vec<f64> = (0..n).map(|k| proj_a[(k * n + i) * c + ch]).collect();
                let b_col: Vec<f64> = (0..n).map(|k| proj_b[(j * n + k) * c + ch]).collect();
                let dot = rect_matmul(disp, &a_col, &b_col, 1, n, 1);
                out[(i * n + j) * c + ch] = dot[0];
            }
        }
    }
    out
}

fn attention_scores_dispatch(
    disp: &Dispatcher,
    q: &[f64],
    k: &[f64],
    n_res: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f64> {
    let scale = (head_dim as f64).sqrt();
    let mut scores = vec![0.0_f64; n_heads * n_res * n_res];

    for h_idx in 0..n_heads {
        let q_head: Vec<f64> = (0..n_res)
            .flat_map(|r| (0..head_dim).map(move |d| q[(r * n_heads + h_idx) * head_dim + d]))
            .collect();
        let k_head: Vec<f64> = (0..n_res)
            .flat_map(|r| (0..head_dim).map(move |d| k[(r * n_heads + h_idx) * head_dim + d]))
            .collect();

        let k_t: Vec<f64> = (0..head_dim)
            .flat_map(|d| {
                let kh = &k_head;
                (0..n_res).map(move |r| kh[r * head_dim + d])
            })
            .collect();

        let qk = rect_matmul(disp, &q_head, &k_t, n_res, head_dim, n_res);
        for idx in 0..n_res * n_res {
            scores[h_idx * n_res * n_res + idx] = qk[idx] / scale;
        }
    }
    scores
}

fn apply_softmax_rows(disp: &Dispatcher, scores: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(n_rows * n_cols);
    for row in scores.chunks(n_cols) {
        let probs = disp.softmax(row);
        result.extend_from_slice(&probs);
    }
    result
}

fn opm_dispatch(
    disp: &Dispatcher,
    a: &[f64],
    b: &[f64],
    n_seq: usize,
    n_res: usize,
    c_a: usize,
    c_b: usize,
) -> Vec<f64> {
    let c_out = c_a * c_b;
    let mut out = vec![0.0_f64; n_res * n_res * c_out];

    for s in 0..n_seq {
        let a_s: Vec<f64> = (0..n_res * c_a)
            .map(|idx| a[s * n_res * c_a + idx])
            .collect();
        let b_s: Vec<f64> = (0..n_res * c_b)
            .map(|idx| b[s * n_res * c_b + idx])
            .collect();

        let b_t: Vec<f64> = (0..c_b)
            .flat_map(|cb| {
                let bs = &b_s;
                (0..n_res).map(move |r| bs[r * c_b + cb])
            })
            .collect();

        let outer = rect_matmul(disp, &a_s, &b_t, n_res * c_a, 1, n_res * c_b);

        for i in 0..n_res {
            for j in 0..n_res {
                for ca in 0..c_a {
                    for cb in 0..c_b {
                        out[(i * n_res + j) * c_out + ca * c_b + cb] +=
                            outer[(i * c_a + ca) * (n_res * c_b) + j * c_b + cb];
                    }
                }
            }
        }
    }

    let inv_n = 1.0 / n_seq as f64;
    for v in &mut out {
        *v *= inv_n;
    }
    out
}

fn ipa_distance_dispatch(
    disp: &Dispatcher,
    points_q: &[f64],
    points_k: &[f64],
    n_res: usize,
    n_qp: usize,
) -> Vec<f64> {
    let mut dist = vec![0.0_f64; n_res * n_res];

    for i in 0..n_res {
        for j in 0..n_res {
            let mut total_dist = 0.0_f64;
            for qp in 0..n_qp {
                let qi = &points_q[(i * n_qp + qp) * 3..(i * n_qp + qp) * 3 + 3];
                let kj = &points_k[(j * n_qp + qp) * 3..(j * n_qp + qp) * 3 + 3];
                let diff: Vec<f64> = qi.iter().zip(kj.iter()).map(|(a, b)| a - b).collect();
                let sq_dist = rect_matmul(disp, &diff, &diff, 1, 3, 1);
                total_dist += sq_dist[0];
            }
            dist[i * n_res + j] = total_dist / n_qp as f64;
        }
    }
    dist
}
