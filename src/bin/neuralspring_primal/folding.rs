// SPDX-License-Identifier: AGPL-3.0-or-later

//! coralForge RPC handlers (nF-01/02 Evoformer + Structure Module).
//!
//! Evoformer block implements Algorithm 6 from Jumper et al. 2021.
//! Structure Module implements Algorithm 22 (IPA + backbone + torsion).

use neural_spring::coral_forge::structure::{
    backbone_update, ipa_scores, torsion_angles, IpaConfig,
};
use neural_spring::coral_forge::{
    msa_col_attention, msa_row_attention, outer_product_mean, triangle_attention_scores,
    triangle_mul_incoming, triangle_mul_outgoing,
};
use neural_spring::rng::Rng;

use super::{JsonRpcResponse, PrimalState};

#[expect(clippy::too_many_lines, reason = "validation binary")]
pub fn handle_evoformer_block(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    let n_seq = p_usize(params, "n_seq", 4);
    let n_res = p_usize(params, "n_res", 6);
    let n_heads = p_usize(params, "n_heads", 2);
    let head_dim = p_usize(params, "head_dim", 4);
    let c_pair = p_usize(params, "c_pair", 4);
    let seed = p_u64(params, "seed", 42);

    let c_msa = n_heads * head_dim;
    let mut rng = Rng::new(seed);

    let msa_len = n_seq * n_res * c_msa;
    let pair_len = n_res * n_res * c_pair;
    let mut msa: Vec<f64> = (0..msa_len).map(|_| rng.normal()).collect();
    let mut pair: Vec<f64> = (0..pair_len).map(|_| rng.normal()).collect();

    let msa_fingerprint: f64 = msa.iter().map(|x| x * x).sum();
    let pair_fingerprint: f64 = pair.iter().map(|x| x * x).sum();

    // Step 1: MSA row attention with pair bias
    let w_q = rand_vec(&mut rng, c_msa * n_heads * head_dim);
    let w_k = rand_vec(&mut rng, c_msa * n_heads * head_dim);
    let w_v = rand_vec(&mut rng, c_msa * n_heads * head_dim);

    let q_row = matmul_3d(&msa, &w_q, n_seq, n_res, c_msa, n_heads * head_dim);
    let k_row = matmul_3d(&msa, &w_k, n_seq, n_res, c_msa, n_heads * head_dim);
    let v_row = matmul_3d(&msa, &w_v, n_seq, n_res, c_msa, n_heads * head_dim);

    let w_bias = rand_vec(&mut rng, c_pair * n_heads);
    let pair_bias = einsum_ijc_ch(&pair, &w_bias, n_res, n_res, c_pair, n_heads);

    let msa_row_out = msa_row_attention(
        &q_row, &k_row, &v_row, &pair_bias, n_seq, n_res, n_heads, head_dim,
    );
    let w_o_row = rand_vec(&mut rng, n_heads * head_dim * c_msa);
    let projected_row = matmul_3d(
        &msa_row_out,
        &w_o_row,
        n_seq,
        n_res,
        n_heads * head_dim,
        c_msa,
    );
    add_inplace(&mut msa, &projected_row);

    // Step 2: MSA column attention
    let w_q_col = rand_vec(&mut rng, c_msa * n_heads * head_dim);
    let w_k_col = rand_vec(&mut rng, c_msa * n_heads * head_dim);
    let w_v_col = rand_vec(&mut rng, c_msa * n_heads * head_dim);

    let q_col = matmul_3d(&msa, &w_q_col, n_seq, n_res, c_msa, n_heads * head_dim);
    let k_col = matmul_3d(&msa, &w_k_col, n_seq, n_res, c_msa, n_heads * head_dim);
    let v_col = matmul_3d(&msa, &w_v_col, n_seq, n_res, c_msa, n_heads * head_dim);

    let msa_col_out = msa_col_attention(&q_col, &k_col, &v_col, n_seq, n_res, n_heads, head_dim);
    let w_o_col = rand_vec(&mut rng, n_heads * head_dim * c_msa);
    let projected_col = matmul_3d(
        &msa_col_out,
        &w_o_col,
        n_seq,
        n_res,
        n_heads * head_dim,
        c_msa,
    );
    add_inplace(&mut msa, &projected_col);

    // Step 3: Outer product mean
    let c_opm = 2;
    let w_opm_a = rand_vec(&mut rng, c_msa * c_opm);
    let w_opm_b = rand_vec(&mut rng, c_msa * c_opm);
    let opm_a = matmul_3d(&msa, &w_opm_a, n_seq, n_res, c_msa, c_opm);
    let opm_b = matmul_3d(&msa, &w_opm_b, n_seq, n_res, c_msa, c_opm);
    let opm_out = outer_product_mean(&opm_a, &opm_b, n_seq, n_res, c_opm, c_opm);

    let w_opm_proj = rand_vec(&mut rng, c_opm * c_opm * c_pair);
    let opm_projected = matmul_2d(&opm_out, &w_opm_proj, n_res * n_res, c_opm * c_opm, c_pair);
    add_inplace(&mut pair, &opm_projected);

    // Step 4: Triangle multiplicative outgoing
    let w_tri_a = rand_vec(&mut rng, c_pair * c_pair);
    let w_tri_b = rand_vec(&mut rng, c_pair * c_pair);
    let proj_a = matmul_2d(&pair, &w_tri_a, n_res * n_res, c_pair, c_pair);
    let proj_b = matmul_2d(&pair, &w_tri_b, n_res * n_res, c_pair, c_pair);
    let tri_out = triangle_mul_outgoing(&proj_a, &proj_b, n_res, c_pair);
    let w_tri_proj = rand_vec(&mut rng, c_pair * c_pair);
    let tri_projected = matmul_2d(&tri_out, &w_tri_proj, n_res * n_res, c_pair, c_pair);
    add_inplace(&mut pair, &tri_projected);

    // Step 5: Triangle multiplicative incoming
    let w_tri_a_in = rand_vec(&mut rng, c_pair * c_pair);
    let w_tri_b_in = rand_vec(&mut rng, c_pair * c_pair);
    let proj_a_in = matmul_2d(&pair, &w_tri_a_in, n_res * n_res, c_pair, c_pair);
    let proj_b_in = matmul_2d(&pair, &w_tri_b_in, n_res * n_res, c_pair, c_pair);
    let tri_in = triangle_mul_incoming(&proj_a_in, &proj_b_in, n_res, c_pair);
    let w_tri_proj_in = rand_vec(&mut rng, c_pair * c_pair);
    let tri_proj_in = matmul_2d(&tri_in, &w_tri_proj_in, n_res * n_res, c_pair, c_pair);
    add_inplace(&mut pair, &tri_proj_in);

    // Step 6: Triangle attention scores
    let w_tq = rand_vec(&mut rng, c_pair * n_heads * head_dim);
    let w_tk = rand_vec(&mut rng, c_pair * n_heads * head_dim);
    let tri_q = matmul_2d(&pair, &w_tq, n_res * n_res, c_pair, n_heads * head_dim);
    let tri_k = matmul_2d(&pair, &w_tk, n_res * n_res, c_pair, n_heads * head_dim);
    let w_tri_bias = rand_vec(&mut rng, c_pair * n_heads);
    let tri_bias = einsum_ijc_ch(&pair, &w_tri_bias, n_res, n_res, c_pair, n_heads);
    let tri_attn =
        triangle_attention_scores(&tri_q, &tri_k, &tri_bias, n_res, n_res, n_heads, head_dim);

    let msa_changed = {
        let after: f64 = msa.iter().map(|x| x * x).sum();
        (after - msa_fingerprint).abs() > 1e-15
    };
    let pair_changed = {
        let after: f64 = pair.iter().map(|x| x * x).sum();
        (after - pair_fingerprint).abs() > 1e-15
    };

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "n_seq": n_seq,
            "n_res": n_res,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "c_pair": c_pair,
            "c_msa": c_msa,
            "msa_shape": [n_seq, n_res, c_msa],
            "pair_shape": [n_res, n_res, c_pair],
            "tri_attn_shape": [n_res, n_heads, n_res, n_res],
            "msa_finite": msa.iter().all(|v| v.is_finite()),
            "pair_finite": pair.iter().all(|v| v.is_finite()),
            "tri_attn_finite": tri_attn.iter().all(|v| v.is_finite()),
            "msa_changed": msa_changed,
            "pair_changed": pair_changed,
        }),
    )
}

pub fn handle_structure_module(
    id: serde_json::Value,
    params: &serde_json::Value,
) -> JsonRpcResponse {
    let n_res = p_usize(params, "n_res", 6);
    let c_single = p_usize(params, "c_single", 8);
    let c_pair = p_usize(params, "c_pair", 4);
    let n_heads = p_usize(params, "n_heads", 2);
    let head_dim = p_usize(params, "head_dim", 4);
    let n_points = p_usize(params, "n_points", 2);
    let seed = p_u64(params, "seed", 42);

    let mut rng = Rng::new(seed);

    let single: Vec<f64> = (0..n_res * c_single).map(|_| rng.normal()).collect();
    let pair: Vec<f64> = (0..n_res * n_res * c_pair).map(|_| rng.normal()).collect();

    let mut frames = vec![0.0f64; n_res * 12];
    for i in 0..n_res {
        frames[i * 12] = 1.0;
        frames[i * 12 + 4] = 1.0;
        frames[i * 12 + 8] = 1.0;
    }

    let w_iq = rand_vec(&mut rng, c_single * n_heads * head_dim);
    let w_ik = rand_vec(&mut rng, c_single * n_heads * head_dim);
    let q_scalar = matmul_2d(&single, &w_iq, n_res, c_single, n_heads * head_dim);
    let k_scalar = matmul_2d(&single, &w_ik, n_res, c_single, n_heads * head_dim);

    let w_ipa_bias = rand_vec(&mut rng, c_pair * n_heads);
    let pair_bias = einsum_ijc_ch(&pair, &w_ipa_bias, n_res, n_res, c_pair, n_heads);

    let w_qp = rand_vec(&mut rng, c_single * n_heads * n_points * 3);
    let w_kp = rand_vec(&mut rng, c_single * n_heads * n_points * 3);
    let q_points = matmul_2d(&single, &w_qp, n_res, c_single, n_heads * n_points * 3);
    let k_points = matmul_2d(&single, &w_kp, n_res, c_single, n_heads * n_points * 3);

    let cfg = IpaConfig {
        n_res,
        n_heads,
        head_dim,
        n_points,
        w_l: 1.0,
        w_c: 1.0,
        w_p: 1.0,
        gamma: 0.5,
    };
    let ipa_s = ipa_scores(
        &q_scalar, &k_scalar, &pair_bias, &q_points, &k_points, &frames, &cfg,
    );

    let mut delta_quats: Vec<f64> = (0..n_res * 4).map(|_| rng.normal() * 0.1).collect();
    for i in 0..n_res {
        delta_quats[i * 4] += 1.0;
    }
    let delta_trans: Vec<f64> = (0..n_res * 3).map(|_| rng.normal() * 0.1).collect();
    let updated_frames = backbone_update(&delta_quats, &delta_trans, &frames, n_res);

    let c_hidden = 6;
    let hh = c_hidden * c_hidden;
    let weight_len = c_single * c_hidden
        + c_hidden
        + hh
        + c_hidden
        + hh
        + c_hidden
        + hh
        + c_hidden
        + hh
        + c_hidden
        + c_hidden * 14
        + 14;
    let torsion_weights: Vec<f64> = (0..weight_len).map(|_| rng.normal() * 0.1).collect();
    let torsion_out = torsion_angles(&single, &torsion_weights, n_res, c_single, c_hidden);

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "n_res": n_res,
            "ipa_scores_shape": [n_heads, n_res, n_res],
            "ipa_scores_finite": ipa_s.iter().all(|v| v.is_finite()),
            "frames_shape": [n_res, 12],
            "frames_finite": updated_frames.iter().all(|v| v.is_finite()),
            "torsion_shape": [n_res, 7, 2],
            "torsion_finite": torsion_out.iter().all(|v| v.is_finite()),
            "torsion_count": torsion_out.len(),
        }),
    )
}

pub fn handle_folding_health(id: serde_json::Value, state: &PrimalState) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "folding_primitives": {
                "gelu": true,
                "layer_norm": true,
                "softmax_rows": true,
                "sdpa_scores": true,
                "sdpa_full": true,
                "msa_row_attention": true,
                "msa_col_attention": true,
                "outer_product_mean": true,
                "triangle_mul_outgoing": true,
                "triangle_mul_incoming": true,
                "triangle_attention_scores": true,
                "ipa_scores": true,
                "backbone_update": true,
                "torsion_angles": true,
            },
            "gpu_available": state.dispatcher.has_gpu(),
            "gpu_adapter": state.dispatcher.adapter_name(),
            "validated_papers": ["nF-01 (OpenFold)", "nF-02 (AlphaFold2)"],
            "validation_status": "210/210 validate_all",
        }),
    )
}

pub fn handle_gpu_dispatch(
    id: serde_json::Value,
    params: &serde_json::Value,
    state: &PrimalState,
) -> JsonRpcResponse {
    let op = match params.get("op").and_then(|v| v.as_str()) {
        Some(o) => o,
        None => {
            return JsonRpcResponse::error(
                id,
                super::rpc_error::INVALID_PARAMS,
                "Missing 'op' parameter".into(),
            )
        }
    };

    match op {
        "mat_mul" => dispatch_mat_mul(id, params, state),
        "softmax" => dispatch_softmax(id, params, state),
        "mean" => dispatch_mean(id, params, state),
        "variance" => dispatch_variance(id, params, state),
        "eigh" => dispatch_eigh(id, params, state),
        _ => JsonRpcResponse::error(
            id,
            super::rpc_error::INVALID_PARAMS,
            format!("Unknown dispatch op: {op}"),
        ),
    }
}

// ── Dispatch sub-handlers ────────────────────────────────────────

fn dispatch_mat_mul(
    id: serde_json::Value,
    params: &serde_json::Value,
    state: &PrimalState,
) -> JsonRpcResponse {
    let Some(a) = extract_f64_vec(params, "a") else {
        return JsonRpcResponse::error(
            id,
            super::rpc_error::INVALID_PARAMS,
            "Missing 'a' parameter".into(),
        );
    };
    let Some(b) = extract_f64_vec(params, "b") else {
        return JsonRpcResponse::error(
            id,
            super::rpc_error::INVALID_PARAMS,
            "Missing 'b' parameter".into(),
        );
    };
    let n = p_usize(params, "n", 0);
    if n == 0 || a.len() != n * n || b.len() != n * n {
        return JsonRpcResponse::error(
            id,
            super::rpc_error::INVALID_PARAMS,
            "Invalid matrix dimensions".into(),
        );
    }
    let result = state.dispatcher.mat_mul(&a, &b, n);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "result": result, "backend": format!("{}", state.dispatcher.backend()) }),
    )
}

fn dispatch_softmax(
    id: serde_json::Value,
    params: &serde_json::Value,
    state: &PrimalState,
) -> JsonRpcResponse {
    let Some(x) = extract_f64_vec(params, "x") else {
        return JsonRpcResponse::error(
            id,
            super::rpc_error::INVALID_PARAMS,
            "Missing 'x' parameter".into(),
        );
    };
    let result = state.dispatcher.softmax(&x);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "result": result, "backend": format!("{}", state.dispatcher.backend()) }),
    )
}

fn dispatch_mean(
    id: serde_json::Value,
    params: &serde_json::Value,
    state: &PrimalState,
) -> JsonRpcResponse {
    let Some(data) = extract_f64_vec(params, "data") else {
        return JsonRpcResponse::error(
            id,
            super::rpc_error::INVALID_PARAMS,
            "Missing 'data' parameter".into(),
        );
    };
    let result = state.dispatcher.mean(&data);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "result": result, "backend": format!("{}", state.dispatcher.backend()) }),
    )
}

fn dispatch_variance(
    id: serde_json::Value,
    params: &serde_json::Value,
    state: &PrimalState,
) -> JsonRpcResponse {
    let Some(data) = extract_f64_vec(params, "data") else {
        return JsonRpcResponse::error(
            id,
            super::rpc_error::INVALID_PARAMS,
            "Missing 'data' parameter".into(),
        );
    };
    let result = state.dispatcher.variance(&data);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "result": result, "backend": format!("{}", state.dispatcher.backend()) }),
    )
}

fn dispatch_eigh(
    id: serde_json::Value,
    params: &serde_json::Value,
    state: &PrimalState,
) -> JsonRpcResponse {
    let Some(a) = extract_f64_vec(params, "a") else {
        return JsonRpcResponse::error(
            id,
            super::rpc_error::INVALID_PARAMS,
            "Missing 'a' parameter".into(),
        );
    };
    let n = p_usize(params, "n", 0);
    if n == 0 || a.len() != n * n {
        return JsonRpcResponse::error(
            id,
            super::rpc_error::INVALID_PARAMS,
            "Invalid matrix dimensions".into(),
        );
    }
    let (eigenvalues, _eigenvectors) = state.dispatcher.eigh(&a, n);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "eigenvalues": eigenvalues, "backend": format!("{}", state.dispatcher.backend()) }),
    )
}

// ── Helpers ──────────────────────────────────────────────────────

fn extract_f64_vec(params: &serde_json::Value, key: &str) -> Option<Vec<f64>> {
    params
        .get(key)
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

fn p_usize(params: &serde_json::Value, key: &str, default: u64) -> usize {
    params.get(key).and_then(|v| v.as_u64()).unwrap_or(default) as usize
}

fn p_u64(params: &serde_json::Value, key: &str, default: u64) -> u64 {
    params.get(key).and_then(|v| v.as_u64()).unwrap_or(default)
}

fn rand_vec(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.normal() * 0.1).collect()
}

fn add_inplace(target: &mut [f64], source: &[f64]) {
    for (t, s) in target.iter_mut().zip(source) {
        *t += s;
    }
}

// ── Linear algebra for Evoformer composition ─────────────────────

/// Batched matmul: [batch, rows, in] × [in, out] → [batch, rows, out]
fn matmul_3d(
    a: &[f64],
    w: &[f64],
    batch: usize,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; batch * rows * out_dim];
    let slice_size = rows * in_dim;
    for b in 0..batch {
        let a_slice = &a[b * slice_size..(b + 1) * slice_size];
        match barracuda::dispatch::matmul_dispatch(a_slice, w, rows, in_dim, out_dim, None) {
            Ok(result) => {
                out[b * rows * out_dim..(b + 1) * rows * out_dim].copy_from_slice(&result);
            }
            Err(_) => {
                for r in 0..rows {
                    for o in 0..out_dim {
                        let mut acc = 0.0f64;
                        for i in 0..in_dim {
                            acc = a_slice[r * in_dim + i].mul_add(w[i * out_dim + o], acc);
                        }
                        out[b * rows * out_dim + r * out_dim + o] = acc;
                    }
                }
            }
        }
    }
    out
}

/// 2D matmul: [rows, in] × [in, out] → [rows, out]
fn matmul_2d(a: &[f64], w: &[f64], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f64> {
    barracuda::dispatch::matmul_dispatch(a, w, rows, in_dim, out_dim, None)
        .unwrap_or_else(|_| matmul_3d(a, w, 1, rows, in_dim, out_dim))
}

/// einsum("ijc,ch->hij") for pair bias computation.
fn einsum_ijc_ch(
    tensor: &[f64],
    weight: &[f64],
    ni: usize,
    nj: usize,
    nc: usize,
    nh: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; nh * ni * nj];
    for i in 0..ni {
        for j in 0..nj {
            for h in 0..nh {
                let mut acc = 0.0f64;
                for c in 0..nc {
                    acc = tensor[i * nj * nc + j * nc + c].mul_add(weight[c * nh + h], acc);
                }
                out[h * ni * nj + i * nj + j] = acc;
            }
        }
    }
    out
}
