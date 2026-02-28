// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pairformer block for `AlphaFold3` (nF-03 Phase B).
//!
//! The Pairformer is `AlphaFold3`'s simplified Evoformer that operates on pair
//! representations only (no MSA track). One block:
//!
//! 1. `LayerNorm` → Triangle multiplicative outgoing (Algorithm 11)
//! 2. `LayerNorm` → Triangle multiplicative incoming (Algorithm 12)
//! 3. `LayerNorm` → Triangle attention (Algorithms 13-14)
//! 4. `LayerNorm` → Pair transition FFN (Linear → GELU → Linear)
//! 5. (Optional) Timestep conditioning
//!
//! ~90% reuse of existing Evoformer primitives from nF-02.
//!
//! Reference: Abramson et al. Nature 630:493-500 (2024)

#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    clippy::similar_names
)]

/// Sinusoidal timestep embedding (Vaswani et al. 2017).
#[must_use]
pub fn sinusoidal_embedding(t: f64, d_model: usize) -> Vec<f64> {
    let d = d_model as f64;
    (0..d_model)
        .map(|i| {
            let base_idx = if i % 2 == 0 { i } else { i - 1 };
            let freq = 10000.0_f64.powf(base_idx as f64 / d);
            if i % 2 == 0 {
                (t / freq).sin()
            } else {
                (t / freq).cos()
            }
        })
        .collect()
}

/// Add timestep conditioning to pair representation via broadcast.
///
/// `pair_repr`: `[n*n*d]`, `t_emb`: `[d_model]`, `w_cond`: `[d_model*d]`, `b_cond`: `[d]`.
/// Projects `t_emb` to `[d]` and adds to every (i,j) pair.
///
/// # Panics
///
/// Panics if tensor dimensions are inconsistent.
#[must_use]
pub fn condition_pair_with_timestep(
    pair_repr: &[f64],
    n: usize,
    d: usize,
    t_emb: &[f64],
    w_cond: &[f64],
    b_cond: &[f64],
) -> Vec<f64> {
    let d_model = t_emb.len();
    assert_eq!(w_cond.len(), d_model * d);
    assert_eq!(b_cond.len(), d);
    assert_eq!(pair_repr.len(), n * n * d);

    // cond = t_emb @ w_cond + b_cond → [d]
    let mut cond = vec![0.0_f64; d];
    for j in 0..d {
        let mut acc = b_cond[j];
        for k in 0..d_model {
            acc = t_emb[k].mul_add(w_cond[k * d + j], acc);
        }
        cond[j] = acc;
    }

    // Broadcast-add to every pair
    let mut out = pair_repr.to_vec();
    for pair in out.chunks_exact_mut(d) {
        for (p, c) in pair.iter_mut().zip(cond.iter()) {
            *p += c;
        }
    }
    out
}

/// Full Pairformer block weights.
pub struct PairformerWeights<'a> {
    pub ln_gamma: &'a [f64],
    pub ln_beta: &'a [f64],
    pub tri_out_wa: &'a [f64],
    pub tri_out_wb: &'a [f64],
    pub tri_out_wg: &'a [f64],
    pub tri_in_wa: &'a [f64],
    pub tri_in_wb: &'a [f64],
    pub tri_in_wg: &'a [f64],
    pub n_heads: usize,
    pub head_dim: usize,
    pub tri_attn_wq: &'a [f64],
    pub tri_attn_wk: &'a [f64],
    pub tri_attn_wv: &'a [f64],
    pub ffn_w1: &'a [f64],
    pub ffn_b1: &'a [f64],
    pub d_hidden: usize,
    pub ffn_w2: &'a [f64],
    pub ffn_b2: &'a [f64],
    pub cond_w: &'a [f64],
    pub cond_b: &'a [f64],
}

/// One Pairformer block iteration.
///
/// `pair`: `[n*n*d]`, `t_emb`: optional timestep embedding `[d]`.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn pairformer_block(
    pair: &[f64],
    n: usize,
    d: usize,
    w: &PairformerWeights<'_>,
    t_emb: Option<&[f64]>,
) -> Vec<f64> {
    let eps = crate::tolerances::LAYER_NORM_EPS;
    let nn = n * n;

    let mut pair_out = pair.to_vec();

    // Helper: layer_norm in-place
    let ln =
        |input: &[f64]| -> Vec<f64> { super::layer_norm(input, nn, d, w.ln_gamma, w.ln_beta, eps) };

    // Helper: linear projection [nn*d] @ [d, out_d] → [nn*out_d]
    let project = |input: &[f64], weight: &[f64], out_d: usize| -> Vec<f64> {
        let mut out = vec![0.0_f64; nn * out_d];
        for row in 0..nn {
            let x = &input[row * d..(row + 1) * d];
            for j in 0..out_d {
                let mut acc = 0.0_f64;
                for k in 0..d {
                    acc = x[k].mul_add(weight[k * out_d + j], acc);
                }
                out[row * out_d + j] = acc;
            }
        }
        out
    };

    // Helper: sigmoid
    let sigmoid_vec =
        |input: &[f64]| -> Vec<f64> { input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect() };

    // 1. Triangle multiplicative outgoing
    {
        let normed = ln(&pair_out);
        let proj_a = project(&normed, w.tri_out_wa, d);
        let proj_b = project(&normed, w.tri_out_wb, d);
        let gate = sigmoid_vec(&project(&normed, w.tri_out_wg, d));
        let tri_out = super::triangle_mul_outgoing(&proj_a, &proj_b, n, d);
        for i in 0..pair_out.len() {
            pair_out[i] += gate[i] * tri_out[i];
        }
    }

    // 2. Triangle multiplicative incoming
    {
        let normed = ln(&pair_out);
        let proj_a = project(&normed, w.tri_in_wa, d);
        let proj_b = project(&normed, w.tri_in_wb, d);
        let gate = sigmoid_vec(&project(&normed, w.tri_in_wg, d));
        let tri_in = super::triangle_mul_incoming(&proj_a, &proj_b, n, d);
        for i in 0..pair_out.len() {
            pair_out[i] += gate[i] * tri_in[i];
        }
    }

    // 3. Triangle attention (Algorithms 13-14)
    {
        let normed = ln(&pair_out);
        let h = w.n_heads;
        let hd = w.head_dim;

        // Project to Q, K, V: [nn, d] → [nn, h*hd]
        let q_flat = project(&normed, w.tri_attn_wq, h * hd);
        let k_flat = project(&normed, w.tri_attn_wk, h * hd);
        let v_flat = project(&normed, w.tri_attn_wv, h * hd);

        // Reshape to [n, n, h, hd] and use pair bias from normed[:,:,0:h]
        let mut bias = vec![0.0_f64; h * n * n]; // [h, n, n]
        for hi in 0..h.min(d) {
            for i in 0..n {
                for j in 0..n {
                    bias[hi * n * n + i * n + j] = normed[(i * n + j) * d + hi];
                }
            }
        }

        // Triangle attention scores: for each row i, compute attention over j,k
        // scores[row, h, j, k] = sum_d Q[row,j,h,d]*K[row,k,h,d]/sqrt(hd) + bias[h,j,k]
        let scale = (hd as f64).sqrt();
        let mut scores = vec![0.0_f64; n * h * n * n]; // [R, H, N, N]
        for row in 0..n {
            for hi in 0..h {
                for j in 0..n {
                    for k in 0..n {
                        let mut dot = 0.0_f64;
                        for dd in 0..hd {
                            let qi = q_flat[(row * n + j) * (h * hd) + hi * hd + dd];
                            let ki = k_flat[(row * n + k) * (h * hd) + hi * hd + dd];
                            dot = qi.mul_add(ki, dot);
                        }
                        scores[row * h * n * n + hi * n * n + j * n + k] =
                            dot / scale + bias[hi * n * n + j * n + k];
                    }
                }
            }
        }

        // Softmax over last dim (k) for each (row, h, j)
        for row in 0..n {
            for hi in 0..h {
                for j in 0..n {
                    let base = row * h * n * n + hi * n * n + j * n;
                    let slice = &mut scores[base..base + n];
                    let max_val = slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut sum = 0.0_f64;
                    for s in slice.iter_mut() {
                        *s = (*s - max_val).exp();
                        sum += *s;
                    }
                    for s in slice.iter_mut() {
                        *s /= sum;
                    }
                }
            }
        }

        // Apply: attended[r,j,h,d] = sum_k attn[r,h,j,k] * V[r,k,h,d]
        let mut attended = vec![0.0_f64; n * n * h * hd]; // [R, N, H, D]
        for row in 0..n {
            for j in 0..n {
                for hi in 0..h {
                    for dd in 0..hd {
                        let mut acc = 0.0_f64;
                        for k in 0..n {
                            let attn_val = scores[row * h * n * n + hi * n * n + j * n + k];
                            let v_val = v_flat[(row * n + k) * (h * hd) + hi * hd + dd];
                            acc = attn_val.mul_add(v_val, acc);
                        }
                        attended[(row * n + j) * (h * hd) + hi * hd + dd] = acc;
                    }
                }
            }
        }

        // Merge heads, truncate to d
        for ij in 0..nn {
            for dd in 0..d.min(h * hd) {
                pair_out[ij * d + dd] += attended[ij * (h * hd) + dd];
            }
        }
    }

    // 4. Pair transition FFN
    {
        let normed = ln(&pair_out);
        let ffn_out = super::diffusion::pair_transition_ffn(
            &normed, n, d, w.ffn_w1, w.ffn_b1, w.d_hidden, w.ffn_w2, w.ffn_b2,
        );
        for i in 0..pair_out.len() {
            pair_out[i] += ffn_out[i];
        }
    }

    // 5. Timestep conditioning
    if let Some(t) = t_emb {
        pair_out = condition_pair_with_timestep(&pair_out, n, d, t, w.cond_w, w.cond_b);
    }

    pair_out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoidal_shape() {
        let emb = sinusoidal_embedding(25.0, 8);
        assert_eq!(emb.len(), 8);
        assert!(emb.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn different_timesteps_different_embeddings() {
        let e0 = sinusoidal_embedding(0.0, 8);
        let e25 = sinusoidal_embedding(25.0, 8);
        assert!(e0.iter().zip(e25.iter()).any(|(a, b)| (a - b).abs() > 1e-6));
    }

    #[test]
    fn conditioning_broadcast() {
        let pair = vec![1.0; 4 * 4 * 2]; // 4x4, d=2
        let t_emb = vec![0.5, 0.5];
        let w_cond = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let b_cond = vec![0.0, 0.0];
        let out = condition_pair_with_timestep(&pair, 4, 2, &t_emb, &w_cond, &b_cond);
        // Each element should be 1.0 + 0.5 = 1.5
        assert!((out[0] - 1.5).abs() < 1e-14);
        assert!((out[1] - 1.5).abs() < 1e-14);
    }
}
