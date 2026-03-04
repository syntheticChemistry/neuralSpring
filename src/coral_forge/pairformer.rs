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

#![expect(
    clippy::cast_precision_loss,
    reason = "domain-specific numeric patterns"
)]

/// Base frequency for sinusoidal positional encoding (Vaswani et al. 2017, §3.5).
const SINUSOIDAL_BASE: f64 = 10_000.0;

/// Sinusoidal timestep embedding (Vaswani et al. 2017).
#[must_use]
pub fn sinusoidal_embedding(t: f64, d_model: usize) -> Vec<f64> {
    let d = d_model as f64;
    (0..d_model)
        .map(|i| {
            let base_idx = if i % 2 == 0 { i } else { i - 1 };
            let freq = SINUSOIDAL_BASE.powf(base_idx as f64 / d);
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
    let cond: Vec<f64> = (0..d)
        .map(|j| {
            t_emb.iter().enumerate().fold(b_cond[j], |acc, (k, &te)| {
                te.mul_add(w_cond[k * d + j], acc)
            })
        })
        .collect();

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
#[expect(
    clippy::too_many_lines,
    reason = "Pairformer block is a single fused attention+MLP pass; splitting would fragment the algorithm"
)]
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
        input
            .chunks_exact(d)
            .flat_map(|x| {
                (0..out_d).map(move |j| {
                    x.iter().enumerate().fold(0.0_f64, |acc, (k, &xk)| {
                        xk.mul_add(weight[k * out_d + j], acc)
                    })
                })
            })
            .collect()
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
        pair_out
            .iter_mut()
            .zip(gate.iter().zip(tri_out.iter()))
            .for_each(|(p, (&g, &t))| *p += g * t);
    }

    // 2. Triangle multiplicative incoming
    {
        let normed = ln(&pair_out);
        let proj_a = project(&normed, w.tri_in_wa, d);
        let proj_b = project(&normed, w.tri_in_wb, d);
        let gate = sigmoid_vec(&project(&normed, w.tri_in_wg, d));
        let tri_in = super::triangle_mul_incoming(&proj_a, &proj_b, n, d);
        pair_out
            .iter_mut()
            .zip(gate.iter().zip(tri_in.iter()))
            .for_each(|(p, (&g, &t))| *p += g * t);
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
        let trunc = d.min(h * hd);
        pair_out
            .chunks_exact_mut(d)
            .zip(attended.chunks_exact(h * hd))
            .for_each(|(po, att)| {
                po[..trunc]
                    .iter_mut()
                    .zip(att[..trunc].iter())
                    .for_each(|(p, &a)| *p += a);
            });
    }

    // 4. Pair transition FFN
    {
        let normed = ln(&pair_out);
        let ffn_out = super::diffusion::pair_transition_ffn(
            &normed, n, d, w.ffn_w1, w.ffn_b1, w.d_hidden, w.ffn_w2, w.ffn_b2,
        );
        pair_out
            .iter_mut()
            .zip(ffn_out.iter())
            .for_each(|(p, &f)| *p += f);
    }

    // 5. Timestep conditioning
    if let Some(t) = t_emb {
        pair_out = condition_pair_with_timestep(&pair_out, n, d, t, w.cond_w, w.cond_b);
    }

    pair_out
}

#[cfg(test)]
#[expect(
    clippy::similar_names,
    reason = "ffn_w1/ffn_b1/ffn_w2/ffn_b2 are standard NN weight/bias names"
)]
mod tests {
    use super::*;

    #[test]
    fn sinusoidal_shape() {
        let emb = sinusoidal_embedding(25.0, 8);
        assert_eq!(emb.len(), 8);
        assert!(emb.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sinusoidal_bounded() {
        for &t in &[0.0, 1.0, 50.0, 1000.0] {
            let emb = sinusoidal_embedding(t, 16);
            assert!(
                emb.iter().all(|v| (-1.0..=1.0).contains(v)),
                "sinusoidal values must be in [-1, 1] for t={t}"
            );
        }
    }

    #[test]
    fn sinusoidal_even_odd_pattern() {
        let emb = sinusoidal_embedding(10.0, 4);
        let freq0 = 10.0 / 10000.0_f64.powi(0);
        assert!((emb[0] - freq0.sin()).abs() < 1e-12, "even index = sin");
        assert!((emb[1] - freq0.cos()).abs() < 1e-12, "odd index = cos");
    }

    #[test]
    fn sinusoidal_dimension_one() {
        let emb = sinusoidal_embedding(5.0, 1);
        assert_eq!(emb.len(), 1);
        assert!(emb[0].is_finite());
    }

    #[test]
    fn different_timesteps_different_embeddings() {
        let e0 = sinusoidal_embedding(0.0, 8);
        let e25 = sinusoidal_embedding(25.0, 8);
        assert!(e0.iter().zip(e25.iter()).any(|(a, b)| (a - b).abs() > 1e-6));
    }

    #[test]
    fn conditioning_broadcast() {
        let pair = vec![1.0; 4 * 4 * 2];
        let t_emb = vec![0.5, 0.5];
        let w_cond = vec![1.0, 0.0, 0.0, 1.0];
        let b_cond = vec![0.0, 0.0];
        let out = condition_pair_with_timestep(&pair, 4, 2, &t_emb, &w_cond, &b_cond);
        assert!((out[0] - 1.5).abs() < 1e-14);
        assert!((out[1] - 1.5).abs() < 1e-14);
    }

    #[test]
    fn conditioning_bias_only() {
        let n = 2;
        let d = 2;
        let pair = vec![0.0; n * n * d];
        let t_emb = vec![0.0, 0.0];
        let w_cond = vec![0.0; 4];
        let b_cond = vec![3.0, 7.0];
        let out = condition_pair_with_timestep(&pair, n, d, &t_emb, &w_cond, &b_cond);
        for chunk in out.chunks_exact(d) {
            assert!((chunk[0] - 3.0).abs() < 1e-14, "bias[0] = 3.0");
            assert!((chunk[1] - 7.0).abs() < 1e-14, "bias[1] = 7.0");
        }
    }

    #[test]
    fn conditioning_preserves_length() {
        let n = 3;
        let d = 4;
        let pair = vec![1.0; n * n * d];
        let t_emb = vec![1.0; d];
        let w_cond = vec![0.1; d * d];
        let b_cond = vec![0.0; d];
        let out = condition_pair_with_timestep(&pair, n, d, &t_emb, &w_cond, &b_cond);
        assert_eq!(out.len(), pair.len());
    }

    #[test]
    fn pairformer_block_output_shape() {
        let n = 2;
        let d = 4;
        let h = 1;
        let hd = 4;
        let d_hidden = 8;

        let pair = vec![0.01; n * n * d];
        let ln_g = vec![1.0; d];
        let ln_b = vec![0.0; d];
        let tri_w = vec![0.01; d * d];
        let attn_w = vec![0.01; d * (h * hd)];
        let ffn_w1 = vec![0.01; d * d_hidden];
        let ffn_b1 = vec![0.0; d_hidden];
        let ffn_w2 = vec![0.01; d_hidden * d];
        let ffn_b2 = vec![0.0; d];
        let cond_w = vec![0.01; d * d];
        let cond_b = vec![0.0; d];

        let weights = PairformerWeights {
            ln_gamma: &ln_g,
            ln_beta: &ln_b,
            tri_out_wa: &tri_w,
            tri_out_wb: &tri_w,
            tri_out_wg: &tri_w,
            tri_in_wa: &tri_w,
            tri_in_wb: &tri_w,
            tri_in_wg: &tri_w,
            n_heads: h,
            head_dim: hd,
            tri_attn_wq: &attn_w,
            tri_attn_wk: &attn_w,
            tri_attn_wv: &attn_w,
            ffn_w1: &ffn_w1,
            ffn_b1: &ffn_b1,
            d_hidden,
            ffn_w2: &ffn_w2,
            ffn_b2: &ffn_b2,
            cond_w: &cond_w,
            cond_b: &cond_b,
        };

        let out = pairformer_block(&pair, n, d, &weights, None);
        assert_eq!(out.len(), n * n * d, "output shape must match input");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
    }

    #[test]
    fn pairformer_block_with_conditioning() {
        let n = 2;
        let d = 4;
        let h = 1;
        let hd = 4;
        let d_hidden = 8;

        let pair = vec![0.01; n * n * d];
        let ln_g = vec![1.0; d];
        let ln_b = vec![0.0; d];
        let tri_w = vec![0.01; d * d];
        let attn_w = vec![0.01; d * (h * hd)];
        let ffn_w1 = vec![0.01; d * d_hidden];
        let ffn_b1 = vec![0.0; d_hidden];
        let ffn_w2 = vec![0.01; d_hidden * d];
        let ffn_b2 = vec![0.0; d];
        let cond_w = vec![0.01; d * d];
        let cond_b = vec![0.0; d];

        let weights = PairformerWeights {
            ln_gamma: &ln_g,
            ln_beta: &ln_b,
            tri_out_wa: &tri_w,
            tri_out_wb: &tri_w,
            tri_out_wg: &tri_w,
            tri_in_wa: &tri_w,
            tri_in_wb: &tri_w,
            tri_in_wg: &tri_w,
            n_heads: h,
            head_dim: hd,
            tri_attn_wq: &attn_w,
            tri_attn_wk: &attn_w,
            tri_attn_wv: &attn_w,
            ffn_w1: &ffn_w1,
            ffn_b1: &ffn_b1,
            d_hidden,
            ffn_w2: &ffn_w2,
            ffn_b2: &ffn_b2,
            cond_w: &cond_w,
            cond_b: &cond_b,
        };

        let t_emb = sinusoidal_embedding(10.0, d);
        let out_no_t = pairformer_block(&pair, n, d, &weights, None);
        let out_t = pairformer_block(&pair, n, d, &weights, Some(&t_emb));
        assert_eq!(out_t.len(), out_no_t.len());
        let max_diff = out_no_t
            .iter()
            .zip(out_t.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-15,
            "timestep conditioning should change output"
        );
    }

    #[test]
    fn pairformer_deterministic() {
        let n = 2;
        let d = 4;
        let h = 1;
        let hd = 4;
        let d_hidden = 8;

        let pair = vec![0.05; n * n * d];
        let ln_g = vec![1.0; d];
        let ln_b = vec![0.0; d];
        let tri_w = vec![0.02; d * d];
        let attn_w = vec![0.02; d * (h * hd)];
        let ffn_w1 = vec![0.02; d * d_hidden];
        let ffn_b1 = vec![0.0; d_hidden];
        let ffn_w2 = vec![0.02; d_hidden * d];
        let ffn_b2 = vec![0.0; d];
        let cond_w = vec![0.01; d * d];
        let cond_b = vec![0.0; d];

        let weights = PairformerWeights {
            ln_gamma: &ln_g,
            ln_beta: &ln_b,
            tri_out_wa: &tri_w,
            tri_out_wb: &tri_w,
            tri_out_wg: &tri_w,
            tri_in_wa: &tri_w,
            tri_in_wb: &tri_w,
            tri_in_wg: &tri_w,
            n_heads: h,
            head_dim: hd,
            tri_attn_wq: &attn_w,
            tri_attn_wk: &attn_w,
            tri_attn_wv: &attn_w,
            ffn_w1: &ffn_w1,
            ffn_b1: &ffn_b1,
            d_hidden,
            ffn_w2: &ffn_w2,
            ffn_b2: &ffn_b2,
            cond_w: &cond_w,
            cond_b: &cond_b,
        };

        let out1 = pairformer_block(&pair, n, d, &weights, None);
        let out2 = pairformer_block(&pair, n, d, &weights, None);
        assert_eq!(out1, out2, "pairformer must be deterministic");
    }
}
