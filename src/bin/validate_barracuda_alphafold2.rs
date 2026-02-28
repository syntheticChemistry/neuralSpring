// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::suboptimal_flops,
    clippy::expect_used,
    clippy::unwrap_used
)]

//! `BarraCUDA` CPU validation: nF-02 `AlphaFold2` operations.
//!
//! Proves `barracuda`'s pure Rust CPU math (`eigh_f64`, matmul via dispatch,
//! stats primitives) produces identical results to `neuralSpring`'s hand-rolled
//! implementations for Evoformer/Structure Module operations.
//!
//! ## Primitives validated
//!
//! - `barracuda::linalg::eigh_f64` — eigendecomposition (IPA attention spectral analysis)
//! - `barracuda::dispatch::matmul_dispatch` (CPU path) — GEMM for SDPA
//! - `barracuda::dispatch::transpose_dispatch` (CPU path) — matrix transposition
//! - `barracuda::stats::mean` — mean computation (layer norm)
//! - `barracuda::dispatch::variance_dispatch` (CPU path) — population variance (layer norm)
//! - `barracuda::stats::dot` — dot product (SDPA scores)
//! - `barracuda::stats::l2_norm` — L2 norm
//!
//! ## Reference
//!
//! Jumper et al. "Highly accurate protein structure prediction with `AlphaFold`"
//! Nature 596:583-589 (2021)

use neural_spring::coral_forge;
use neural_spring::require;
use neural_spring::rng::Rng;
use neural_spring::spectral_commutativity::{mat_mul, random_matrix, random_symmetric, transpose};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_alphafold2");
    let mut rng = Rng::new(42);

    validate_matmul_square(&mut h, &mut rng);
    validate_matmul_sdpa_shape(&mut h, &mut rng);
    validate_transpose(&mut h, &mut rng);
    validate_mean(&mut h, &mut rng);
    validate_variance_population(&mut h, &mut rng);
    validate_dot(&mut h, &mut rng);
    validate_l2_norm(&mut h, &mut rng);
    validate_sdpa_scores(&mut h, &mut rng);
    validate_layer_norm(&mut h, &mut rng);
    validate_eigh_symmetric(&mut h, &mut rng);
    validate_eigh_reconstruct(&mut h, &mut rng);
    validate_eigh_vs_householder(&mut h, &mut rng);
    validate_attention_weight_eigh(&mut h, &mut rng);

    h.finish();
}

/// 1. Square matmul: `neuralSpring` `spectral_commutativity` vs `barracuda` CPU dispatch.
fn validate_matmul_square(h: &mut ValidationHarness, rng: &mut Rng) {
    let n = 8_usize;
    let a = random_matrix(n, rng);
    let b = random_matrix(n, rng);

    let ns_result = mat_mul(&a, &b, n);
    let bc_result = require!(
        h,
        barracuda::dispatch::matmul_dispatch(&a, &b, n, n, n, None),
        "matmul_dispatch CPU"
    );

    let max_diff = ns_result
        .iter()
        .zip(bc_result.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("matmul square {n}×{n} (max_diff={max_diff:.2e})"),
        max_diff,
        tolerances::CROSS_LANGUAGE,
    );
}

/// 2. Non-square matmul (SDPA shape): `Q[q_len, head_dim]` @ `K^T[head_dim, kv_len]`.
fn validate_matmul_sdpa_shape(h: &mut ValidationHarness, rng: &mut Rng) {
    let q_len = 4_usize;
    let kv_len = 6_usize;
    let head_dim = 8_usize;

    let scale = 0.5 / (head_dim as f64).sqrt();
    let q: Vec<f64> = (0..q_len * head_dim)
        .map(|_| rng.uniform() * scale + 0.5)
        .collect();
    let k: Vec<f64> = (0..kv_len * head_dim)
        .map(|_| rng.uniform() * scale + 0.5)
        .collect();

    let ns_scores = coral_forge::sdpa_scores(&q, &k, 1, 1, q_len, kv_len, head_dim);

    let k_t = require!(
        h,
        barracuda::dispatch::transpose_dispatch(&k, kv_len, head_dim, None),
        "transpose CPU"
    );
    let bc_scores_raw = require!(
        h,
        barracuda::dispatch::matmul_dispatch(&q, &k_t, q_len, head_dim, kv_len, None),
        "matmul CPU"
    );
    let scale_factor = 1.0 / (head_dim as f64).sqrt();
    let bc_scores: Vec<f64> = bc_scores_raw.iter().map(|x| x * scale_factor).collect();

    let max_diff = ns_scores
        .iter()
        .zip(bc_scores.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("SDPA matmul Q@K^T/√d (max_diff={max_diff:.2e})"),
        max_diff,
        tolerances::CROSS_LANGUAGE,
    );
}

/// 3. Transpose: `neuralSpring` vs `barracuda` CPU dispatch.
fn validate_transpose(h: &mut ValidationHarness, rng: &mut Rng) {
    let n = 8_usize;
    let a = random_matrix(n, rng);

    let ns_t = transpose(&a, n);
    let bc_t = require!(
        h,
        barracuda::dispatch::transpose_dispatch(&a, n, n, None),
        "transpose CPU"
    );

    let max_diff = ns_t
        .iter()
        .zip(bc_t.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("transpose {n}×{n} (max_diff={max_diff:.2e})"),
        max_diff,
        tolerances::CROSS_LANGUAGE,
    );
}

/// 4. Mean: `barracuda::stats::mean` vs manual.
fn validate_mean(h: &mut ValidationHarness, rng: &mut Rng) {
    let data: Vec<f64> = (0..64)
        .map(|i| f64::from(i) * 0.1 + rng.uniform() * 0.01)
        .collect();

    let bc_mean = barracuda::stats::mean(&data);
    let ns_mean = data.iter().sum::<f64>() / data.len() as f64;

    h.check_abs(
        "stats::mean",
        (bc_mean - ns_mean).abs(),
        0.0,
        tolerances::EXACT_F64,
    );
}

/// 5. Population variance: `barracuda` `variance_dispatch` (CPU) vs `layer_norm` convention.
fn validate_variance_population(h: &mut ValidationHarness, rng: &mut Rng) {
    let row: Vec<f64> = (0..16)
        .map(|i| f64::from(i) * 0.2 + rng.uniform() * 0.05)
        .collect();

    let bc_var = require!(
        h,
        barracuda::dispatch::variance_dispatch(&row, None),
        "variance CPU"
    );
    let mean = row.iter().sum::<f64>() / row.len() as f64;
    let ns_var = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / row.len() as f64;

    h.check_abs(
        "variance_dispatch (population) vs manual",
        (bc_var - ns_var).abs(),
        0.0,
        tolerances::EXACT_F64,
    );
}

/// 6. Dot product: `barracuda::stats::dot` vs manual.
fn validate_dot(h: &mut ValidationHarness, rng: &mut Rng) {
    let a: Vec<f64> = (0..32).map(|_| rng.uniform() * 0.5 + 0.5).collect();
    let b: Vec<f64> = (0..32).map(|_| rng.uniform() * 0.5 + 0.5).collect();

    let bc_dot = barracuda::stats::dot(&a, &b);
    let ns_dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    h.check_abs(
        "stats::dot",
        (bc_dot - ns_dot).abs(),
        0.0,
        tolerances::EXACT_F64,
    );
}

/// 7. L2 norm: `barracuda::stats::l2_norm` vs manual.
fn validate_l2_norm(h: &mut ValidationHarness, rng: &mut Rng) {
    let x: Vec<f64> = (0..32).map(|_| rng.uniform() * 0.5 + 0.5).collect();

    let bc_norm = barracuda::stats::l2_norm(&x);
    let ns_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();

    h.check_abs(
        "stats::l2_norm",
        (bc_norm - ns_norm).abs(),
        0.0,
        tolerances::EXACT_F64,
    );
}

/// 8. SDPA scores: `coral_forge::sdpa_scores` vs `barracuda` matmul + scale.
fn validate_sdpa_scores(h: &mut ValidationHarness, rng: &mut Rng) {
    let batch = 1_usize;
    let heads = 2_usize;
    let q_len = 4_usize;
    let kv_len = 4_usize;
    let head_dim = 8_usize;

    let scale = 0.5 / (head_dim as f64).sqrt();
    let q: Vec<f64> = (0..batch * heads * q_len * head_dim)
        .map(|_| rng.uniform() * scale + 0.5)
        .collect();
    let k: Vec<f64> = (0..batch * heads * kv_len * head_dim)
        .map(|_| rng.uniform() * scale + 0.5)
        .collect();

    let ns_scores = coral_forge::sdpa_scores(&q, &k, batch, heads, q_len, kv_len, head_dim);

    let mut bc_scores = vec![0.0; batch * heads * q_len * kv_len];
    let scale_factor = 1.0 / (head_dim as f64).sqrt();

    for b in 0..batch {
        for hd in 0..heads {
            let q_base = ((b * heads + hd) * q_len) * head_dim;
            let k_base = ((b * heads + hd) * kv_len) * head_dim;
            let q_head = &q[q_base..q_base + q_len * head_dim];
            let k_head = &k[k_base..k_base + kv_len * head_dim];
            let k_t = barracuda::dispatch::transpose_dispatch(k_head, kv_len, head_dim, None);
            let k_t = match k_t {
                Ok(kt) => kt,
                Err(e) => {
                    h.check_bool(&format!("SDPA transpose [ERROR: {e}]"), false);
                    return;
                }
            };
            let scores =
                barracuda::dispatch::matmul_dispatch(q_head, &k_t, q_len, head_dim, kv_len, None);
            let scores = match scores {
                Ok(s) => s,
                Err(e) => {
                    h.check_bool(&format!("SDPA matmul [ERROR: {e}]"), false);
                    return;
                }
            };
            let out_base = ((b * heads + hd) * q_len) * kv_len;
            for (i, &v) in scores.iter().enumerate() {
                bc_scores[out_base + i] = v * scale_factor;
            }
        }
    }

    let max_diff = ns_scores
        .iter()
        .zip(bc_scores.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("sdpa_scores full (max_diff={max_diff:.2e})"),
        max_diff,
        tolerances::CROSS_LANGUAGE,
    );
}

/// 9. Layer norm: `coral_forge::layer_norm` vs `barracuda` mean + variance + normalize.
fn validate_layer_norm(h: &mut ValidationHarness, rng: &mut Rng) {
    let rows = 2_usize;
    let dim = 8_usize;
    let x: Vec<f64> = (0_i32..i32::try_from(rows * dim).expect("test size"))
        .map(|i| f64::from(i) * 0.3 + rng.uniform() * 0.1)
        .collect();
    let gamma: Vec<f64> = (0_i32..i32::try_from(dim).expect("test size"))
        .map(|i| 1.0 + f64::from(i) * 0.05)
        .collect();
    let beta: Vec<f64> = (0_i32..i32::try_from(dim).expect("test size"))
        .map(|i| f64::from(i) * 0.02)
        .collect();

    let ns_out = coral_forge::layer_norm(&x, rows, dim, &gamma, &beta, tolerances::LAYER_NORM_EPS);

    let mut bc_out = Vec::with_capacity(rows * dim);
    for row in x.chunks_exact(dim) {
        let mean = require!(h, barracuda::dispatch::mean_dispatch(row, None), "mean CPU");
        let var = require!(
            h,
            barracuda::dispatch::variance_dispatch(row, None),
            "variance CPU"
        );
        let inv_std = 1.0 / (var + tolerances::LAYER_NORM_EPS).sqrt();
        for (&xd, (&g, &b)) in row.iter().zip(gamma.iter().zip(beta.iter())) {
            bc_out.push(g.mul_add((xd - mean) * inv_std, b));
        }
    }

    let max_diff = ns_out
        .iter()
        .zip(bc_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("layer_norm (max_diff={max_diff:.2e})"),
        max_diff,
        tolerances::CROSS_LANGUAGE,
    );
}

/// 10. Eigh symmetric: `barracuda::linalg::eigh_f64` produces real sorted eigenvalues.
fn validate_eigh_symmetric(h: &mut ValidationHarness, rng: &mut Rng) {
    let n = 8_usize;
    let sym = random_symmetric(n, rng);

    match barracuda::linalg::eigh_f64(&sym, n) {
        Ok(eig) => {
            h.check_bool(
                &format!("eigh_f64 returns {n} eigenvalues"),
                eig.eigenvalues.len() == n,
            );
            let all_finite = eig.eigenvalues.iter().all(|&v| v.is_finite());
            h.check_bool("eigh eigenvalues finite", all_finite);
            let sorted = eig.eigenvalues.windows(2).all(|w| w[0] <= w[1]);
            h.check_bool("eigh eigenvalues sorted ascending", sorted);
        }
        Err(e) => {
            h.check_bool(&format!("eigh_f64 [ERROR: {e}]"), false);
        }
    }
}

/// 11. Eigh reconstruction: V*D*V^T ≈ A.
fn validate_eigh_reconstruct(h: &mut ValidationHarness, rng: &mut Rng) {
    let n = 8_usize;
    let sym = random_symmetric(n, rng);

    match barracuda::linalg::eigh_f64(&sym, n) {
        Ok(eig) => {
            let reconstructed = eig.reconstruct();
            let recon_err: f64 = sym
                .iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm: f64 = sym.iter().map(|x| x * x).sum::<f64>().sqrt();
            let rel_err = if norm > tolerances::ZERO_DETECTION {
                recon_err / norm
            } else {
                recon_err
            };
            h.check_upper(
                &format!("eigh reconstruct V*D*V^T (rel_err={rel_err:.2e})"),
                rel_err,
                tolerances::EIGH_JACOBI_RECONSTRUCT,
            );
        }
        Err(e) => {
            h.check_bool(&format!("eigh_f64 reconstruct [ERROR: {e}]"), false);
        }
    }
}

/// 12. Eigh Jacobi vs `Householder`: eigenvalues agree within algorithm tolerance.
fn validate_eigh_vs_householder(h: &mut ValidationHarness, rng: &mut Rng) {
    let n = 6_usize;
    let sym = random_symmetric(n, rng);

    let jacobi = match barracuda::linalg::eigh_f64(&sym, n) {
        Ok(e) => e,
        Err(e) => {
            h.check_bool(&format!("eigh_f64 Jacobi [ERROR: {e}]"), false);
            return;
        }
    };

    let householder = neural_spring::eigh::eigh_householder_qr(&sym, n);

    let max_diff = jacobi
        .eigenvalues
        .iter()
        .zip(householder.eigenvalues.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_upper(
        &format!("eigh Jacobi vs Householder (max_diff={max_diff:.2e})"),
        max_diff,
        tolerances::EIGH_JACOBI_EIGENVALUE,
    );
}

/// 13. IPA-style: `eigh` on attention weight matrix (symmetric) for spectral analysis.
fn validate_attention_weight_eigh(h: &mut ValidationHarness, rng: &mut Rng) {
    let n = 6_usize;
    let scale = 0.5 / (n as f64).sqrt();
    let attn: Vec<f64> = (0..n * n).map(|_| rng.uniform() * scale + 0.3).collect();
    let attn_sym: Vec<f64> = (0..n * n)
        .map(|ij| {
            let i = ij / n;
            let j = ij % n;
            (attn[i * n + j] + attn[j * n + i]) * 0.5
        })
        .collect();

    match barracuda::linalg::eigh_f64(&attn_sym, n) {
        Ok(eig) => {
            let trace: f64 = eig.eigenvalues.iter().sum();
            let trace_expected = attn_sym.iter().step_by(n + 1).sum::<f64>();
            h.check_abs(
                "eigh trace = sum(diag(A))",
                (trace - trace_expected).abs(),
                0.0,
                tolerances::EIGH_JACOBI_EIGENVALUE,
            );
            h.check_bool(
                "attention-weight eigh returns n eigenvalues",
                eig.eigenvalues.len() == n,
            );
            let all_finite = eig.eigenvalues.iter().all(|&v| v.is_finite());
            h.check_bool("attention-weight eigh eigenvalues finite", all_finite);
        }
        Err(e) => {
            h.check_bool(&format!("attention eigh [ERROR: {e}]"), false);
        }
    }
}
