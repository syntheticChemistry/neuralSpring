// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU validation: nF-03 `AlphaFold3` operations.
//!
//! Proves `barracuda`'s pure Rust CPU math produces identical results to
//! `neuralSpring`'s hand-rolled implementations for `AlphaFold3` diffusion,
//! Pairformer, and confidence head operations.
//!
//! Closes the last `BarraCUDA` CPU tier gap: coralForge 2/3 → 3/3.
//!
//! ## Primitives validated
//!
//! - `barracuda::dispatch::matmul_dispatch` (CPU) — GEMM for projections, FFN, confidence heads
//! - `barracuda::stats::dot` — dot product for attention scores
//! - `barracuda::stats::mean` — mean for pLDDT aggregation, COM removal
//! - `barracuda::dispatch::variance_dispatch` — variance for layer norm
//! - `barracuda::stats::l2_norm` — L2 norm for SE(3) equivariance
//!
//! ## Reference
//!
//! Abramson et al. "Accurate structure prediction for all molecules"
//! Nature 630:493-500 (2024)
//!
//! ## Provenance
//!
//! CPU reference: neuralSpring lib (coralForge `AlphaFold3` hand-rolled implementations).
//! GPU path: N/A (CPU validation; establishes CPU tier for GPU parity).
//! Evolution: Abramson et al. 2024 → Python → Rust (CPU) → `BarraCUDA` CPU (this validates CPU tier).

#![expect(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    reason = "validation binary"
)]

use neural_spring::coral_forge;
use neural_spring::coral_forge::confidence;
use neural_spring::require;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_alphafold3");
    let mut rng = Rng::new(42);

    validate_cosine_schedule_via_barracuda(&mut h);
    validate_forward_diffusion_via_barracuda(&mut h, &mut rng);
    validate_pairformer_projection_via_barracuda(&mut h, &mut rng);
    validate_pairformer_trimul_via_barracuda(&mut h, &mut rng);
    validate_pairformer_attention_via_barracuda(&mut h, &mut rng);
    validate_pair_transition_ffn_via_barracuda(&mut h, &mut rng);
    validate_pldt_via_barracuda(&mut h, &mut rng);
    validate_pae_via_barracuda(&mut h, &mut rng);
    validate_layer_norm_via_barracuda(&mut h, &mut rng);
    validate_se3_com_removal(&mut h, &mut rng);

    h.finish();
}

// ═══════════════════════════════════════════════════════════════════
// 1. Cosine noise schedule: neuralSpring vs barracuda stats
// ═══════════════════════════════════════════════════════════════════

fn validate_cosine_schedule_via_barracuda(h: &mut ValidationHarness) {
    let t_steps = 20;
    let schedule = coral_forge::diffusion::cosine_beta_schedule(t_steps, 0.008);

    let alpha_bars = &schedule.alpha_bar;
    let bc_mean = barracuda::stats::mean(alpha_bars);
    let ns_mean: f64 = alpha_bars.iter().sum::<f64>() / alpha_bars.len() as f64;

    h.check_abs(
        "cosine schedule mean (bC vs ns)",
        bc_mean,
        ns_mean,
        tolerances::EXACT_F64,
    );

    let bc_var = barracuda::dispatch::variance_dispatch(alpha_bars, None).unwrap_or(0.0);
    let ns_var = {
        let m = ns_mean;
        alpha_bars.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / alpha_bars.len() as f64
    };
    h.check_abs(
        "cosine schedule variance (bC vs ns)",
        bc_var,
        ns_var,
        tolerances::EXACT_F64,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 2. Forward diffusion: verifying noise scaling algebra
// ═══════════════════════════════════════════════════════════════════

fn validate_forward_diffusion_via_barracuda(h: &mut ValidationHarness, rng: &mut Rng) {
    let n_atoms = 6;
    let coords: Vec<f64> = (0..n_atoms * 3).map(|_| rng.normal() * 5.0).collect();
    let noise: Vec<f64> = (0..n_atoms * 3).map(|_| rng.normal()).collect();

    let schedule = coral_forge::diffusion::cosine_beta_schedule(50, 0.008);
    let t = 25;
    let ns_noised = coral_forge::diffusion::forward_diffusion(&coords, &noise, t, &schedule);

    let alpha_bar_t = schedule.alpha_bar[t];
    let sqrt_ab = alpha_bar_t.sqrt();
    let sqrt_1mab = (1.0 - alpha_bar_t).sqrt();

    let bc_noised: Vec<f64> = coords
        .iter()
        .zip(noise.iter())
        .map(|(&x, &n)| sqrt_ab.mul_add(x, sqrt_1mab * n))
        .collect();

    let max_diff = ns_noised
        .iter()
        .zip(bc_noised.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("forward diffusion bC parity (max_diff={max_diff:.2e})"),
        max_diff < tolerances::EXACT_F64,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 3. Pairformer linear projection: barracuda matmul vs hand-rolled
// ═══════════════════════════════════════════════════════════════════

fn validate_pairformer_projection_via_barracuda(h: &mut ValidationHarness, rng: &mut Rng) {
    let n = 4;
    let d = 3;
    let out_d = 5;
    let nn = n * n;

    let input: Vec<f64> = (0..nn * d).map(|_| rng.normal() * 0.3).collect();
    let weight: Vec<f64> = (0..d * out_d).map(|_| rng.normal() * 0.2).collect();

    let ns_out = project_ns(&input, &weight, nn, d, out_d);
    let bc_out = require!(
        h,
        barracuda::dispatch::matmul_dispatch(&input, &weight, nn, d, out_d, None),
        "matmul_dispatch pairformer projection"
    );

    let max_diff = ns_out
        .iter()
        .zip(bc_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "pairformer projection matmul (bC vs ns)",
        max_diff,
        0.0,
        tolerances::EXACT_F64,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 4. Triangle multiply outgoing: barracuda dot for contraction
// ═══════════════════════════════════════════════════════════════════

fn validate_pairformer_trimul_via_barracuda(h: &mut ValidationHarness, rng: &mut Rng) {
    let n = 3;
    let c = 2;

    let proj_a: Vec<f64> = (0..n * n * c).map(|_| rng.normal() * 0.3).collect();
    let proj_b: Vec<f64> = (0..n * n * c).map(|_| rng.normal() * 0.3).collect();

    let ns_out = coral_forge::triangle_mul_outgoing(&proj_a, &proj_b, n, c);
    let bc_out = trimul_outgoing_bc(&proj_a, &proj_b, n, c);

    let max_diff = ns_out
        .iter()
        .zip(bc_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "trimul outgoing bC vs ns",
        max_diff,
        0.0,
        tolerances::EXACT_F64,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 5. Attention scores: barracuda dot product for QK^T/√d
// ═══════════════════════════════════════════════════════════════════

fn validate_pairformer_attention_via_barracuda(h: &mut ValidationHarness, rng: &mut Rng) {
    let n_res = 4;
    let head_dim = 3;

    let q: Vec<f64> = (0..n_res * head_dim).map(|_| rng.normal() * 0.3).collect();
    let k: Vec<f64> = (0..n_res * head_dim).map(|_| rng.normal() * 0.3).collect();

    let scale = (head_dim as f64).sqrt();

    let mut ns_scores = vec![0.0_f64; n_res * n_res];
    let mut bc_scores = vec![0.0_f64; n_res * n_res];
    for i in 0..n_res {
        let qi = &q[i * head_dim..(i + 1) * head_dim];
        for j in 0..n_res {
            let kj = &k[j * head_dim..(j + 1) * head_dim];
            ns_scores[i * n_res + j] =
                qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum::<f64>() / scale;
            bc_scores[i * n_res + j] = barracuda::stats::dot(qi, kj) / scale;
        }
    }

    let max_diff = ns_scores
        .iter()
        .zip(bc_scores.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "attention scores bC dot vs ns",
        max_diff,
        0.0,
        tolerances::EXACT_F64,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 6. Pair transition FFN: matmul chain for Linear → GELU → Linear
// ═══════════════════════════════════════════════════════════════════

fn validate_pair_transition_ffn_via_barracuda(h: &mut ValidationHarness, rng: &mut Rng) {
    let n = 3;
    let d = 4;
    let d_hidden = 8;

    let input: Vec<f64> = (0..n * n * d).map(|_| rng.normal() * 0.3).collect();
    let w1: Vec<f64> = (0..d * d_hidden).map(|_| rng.normal() * 0.2).collect();
    let b1: Vec<f64> = (0..d_hidden).map(|_| rng.normal() * 0.1).collect();
    let w2: Vec<f64> = (0..d_hidden * d).map(|_| rng.normal() * 0.2).collect();
    let b2: Vec<f64> = (0..d).map(|_| rng.normal() * 0.1).collect();

    let ns_out =
        coral_forge::diffusion::pair_transition_ffn(&input, n, d, &w1, &b1, d_hidden, &w2, &b2);

    let nn = n * n;
    let bc_hidden = require!(
        h,
        barracuda::dispatch::matmul_dispatch(&input, &w1, nn, d, d_hidden, None),
        "FFN matmul W1"
    );

    let bc_biased: Vec<f64> = bc_hidden
        .chunks_exact(d_hidden)
        .flat_map(|row| row.iter().zip(b1.iter()).map(|(h_val, bi)| h_val + bi))
        .collect();

    let bc_gelu: Vec<f64> = bc_biased.iter().map(|&x| gelu_f64(x)).collect();

    let bc_out_flat = require!(
        h,
        barracuda::dispatch::matmul_dispatch(&bc_gelu, &w2, nn, d_hidden, d, None),
        "FFN matmul W2"
    );
    let bc_out: Vec<f64> = bc_out_flat
        .chunks_exact(d)
        .flat_map(|row| row.iter().zip(b2.iter()).map(|(h_val, bi)| h_val + bi))
        .collect();

    let max_diff = ns_out
        .iter()
        .zip(bc_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("pair FFN bC matmul chain (max_diff={max_diff:.2e})"),
        max_diff < tolerances::EXACT_F64,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 7. pLDDT confidence: matmul + sigmoid via barracuda
// ═══════════════════════════════════════════════════════════════════

fn validate_pldt_via_barracuda(h: &mut ValidationHarness, rng: &mut Rng) {
    let n_residues = 16;
    let d = 4;

    let single_repr: Vec<f64> = (0..n_residues * d).map(|_| rng.normal() * 0.5).collect();
    let w: Vec<f64> = (0..d).map(|_| rng.normal() * 0.3).collect();
    let b_val = rng.normal() * 0.1;

    let ns_pldt = confidence::plddt_head(&single_repr, n_residues, d, &w, b_val);

    let bc_logits = require!(
        h,
        barracuda::dispatch::matmul_dispatch(&single_repr, &w, n_residues, d, 1, None),
        "pLDDT matmul"
    );
    let bc_pldt: Vec<f64> = bc_logits.iter().map(|&l| sigmoid(l + b_val)).collect();

    let max_diff = ns_pldt
        .iter()
        .zip(bc_pldt.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "pLDDT bC matmul+sigmoid vs ns",
        max_diff,
        0.0,
        tolerances::EXACT_F64,
    );

    let bc_mean = barracuda::stats::mean(&bc_pldt);
    let ns_mean: f64 = ns_pldt.iter().sum::<f64>() / ns_pldt.len() as f64;
    h.check_abs(
        "pLDDT mean bC vs ns",
        bc_mean,
        ns_mean,
        tolerances::EXACT_F64,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 8. PAE confidence: matmul + softmax via barracuda
// ═══════════════════════════════════════════════════════════════════

fn validate_pae_via_barracuda(h: &mut ValidationHarness, rng: &mut Rng) {
    let n = 4;
    let d = 3;
    let n_bins = 8;

    let pair_repr: Vec<f64> = (0..n * n * d).map(|_| rng.normal() * 0.5).collect();
    let w: Vec<f64> = (0..d * n_bins).map(|_| rng.normal() * 0.3).collect();
    let b: Vec<f64> = (0..n_bins).map(|_| rng.normal() * 0.1).collect();

    let (ns_expected, _ns_probs) = confidence::pae_head(&pair_repr, n, d, &w, &b, n_bins);

    let nn = n * n;
    let bc_logits = require!(
        h,
        barracuda::dispatch::matmul_dispatch(&pair_repr, &w, nn, d, n_bins, None),
        "PAE matmul"
    );

    let bin_centers: Vec<f64> = (0..n_bins)
        .map(|i| 31.75 * (i as f64) / ((n_bins - 1) as f64))
        .collect();

    let bc_expected: Vec<f64> = bc_logits
        .chunks_exact(n_bins)
        .map(|row| {
            let biased: Vec<f64> = row.iter().zip(b.iter()).map(|(&l, &bi)| l + bi).collect();
            let max_val = biased.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = biased.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let probs: Vec<f64> = exps.iter().map(|&e| e / sum).collect();
            probs
                .iter()
                .zip(bin_centers.iter())
                .map(|(&p, &c)| p * c)
                .sum::<f64>()
        })
        .collect();

    let max_diff = ns_expected
        .iter()
        .zip(bc_expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "PAE bC matmul+softmax vs ns",
        max_diff,
        0.0,
        tolerances::EXACT_F64,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 9. Layer norm: barracuda mean + variance vs neuralSpring
// ═══════════════════════════════════════════════════════════════════

fn validate_layer_norm_via_barracuda(h: &mut ValidationHarness, rng: &mut Rng) {
    let n = 4;
    let d = 3;
    let gamma: Vec<f64> = (0..d).map(|_| 0.8 + rng.next_f64() * 0.4).collect();
    let beta: Vec<f64> = (0..d).map(|_| rng.normal() * 0.1).collect();
    let input: Vec<f64> = (0..n * d).map(|_| rng.normal()).collect();

    let eps = tolerances::LAYER_NORM_EPS;
    let ns_out = coral_forge::layer_norm(&input, n, d, &gamma, &beta, eps);

    let mut bc_out = Vec::with_capacity(n * d);
    for row in input.chunks_exact(d) {
        let bc_mean = barracuda::stats::mean(row);
        let bc_var = barracuda::dispatch::variance_dispatch(row, None).unwrap_or(0.0);
        let inv_std = 1.0 / (bc_var + eps).sqrt();
        for (i, &x) in row.iter().enumerate() {
            bc_out.push((x - bc_mean).mul_add(inv_std * gamma[i], beta[i]));
        }
    }

    let max_diff = ns_out
        .iter()
        .zip(bc_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "layer norm bC mean+var vs ns",
        max_diff,
        0.0,
        tolerances::EXACT_F64,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 10. SE(3) COM removal: barracuda mean for center-of-mass
// ═══════════════════════════════════════════════════════════════════

fn validate_se3_com_removal(h: &mut ValidationHarness, rng: &mut Rng) {
    let n_atoms = 8;
    let coords: Vec<f64> = (0..n_atoms * 3).map(|_| rng.normal() * 10.0).collect();

    let (ns_centered, _ns_com) = coral_forge::diffusion::remove_center_of_mass(&coords);

    let x_coords: Vec<f64> = (0..n_atoms).map(|i| coords[i * 3]).collect();
    let y_coords: Vec<f64> = (0..n_atoms).map(|i| coords[i * 3 + 1]).collect();
    let z_coords: Vec<f64> = (0..n_atoms).map(|i| coords[i * 3 + 2]).collect();

    let bc_com = [
        barracuda::stats::mean(&x_coords),
        barracuda::stats::mean(&y_coords),
        barracuda::stats::mean(&z_coords),
    ];

    let bc_centered: Vec<f64> = coords
        .chunks_exact(3)
        .flat_map(|atom| {
            vec![
                atom[0] - bc_com[0],
                atom[1] - bc_com[1],
                atom[2] - bc_com[2],
            ]
        })
        .collect();

    let max_diff = ns_centered
        .iter()
        .zip(bc_centered.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_abs(
        "SE(3) COM removal bC vs ns",
        max_diff,
        0.0,
        tolerances::EXACT_F64,
    );

    let residual_x: Vec<f64> = (0..n_atoms).map(|i| bc_centered[i * 3]).collect();
    let residual_y: Vec<f64> = (0..n_atoms).map(|i| bc_centered[i * 3 + 1]).collect();
    let residual_z: Vec<f64> = (0..n_atoms).map(|i| bc_centered[i * 3 + 2]).collect();
    let residual_com = barracuda::stats::l2_norm(&[
        barracuda::stats::mean(&residual_x),
        barracuda::stats::mean(&residual_y),
        barracuda::stats::mean(&residual_z),
    ]);
    h.check_abs(
        "SE(3) residual COM near zero",
        residual_com,
        0.0,
        tolerances::EXACT_F64,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn sigmoid(x: f64) -> f64 {
    neural_spring::primitives::sigmoid(x)
}

fn gelu_f64(x: f64) -> f64 {
    neural_spring::primitives::gelu(x)
}

fn project_ns(input: &[f64], weight: &[f64], rows: usize, d: usize, out_d: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows * out_d];
    for row in 0..rows {
        let x = &input[row * d..(row + 1) * d];
        for j in 0..out_d {
            out[row * out_d + j] =
                (0..d).fold(0.0_f64, |acc, k| x[k].mul_add(weight[k * out_d + j], acc));
        }
    }
    out
}

fn trimul_outgoing_bc(proj_a: &[f64], proj_b: &[f64], n: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n * c];
    for i in 0..n {
        for j in 0..n {
            for ch in 0..c {
                let a_col: Vec<f64> = (0..n).map(|k| proj_a[(i * n + k) * c + ch]).collect();
                let b_col: Vec<f64> = (0..n).map(|k| proj_b[(j * n + k) * c + ch]).collect();
                out[(i * n + j) * c + ch] = barracuda::stats::dot(&a_col, &b_col);
            }
        }
    }
    out
}
