// SPDX-License-Identifier: AGPL-3.0-or-later

//! Diffusion model primitives for `AlphaFold3` (nF-03 Phase A).
//!
//! Implements core diffusion operations from Abramson et al. "Accurate
//! structure prediction for all molecules" Nature 630:493-500 (2024):
//!
//! - Noise schedules (cosine, linear)
//! - Forward diffusion `q(x_t | x_0)`
//! - DDPM reverse step (stochastic)
//! - DDIM reverse step (deterministic)
//! - SE(3)-equivariant operations (center-of-mass removal)
//! - Pairformer transition FFN (Linear → GELU → Linear)
//! - Confidence heads (`pLDDT`, PAE)
//!
//! ## References
//!
//! - Ho et al. "Denoising Diffusion Probabilistic Models" `NeurIPS` (2020)
//! - Song et al. "Denoising Diffusion Implicit Models" ICLR (2021)
//! - Nichol & Dhariwal "Improved Denoising Diffusion" ICML (2021)

#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use std::f64::consts::PI;

/// Noise schedule: beta values and cumulative alpha-bar values.
#[derive(Debug, Clone)]
pub struct NoiseSchedule {
    pub betas: Vec<f64>,
    pub alpha_bar: Vec<f64>,
}

/// Cosine noise schedule (Nichol & Dhariwal 2021).
///
/// `alpha_bar_t = f(t)/f(0)` where `f(t) = cos((t/T + s)/(1+s) * π/2)²`.
#[must_use]
pub fn cosine_beta_schedule(t_steps: usize, s: f64) -> NoiseSchedule {
    let t = t_steps as f64;
    let f_vals: Vec<f64> = (0..=t_steps)
        .map(|i| ((i as f64 / t + s) / (1.0 + s) * (PI / 2.0)).cos().powi(2))
        .collect();

    let f0 = f_vals[0];
    let (betas, alpha_bar): (Vec<f64>, Vec<f64>) = f_vals
        .windows(2)
        .map(|w| {
            let ab = (w[1] / f0).clamp(crate::tolerances::DIFFUSION_ALPHA_BAR_FLOOR, 1.0);
            let ab_prev = (w[0] / f0).clamp(crate::tolerances::DIFFUSION_ALPHA_BAR_FLOOR, 1.0);
            let beta = (1.0 - ab / ab_prev).clamp(crate::tolerances::DIFFUSION_BETA_FLOOR, 0.999);
            (beta, ab)
        })
        .unzip();

    NoiseSchedule { betas, alpha_bar }
}

/// Linear noise schedule (Ho et al. 2020).
#[must_use]
pub fn linear_beta_schedule(t_steps: usize, beta_start: f64, beta_end: f64) -> NoiseSchedule {
    let denom = (t_steps - 1) as f64;
    let betas: Vec<f64> = (0..t_steps)
        .map(|i| (beta_end - beta_start).mul_add(i as f64 / denom, beta_start))
        .collect();

    let mut cum_prod = 1.0_f64;
    let alpha_bar: Vec<f64> = betas
        .iter()
        .map(|&b| {
            cum_prod *= 1.0 - b;
            cum_prod
        })
        .collect();

    NoiseSchedule { betas, alpha_bar }
}

/// Forward diffusion: `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise`.
///
/// `coords`: flat `[n_atoms, 3]`, `noise`: same shape, `t`: timestep index.
/// Returns `x_t` in the same layout.
#[must_use]
pub fn forward_diffusion(
    coords: &[f64],
    noise: &[f64],
    t: usize,
    schedule: &NoiseSchedule,
) -> Vec<f64> {
    let a_bar_t = schedule.alpha_bar[t];
    let sqrt_ab = a_bar_t.sqrt();
    let sqrt_one_minus = (1.0 - a_bar_t).sqrt();

    coords
        .iter()
        .zip(noise.iter())
        .map(|(&x, &eps)| sqrt_ab.mul_add(x, sqrt_one_minus * eps))
        .collect()
}

/// DDPM reverse step: stochastic denoising.
///
/// Returns `x_{t-1}` given `x_t`, predicted noise, and optional random noise `z`.
/// When `t == 0`, the variance term is dropped (deterministic final step).
#[must_use]
pub fn ddpm_reverse_step(
    x_t: &[f64],
    predicted_noise: &[f64],
    z: &[f64],
    t: usize,
    schedule: &NoiseSchedule,
) -> Vec<f64> {
    let beta_t = schedule.betas[t];
    let alpha_t = 1.0 - beta_t;
    let a_bar_t = schedule.alpha_bar[t];

    let coeff_x = 1.0 / alpha_t.sqrt();
    let coeff_eps = beta_t / (1.0 - a_bar_t).sqrt();

    if t == 0 {
        x_t.iter()
            .zip(predicted_noise.iter())
            .map(|(&x, &eps)| coeff_x * (x - coeff_eps * eps))
            .collect()
    } else {
        let sigma_t = beta_t.sqrt();
        x_t.iter()
            .zip(predicted_noise.iter())
            .zip(z.iter())
            .map(|((&x, &eps), &zi)| coeff_x * (x - coeff_eps * eps) + sigma_t * zi)
            .collect()
    }
}

/// DDIM reverse step: deterministic denoising (Song et al. 2021).
///
/// Returns `(x_{t-1}, predicted_x_0)`.
#[must_use]
pub fn ddim_reverse_step(
    x_t: &[f64],
    predicted_noise: &[f64],
    t: usize,
    schedule: &NoiseSchedule,
) -> (Vec<f64>, Vec<f64>) {
    let a_bar_t = schedule.alpha_bar[t];
    let a_bar_prev = if t > 0 {
        schedule.alpha_bar[t - 1]
    } else {
        1.0
    };

    let sqrt_ab_t = a_bar_t.sqrt();
    let sqrt_one_minus_t = (1.0 - a_bar_t).sqrt();
    let sqrt_ab_prev = a_bar_prev.sqrt();
    let sqrt_one_minus_prev = (1.0 - a_bar_prev).sqrt();

    let pred_x_0: Vec<f64> = x_t
        .iter()
        .zip(predicted_noise.iter())
        .map(|(&x, &eps)| (x - sqrt_one_minus_t * eps) / sqrt_ab_t)
        .collect();

    let x_prev: Vec<f64> = pred_x_0
        .iter()
        .zip(predicted_noise.iter())
        .map(|(&x0, &eps)| sqrt_ab_prev.mul_add(x0, sqrt_one_minus_prev * eps))
        .collect();

    (x_prev, pred_x_0)
}

/// Remove center of mass from coordinates. Returns `(centered, com)`.
///
/// `coords`: flat `[n_atoms * 3]`, interpreted as `n_atoms` 3D points.
///
/// # Panics
///
/// Panics if `coords` is empty or not a multiple of 3.
#[must_use]
pub fn remove_center_of_mass(coords: &[f64]) -> (Vec<f64>, [f64; 3]) {
    let n = coords.len() / 3;
    assert!(n > 0 && coords.len() == n * 3);
    let n_f = n as f64;

    let com = coords.chunks_exact(3).fold([0.0_f64; 3], |mut acc, atom| {
        acc[0] += atom[0];
        acc[1] += atom[1];
        acc[2] += atom[2];
        acc
    });
    let com = [com[0] / n_f, com[1] / n_f, com[2] / n_f];

    let centered: Vec<f64> = coords
        .chunks_exact(3)
        .flat_map(|atom| [atom[0] - com[0], atom[1] - com[1], atom[2] - com[2]])
        .collect();

    (centered, com)
}

/// SE(3)-equivariant forward diffusion: center → noise → re-center.
///
/// Returns `(noisy_centered, noise, original_com)`.
#[must_use]
pub fn se3_equivariant_noise(
    coords: &[f64],
    noise: &[f64],
    t: usize,
    schedule: &NoiseSchedule,
) -> (Vec<f64>, [f64; 3]) {
    let (centered, com) = remove_center_of_mass(coords);
    let noisy = forward_diffusion(&centered, noise, t, schedule);
    let (noisy_centered, _) = remove_center_of_mass(&noisy);
    (noisy_centered, com)
}

/// Pairformer transition FFN: `Linear → GELU → Linear`.
///
/// `pair_repr`: flat `[n * n * d_pair]`.
/// `w1`: `[d_pair, d_hidden]`, `b1`: `[d_hidden]`.
/// `w2`: `[d_hidden, d_pair]`, `b2`: `[d_pair]`.
///
/// # Panics
///
/// Panics if weight/bias dimensions are inconsistent.
#[must_use]
pub fn pair_transition_ffn(
    pair_repr: &[f64],
    n: usize,
    d_pair: usize,
    w1: &[f64],
    b1: &[f64],
    d_hidden: usize,
    w2: &[f64],
    b2: &[f64],
) -> Vec<f64> {
    let n_elements = n * n;
    assert_eq!(pair_repr.len(), n_elements * d_pair);
    assert_eq!(w1.len(), d_pair * d_hidden);
    assert_eq!(w2.len(), d_hidden * d_pair);

    let mut output = vec![0.0_f64; n_elements * d_pair];

    for row in 0..n_elements {
        let x = &pair_repr[row * d_pair..(row + 1) * d_pair];

        // Hidden = GELU(x @ W1 + b1)
        let mut hidden = vec![0.0_f64; d_hidden];
        for j in 0..d_hidden {
            let mut acc = b1[j];
            for k in 0..d_pair {
                acc = x[k].mul_add(w1[k * d_hidden + j], acc);
            }
            hidden[j] = super::gelu(acc);
        }

        // Output = hidden @ W2 + b2
        for j in 0..d_pair {
            let mut acc = b2[j];
            for k in 0..d_hidden {
                acc = hidden[k].mul_add(w2[k * d_pair + j], acc);
            }
            output[row * d_pair + j] = acc;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_schedule_monotone() {
        let sched = cosine_beta_schedule(50, 0.008);
        assert_eq!(sched.betas.len(), 50);
        assert_eq!(sched.alpha_bar.len(), 50);
        for w in sched.alpha_bar.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-15,
                "alpha_bar not monotonically decreasing"
            );
        }
        assert!(sched.alpha_bar[0] > 0.99);
    }

    #[test]
    fn linear_schedule_bounds() {
        let sched = linear_beta_schedule(50, 1e-4, 0.02);
        assert!((sched.betas[0] - 1e-4).abs() < 1e-10);
        assert!((sched.betas[49] - 0.02).abs() < 1e-10);
    }

    #[test]
    fn forward_preserves_shape() {
        let sched = cosine_beta_schedule(50, 0.008);
        let coords = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let x_t = forward_diffusion(&coords, &noise, 25, &sched);
        assert_eq!(x_t.len(), coords.len());
    }

    #[test]
    fn ddim_deterministic() {
        let sched = cosine_beta_schedule(50, 0.008);
        let x_t = vec![1.0, 2.0, 3.0];
        let eps = vec![0.5, 0.5, 0.5];
        let (a, _) = ddim_reverse_step(&x_t, &eps, 25, &sched);
        let (b, _) = ddim_reverse_step(&x_t, &eps, 25, &sched);
        assert_eq!(a, b, "DDIM should be deterministic");
    }

    #[test]
    fn com_removal_centers() {
        let coords = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let (centered, com) = remove_center_of_mass(&coords);
        let new_com: f64 = centered.chunks_exact(3).map(|c| c[0]).sum::<f64>() / 3.0;
        assert!(new_com.abs() < 1e-14);
        assert!((com[0] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn pair_transition_ffn_shape() {
        let n = 3;
        let d_pair = 4;
        let d_hidden = 8;
        let pair = vec![0.1; n * n * d_pair];
        let w1 = vec![0.01; d_pair * d_hidden];
        let b1 = vec![0.0; d_hidden];
        let w2 = vec![0.01; d_hidden * d_pair];
        let b2 = vec![0.0; d_pair];
        let out = pair_transition_ffn(&pair, n, d_pair, &w1, &b1, d_hidden, &w2, &b2);
        assert_eq!(out.len(), n * n * d_pair);
    }
}
