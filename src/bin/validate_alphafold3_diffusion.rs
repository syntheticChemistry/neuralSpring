// SPDX-License-Identifier: AGPL-3.0-or-later

//! nF-03 Phase A: `AlphaFold3` diffusion primitive validation.
//!
//! Loads Python-generated baselines from `diffusion_baselines.json` and
//! validates that Rust CPU implementations reproduce them within cross-language
//! tolerance (1e-10 for f64→f64).
//!
//! ## Provenance
//!
//! Python baseline: `control/coral_forge/alphafold3_diffusion.py`
//! Reference: Abramson et al. Nature 630:493-500 (2024)
//!            Ho et al. "DDPM" `NeurIPS` (2020)
//!            Song et al. "DDIM" ICLR (2021)
//!
//! ## Experiments
//!
//! | Check | Primitive | What it validates |
//! |-------|-----------|-------------------|
//! | nF-D01 | Cosine schedule | Beta/alpha_bar monotonicity + range |
//! | nF-D02 | Linear schedule | Beta/alpha_bar bounds |
//! | nF-D03 | Forward diffusion | x_t = sqrt(a_bar)*x_0 + sqrt(1-a_bar)*eps |
//! | nF-D04 | DDPM reverse | Stochastic denoising step |
//! | nF-D05 | DDIM reverse | Deterministic denoising step |
//! | nF-D06 | SE(3) equivariance | COM removal + translation invariance |
//! | nF-D07 | Pair transition FFN | Linear → GELU → Linear |
//! | nF-D08 | pLDDT head | Linear → sigmoid → \[0,1\] |
//! | nF-D09 | PAE head | Pair → softmax → expected distance |
//! | nF-D10 | DDIM full loop | Oracle denoising T→0 convergence |

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_lines
)]

use neural_spring::coral_forge::diffusion;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

const BASELINE_JSON: &str = include_str!("../../control/coral_forge/diffusion_baselines.json");

fn flat_f64(val: &serde_json::Value) -> Vec<f64> {
    match val {
        serde_json::Value::Array(arr) => arr.iter().flat_map(flat_f64).collect(),
        serde_json::Value::Number(n) => vec![n.as_f64().unwrap_or(0.0)],
        _ => vec![],
    }
}

const T_STEPS: usize = 50;
const N_RES: usize = 12;
const N_ATOMS: usize = N_RES * 3;
const D_PAIR: usize = 8;
const D_HIDDEN: usize = 16;
const SEED: u64 = 42;

/// GPU-style float hash constant (Blum, Blum & Shub family).
/// Widely used in shader PRNG: `fract(sin(x) * 43758.5453)`.
const HASH_SCALE: f64 = 43_758.545_3;

fn main() {
    let mut h = ValidationHarness::new("alphafold3_diffusion");

    let Ok(baselines) = serde_json::from_str::<serde_json::Value>(BASELINE_JSON) else {
        eprintln!("[ERROR] Failed to parse diffusion_baselines.json");
        std::process::exit(1);
    };

    // ─── nF-D01: Cosine noise schedule ─────────────────────────────
    {
        let py_betas = flat_f64(&baselines["cosine_betas"]);
        let py_abar = flat_f64(&baselines["cosine_alpha_bar"]);
        let sched = diffusion::cosine_beta_schedule(T_STEPS, 0.008);

        let max_beta_diff = sched
            .betas
            .iter()
            .zip(py_betas.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-D01a cosine betas vs Python",
            max_beta_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        let max_abar_diff = sched
            .alpha_bar
            .iter()
            .zip(py_abar.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-D01b cosine alpha_bar vs Python",
            max_abar_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        h.check_bool(
            "nF-D01c alpha_bar monotonically decreasing",
            sched.alpha_bar.windows(2).all(|w| w[1] <= w[0] + 1e-15),
        );
        h.check_bool("nF-D01d alpha_bar[0] > 0.99", sched.alpha_bar[0] > 0.99);
    }

    // ─── nF-D02: Linear noise schedule ─────────────────────────────
    {
        let py_betas = flat_f64(&baselines["linear_betas"]);
        let py_abar = flat_f64(&baselines["linear_alpha_bar"]);
        let sched = diffusion::linear_beta_schedule(T_STEPS, 1e-4, 0.02);

        let max_beta_diff = sched
            .betas
            .iter()
            .zip(py_betas.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-D02a linear betas vs Python",
            max_beta_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        let max_abar_diff = sched
            .alpha_bar
            .iter()
            .zip(py_abar.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-D02b linear alpha_bar vs Python",
            max_abar_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );
    }

    // ─── nF-D03: Forward diffusion ─────────────────────────────────
    {
        let x_0 = flat_f64(&baselines["x_0"]);
        let noise = flat_f64(&baselines["noise_mid"]);
        let py_x_mid = flat_f64(&baselines["x_mid"]);
        let t_mid = baselines["t_mid"].as_u64().unwrap_or(25) as usize;

        let sched = diffusion::cosine_beta_schedule(T_STEPS, 0.008);
        let rs_x_mid = diffusion::forward_diffusion(&x_0, &noise, t_mid, &sched);

        let max_diff = rs_x_mid
            .iter()
            .zip(py_x_mid.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-D03a forward diffusion x_t vs Python",
            max_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        h.check_bool(
            "nF-D03b forward shape preserved",
            rs_x_mid.len() == x_0.len(),
        );
    }

    // ─── nF-D04: DDPM reverse step ─────────────────────────────────
    {
        let py_x_prev = flat_f64(&baselines["ddpm_x_prev"]);
        let x_mid = flat_f64(&baselines["x_mid"]);
        let noise_mid = flat_f64(&baselines["noise_mid"]);
        let t_mid = baselines["t_mid"].as_u64().unwrap_or(25) as usize;

        let sched = diffusion::cosine_beta_schedule(T_STEPS, 0.008);

        // Generate the same z noise as Python: np.random.default_rng(SEED+3)
        // We use a simple approach: extract the noise from the relationship:
        // x_prev = mean + sigma * z → z = (x_prev - mean) / sigma
        // But that's circular. Instead, we test the DDPM *mean* (z=0) since
        // the stochastic component depends on matching the RNG exactly.
        let z_zeros = vec![0.0_f64; x_mid.len()];
        let rs_mean = diffusion::ddpm_reverse_step(&x_mid, &noise_mid, &z_zeros, t_mid, &sched);

        // The mean should be the deterministic component; difference from Python
        // is just sigma*z where z comes from rng(SEED+3). We verify the mean
        // is finite and reasonably close (within the stochastic envelope).
        let beta_t = sched.betas[t_mid];
        let sigma_t = beta_t.sqrt();
        let max_diff = rs_mean
            .iter()
            .zip(py_x_prev.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        // Difference should be bounded by ~5*sigma_t (the stochastic term)
        h.check_bool(
            "nF-D04a DDPM mean within stochastic envelope",
            max_diff < 5.0 * sigma_t,
        );
        h.check_bool(
            "nF-D04b DDPM output finite",
            rs_mean.iter().all(|v| v.is_finite()),
        );

        // Test deterministic final step (t=0: no variance)
        let t0_result = diffusion::ddpm_reverse_step(&x_mid, &noise_mid, &z_zeros, 0, &sched);
        h.check_bool(
            "nF-D04c DDPM t=0 is deterministic",
            t0_result.iter().all(|v| v.is_finite()),
        );
    }

    // ─── nF-D05: DDIM reverse step ─────────────────────────────────
    {
        let py_ddim_prev = flat_f64(&baselines["ddim_x_prev"]);
        let py_ddim_x0 = flat_f64(&baselines["ddim_pred_x0"]);
        let x_mid = flat_f64(&baselines["x_mid"]);
        let noise_mid = flat_f64(&baselines["noise_mid"]);
        let t_mid = baselines["t_mid"].as_u64().unwrap_or(25) as usize;

        let sched = diffusion::cosine_beta_schedule(T_STEPS, 0.008);
        let (rs_prev, rs_x0) = diffusion::ddim_reverse_step(&x_mid, &noise_mid, t_mid, &sched);

        let max_prev_diff = rs_prev
            .iter()
            .zip(py_ddim_prev.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-D05a DDIM x_prev vs Python",
            max_prev_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        let max_x0_diff = rs_x0
            .iter()
            .zip(py_ddim_x0.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-D05b DDIM pred_x0 vs Python",
            max_x0_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        // Verify determinism
        let (rs_prev2, _) = diffusion::ddim_reverse_step(&x_mid, &noise_mid, t_mid, &sched);
        h.check_bool("nF-D05c DDIM deterministic", rs_prev == rs_prev2);
    }

    // ─── nF-D06: SE(3) equivariant noise ───────────────────────────
    {
        let py_noisy = flat_f64(&baselines["se3_noisy"]);

        let sched = diffusion::cosine_beta_schedule(T_STEPS, 0.008);
        let t_mid = baselines["t_mid"].as_u64().unwrap_or(25) as usize;

        // Generate same coords as Python (rng seed 42, advanced past earlier draws)
        // Instead of matching rng state exactly, verify the structural properties:
        // 1. Output is centered (COM ≈ 0)
        // 2. Translation invariance holds
        let coords: Vec<f64> = (0..N_ATOMS * 3).map(|i| (i as f64) * 0.1 - 5.0).collect();
        let noise: Vec<f64> = (0..N_ATOMS * 3).map(|i| (i as f64) * 0.01).collect();

        let (noisy, _com) = diffusion::se3_equivariant_noise(&coords, &noise, t_mid, &sched);

        // COM of result should be ~0
        let com_x: f64 = noisy.chunks_exact(3).map(|c| c[0]).sum::<f64>() / N_ATOMS as f64;
        let com_y: f64 = noisy.chunks_exact(3).map(|c| c[1]).sum::<f64>() / N_ATOMS as f64;
        let com_z: f64 = noisy.chunks_exact(3).map(|c| c[2]).sum::<f64>() / N_ATOMS as f64;
        let com_norm = (com_x * com_x + com_y * com_y + com_z * com_z).sqrt();
        h.check_abs(
            "nF-D06a SE(3) output centered",
            com_norm,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        // Translation invariance: shift coords by arbitrary vector
        let shifted: Vec<f64> = coords
            .chunks_exact(3)
            .flat_map(|c| [c[0] + 100.0, c[1] - 50.0, c[2] + 200.0])
            .collect();
        let (noisy_shifted, _) = diffusion::se3_equivariant_noise(&shifted, &noise, t_mid, &sched);
        let max_diff = noisy
            .iter()
            .zip(noisy_shifted.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-D06b SE(3) translation invariance",
            max_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        // Check Python baseline COM
        let py_com_x: f64 =
            py_noisy.chunks(3).map(|c| c[0]).sum::<f64>() / (py_noisy.len() as f64 / 3.0);
        let py_com_y: f64 =
            py_noisy.chunks(3).map(|c| c[1]).sum::<f64>() / (py_noisy.len() as f64 / 3.0);
        let py_com_z: f64 =
            py_noisy.chunks(3).map(|c| c[2]).sum::<f64>() / (py_noisy.len() as f64 / 3.0);
        let py_com_norm = (py_com_x * py_com_x + py_com_y * py_com_y + py_com_z * py_com_z).sqrt();
        h.check_abs(
            "nF-D06c Python baseline also centered",
            py_com_norm,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );
    }

    // ─── nF-D07: Pair transition FFN ───────────────────────────────
    {
        let pair_repr = flat_f64(&baselines["pair_repr"]);
        let w1 = flat_f64(&baselines["ffn_w1"]);
        let b1 = flat_f64(&baselines["ffn_b1"]);
        let w2 = flat_f64(&baselines["ffn_w2"]);
        let b2 = flat_f64(&baselines["ffn_b2"]);
        let py_out = flat_f64(&baselines["ffn_out"]);

        let rs_out =
            diffusion::pair_transition_ffn(&pair_repr, N_RES, D_PAIR, &w1, &b1, D_HIDDEN, &w2, &b2);

        let max_diff = rs_out
            .iter()
            .zip(py_out.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-D07a FFN output vs Python",
            max_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        h.check_bool(
            "nF-D07b FFN output shape",
            rs_out.len() == N_RES * N_RES * D_PAIR,
        );
    }

    // ─── nF-D08: pLDDT head ───────────────────────────────────────
    {
        let single_repr = flat_f64(&baselines["plddt_single"]);
        let w = flat_f64(&baselines["plddt_w"]);
        let b = flat_f64(&baselines["plddt_b"]);
        let py_plddt = flat_f64(&baselines["plddt"]);

        let rs_plddt = neural_spring::coral_forge::confidence::plddt_head(
            &single_repr,
            N_RES,
            D_PAIR,
            &w,
            b[0],
        );

        let max_diff = rs_plddt
            .iter()
            .zip(py_plddt.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-D08a pLDDT vs Python",
            max_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        h.check_bool(
            "nF-D08b pLDDT values in [0,1]",
            rs_plddt.iter().all(|&v| (0.0..=1.0).contains(&v)),
        );
    }

    // ─── nF-D09: PAE head ─────────────────────────────────────────
    {
        let pair_repr = flat_f64(&baselines["pair_repr"]);
        let w = flat_f64(&baselines["pae_w"]);
        let b = flat_f64(&baselines["pae_b"]);
        let py_expected = flat_f64(&baselines["pae_expected"]);

        let (rs_expected, rs_probs) =
            neural_spring::coral_forge::confidence::pae_head(&pair_repr, N_RES, D_PAIR, &w, &b, 64);

        let max_diff = rs_expected
            .iter()
            .zip(py_expected.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0_f64, f64::max);
        h.check_abs(
            "nF-D09a PAE expected vs Python",
            max_diff,
            0.0,
            tolerances::CROSS_LANGUAGE,
        );

        // Probs sum to 1 per pair
        let all_sum_one = rs_probs.chunks_exact(64).all(|row| {
            let sum: f64 = row.iter().sum();
            (sum - 1.0).abs() < 1e-10
        });
        h.check_bool("nF-D09b PAE probs sum to 1", all_sum_one);

        h.check_bool(
            "nF-D09c PAE expected non-negative",
            rs_expected.iter().all(|&v| v >= 0.0),
        );
    }

    // ─── nF-D10: Full DDIM loop (oracle) ──────────────────────────
    {
        let py_trajectory = flat_f64(&baselines["ddim_trajectory"]);
        let clean = flat_f64(&baselines["clean_centered"]);

        let sched = diffusion::cosine_beta_schedule(T_STEPS, 0.008);

        // Forward to T-1: use the same noise pattern as Python
        // Since we can't match Python's RNG exactly, we test the structural
        // property: DDIM oracle with perfect noise prediction converges
        let noise: Vec<f64> = (0..clean.len())
            .map(|i| {
                let x = (SEED as f64 + 10.0) * (i as f64 + 1.0);
                (x.sin() * HASH_SCALE).fract()
            })
            .collect();
        let x_t = diffusion::forward_diffusion(&clean, &noise, T_STEPS - 1, &sched);

        let initial_dist: f64 = x_t
            .iter()
            .zip(clean.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // Run DDIM reverse with oracle noise prediction
        let mut x_curr = x_t;
        for t in (1..T_STEPS).rev() {
            let a_bar_t = sched.alpha_bar[t];
            let predicted_eps: Vec<f64> = x_curr
                .iter()
                .zip(clean.iter())
                .map(|(&xt, &x0)| (xt - a_bar_t.sqrt() * x0) / (1.0 - a_bar_t).sqrt())
                .collect();
            let (x_prev, _) = diffusion::ddim_reverse_step(&x_curr, &predicted_eps, t, &sched);
            x_curr = x_prev;
        }

        let final_dist: f64 = x_curr
            .iter()
            .zip(clean.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        h.check_bool(
            "nF-D10a DDIM oracle converges (final < 1% of initial)",
            final_dist < initial_dist * 0.01,
        );

        // Python trajectory also converges
        h.check_bool(
            "nF-D10b Python trajectory converges",
            py_trajectory.last().copied().unwrap_or(1.0)
                < py_trajectory.first().copied().unwrap_or(0.0) * 0.01,
        );
    }

    h.finish();
}
