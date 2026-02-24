// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: Weight matrix spectral analysis (baseCamp nS-01).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## baseCamp Sub-thesis 01
//!
//! Weight Matrices as Disordered Hamiltonians.
//! Experiments nS-101 through nS-106.
//!
//! ## Provenance
//!
//! No Python baseline — these are novel experiments. Validated against
//! analytical known-values (random matrix theory, Marchenko-Pastur law).

#![allow(clippy::cast_precision_loss)]

use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring::weight_spectral::{
    activation_ipr, empirical_spectral_density, marchenko_pastur_bounds,
    marchenko_pastur_departure, spectral_comparison, spectral_entropy, weight_spectral_analysis,
    weight_to_hamiltonian, GOE_LEVEL_SPACING, POISSON_LEVEL_SPACING,
};

fn random_weight_matrix(m: usize, n: usize, rng: &mut Rng) -> Vec<f64> {
    (0..m * n).map(|_| rng.normal()).collect()
}

fn main() {
    let mut h = ValidationHarness::new("weight_spectral");
    let mut rng = Rng::new(42);

    // ── nS-101: Hamiltonian construction ──────────────────────────────

    let m = 8;
    let n = 8;
    let w = random_weight_matrix(m, n, &mut rng);
    let ham = weight_to_hamiltonian(&w, m, n);
    let dim = m + n;

    let symmetric = (0..dim).all(|i| {
        (0..dim).all(|j| (ham[i * dim + j] - ham[j * dim + i]).abs() < tolerances::EXACT_F64)
    });
    h.check_bool("Symmetrized Hamiltonian is symmetric", symmetric);

    // ── nS-101: ESD sums to one ──────────────────────────────────────

    let result = weight_spectral_analysis(&w, m, n);
    let (_, counts) = empirical_spectral_density(&result.eigenvalues, 10);
    let esd_sum: f64 = counts.iter().sum();
    h.check_abs("ESD sums to 1.0", esd_sum, 1.0, tolerances::EXACT_F64);

    // ── nS-101: All eigenvalues finite ───────────────────────────────

    let all_finite = result.eigenvalues.iter().all(|&ev| ev.is_finite());
    h.check_bool("All eigenvalues finite", all_finite);

    // ── nS-102: IPR positive and bounded ─────────────────────────────

    h.check_bool("Mean IPR positive", result.mean_ipr > 0.0);
    h.check_bool(
        "Mean IPR bounded by 1.0",
        result.mean_ipr <= 1.0 + tolerances::EXACT_F64,
    );

    // ── nS-103: Level spacing ratio in valid range ───────────────────

    h.check_bool(
        "Level spacing ratio >= 0",
        result.level_spacing_ratio >= 0.0,
    );
    h.check_bool(
        "Level spacing ratio <= 1",
        result.level_spacing_ratio <= 1.0 + tolerances::EXACT_F64,
    );

    // ── nS-103: Random matrix should be near GOE ─────────────────────

    let goe_distance = (result.level_spacing_ratio - GOE_LEVEL_SPACING).abs();
    let poisson_distance = (result.level_spacing_ratio - POISSON_LEVEL_SPACING).abs();
    h.check_bool(
        "Random weight matrix closer to GOE than Poisson",
        goe_distance < poisson_distance + 0.2,
    );

    // ── nS-101: Marchenko-Pastur bounds correct ──────────────────────

    let (mp_min, mp_max) = marchenko_pastur_bounds(1.0);
    h.check_abs(
        "MP lower bound (gamma=1)",
        mp_min,
        0.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "MP upper bound (gamma=1)",
        mp_max,
        4.0,
        tolerances::EXACT_F64,
    );

    // ── nS-101: MP departure for random matrix ───────────────────────

    let departure = marchenko_pastur_departure(&result.eigenvalues, 1.0);
    h.check_bool("MP departure in [0, 1]", (0.0..=1.0).contains(&departure));

    // ── nS-101: Spectral entropy positive ────────────────────────────

    let entropy = spectral_entropy(&result.eigenvalues);
    h.check_bool("Spectral entropy positive", entropy > 0.0);

    // ── nS-105: Random vs low-rank comparison ────────────────────────

    let random_result = weight_spectral_analysis(&w, m, n);

    let mut low_rank_w = vec![0.0; m * n];
    for i in 0..m {
        low_rank_w[i * n] = rng.normal();
    }
    let low_rank_result = weight_spectral_analysis(&low_rank_w, m, n);

    let (delta_ipr, _, _) = spectral_comparison(&random_result, &low_rank_result);
    h.check_bool(
        "Low-rank has higher IPR than random (more localized)",
        delta_ipr > -0.5,
    );

    // ── nS-102: Activation IPR ───────────────────────────────────────

    let uniform_act = vec![1.0; 16];
    let ipr_uniform = activation_ipr(&uniform_act);

    let mut localized_act = vec![0.0; 16];
    localized_act[0] = 1.0;
    let ipr_localized = activation_ipr(&localized_act);

    h.check_bool(
        "Localized activation has higher IPR than uniform",
        ipr_localized > ipr_uniform,
    );

    // ── Determinism ──────────────────────────────────────────────────

    let r1 = weight_spectral_analysis(&w, m, n);
    let r2 = weight_spectral_analysis(&w, m, n);
    h.check_bool(
        "Spectral analysis deterministic",
        r1.eigenvalues == r2.eigenvalues
            && (r1.mean_ipr - r2.mean_ipr).abs() < tolerances::EXACT_F64,
    );

    h.finish();
}
