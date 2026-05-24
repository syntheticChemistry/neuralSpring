// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: Exp-053 — Anderson Localization in Multi-Agent
//! AI Coordination.
//!
//! Paper C: Anderson localization predicts phase transitions in
//! decentralized multi-agent coordination. Validates size-independence
//! of the localization transition, dimensional topology effects, and
//! disorder-driven IPR/LSR changes against Python baselines.
//!
//! ## Provenance
//!
//! - **Python baseline**: `control/anderson_multiagent/anderson_multiagent.py`
//! - **Baseline values**: `control/anderson_multiagent/baseline_values.json`
//! - **All data**: synthetic, deterministic seed 42
//! - **Commit**: `BASELINE_COMMIT` (`f9ad0268`)
//! - **Date**: 2026-02-26
//! - **Command**: `python3 control/anderson_multiagent/anderson_multiagent.py`
//! - **Environment**: Python 3.12, `NumPy`, seed=42
//! - **Hardware**: southGate (Ryzen 7 5800X3D, 128GB DDR4, Pop!_OS 22.04)
//! - **Provenance record**: `provenance::ANDERSON_MULTIAGENT_PROVENANCE`

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use neural_spring::agent_coordination::{coordination_spectral_analysis, generate_lattice_agents};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("anderson_multiagent (Exp-053)");

    // ── 1. Disorder sweep: IPR increases with disorder ──────────────

    let r_clean = spectral_at(4, 3, 0.0, 2.5);
    let r_dirty = spectral_at(4, 3, 8.0, 2.5);

    h.check_bool(
        "IPR increases with disorder (W=0 → W=8)",
        r_dirty.mean_ipr > r_clean.mean_ipr,
    );

    // Rust and Python use different RNG (Xoshiro256** vs PCG64), so
    // IPR values won't match exactly — validate qualitative behavior.
    h.check_bool(
        "IPR(W=0) in physically reasonable range [0.01, 0.3]",
        (0.01..=0.3).contains(&r_clean.mean_ipr),
    );

    h.check_bool(
        "IPR(W=8) in physically reasonable range [0.1, 0.5]",
        (0.1..=0.5).contains(&r_dirty.mean_ipr),
    );

    // ── 2. IPR ratio size-independence ──────────────────────────────

    let sizes: &[(usize, usize)] = &[(4, 64), (5, 125), (6, 216), (8, 512)];
    let mut ipr_ratios = Vec::with_capacity(sizes.len());

    for &(n_side, n_agents) in sizes {
        let r_low = spectral_at(n_side, 3, 0.1, 2.5);
        let r_high = spectral_at(n_side, 3, 4.0, 2.5);
        let ratio = r_high.mean_ipr / r_low.mean_ipr.max(1e-15);
        ipr_ratios.push(ratio);

        h.check_bool(
            &format!("N={n_agents}: disorder localizes (ratio > 1)"),
            ratio > 1.0,
        );
    }

    let ratio_mean: f64 = ipr_ratios.iter().sum::<f64>() / ipr_ratios.len() as f64;
    let ratio_max = ipr_ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ratio_min = ipr_ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let spread = (ratio_max - ratio_min) / ratio_mean;

    h.check_bool(
        "IPR ratio size-independent (spread < 40%)",
        spread < tolerances::IPR_RATIO_SPREAD_MAX,
    );

    // ── 3. Dimensional topology: 3D more connected than 1D ─────────

    let r_1d = spectral_at_dim(64, 1, 0.1, 2.5);
    let r_3d = spectral_at_dim(4, 3, 0.1, 2.5);

    h.check_bool(
        "3D algebraic connectivity > 1D algebraic connectivity",
        r_3d.algebraic_connectivity > r_1d.algebraic_connectivity,
    );

    h.check_bool(
        "1D algebraic connectivity positive",
        r_1d.algebraic_connectivity > 0.0,
    );

    h.check_bool(
        "3D algebraic connectivity > 1.0",
        r_3d.algebraic_connectivity > 1.0,
    );

    // ── 4. Disorder disrupts all dimensions ─────────────────────────

    for (dim, n_side) in [(1, 64), (2, 8), (3, 4)] {
        let r_low = spectral_at_dim(n_side, dim, 0.1, 2.5);
        let r_high = spectral_at_dim(n_side, dim, 8.0, 2.5);

        h.check_bool(
            &format!("dim={dim}: disorder increases IPR"),
            r_high.mean_ipr > r_low.mean_ipr,
        );
    }

    // ── 5. Normalized IPR scaling ───────────────────────────────────

    let n_agents = 64u32;
    let nipr_ext = r_clean.mean_ipr * f64::from(n_agents);
    let nipr_loc = spectral_at(4, 3, 10.0, 2.5).mean_ipr * f64::from(n_agents);

    h.check_bool(
        "Localized N*IPR > extended N*IPR (by 1.5×)",
        nipr_loc > nipr_ext * 1.5,
    );

    // ── 6. Determinism ──────────────────────────────────────────────

    let r_a = spectral_at(4, 3, 2.0, 2.5);
    let r_b = spectral_at(4, 3, 2.0, 2.5);

    h.check_abs(
        "Deterministic IPR",
        r_a.mean_ipr,
        r_b.mean_ipr,
        tolerances::EXACT_F64,
    );

    h.check_abs(
        "Deterministic LSR",
        r_a.level_spacing_ratio,
        r_b.level_spacing_ratio,
        tolerances::EXACT_F64,
    );

    // ── 7. Cross-validate qualitative behavior ────────────────────

    h.check_bool(
        "Deterministic IPR at W=2 in expected range [0.05, 0.3]",
        (0.05..=0.3).contains(&r_a.mean_ipr),
    );

    h.finish();
}

/// Spectral analysis on a 3D lattice with given `n_per_side`.
fn spectral_at(
    n_per_side: usize,
    dim: usize,
    disorder: f64,
    comm_range: f64,
) -> neural_spring::agent_coordination::CoordinationResult {
    let mut rng = Rng::new(42);
    let agents = generate_lattice_agents(n_per_side, dim, 0.3, &mut rng);
    coordination_spectral_analysis(&agents, comm_range, disorder)
}

/// Spectral analysis with explicit dim and `n_per_side`.
fn spectral_at_dim(
    n_per_side: usize,
    dim: usize,
    disorder: f64,
    comm_range: f64,
) -> neural_spring::agent_coordination::CoordinationResult {
    spectral_at(n_per_side, dim, disorder, comm_range)
}
