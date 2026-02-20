// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: Anderson localization (Paper 023).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/anderson_localization/anderson_localization.py`
//! Paper: Bourgain & Kachkovskiy (2018) GAFA 29:3-43.
//! Command: `python3 control/anderson_localization/anderson_localization.py`
//! Result: 8/8 PASS (seed=42, N=64, Aubry-André model)

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

use neural_spring::anderson_localization::{
    anderson_hamiltonian_random, aubry_andre_hamiltonian, disorder_sweep, jacobi_eigh, mean_ipr,
    two_particle_hamiltonian, GOLDEN_RATIO,
};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn is_symmetric(h: &[Vec<f64>]) -> bool {
    let n = h.len();
    for i in 0..n {
        for j in (i + 1)..n {
            if (h[i][j] - h[j][i]).abs() > 1e-12 {
                return false;
            }
        }
    }
    true
}

fn main() {
    let mut h = ValidationHarness::new("anderson_localization");
    let mut rng = Rng::new(42);

    let n = 32;
    let t = 1.0;

    // Hamiltonian is Hermitian (symmetric)
    let h_rand = anderson_hamiltonian_random(n, t, 1.0, &mut rng);
    h.check_bool("Anderson H is symmetric", is_symmetric(&h_rand));

    // Eigenvalues real
    let (eigvals, _) = jacobi_eigh(&h_rand);
    let all_real = eigvals.iter().all(|&x| x.is_finite() && !x.is_nan());
    h.check_bool("All eigenvalues real and finite", all_real);

    // Weak disorder → extended (low IPR)
    rng = Rng::new(42);
    let h_weak = anderson_hamiltonian_random(n, t, 0.5, &mut rng);
    let (_, ev_weak) = jacobi_eigh(&h_weak);
    let ipr_weak = mean_ipr(&ev_weak);

    // Strong disorder → localized (high IPR)
    rng = Rng::new(42);
    let h_strong = anderson_hamiltonian_random(n, t, 8.0, &mut rng);
    let (_, ev_strong) = jacobi_eigh(&h_strong);
    let ipr_strong = mean_ipr(&ev_strong);
    h.check_lower("Strong disorder: localized (IPR > 0.05)", ipr_strong, 0.05);
    h.check_bool(
        "Extended (weak) has lower IPR than localized (strong)",
        ipr_weak < ipr_strong,
    );

    // IPR trend: stronger disorder gives higher mean IPR
    rng = Rng::new(42);
    let w_vals = [0.5, 1.0, 2.0, 4.0];
    let ipr_vals = disorder_sweep(n, t, &w_vals, &mut rng);
    let trend = ipr_vals.len() >= 2 && ipr_vals[ipr_vals.len() - 1] > ipr_vals[0];
    h.check_bool("IPR trend: strong disorder (W=4) > weak (W=0.5)", trend);

    // Aubry-André transition near W_c = 2
    let alpha = 1.0 / GOLDEN_RATIO;
    let h_below = aubry_andre_hamiltonian(n, t, 1.5, alpha, 0.0);
    let h_above = aubry_andre_hamiltonian(n, t, 3.0, alpha, 0.0);
    let (_, ev_below) = jacobi_eigh(&h_below);
    let (_, ev_above) = jacobi_eigh(&h_above);
    let ipr_below = mean_ipr(&ev_below);
    let ipr_above = mean_ipr(&ev_above);
    h.check_bool(
        "Aubry-André: W<W_c has lower IPR than W>W_c",
        ipr_below < ipr_above,
    );

    // Two-particle: finite, normalized
    let n2 = 6;
    let h2 = two_particle_hamiltonian(n2, t, 2.0, 0.5, alpha);
    let (eig2, ev2) = jacobi_eigh(&h2);
    let all_finite = eig2.iter().all(|&x| x.is_finite())
        && ev2.iter().all(|row| row.iter().all(|&x| x.is_finite()));
    h.check_bool("Two-particle: all finite", all_finite);

    let dim = n2 * n2;
    let norms_ok = (0..dim).all(|k| {
        let norm: f64 = ev2.iter().map(|row| row[k] * row[k]).sum();
        (norm - 1.0).abs() < tolerances::CROSS_LANGUAGE
    });
    h.check_bool("Two-particle: eigenvectors normalized", norms_ok);

    h.finish();
}
