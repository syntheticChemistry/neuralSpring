// SPDX-License-Identifier: AGPL-3.0-or-later

//! nS-604: Three-compartment tissue lattice and Anderson Hamiltonian construction.
//!
//! Builds 1D Anderson lattice Hamiltonians for the skin tissue stack and
//! computes level spacing ratio spectral sweeps across barrier states.

use super::{dimensional_promotion, evenness_to_disorder, pielou_evenness};
use crate::tolerances;

/// Per-compartment disorder analysis.
///
/// McCandless (2014) G6: IL-31 targets immune, skin, and neural cells.
/// Each compartment has its own cell-type distribution and thus its own
/// Pielou evenness / disorder W value.
#[derive(Debug, Clone)]
pub struct ThreeCompartmentDisorder {
    pub immune_w: f64,
    pub skin_w: f64,
    pub neural_w: f64,
    pub cross_compartment_variance: f64,
}

/// Compute three-compartment disorder from cell-type fractions.
///
/// Higher variance across compartments = more heterogeneous tissue =
/// cytokines face different propagation regimes in each compartment.
#[must_use]
pub fn three_compartment_disorder(
    immune_fracs: &[f64],
    skin_fracs: &[f64],
    neural_fracs: &[f64],
    w_scale: f64,
) -> ThreeCompartmentDisorder {
    let w_immune = evenness_to_disorder(pielou_evenness(immune_fracs), w_scale);
    let w_skin = evenness_to_disorder(pielou_evenness(skin_fracs), w_scale);
    let w_neural = evenness_to_disorder(pielou_evenness(neural_fracs), w_scale);
    let mean_w = (w_immune + w_skin + w_neural) / 3.0;
    let di = w_immune - mean_w;
    let ds = w_skin - mean_w;
    let dn = w_neural - mean_w;
    let variance = di.mul_add(di, ds.mul_add(ds, dn * dn)) / 3.0;
    ThreeCompartmentDisorder {
        immune_w: w_immune,
        skin_w: w_skin,
        neural_w: w_neural,
        cross_compartment_variance: variance,
    }
}

/// Build a multi-layer 1D Anderson Hamiltonian for the skin tissue stack.
///
/// Each layer contributes sites with on-site energies sampled from a
/// distribution whose width is set by the layer's effective dimension
/// and the cell-type disorder W. Returns a flat symmetric matrix.
#[must_use]
pub fn tissue_lattice_hamiltonian(
    layer_sizes: &[usize],
    layer_disorders: &[f64],
    hopping: f64,
    seed: u64,
) -> Vec<f64> {
    let n: usize = layer_sizes.iter().sum();
    let mut h = vec![0.0; n * n];
    let mut rng = crate::rng::Rng::new(seed);

    let mut site = 0;
    for (layer_idx, &layer_n) in layer_sizes.iter().enumerate() {
        let w = layer_disorders[layer_idx.min(layer_disorders.len() - 1)];
        for _ in 0..layer_n {
            h[site * n + site] = w * rng.normal();
            site += 1;
        }
    }

    for i in 0..(n - 1) {
        h[i * n + (i + 1)] = hopping;
        h[(i + 1) * n + i] = hopping;
    }
    h
}

/// Barrier promotion spectral sweep: eigenvalues across barrier states.
///
/// Sweeps intact_fraction from 1.0 (intact) to 0.0 (fully breached),
/// building a lattice at each state and computing the level spacing ratio r.
#[must_use]
pub fn barrier_promotion_spectrum(
    n_sites: usize,
    n_steps: usize,
    base_disorder: f64,
    hopping: f64,
) -> Vec<(f64, f64, f64)> {
    let mut results = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        #[expect(
            clippy::cast_precision_loss,
            reason = "simulation step index → f64 for barrier breach fraction"
        )]
        let intact = 1.0 - step as f64 / (n_steps - 1).max(1) as f64;
        let d_eff = dimensional_promotion(intact, 2.0, 3.0);
        let w_eff = base_disorder * (3.0 - d_eff + 1.0);

        let ham = tissue_lattice_hamiltonian(&[n_sites], &[w_eff], hopping, 42 + step as u64);
        let decomp = crate::eigh::eigh_householder_qr(&ham, n_sites);
        let mut evals = decomp.eigenvalues;
        evals.sort_by(f64::total_cmp);

        let r = level_spacing_ratio(&evals);
        results.push((intact, d_eff, r));
    }
    results
}

/// Level spacing ratio (mean) from sorted eigenvalues.
///
/// `r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})` where `s_n = E_{n+1} - E_n`.
/// GOE: r ≈ 0.5307, Poisson: r ≈ 0.386.
#[must_use]
pub fn level_spacing_ratio(sorted_eigenvalues: &[f64]) -> f64 {
    if sorted_eigenvalues.len() < 3 {
        return 0.0;
    }
    let spacings: Vec<f64> = sorted_eigenvalues
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .collect();
    let ratios: Vec<f64> = spacings
        .windows(2)
        .filter_map(|w| {
            let (a, b) = (w[0], w[1]);
            if a > tolerances::NUMERICAL_DISTINCTNESS || b > tolerances::NUMERICAL_DISTINCTNESS {
                Some(a.min(b) / a.max(b))
            } else {
                None
            }
        })
        .collect();
    if ratios.is_empty() {
        return 0.0;
    }
    #[expect(clippy::cast_precision_loss, reason = "ratio count → f64 for mean")]
    let n = ratios.len() as f64;
    ratios.iter().sum::<f64>() / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_three_compartment_disorder() {
        let immune = [0.25, 0.25, 0.25, 0.25];
        let skin = [0.80, 0.10, 0.05, 0.05];
        let neural = [0.50, 0.50];
        let result = three_compartment_disorder(&immune, &skin, &neural, 10.0);
        assert!(
            result.immune_w > result.skin_w,
            "even immune > dominated skin"
        );
        assert!(result.cross_compartment_variance > 0.0);
    }

    #[test]
    fn test_tissue_lattice_hamiltonian_symmetric() {
        let h = tissue_lattice_hamiltonian(&[4, 4], &[1.0, 2.0], 1.0, 42);
        let n = 8;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (h[i * n + j] - h[j * n + i]).abs() < 1e-15,
                    "Hamiltonian must be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_level_spacing_ratio_bounds() {
        let evals: Vec<f64> = (0..20).map(|i| f64::from(i) * 1.0).collect();
        let r = level_spacing_ratio(&evals);
        assert!(r > 0.0 && r <= 1.0, "r must be in (0,1], got {r}");
    }

    #[test]
    fn test_barrier_promotion_spectrum() {
        let results = barrier_promotion_spectrum(16, 5, 1.0, 1.0);
        assert_eq!(results.len(), 5);
        assert!((results[0].0 - 1.0).abs() < 1e-10, "first step = intact");
        assert!((results[4].0).abs() < 1e-10, "last step = fully breached");
        for &(_, d_eff, r) in &results {
            assert!((2.0..=3.0).contains(&d_eff));
            assert!((0.0..=1.0).contains(&r), "r out of bounds: {r}");
        }
    }
}
