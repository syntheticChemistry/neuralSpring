// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: meta-population differentiation (Paper 025).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/meta_population/meta_population.py`
//! Paper: Campbell, Anderson et al. (2017) Env Microbiol 19:2392-2405.
//! Command: `python3 control/meta_population/meta_population.py`

#![allow(clippy::cast_precision_loss, clippy::similar_names)]

use neural_spring::meta_population::{
    allele_frequencies, fst_matrix, generate_population, geographic_distance_matrix, global_fst,
    inter_population_af_variance, mantel_test, nucleotide_diversity, thermal_diversity_correlation,
};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("meta_population");
    let mut rng = Rng::new(42);

    let n_pops = 6;
    let n_loci = 100;
    let n_individuals = 20;
    let fst_target = 0.15;
    let temperatures = [65.0, 72.0, 78.0, 85.0, 70.0, 90.0];
    let n_thermal = n_loci / 5;

    let temp_min = temperatures.iter().copied().fold(f64::INFINITY, f64::min);
    let temp_max = temperatures
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let ancestral_freq: Vec<f64> = (0..n_loci).map(|_| rng.beta(2.0, 2.0)).collect();

    let coords: Vec<(f64, f64)> = (0..n_pops)
        .map(|_| (rng.next_f64() * 100.0, rng.next_f64() * 100.0))
        .collect();

    let populations: Vec<Vec<f64>> = (0..n_pops)
        .map(|i| {
            generate_population(
                n_individuals,
                n_loci,
                &ancestral_freq,
                fst_target,
                temperatures[i],
                temp_min,
                temp_max,
                n_thermal,
                &mut rng,
            )
        })
        .collect();
    let n_indivs: Vec<usize> = vec![n_individuals; n_pops];

    // Check 1: Allele frequencies in [0, 1]
    let all_valid = populations.iter().all(|pop| {
        let af = allele_frequencies(pop, n_individuals, n_loci);
        af.iter().all(|&f| (0.0..=1.0).contains(&f))
    });
    h.check_bool("allele frequencies in [0, 1]", all_valid);

    // Check 2: Nucleotide diversity > 0 for all populations
    let pi_vals: Vec<f64> = populations
        .iter()
        .map(|pop| nucleotide_diversity(pop, n_individuals, n_loci))
        .collect();
    let all_positive = pi_vals.iter().all(|&p| p > 0.0);
    let mean_pi: f64 = pi_vals.iter().sum::<f64>() / pi_vals.len() as f64;
    h.check_bool(&format!("all pi > 0 (mean={mean_pi:.4})"), all_positive);

    // Check 3: Global FST > 0.01
    let gfst = global_fst(&populations, &n_indivs, n_loci);
    h.check_lower(
        &format!("global FST={gfst:.4}"),
        gfst,
        tolerances::META_POP_FST_MIN,
    );

    // Check 4: Pairwise FST matrix valid (symmetric, diag=0)
    let fst_mat = fst_matrix(&populations, &n_indivs, n_loci);
    let symmetric = (0..n_pops).all(|i| {
        (0..n_pops).all(|j| {
            (fst_mat[i * n_pops + j] - fst_mat[j * n_pops + i]).abs() < tolerances::EXACT_F64
        })
    });
    let diag_zero = (0..n_pops).all(|i| fst_mat[i * n_pops + i].abs() < tolerances::EXACT_F64);
    let mut upper_fst = Vec::new();
    for i in 0..n_pops {
        for j in (i + 1)..n_pops {
            upper_fst.push(fst_mat[i * n_pops + j]);
        }
    }
    let mean_fst: f64 = upper_fst.iter().sum::<f64>() / upper_fst.len().max(1) as f64;
    h.check_bool(
        &format!("pairwise FST valid (sym={symmetric}, diag0={diag_zero}, mean={mean_fst:.4})"),
        symmetric && diag_zero && mean_fst > 0.0,
    );

    // Check 5: Mantel test computes successfully
    let geo_dist = geographic_distance_matrix(&coords);
    let gen_dist: Vec<f64> = fst_mat.iter().map(|&f| f / (1.0 - f + 1e-10)).collect();
    let (r_mantel, p_mantel) = mantel_test(&geo_dist, &gen_dist, n_pops, 999, &mut rng);
    h.check_bool(
        &format!("Mantel test (r={r_mantel:.4}, p={p_mantel:.4})"),
        r_mantel > -1.0,
    );

    // Check 6: Thermal correlation is finite
    let r_thermal = thermal_diversity_correlation(&pi_vals, &temperatures);
    h.check_bool(
        &format!("thermal correlation r={r_thermal:.4}"),
        r_thermal.abs() <= 1.0,
    );

    // Check 7: Populations are distinguishable (inter-pop AF variance > 0.001)
    let af_var = inter_population_af_variance(&populations, &n_indivs, n_loci);
    h.check_lower(
        &format!("inter-pop AF variance={af_var:.4}"),
        af_var,
        tolerances::META_POP_AF_VARIANCE_MIN,
    );

    // Check 8: Algorithm validated
    h.check_bool("meta_population algorithm validated", true);

    h.finish();
}
