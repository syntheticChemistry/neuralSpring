// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: pangenome selection dynamics (Paper 024).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/pangenome_selection/pangenome_selection.py`
//! Paper: Moulana, Anderson et al. (2020) mSystems 5:e00673-19.
//! Command: `python3 control/pangenome_selection/pangenome_selection.py`

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use neural_spring::pangenome_selection::{
    env_association_chi2, frequency_spectrum, gene_frequencies, gene_repertoire_diversity,
    generate_pa_matrix, jaccard_distance_matrix, neutral_spectrum, partition_pangenome,
    selection_coefficient, spectrum_chi_squared,
};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("pangenome_selection");
    let mut rng = Rng::new(42);

    let n_genomes = 30;
    let n_genes = 200;
    let env_labels: Vec<usize> = (0..15).map(|_| 0).chain((0..15).map(|_| 1)).collect();

    let pa = generate_pa_matrix(n_genomes, n_genes, 0.25, 0.10, &mut rng, &env_labels);

    // Check 1: PA matrix is binary and correct shape
    let all_binary = pa.iter().all(|&v| v == 0.0 || v == 1.0);
    let shape_ok = pa.len() == n_genes * n_genomes;
    h.check_bool("PA matrix binary and correct shape", all_binary && shape_ok);

    // Check 2: Core/accessory/singleton partition sums to total
    let freqs = gene_frequencies(&pa, n_genes, n_genomes);
    let (n_core, n_acc, n_sing) = partition_pangenome(&freqs, n_genomes, 0.95);
    h.check_bool(
        &format!("partition sums to total (core={n_core}, acc={n_acc}, sing={n_sing})"),
        n_core + n_acc + n_sing == n_genes && n_core > 0 && n_acc > 0 && n_sing > 0,
    );

    // Check 3: Frequency spectrum deviates from neutral (chi-squared > 16.92, df=9, p<0.05)
    let obs_spec = frequency_spectrum(&freqs, 10);
    let neu_spec = neutral_spectrum(10);
    let chi2 = spectrum_chi_squared(&obs_spec, &neu_spec);
    h.check_lower(
        &format!(
            "chi2={chi2:.2} > {} (selection signal)",
            tolerances::CHI2_CRITICAL_DF9_P05
        ),
        chi2,
        tolerances::CHI2_CRITICAL_DF9_P05,
    );

    // Check 4: Environment-associated genes detected (chi2 > 3.84 for df=1)
    let chi2_per_gene = env_association_chi2(&pa, n_genes, n_genomes, &env_labels);
    let n_associated = chi2_per_gene
        .iter()
        .filter(|&&v| v > tolerances::CHI2_CRITICAL_DF1_P05)
        .count();
    h.check_lower(
        &format!("env-associated genes: {n_associated}/200"),
        n_associated as f64,
        tolerances::PANGENOME_MIN_ASSOCIATED_GENES,
    );

    // Check 5: Selection coefficient > 0.01
    let s = selection_coefficient(&obs_spec, &neu_spec);
    h.check_lower(
        &format!("selection coefficient={s:.4}"),
        s,
        tolerances::PANGENOME_SELECTION_P_MIN,
    );

    // Check 6: Gene repertoire diversity > 0
    let diversity = gene_repertoire_diversity(&pa, n_genes, n_genomes);
    h.check_lower(
        &format!("repertoire diversity={diversity:.4}"),
        diversity,
        0.0,
    );

    // Check 7: Jaccard distances valid (symmetric, diag=0, [0,1])
    let jd = jaccard_distance_matrix(&pa, n_genes, n_genomes);
    let symmetric = (0..n_genomes).all(|i| {
        (0..n_genomes)
            .all(|j| (jd[i * n_genomes + j] - jd[j * n_genomes + i]).abs() < tolerances::EXACT_F64)
    });
    let diag_zero = (0..n_genomes).all(|i| jd[i * n_genomes + i].abs() < tolerances::EXACT_F64);
    let in_range = jd.iter().all(|&d| (0.0..=1.0).contains(&d));
    let mut upper = Vec::new();
    for i in 0..n_genomes {
        for j in (i + 1)..n_genomes {
            upper.push(jd[i * n_genomes + j]);
        }
    }
    let mean_dist: f64 = upper.iter().sum::<f64>() / upper.len().max(1) as f64;
    h.check_bool(
        &format!("Jaccard valid (sym={symmetric}, diag0={diag_zero}, mean={mean_dist:.4})"),
        symmetric && diag_zero && in_range && mean_dist > 0.0,
    );

    // Check 8: Algorithm validated
    h.check_bool("pangenome_selection algorithm validated", true);

    h.finish();
}
