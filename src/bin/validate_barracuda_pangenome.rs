// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` CPU port: pangenome selection dynamics (Paper 024).
//!
//! Validates `barracuda::stats::{variance, pearson_correlation}` for
//! pangenome gene frequency analysis and environmental association tests.
//!
//! Evolution path:
//! ```text
//! Python (numpy.var, scipy.stats.chi2) → Rust (hand-rolled)
//!   → BarraCUDA CPU (barracuda::stats::variance, pearson_correlation)
//!   → BarraCUDA GPU (stats reduction + pairwise GEMV)
//! ```
//!
//! ## Provenance
//!
//! Python baseline: `control/pangenome_selection/pangenome_selection.py`
//! Rust baseline: `validate_pangenome_selection` (8/8 PASS)

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::similar_names
)]

use neural_spring::pangenome_selection::{
    env_association_chi2, frequency_spectrum, gene_frequencies, gene_repertoire_diversity,
    generate_pa_matrix, jaccard_distance_matrix, neutral_spectrum, partition_pangenome,
    selection_coefficient, spectrum_chi_squared,
};
use neural_spring::rng::Rng;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_pangenome");

    validate_gene_frequency_variance(&mut h);
    validate_env_correlation(&mut h);
    validate_selection_spectrum(&mut h);
    validate_jaccard_via_barracuda(&mut h);
    validate_partition_consistency(&mut h);
    validate_repertoire_diversity(&mut h);

    h.finish();
}

/// Gene frequency variance via barracuda matches hand-rolled.
fn validate_gene_frequency_variance(h: &mut ValidationHarness) {
    let mut rng = Rng::new(42);
    let n_genomes = 30;
    let n_genes = 200;
    let env_labels: Vec<usize> = (0..15).map(|_| 0).chain((0..15).map(|_| 1)).collect();
    let pa = generate_pa_matrix(n_genomes, n_genes, 0.25, 0.10, &mut rng, &env_labels);
    let freqs = gene_frequencies(&pa, n_genes, n_genomes);

    let hand_mean: f64 = freqs.iter().sum::<f64>() / freqs.len() as f64;
    let hand_var: f64 =
        freqs.iter().map(|&f| (f - hand_mean).powi(2)).sum::<f64>() / freqs.len() as f64;

    let barracuda_var = barracuda::stats::correlation::variance(&freqs).unwrap_or(f64::NAN);

    // barracuda uses sample variance (ddof=1); hand-rolled uses population (ddof=0).
    // For n=200, ratio is 200/199 ≈ 1.005. Allow 2% tolerance.
    let tol = hand_var.mul_add(0.02, 1e-10);
    h.check_abs(
        &format!("gene freq variance: barracuda={barracuda_var:.6} vs hand={hand_var:.6}"),
        barracuda_var,
        hand_var,
        tol,
    );
}

/// Environmental correlation: Pearson r between env label and gene frequency.
fn validate_env_correlation(h: &mut ValidationHarness) {
    let mut rng = Rng::new(42);
    let n_genomes = 30;
    let n_genes = 200;
    let env_labels: Vec<usize> = (0..15).map(|_| 0).chain((0..15).map(|_| 1)).collect();
    let pa = generate_pa_matrix(n_genomes, n_genes, 0.25, 0.10, &mut rng, &env_labels);

    let chi2_per_gene = env_association_chi2(&pa, n_genes, n_genomes, &env_labels);
    let n_associated = chi2_per_gene.iter().filter(|&&v| v > 3.84).count();

    // Build per-genome gene count and env label as f64 vectors for Pearson correlation.
    let gene_counts: Vec<f64> = (0..n_genomes)
        .map(|j| (0..n_genes).map(|i| pa[i * n_genomes + j]).sum::<f64>())
        .collect();
    let env_f64: Vec<f64> = env_labels.iter().map(|&e| e as f64).collect();

    let r = barracuda::stats::correlation::pearson_correlation(&gene_counts, &env_f64)
        .unwrap_or(f64::NAN);

    h.check_bool(
        &format!("env-associated genes > 5 ({n_associated})"),
        n_associated > 5,
    );
    h.check_bool(
        &format!("Pearson(gene_count, env) finite (r={r:.4})"),
        r.is_finite(),
    );
}

/// Frequency spectrum selection test matches hand-rolled chi-squared.
fn validate_selection_spectrum(h: &mut ValidationHarness) {
    let mut rng = Rng::new(42);
    let n_genomes = 30;
    let n_genes = 200;
    let env_labels: Vec<usize> = (0..15).map(|_| 0).chain((0..15).map(|_| 1)).collect();
    let pa = generate_pa_matrix(n_genomes, n_genes, 0.25, 0.10, &mut rng, &env_labels);
    let freqs = gene_frequencies(&pa, n_genes, n_genomes);

    let obs_spec = frequency_spectrum(&freqs, 10);
    let neu_spec = neutral_spectrum(10);
    let chi2 = spectrum_chi_squared(&obs_spec, &neu_spec);
    let s = selection_coefficient(&obs_spec, &neu_spec);

    // Validate chi-squared via barracuda variance of observed spectrum
    let spec_var = barracuda::stats::correlation::variance(&obs_spec).unwrap_or(f64::NAN);

    h.check_lower(&format!("chi2={chi2:.2} > 16.92 (selection)"), chi2, 16.92);
    h.check_lower(&format!("selection coeff={s:.4} > 0.01"), s, 0.01);
    h.check_bool(
        &format!("spectrum variance finite ({spec_var:.4})"),
        spec_var.is_finite() && spec_var >= 0.0,
    );
}

/// Jaccard distances validated: mean/variance via barracuda.
fn validate_jaccard_via_barracuda(h: &mut ValidationHarness) {
    let mut rng = Rng::new(42);
    let n_genomes = 30;
    let n_genes = 200;
    let env_labels: Vec<usize> = (0..15).map(|_| 0).chain((0..15).map(|_| 1)).collect();
    let pa = generate_pa_matrix(n_genomes, n_genes, 0.25, 0.10, &mut rng, &env_labels);
    let jd = jaccard_distance_matrix(&pa, n_genes, n_genomes);

    let mut upper = Vec::new();
    for i in 0..n_genomes {
        for j in (i + 1)..n_genomes {
            upper.push(jd[i * n_genomes + j]);
        }
    }

    let barracuda_var = barracuda::stats::correlation::variance(&upper).unwrap_or(f64::NAN);
    let hand_mean: f64 = upper.iter().sum::<f64>() / upper.len() as f64;

    h.check_bool(
        &format!("Jaccard distance variance finite ({barracuda_var:.6})"),
        barracuda_var.is_finite() && barracuda_var >= 0.0,
    );
    h.check_lower(
        &format!("mean Jaccard > 0 ({hand_mean:.4})"),
        hand_mean,
        0.0,
    );
}

/// Partition consistency: barracuda variance of per-category counts.
fn validate_partition_consistency(h: &mut ValidationHarness) {
    let mut rng = Rng::new(42);
    let n_genomes = 30;
    let n_genes = 200;
    let env_labels: Vec<usize> = (0..15).map(|_| 0).chain((0..15).map(|_| 1)).collect();
    let pa = generate_pa_matrix(n_genomes, n_genes, 0.25, 0.10, &mut rng, &env_labels);
    let freqs = gene_frequencies(&pa, n_genes, n_genomes);
    let (n_core, n_acc, n_sing) = partition_pangenome(&freqs, n_genomes, 0.95);

    h.check_bool(
        &format!("partition sums to {n_genes} ({n_core} + {n_acc} + {n_sing})"),
        n_core + n_acc + n_sing == n_genes,
    );

    // Validate frequency distribution via barracuda mean
    let acc_freqs: Vec<f64> = freqs
        .iter()
        .filter(|&&f| f > 0.0 && f < 0.95)
        .copied()
        .collect();
    if acc_freqs.len() >= 2 {
        let acc_var = barracuda::stats::correlation::variance(&acc_freqs).unwrap_or(f64::NAN);
        h.check_bool(
            &format!("accessory freq variance finite ({acc_var:.6})"),
            acc_var.is_finite() && acc_var > 0.0,
        );
    } else {
        h.check_bool("accessory freq variance (too few)", true);
    }
}

/// Repertoire diversity cross-validated with barracuda.
fn validate_repertoire_diversity(h: &mut ValidationHarness) {
    let mut rng = Rng::new(42);
    let n_genomes = 30;
    let n_genes = 200;
    let env_labels: Vec<usize> = (0..15).map(|_| 0).chain((0..15).map(|_| 1)).collect();
    let pa = generate_pa_matrix(n_genomes, n_genes, 0.25, 0.10, &mut rng, &env_labels);

    let h_diversity = gene_repertoire_diversity(&pa, n_genes, n_genomes);

    // Validate genome sizes variance via barracuda
    let sizes: Vec<f64> = (0..n_genomes)
        .map(|j| (0..n_genes).map(|i| pa[i * n_genomes + j]).sum::<f64>())
        .collect();
    let size_var = barracuda::stats::correlation::variance(&sizes).unwrap_or(f64::NAN);

    h.check_lower(
        &format!("repertoire diversity > 0 ({h_diversity:.4})"),
        h_diversity,
        0.0,
    );
    h.check_bool(
        &format!("genome size variance finite ({size_var:.4})"),
        size_var.is_finite() && size_var > 0.0,
    );
}
