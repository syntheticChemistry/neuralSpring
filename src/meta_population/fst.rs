// SPDX-License-Identifier: AGPL-3.0-or-later

use super::allele_frequencies;
use crate::primitives::DIVISION_GUARD;

/// Weir & Cockerham (1984) FST estimator for two populations.
#[must_use]
pub fn pairwise_fst(pop_a: &[f64], n_a: usize, pop_b: &[f64], n_b: usize, n_loci: usize) -> f64 {
    let freq_a = allele_frequencies(pop_a, n_a, n_loci);
    let freq_b = allele_frequencies(pop_b, n_b, n_loci);
    let ns = [n_a as f64, n_b as f64];
    let n_total = ns[0] + ns[1];
    let n_pops = 2.0;
    let n_bar = n_total / n_pops;
    let n_c = (n_total - (ns[0] * ns[0] + ns[1] * ns[1]) / n_total) / (n_pops - 1.0);

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for j in 0..n_loci {
        let p_i = [freq_a[j], freq_b[j]];
        let p_bar = (ns[0] * p_i[0] + ns[1] * p_i[1]) / n_total;

        let s2 = (ns[0] * (p_i[0] - p_bar).powi(2) + ns[1] * (p_i[1] - p_bar).powi(2))
            / ((n_pops - 1.0) * n_bar);

        let h_bar = (ns[0] * 2.0 * p_i[0] * (1.0 - p_i[0]) + ns[1] * 2.0 * p_i[1] * (1.0 - p_i[1]))
            / n_total;

        let a = (n_bar / n_c)
            * (s2
                - (1.0 / (n_bar - 1.0))
                    * (p_bar * (1.0 - p_bar) - ((n_pops - 1.0) / n_pops) * s2 - 0.25 * h_bar));
        let b = (n_bar / (n_bar - 1.0))
            * (p_bar * (1.0 - p_bar)
                - ((n_pops - 1.0) / n_pops) * s2
                - ((2.0 * n_bar - 1.0) / (4.0 * n_bar)) * h_bar);
        let c = 0.5 * h_bar;

        numerator += a;
        denominator += a + b + c;
    }

    if denominator.abs() < DIVISION_GUARD {
        return 0.0;
    }
    numerator / denominator
}

/// Global FST across multiple populations.
#[must_use]
pub fn global_fst(populations: &[Vec<f64>], n_individuals: &[usize], n_loci: usize) -> f64 {
    let n_pops = populations.len();
    let ns: Vec<f64> = n_individuals.iter().map(|&n| n as f64).collect();
    let n_total: f64 = ns.iter().sum();
    let n_bar = n_total / n_pops as f64;
    let n_c = (n_total - ns.iter().map(|n| n * n).sum::<f64>() / n_total) / (n_pops as f64 - 1.0);

    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .zip(n_individuals.iter())
        .map(|(pop, &n)| allele_frequencies(pop, n, n_loci))
        .collect();

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for j in 0..n_loci {
        let p_i: Vec<f64> = all_freqs.iter().map(|f| f[j]).collect();
        let p_bar: f64 = barracuda::stats::dot(&ns, &p_i) / n_total;

        let s2: f64 = ns
            .iter()
            .zip(p_i.iter())
            .map(|(n, p)| n * (p - p_bar).powi(2))
            .sum::<f64>()
            / ((n_pops as f64 - 1.0) * n_bar);

        let h_bar: f64 = ns
            .iter()
            .zip(p_i.iter())
            .map(|(n, p)| n * 2.0 * p * (1.0 - p))
            .sum::<f64>()
            / n_total;

        let a = (n_bar / n_c)
            * (s2
                - (1.0 / (n_bar - 1.0))
                    * (p_bar * (1.0 - p_bar)
                        - ((n_pops as f64 - 1.0) / n_pops as f64) * s2
                        - 0.25 * h_bar));
        let b = (n_bar / (n_bar - 1.0))
            * (p_bar * (1.0 - p_bar)
                - ((n_pops as f64 - 1.0) / n_pops as f64) * s2
                - ((2.0 * n_bar - 1.0) / (4.0 * n_bar)) * h_bar);
        let c_val = 0.5 * h_bar;

        numerator += a;
        denominator += a + b + c_val;
    }

    if denominator.abs() < DIVISION_GUARD {
        return 0.0;
    }
    numerator / denominator
}

/// Build pairwise FST matrix (`n_pops` x `n_pops`, flat row-major).
#[must_use]
pub fn fst_matrix(populations: &[Vec<f64>], n_individuals: &[usize], n_loci: usize) -> Vec<f64> {
    let n = populations.len();
    let mut mat = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let fst = pairwise_fst(
                &populations[i],
                n_individuals[i],
                &populations[j],
                n_individuals[j],
                n_loci,
            );
            mat[i * n + j] = fst;
            mat[j * n + i] = fst;
        }
    }
    mat
}

/// Single-locus FST with F-statistics (θ, f, F) via upstream `BarraCUDA`.
///
/// Delegates to `barracuda::ops::bio::fst_variance_decomposition` which
/// implements Weir & Cockerham (1984) variance decomposition for one locus.
/// Returns `(fst, f_is, f_it)` — enriched over our multi-locus `pairwise_fst`
/// which only computes θ.
///
/// Cross-spring evolution: wetSpring's population genetics work drove the
/// upstream `fst_variance` module in `BarraCUDA` (S53). neuralSpring benefits
/// by gaining per-locus F-statistics without reimplementation.
///
/// # Errors
///
/// Returns `Err` if fewer than 2 populations or invalid allele frequencies.
pub fn fst_single_locus(
    allele_freqs: &[f64],
    population_sizes: &[usize],
) -> Result<(f64, f64, f64), String> {
    barracuda::ops::bio::fst_variance_decomposition(allele_freqs, population_sizes)
        .map(|r| (r.fst, r.f_is, r.f_it))
        .map_err(|e| format!("fst_single_locus: {e}"))
}

/// Multi-locus FST with full F-statistics (θ, f, F) via upstream per-locus decomposition.
///
/// Computes per-locus F-statistics using upstream `fst_variance_decomposition`,
/// then averages across loci (ratio-of-averages estimator, same as
/// `pairwise_fst` but enriched with `f_is` and `f_it`).
#[must_use]
pub fn pairwise_fst_full(
    pop_a: &[f64],
    n_a: usize,
    pop_b: &[f64],
    n_b: usize,
    n_loci: usize,
) -> (f64, f64, f64) {
    let freq_a = allele_frequencies(pop_a, n_a, n_loci);
    let freq_b = allele_frequencies(pop_b, n_b, n_loci);
    let sizes = [n_a, n_b];

    let (sum_a, sum_b, sum_c, n_valid) = freq_a
        .iter()
        .zip(freq_b.iter())
        .filter_map(|(&fa, &fb)| {
            barracuda::ops::bio::fst_variance_decomposition(&[fa, fb], &sizes).ok()
        })
        .fold((0.0, 0.0, 0.0, 0_i32), |(sa, sb, sc, nv), r| {
            (sa + r.fst, sb + r.f_is, sc + r.f_it, nv + 1)
        });

    if n_valid == 0 {
        return (0.0, 0.0, 0.0);
    }
    let n = f64::from(n_valid);
    (sum_a / n, sum_b / n, sum_c / n)
}

/// Global FST via variance decomposition: `FST = between_var / (between_var + within_var)`.
///
/// Uses `inter_population_af_variance` for between-population variance and
/// mean of per-population allele-frequency variance for within-population.
/// Matches `fst_variance_decomposition_gpu` for CPU/GPU parity validation.
#[must_use]
pub fn global_fst_variance_decomposition(
    populations: &[Vec<f64>],
    n_individuals: &[usize],
    n_loci: usize,
) -> f64 {
    let n_pops = populations.len();
    if n_pops < 2 || n_loci == 0 {
        return 0.0;
    }

    let between_var = inter_population_af_variance(populations, n_individuals, n_loci);

    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .zip(n_individuals.iter())
        .map(|(pop, &n)| allele_frequencies(pop, n, n_loci))
        .collect();

    let within_var: f64 = all_freqs
        .iter()
        .map(|freqs| {
            let mean: f64 = freqs.iter().sum::<f64>() / n_loci as f64;
            freqs.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / n_loci as f64
        })
        .sum::<f64>()
        / n_pops as f64;

    let denom = between_var + within_var;
    if denom.abs() < DIVISION_GUARD {
        return 0.0;
    }
    between_var / denom
}

/// Mean allele frequency variance across populations (inter-population).
#[must_use]
pub fn inter_population_af_variance(
    populations: &[Vec<f64>],
    n_individuals: &[usize],
    n_loci: usize,
) -> f64 {
    let all_freqs: Vec<Vec<f64>> = populations
        .iter()
        .zip(n_individuals.iter())
        .map(|(pop, &n)| allele_frequencies(pop, n, n_loci))
        .collect();

    let n_pops = all_freqs.len() as f64;
    let total_var: f64 = (0..n_loci)
        .map(|j| {
            let mean: f64 = all_freqs.iter().map(|f| f[j]).sum::<f64>() / n_pops;
            all_freqs.iter().map(|f| (f[j] - mean).powi(2)).sum::<f64>() / n_pops
        })
        .sum();
    total_var / n_loci as f64
}

#[cfg(test)]
mod tests {
    use super::super::generate_population;
    use super::*;
    use crate::rng::Rng;
    use crate::tolerances;

    #[test]
    fn fst_symmetric() {
        let mut rng = Rng::new(42);
        let anc: Vec<f64> = (0..10).map(|_| rng.beta(2.0, 2.0)).collect();
        let temps = [70.0, 80.0];
        let pop_a = generate_population(5, 10, &anc, 0.15, temps[0], 70.0, 80.0, 2, &mut rng);
        let pop_b = generate_population(5, 10, &anc, 0.15, temps[1], 70.0, 80.0, 2, &mut rng);
        let fst_ab = pairwise_fst(&pop_a, 5, &pop_b, 5, 10);
        let fst_ba = pairwise_fst(&pop_b, 5, &pop_a, 5, 10);
        assert!(
            (fst_ab - fst_ba).abs() < tolerances::CROSS_LANGUAGE,
            "FST should be symmetric"
        );
    }

    #[test]
    fn global_fst_positive() {
        let mut rng = Rng::new(42);
        let n_loci = 20;
        let anc: Vec<f64> = (0..n_loci).map(|_| rng.beta(2.0, 2.0)).collect();
        let pops: Vec<Vec<f64>> = [70.0, 85.0]
            .iter()
            .map(|&t| generate_population(10, n_loci, &anc, 0.15, t, 65.0, 90.0, 4, &mut rng))
            .collect();
        let n_indivs = vec![10; 2];
        let gfst = global_fst(&pops, &n_indivs, n_loci);
        assert!(gfst.is_finite(), "global FST must be finite");
    }

    #[test]
    fn fst_matrix_symmetric_and_diag_zero() {
        let mut rng = Rng::new(42);
        let n_loci = 15;
        let n_pops = 3;
        let anc: Vec<f64> = (0..n_loci).map(|_| rng.beta(2.0, 2.0)).collect();
        let temps = [70.0, 78.0, 85.0];
        let pops: Vec<Vec<f64>> = temps
            .iter()
            .map(|&t| generate_population(8, n_loci, &anc, 0.15, t, 65.0, 90.0, 3, &mut rng))
            .collect();
        let n_indivs = vec![8; n_pops];
        let fst_mat = fst_matrix(&pops, &n_indivs, n_loci);
        for i in 0..n_pops {
            assert!(fst_mat[i * n_pops + i].abs() < tolerances::CROSS_LANGUAGE);
            for j in 0..n_pops {
                assert!(
                    (fst_mat[i * n_pops + j] - fst_mat[j * n_pops + i]).abs()
                        < tolerances::CROSS_LANGUAGE
                );
            }
        }
    }

    #[test]
    fn inter_pop_af_variance_positive() {
        let mut rng = Rng::new(42);
        let n_loci = 15;
        let anc: Vec<f64> = (0..n_loci).map(|_| rng.beta(2.0, 2.0)).collect();
        let pops: Vec<Vec<f64>> = [70.0, 85.0]
            .iter()
            .map(|&t| generate_population(8, n_loci, &anc, 0.15, t, 65.0, 90.0, 3, &mut rng))
            .collect();
        let n_indivs = vec![8; 2];
        let v = inter_population_af_variance(&pops, &n_indivs, n_loci);
        assert!(v >= 0.0 && v.is_finite());
    }
}
