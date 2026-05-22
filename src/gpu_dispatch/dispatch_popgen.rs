// SPDX-License-Identifier: AGPL-3.0-or-later

//! Population genetics dispatch operations (Campbell 025).

use super::Dispatcher;

impl Dispatcher {
    /// Allele frequencies: GPU column-sum if available, CPU fallback.
    #[must_use]
    pub fn allele_frequencies(&self, pop: &[f64], n_individuals: usize, n_loci: usize) -> Vec<f64> {
        self.gpu_or_cpu(
            "allele_frequencies",
            |dev| crate::gpu_ops::allele_frequencies_gpu(pop, n_individuals, n_loci, dev),
            || crate::meta_population::allele_frequencies(pop, n_individuals, n_loci),
        )
    }

    /// Nucleotide diversity: GPU if available, CPU fallback.
    #[must_use]
    pub fn nucleotide_diversity(&self, pop: &[f64], n_individuals: usize, n_loci: usize) -> f64 {
        self.gpu_or_cpu(
            "nucleotide_diversity",
            |dev| crate::gpu_ops::nucleotide_diversity_gpu(pop, n_individuals, n_loci, dev),
            || crate::meta_population::nucleotide_diversity(pop, n_individuals, n_loci),
        )
    }

    /// Matrix correlation (upper triangle Pearson): GPU if available, CPU fallback.
    #[must_use]
    pub fn matrix_correlation(&self, a: &[f64], b: &[f64], n: usize) -> f64 {
        self.gpu_or_cpu(
            "matrix_correlation",
            |dev| crate::gpu_ops::matrix_correlation_gpu(a, b, n, dev),
            || crate::meta_population::matrix_correlation(a, b, n),
        )
    }

    /// Geographic distance matrix: GPU if available, CPU fallback.
    #[must_use]
    pub fn geographic_distances(&self, coords: &[(f64, f64)]) -> Vec<f64> {
        self.gpu_or_cpu(
            "geographic_distances",
            |dev| crate::gpu_ops::geographic_distance_matrix_gpu(coords, dev),
            || crate::meta_population::geographic_distance_matrix(coords),
        )
    }

    /// Thermal diversity correlation: GPU Pearson if available, CPU fallback.
    #[must_use]
    pub fn thermal_diversity_correlation(&self, pi_values: &[f64], temperatures: &[f64]) -> f64 {
        self.gpu_or_cpu(
            "thermal_diversity_correlation",
            |dev| crate::gpu_ops::thermal_diversity_correlation_gpu(pi_values, temperatures, dev),
            || crate::meta_population::thermal_diversity_correlation(pi_values, temperatures),
        )
    }

    /// Inter-population allele frequency variance: GPU if available, CPU fallback.
    #[must_use]
    pub fn inter_population_af_variance(
        &self,
        populations: &[&[f64]],
        n_individuals: &[usize],
        n_loci: usize,
    ) -> f64 {
        self.gpu_or_cpu(
            "inter_population_af_variance",
            |dev| {
                crate::gpu_ops::inter_population_af_variance_gpu(
                    populations,
                    n_individuals,
                    n_loci,
                    dev,
                )
            },
            || {
                crate::meta_population::inter_population_af_variance(populations, n_individuals, n_loci)
            },
        )
    }

    /// Pairwise FST (Weir-Cockerham): GPU allele freqs + per-locus decomposition.
    #[must_use]
    pub fn pairwise_fst(
        &self,
        pop_a: &[f64],
        n_a: usize,
        pop_b: &[f64],
        n_b: usize,
        n_loci: usize,
    ) -> f64 {
        self.gpu_or_cpu(
            "pairwise_fst",
            |dev| crate::gpu_ops::pairwise_fst_gpu(pop_a, n_a, pop_b, n_b, n_loci, dev),
            || crate::meta_population::pairwise_fst(pop_a, n_a, pop_b, n_b, n_loci),
        )
    }

    /// Single-locus FST with full F-statistics.
    ///
    /// Returns `(fst, f_is, f_it)`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if fewer than 2 populations or invalid allele frequencies.
    pub fn fst_single_locus(
        &self,
        allele_freqs: &[f64],
        population_sizes: &[usize],
    ) -> Result<(f64, f64, f64), String> {
        crate::meta_population::fst_single_locus(allele_freqs, population_sizes)
    }

    /// Multi-locus FST with full F-statistics (θ, f, F).
    #[must_use]
    pub fn pairwise_fst_full(
        &self,
        pop_a: &[f64],
        n_a: usize,
        pop_b: &[f64],
        n_b: usize,
        n_loci: usize,
    ) -> (f64, f64, f64) {
        crate::meta_population::pairwise_fst_full(pop_a, n_a, pop_b, n_b, n_loci)
    }

    /// Global FST (multi-population Weir-Cockerham): GPU allele freqs + reduction.
    #[must_use]
    pub fn global_fst(
        &self,
        populations: &[Vec<f64>],
        n_individuals: &[usize],
        n_loci: usize,
    ) -> f64 {
        self.gpu_or_cpu(
            "global_fst",
            |dev| crate::gpu_ops::global_fst_gpu(populations, n_individuals, n_loci, dev),
            || crate::meta_population::global_fst(populations, n_individuals, n_loci),
        )
    }

    /// Global FST via variance decomposition.
    #[must_use]
    pub fn global_fst_variance_decomposition(
        &self,
        populations: &[Vec<f64>],
        n_individuals: &[usize],
        n_loci: usize,
    ) -> f64 {
        self.gpu_or_cpu(
            "global_fst_variance_decomposition",
            |dev| {
                let refs: Vec<&[f64]> = populations.iter().map(Vec::as_slice).collect();
                crate::gpu_ops::fst_variance_decomposition_gpu(&refs, n_individuals, n_loci, dev)
            },
            || {
                crate::meta_population::global_fst_variance_decomposition(
                    populations,
                    n_individuals,
                    n_loci,
                )
            },
        )
    }
}
