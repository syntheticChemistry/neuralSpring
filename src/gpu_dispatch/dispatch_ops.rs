// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain-specific dispatched operations on [`Dispatcher`].
//!
//! Each method routes to either GPU or CPU based on runtime capabilities.
//! GPU paths delegate to `barracuda::dispatch` or `crate::gpu_ops`;
//! CPU paths use local reference implementations.

use super::cpu_fallback;
use super::Dispatcher;

impl Dispatcher {
    // ═══════════════════════════════════════════════════════════════
    // Linear algebra
    // (mat_mul, frobenius_norm, transpose delegate to upstream
    //  barracuda::dispatch::domain_ops which handles GPU/CPU routing
    //  with size-based thresholds — cross-spring evolved from
    //  hotSpring precision shaders + wetSpring bio shaders)
    // ═══════════════════════════════════════════════════════════════

    /// Matrix multiply: delegates to upstream `barracuda::dispatch::matmul_dispatch`.
    #[must_use]
    pub fn mat_mul(&self, a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        barracuda::dispatch::matmul_dispatch(a, b, n, n, n, self.wgpu_device()).unwrap_or_else(
            |e| {
                eprintln!("[dispatch] mat_mul upstream failed: {e}");
                crate::spectral_commutativity::mat_mul(a, b, n)
            },
        )
    }

    /// Frobenius norm: delegates to upstream `barracuda::dispatch::frobenius_norm_dispatch`.
    #[must_use]
    pub fn frobenius_norm(&self, a: &[f64]) -> f64 {
        barracuda::dispatch::frobenius_norm_dispatch(a, self.wgpu_device()).unwrap_or_else(|e| {
            eprintln!("[dispatch] frobenius_norm upstream failed: {e}");
            crate::spectral_commutativity::frobenius_norm(a)
        })
    }

    /// Transpose: delegates to upstream `barracuda::dispatch::transpose_dispatch`.
    #[must_use]
    pub fn transpose(&self, a: &[f64], n: usize) -> Vec<f64> {
        barracuda::dispatch::transpose_dispatch(a, n, n, self.wgpu_device()).unwrap_or_else(|e| {
            eprintln!("[dispatch] transpose upstream failed: {e}");
            crate::spectral_commutativity::transpose(a, n)
        })
    }

    /// Commutator `[A,B]` = AB - BA: GPU if available, CPU fallback.
    #[must_use]
    pub fn commutator(&self, a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        self.gpu_or_cpu(
            "commutator",
            |dev| crate::gpu_ops::commutator_gpu(a, b, n, dev),
            || crate::spectral_commutativity::commutator(a, b, n),
        )
    }

    /// Distance to normal: GPU if available, CPU fallback.
    #[must_use]
    pub fn distance_to_normal(&self, a: &[f64], n: usize) -> f64 {
        self.gpu_or_cpu(
            "distance_to_normal",
            |dev| crate::gpu_ops::distance_to_normal_gpu(a, n, dev),
            || crate::spectral_commutativity::distance_to_normal(a, n),
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // Activations / distributions
    // ═══════════════════════════════════════════════════════════════

    /// Softmax (global): delegates to upstream `barracuda::dispatch::softmax_dispatch`.
    #[must_use]
    pub fn softmax(&self, x: &[f64]) -> Vec<f64> {
        barracuda::dispatch::softmax_dispatch(x, self.wgpu_device()).unwrap_or_else(|e| {
            eprintln!("[dispatch] softmax upstream failed: {e}");
            crate::transformer::softmax(x)
        })
    }

    /// Row-wise softmax: uses upstream `Tensor::softmax_dim(1)` (rewired S72).
    ///
    /// Cross-spring evolution: neuralSpring requested `softmax_dim(axis)` in V20;
    /// `ToadStool` implemented in `tensor_axis_ops.rs` (S60). Previously, row-wise
    /// softmax required manual per-row dispatch or `ScaledDotProductAttention`.
    #[must_use]
    pub fn softmax_row_wise(&self, matrix: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
        if let Some(dev) = self.wgpu_device() {
            let m_f32: Vec<f32> = matrix.iter().map(|&v| v as f32).collect();
            if let Ok(t) =
                barracuda::tensor::Tensor::from_data(&m_f32, vec![n_rows, n_cols], dev.clone())
            {
                if let Ok(sm) = t.softmax_dim(1) {
                    if let Ok(out) = sm.to_vec() {
                        return out.into_iter().map(f64::from).collect();
                    }
                }
            }
        }
        crate::neural_pgm::weight_to_transition(matrix, n_rows, n_cols)
    }

    /// Boltzmann distribution: GPU if available, CPU fallback.
    #[must_use]
    pub fn boltzmann(&self, fitnesses: &[f64], beta: f64) -> Vec<f64> {
        self.gpu_or_cpu(
            "boltzmann",
            |dev| crate::gpu_ops::boltzmann_gpu(fitnesses, beta, dev),
            || crate::counterdiabatic::boltzmann_distribution(fitnesses, beta),
        )
    }

    /// GELU activation: delegates to upstream `barracuda::dispatch::gelu_dispatch`.
    ///
    /// GPU path uses fused WGSL kernel; CPU fallback uses `crate::transformer::gelu`.
    #[must_use]
    pub fn gelu(&self, x: &[f64]) -> Vec<f64> {
        barracuda::dispatch::gelu_dispatch(x, self.wgpu_device()).unwrap_or_else(|e| {
            eprintln!("[dispatch] gelu upstream failed: {e}");
            x.iter().map(|&v| crate::transformer::gelu(v)).collect()
        })
    }

    /// Hill activation batch: GPU if available, CPU fallback.
    #[must_use]
    pub fn hill_activation_batch(&self, x: &[f64], vmax: f64, k: f64, n_hill: f64) -> Vec<f64> {
        self.gpu_or_cpu(
            "hill_activation_batch",
            |dev| crate::gpu_ops::hill_activation_batch_gpu(x, vmax, k, n_hill, dev),
            || {
                x.iter()
                    .map(|&xi| crate::primitives::hill_activation(xi, vmax, k, n_hill))
                    .collect()
            },
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // Reductions / statistics
    // ═══════════════════════════════════════════════════════════════

    /// L2 distance: delegates to upstream `barracuda::dispatch::l2_distance_dispatch`.
    #[must_use]
    pub fn l2_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        barracuda::dispatch::l2_distance_dispatch(a, b, self.wgpu_device()).unwrap_or_else(|e| {
            eprintln!("[dispatch] l2_distance upstream failed: {e}");
            crate::modes::l2_distance(a, b)
        })
    }

    /// Shannon entropy: GPU if available, CPU fallback.
    #[must_use]
    pub fn shannon_entropy(&self, p: &[f64]) -> f64 {
        self.gpu_or_cpu(
            "shannon_entropy",
            |dev| crate::gpu_ops::shannon_entropy_gpu(p, dev),
            || crate::primitives::shannon_entropy(p),
        )
    }

    /// Mean: delegates to upstream `barracuda::dispatch::mean_dispatch`.
    #[must_use]
    pub fn mean(&self, data: &[f64]) -> f64 {
        barracuda::dispatch::mean_dispatch(data, self.wgpu_device()).unwrap_or_else(|e| {
            eprintln!("[dispatch] mean upstream failed: {e}");
            if data.is_empty() {
                0.0
            } else {
                data.iter().sum::<f64>() / data.len() as f64
            }
        })
    }

    /// Variance: delegates to upstream `barracuda::dispatch::variance_dispatch`.
    #[must_use]
    pub fn variance(&self, data: &[f64]) -> f64 {
        barracuda::dispatch::variance_dispatch(data, self.wgpu_device()).unwrap_or_else(|e| {
            eprintln!("[dispatch] variance upstream failed: {e}");
            cpu_fallback::variance(data)
        })
    }

    /// Pearson correlation: GPU if available, CPU fallback.
    #[must_use]
    pub fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        self.gpu_or_cpu(
            "pearson_correlation",
            |dev| crate::gpu_ops::pearson_correlation_gpu(x, y, dev),
            || cpu_fallback::pearson(x, y),
        )
    }

    /// Chi-squared statistic: GPU if available, CPU fallback.
    #[must_use]
    pub fn chi_squared(&self, observed: &[f64], expected: &[f64]) -> f64 {
        self.gpu_or_cpu(
            "chi_squared",
            |dev| crate::gpu_ops::chi_squared_gpu(observed, expected, dev),
            || cpu_fallback::chi_squared(observed, expected),
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // HMM (Liu 016–018)
    // ═══════════════════════════════════════════════════════════════

    /// HMM backward step: GPU if available, CPU fallback.
    #[must_use]
    pub fn hmm_backward_step(
        &self,
        beta_next: &[f64],
        transition: &[f64],
        emission_col: &[f64],
        scale: f64,
        n_states: usize,
    ) -> Vec<f64> {
        self.gpu_or_cpu(
            "hmm_backward_step",
            |dev| {
                crate::gpu_ops::hmm_backward_step_gpu(
                    beta_next,
                    transition,
                    emission_col,
                    scale,
                    n_states,
                    dev,
                )
            },
            || {
                cpu_fallback::hmm_backward_step(
                    beta_next,
                    transition,
                    emission_col,
                    scale,
                    n_states,
                )
            },
        )
    }

    /// HMM Viterbi step: GPU if available, CPU fallback.
    /// Returns `(delta_new, psi)`.
    #[must_use]
    pub fn hmm_viterbi_step(
        &self,
        delta_prev: &[f64],
        log_transition: &[f64],
        log_emission_col: &[f64],
        n_states: usize,
    ) -> (Vec<f64>, Vec<usize>) {
        self.gpu_or_cpu(
            "hmm_viterbi_step",
            |dev| {
                crate::gpu_ops::hmm_viterbi_step_gpu(
                    delta_prev,
                    log_transition,
                    log_emission_col,
                    n_states,
                    dev,
                )
            },
            || {
                cpu_fallback::hmm_viterbi_step(
                    delta_prev,
                    log_transition,
                    log_emission_col,
                    n_states,
                )
            },
        )
    }

    /// HMM forward chain: GPU full forward algorithm if available, CPU fallback.
    ///
    /// Composes GPU GEMV steps over all observations, returning log-likelihood.
    #[must_use]
    pub fn hmm_forward_chain(
        &self,
        initial: &[f64],
        transition: &[f64],
        emission: &[f64],
        observations: &[usize],
        n_states: usize,
        n_obs: usize,
    ) -> f64 {
        self.gpu_or_cpu(
            "hmm_forward_chain",
            |dev| {
                crate::gpu_ops::hmm_forward_chain_gpu(
                    initial,
                    transition,
                    emission,
                    observations,
                    n_states,
                    n_obs,
                    dev,
                )
            },
            || {
                let hmm = crate::hmm::Hmm::from_flat(
                    transition.to_vec(),
                    emission.to_vec(),
                    initial.to_vec(),
                    n_states,
                    n_obs,
                );
                hmm.forward(observations).1
            },
        )
    }

    /// HMM Viterbi chain: GPU full Viterbi if available, CPU fallback.
    ///
    /// Returns `(state_sequence, log_probability)`.
    #[must_use]
    pub fn hmm_viterbi_chain(
        &self,
        initial: &[f64],
        transition: &[f64],
        emission: &[f64],
        observations: &[usize],
        n_states: usize,
        n_obs: usize,
    ) -> (Vec<usize>, f64) {
        self.gpu_or_cpu(
            "hmm_viterbi_chain",
            |dev| {
                crate::gpu_ops::hmm_viterbi_chain_gpu(
                    initial,
                    transition,
                    emission,
                    observations,
                    n_states,
                    n_obs,
                    dev,
                )
            },
            || {
                let hmm = crate::hmm::Hmm::from_flat(
                    transition.to_vec(),
                    emission.to_vec(),
                    initial.to_vec(),
                    n_states,
                    n_obs,
                );
                hmm.viterbi(observations)
            },
        )
    }

    /// HMM forward step: delegates to upstream `barracuda::dispatch::hmm_forward_dispatch`.
    ///
    /// GPU path uses fused WGSL kernel; CPU fallback uses local matrix-vector multiply.
    /// Completes the `domain_ops` pattern started in S58.
    #[must_use]
    pub fn hmm_forward_step(
        &self,
        alpha_prev: &[f64],
        transition: &[f64],
        emission_col: &[f64],
        n_states: usize,
    ) -> (Vec<f64>, f64) {
        barracuda::dispatch::hmm_forward_dispatch(
            alpha_prev,
            transition,
            emission_col,
            n_states,
            self.wgpu_device(),
        )
        .unwrap_or_else(|e| {
            eprintln!("[dispatch] hmm_forward_step upstream failed: {e}");
            cpu_fallback::hmm_forward_step(alpha_prev, transition, emission_col, n_states)
        })
    }

    // ═══════════════════════════════════════════════════════════════
    // Population genetics (Campbell 025)
    // ═══════════════════════════════════════════════════════════════

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
                let vecs: Vec<Vec<f64>> = populations.iter().map(|s| s.to_vec()).collect();
                crate::meta_population::inter_population_af_variance(&vecs, n_individuals, n_loci)
            },
        )
    }

    /// Pairwise FST (Weir-Cockerham): GPU allele freqs + per-locus decomposition.
    ///
    /// Uses GPU for allele frequency computation, locus-level
    /// Weir-Cockerham terms computed on CPU (reduction-heavy, low-N).
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

    /// Single-locus FST with full F-statistics via upstream `BarraCUDA`.
    ///
    /// Delegates to `barracuda::ops::bio::fst_variance_decomposition` (CPU,
    /// cross-spring evolved from wetSpring S53 population genetics work).
    /// Returns `(fst, f_is, f_it)` — richer than the θ-only `pairwise_fst`.
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
    ///
    /// Uses upstream `fst_variance_decomposition` per-locus, then averages.
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

    // ═══════════════════════════════════════════════════════════════
    // Game theory (Bruger/Waters 019)
    // ═══════════════════════════════════════════════════════════════

    /// Replicator dynamics step: GPU matmul if available, CPU fallback.
    #[must_use]
    pub fn replicator_step(&self, freq: &[f64; 2], payoff: &[[f64; 2]; 2], dt: f64) -> [f64; 2] {
        self.gpu_or_cpu(
            "replicator_step",
            |dev| crate::gpu_ops::replicator_step_gpu(freq, payoff, dt, dev),
            || cpu_fallback::replicator_step(freq, payoff, dt),
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // Eigensolvers (Session 47)
    // ═══════════════════════════════════════════════════════════════

    /// Symmetric eigenvalue decomposition: GPU (`BatchedEighGpu`) if available.
    #[must_use]
    pub fn eigh(&self, a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        self.gpu_or_cpu(
            "eigh",
            |dev| crate::gpu_ops::eigh_gpu(a, n, dev),
            || {
                let r = crate::eigh::eigh_householder_qr(a, n);
                (r.eigenvalues, r.eigenvectors)
            },
        )
    }

    /// Batch disorder sweep on GPU: eigensolve + mean IPR for all W values.
    #[must_use]
    pub fn disorder_sweep(
        &self,
        hamiltonians: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Option<Vec<f64>> {
        let dev = self.wgpu_device()?;
        crate::gpu_ops::disorder_sweep_gpu(hamiltonians, n, batch_size, dev).ok()
    }

    // ═══════════════════════════════════════════════════════════════
    // Pangenome selection (Moulana 024)
    // ═══════════════════════════════════════════════════════════════

    /// Spectrum chi-squared with GPU dispatch.
    #[must_use]
    pub fn spectrum_chi_squared(&self, observed: &[f64], expected_frac: &[f64]) -> f64 {
        self.gpu_or_cpu(
            "spectrum_chi_squared",
            |dev| crate::gpu_ops::spectrum_chi_squared_gpu(observed, expected_frac, dev),
            || crate::pangenome_selection::spectrum_chi_squared(observed, expected_frac),
        )
    }

    /// Selection coefficient with GPU dispatch.
    #[must_use]
    pub fn selection_coefficient(&self, observed: &[f64], neutral: &[f64]) -> f64 {
        self.gpu_or_cpu(
            "selection_coefficient",
            |dev| crate::gpu_ops::selection_coefficient_gpu(observed, neutral, dev),
            || crate::pangenome_selection::selection_coefficient(observed, neutral),
        )
    }
}
