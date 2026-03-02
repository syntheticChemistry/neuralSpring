// SPDX-License-Identifier: AGPL-3.0-or-later

//! baseCamp GPU promotion methods for `Dispatcher`.
//!
//! Extends the `Dispatcher` with GPU-accelerated paths for all 5
//! baseCamp sub-theses (biophysical AI interpretability):
//!
//! - Sub-01: Weight spectral analysis (eigensolve on GPU)
//! - Sub-02: Information flow (attention eigensolve + signal matmul on GPU)
//! - Sub-03: Loss landscape (Hessian eigensolve on GPU)
//! - Sub-04: Belief propagation (f64 GEMV chain via `matmul_dispatch`)
//! - Sub-05: Agent interaction graph (pairwise L2 on GPU)

use super::Dispatcher;
use crate::primitives::LOG_GUARD;

impl Dispatcher {
    /// Weight-to-Hamiltonian construction + spectral analysis (Sub-01).
    ///
    /// Constructs the Anderson Hamiltonian from a weight matrix, then
    /// computes eigenvalues and eigenvectors via GPU-accelerated
    /// `eigh_gpu` (Jacobi eigensolver). Hamiltonian construction is
    /// O(mn) — negligible vs the O(n^3) eigensolve that runs on GPU.
    #[must_use]
    pub fn weight_spectral_analysis(
        &self,
        weights: &[f64],
        rows: usize,
        cols: usize,
    ) -> crate::weight_spectral::WeightSpectralResult {
        let ham = crate::weight_spectral::weight_to_hamiltonian(weights, rows, cols);
        let dim = rows + cols;
        let (eigenvalues, eigenvectors) = self.eigh(&ham, dim);
        let gamma = rows as f64 / cols.max(1) as f64;
        crate::weight_spectral::spectral_result_from_decomposition(
            eigenvalues,
            &eigenvectors,
            dim,
            gamma,
        )
    }

    /// Numerical Hessian + GPU eigensolve for landscape analysis (Sub-03).
    ///
    /// The Hessian is computed via central finite differences (CPU —
    /// requires arbitrary function evaluation). The expensive eigensolve
    /// routes through GPU via `eigh_gpu`.
    #[must_use]
    pub fn numerical_hessian(
        &self,
        f: impl Fn(&[f64]) -> f64,
        point: &[f64],
        h_step: f64,
    ) -> Vec<f64> {
        crate::loss_landscape::numerical_hessian(&f, point, h_step)
    }

    /// Full loss landscape analysis with GPU-accelerated eigensolve (Sub-03).
    ///
    /// Computes the Hessian via CPU finite differences, then routes the
    /// O(n^3) eigensolve through GPU. Scalar metrics (flatness, sharpness,
    /// saddle index, spectral gap) are computed from the GPU eigenvalues.
    #[must_use]
    pub fn landscape_analysis(
        &self,
        loss_fn: &dyn Fn(&[f64]) -> f64,
        params: &[f64],
        epsilon: f64,
        flatness_threshold: f64,
    ) -> crate::loss_landscape::LandscapeResult {
        let hessian = crate::loss_landscape::numerical_hessian(loss_fn, params, epsilon);
        let n = params.len();
        let (mut eigenvalues, _eigenvectors) = self.eigh(&hessian, n);
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let loss = loss_fn(params);

        crate::loss_landscape::LandscapeResult {
            loss,
            flatness: crate::loss_landscape::landscape_flatness(&eigenvalues, flatness_threshold),
            sharpness: crate::loss_landscape::landscape_sharpness(&eigenvalues),
            saddle_index: crate::loss_landscape::saddle_index(&eigenvalues),
            spectral_gap: crate::loss_landscape::spectral_gap(&eigenvalues),
            hessian_eigenvalues: eigenvalues,
        }
    }

    /// Belief propagation chain via GPU f64 GEMV dispatch (Sub-04).
    ///
    /// Propagates a probability distribution through a chain of
    /// transition matrices (row-stochastic). Each step is a
    /// matrix-vector multiply dispatched through `matmul_dispatch`
    /// for full f64 fidelity (no f32 truncation).
    #[must_use]
    pub fn belief_propagation(
        &self,
        input: &[f64],
        transitions: &[&[f64]],
        dims: &[usize],
    ) -> Vec<Vec<f64>> {
        let mut distributions = Vec::with_capacity(transitions.len() + 1);
        distributions.push(input.to_vec());
        let mut current = input.to_vec();
        for (idx, &trans) in transitions.iter().enumerate() {
            let out_dim = dims[idx];
            let in_dim = current.len();
            let next =
                barracuda::dispatch::transpose_dispatch(trans, in_dim, out_dim, self.wgpu_device())
                    .and_then(|trans_t| {
                        barracuda::dispatch::matmul_dispatch(
                            &trans_t,
                            &current,
                            out_dim,
                            in_dim,
                            1,
                            self.wgpu_device(),
                        )
                    })
                    .unwrap_or_else(|_| {
                        (0..out_dim)
                            .map(|j| {
                                trans
                                    .chunks_exact(out_dim)
                                    .zip(current.iter())
                                    .map(|(row, &cur)| row[j] * cur)
                                    .sum()
                            })
                            .collect()
                    });
            let sum: f64 = next.iter().sum();
            let normalized: Vec<f64> = if sum > LOG_GUARD {
                next.iter().map(|&v| v / sum).collect()
            } else {
                next
            };
            distributions.push(normalized.clone());
            current = normalized;
        }
        distributions
    }

    /// Attention spectral analysis with GPU eigensolve (Sub-02).
    ///
    /// Routes the O(n^3) eigendecomposition of the symmetrized attention
    /// Hamiltonian through GPU, returning eigenvalues, mean IPR, and
    /// level spacing ratio.
    #[must_use]
    pub fn attention_spectral_analysis(
        &self,
        attention: &[f64],
        n: usize,
    ) -> crate::information_flow::AttentionSpectralResult {
        let h = crate::information_flow::attention_to_hamiltonian(attention, n);
        let (mut eigenvalues, eigenvectors) = self.eigh(&h, n);
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean_ipr_val = crate::anderson_localization::mean_ipr(&eigenvectors, n);
        let lsr = crate::weight_spectral::level_spacing_ratio(&eigenvalues);
        crate::information_flow::AttentionSpectralResult {
            eigenvalues,
            mean_ipr: mean_ipr_val,
            level_spacing_ratio: lsr,
        }
    }

    /// Signal propagation with GPU matmul (Sub-02).
    ///
    /// Propagates input through weight matrices using `matmul_dispatch`
    /// for the matrix-vector products, with `ReLU` activation.
    #[must_use]
    pub fn mlp_signal_propagation(
        &self,
        input: &[f64],
        weight_matrices: &[&[f64]],
        layer_dims: &[usize],
    ) -> Vec<f64> {
        let mut variances = Vec::with_capacity(weight_matrices.len() + 1);
        let input_var = input.iter().map(|&x| x * x).sum::<f64>() / input.len().max(1) as f64;
        variances.push(input_var);

        let mut signal = input.to_vec();
        for (layer_idx, &weights) in weight_matrices.iter().enumerate() {
            let n_in = if layer_idx == 0 {
                input.len()
            } else {
                layer_dims[layer_idx - 1]
            };
            let n_out = layer_dims[layer_idx];

            let raw_output = barracuda::dispatch::matmul_dispatch(
                weights,
                &signal,
                n_out,
                n_in,
                1,
                self.wgpu_device(),
            )
            .unwrap_or_else(|_| {
                weights
                    .chunks_exact(n_in)
                    .take(n_out)
                    .map(|row| row.iter().zip(signal.iter()).map(|(&w, &s)| w * s).sum())
                    .collect()
            });

            let output: Vec<f64> = raw_output.iter().map(|&v| v.max(0.0)).collect();
            let var = output.iter().map(|&x| x * x).sum::<f64>() / n_out.max(1) as f64;
            variances.push(var);
            signal = output;
        }
        variances
    }

    /// Interaction graph as pairwise distance matrix (Sub-05).
    ///
    /// Computes the pairwise L2 distance matrix for a set of agent
    /// positions, then applies a communication range threshold.
    /// GPU-accelerated via `pairwise_l2_matrix_gpu`.
    #[must_use]
    pub fn agent_interaction_graph(
        &self,
        positions: &[f64],
        n_agents: usize,
        dim: usize,
        comm_range: f64,
    ) -> Vec<f64> {
        let upper_tri = self.gpu_or_cpu(
            "pairwise_l2",
            |dev| crate::gpu_ops::pairwise_l2_matrix_gpu(positions, n_agents, dim, dev),
            || {
                (0..n_agents)
                    .flat_map(|i| {
                        let pos_i = &positions[i * dim..(i + 1) * dim];
                        (i + 1..n_agents).map(move |j| {
                            let pos_j = &positions[j * dim..(j + 1) * dim];
                            pos_i
                                .iter()
                                .zip(pos_j)
                                .map(|(&a, &b)| (a - b) * (a - b))
                                .sum::<f64>()
                                .sqrt()
                        })
                    })
                    .collect()
            },
        );
        let mut adj = vec![0.0; n_agents * n_agents];
        for ((i, j), &dist) in (0..n_agents)
            .flat_map(|i| (i + 1..n_agents).map(move |j| (i, j)))
            .zip(upper_tri.iter())
        {
            if dist < comm_range && dist > LOG_GUARD {
                let weight = dist.recip();
                adj[i * n_agents + j] = weight;
                adj[j * n_agents + i] = weight;
            }
        }
        adj
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::expect_used, clippy::suboptimal_flops)]
mod tests {
    use crate::gpu_dispatch::Dispatcher;
    use crate::rng::Rng;
    use crate::tolerances;

    fn make_dispatcher() -> Dispatcher {
        // Tokio runtime creation is genuinely fatal for GPU device init: without it we cannot
        // run async Dispatcher::new(). .expect() is intentional — test harness cannot recover.
        tokio::runtime::Runtime::new()
            .expect("tokio runtime creation failed — required for async test harness")
            .block_on(async { Dispatcher::new().await })
    }

    #[test]
    fn weight_spectral_analysis_finite() {
        let d = make_dispatcher();
        let mut rng = Rng::new(42);
        let n = 8;
        let w: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();
        let result = d.weight_spectral_analysis(&w, n, n);
        assert!(result.eigenvalues.iter().all(|e| e.is_finite()));
        assert!(result.mean_ipr > 0.0);
        assert!(result.bandwidth > 0.0);
        assert!(result.condition_number >= 1.0);
    }

    #[test]
    fn numerical_hessian_quadratic() {
        let d = make_dispatcher();
        let f = |x: &[f64]| x[0] * x[0] + 2.0 * x[1] * x[1];
        let hess = d.numerical_hessian(f, &[1.0, 1.0], 1e-4);
        assert_eq!(hess.len(), 4);
        assert!((hess[0] - 2.0).abs() < 0.01, "d²f/dx² ≈ 2");
        assert!((hess[3] - 4.0).abs() < 0.01, "d²f/dy² ≈ 4");
        assert!(hess[1].abs() < 0.01, "cross term ≈ 0");
    }

    #[test]
    fn landscape_analysis_convex() {
        let d = make_dispatcher();
        let f = |x: &[f64]| -> f64 { x.iter().map(|&v| v * v).sum() };
        let result = d.landscape_analysis(&f, &[0.5, -0.3], 1e-4, 1.0);
        assert!(result.loss > 0.0);
        assert!(
            result.hessian_eigenvalues.iter().all(|&e| e > 0.0),
            "convex quadratic should have all positive eigenvalues"
        );
        assert_eq!(result.saddle_index, 0);
    }

    #[test]
    fn belief_propagation_preserves_normalization() {
        let d = make_dispatcher();
        let input = vec![0.5, 0.3, 0.2];
        let trans = vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.2, 0.6];
        let distributions = d.belief_propagation(&input, &[&trans], &[3]);
        assert_eq!(distributions.len(), 2);
        let sum: f64 = distributions[1].iter().sum();
        assert!(
            (sum - 1.0).abs() < tolerances::CROSS_LANGUAGE,
            "output distribution should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn attention_spectral_result_finite() {
        let d = make_dispatcher();
        let n = 4;
        let mut rng = Rng::new(42);
        let attn: Vec<f64> = (0..n * n).map(|_| rng.uniform()).collect();
        let result = d.attention_spectral_analysis(&attn, n);
        assert_eq!(result.eigenvalues.len(), n);
        assert!(result.eigenvalues.iter().all(|e| e.is_finite()));
        assert!(result.mean_ipr > 0.0);
        assert!(result.level_spacing_ratio.is_finite());
    }

    #[test]
    fn mlp_signal_propagation_correct_length() {
        let d = make_dispatcher();
        let input = vec![1.0, 2.0, 3.0];
        let mut rng = Rng::new(42);
        let w1: Vec<f64> = (0..4 * 3).map(|_| rng.normal() * 0.1).collect();
        let w2: Vec<f64> = (0..2 * 4).map(|_| rng.normal() * 0.1).collect();
        let variances = d.mlp_signal_propagation(&input, &[&w1, &w2], &[4, 2]);
        assert_eq!(variances.len(), 3, "input + 2 layers = 3 variance values");
        assert!(variances.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn agent_interaction_graph_symmetric() {
        let d = make_dispatcher();
        let positions = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0];
        let n = 4;
        let adj = d.agent_interaction_graph(&positions, n, 2, 2.0);
        assert_eq!(adj.len(), n * n);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (adj[i * n + j] - adj[j * n + i]).abs() < tolerances::ZERO_DETECTION,
                    "adjacency not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn agent_interaction_graph_zero_diagonal() {
        let d = make_dispatcher();
        let positions = vec![0.0, 0.0, 1.0, 1.0];
        let n = 2;
        let adj = d.agent_interaction_graph(&positions, n, 2, 10.0);
        assert!(
            (adj[0]).abs() < tolerances::ZERO_DETECTION,
            "diagonal should be 0"
        );
        assert!(
            (adj[3]).abs() < tolerances::ZERO_DETECTION,
            "diagonal should be 0"
        );
    }
}
