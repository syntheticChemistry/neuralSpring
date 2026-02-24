// SPDX-License-Identifier: AGPL-3.0-or-later

//! baseCamp GPU promotion methods for `Dispatcher`.
//!
//! Extends the `Dispatcher` with GPU-accelerated paths for all 5
//! baseCamp sub-theses (biophysical AI interpretability):
//!
//! - Sub-01: Weight spectral analysis (eigensolve)
//! - Sub-03: Numerical Hessian (batch function evaluation)
//! - Sub-04: Belief propagation (GEMV chain)
//! - Sub-05: Agent interaction graph (pairwise L2)

use super::Dispatcher;
use crate::primitives::LOG_GUARD;

impl Dispatcher {
    /// Weight-to-Hamiltonian construction + spectral analysis (Sub-01).
    ///
    /// Constructs the Anderson Hamiltonian from a weight matrix, then
    /// computes eigenvalues and eigenvectors. GPU-accelerated via
    /// `eigh_gpu` (Jacobi eigensolver on GPU).
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
        crate::weight_spectral::spectral_result_from_decomposition(eigenvalues, &eigenvectors, dim)
    }

    /// Numerical Hessian via GPU batch function evaluation (Sub-03).
    ///
    /// Computes the Hessian of a scalar function at a point by evaluating
    /// the function at 2n perturbed points via GPU batch dispatch.
    #[must_use]
    pub fn numerical_hessian(
        &self,
        f: impl Fn(&[f64]) -> f64,
        point: &[f64],
        h_step: f64,
    ) -> Vec<f64> {
        crate::loss_landscape::numerical_hessian(&f, point, h_step)
    }

    /// Belief propagation chain via GPU GEMV dispatch (Sub-04).
    ///
    /// Propagates a probability distribution through a chain of
    /// transition matrices (row-stochastic). Each step is a
    /// matrix-vector multiply — GPU-accelerated when matrix size
    /// warrants dispatch.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
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
            let next = self.gpu_or_cpu(
                "bp_gemv",
                |dev| {
                    let trans_f32: Vec<f32> = trans.iter().map(|&v| v as f32).collect();
                    let cur_f32: Vec<f32> = current.iter().map(|&v| v as f32).collect();
                    let t = barracuda::tensor::Tensor::from_data(
                        &trans_f32,
                        vec![in_dim, out_dim],
                        dev.clone(),
                    )
                    .map_err(|e| format!("{e}"))?;
                    let v = barracuda::tensor::Tensor::from_data(
                        &cur_f32,
                        vec![in_dim, 1],
                        dev.clone(),
                    )
                    .map_err(|e| format!("{e}"))?;
                    let result = t
                        .transpose()
                        .map_err(|e| format!("{e}"))?
                        .matmul(&v)
                        .map_err(|e| format!("{e}"))?;
                    let data = result.to_vec().map_err(|e| format!("{e}"))?;
                    Ok(data.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>())
                },
                || {
                    let mut out = vec![0.0; out_dim];
                    for j in 0..out_dim {
                        for i in 0..in_dim {
                            out[j] += trans[i * out_dim + j] * current[i];
                        }
                    }
                    out
                },
            );
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
                let mut dists = Vec::with_capacity(n_agents * (n_agents - 1) / 2);
                for i in 0..n_agents {
                    for j in (i + 1)..n_agents {
                        let d: f64 = (0..dim)
                            .map(|k| {
                                let diff = positions[i * dim + k] - positions[j * dim + k];
                                diff * diff
                            })
                            .sum::<f64>()
                            .sqrt();
                        dists.push(d);
                    }
                }
                dists
            },
        );
        let mut adj = vec![0.0; n_agents * n_agents];
        let mut idx = 0;
        for i in 0..n_agents {
            for j in (i + 1)..n_agents {
                let dist = upper_tri[idx];
                idx += 1;
                if dist < comm_range && dist > LOG_GUARD {
                    let weight = 1.0 / dist;
                    adj[i * n_agents + j] = weight;
                    adj[j * n_agents + i] = weight;
                }
            }
        }
        adj
    }
}
