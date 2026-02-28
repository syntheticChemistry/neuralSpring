// SPDX-License-Identifier: AGPL-3.0-or-later

//! Gate / fitness / swarm dispatch operations.

use super::Dispatcher;

impl Dispatcher {
    /// Two-input Hill gate: `f(a,b) = V_max × H(a) × H(b)` on GPU grid.
    #[must_use]
    pub fn hill_gate(
        &self,
        input_a: &[f64],
        input_b: &[f64],
        cfg: &crate::gpu_ops::HillGateConfig,
    ) -> Vec<f64> {
        self.gpu_or_cpu(
            "hill_gate",
            |dev| crate::gpu_ops::hill_gate_gpu(input_a, input_b, cfg, dev),
            || {
                input_a
                    .iter()
                    .flat_map(|&a| {
                        input_b.iter().map(move |&b| {
                            crate::signal_integration::two_input_hill(
                                a, b, cfg.vmax, cfg.k_a, cfg.k_b, cfg.n_a, cfg.n_b,
                            )
                        })
                    })
                    .collect()
            },
        )
    }

    /// Multi-objective fitness evaluation on GPU.
    #[must_use]
    pub fn multi_obj_fitness(
        &self,
        genotypes: &[f64],
        pop_size: usize,
        genome_len: usize,
        n_objectives: usize,
    ) -> Vec<f64> {
        self.gpu_or_cpu(
            "multi_obj_fitness",
            |dev| {
                crate::gpu_ops::multi_obj_fitness_gpu(
                    genotypes,
                    pop_size,
                    genome_len,
                    n_objectives,
                    dev,
                )
            },
            || {
                genotypes
                    .chunks_exact(genome_len)
                    .flat_map(|geno| {
                        crate::directed_evolution::multi_objective_fitness(geno, n_objectives)
                    })
                    .collect()
            },
        )
    }

    /// Swarm neural-network forward pass on GPU.
    #[must_use]
    pub fn swarm_nn_forward(
        &self,
        weights: &[f64],
        inputs: &[f64],
        dims: &crate::gpu_ops::SwarmNnDims,
    ) -> Vec<u32> {
        self.gpu_or_cpu(
            "swarm_nn_forward",
            |dev| crate::gpu_ops::swarm_nn_forward_gpu(weights, inputs, dims, dev),
            || {
                let weights_per = dims.input_dim * dims.hidden_dim
                    + dims.hidden_dim
                    + dims.hidden_dim * dims.output_dim
                    + dims.output_dim;
                (0..dims.n_controllers)
                    .flat_map(|c| {
                        let params = &weights[c * weights_per..(c + 1) * weights_per];
                        (0..dims.n_evals).map(move |e| {
                            let i_start = (c * dims.n_evals + e) * dims.input_dim;
                            let sense = inputs[i_start];
                            crate::swarm_robotics::neural_forward(params, sense) as u32
                        })
                    })
                    .collect()
            },
        )
    }
}
