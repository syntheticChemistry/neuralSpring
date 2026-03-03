// SPDX-License-Identifier: AGPL-3.0-or-later

//! Activation / distribution dispatch operations.

use super::Dispatcher;

impl Dispatcher {
    /// Softmax (global): delegates to upstream `barracuda::dispatch::softmax_dispatch`.
    #[must_use]
    pub fn softmax(&self, x: &[f64]) -> Vec<f64> {
        barracuda::dispatch::softmax_dispatch(x, self.wgpu_device()).unwrap_or_else(|e| {
            log::warn!("softmax upstream failed: {e}");
            crate::transformer::softmax(x)
        })
    }

    /// Row-wise softmax: uses upstream `Tensor::softmax_dim(1)`.
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
    #[must_use]
    pub fn gelu(&self, x: &[f64]) -> Vec<f64> {
        barracuda::dispatch::gelu_dispatch(x, self.wgpu_device()).unwrap_or_else(|e| {
            log::warn!("gelu upstream failed: {e}");
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
}
