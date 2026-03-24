// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reductions / statistics dispatch operations.

#![expect(
    clippy::cast_precision_loss,
    reason = "slice lengths → f64 for mean/variance in reduction dispatch"
)]

use super::Dispatcher;
use super::cpu_fallback;

impl Dispatcher {
    /// L2 distance: delegates to upstream `barracuda::dispatch::l2_distance_dispatch`.
    #[must_use]
    pub fn l2_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        barracuda::dispatch::l2_distance_dispatch(a, b, self.wgpu_device()).unwrap_or_else(|e| {
            log::warn!("l2_distance upstream failed: {e}");
            crate::modes::l2_distance(a, b)
        })
    }

    /// Shannon entropy: GPU if available, CPU fallback.
    #[must_use]
    pub fn shannon_entropy(&self, p: &[f64]) -> f64 {
        self.gpu_or_cpu(
            "shannon_entropy",
            |dev| crate::gpu_ops::shannon_entropy_gpu(p, dev).map_err(|e| e.to_string()),
            || crate::primitives::shannon_entropy(p),
        )
    }

    /// Mean: delegates to upstream `barracuda::dispatch::mean_dispatch`.
    #[must_use]
    pub fn mean(&self, data: &[f64]) -> f64 {
        barracuda::dispatch::mean_dispatch(data, self.wgpu_device()).unwrap_or_else(|e| {
            log::warn!("mean upstream failed: {e}");
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
            log::warn!("variance upstream failed: {e}");
            cpu_fallback::variance(data)
        })
    }

    /// Pearson correlation: GPU if available, CPU fallback.
    #[must_use]
    pub fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        self.gpu_or_cpu(
            "pearson_correlation",
            |dev| crate::gpu_ops::pearson_correlation_gpu(x, y, dev).map_err(|e| e.to_string()),
            || cpu_fallback::pearson(x, y),
        )
    }

    /// Chi-squared statistic: GPU if available, CPU fallback.
    #[must_use]
    pub fn chi_squared(&self, observed: &[f64], expected: &[f64]) -> f64 {
        self.gpu_or_cpu(
            "chi_squared",
            |dev| {
                crate::gpu_ops::chi_squared_gpu(observed, expected, dev).map_err(|e| e.to_string())
            },
            || cpu_fallback::chi_squared(observed, expected),
        )
    }

    /// KL divergence: GPU fused f64 if available, CPU fallback.
    ///
    /// Cross-spring evolution: neuralSpring `kl_divergence_f64.wgsl`
    /// → `BarraCUDA` (via `ToadStool` S76) → `FusedKlDivergenceGpu`.
    #[must_use]
    pub fn kl_divergence(&self, p: &[f64], q: &[f64]) -> f64 {
        self.gpu_or_cpu(
            "kl_divergence",
            |dev| crate::gpu_ops::kl_divergence_gpu(p, q, dev).map_err(|e| e.to_string()),
            || cpu_fallback::kl_divergence(p, q),
        )
    }
}
