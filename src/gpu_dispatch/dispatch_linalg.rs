// SPDX-License-Identifier: AGPL-3.0-or-later

//! Linear algebra dispatch operations.

use super::Dispatcher;

impl Dispatcher {
    /// Matrix multiply: delegates to upstream `barracuda::dispatch::matmul_dispatch`.
    #[must_use]
    pub fn mat_mul(&self, a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        barracuda::dispatch::matmul_dispatch(a, b, n, n, n, self.wgpu_device()).unwrap_or_else(
            |e| {
                log::warn!("mat_mul upstream failed: {e}");
                crate::spectral_commutativity::mat_mul(a, b, n)
            },
        )
    }

    /// Frobenius norm: delegates to upstream `barracuda::dispatch::frobenius_norm_dispatch`.
    #[must_use]
    pub fn frobenius_norm(&self, a: &[f64]) -> f64 {
        barracuda::dispatch::frobenius_norm_dispatch(a, self.wgpu_device()).unwrap_or_else(|e| {
            log::warn!("frobenius_norm upstream failed: {e}");
            crate::spectral_commutativity::frobenius_norm(a)
        })
    }

    /// Transpose: delegates to upstream `barracuda::dispatch::transpose_dispatch`.
    #[must_use]
    pub fn transpose(&self, a: &[f64], n: usize) -> Vec<f64> {
        barracuda::dispatch::transpose_dispatch(a, n, n, self.wgpu_device()).unwrap_or_else(|e| {
            log::warn!("transpose upstream failed: {e}");
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
}
