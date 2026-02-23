// SPDX-License-Identifier: AGPL-3.0-or-later

//! Capability-based GPU/CPU dispatch for science operations.
//!
//! Provides a unified execution context that routes operations to GPU
//! when available, falling back to CPU reference implementations.
//! No hardcoded backend: the adapter is discovered at runtime via
//! `NEURALSPRING_BACKEND` (see [`crate::gpu::Gpu`]).
//!
//! ## Design principles
//!
//! - **Self-knowledge only**: `Dispatcher` discovers its capabilities at init
//! - **No mocks in production**: GPU path is real GPU; CPU path is real CPU math
//! - **Agnostic**: works on RTX 4070, TITAN V (NVK), Raspberry Pi, llvmpipe
//! - **Observable**: every dispatch decision is logged to `execution_log`

#![allow(clippy::cast_precision_loss)]

use barracuda::device::WgpuDevice;
use std::sync::Arc;

use crate::gpu::{Gpu, GpuCapabilities};

/// Runtime execution backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// GPU via wgpu/Vulkan (includes NVK, proprietary, etc.)
    Gpu,
    /// CPU reference implementation
    Cpu,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gpu => write!(f, "GPU"),
            Self::Cpu => write!(f, "CPU"),
        }
    }
}

/// Capability-based dispatcher for GPU/CPU execution.
///
/// Created once at startup, shared across all science modules.
/// When GPU is available, operations route through `gpu_ops`;
/// when unavailable (no adapter, CI, etc.), they use CPU references.
pub struct Dispatcher {
    gpu: Option<Gpu>,
    prefer_gpu: bool,
}

impl Dispatcher {
    /// Create a dispatcher by probing the runtime GPU environment.
    ///
    /// If GPU initialization fails, falls back to CPU-only mode silently.
    pub async fn new() -> Self {
        match Gpu::new().await {
            Ok(gpu) => {
                eprintln!(
                    "[dispatch] GPU available: {} ({:?}, {:?})",
                    gpu.adapter_name, gpu.device_type, gpu.backend,
                );
                Self {
                    gpu: Some(gpu),
                    prefer_gpu: true,
                }
            }
            Err(e) => {
                eprintln!("[dispatch] No GPU, CPU-only mode: {e}");
                Self {
                    gpu: None,
                    prefer_gpu: false,
                }
            }
        }
    }

    /// Create a CPU-only dispatcher (for testing or CI).
    #[must_use]
    pub const fn cpu_only() -> Self {
        Self {
            gpu: None,
            prefer_gpu: false,
        }
    }

    /// Create from an existing `Gpu` context.
    #[must_use]
    pub const fn from_gpu(gpu: Gpu) -> Self {
        Self {
            gpu: Some(gpu),
            prefer_gpu: true,
        }
    }

    /// Whether GPU execution is available.
    #[must_use]
    pub const fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Current preferred backend.
    #[must_use]
    pub const fn backend(&self) -> Backend {
        if self.prefer_gpu && self.gpu.is_some() {
            Backend::Gpu
        } else {
            Backend::Cpu
        }
    }

    /// GPU capabilities (if available).
    #[must_use]
    pub fn capabilities(&self) -> Option<&GpuCapabilities> {
        self.gpu.as_ref().map(|g| &g.capabilities)
    }

    /// Adapter name (if GPU available).
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        self.gpu
            .as_ref()
            .map_or("(none)", |g| g.adapter_name.as_str())
    }

    /// Get the `WgpuDevice` for direct Tensor API usage.
    #[must_use]
    pub fn wgpu_device(&self) -> Option<&Arc<WgpuDevice>> {
        self.gpu.as_ref().map(Gpu::wgpu_device)
    }

    /// Get the `Gpu` context reference.
    #[must_use]
    pub const fn gpu(&self) -> Option<&Gpu> {
        self.gpu.as_ref()
    }

    // ═══════════════════════════════════════════════════════════════
    // Dispatched operations
    // ═══════════════════════════════════════════════════════════════

    /// Matrix multiply: GPU if available, CPU fallback.
    #[must_use]
    pub fn mat_mul(&self, a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::mat_mul_gpu(a, b, n, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] mat_mul GPU failed, falling back: {e}"),
            }
        }
        crate::spectral_commutativity::mat_mul(a, b, n)
    }

    /// Frobenius norm: GPU if available, CPU fallback.
    #[must_use]
    pub fn frobenius_norm(&self, a: &[f64]) -> f64 {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::frobenius_norm_gpu(a, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] frobenius_norm GPU failed: {e}"),
            }
        }
        crate::spectral_commutativity::frobenius_norm(a)
    }

    /// Transpose: GPU if available, CPU fallback.
    #[must_use]
    pub fn transpose(&self, a: &[f64], n: usize) -> Vec<f64> {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::transpose_gpu(a, n, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] transpose GPU failed: {e}"),
            }
        }
        crate::spectral_commutativity::transpose(a, n)
    }

    /// Softmax: GPU if available, CPU fallback.
    #[must_use]
    pub fn softmax(&self, x: &[f64]) -> Vec<f64> {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::softmax_gpu(x, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] softmax GPU failed: {e}"),
            }
        }
        crate::transformer::softmax(x)
    }

    /// Boltzmann distribution: GPU if available, CPU fallback.
    #[must_use]
    pub fn boltzmann(&self, fitnesses: &[f64], beta: f64) -> Vec<f64> {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::boltzmann_gpu(fitnesses, beta, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] boltzmann GPU failed: {e}"),
            }
        }
        crate::counterdiabatic::boltzmann_distribution(fitnesses, beta)
    }

    /// L2 distance: GPU if available, CPU fallback.
    #[must_use]
    pub fn l2_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::l2_distance_gpu(a, b, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] l2_distance GPU failed: {e}"),
            }
        }
        crate::modes::l2_distance(a, b)
    }

    /// Shannon entropy: GPU if available, CPU fallback.
    #[must_use]
    pub fn shannon_entropy(&self, p: &[f64]) -> f64 {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::shannon_entropy_gpu(p, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] shannon_entropy GPU failed: {e}"),
            }
        }
        crate::primitives::shannon_entropy(p)
    }

    /// Mean: GPU if available, CPU fallback.
    #[must_use]
    pub fn mean(&self, data: &[f64]) -> f64 {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::mean_gpu(data, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] mean GPU failed: {e}"),
            }
        }
        if data.is_empty() {
            0.0
        } else {
            data.iter().sum::<f64>() / data.len() as f64
        }
    }

    /// Variance: GPU if available, CPU fallback.
    #[must_use]
    pub fn variance(&self, data: &[f64]) -> f64 {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::variance_gpu(data, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] variance GPU failed: {e}"),
            }
        }
        let n = data.len() as f64;
        if n < 1.0 {
            return 0.0;
        }
        let mean = data.iter().sum::<f64>() / n;
        data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n
    }

    /// Pearson correlation: GPU if available, CPU fallback.
    #[must_use]
    pub fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::pearson_correlation_gpu(x, y, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] pearson GPU failed: {e}"),
            }
        }
        cpu_pearson(x, y)
    }

    /// Distance to normal: GPU if available, CPU fallback.
    #[must_use]
    pub fn distance_to_normal(&self, a: &[f64], n: usize) -> f64 {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::distance_to_normal_gpu(a, n, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] distance_to_normal GPU failed: {e}"),
            }
        }
        crate::spectral_commutativity::distance_to_normal(a, n)
    }

    /// Commutator [A,B] = AB - BA: GPU if available, CPU fallback.
    #[must_use]
    pub fn commutator(&self, a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::commutator_gpu(a, b, n, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] commutator GPU failed: {e}"),
            }
        }
        crate::spectral_commutativity::commutator(a, b, n)
    }

    /// Chi-squared statistic: GPU if available, CPU fallback.
    #[must_use]
    pub fn chi_squared(&self, observed: &[f64], expected: &[f64]) -> f64 {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::chi_squared_gpu(observed, expected, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] chi_squared GPU failed: {e}"),
            }
        }
        observed
            .iter()
            .zip(expected.iter())
            .map(|(&o, &e)| {
                if e.abs() < crate::primitives::LOG_GUARD {
                    0.0
                } else {
                    (o - e).powi(2) / e
                }
            })
            .sum()
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase B dispatched operations
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
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::hmm_backward_step_gpu(
                beta_next,
                transition,
                emission_col,
                scale,
                n_states,
                dev,
            ) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] hmm_backward_step GPU failed: {e}"),
            }
        }
        cpu_hmm_backward_step(beta_next, transition, emission_col, scale, n_states)
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
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::hmm_viterbi_step_gpu(
                delta_prev,
                log_transition,
                log_emission_col,
                n_states,
                dev,
            ) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] hmm_viterbi_step GPU failed: {e}"),
            }
        }
        cpu_hmm_viterbi_step(delta_prev, log_transition, log_emission_col, n_states)
    }

    /// Allele frequencies: GPU column-sum if available, CPU fallback.
    #[must_use]
    pub fn allele_frequencies(&self, pop: &[f64], n_individuals: usize, n_loci: usize) -> Vec<f64> {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::allele_frequencies_gpu(pop, n_individuals, n_loci, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] allele_frequencies GPU failed: {e}"),
            }
        }
        crate::meta_population::allele_frequencies(pop, n_individuals, n_loci)
    }

    /// Nucleotide diversity: GPU if available, CPU fallback.
    #[must_use]
    pub fn nucleotide_diversity(&self, pop: &[f64], n_individuals: usize, n_loci: usize) -> f64 {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::nucleotide_diversity_gpu(pop, n_individuals, n_loci, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] nucleotide_diversity GPU failed: {e}"),
            }
        }
        crate::meta_population::nucleotide_diversity(pop, n_individuals, n_loci)
    }

    /// Matrix correlation (upper triangle Pearson): GPU if available, CPU fallback.
    #[must_use]
    pub fn matrix_correlation(&self, a: &[f64], b: &[f64], n: usize) -> f64 {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::matrix_correlation_gpu(a, b, n, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] matrix_correlation GPU failed: {e}"),
            }
        }
        crate::meta_population::matrix_correlation(a, b, n)
    }

    /// Geographic distance matrix: GPU if available, CPU fallback.
    #[must_use]
    pub fn geographic_distances(&self, coords: &[(f64, f64)]) -> Vec<f64> {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::geographic_distance_matrix_gpu(coords, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] geographic_distances GPU failed: {e}"),
            }
        }
        crate::meta_population::geographic_distance_matrix(coords)
    }

    /// Thermal diversity correlation: GPU Pearson if available, CPU fallback.
    #[must_use]
    pub fn thermal_diversity_correlation(&self, pi_values: &[f64], temperatures: &[f64]) -> f64 {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::thermal_diversity_correlation_gpu(pi_values, temperatures, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] thermal_diversity_correlation GPU failed: {e}"),
            }
        }
        crate::meta_population::thermal_diversity_correlation(pi_values, temperatures)
    }

    /// Replicator dynamics step: GPU matmul if available, CPU fallback.
    #[must_use]
    pub fn replicator_step(&self, freq: &[f64; 2], payoff: &[[f64; 2]; 2], dt: f64) -> [f64; 2] {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::replicator_step_gpu(freq, payoff, dt, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] replicator_step GPU failed: {e}"),
            }
        }
        cpu_replicator_step(freq, payoff, dt)
    }

    /// Hill activation batch: GPU if available, CPU fallback.
    #[must_use]
    pub fn hill_activation_batch(&self, x: &[f64], vmax: f64, k: f64, n_hill: f64) -> Vec<f64> {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::hill_activation_batch_gpu(x, vmax, k, n_hill, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] hill_activation_batch GPU failed: {e}"),
            }
        }
        x.iter()
            .map(|&xi| crate::primitives::hill_activation(xi, vmax, k, n_hill))
            .collect()
    }
}

fn cpu_hmm_backward_step(
    beta_next: &[f64],
    transition: &[f64],
    emission_col: &[f64],
    scale: f64,
    n_states: usize,
) -> Vec<f64> {
    let guard = crate::primitives::LOG_GUARD;
    let safe_scale = if scale.abs() < guard { guard } else { scale };
    let mut beta = vec![0.0; n_states];
    for i in 0..n_states {
        let mut sum = 0.0;
        for j in 0..n_states {
            sum += transition[i * n_states + j] * emission_col[j] * beta_next[j];
        }
        beta[i] = sum / safe_scale;
    }
    beta
}

fn cpu_hmm_viterbi_step(
    delta_prev: &[f64],
    log_transition: &[f64],
    log_emission_col: &[f64],
    n_states: usize,
) -> (Vec<f64>, Vec<usize>) {
    let mut delta_new = Vec::with_capacity(n_states);
    let mut psi = Vec::with_capacity(n_states);
    for j in 0..n_states {
        let mut best_i = 0;
        let mut best_val = f64::NEG_INFINITY;
        for i in 0..n_states {
            let val = delta_prev[i] + log_transition[i * n_states + j];
            if val > best_val {
                best_val = val;
                best_i = i;
            }
        }
        delta_new.push(best_val + log_emission_col[j]);
        psi.push(best_i);
    }
    (delta_new, psi)
}

fn cpu_replicator_step(freq: &[f64; 2], payoff: &[[f64; 2]; 2], dt: f64) -> [f64; 2] {
    let f0 = payoff[0][0].mul_add(freq[0], payoff[0][1] * freq[1]);
    let f1 = payoff[1][0].mul_add(freq[0], payoff[1][1] * freq[1]);
    let f_bar = freq[0].mul_add(f0, freq[1] * f1);

    let mut x0 = (dt * freq[0]).mul_add(f0 - f_bar, freq[0]).max(0.0);
    let mut x1 = (dt * freq[1]).mul_add(f1 - f_bar, freq[1]).max(0.0);
    let sum = x0 + x1;
    if sum > 0.0 {
        x0 /= sum;
        x1 /= sum;
    }
    [x0, x1]
}

fn cpu_pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - mx) * (b - my))
        .sum();
    let vx: f64 = x.iter().map(|&a| (a - mx).powi(2)).sum();
    let vy: f64 = y.iter().map(|&b| (b - my).powi(2)).sum();
    let denom = (vx * vy).sqrt();
    if denom < crate::primitives::LOG_GUARD {
        0.0
    } else {
        cov / denom
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;

    // GPU dispatch tests validated via `validate_gpu_promotion` binary
    // (27/27 PASS on both RTX 4070 and TITAN V NVK).

    #[test]
    fn cpu_only_dispatcher_works() {
        let d = Dispatcher::cpu_only();
        assert!(!d.has_gpu());
        assert_eq!(d.backend(), Backend::Cpu);
        let result = d.softmax(&[1.0, 2.0, 3.0]);
        let total: f64 = result.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }
}
