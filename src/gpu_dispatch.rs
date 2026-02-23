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

    /// Commutator `[A,B]` = AB - BA: GPU if available, CPU fallback.
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

    // ═══════════════════════════════════════════════════════════════
    // Eigensolvers (Session 47 — GPU promotion)
    // ═══════════════════════════════════════════════════════════════

    /// Symmetric eigenvalue decomposition: GPU (`BatchedEighGpu`) if available.
    #[must_use]
    pub fn eigh(&self, a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::eigh_gpu(a, n, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] eigh GPU failed: {e}"),
            }
        }
        let r = crate::eigh::eigh_householder_qr(a, n);
        (r.eigenvalues, r.eigenvectors)
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

    /// Spectrum chi-squared with GPU dispatch (pangenome selection).
    #[must_use]
    pub fn spectrum_chi_squared(&self, observed: &[f64], expected_frac: &[f64]) -> f64 {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::spectrum_chi_squared_gpu(observed, expected_frac, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] spectrum_chi_squared GPU failed: {e}"),
            }
        }
        crate::pangenome_selection::spectrum_chi_squared(observed, expected_frac)
    }

    /// Selection coefficient with GPU dispatch (pangenome selection).
    #[must_use]
    pub fn selection_coefficient(&self, observed: &[f64], neutral: &[f64]) -> f64 {
        if let Some(dev) = self.wgpu_device() {
            match crate::gpu_ops::selection_coefficient_gpu(observed, neutral, dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] selection_coefficient GPU failed: {e}"),
            }
        }
        crate::pangenome_selection::selection_coefficient(observed, neutral)
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

    fn cpu() -> Dispatcher {
        Dispatcher::cpu_only()
    }

    // ── Metadata ────────────────────────────────────────────────

    #[test]
    fn cpu_only_no_gpu() {
        let d = cpu();
        assert!(!d.has_gpu());
        assert_eq!(d.backend(), Backend::Cpu);
        assert!(d.capabilities().is_none());
        assert_eq!(d.adapter_name(), "(none)");
        assert!(d.wgpu_device().is_none());
        assert!(d.gpu().is_none());
    }

    #[test]
    fn backend_display() {
        assert_eq!(format!("{}", Backend::Gpu), "GPU");
        assert_eq!(format!("{}", Backend::Cpu), "CPU");
    }

    // ── Linear algebra ──────────────────────────────────────────

    #[test]
    fn cpu_mat_mul_identity() {
        let d = cpu();
        #[rustfmt::skip]
        let eye = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let result = d.mat_mul(&eye, &eye, 3);
        for (i, &v) in result.iter().enumerate() {
            let expected = if i / 3 == i % 3 { 1.0 } else { 0.0 };
            assert!((v - expected).abs() < 1e-15, "mat_mul identity [{i}]");
        }
    }

    #[test]
    fn cpu_frobenius_norm() {
        let d = cpu();
        let a = vec![3.0, 4.0];
        assert!((d.frobenius_norm(&a) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn cpu_transpose() {
        let d = cpu();
        #[rustfmt::skip]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let t = d.transpose(&a, 2);
        assert!((t[0] - 1.0).abs() < 1e-15);
        assert!((t[1] - 3.0).abs() < 1e-15);
        assert!((t[2] - 2.0).abs() < 1e-15);
        assert!((t[3] - 4.0).abs() < 1e-15);
    }

    #[test]
    fn cpu_distance_to_normal() {
        let d = cpu();
        #[rustfmt::skip]
        let sym = vec![
            2.0, 1.0,
            1.0, 2.0,
        ];
        let dist = d.distance_to_normal(&sym, 2);
        assert!(
            dist < 1e-12,
            "symmetric matrix should commute with transpose"
        );
    }

    #[test]
    fn cpu_commutator_symmetric_zero() {
        let d = cpu();
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let comm = d.commutator(&a, &a, 2);
        for &v in &comm {
            assert!(v.abs() < 1e-15, "A commutes with itself");
        }
    }

    // ── Activations / distributions ─────────────────────────────

    #[test]
    fn cpu_softmax_sums_to_one() {
        let d = cpu();
        let result = d.softmax(&[1.0, 2.0, 3.0]);
        let total: f64 = result.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(result[2] > result[1] && result[1] > result[0]);
    }

    #[test]
    fn cpu_boltzmann_sums_to_one() {
        let d = cpu();
        let result = d.boltzmann(&[1.0, 2.0, 3.0], 1.0);
        let total: f64 = result.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    // ── Reductions / statistics ─────────────────────────────────

    #[test]
    fn cpu_l2_distance() {
        let d = cpu();
        let dist = d.l2_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((dist - 5.0).abs() < 1e-12);
    }

    #[test]
    fn cpu_shannon_entropy() {
        let d = cpu();
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let h = d.shannon_entropy(&p);
        let expected = 4.0_f64.ln();
        assert!((h - expected).abs() < 1e-10);
    }

    #[test]
    fn cpu_mean() {
        let d = cpu();
        assert!((d.mean(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-15);
        assert!((d.mean(&[]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn cpu_variance() {
        let d = cpu();
        let v = d.variance(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((v - 4.0).abs() < 1e-12);
        assert!((d.variance(&[]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn cpu_pearson_correlation() {
        let d = cpu();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = d.pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-12, "perfect positive correlation");
    }

    #[test]
    fn cpu_pearson_short() {
        let d = cpu();
        assert!((d.pearson_correlation(&[1.0], &[2.0]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn cpu_pearson_zero_variance() {
        let d = cpu();
        let r = d.pearson_correlation(&[3.0, 3.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((r - 0.0).abs() < 1e-15);
    }

    #[test]
    fn cpu_chi_squared() {
        let d = cpu();
        let obs = vec![10.0, 20.0, 30.0];
        let exp = vec![20.0, 20.0, 20.0];
        let chi2 = d.chi_squared(&obs, &exp);
        assert!((chi2 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn cpu_chi_squared_zero_expected() {
        let d = cpu();
        let chi2 = d.chi_squared(&[5.0], &[0.0]);
        assert!((chi2 - 0.0).abs() < 1e-15, "zero expected → 0 contribution");
    }

    // ── HMM ─────────────────────────────────────────────────────

    #[test]
    fn cpu_hmm_backward_step_basic() {
        let d = cpu();
        let beta_next = vec![1.0, 1.0];
        #[rustfmt::skip]
        let trans = vec![0.7, 0.3, 0.4, 0.6];
        let emit = vec![0.5, 0.5];
        let result = d.hmm_backward_step(&beta_next, &trans, &emit, 1.0, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 0.5).abs() < 1e-12);
        assert!((result[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn cpu_hmm_backward_step_zero_scale() {
        let d = cpu();
        let result = d.hmm_backward_step(&[1.0], &[1.0], &[1.0], 0.0, 1);
        assert!(result[0].is_finite(), "zero scale should use guard");
    }

    #[test]
    fn cpu_hmm_viterbi_step() {
        let d = cpu();
        let delta_prev = vec![0.0_f64.ln(), (-1.0_f64).exp().ln()];
        #[rustfmt::skip]
        let log_trans = vec![
            0.7_f64.ln(), 0.3_f64.ln(),
            0.4_f64.ln(), 0.6_f64.ln(),
        ];
        let log_emit = vec![0.6_f64.ln(), 0.4_f64.ln()];
        let (delta, psi) = d.hmm_viterbi_step(&delta_prev, &log_trans, &log_emit, 2);
        assert_eq!(delta.len(), 2);
        assert_eq!(psi.len(), 2);
    }

    // ── Population genetics ─────────────────────────────────────

    #[test]
    fn cpu_allele_frequencies() {
        let d = cpu();
        // 2 individuals × 2 loci, diploid division: sum / (2*n_ind)
        let pop = vec![2.0, 0.0, 0.0, 2.0];
        let freq = d.allele_frequencies(&pop, 2, 2);
        assert_eq!(freq.len(), 2);
        assert!((freq[0] - 0.5).abs() < 1e-12);
        assert!((freq[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn cpu_nucleotide_diversity() {
        let d = cpu();
        let pop = vec![0.0, 1.0, 1.0, 0.0];
        let pi = d.nucleotide_diversity(&pop, 2, 2);
        assert!(pi >= 0.0);
    }

    #[test]
    fn cpu_matrix_correlation() {
        let d = cpu();
        #[rustfmt::skip]
        let a = vec![
            0.0, 1.0, 2.0,
            1.0, 0.0, 3.0,
            2.0, 3.0, 0.0,
        ];
        let r = d.matrix_correlation(&a, &a, 3);
        assert!((r - 1.0).abs() < 1e-10, "self-correlation = 1.0");
    }

    #[test]
    fn cpu_geographic_distances() {
        let d = cpu();
        let coords = vec![(0.0, 0.0), (3.0, 4.0)];
        let dist = d.geographic_distances(&coords);
        assert_eq!(dist.len(), 4);
        assert!((dist[0] - 0.0).abs() < 1e-12);
        assert!((dist[1] - 5.0).abs() < 1e-12);
        assert!((dist[3] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn cpu_thermal_diversity_correlation() {
        let d = cpu();
        let r = d.thermal_diversity_correlation(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]);
        assert!((r - 1.0).abs() < 1e-10, "perfect linear → r≈1");
    }

    #[test]
    fn cpu_thermal_diversity_short() {
        let d = cpu();
        let r = d.thermal_diversity_correlation(&[1.0], &[10.0]);
        assert!((r - 0.0).abs() < 1e-15, "n<2 → 0");
    }

    // ── Game theory ─────────────────────────────────────────────

    #[test]
    fn cpu_replicator_step_preserves_simplex() {
        let d = cpu();
        let freq = [0.6, 0.4];
        let payoff = [[3.0, 0.0], [5.0, 1.0]];
        let next = d.replicator_step(&freq, &payoff, 0.01);
        let sum: f64 = next.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "frequencies sum to 1");
        assert!(next[0] >= 0.0 && next[1] >= 0.0, "non-negative");
    }

    // ── Regulatory ──────────────────────────────────────────────

    #[test]
    fn cpu_hill_activation_batch() {
        let d = cpu();
        let result = d.hill_activation_batch(&[0.0, 1.0, 10.0], 1.0, 1.0, 2.0);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.0).abs() < 1e-10, "hill(0)≈0");
        assert!((result[1] - 0.5).abs() < 0.01, "hill(k)≈Vmax/2");
        assert!(result[2] > 0.9, "hill(10k)≈Vmax");
    }

    // ── Eigensolvers ────────────────────────────────────────────

    #[test]
    fn cpu_eigh_diagonal() {
        let d = cpu();
        let a = vec![2.0, 0.0, 0.0, 3.0];
        let (vals, _vecs) = d.eigh(&a, 2);
        let mut sorted = vals;
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in eigenvalues"));
        assert!((sorted[0] - 2.0).abs() < 1e-10);
        assert!((sorted[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn cpu_disorder_sweep_no_gpu() {
        let d = cpu();
        assert!(d.disorder_sweep(&[1.0, 0.0, 0.0, 1.0], 2, 1).is_none());
    }

    // ── Pangenome selection ─────────────────────────────────────

    #[test]
    fn cpu_spectrum_chi_squared() {
        let d = cpu();
        let obs = vec![10.0, 20.0, 30.0];
        let frac = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let chi2 = d.spectrum_chi_squared(&obs, &frac);
        assert!(chi2 >= 0.0);
    }

    #[test]
    fn cpu_selection_coefficient() {
        let d = cpu();
        let obs = vec![10.0, 20.0, 30.0];
        let neutral = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let s = d.selection_coefficient(&obs, &neutral);
        assert!(s.is_finite());
    }
}
