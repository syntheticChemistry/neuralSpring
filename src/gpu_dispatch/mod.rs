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

mod basecamp;
mod cpu_fallback;
mod dispatch_ops;

use barracuda::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use barracuda::device::WgpuDevice;
use barracuda::unified_hardware::BandwidthTier;
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
/// When GPU is available, operations route through upstream
/// `barracuda::dispatch::domain_ops` (cross-spring evolved from hotSpring
/// precision shaders and wetSpring bio shaders). When unavailable,
/// local CPU references are used.
///
/// The `driver_profile` field (from `barracuda::device::GpuDriverProfile`)
/// provides hardware-adaptive f64 strategy, driver workaround detection,
/// and optimal eigensolve configuration — evolved from hotSpring's
/// core-streaming discovery work.
pub struct Dispatcher {
    gpu: Option<Gpu>,
    prefer_gpu: bool,
    driver_profile: Option<GpuDriverProfile>,
}

impl Dispatcher {
    /// Create a dispatcher by probing the runtime GPU environment.
    ///
    /// If GPU initialization fails, falls back to CPU-only mode silently.
    pub async fn new() -> Self {
        match Gpu::new().await {
            Ok(gpu) => {
                let profile = GpuDriverProfile::from_device(gpu.wgpu_device());
                let tier = BandwidthTier::detect_from_adapter_name(&gpu.adapter_name);
                eprintln!(
                    "[dispatch] GPU available: {} ({:?}, {:?}, f64={:?}, pcie={tier:?})",
                    gpu.adapter_name,
                    gpu.device_type,
                    gpu.backend,
                    profile.fp64_strategy(),
                );
                Self {
                    gpu: Some(gpu),
                    prefer_gpu: true,
                    driver_profile: Some(profile),
                }
            }
            Err(e) => {
                eprintln!("[dispatch] No GPU, CPU-only mode: {e}");
                Self {
                    gpu: None,
                    prefer_gpu: false,
                    driver_profile: None,
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
            driver_profile: None,
        }
    }

    /// Create from an existing `Gpu` context.
    #[must_use]
    pub fn from_gpu(gpu: Gpu) -> Self {
        let profile = GpuDriverProfile::from_device(gpu.wgpu_device());
        Self {
            gpu: Some(gpu),
            prefer_gpu: true,
            driver_profile: Some(profile),
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

    /// Upstream driver profile (hotSpring-evolved hardware detection).
    #[must_use]
    pub const fn driver_profile(&self) -> Option<&GpuDriverProfile> {
        self.driver_profile.as_ref()
    }

    /// Hardware-adaptive f64 strategy: `Native` on compute-class GPUs
    /// (Titan V, V100, A100 — 1:2 FP64:FP32), `Hybrid` on consumer
    /// GPUs (RTX 4070 — 1:64 ratio, routes bulk math through df64 f32-pairs).
    #[must_use]
    pub fn fp64_strategy(&self) -> Fp64Strategy {
        self.driver_profile
            .as_ref()
            .map_or(Fp64Strategy::Native, GpuDriverProfile::fp64_strategy)
    }

    /// Whether the GPU driver needs `pow(f64,f64)` polyfill workaround.
    #[must_use]
    pub fn needs_pow_workaround(&self) -> bool {
        self.driver_profile
            .as_ref()
            .is_some_and(GpuDriverProfile::needs_pow_f64_workaround)
    }

    /// `PCIe` bandwidth tier detected from the GPU adapter name.
    ///
    /// Returns `BandwidthTier::Unknown` when no GPU is available.
    /// Evolved from hotSpring's cross-device transfer cost modelling.
    #[must_use]
    pub fn bandwidth_tier(&self) -> BandwidthTier {
        self.gpu.as_ref().map_or(BandwidthTier::Unknown, |g| {
            BandwidthTier::detect_from_adapter_name(&g.adapter_name)
        })
    }

    /// Check whether a combined GPU allocation of `total_bytes` is safe.
    ///
    /// On NVK (nouveau), the kernel driver PTE-faults at ~1.4 GB combined.
    /// Returns `Ok(())` if safe or no GPU, `Err` with a diagnostic message
    /// if the allocation would exceed the driver's safe limit.
    ///
    /// # Errors
    ///
    /// Returns [`barracuda::error::BarracudaError::DeviceLimitExceeded`] when
    /// `total_bytes` exceeds the NVK safe allocation limit (~1.2 GB).
    pub fn check_allocation_safe(&self, total_bytes: u64) -> barracuda::error::Result<()> {
        if let Some(ref profile) = self.driver_profile {
            profile.check_allocation_safe(total_bytes)?;
        }
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════
    // Core dispatch helper
    // ═══════════════════════════════════════════════════════════════

    /// Attempt `gpu_fn` on the GPU device; on failure (or absence) run `cpu_fn`.
    ///
    /// Centralises the "try GPU, log-and-fallback" pattern so each public
    /// method only specifies *what* to compute on each backend.
    fn gpu_or_cpu<T>(
        &self,
        op: &str,
        gpu_fn: impl FnOnce(&Arc<WgpuDevice>) -> Result<T, String>,
        cpu_fn: impl FnOnce() -> T,
    ) -> T {
        if let Some(dev) = self.wgpu_device() {
            match gpu_fn(dev) {
                Ok(result) => return result,
                Err(e) => eprintln!("[dispatch] {op} GPU failed, falling back: {e}"),
            }
        }
        cpu_fn()
    }

    // ═══════════════════════════════════════════════════════════════
    // Mixed-hardware dispatch (metalForge integration)
    // ═══════════════════════════════════════════════════════════════

    /// Route a workload using the `metalForge` mixed-hardware cost model.
    ///
    /// Combines `dispatch.rs` substrate heuristics with `mixed.rs` transfer
    /// cost estimation to select the optimal execution path. This is the
    /// wiring point for `ToadStool` to absorb into `barracuda::unified_hardware`.
    #[allow(clippy::too_many_arguments)]
    pub fn mixed_dispatch<T>(
        &self,
        op: &str,
        compute_us: f64,
        data_bytes: u64,
        npu_available: bool,
        needs_realtime: bool,
        gpu_fn: impl FnOnce(&Arc<WgpuDevice>) -> Result<T, String>,
        cpu_fn: impl FnOnce() -> T,
    ) -> (T, neural_spring_forge::mixed::MixedSubstrate) {
        use neural_spring_forge::mixed::{mixed_substrate, MixedSubstrate};

        let substrate = mixed_substrate(
            compute_us,
            data_bytes,
            self.has_gpu(),
            npu_available,
            needs_realtime,
        );

        match substrate {
            MixedSubstrate::GpuOnly | MixedSubstrate::CpuToGpu => {
                if let Some(dev) = self.wgpu_device() {
                    match gpu_fn(dev) {
                        Ok(result) => return (result, substrate),
                        Err(e) => {
                            eprintln!("[mixed-dispatch] {op} GPU failed, falling back: {e}");
                        }
                    }
                }
                (cpu_fn(), MixedSubstrate::CpuOnly)
            }
            MixedSubstrate::GpuToNpu | MixedSubstrate::NpuToGpu | MixedSubstrate::NpuOnly => {
                eprintln!(
                    "[mixed-dispatch] {op} NPU substrate selected but not available, using GPU"
                );
                if let Some(dev) = self.wgpu_device() {
                    match gpu_fn(dev) {
                        Ok(result) => return (result, substrate),
                        Err(e) => {
                            eprintln!("[mixed-dispatch] {op} GPU fallback failed: {e}");
                        }
                    }
                }
                (cpu_fn(), MixedSubstrate::CpuOnly)
            }
            MixedSubstrate::CpuOnly | MixedSubstrate::GpuToCpu => (cpu_fn(), substrate),
        }
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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!((sorted[0] - 2.0).abs() < 1e-10);
        assert!((sorted[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn cpu_disorder_sweep_no_gpu() {
        let d = cpu();
        assert!(d.disorder_sweep(&[1.0, 0.0, 0.0, 1.0], 2, 1).is_none());
    }

    // ── Inter-population AF variance ──────────────────────────

    #[test]
    fn cpu_inter_population_af_variance_basic() {
        let d = cpu();
        let population_a = vec![2.0, 0.0, 0.0, 2.0];
        let population_b = vec![0.0, 2.0, 2.0, 0.0];
        let populations: Vec<&[f64]> = vec![&population_a, &population_b];
        let var = d.inter_population_af_variance(&populations, &[2, 2], 2);
        assert!(var >= 0.0, "AF variance must be non-negative");
    }

    // ── FST ──────────────────────────────────────────────────────

    #[test]
    fn cpu_pairwise_fst_divergent() {
        let d = cpu();
        let pop_a = vec![2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0];
        let pop_b = vec![0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0];
        let fst = d.pairwise_fst(&pop_a, 5, &pop_b, 5, 2);
        assert!(fst.is_finite(), "FST must be finite");
    }

    #[test]
    fn cpu_global_fst_two_pops() {
        let d = cpu();
        let pop1 = vec![2.0, 0.0, 2.0, 0.0];
        let pop2 = vec![0.0, 2.0, 0.0, 2.0];
        let fst = d.global_fst(&[pop1, pop2], &[2, 2], 2);
        assert!(fst.is_finite(), "FST must be finite");
    }

    // ── HMM chains ──────────────────────────────────────────────

    #[test]
    fn cpu_hmm_forward_chain_basic() {
        let d = cpu();
        let initial = vec![0.6, 0.4];
        #[rustfmt::skip]
        let transition = vec![0.7, 0.3, 0.4, 0.6];
        #[rustfmt::skip]
        let emission = vec![0.5, 0.4, 0.1, 0.1, 0.3, 0.6];
        let obs = vec![0, 1, 2, 0];
        let ll = d.hmm_forward_chain(&initial, &transition, &emission, &obs, 2, 3);
        assert!(ll.is_finite(), "log-likelihood must be finite");
        assert!(ll < 0.0, "log-likelihood should be negative");
    }

    #[test]
    fn cpu_hmm_viterbi_chain_basic() {
        let d = cpu();
        let initial = vec![0.6, 0.4];
        #[rustfmt::skip]
        let transition = vec![0.7, 0.3, 0.4, 0.6];
        #[rustfmt::skip]
        let emission = vec![0.5, 0.4, 0.1, 0.1, 0.3, 0.6];
        let obs = vec![0, 1, 2, 0];
        let (path, log_prob) = d.hmm_viterbi_chain(&initial, &transition, &emission, &obs, 2, 3);
        assert_eq!(path.len(), 4);
        assert!(log_prob.is_finite());
        for &s in &path {
            assert!(s < 2, "state must be < n_states");
        }
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

    // ── baseCamp (gpu_dispatch/basecamp.rs coverage) ─────────────

    #[test]
    fn basecamp_weight_spectral_analysis() {
        let d = cpu();
        let weights = vec![1.0, 0.0, 0.0, 1.0];
        let result = d.weight_spectral_analysis(&weights, 2, 2);
        assert_eq!(result.eigenvalues.len(), 4);
        assert!(result.mean_ipr.is_finite());
        assert!(result.level_spacing_ratio.is_finite());
        assert!(result.spectral_entropy.is_finite());
        assert!(result.mp_departure.is_finite());
    }

    #[test]
    fn basecamp_numerical_hessian_quadratic() {
        let d = cpu();
        let quadratic = |x: &[f64]| -> f64 { x.iter().map(|&v| v * v).sum() };
        let point = vec![1.0, 2.0];
        let hess = d.numerical_hessian(quadratic, &point, 1e-5);
        assert_eq!(hess.len(), 4);
        assert!((hess[0] - 2.0).abs() < 1e-4, "d²/dx² of x² = 2");
        assert!((hess[3] - 2.0).abs() < 1e-4, "d²/dy² of y² = 2");
        assert!(hess[1].abs() < 1e-4, "cross-term ≈ 0");
    }

    #[test]
    fn basecamp_belief_propagation_preserves_probability() {
        let d = cpu();
        let input = vec![0.25, 0.25, 0.25, 0.25];
        #[rustfmt::skip]
        let transition = vec![
            0.7, 0.3,
            0.6, 0.4,
            0.5, 0.5,
            0.4, 0.6,
        ];
        let dists = d.belief_propagation(&input, &[transition.as_slice()], &[2]);
        assert_eq!(dists.len(), 2);
        let final_sum: f64 = dists.last().expect("non-empty").iter().sum();
        assert!(
            (final_sum - 1.0).abs() < 1e-10,
            "output should be normalized, got sum={final_sum}"
        );
    }

    #[test]
    fn basecamp_agent_interaction_graph() {
        let d = cpu();
        let positions = vec![0.0, 0.0, 1.0, 0.0, 5.0, 5.0];
        let adj = d.agent_interaction_graph(&positions, 3, 2, 2.0);
        assert_eq!(adj.len(), 9);
        assert!(adj[1] > 0.0, "agents 0-1 within range");
        assert!(adj[3] > 0.0, "symmetric: adj[1][0]");
        assert!(adj[2].abs() < 1e-15, "agents 0-2 outside range");
    }
}
