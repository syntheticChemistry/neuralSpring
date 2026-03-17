// SPDX-License-Identifier: AGPL-3.0-or-later

//! Game theory, eigensolvers, ODE batch, and pangenome selection dispatch.

#![expect(
    clippy::cast_possible_truncation,
    reason = "domain-specific numeric patterns"
)]

use super::Dispatcher;
use super::cpu_fallback;

impl Dispatcher {
    // ─── Game theory ─────────────────────────────────────────────

    /// Replicator dynamics step: GPU matmul if available, CPU fallback.
    #[must_use]
    pub fn replicator_step(&self, freq: &[f64; 2], payoff: &[[f64; 2]; 2], dt: f64) -> [f64; 2] {
        self.gpu_or_cpu(
            "replicator_step",
            |dev| crate::gpu_ops::replicator_step_gpu(freq, payoff, dt, dev),
            || cpu_fallback::replicator_step(freq, payoff, dt),
        )
    }

    // ─── Eigensolvers ────────────────────────────────────────────

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

    // ─── ODE batch integration ───────────────────────────────────

    /// Batch ODE integration: N systems × T timesteps, final state only.
    #[must_use]
    pub fn integrate_ode_batch(
        &self,
        states: &[f64],
        coeffs: &[f64],
        n_systems: usize,
        dim: usize,
        n_steps: usize,
        dt: f64,
    ) -> Vec<f64> {
        let states_f32: Vec<f32> = states.iter().map(|&v| v as f32).collect();
        let coeffs_f32: Vec<f32> = coeffs.iter().map(|&v| v as f32).collect();
        let n_coeffs = dim * 3;

        self.gpu_or_cpu(
            "integrate_ode_batch",
            |dev| {
                crate::gpu_ops::integrate_ode_batch_gpu(
                    &states_f32,
                    &coeffs_f32,
                    n_systems as u32,
                    dim as u32,
                    n_steps as u32,
                    dt as f32,
                    n_coeffs as u32,
                    dev,
                )
                .map(|v| v.into_iter().map(f64::from).collect())
            },
            || cpu_fallback::cpu_ode_batch_hill(states, coeffs, n_systems, dim, n_steps, dt),
        )
    }

    // ─── Pangenome selection ─────────────────────────────────────

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
