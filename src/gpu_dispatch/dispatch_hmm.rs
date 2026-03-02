// SPDX-License-Identifier: AGPL-3.0-or-later

//! HMM dispatch operations (Liu 016–018).

use super::cpu_fallback;
use super::Dispatcher;

impl Dispatcher {
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
        self.gpu_or_cpu(
            "hmm_backward_step",
            |dev| {
                crate::gpu_ops::hmm_backward_step_gpu(
                    beta_next,
                    transition,
                    emission_col,
                    scale,
                    n_states,
                    dev,
                )
            },
            || {
                cpu_fallback::hmm_backward_step(
                    beta_next,
                    transition,
                    emission_col,
                    scale,
                    n_states,
                )
            },
        )
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
        self.gpu_or_cpu(
            "hmm_viterbi_step",
            |dev| {
                crate::gpu_ops::hmm_viterbi_step_gpu(
                    delta_prev,
                    log_transition,
                    log_emission_col,
                    n_states,
                    dev,
                )
            },
            || {
                cpu_fallback::hmm_viterbi_step(
                    delta_prev,
                    log_transition,
                    log_emission_col,
                    n_states,
                )
            },
        )
    }

    /// HMM forward chain: GPU full forward algorithm if available, CPU fallback.
    #[must_use]
    pub fn hmm_forward_chain(
        &self,
        initial: &[f64],
        transition: &[f64],
        emission: &[f64],
        observations: &[usize],
        n_states: usize,
        n_obs: usize,
    ) -> f64 {
        self.gpu_or_cpu(
            "hmm_forward_chain",
            |dev| {
                crate::gpu_ops::hmm_forward_chain_gpu(
                    initial,
                    transition,
                    emission,
                    observations,
                    n_states,
                    n_obs,
                    dev,
                )
            },
            || {
                let hmm = crate::hmm::Hmm::from_flat(
                    transition.to_vec(),
                    emission.to_vec(),
                    initial.to_vec(),
                    n_states,
                    n_obs,
                );
                hmm.forward(observations).1
            },
        )
    }

    /// HMM Viterbi chain: GPU full Viterbi if available, CPU fallback.
    /// Returns `(state_sequence, log_probability)`.
    #[must_use]
    pub fn hmm_viterbi_chain(
        &self,
        initial: &[f64],
        transition: &[f64],
        emission: &[f64],
        observations: &[usize],
        n_states: usize,
        n_obs: usize,
    ) -> (Vec<usize>, f64) {
        self.gpu_or_cpu(
            "hmm_viterbi_chain",
            |dev| {
                crate::gpu_ops::hmm_viterbi_chain_gpu(
                    initial,
                    transition,
                    emission,
                    observations,
                    n_states,
                    n_obs,
                    dev,
                )
            },
            || {
                let hmm = crate::hmm::Hmm::from_flat(
                    transition.to_vec(),
                    emission.to_vec(),
                    initial.to_vec(),
                    n_states,
                    n_obs,
                );
                hmm.viterbi(observations)
            },
        )
    }

    /// HMM chain: full forward + Viterbi over all observations.
    /// Returns `(path, log_prob, log_likelihood)`.
    #[must_use]
    pub fn hmm_chain(
        &self,
        initial: &[f64],
        transition: &[f64],
        emission: &[f64],
        observations: &[usize],
        n_states: usize,
        n_obs: usize,
    ) -> (Vec<usize>, f64, f64) {
        self.gpu_or_cpu(
            "hmm_chain",
            |dev| {
                let log_lik = crate::gpu_ops::hmm_forward_chain_gpu(
                    initial,
                    transition,
                    emission,
                    observations,
                    n_states,
                    n_obs,
                    dev,
                )?;
                let (path, log_prob) = crate::gpu_ops::hmm_viterbi_chain_gpu(
                    initial,
                    transition,
                    emission,
                    observations,
                    n_states,
                    n_obs,
                    dev,
                )?;
                Ok((path, log_prob, log_lik))
            },
            || {
                cpu_fallback::hmm_chain(
                    initial,
                    transition,
                    emission,
                    observations,
                    n_states,
                    n_obs,
                )
            },
        )
    }

    /// Detect introgression regions via Viterbi decoding.
    #[must_use]
    pub fn detect_introgression(
        &self,
        hmm: &crate::hmm::Hmm,
        observations: &[usize],
    ) -> (Vec<usize>, f64) {
        self.gpu_or_cpu(
            "detect_introgression",
            |dev| {
                let (path, log_prob) = crate::gpu_ops::hmm_viterbi_chain_gpu(
                    &hmm.initial,
                    &hmm.transition,
                    &hmm.emission,
                    observations,
                    hmm.num_states(),
                    hmm.num_symbols(),
                    dev,
                )?;
                Ok((path, log_prob))
            },
            || crate::introgression::detect_introgression(hmm, observations),
        )
    }

    /// HMM forward step: delegates to upstream `barracuda::dispatch::hmm_forward_dispatch`.
    #[must_use]
    pub fn hmm_forward_step(
        &self,
        alpha_prev: &[f64],
        transition: &[f64],
        emission_col: &[f64],
        n_states: usize,
    ) -> (Vec<f64>, f64) {
        barracuda::dispatch::hmm_forward_dispatch(
            alpha_prev,
            transition,
            emission_col,
            n_states,
            self.wgpu_device(),
        )
        .unwrap_or_else(|e| {
            log::warn!("hmm_forward_step upstream failed: {e}");
            cpu_fallback::hmm_forward_step(alpha_prev, transition, emission_col, n_states)
        })
    }
}
