// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU reference implementations for dispatched operations.
//!
//! These are the fallback paths when no GPU adapter is available.
//! Each function mirrors a GPU kernel but uses scalar CPU arithmetic.
//! Kept in a separate module so the dispatcher focuses on routing
//! and these implementations are independently testable.
//!
//! ## Variance convention
//!
//! `variance()` uses **population variance** (divides by N). As of `ToadStool`
//! S66, `barracuda::dispatch::variance_dispatch` also uses population variance
//! (ddof=0), so the dispatcher and this fallback now agree. Note that
//! `barracuda::stats::correlation::variance` still uses sample variance (N-1)
//! — do NOT confuse the two.

/// Population variance (biased, matching GPU kernel and `variance_dispatch` convention).
#[must_use]
pub fn variance(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 1.0 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n
}

/// Pearson product-moment correlation coefficient.
///
/// Delegates to `barracuda::stats::correlation::pearson_correlation`
/// (identical formula). Returns 0.0 for degenerate inputs (too short,
/// zero variance, NaN).
#[must_use]
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    match barracuda::stats::correlation::pearson_correlation(x, y) {
        Ok(r) if r.is_finite() => r,
        _ => 0.0,
    }
}

/// Chi-squared statistic: `sum((O-E)^2 / E)`.
///
/// Delegates to `barracuda::special::chi_squared_statistic`
/// (identical formula). Falls back to 0.0 on error.
#[must_use]
pub fn chi_squared(observed: &[f64], expected: &[f64]) -> f64 {
    barracuda::special::chi_squared_statistic(observed, expected).unwrap_or(0.0)
}

/// KL divergence: `sum(p * ln(p/q))`.
///
/// Delegates to `counterdiabatic::kl_divergence` (normalizes inputs,
/// guards against zero). Falls back to 0.0 on empty input.
#[must_use]
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    crate::counterdiabatic::kl_divergence(p, q)
}

/// HMM forward step: `alpha[j] = B[j] * sum_i(alpha_prev[i] * T[i,j])`, then normalize.
///
/// Returns `(alpha_new, scale)` where `scale = sum(alpha_new)` before normalization.
#[must_use]
pub fn hmm_forward_step(
    alpha_prev: &[f64],
    transition: &[f64],
    emission_col: &[f64],
    n_states: usize,
) -> (Vec<f64>, f64) {
    let mut alpha: Vec<f64> = (0..n_states)
        .map(|j| {
            let sum: f64 = alpha_prev
                .iter()
                .enumerate()
                .map(|(i, &a)| a * transition[i * n_states + j])
                .sum();
            sum * emission_col[j]
        })
        .collect();
    let scale: f64 = alpha.iter().sum();
    if scale > 0.0 {
        for a in &mut alpha {
            *a /= scale;
        }
    }
    (alpha, scale)
}

/// HMM backward step: `beta[i] = (1/scale) * sum_j(T[i,j] * B[j] * beta_next[j])`.
#[must_use]
pub fn hmm_backward_step(
    beta_next: &[f64],
    transition: &[f64],
    emission_col: &[f64],
    scale: f64,
    n_states: usize,
) -> Vec<f64> {
    let guard = crate::primitives::LOG_GUARD;
    let safe_scale = if scale.abs() < guard { guard } else { scale };
    (0..n_states)
        .map(|i| {
            let sum: f64 = (0..n_states)
                .map(|j| transition[i * n_states + j] * emission_col[j] * beta_next[j])
                .sum();
            sum / safe_scale
        })
        .collect()
}

/// HMM chain: T forward steps + T Viterbi steps (step-level composition).
///
/// Returns `(path, log_prob, log_likelihood)`. Used as CPU fallback when
/// `Dispatcher::hmm_chain` has no GPU.
#[must_use]
pub fn hmm_chain(
    initial: &[f64],
    transition: &[f64],
    emission: &[f64],
    observations: &[usize],
    n_states: usize,
    n_obs: usize,
) -> (Vec<usize>, f64, f64) {
    let t_len = observations.len();
    if t_len == 0 {
        return (Vec::new(), 0.0, 0.0);
    }

    let ob0 = observations[0].min(n_obs.saturating_sub(1));
    let mut alpha: Vec<f64> = (0..n_states)
        .map(|i| initial[i] * emission[i * n_obs + ob0])
        .collect();
    let scale0 = alpha.iter().sum::<f64>().max(crate::primitives::LOG_GUARD);
    for v in &mut alpha {
        *v /= scale0;
    }
    let mut log_lik = scale0.ln();

    for &ob_raw in observations.iter().skip(1).take(t_len.saturating_sub(1)) {
        let ob = ob_raw.min(n_obs.saturating_sub(1));
        let emission_col: Vec<f64> = (0..n_states).map(|j| emission[j * n_obs + ob]).collect();
        let (new_alpha, scale) = hmm_forward_step(&alpha, transition, &emission_col, n_states);
        log_lik += scale.max(crate::primitives::LOG_GUARD).ln();
        alpha = new_alpha;
    }

    let log_trans: Vec<f64> = transition
        .iter()
        .map(|&x| x.max(crate::primitives::LOG_GUARD).ln())
        .collect();

    let mut delta: Vec<f64> = (0..n_states)
        .map(|i| {
            initial[i].max(crate::primitives::LOG_GUARD).ln()
                + emission[i * n_obs + observations[0].min(n_obs.saturating_sub(1))]
                    .max(crate::primitives::LOG_GUARD)
                    .ln()
        })
        .collect();

    let mut psi_all = Vec::with_capacity(t_len);

    for &ob_raw in observations.iter().skip(1).take(t_len.saturating_sub(1)) {
        let ob = ob_raw.min(n_obs.saturating_sub(1));
        let log_emit: Vec<f64> = (0..n_states)
            .map(|j| {
                emission[j * n_obs + ob]
                    .max(crate::primitives::LOG_GUARD)
                    .ln()
            })
            .collect();
        let (new_delta, psi) = hmm_viterbi_step(&delta, &log_trans, &log_emit, n_states);
        psi_all.push(psi);
        delta = new_delta;
    }

    let (best_state, log_prob) = delta
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or((0, f64::NEG_INFINITY), |(i, &v)| (i, v));

    let mut path = vec![0_usize; t_len];
    path[t_len - 1] = best_state;
    for t in (0..t_len.saturating_sub(1)).rev() {
        path[t] = psi_all[t][path[t + 1]];
    }

    (path, log_prob, log_lik)
}

/// HMM Viterbi step: `delta[j] = max_i(delta_prev[i] + log_T[i,j]) + log_B[j]`.
#[must_use]
pub fn hmm_viterbi_step(
    delta_prev: &[f64],
    log_transition: &[f64],
    log_emission_col: &[f64],
    n_states: usize,
) -> (Vec<f64>, Vec<usize>) {
    let (delta_new, psi): (Vec<f64>, Vec<usize>) = (0..n_states)
        .map(|j| {
            let (best_i, best_val) = (0..n_states)
                .map(|i| (i, delta_prev[i] + log_transition[i * n_states + j]))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, f64::NEG_INFINITY));
            (best_val + log_emission_col[j], best_i)
        })
        .unzip();
    (delta_new, psi)
}

/// CPU batch ODE integration: Hill-function RHS, matches `rk4_parallel.wgsl`.
///
/// Coeffs per dimension: `[prod, deg, activator_idx]`. Hill uses k=0.5, n=2.
///
/// # Panics
///
/// Panics if any `activator_idx` in `coeffs` is negative or >= `dim`.
#[expect(
    clippy::cast_sign_loss,
    reason = "activator_idx from coeffs is asserted non-negative"
)]
#[must_use]
pub fn cpu_ode_batch_hill(
    states: &[f64],
    coeffs: &[f64],
    n_systems: usize,
    dim: usize,
    n_steps: usize,
    dt: f64,
) -> Vec<f64> {
    fn hill(x: f64, k: f64, n: f64) -> f64 {
        let xn = x.powf(n);
        xn / (k.powf(n) + xn)
    }

    fn safe_idx(raw: f64, dim: usize) -> usize {
        assert!(
            raw >= 0.0 && (raw as usize) < dim,
            "activator_idx {raw} out of range [0, {dim})"
        );
        raw as usize
    }

    let n_coeffs = dim * 3;
    let half_dt = 0.5 * dt;
    let sixth_dt = dt / 6.0;

    let mut out = Vec::with_capacity(n_systems * dim);
    for sys in 0..n_systems {
        let mut y: Vec<f64> = (0..dim).map(|d| states[sys * dim + d]).collect();
        let coeff_base = sys * n_coeffs;

        for _ in 0..n_steps {
            let mut k1 = vec![0.0; dim];
            for d in 0..dim {
                let c = coeff_base + d * 3;
                let prod = coeffs[c];
                let deg = coeffs[c + 1];
                let act_idx = safe_idx(coeffs[c + 2], dim);
                k1[d] = prod.mul_add(hill(y[act_idx], 0.5, 2.0), -(deg * y[d]));
            }

            let mut k2 = vec![0.0; dim];
            for d in 0..dim {
                let y2_d = half_dt.mul_add(k1[d], y[d]);
                let c = coeff_base + d * 3;
                let prod = coeffs[c];
                let deg = coeffs[c + 1];
                let act_idx = safe_idx(coeffs[c + 2], dim);
                let act_val = half_dt.mul_add(k1[act_idx], y[act_idx]);
                k2[d] = prod.mul_add(hill(act_val, 0.5, 2.0), -(deg * y2_d));
            }

            let mut k3 = vec![0.0; dim];
            for d in 0..dim {
                let y3_d = half_dt.mul_add(k2[d], y[d]);
                let c = coeff_base + d * 3;
                let prod = coeffs[c];
                let deg = coeffs[c + 1];
                let act_idx = safe_idx(coeffs[c + 2], dim);
                let act_val = half_dt.mul_add(k2[act_idx], y[act_idx]);
                k3[d] = prod.mul_add(hill(act_val, 0.5, 2.0), -(deg * y3_d));
            }

            let mut k4 = vec![0.0; dim];
            for d in 0..dim {
                let y4_d = dt.mul_add(k3[d], y[d]);
                let c = coeff_base + d * 3;
                let prod = coeffs[c];
                let deg = coeffs[c + 1];
                let act_idx = safe_idx(coeffs[c + 2], dim);
                let act_val = dt.mul_add(k3[act_idx], y[act_idx]);
                k4[d] = prod.mul_add(hill(act_val, 0.5, 2.0), -(deg * y4_d));
            }

            for d in 0..dim {
                let w = 2.0f64.mul_add(k2[d], k1[d]) + 2.0f64.mul_add(k3[d], k4[d]);
                y[d] = sixth_dt.mul_add(w, y[d]);
            }
        }
        out.extend_from_slice(&y);
    }
    out
}

/// Single replicator dynamics step with simplex projection.
#[must_use]
pub fn replicator_step(freq: &[f64; 2], payoff: &[[f64; 2]; 2], dt: f64) -> [f64; 2] {
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

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn variance_basic() {
        let v = variance(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((v - 4.0).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn variance_empty() {
        assert!((variance(&[]) - 0.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn pearson_perfect() {
        let r = pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]);
        assert!((r - 1.0).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn pearson_short() {
        assert!((pearson(&[1.0], &[2.0]) - 0.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn chi_squared_known() {
        let chi2 = chi_squared(&[10.0, 20.0, 30.0], &[20.0, 20.0, 20.0]);
        assert!((chi2 - 10.0).abs() < tolerances::CROSS_LANGUAGE);
    }

    #[test]
    fn hmm_backward_basic() {
        let beta = hmm_backward_step(&[1.0, 1.0], &[0.7, 0.3, 0.4, 0.6], &[0.5, 0.5], 1.0, 2);
        assert!((beta[0] - 0.5).abs() < tolerances::EXACT_F64);
        assert!((beta[1] - 0.5).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn hmm_backward_zero_scale() {
        let beta = hmm_backward_step(&[1.0], &[1.0], &[1.0], 0.0, 1);
        assert!(beta[0].is_finite());
    }

    #[test]
    fn replicator_simplex() {
        let next = replicator_step(&[0.6, 0.4], &[[3.0, 0.0], [5.0, 1.0]], 0.01);
        let sum: f64 = next.iter().sum();
        assert!((sum - 1.0).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn hmm_forward_step_normalizes() {
        let alpha = vec![0.6, 0.4];
        #[rustfmt::skip]
        let trans = vec![0.7, 0.3, 0.4, 0.6];
        let emit = vec![0.5, 0.5];
        let (new_alpha, scale) = hmm_forward_step(&alpha, &trans, &emit, 2);
        assert_eq!(new_alpha.len(), 2);
        assert!(scale > 0.0);
        let sum: f64 = new_alpha.iter().sum();
        assert!((sum - 1.0).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn hmm_forward_step_zero_emission() {
        let alpha = vec![0.5, 0.5];
        let trans = vec![1.0, 0.0, 0.0, 1.0];
        let emit = vec![0.0, 0.0];
        let (alpha_new, scale) = hmm_forward_step(&alpha, &trans, &emit, 2);
        assert_eq!(alpha_new.len(), 2);
        assert!((scale - 0.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn hmm_viterbi_step_known() {
        let delta_prev = vec![0.0, -1.0];
        #[rustfmt::skip]
        let log_trans = vec![
            0.7_f64.ln(), 0.3_f64.ln(),
            0.4_f64.ln(), 0.6_f64.ln(),
        ];
        let log_emit = vec![0.6_f64.ln(), 0.4_f64.ln()];
        let (delta, psi) = hmm_viterbi_step(&delta_prev, &log_trans, &log_emit, 2);
        assert_eq!(delta.len(), 2);
        assert_eq!(psi.len(), 2);
        assert!(delta[0].is_finite());
        assert!(delta[1].is_finite());
        assert!(psi[0] < 2);
        assert!(psi[1] < 2);
    }

    #[test]
    fn pearson_zero_variance_returns_zero() {
        let r = pearson(&[3.0, 3.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((r - 0.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn chi_squared_zero_expected() {
        let c = chi_squared(&[5.0], &[0.0]);
        assert!(c.is_finite());
    }

    #[test]
    fn replicator_zero_frequencies() {
        let next = replicator_step(&[0.0, 0.0], &[[1.0, 0.0], [0.0, 1.0]], 0.01);
        assert!(next[0].is_finite());
        assert!(next[1].is_finite());
    }

    #[test]
    fn variance_single_element() {
        assert!((variance(&[5.0]) - 0.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn cpu_ode_batch_hill_single_system() {
        let dim = 2;
        let coeffs = vec![0.5, 0.1, 1.0, 0.3, 0.2, 0.0];
        let states = vec![1.0, 0.5];
        let result = cpu_ode_batch_hill(&states, &coeffs, 1, dim, 10, 0.01);
        assert_eq!(result.len(), 2);
        assert!(result[0].is_finite() && result[1].is_finite());
    }

    #[test]
    fn cpu_ode_batch_hill_multi_system() {
        let dim = 2;
        let n_systems = 3;
        let coeffs: Vec<f64> = (0..n_systems)
            .flat_map(|_| vec![0.5, 0.1, 1.0, 0.3, 0.2, 0.0])
            .collect();
        let states: Vec<f64> = (0..n_systems)
            .flat_map(|i| vec![1.0 + i as f64, 0.5])
            .collect();
        let result = cpu_ode_batch_hill(&states, &coeffs, n_systems, dim, 50, 0.01);
        assert_eq!(result.len(), n_systems * dim);
        for v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn hmm_chain_produces_valid_path() {
        let trans = vec![0.7, 0.3, 0.4, 0.6];
        let emission = vec![0.1, 0.4, 0.5, 0.6, 0.3, 0.1];
        let initial = vec![0.6, 0.4];
        let obs = vec![0, 1, 2, 0, 1];
        let (path, log_prob, log_lik) = hmm_chain(&initial, &trans, &emission, &obs, 2, 3);
        assert_eq!(path.len(), obs.len());
        assert!(log_prob.is_finite());
        assert!(log_lik.is_finite());
        assert!(log_lik < 0.0, "log-likelihood should be negative");
        for &s in &path {
            assert!(s < 2, "state out of range");
        }
    }

    #[test]
    fn hmm_chain_empty_obs() {
        let (path, log_prob, log_lik) =
            hmm_chain(&[0.5, 0.5], &[1.0, 0.0, 0.0, 1.0], &[0.5, 0.5], &[], 2, 1);
        assert!(path.is_empty());
        assert!((log_prob - 0.0).abs() < tolerances::ZERO_DETECTION);
        assert!((log_lik - 0.0).abs() < tolerances::ZERO_DETECTION);
    }

    #[test]
    fn hmm_chain_single_obs() {
        let trans = vec![0.7, 0.3, 0.4, 0.6];
        let emission = vec![0.9, 0.1, 0.2, 0.8];
        let initial = vec![0.6, 0.4];
        let (path, log_prob, log_lik) = hmm_chain(&initial, &trans, &emission, &[0], 2, 2);
        assert_eq!(path.len(), 1);
        assert!(log_prob.is_finite());
        assert!(log_lik.is_finite());
    }

    #[test]
    fn kl_divergence_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let d = kl_divergence(&p, &p);
        assert!((d - 0.0).abs() < tolerances::EXACT_F64);
    }

    #[test]
    fn kl_divergence_different() {
        let p = vec![0.9, 0.1];
        let q = vec![0.5, 0.5];
        let d = kl_divergence(&p, &q);
        assert!(d > 0.0, "KL(P||Q) > 0 for P != Q");
        assert!(d.is_finite());
    }
}
