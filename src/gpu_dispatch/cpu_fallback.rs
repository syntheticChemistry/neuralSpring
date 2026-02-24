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
//! `variance()` uses **population variance** (divides by N), matching
//! the GPU kernel convention. This differs from `barracuda::stats::variance`
//! which uses **sample variance** (divides by N-1). Do NOT rewire to
//! barracuda's variance — the conventions are intentionally different.

/// Population variance (biased, matching GPU kernel convention).
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
pub fn chi_squared(observed: &[f64], expected: &[f64]) -> f64 {
    barracuda::special::chi_squared_statistic(observed, expected).unwrap_or(0.0)
}

/// HMM backward step: `beta[i] = (1/scale) * sum_j(T[i,j] * B[j] * beta_next[j])`.
pub fn hmm_backward_step(
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

/// HMM Viterbi step: `delta[j] = max_i(delta_prev[i] + log_T[i,j]) + log_B[j]`.
pub fn hmm_viterbi_step(
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

/// Single replicator dynamics step with simplex projection.
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

    #[test]
    fn variance_basic() {
        let v = variance(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((v - 4.0).abs() < 1e-12);
    }

    #[test]
    fn variance_empty() {
        assert!((variance(&[]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn pearson_perfect() {
        let r = pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_short() {
        assert!((pearson(&[1.0], &[2.0]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn chi_squared_known() {
        let chi2 = chi_squared(&[10.0, 20.0, 30.0], &[20.0, 20.0, 20.0]);
        assert!((chi2 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn hmm_backward_basic() {
        let beta = hmm_backward_step(&[1.0, 1.0], &[0.7, 0.3, 0.4, 0.6], &[0.5, 0.5], 1.0, 2);
        assert!((beta[0] - 0.5).abs() < 1e-12);
        assert!((beta[1] - 0.5).abs() < 1e-12);
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
        assert!((sum - 1.0).abs() < 1e-12);
    }
}
