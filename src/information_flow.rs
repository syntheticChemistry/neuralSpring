// SPDX-License-Identifier: AGPL-3.0-or-later

//! Information flow analysis in neural network layers.
//!
//! baseCamp Sub-thesis 02: Information Flow as Wave Propagation.
//!
//! Models information propagation through neural network layers as
//! wave propagation through a disordered lattice. LSTM gates become
//! lattice site potentials; transformer attention matrices become
//! coupling matrices in an Anderson Hamiltonian.
//!
//! ## Grounding papers
//!
//! - Schoenholz et al. (2017) "Deep Information Propagation" (ICLR)
//! - Gu et al. (2020) "Improving the Gating Mechanism of RNNs" (ICML)
//! - Yang et al. (2025) "GLU Spectral Analysis"
//!
//! ## Validated primitives
//!
//! - [`crate::anderson_localization::ipr`] — inverse participation ratio
//! - [`crate::eigh::eigh_householder_qr`] — eigendecomposition
//! - [`crate::hmm`] — forward algorithm as GEMM chain

#![expect(
    clippy::cast_precision_loss,
    reason = "matrix dimension and bin counts → f64 for information-theoretic metrics"
)]

use crate::anderson_localization::ipr;
use crate::eigh::eigh_householder_qr;
use crate::primitives::LOG_GUARD;

/// Compute signal variance at each layer for depth-scale analysis.
///
/// Given per-layer output variances, returns the depth scale `xi_c`:
/// the characteristic depth at which signal variance decays to 1/e
/// of the initial value. Returns `f64::INFINITY` if no decay detected.
///
/// Schoenholz et al. (2017): trainability requires `xi_c` → ∞.
#[must_use]
pub fn depth_scale(layer_variances: &[f64]) -> f64 {
    if layer_variances.len() < 2 || layer_variances[0] < LOG_GUARD {
        return f64::INFINITY;
    }
    let initial = layer_variances[0];
    let threshold = initial / std::f64::consts::E;

    for (i, &var) in layer_variances.iter().enumerate().skip(1) {
        if var < threshold {
            let prev = layer_variances[i - 1];
            if (prev - var).abs() < LOG_GUARD {
                return i as f64;
            }
            let frac = (prev - threshold) / (prev - var);
            return (i - 1) as f64 + frac;
        }
    }
    f64::INFINITY
}

/// Compute the Anderson disorder parameter W from gate value distribution.
///
/// Maps LSTM gate values (sigmoid outputs in \[0,1\]) to an effective
/// disorder strength. High saturation (gates near 0 or 1) = high disorder.
/// Moderate gates (near 0.5) = low disorder.
///
/// `W = 4 * std_dev(gate_values)`, scaled so that uniform \[0,1\] gives W ≈ 1.15.
#[must_use]
pub fn gate_disorder_parameter(gate_values: &[f64]) -> f64 {
    if gate_values.is_empty() {
        return 0.0;
    }
    let n = gate_values.len() as f64;
    let mean = gate_values.iter().sum::<f64>() / n;
    let variance = gate_values.iter().map(|&g| (g - mean).powi(2)).sum::<f64>() / n;
    4.0 * variance.sqrt()
}

/// Compute gate saturation fraction: what fraction of gates are near 0 or 1.
///
/// A gate is "saturated" if it is within `threshold` of 0 or 1.
/// High saturation = strong Anderson disorder (information localization).
#[must_use]
pub fn gate_saturation(gate_values: &[f64], threshold: f64) -> f64 {
    if gate_values.is_empty() {
        return 0.0;
    }
    let saturated = gate_values
        .iter()
        .filter(|&&g| g < threshold || g > (1.0 - threshold))
        .count();
    saturated as f64 / gate_values.len() as f64
}

/// Information IPR: inverse participation ratio of an activation vector.
///
/// Measures how concentrated the neural activation is.
/// High IPR = information in few neurons (localized, bottleneck).
/// Low IPR = information spread evenly (delocalized, distributed).
///
/// Normalizes the activation vector to a probability distribution first.
#[must_use]
pub fn information_ipr(activations: &[f64]) -> f64 {
    let norm_sq: f64 = activations.iter().map(|&x| x * x).sum();
    if norm_sq < LOG_GUARD {
        return 0.0;
    }
    let normalized: Vec<f64> = activations.iter().map(|&x| x / norm_sq.sqrt()).collect();
    ipr(&normalized)
}

/// Construct an Anderson Hamiltonian from a transformer attention matrix.
///
/// Treats the attention matrix A (n×n) as a hopping matrix of a
/// tight-binding Hamiltonian. The on-site potential is derived from
/// the diagonal elements; off-diagonal elements are hopping amplitudes.
///
/// Returns a symmetric n×n matrix suitable for eigendecomposition.
#[must_use]
pub fn attention_to_hamiltonian(attention: &[f64], n: usize) -> Vec<f64> {
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let aij = attention[i * n + j];
            let aji = attention[j * n + i];
            h[i * n + j] = f64::midpoint(aij, aji);
        }
    }
    h
}

/// Spectral analysis of attention matrix.
///
/// Returns eigenvalues, mean IPR, and level spacing ratio of the
/// symmetrized attention Hamiltonian.
#[must_use]
pub fn attention_spectral_analysis(attention: &[f64], n: usize) -> AttentionSpectralResult {
    let h = attention_to_hamiltonian(attention, n);
    let decomp = eigh_householder_qr(&h, n);

    let mut eigenvalues = decomp.eigenvalues.clone();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean_ipr_val = crate::anderson_localization::mean_ipr(&decomp.eigenvectors, n);
    let lsr = crate::weight_spectral::level_spacing_ratio(&eigenvalues);

    AttentionSpectralResult {
        eigenvalues,
        mean_ipr: mean_ipr_val,
        level_spacing_ratio: lsr,
    }
}

/// Result of attention matrix spectral analysis.
#[derive(Debug, Clone)]
pub struct AttentionSpectralResult {
    /// Sorted eigenvalues.
    pub eigenvalues: Vec<f64>,
    /// Mean IPR of eigenstates.
    pub mean_ipr: f64,
    /// Level spacing ratio (GOE ≈ 0.531, Poisson ≈ 0.386).
    pub level_spacing_ratio: f64,
}

/// Compute per-layer signal propagation for an MLP.
///
/// Given weight matrices (as flat row-major) and layer dimensions,
/// propagates an input signal vector through random `ReLU` layers
/// and returns the variance at each layer.
///
/// This is the mean-field analysis from Schoenholz et al. (2017).
#[must_use]
pub fn mlp_signal_propagation(
    input: &[f64],
    weight_matrices: &[&[f64]],
    layer_dims: &[usize],
) -> Vec<f64> {
    let mut variances = Vec::with_capacity(weight_matrices.len() + 1);
    let input_var = input.iter().map(|&x| x * x).sum::<f64>() / input.len().max(1) as f64;
    variances.push(input_var);

    let mut signal = input.to_vec();
    for (layer_idx, &weights) in weight_matrices.iter().enumerate() {
        let n_in = if layer_idx == 0 {
            input.len()
        } else {
            layer_dims[layer_idx - 1]
        };
        let n_out = layer_dims[layer_idx];

        let mut output = vec![0.0; n_out];
        for i in 0..n_out {
            for j in 0..n_in.min(signal.len()) {
                output[i] = weights[i * n_in + j].mul_add(signal[j], output[i]);
            }
            output[i] = output[i].max(0.0);
        }

        let var = output.iter().map(|&x| x * x).sum::<f64>() / n_out.max(1) as f64;
        variances.push(var);
        signal = output;
    }

    variances
}

/// Edge-of-chaos diagnostic: compute the Jacobian spectral radius
/// at a given layer.
///
/// For a weight matrix W and `ReLU` activation, the Jacobian is
/// `diag(mask) * W` where `mask[i] = 1` if `pre-activation[i] > 0`.
/// The spectral radius determines signal amplification:
/// - ρ < 1: ordered phase (signal dies)
/// - ρ = 1: edge of chaos (signal propagates)
/// - ρ > 1: chaotic phase (signal explodes)
#[must_use]
pub fn jacobian_spectral_radius(weights: &[f64], pre_activations: &[f64], n: usize) -> f64 {
    let mut jacobian = vec![0.0; n * n];
    for i in 0..n {
        let mask = if pre_activations.get(i).copied().unwrap_or(0.0) > 0.0 {
            1.0
        } else {
            0.0
        };
        for j in 0..n {
            jacobian[i * n + j] = mask * weights[i * n + j];
        }
    }

    let jtj = mat_mul_transpose(&jacobian, n);
    let decomp = eigh_householder_qr(&jtj, n);
    decomp
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .sqrt()
}

fn mat_mul_transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i * n + j] = a[k * n + i].mul_add(a[k * n + j], result[i * n + j]);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::tolerances;

    #[test]
    fn depth_scale_infinite_for_constant() {
        let variances = vec![1.0; 10];
        let xi = depth_scale(&variances);
        assert!(
            xi.is_infinite(),
            "constant variance should give infinite depth scale"
        );
    }

    #[test]
    fn depth_scale_finite_for_decay() {
        let variances: Vec<f64> = (0..10).map(|i| (-0.5 * f64::from(i)).exp()).collect();
        let xi = depth_scale(&variances);
        assert!(
            xi > 0.0 && xi < 10.0,
            "exponential decay should give finite xi, got {xi}"
        );
    }

    #[test]
    fn gate_disorder_zero_for_constant() {
        let gates = vec![0.5; 100];
        let w = gate_disorder_parameter(&gates);
        assert!(
            w.abs() < tolerances::EXACT_F64,
            "constant gates should give zero disorder, got {w}"
        );
    }

    #[test]
    fn gate_disorder_positive_for_spread() {
        let mut rng = Rng::new(42);
        let gates: Vec<f64> = (0..100).map(|_| rng.uniform()).collect();
        let w = gate_disorder_parameter(&gates);
        assert!(
            w > 0.0,
            "spread gates should give positive disorder, got {w}"
        );
    }

    #[test]
    fn saturation_fraction_bounds() {
        let gates = vec![0.01, 0.99, 0.5, 0.02, 0.98, 0.5];
        let sat = gate_saturation(&gates, 0.05);
        assert!(
            (sat - 4.0 / 6.0).abs() < tolerances::EXACT_F64,
            "expected 4/6 saturation, got {sat}"
        );
    }

    #[test]
    fn information_ipr_bounds() {
        let uniform = vec![1.0; 8];
        let localized = {
            let mut v = vec![0.0; 8];
            v[0] = 1.0;
            v
        };
        let ipr_uniform = information_ipr(&uniform);
        let ipr_localized = information_ipr(&localized);
        assert!(
            ipr_localized > ipr_uniform,
            "localized should have higher IPR: {ipr_localized} vs {ipr_uniform}"
        );
    }

    #[test]
    fn attention_hamiltonian_symmetric() {
        let mut rng = Rng::new(42);
        let n = 8;
        let attention: Vec<f64> = (0..n * n).map(|_| rng.uniform()).collect();
        let h = attention_to_hamiltonian(&attention, n);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (h[i * n + j] - h[j * n + i]).abs() < tolerances::ZERO_DETECTION,
                    "H not symmetric"
                );
            }
        }
    }

    #[test]
    fn determinism() {
        let gates = vec![0.1, 0.9, 0.5, 0.2, 0.8];
        let w1 = gate_disorder_parameter(&gates);
        let w2 = gate_disorder_parameter(&gates);
        assert!((w1 - w2).abs() < f64::EPSILON, "determinism: {w1} != {w2}");
    }

    #[test]
    fn depth_scale_edge_cases() {
        assert!(depth_scale(&[]).is_infinite(), "empty → infinite");
        assert!(depth_scale(&[1.0]).is_infinite(), "single → infinite");
        assert!(
            depth_scale(&[0.0, 0.0]).is_infinite(),
            "zero initial → infinite"
        );

        let below_threshold = vec![1.0, 0.3];
        let xi = depth_scale(&below_threshold);
        assert!(xi.is_finite(), "below-threshold decay should resolve");
        assert!(xi > 0.0 && xi < 2.0, "xi in (0, 2), got {xi}");

        let just_above = vec![1.0, 0.37];
        assert!(
            depth_scale(&just_above).is_infinite(),
            "above 1/e → infinite"
        );
    }

    #[test]
    fn gate_disorder_empty() {
        assert!((gate_disorder_parameter(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gate_saturation_empty() {
        assert!((gate_saturation(&[], 0.1) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gate_saturation_none_saturated() {
        let gates = vec![0.5; 10];
        let sat = gate_saturation(&gates, 0.1);
        assert!(sat.abs() < f64::EPSILON, "mid-range → 0 saturation");
    }

    #[test]
    fn information_ipr_zero_activations() {
        let zeros = vec![0.0; 8];
        assert!(
            information_ipr(&zeros).abs() < f64::EPSILON,
            "zero activations → 0 IPR"
        );
    }

    #[test]
    fn mlp_signal_propagation_basic() {
        let input = vec![1.0, 0.5, -0.3, 0.8];
        let w1: Vec<f64> = vec![
            0.2, 0.1, 0.3, -0.1, -0.2, 0.4, 0.1, 0.2, 0.3, -0.3, 0.2, 0.1,
        ];
        let w2: Vec<f64> = vec![0.5, -0.2, 0.3, -0.4, 0.1, 0.6];
        let vars = mlp_signal_propagation(&input, &[w1.as_slice(), w2.as_slice()], &[3, 2]);
        assert_eq!(vars.len(), 3, "input + 2 layers = 3 variances");
        assert!(vars[0] > 0.0, "input variance positive");
        assert!(vars.iter().all(|v| v.is_finite()), "all variances finite");
    }

    #[test]
    fn mlp_signal_propagation_single_layer() {
        let input = vec![1.0, 1.0];
        let w: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
        let vars = mlp_signal_propagation(&input, &[w.as_slice()], &[2]);
        assert_eq!(vars.len(), 2);
        assert!(
            (vars[0] - 1.0).abs() < tolerances::EXACT_F64,
            "identity preserves variance"
        );
    }

    #[test]
    fn attention_spectral_analysis_produces_results() {
        let mut rng = Rng::new(42);
        let n = 8;
        let attention: Vec<f64> = (0..n * n).map(|_| rng.uniform()).collect();
        let result = attention_spectral_analysis(&attention, n);
        assert_eq!(result.eigenvalues.len(), n);
        assert!(result.mean_ipr > 0.0, "IPR should be positive");
        assert!(
            result.level_spacing_ratio.is_finite(),
            "LSR should be finite"
        );
        let sorted: Vec<f64> = result.eigenvalues;
        for w in sorted.windows(2) {
            assert!(w[0] <= w[1], "eigenvalues should be sorted");
        }
    }

    #[test]
    fn jacobian_spectral_radius_identity() {
        let n = 4;
        let mut weights = vec![0.0; n * n];
        for i in 0..n {
            weights[i * n + i] = 1.0;
        }
        let pre_act = vec![1.0; n]; // all positive → ReLU mask all 1
        let rho = jacobian_spectral_radius(&weights, &pre_act, n);
        assert!(
            (rho - 1.0).abs() < 0.05,
            "identity weight + all-positive pre-act → ρ ≈ 1, got {rho}"
        );
    }

    #[test]
    fn jacobian_spectral_radius_zero_pre_activations() {
        let n = 4;
        let mut rng = Rng::new(42);
        let weights: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();
        let pre_act = vec![-1.0; n]; // all negative → ReLU mask all 0
        let rho = jacobian_spectral_radius(&weights, &pre_act, n);
        assert!(
            rho < tolerances::CROSS_LANGUAGE,
            "all-negative pre-act → ρ ≈ 0 (dead ReLU), got {rho}"
        );
    }

    #[test]
    fn jacobian_spectral_radius_random() {
        let mut rng = Rng::new(42);
        let n = 8;
        let weights: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();
        let pre_act: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let rho = jacobian_spectral_radius(&weights, &pre_act, n);
        assert!(
            rho.is_finite() && rho >= 0.0,
            "ρ must be finite and non-negative"
        );
    }

    #[test]
    fn mat_mul_transpose_symmetry() {
        let mut rng = Rng::new(42);
        let n = 4;
        let a: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();
        let ata = mat_mul_transpose(&a, n);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (ata[i * n + j] - ata[j * n + i]).abs() < tolerances::EXACT_F64,
                    "A^T A must be symmetric"
                );
            }
        }
    }
}
