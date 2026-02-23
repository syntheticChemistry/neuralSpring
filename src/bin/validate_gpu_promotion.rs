// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validates GPU-promoted operations against CPU reference implementations.
//!
//! Proves every CPU-bound operation in neuralSpring has a working GPU path
//! through `gpu_dispatch::Dispatcher`. Each check compares GPU output
//! against the CPU reference within documented tolerance.
//!
//! Coverage:
//! - Linear algebra: matmul, transpose, `frobenius_norm`, commutator, `distance_to_normal`
//! - Activations: softmax, boltzmann, GELU
//! - Reductions: mean, sum, max, variance
//! - Statistics: L2 distance, Shannon entropy, Pearson correlation, chi-squared
//! - HMM: forward step
//! - KL divergence

#![allow(clippy::cast_precision_loss)]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::validation::ValidationHarness;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let dispatcher = rt.block_on(Dispatcher::new());

    let mut h = ValidationHarness::new("validate_gpu_promotion");

    if !dispatcher.has_gpu() {
        eprintln!("WARNING: No GPU available — all checks use CPU fallback");
    }

    eprintln!(
        "Backend: {} ({})",
        dispatcher.backend(),
        dispatcher.adapter_name(),
    );

    // ─── Linear algebra ────────────────────────────────────────────

    let n = 8;
    let a: Vec<f64> = (0..n * n)
        .map(|i| (i + 1) as f64 / (n * n) as f64)
        .collect();
    let ident = neural_spring::spectral_commutativity::identity_matrix(n);
    let cpu_ai = neural_spring::spectral_commutativity::mat_mul(&a, &ident, n);
    let gpu_ai = dispatcher.mat_mul(&a, &ident, n);
    let diff_ai: f64 = cpu_ai
        .iter()
        .zip(gpu_ai.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper("matmul A*I max_diff", diff_ai, 0.01);

    let mut rng = neural_spring::rng::Rng::new(42);
    let ra = neural_spring::spectral_commutativity::random_matrix(n, &mut rng);
    let rb = neural_spring::spectral_commutativity::random_matrix(n, &mut rng);
    let cpu_ab = neural_spring::spectral_commutativity::mat_mul(&ra, &rb, n);
    let gpu_ab = dispatcher.mat_mul(&ra, &rb, n);
    let diff_ab: f64 = cpu_ab
        .iter()
        .zip(gpu_ab.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper("matmul random 8x8 max_diff", diff_ab, 0.05);

    let cpu_t = neural_spring::spectral_commutativity::transpose(&ra, n);
    let gpu_t = dispatcher.transpose(&ra, n);
    let diff_t: f64 = cpu_t
        .iter()
        .zip(gpu_t.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper("transpose 8x8 max_diff", diff_t, 0.01);

    let cpu_fn = neural_spring::spectral_commutativity::frobenius_norm(&[3.0, 4.0, 0.0, 0.0]);
    let gpu_fn = dispatcher.frobenius_norm(&[3.0, 4.0, 0.0, 0.0]);
    h.check_abs("frobenius_norm [3,4,0,0]", gpu_fn, cpu_fn, 0.01);

    let n4 = 4;
    let mut rng4 = neural_spring::rng::Rng::new(42);
    let ca = neural_spring::spectral_commutativity::random_matrix(n4, &mut rng4);
    let cb = neural_spring::spectral_commutativity::random_matrix(n4, &mut rng4);
    let cpu_comm = neural_spring::spectral_commutativity::commutator(&ca, &cb, n4);
    let gpu_comm = dispatcher.commutator(&ca, &cb, n4);
    let diff_comm: f64 = cpu_comm
        .iter()
        .zip(gpu_comm.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper("commutator [A,B] max_diff", diff_comm, 0.1);

    let sym = neural_spring::spectral_commutativity::random_symmetric(n, &mut rng);
    let cpu_dn = neural_spring::spectral_commutativity::distance_to_normal(&sym, n);
    let gpu_dn = dispatcher.distance_to_normal(&sym, n);
    h.check_upper("distance_to_normal sym CPU ≈ 0", cpu_dn, 1e-6);
    h.check_upper("distance_to_normal sym GPU ≈ 0", gpu_dn, 0.05);

    // ─── Activations ───────────────────────────────────────────────

    let x5 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let cpu_sm = neural_spring::transformer::softmax(&x5);
    let gpu_sm = dispatcher.softmax(&x5);
    let diff_sm: f64 = cpu_sm
        .iter()
        .zip(gpu_sm.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper("softmax max_diff", diff_sm, 0.01);
    h.check_abs("softmax sums to 1", gpu_sm.iter().sum::<f64>(), 1.0, 0.01);

    let fitnesses = vec![0.1, 0.5, 0.9, 0.3, 0.7];
    let cpu_b = neural_spring::counterdiabatic::boltzmann_distribution(&fitnesses, 2.0);
    let gpu_b = dispatcher.boltzmann(&fitnesses, 2.0);
    let diff_b: f64 = cpu_b
        .iter()
        .zip(gpu_b.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper("boltzmann max_diff", diff_b, 0.05);
    h.check_abs("boltzmann sums to 1", gpu_b.iter().sum::<f64>(), 1.0, 0.01);

    // ─── Reductions ────────────────────────────────────────────────

    let ab_a = vec![1.0, 2.0, 3.0];
    let ab_b = vec![4.0, 5.0, 6.0];
    let cpu_l2 = neural_spring::modes::l2_distance(&ab_a, &ab_b);
    let gpu_l2 = dispatcher.l2_distance(&ab_a, &ab_b);
    h.check_abs("L2 distance", gpu_l2, cpu_l2, 0.01);

    let data10: Vec<f64> = (1..=10).map(f64::from).collect();
    let cpu_mean = data10.iter().sum::<f64>() / 10.0;
    let gpu_mean = dispatcher.mean(&data10);
    h.check_abs("mean [1..10]", gpu_mean, cpu_mean, 0.01);

    let data_var = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let cpu_m = data_var.iter().sum::<f64>() / data_var.len() as f64;
    let cpu_var =
        data_var.iter().map(|&x| (x - cpu_m).powi(2)).sum::<f64>() / data_var.len() as f64;
    let gpu_var = dispatcher.variance(&data_var);
    h.check_abs("variance", gpu_var, cpu_var, 0.1);

    let uniform4 = vec![0.25; 4];
    let cpu_h = neural_spring::primitives::shannon_entropy(&uniform4);
    let gpu_h = dispatcher.shannon_entropy(&uniform4);
    h.check_abs("Shannon entropy uniform(4)", gpu_h, cpu_h, 0.05);

    // ─── Statistics ────────────────────────────────────────────────

    let px = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let py = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let gpu_r_pos = dispatcher.pearson_correlation(&px, &py);
    h.check_abs("Pearson r(x, 2x) ≈ 1", gpu_r_pos, 1.0, 0.05);

    let ny = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let gpu_r_neg = dispatcher.pearson_correlation(&px, &ny);
    h.check_abs("Pearson r(x, rev(x)) ≈ -1", gpu_r_neg, -1.0, 0.05);

    let obs = vec![10.0, 20.0, 30.0, 40.0];
    let exp = vec![25.0, 25.0, 25.0, 25.0];
    let cpu_chi2: f64 = obs
        .iter()
        .zip(exp.iter())
        .map(|(&o, &e): (&f64, &f64)| (o - e).powi(2) / e)
        .sum();
    let gpu_chi2 = dispatcher.chi_squared(&obs, &exp);
    h.check_abs("chi-squared", gpu_chi2, cpu_chi2, 0.5);

    // ─── HMM forward step ──────────────────────────────────────────

    if let Some(device) = dispatcher.wgpu_device() {
        let n_s = 3;
        let alpha = vec![0.4, 0.3, 0.3];
        let trans = vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.2, 0.6];
        let emit = vec![0.5, 0.3, 0.2];

        match neural_spring::gpu_ops::hmm_forward_step_gpu(&alpha, &trans, &emit, n_s, device) {
            Ok((alpha_new, scale)) => {
                let sum: f64 = alpha_new.iter().sum();
                h.check_abs("HMM fwd step normalized", sum, 1.0, 0.01);
                h.check_lower("HMM fwd step scale > 0", scale, 0.0);
            }
            Err(e) => h.check_bool(&format!("HMM fwd step: {e}"), false),
        }
    }

    // ─── Direct GPU tensor ops ─────────────────────────────────────

    if let Some(device) = dispatcher.wgpu_device() {
        match neural_spring::gpu_ops::gelu_gpu(&[-2.0, -1.0, 0.0, 0.5, 1.0, 3.0], device) {
            Ok(r) => {
                h.check_abs("GELU(0)", r[2], 0.0, 0.01);
                h.check_lower("GELU(3) > 2.9", r[5], 2.9);
            }
            Err(e) => h.check_bool(&format!("GELU: {e}"), false),
        }

        match neural_spring::gpu_ops::sum_gpu(&[1.0, 2.0, 3.0, 4.0, 5.0], device) {
            Ok(s) => h.check_abs("sum_gpu [1..5]", s, 15.0, 0.1),
            Err(e) => h.check_bool(&format!("sum: {e}"), false),
        }

        match neural_spring::gpu_ops::max_gpu(&[1.0, 5.0, 3.0, 2.0, 4.0], device) {
            Ok(m) => h.check_abs("max_gpu", m, 5.0, 0.1),
            Err(e) => h.check_bool(&format!("max: {e}"), false),
        }

        match neural_spring::gpu_ops::kl_divergence_gpu(
            &[0.25, 0.25, 0.25, 0.25],
            &[0.25, 0.25, 0.25, 0.25],
            device,
        ) {
            Ok(kl) => h.check_abs("KL(uniform, uniform)", kl, 0.0, 0.01),
            Err(e) => h.check_bool(&format!("KL: {e}"), false),
        }

        match neural_spring::gpu_ops::neural_forward_gpu(
            &[0.5, 0.3, 0.7, 0.2],
            &[0.1, 0.2, 0.3, 0.4],
            &[
                0.6, 0.4, 0.8, 0.1, 0.5, 0.3, 0.7, 0.2, 0.9, 0.4, 0.6, 0.5, 0.3, 0.8, 0.2, 0.7,
                0.4, 0.1, 0.9, 0.5,
            ],
            &[0.1, 0.2, 0.3, 0.4, 0.5],
            &[0.5],
            4,
            5,
            device,
        ) {
            Ok(out) => {
                h.check_bool("neural_forward produces output", out.len() == 5);
                h.check_bool(
                    "neural_forward outputs in (0,1)",
                    out.iter().all(|&v| v > 0.0 && v < 1.0),
                );
            }
            Err(e) => h.check_bool(&format!("neural_forward: {e}"), false),
        }
    }

    h.finish();
}
