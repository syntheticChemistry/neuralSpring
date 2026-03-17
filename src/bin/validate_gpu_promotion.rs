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
//!
//! ## Provenance
//!
//! CPU reference: neuralSpring library modules (Rust CPU).
//! GPU dispatch: `BarraCUDA` typed GPU ops via WGSL shaders.
//! Validated on: llvmpipe (software Vulkan) and RTX 4070 (hardware Vulkan).
//! No Python baseline — GPU parity validated against Rust CPU reference.

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::tolerances;
use neural_spring::validation::{max_abs_diff_f64, ValidationHarness};

fn main() {
    let Ok(rt) = tokio::runtime::Runtime::new() else {
        println!("FATAL: could not create tokio runtime");
        std::process::exit(1);
    };
    let dispatcher = rt.block_on(Dispatcher::new());

    let mut h = ValidationHarness::new("validate_gpu_promotion");

    if !dispatcher.has_gpu() {
        println!("WARNING: No GPU available — all checks use CPU fallback");
    }

    println!(
        "Backend: {} ({})",
        dispatcher.backend(),
        dispatcher.adapter_name(),
    );

    validate_linear_algebra(&mut h, &dispatcher);
    validate_activations(&mut h, &dispatcher);
    validate_reductions(&mut h, &dispatcher);
    validate_statistics(&mut h, &dispatcher);
    validate_hmm(&mut h, &dispatcher);
    validate_tensor_ops(&mut h, &dispatcher);

    h.finish();
}

fn validate_linear_algebra(h: &mut ValidationHarness, disp: &Dispatcher) {
    let n = 8;
    let sequential: Vec<f64> = (0..n * n)
        .map(|i| (i + 1) as f64 / (n * n) as f64)
        .collect();
    let ident = neural_spring::spectral_commutativity::identity_matrix(n);
    let cpu_identity_prod = neural_spring::spectral_commutativity::mat_mul(&sequential, &ident, n);
    let gpu_identity_prod = disp.mat_mul(&sequential, &ident, n);
    h.check_upper(
        "matmul A*I max_diff",
        max_abs_diff_f64(&cpu_identity_prod, &gpu_identity_prod),
        tolerances::GPU_MATMUL_IDENTITY_F32,
    );

    let mut rng = neural_spring::rng::Rng::new(42);
    let rand_a = neural_spring::spectral_commutativity::random_matrix(n, &mut rng);
    let rand_b = neural_spring::spectral_commutativity::random_matrix(n, &mut rng);
    let cpu_random_prod = neural_spring::spectral_commutativity::mat_mul(&rand_a, &rand_b, n);
    let gpu_random_prod = disp.mat_mul(&rand_a, &rand_b, n);
    h.check_upper(
        "matmul random 8x8 max_diff",
        max_abs_diff_f64(&cpu_random_prod, &gpu_random_prod),
        tolerances::GPU_MATMUL_RANDOM_F32,
    );

    let cpu_transposed = neural_spring::spectral_commutativity::transpose(&rand_a, n);
    let gpu_transposed = disp.transpose(&rand_a, n);
    h.check_upper(
        "transpose 8x8 max_diff",
        max_abs_diff_f64(&cpu_transposed, &gpu_transposed),
        tolerances::GPU_TRANSPOSE_F32,
    );

    let cpu_frob = neural_spring::spectral_commutativity::frobenius_norm(&[3.0, 4.0, 0.0, 0.0]);
    let gpu_frob = disp.frobenius_norm(&[3.0, 4.0, 0.0, 0.0]);
    h.check_abs(
        "frobenius_norm [3,4,0,0]",
        gpu_frob,
        cpu_frob,
        tolerances::GPU_FROBENIUS_F32,
    );

    let n4 = 4;
    let mut rng4 = neural_spring::rng::Rng::new(42);
    let comm_a = neural_spring::spectral_commutativity::random_matrix(n4, &mut rng4);
    let comm_b = neural_spring::spectral_commutativity::random_matrix(n4, &mut rng4);
    let cpu_comm = neural_spring::spectral_commutativity::commutator(&comm_a, &comm_b, n4);
    let gpu_comm = disp.commutator(&comm_a, &comm_b, n4);
    h.check_upper(
        "commutator `[A,B]` max_diff",
        max_abs_diff_f64(&cpu_comm, &gpu_comm),
        tolerances::GPU_COMMUTATOR_F32,
    );

    let sym = neural_spring::spectral_commutativity::random_symmetric(n, &mut rng);
    let cpu_dist_normal = neural_spring::spectral_commutativity::distance_to_normal(&sym, n);
    let gpu_dist_normal = disp.distance_to_normal(&sym, n);
    h.check_upper(
        "distance_to_normal sym CPU ≈ 0",
        cpu_dist_normal,
        tolerances::CPU_NORMAL_DISTANCE_SYMMETRIC_F64,
    );
    h.check_upper(
        "distance_to_normal sym GPU ≈ 0",
        gpu_dist_normal,
        tolerances::GPU_NORMAL_DISTANCE_SYMMETRIC_F32,
    );
}

fn validate_activations(h: &mut ValidationHarness, disp: &Dispatcher) {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let cpu_softmax = neural_spring::transformer::softmax(&input);
    let gpu_softmax = disp.softmax(&input);
    h.check_upper(
        "softmax max_diff",
        max_abs_diff_f64(&cpu_softmax, &gpu_softmax),
        tolerances::GPU_SOFTMAX_DISPATCH_F32,
    );
    h.check_abs(
        "softmax sums to 1",
        gpu_softmax.iter().sum::<f64>(),
        1.0,
        tolerances::GPU_SOFTMAX_SUM_F32,
    );

    let fitnesses = vec![0.1, 0.5, 0.9, 0.3, 0.7];
    let cpu_boltz = neural_spring::counterdiabatic::boltzmann_distribution(&fitnesses, 2.0);
    let gpu_boltz = disp.boltzmann(&fitnesses, 2.0);
    h.check_upper(
        "boltzmann max_diff",
        max_abs_diff_f64(&cpu_boltz, &gpu_boltz),
        tolerances::GPU_BOLTZMANN_F32,
    );
    h.check_abs(
        "boltzmann sums to 1",
        gpu_boltz.iter().sum::<f64>(),
        1.0,
        tolerances::GPU_SOFTMAX_SUM_F32,
    );
}

fn validate_reductions(h: &mut ValidationHarness, disp: &Dispatcher) {
    let vec_a = vec![1.0, 2.0, 3.0];
    let vec_b = vec![4.0, 5.0, 6.0];
    let cpu_l2 = neural_spring::modes::l2_distance(&vec_a, &vec_b);
    let gpu_l2 = disp.l2_distance(&vec_a, &vec_b);
    h.check_abs(
        "L2 distance",
        gpu_l2,
        cpu_l2,
        tolerances::GPU_L2_DISPATCH_F32,
    );

    let data10: Vec<f64> = (1..=10).map(f64::from).collect();
    let cpu_mean = data10.iter().sum::<f64>() / 10.0;
    let gpu_mean = disp.mean(&data10);
    h.check_abs(
        "mean [1..10]",
        gpu_mean,
        cpu_mean,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );

    let data_var = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let var_mean = data_var.iter().sum::<f64>() / data_var.len() as f64;
    let cpu_var = data_var
        .iter()
        .map(|&x| (x - var_mean).powi(2))
        .sum::<f64>()
        / data_var.len() as f64;
    let gpu_var = disp.variance(&data_var);
    h.check_abs(
        "variance",
        gpu_var,
        cpu_var,
        tolerances::GPU_VARIANCE_DISPATCH_F32,
    );

    let uniform4 = vec![0.25; 4];
    let cpu_entropy = neural_spring::primitives::shannon_entropy(&uniform4);
    let gpu_entropy = disp.shannon_entropy(&uniform4);
    h.check_abs(
        "Shannon entropy uniform(4)",
        gpu_entropy,
        cpu_entropy,
        tolerances::GPU_ENTROPY_F32,
    );
}

fn validate_statistics(h: &mut ValidationHarness, disp: &Dispatcher) {
    let px = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let py = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let gpu_r_pos = disp.pearson_correlation(&px, &py);
    h.check_abs(
        "Pearson r(x, 2x) ≈ 1",
        gpu_r_pos,
        1.0,
        tolerances::GPU_PEARSON_F32,
    );

    let ny = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let gpu_r_neg = disp.pearson_correlation(&px, &ny);
    h.check_abs(
        "Pearson r(x, rev(x)) ≈ -1",
        gpu_r_neg,
        -1.0,
        tolerances::GPU_PEARSON_F32,
    );

    let obs = vec![10.0, 20.0, 30.0, 40.0];
    let expected = vec![25.0, 25.0, 25.0, 25.0];
    let cpu_chi2: f64 = obs
        .iter()
        .zip(expected.iter())
        .map(|(&o, &e): (&f64, &f64)| (o - e).powi(2) / e)
        .sum();
    let gpu_chi2 = disp.chi_squared(&obs, &expected);
    h.check_abs(
        "chi-squared",
        gpu_chi2,
        cpu_chi2,
        tolerances::GPU_CHI_SQUARED_F32,
    );
}

fn validate_hmm(h: &mut ValidationHarness, disp: &Dispatcher) {
    let Some(device) = disp.wgpu_device() else {
        return;
    };
    let n_states = 3;
    let alpha = vec![0.4, 0.3, 0.3];
    let trans = vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.2, 0.6];
    let emit = vec![0.5, 0.3, 0.2];

    match neural_spring::gpu_ops::hmm_forward_step_gpu(&alpha, &trans, &emit, n_states, device) {
        Ok((alpha_new, scale)) => {
            let sum: f64 = alpha_new.iter().sum();
            h.check_abs(
                "HMM fwd step normalized",
                sum,
                1.0,
                tolerances::GPU_HMM_STEP_F32,
            );
            h.check_lower("HMM fwd step scale > 0", scale, 0.0);
        }
        Err(e) => h.check_bool(&format!("HMM fwd step: {e}"), false),
    }
}

fn validate_tensor_ops(h: &mut ValidationHarness, disp: &Dispatcher) {
    let Some(device) = disp.wgpu_device() else {
        return;
    };

    match neural_spring::gpu_ops::gelu_gpu(&[-2.0, -1.0, 0.0, 0.5, 1.0, 3.0], device) {
        Ok(r) => {
            h.check_abs("GELU(0)", r[2], 0.0, tolerances::GPU_GELU_F32);
            h.check_lower("GELU(3) > 2.9", r[5], 2.9);
        }
        Err(e) => h.check_bool(&format!("GELU: {e}"), false),
    }

    match neural_spring::gpu_ops::sum_gpu(&[1.0, 2.0, 3.0, 4.0, 5.0], device) {
        Ok(s) => h.check_abs("sum_gpu [1..5]", s, 15.0, tolerances::GPU_SUM_DISPATCH_F32),
        Err(e) => h.check_bool(&format!("sum: {e}"), false),
    }

    match neural_spring::gpu_ops::max_gpu(&[1.0, 5.0, 3.0, 2.0, 4.0], device) {
        Ok(m) => h.check_abs("max_gpu", m, 5.0, tolerances::GPU_MAX_DISPATCH_F32),
        Err(e) => h.check_bool(&format!("max: {e}"), false),
    }

    match neural_spring::gpu_ops::kl_divergence_gpu(
        &[0.25, 0.25, 0.25, 0.25],
        &[0.25, 0.25, 0.25, 0.25],
        device,
    ) {
        Ok(kl) => h.check_abs(
            "KL(uniform, uniform)",
            kl,
            0.0,
            tolerances::GPU_KL_DISPATCH_F32,
        ),
        Err(e) => h.check_bool(&format!("KL: {e}"), false),
    }

    let nn_params = neural_spring::gpu_ops::NeuralForwardParams {
        weights_hidden: &[0.5, 0.3, 0.7, 0.2],
        bias_hidden: &[0.1, 0.2, 0.3, 0.4],
        weights_output: &[
            0.6, 0.4, 0.8, 0.1, 0.5, 0.3, 0.7, 0.2, 0.9, 0.4, 0.6, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4,
            0.1, 0.9, 0.5,
        ],
        bias_output: &[0.1, 0.2, 0.3, 0.4, 0.5],
        input: &[0.5],
        hidden_size: 4,
        output_size: 5,
    };
    match neural_spring::gpu_ops::neural_forward_gpu(&nn_params, device) {
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
