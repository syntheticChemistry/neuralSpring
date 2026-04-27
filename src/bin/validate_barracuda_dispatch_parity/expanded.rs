// SPDX-License-Identifier: AGPL-3.0-or-later

//! S115/S127 expanded dispatch parity: bio ops, ODE integration, HMM
//! chain operations, popgen introgression, and glucose domain (Paper 026).

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::gpu_ops::SwarmNnDims;
use neural_spring::hmm::Hmm;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

// ─── S115: bio, ODE, HMM, popgen ops ────────────────────────────────

pub fn validate_multi_obj_fitness(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let pop_size = 8;
    let genome_len = 6;
    let n_objectives = 3;
    let genotypes: Vec<f64> = (0..pop_size * genome_len)
        .map(|i| (i as f64).mul_add(0.1, 0.05).sin().abs())
        .collect();
    let g = gpu.multi_obj_fitness(&genotypes, pop_size, genome_len, n_objectives);
    let c = cpu.multi_obj_fitness(&genotypes, pop_size, genome_len, n_objectives);
    assert_eq!(g.len(), c.len(), "multi_obj_fitness length mismatch");
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "multi_obj_fitness CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_MATMUL_F32,
    );
}

pub fn validate_swarm_nn_forward(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let dims = SwarmNnDims {
        n_controllers: 4,
        n_evals: 2,
        input_dim: 1,
        hidden_dim: 4,
        output_dim: 5,
    };
    let weights_per = dims.input_dim * dims.hidden_dim
        + dims.hidden_dim
        + dims.hidden_dim * dims.output_dim
        + dims.output_dim;
    let weights: Vec<f64> = (0..dims.n_controllers * weights_per)
        .map(|i| (i as f64 * 0.3).sin())
        .collect();
    let inputs: Vec<f64> = (0..dims.n_controllers * dims.n_evals * dims.input_dim)
        .map(|i| (i as f64 * 0.7).cos())
        .collect();
    let g = gpu.swarm_nn_forward(&weights, &inputs, &dims);
    let c = cpu.swarm_nn_forward(&weights, &inputs, &dims);
    h.check_bool("swarm_nn_forward CPU↔GPU action vectors match", g == c);
}

pub fn validate_integrate_ode_batch(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let dim = 2;
    let n_systems = 4;
    let n_steps = 100;
    let dt = 0.01;
    let states: Vec<f64> = (0..n_systems * dim)
        .map(|i| (i as f64).mul_add(0.05, 0.1))
        .collect();
    let n_coeffs = dim * 3;
    let coeffs: Vec<f64> = (0..n_systems * n_coeffs)
        .map(|i| ((i % n_coeffs) as f64).mul_add(0.1, 0.5))
        .collect();
    let g = gpu.integrate_ode_batch(&states, &coeffs, n_systems, dim, n_steps, dt);
    let c = cpu.integrate_ode_batch(&states, &coeffs, n_systems, dim, n_steps, dt);
    assert_eq!(g.len(), c.len(), "ODE batch output length mismatch");
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "integrate_ode_batch CPU↔GPU max diff",
        max_diff,
        tolerances::GPU_RK4_F32,
    );
}

pub fn validate_inter_pop_af_variance(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
) {
    let n_loci = 2;
    let pop_a: Vec<f64> = vec![2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.5, 0.5];
    let pop_b: Vec<f64> = vec![0.0, 2.0, 0.5, 1.5, 2.0, 0.0, 0.5, 1.5];
    let populations: Vec<&[f64]> = vec![&pop_a, &pop_b];
    let n_individuals = vec![4, 4];
    let g = gpu.inter_population_af_variance(&populations, &n_individuals, n_loci);
    let c = cpu.inter_population_af_variance(&populations, &n_individuals, n_loci);
    h.check_abs(
        "inter_pop_af_variance CPU↔GPU",
        g,
        c,
        tolerances::TENSOR_EXACT_F32,
    );
}

pub fn validate_hmm_backward_step(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let beta_next = vec![0.5, 0.5];
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emit_col = vec![0.1, 0.6];
    let scale = 0.5;
    let g = gpu.hmm_backward_step(&beta_next, &trans, &emit_col, scale, 2);
    let c = cpu.hmm_backward_step(&beta_next, &trans, &emit_col, scale, 2);
    let max_diff = g
        .iter()
        .zip(c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "hmm_backward_step CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

pub fn validate_hmm_viterbi_step(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let delta_prev = vec![-0.5, -1.0];
    let log_trans: Vec<f64> = vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()];
    let log_emit = vec![0.1_f64.ln(), 0.6_f64.ln()];
    let (g_delta, g_psi) = gpu.hmm_viterbi_step(&delta_prev, &log_trans, &log_emit, 2);
    let (c_delta, c_psi) = cpu.hmm_viterbi_step(&delta_prev, &log_trans, &log_emit, 2);
    let max_diff = g_delta
        .iter()
        .zip(c_delta.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "hmm_viterbi_step delta CPU↔GPU max diff",
        max_diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
    h.check_bool("hmm_viterbi_step psi CPU↔GPU", g_psi == c_psi);
}

pub fn validate_hmm_chain(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emission = vec![0.1, 0.4, 0.5, 0.6, 0.3, 0.1];
    let initial = vec![0.6, 0.4];
    let obs = vec![0, 1, 2, 0, 1];
    let (g_path, g_prob, g_lik) = gpu.hmm_chain(&initial, &trans, &emission, &obs, 2, 3);
    let (c_path, c_prob, c_lik) = cpu.hmm_chain(&initial, &trans, &emission, &obs, 2, 3);
    h.check_bool("hmm_chain path CPU↔GPU", g_path == c_path);
    h.check_abs(
        "hmm_chain log_prob CPU↔GPU",
        g_prob,
        c_prob,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
    h.check_abs(
        "hmm_chain log_lik CPU↔GPU",
        g_lik,
        c_lik,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

pub fn validate_detect_introgression(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
) {
    let trans = vec![0.95, 0.05, 0.10, 0.90];
    let emission = vec![0.9, 0.1, 0.1, 0.9];
    let initial = vec![0.7, 0.3];
    let hmm = Hmm::from_flat(trans, emission, initial, 2, 2);
    let obs = vec![0, 0, 0, 1, 1, 1, 0, 0];
    let (g_path, g_prob) = gpu.detect_introgression(&hmm, &obs);
    let (c_path, c_prob) = cpu.detect_introgression(&hmm, &obs);
    h.check_bool("detect_introgression path CPU↔GPU", g_path == c_path);
    h.check_abs(
        "detect_introgression log_prob CPU↔GPU",
        g_prob,
        c_prob,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

// ─── S127: Paper 026 glucose domain ──────────────────────────────────

pub fn validate_glucose_variance_pearson(
    h: &mut ValidationHarness,
    gpu: &Dispatcher,
    cpu: &Dispatcher,
) {
    let glucose = neural_spring::glucose_prediction::generate_synthetic_cgm(7, 42);

    let g_var = gpu.variance(&glucose);
    let c_var = cpu.variance(&glucose);
    h.check_abs(
        "glucose variance CPU↔GPU (2016 pts)",
        g_var,
        c_var,
        tolerances::TENSOR_EXACT_F32,
    );

    let half = glucose.len() / 2;
    let g_pear = gpu.pearson_correlation(&glucose[..half], &glucose[half..half * 2]);
    let c_pear = cpu.pearson_correlation(&glucose[..half], &glucose[half..half * 2]);
    // DF64 correlation over 1008 elements: the f32-pair (48-bit mantissa)
    // 5-accumulator reduction accumulates more rounding than native f64.
    // On Hybrid FP64 strategy (Ada Lovelace, consumer GPUs) the DF64 path
    // is used; observed diff ~1.7e-5 for 1008 elements.
    h.check_abs(
        "glucose pearson CPU↔GPU (1008 vs 1008 pts)",
        g_pear,
        c_pear,
        tolerances::GPU_DF64_TRANSCENDENTAL,
    );
}
