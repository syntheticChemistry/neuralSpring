// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation: `ToadStool` compute dispatch routing.
//!
//! Validates that `metalForge::forge::dispatch` substrate heuristics
//! correctly recommend GPU vs CPU for various workload sizes, and that
//! both paths produce identical results when exercised.

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::many_single_char_names
)]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use neural_spring_forge::dispatch::{
    batch_fitness_substrate, batch_ipr_substrate, hmm_substrate, logsumexp_substrate,
    ode_substrate, pairwise_substrate, spatial_substrate, stochastic_substrate, Substrate,
};

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("toadstool_dispatch");

    validate_pairwise_routing(&mut h);
    validate_fitness_routing(&mut h);
    validate_ode_routing(&mut h);
    validate_hmm_routing(&mut h);
    validate_spatial_routing(&mut h);
    validate_ipr_routing(&mut h);
    validate_logsumexp_routing(&mut h);
    validate_stochastic_routing(&mut h);

    validate_compute_parity(&mut h).await;

    h.finish();
}

fn validate_pairwise_routing(h: &mut ValidationHarness) {
    // pairwise: estimated_work > 500_000 → GPU. 20×500 → work=95_000 → CPU.
    h.check_bool(
        "pairwise 20×500 → CPU",
        pairwise_substrate(20, 500) == Substrate::Cpu,
    );
    // 200×1000 → work=19_900_000 → GPU.
    h.check_bool(
        "pairwise 200×1000 → GPU",
        pairwise_substrate(200, 1000) == Substrate::Gpu,
    );
}

fn validate_fitness_routing(h: &mut ValidationHarness) {
    // batch_fitness: total_work > 50_000 → GPU. 100×100 → CPU.
    h.check_bool(
        "batch_fitness 100×100 → CPU",
        batch_fitness_substrate(100, 100) == Substrate::Cpu,
    );
    // 1000×100 → GPU.
    h.check_bool(
        "batch_fitness 1000×100 → GPU",
        batch_fitness_substrate(1000, 100) == Substrate::Gpu,
    );
}

fn validate_ode_routing(h: &mut ValidationHarness) {
    // ode: total_work > 10_000 → GPU. 10×100 → CPU.
    h.check_bool("ode 10×100 → CPU", ode_substrate(10, 100) == Substrate::Cpu);
    // 100×200 → GPU.
    h.check_bool(
        "ode 100×200 → GPU",
        ode_substrate(100, 200) == Substrate::Gpu,
    );
}

fn validate_hmm_routing(h: &mut ValidationHarness) {
    // hmm: total_work > 5_000 → GPU. 3×100 → CPU.
    h.check_bool("hmm 3×100 → CPU", hmm_substrate(3, 100) == Substrate::Cpu);
    // 10×1000 → GPU.
    h.check_bool(
        "hmm 10×1000 → GPU",
        hmm_substrate(10, 1000) == Substrate::Gpu,
    );
}

fn validate_spatial_routing(h: &mut ValidationHarness) {
    // spatial: grid_cells > 4_000 → GPU. 100 → CPU.
    h.check_bool(
        "spatial 100 → CPU",
        spatial_substrate(100) == Substrate::Cpu,
    );
    // 10_000 → GPU.
    h.check_bool(
        "spatial 10_000 → GPU",
        spatial_substrate(10_000) == Substrate::Gpu,
    );
}

fn validate_ipr_routing(h: &mut ValidationHarness) {
    // batch_ipr: total_work > 50_000 → GPU. 100×100 → CPU.
    h.check_bool(
        "batch_ipr 100×100 → CPU",
        batch_ipr_substrate(100, 100) == Substrate::Cpu,
    );
    // 1000×100 → GPU.
    h.check_bool(
        "batch_ipr 1000×100 → GPU",
        batch_ipr_substrate(1000, 100) == Substrate::Gpu,
    );
}

fn validate_logsumexp_routing(h: &mut ValidationHarness) {
    // logsumexp: total_work > 20_000 → GPU. 100×100 → CPU.
    h.check_bool(
        "logsumexp 100×100 → CPU",
        logsumexp_substrate(100, 100) == Substrate::Cpu,
    );
    // 500×100 → GPU.
    h.check_bool(
        "logsumexp 500×100 → GPU",
        logsumexp_substrate(500, 100) == Substrate::Gpu,
    );
}

fn validate_stochastic_routing(h: &mut ValidationHarness) {
    // stochastic: total_work > 100_000 → GPU. 10×10×100 → CPU.
    h.check_bool(
        "stochastic 10×10×100 → CPU",
        stochastic_substrate(10, 10, 100) == Substrate::Cpu,
    );
    // 100×100×20 → GPU.
    h.check_bool(
        "stochastic 100×100×20 → GPU",
        stochastic_substrate(100, 100, 20) == Substrate::Gpu,
    );
}

async fn validate_compute_parity(h: &mut ValidationHarness) {
    let Ok(gpu) = Gpu::new().await else {
        h.check_bool("compute_parity: GPU available", false);
        return;
    };
    let gpu_disp = Dispatcher::from_gpu(gpu);
    let cpu_disp = Dispatcher::cpu_only();

    // HMM forward chain: same log-likelihood on CPU and GPU
    let trans = vec![0.7, 0.3, 0.4, 0.6];
    let emit = vec![0.1, 0.4, 0.5, 0.6, 0.3, 0.1];
    let init = vec![0.6, 0.4];
    let obs = vec![0, 1, 2, 0, 1];
    let g_ll = gpu_disp.hmm_forward_chain(&init, &trans, &emit, &obs, 2, 3);
    let c_ll = cpu_disp.hmm_forward_chain(&init, &trans, &emit, &obs, 2, 3);
    h.check_abs(
        "compute_parity: HMM forward chain CPU↔GPU",
        g_ll,
        c_ll,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    // Variance: same result on both paths
    let data: Vec<f64> = (0..100).map(|i| f64::from(i).sin()).collect();
    let g_var = gpu_disp.variance(&data);
    let c_var = cpu_disp.variance(&data);
    h.check_abs(
        "compute_parity: variance CPU↔GPU",
        g_var,
        c_var,
        tolerances::TENSOR_EXACT_F32,
    );

    // Eigendecomposition: same eigenvalues
    let sym = vec![4.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 5.0];
    let (mut g_evals, _) = gpu_disp.eigh(&sym, 3);
    let (mut c_evals, _) = cpu_disp.eigh(&sym, 3);
    g_evals.sort_by(f64::total_cmp);
    c_evals.sort_by(f64::total_cmp);
    let eval_diff = g_evals
        .iter()
        .zip(c_evals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "compute_parity: eigh eigenvalues CPU↔GPU",
        eval_diff,
        tolerances::TENSOR_MATMUL_F32,
    );

    // Matmul: same matrix product
    let n = 8;
    let a: Vec<f64> = (0..n * n).map(|i| ((i * 7 + 3) % 17) as f64).collect();
    let b: Vec<f64> = (0..n * n).map(|i| ((i * 11 + 5) % 13) as f64).collect();
    let g_prod = gpu_disp.mat_mul(&a, &b, n);
    let c_prod = cpu_disp.mat_mul(&a, &b, n);
    let mm_diff = g_prod
        .iter()
        .zip(c_prod.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "compute_parity: matmul 8×8 CPU↔GPU",
        mm_diff,
        tolerances::TENSOR_MATMUL_F32,
    );

    // Shannon entropy: same result
    let p = vec![0.1, 0.2, 0.3, 0.4];
    let g_ent = gpu_disp.shannon_entropy(&p);
    let c_ent = cpu_disp.shannon_entropy(&p);
    h.check_abs(
        "compute_parity: entropy CPU↔GPU",
        g_ent,
        c_ent,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    // Popgen: allele frequencies
    let pop = vec![2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.5, 0.5];
    let g_af = gpu_disp.allele_frequencies(&pop, 4, 2);
    let c_af = cpu_disp.allele_frequencies(&pop, 4, 2);
    let af_diff = g_af
        .iter()
        .zip(c_af.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_upper(
        "compute_parity: allele_freq CPU↔GPU",
        af_diff,
        tolerances::TENSOR_EXACT_F32,
    );
}
