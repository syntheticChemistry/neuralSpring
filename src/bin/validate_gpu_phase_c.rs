// SPDX-License-Identifier: AGPL-3.0-or-later

//! Phase C GPU promotion validator: HMM chain (forward + Viterbi), FST,
//! inter-population AF variance, and introgression detection via GPU.
//!
//! Proves composed GPU operations match CPU references within tolerance.

#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::hmm::Hmm;
use neural_spring::meta_population;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let Ok(rt) = tokio::runtime::Runtime::new() else {
        eprintln!("FATAL: could not create tokio runtime");
        std::process::exit(1);
    };
    let dispatcher = rt.block_on(Dispatcher::new());

    let mut h = ValidationHarness::new("validate_gpu_phase_c");

    if !dispatcher.has_gpu() {
        eprintln!("WARNING: No GPU available — all checks use CPU fallback");
    }

    eprintln!(
        "Backend: {} ({})",
        dispatcher.backend(),
        dispatcher.adapter_name(),
    );

    // ═══════════════════════════════════════════════════════════════
    // HMM forward chain (Papers 016–018)
    // ═══════════════════════════════════════════════════════════════

    let hmm2 = Hmm::from_flat(
        vec![0.7, 0.3, 0.4, 0.6],
        vec![0.5, 0.4, 0.1, 0.1, 0.3, 0.6],
        vec![0.6, 0.4],
        2,
        3,
    );
    let obs2 = [0_usize, 1, 2, 0, 1, 2, 0, 1];
    let cpu_fwd = hmm2.forward(&obs2);

    let gpu_ll = dispatcher.hmm_forward_chain(
        &hmm2.initial,
        &hmm2.transition,
        &hmm2.emission,
        &obs2,
        2,
        3,
    );
    h.check_abs(
        "hmm2 forward chain log-likelihood",
        gpu_ll,
        cpu_fwd.1,
        tolerances::GPU_HMM_LOG_LIKELIHOOD_F32,
    );

    // 4-state HMM
    let hmm4 = Hmm::from_flat(
        vec![
            0.6, 0.2, 0.1, 0.1, 0.15, 0.5, 0.2, 0.15, 0.1, 0.15, 0.6, 0.15, 0.2, 0.1, 0.15,
            0.55,
        ],
        vec![
            0.3, 0.2, 0.2, 0.15, 0.15, 0.1, 0.3, 0.25, 0.2, 0.15, 0.15, 0.15, 0.3, 0.25, 0.15,
            0.25, 0.2, 0.15, 0.15, 0.25,
        ],
        vec![0.3, 0.3, 0.2, 0.2],
        4,
        5,
    );

    let mut rng = Rng::new(42);
    let (_, obs4) = hmm4.generate_sequence(50, &mut rng);
    let cpu_fwd4 = hmm4.forward(&obs4);

    let gpu_ll4 = dispatcher.hmm_forward_chain(
        &hmm4.initial,
        &hmm4.transition,
        &hmm4.emission,
        &obs4,
        4,
        5,
    );
    h.check_abs(
        "hmm4 forward chain log-likelihood",
        gpu_ll4,
        cpu_fwd4.1,
        tolerances::GPU_HMM_LOG_LIKELIHOOD_F32,
    );

    // 100-step chain
    let (_, obs100) = hmm4.generate_sequence(100, &mut rng);
    let cpu_fwd100 = hmm4.forward(&obs100);
    let gpu_ll100 = dispatcher.hmm_forward_chain(
        &hmm4.initial,
        &hmm4.transition,
        &hmm4.emission,
        &obs100,
        4,
        5,
    );
    h.check_abs(
        "hmm4 forward chain 100-step",
        gpu_ll100,
        cpu_fwd100.1,
        tolerances::GPU_HMM_LOG_LIKELIHOOD_F32 * 2.0,
    );

    // ═══════════════════════════════════════════════════════════════
    // HMM Viterbi chain (Papers 016–018)
    // ═══════════════════════════════════════════════════════════════

    let cpu_vit2 = hmm2.viterbi(&obs2);
    let (gpu_path2, gpu_lp2) = dispatcher.hmm_viterbi_chain(
        &hmm2.initial,
        &hmm2.transition,
        &hmm2.emission,
        &obs2,
        2,
        3,
    );
    h.check_bool("hmm2 viterbi chain path matches", gpu_path2 == cpu_vit2.0);
    h.check_abs(
        "hmm2 viterbi chain log-prob",
        gpu_lp2,
        cpu_vit2.1,
        tolerances::GPU_HMM_VITERBI_LOGPROB_F64,
    );

    let cpu_vit4 = hmm4.viterbi(&obs4);
    let (gpu_path4, gpu_lp4) = dispatcher.hmm_viterbi_chain(
        &hmm4.initial,
        &hmm4.transition,
        &hmm4.emission,
        &obs4,
        4,
        5,
    );
    h.check_bool("hmm4 viterbi chain path matches", gpu_path4 == cpu_vit4.0);
    h.check_abs(
        "hmm4 viterbi chain log-prob",
        gpu_lp4,
        cpu_vit4.1,
        tolerances::GPU_HMM_VITERBI_LOGPROB_F64,
    );

    // Viterbi on longer sequence
    let (_, obs_long) = hmm4.generate_sequence(200, &mut rng);
    let cpu_vit_long = hmm4.viterbi(&obs_long);
    let (gpu_path_long, gpu_lp_long) = dispatcher.hmm_viterbi_chain(
        &hmm4.initial,
        &hmm4.transition,
        &hmm4.emission,
        &obs_long,
        4,
        5,
    );
    // f32 accumulation over 200 steps can cause boundary argmax differences;
    // require >=90% path agreement rather than exact match
    let agree: usize = gpu_path_long
        .iter()
        .zip(cpu_vit_long.0.iter())
        .filter(|(a, b)| a == b)
        .count();
    let agreement = agree as f64 / obs_long.len() as f64;
    h.check_lower(
        "hmm4 viterbi chain 200-step path agreement",
        agreement,
        0.90,
    );
    h.check_abs(
        "hmm4 viterbi chain 200-step log-prob",
        gpu_lp_long,
        cpu_vit_long.1,
        tolerances::GPU_HMM_VITERBI_LOGPROB_F64 * 2.0,
    );

    // ═══════════════════════════════════════════════════════════════
    // Introgression detection via HMM Viterbi chain (Paper 018)
    // ═══════════════════════════════════════════════════════════════

    let intro_hmm = Hmm::from_flat(
        vec![0.95, 0.05, 0.05, 0.95],
        vec![0.7, 0.2, 0.1, 0.2, 0.3, 0.5],
        vec![0.8, 0.2],
        2,
        3,
    );
    let intro_obs = [0_usize, 0, 0, 2, 2, 2, 1, 0, 0, 0];
    let cpu_intro = intro_hmm.viterbi(&intro_obs);
    let (gpu_intro_path, gpu_intro_lp) = dispatcher.hmm_viterbi_chain(
        &intro_hmm.initial,
        &intro_hmm.transition,
        &intro_hmm.emission,
        &intro_obs,
        2,
        3,
    );
    h.check_bool(
        "introgression viterbi path matches",
        gpu_intro_path == cpu_intro.0,
    );
    h.check_abs(
        "introgression viterbi log-prob",
        gpu_intro_lp,
        cpu_intro.1,
        tolerances::GPU_HMM_VITERBI_LOGPROB_F64,
    );

    // ═══════════════════════════════════════════════════════════════
    // Pairwise FST (Paper 025)
    // ═══════════════════════════════════════════════════════════════

    let n_loci = 20;
    let ancestral: Vec<f64> = (0..n_loci).map(|_| rng.beta(2.0, 2.0)).collect();

    let pop_a = meta_population::generate_population(
        10, n_loci, &ancestral, 0.15, 70.0, 65.0, 90.0, 2, &mut rng,
    );
    let pop_b = meta_population::generate_population(
        10, n_loci, &ancestral, 0.15, 85.0, 65.0, 90.0, 2, &mut rng,
    );

    let cpu_fst = meta_population::pairwise_fst(&pop_a, 10, &pop_b, 10, n_loci);
    let gpu_fst = dispatcher.pairwise_fst(&pop_a, 10, &pop_b, 10, n_loci);
    // f32 allele frequency intermediary widens FST diff to ~0.05
    h.check_abs("pairwise_fst", gpu_fst, cpu_fst, 0.1);

    // FST finite check
    h.check_bool("pairwise_fst finite", gpu_fst.is_finite());

    // ═══════════════════════════════════════════════════════════════
    // Global FST (Paper 025)
    // ═══════════════════════════════════════════════════════════════

    let pop_c = meta_population::generate_population(
        10, n_loci, &ancestral, 0.15, 78.0, 65.0, 90.0, 2, &mut rng,
    );

    let pops = vec![pop_a.clone(), pop_b.clone(), pop_c];
    let n_indivs = vec![10_usize, 10, 10];
    let cpu_global_fst = meta_population::global_fst(&pops, &n_indivs, n_loci);
    let gpu_global_fst = dispatcher.global_fst(&pops, &n_indivs, n_loci);
    h.check_abs("global_fst 3-pop", gpu_global_fst, cpu_global_fst, 0.1);
    h.check_bool("global_fst finite", gpu_global_fst.is_finite());

    // ═══════════════════════════════════════════════════════════════
    // Inter-population AF variance via Dispatcher (Paper 025)
    // ═══════════════════════════════════════════════════════════════

    let pop_refs: Vec<&[f64]> = pops.iter().map(Vec::as_slice).collect();
    let cpu_var = meta_population::inter_population_af_variance(&pops, &n_indivs, n_loci);
    let gpu_var = dispatcher.inter_population_af_variance(&pop_refs, &n_indivs, n_loci);
    h.check_abs(
        "inter_pop_af_variance dispatch",
        gpu_var,
        cpu_var,
        tolerances::GPU_AF_VARIANCE_F32,
    );

    // ═══════════════════════════════════════════════════════════════
    // Cross-validation: FST consistency
    // ═══════════════════════════════════════════════════════════════

    let fst_ab = dispatcher.pairwise_fst(&pop_a, 10, &pop_b, 10, n_loci);
    h.check_bool(
        "fst consistency: pairwise finite",
        fst_ab.is_finite(),
    );

    let fst_global = dispatcher.global_fst(&[pop_a, pop_b], &[10, 10], n_loci);
    h.check_bool(
        "fst consistency: global finite",
        fst_global.is_finite(),
    );

    h.finish();
}
