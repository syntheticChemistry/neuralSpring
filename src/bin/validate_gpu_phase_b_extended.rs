// SPDX-License-Identifier: AGPL-3.0-or-later

//! Phase B extended GPU validator: FST variance decomposition and introgression HMM chain.
//!
//! Closes remaining `PURE_GPU_ROADMAP` Phase B gaps:
//! - Gap 1: FST via variance decomposition (GPU primitives only)
//! - Gap 2: Introgression detection via HMM chain (forward + Viterbi)
//!
//! Validates GPU paths match CPU references within documented tolerance.

#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::introgression::{self, phylonet_hmm};
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

    let mut h = ValidationHarness::new("validate_gpu_phase_b_extended");

    if !dispatcher.has_gpu() {
        eprintln!("WARNING: No GPU available — all checks use CPU fallback");
    }

    eprintln!(
        "Backend: {} ({})",
        dispatcher.backend(),
        dispatcher.adapter_name(),
    );

    // ─── Gap 1: GPU FST matches CPU FST (meta_population::global_fst) ───────

    let mut rng = Rng::new(42);
    let n_loci = 20_usize;
    let anc: Vec<f64> = (0..n_loci).map(|_| rng.beta(2.0, 2.0)).collect();
    let pops: Vec<Vec<f64>> = [70.0, 85.0]
        .iter()
        .map(|&t| {
            meta_population::generate_population(10, n_loci, &anc, 0.15, t, 65.0, 90.0, 4, &mut rng)
        })
        .collect();
    let n_indivs = vec![10_usize; 2];

    let cpu_global_fst = meta_population::global_fst(&pops, &n_indivs, n_loci);
    let gpu_global_fst = dispatcher.global_fst(&pops, &n_indivs, n_loci);
    h.check_abs(
        "global_fst GPU vs CPU",
        gpu_global_fst,
        cpu_global_fst,
        tolerances::GPU_AF_VARIANCE_F32,
    );

    let cpu_pairwise =
        meta_population::pairwise_fst(&pops[0], n_indivs[0], &pops[1], n_indivs[1], n_loci);
    let gpu_pairwise =
        dispatcher.pairwise_fst(&pops[0], n_indivs[0], &pops[1], n_indivs[1], n_loci);
    h.check_abs(
        "pairwise_fst GPU vs CPU",
        gpu_pairwise,
        cpu_pairwise,
        tolerances::GPU_AF_VARIANCE_F32,
    );

    let cpu_fst_var = meta_population::global_fst_variance_decomposition(&pops, &n_indivs, n_loci);
    let gpu_fst_var = dispatcher.global_fst_variance_decomposition(&pops, &n_indivs, n_loci);
    h.check_abs(
        "global_fst_variance_decomposition GPU vs CPU",
        gpu_fst_var,
        cpu_fst_var,
        tolerances::GPU_AF_VARIANCE_F32,
    );

    h.check_bool(
        "global_fst finite",
        cpu_global_fst.is_finite() && gpu_global_fst.is_finite(),
    );

    // ─── Gap 2: GPU HMM chain matches CPU introgression detection ───────────

    let hmm = phylonet_hmm();
    let mut rng_obs = Rng::new(99);
    let (_, obs) = introgression::generate_synthetic_loci(200, &hmm, &mut rng_obs);

    let (cpu_path, cpu_logprob) = introgression::detect_introgression(&hmm, &obs);
    let (gpu_path, gpu_logprob) = dispatcher.detect_introgression(&hmm, &obs);

    h.check_bool("detect_introgression path matches", cpu_path == gpu_path);

    h.check_abs(
        "detect_introgression log_prob",
        gpu_logprob,
        cpu_logprob,
        tolerances::GPU_HMM_VITERBI_LOGPROB_F64,
    );

    h.check_bool(
        "detect_introgression path length",
        gpu_path.len() == obs.len(),
    );

    let (_, chain_logprob, chain_loglik) = dispatcher.hmm_chain(
        &hmm.initial,
        &hmm.transition,
        &hmm.emission,
        &obs,
        hmm.num_states(),
        hmm.num_symbols(),
    );

    h.check_abs(
        "hmm_chain log_prob vs detect_introgression",
        chain_logprob,
        cpu_logprob,
        tolerances::GPU_HMM_VITERBI_LOGPROB_F64,
    );

    h.check_bool("hmm_chain log_likelihood finite", chain_loglik.is_finite());

    h.finish();
}
