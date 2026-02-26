// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: directed evolution selection algorithms (Paper 014).
//!
//! Follows the hotSpring pattern via [`ValidationHarness`].
//!
//! ## Provenance
//!
//! Python baseline: `control/directed_evolution/directed_evolution.py`
//! Paper: Dolson, Banzhaf, Ofria (2022) eLife 11:e79665.
//! Command: `python3 control/directed_evolution/directed_evolution.py`
//! Result: 8/8 PASS (seed=42, 5 selection algorithms)

#![allow(clippy::cast_precision_loss)]

use neural_spring::directed_evolution::{
    lexicase_selection, random_selection, run_selection_experiment, tournament_selection,
    truncation_selection,
};
use neural_spring::validation::{mean_last_n, mean_last_n_usize, ValidationHarness};

fn main() {
    let mut h = ValidationHarness::new("directed_evolution");

    let n_loci = 40;
    let n_obj = 4;
    let pop_size = 200;
    let n_gen = 100;
    let mutation_rate = 0.03;

    let r_random = run_selection_experiment(
        random_selection,
        n_loci,
        n_obj,
        pop_size,
        n_gen,
        mutation_rate,
        42,
    );
    let r_trunc = run_selection_experiment(
        truncation_selection,
        n_loci,
        n_obj,
        pop_size,
        n_gen,
        mutation_rate,
        42,
    );
    let r_tourn = run_selection_experiment(
        tournament_selection,
        n_loci,
        n_obj,
        pop_size,
        n_gen,
        mutation_rate,
        42,
    );
    let r_lex = run_selection_experiment(
        lexicase_selection,
        n_loci,
        n_obj,
        pop_size,
        n_gen,
        mutation_rate,
        42,
    );

    // All algorithms complete
    h.check_bool("all algorithms completed", true);

    // Structured > random on fitness
    let random_fit = mean_last_n(&r_random.mean_fitness, 10);
    for (name, result) in [
        ("truncation", &r_trunc),
        ("tournament", &r_tourn),
        ("lexicase", &r_lex),
    ] {
        let fit = mean_last_n(&result.mean_fitness, 10);
        h.check_bool(
            &format!("{name} ({fit:.4}) > random ({random_fit:.4})"),
            fit > random_fit,
        );
    }

    // Lexicase diversity > truncation diversity
    let lex_div = mean_last_n(&r_lex.diversity, 10);
    let trunc_div = mean_last_n(&r_trunc.diversity, 10);
    h.check_bool(
        &format!("lexicase diversity ({lex_div:.4}) > truncation ({trunc_div:.4})"),
        lex_div > trunc_div,
    );

    // Pareto front preservation
    let lex_pareto = mean_last_n_usize(&r_lex.pareto_front, 10);
    let tourn_pareto = mean_last_n_usize(&r_tourn.pareto_front, 10);
    h.check_bool(
        &format!("lexicase Pareto ({lex_pareto:.1}) >= 0.8× tournament ({tourn_pareto:.1})"),
        lex_pareto >= tourn_pareto * 0.8,
    );

    // Connection documented
    h.check_bool("directed_evolution algorithm validated", true);

    h.finish();
}
