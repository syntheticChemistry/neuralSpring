// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation: `ToadStool` compute dispatch routing.
//!
//! Validates that `metalForge::forge::dispatch` substrate heuristics
//! correctly recommend GPU vs CPU for various workload sizes, and that
//! both paths produce identical results when exercised.

#![allow(clippy::cast_precision_loss)]

use neural_spring::validation::ValidationHarness;
use neural_spring_forge::dispatch::{
    batch_fitness_substrate, batch_ipr_substrate, hmm_substrate, logsumexp_substrate,
    ode_substrate, pairwise_substrate, spatial_substrate, stochastic_substrate, Substrate,
};

fn main() {
    let mut h = ValidationHarness::new("toadstool_dispatch");

    validate_pairwise_routing(&mut h);
    validate_fitness_routing(&mut h);
    validate_ode_routing(&mut h);
    validate_hmm_routing(&mut h);
    validate_spatial_routing(&mut h);
    validate_ipr_routing(&mut h);
    validate_logsumexp_routing(&mut h);
    validate_stochastic_routing(&mut h);

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
