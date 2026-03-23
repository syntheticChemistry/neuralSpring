// SPDX-License-Identifier: AGPL-3.0-or-later

//! Modern cross-spring evolution validator: exercises `BarraCUDA` S86 universal
//! precision pipeline and traces shader provenance across all five springs.
//!
//! ## Provenance map
//!
//! ```text
//! hotSpring  → DF64 core-streaming, Fp64Strategy, split_workgroups, lattice QCD
//!              → `BarraCUDA` S68+: precision-routed compilation F32/F64/Df64
//! wetSpring  → diversity (Shannon, Bray-Curtis), bio (Smith-Waterman, Gillespie,
//!              Felsenstein, HMM), NMF, ODE bio
//! neuralSpring → batch_fitness, pairwise ops, eigh, swarm_nn, ValidationHarness
//! airSpring  → hydrology, regression, moving_window, stats metrics
//! groundSpring → bootstrap (rawr_mean), multinomial sampling
//! ```
//!
//! Each check is annotated with its provenance chain showing the evolution path
//! from source spring → `ToadStool` / `BarraCUDA` → neuralSpring.
//!
//! ```text
//! cargo run --bin validate_modern_cross_spring
//! ```

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::expect_used,
    reason = "validation binary"
)]

mod airspring_stats;
mod groundspring_bootstrap;
mod hotspring_precision;
mod neuralspring_dispatch;
mod report;
mod throughput;
mod toadstool_s68;
mod toadstool_s86;
mod wetspring_bio;

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::validation::ValidationHarness;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("modern_cross_spring");

    let dispatcher = Dispatcher::new().await;

    println!(
        "[modern] backend={}, gpu={}, adapter={}, fp64={:?}",
        dispatcher.backend(),
        dispatcher.has_gpu(),
        dispatcher.adapter_name(),
        dispatcher.fp64_strategy(),
    );

    hotspring_precision::validate_hotspring_precision(&mut h, &dispatcher);
    wetspring_bio::validate_wetspring_bio(&mut h);
    airspring_stats::validate_airspring_stats(&mut h);
    groundspring_bootstrap::validate_groundspring_bootstrap(&mut h);
    neuralspring_dispatch::validate_neuralspring_dispatch(&mut h, &dispatcher);
    toadstool_s68::validate_toadstool_s68_precision(&mut h, &dispatcher);
    toadstool_s86::validate_toadstool_s86_evolution(&mut h);
    throughput::benchmark_cross_spring_throughput(&mut h, &dispatcher);

    report::report_provenance_summary();

    h.finish();
}
