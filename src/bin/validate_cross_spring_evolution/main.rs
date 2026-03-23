// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-spring evolution benchmark: validates and benchmarks functions
//! rewired to upstream `barracuda` APIs, and reports driver profile information
//! from `DeviceCapabilities` (hotSpring-evolved).
//!
//! ## What this proves
//!
//! - **Upstream rewiring**: Dispatcher methods + library functions delegate
//!   to upstream `BarraCUDA` and produce correct results
//! - **Cross-spring evolution**: shaders and dispatch logic evolved from
//!   hotSpring (precision), wetSpring (bio), and neuralSpring (validation)
//! - **Driver awareness**: `DeviceCapabilities` correctly detects hardware
//!   and selects appropriate f64 strategy
//! - **Performance**: benchmarks upstream dispatch vs local CPU reference
//! - **S72 rewires**: `softmax_dim(axis)`, `argmax_dim(axis)`,
//!   `fst_variance_decomposition` — APIs previously blocked, now absorbed
//!
//! ## Cross-spring shader lineage
//!
//! ```text
//! hotSpring → df64_core, pow_f64, Taylor trig, Lanczos → BarraCUDA precision
//! wetSpring → HMM forward, ODE bio, NMF, Anderson, FST → BarraCUDA bio+spectral
//! neuralSpring → batch_fitness, pairwise_l2, eigh, ValidationHarness → BarraCUDA ops
//! All three → `ToadStool` (GPU sovereign pipeline)
//! ```
//!
//! ```text
//! cargo run --release --bin validate_cross_spring_evolution
//! ```
//!
//! ## Provenance
//!
//! Cross-spring origin: hotSpring, wetSpring, neuralSpring → `BarraCUDA`/`ToadStool` → neuralSpring.
//! Absorption: S72 rewires (`softmax_dim`, `argmax_dim`, `fst_variance_decomposition`), driver profile.
//! Validation: Dispatcher methods vs upstream `BarraCUDA`, `DeviceCapabilities` hardware detection, CPU reference parity.

#![expect(
    clippy::cast_precision_loss,
    clippy::similar_names,
    reason = "validation binary"
)]

mod device_caps;
mod dispatch_s58;
mod helpers;
mod lineage;
mod s59_gelu_spectral;
mod s72_tensor_fst;
mod s78_primitives;
mod throughput_benchmark;

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::validation::ValidationHarness;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("cross_spring_evolution");

    let dispatcher = Dispatcher::new().await;
    let cpu = Dispatcher::cpu_only();

    println!(
        "[evolution] GPU: {} ({}), f64 strategy: {:?}, pow workaround: {}",
        dispatcher.has_gpu(),
        dispatcher.adapter_name(),
        dispatcher.fp64_strategy(),
        dispatcher.needs_pow_workaround(),
    );

    println!("\n--- Rewired Dispatcher Methods (S58) ---\n");
    dispatch_s58::validate_rewired_matmul(&mut h, &dispatcher, &cpu);
    dispatch_s58::validate_rewired_frobenius(&mut h, &dispatcher, &cpu);
    dispatch_s58::validate_rewired_transpose(&mut h, &dispatcher, &cpu);
    dispatch_s58::validate_rewired_softmax(&mut h, &dispatcher, &cpu);
    dispatch_s58::validate_rewired_l2(&mut h, &dispatcher, &cpu);
    dispatch_s58::validate_rewired_mean(&mut h, &dispatcher, &cpu);
    dispatch_s58::validate_rewired_variance(&mut h, &dispatcher, &cpu);

    println!("\n--- Rewired S59: Dispatcher + Library Functions ---\n");
    s59_gelu_spectral::validate_rewired_gelu(&mut h, &dispatcher, &cpu);
    s59_gelu_spectral::validate_rewired_hmm_forward(&mut h, &dispatcher, &cpu);
    s59_gelu_spectral::validate_rewired_esd(&mut h);
    s59_gelu_spectral::validate_rewired_mp_bounds(&mut h);
    s59_gelu_spectral::validate_rewired_effective_rank(&mut h);

    println!("\n--- S72 Rewires: Upstream Tensor APIs + FST ---\n");
    s72_tensor_fst::validate_rewired_softmax_row_wise(&mut h, &dispatcher, &cpu);
    s72_tensor_fst::validate_rewired_fst_single_locus(&mut h, &dispatcher);
    s72_tensor_fst::validate_rewired_pairwise_fst_full(&mut h, &dispatcher, &cpu);
    s72_tensor_fst::validate_rewired_viterbi_argmax(&mut h, &dispatcher, &cpu);

    println!("\n--- S78 Rewires: Stats Absorption + Cross-Spring Primitives ---\n");
    s78_primitives::validate_rewired_mae_s78(&mut h);
    s78_primitives::validate_rewired_shannon_from_frequencies_s78(&mut h);
    s78_primitives::validate_rewired_hill_s78(&mut h);
    s78_primitives::validate_rewired_l2_distance_s78(&mut h);
    s78_primitives::validate_rewired_complexity_metric_s78(&mut h);

    println!("\n--- Driver Profile Validation ---\n");
    device_caps::validate_device_capabilities(&mut h, &dispatcher);

    throughput_benchmark::benchmark_throughput(&dispatcher, &cpu);
    throughput_benchmark::benchmark_s72_throughput(&dispatcher, &cpu);
    lineage::report_cross_spring_lineage();

    h.finish();
}
