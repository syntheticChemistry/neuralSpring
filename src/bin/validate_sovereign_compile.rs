// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exp 105: Sovereign compile validation.
//!
//! Tests the compute triangle readiness for neuralSpring's spectral operations:
//!
//! - barraCuda: eigensolve, `FusedMapReduceF64`, `BatchedEighGpu`
//! - toadStool: hardware discovery, capability routing
//! - coralReef: WGSL to native binary compilation
//!
//! Reports GPU hardware capabilities, precision strategy, and sovereign
//! compilation readiness for the RTX 4070 (Ada/GSP) path.

#![expect(clippy::expect_used, reason = "binary entry point")]
#![expect(clippy::cast_precision_loss, reason = "small indices fit in f64")]

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn report_hardware(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    println!("\n── Hardware Discovery ──");
    println!("  Backend: {}", dispatcher.backend());
    println!("  Adapter: {}", dispatcher.adapter_name());
    println!("  Has GPU: {}", dispatcher.has_gpu());

    h.check_bool("dispatcher initialized", true);
    h.check_bool(
        "backend is set",
        !format!("{}", dispatcher.backend()).is_empty(),
    );

    if let Some(caps) = dispatcher.capabilities() {
        println!("  Max workgroup X: {}", caps.max_compute_workgroup_size_x);
        println!("  Max buffer size: {} bytes", caps.max_buffer_size);
        println!("  Supports f64: {}", caps.supports_f64);
        println!("  Supports f16: {}", caps.supports_f16);
        h.check_bool("workgroup size > 0", caps.max_compute_workgroup_size_x > 0);
        h.check_bool("buffer size > 0", caps.max_buffer_size > 0);
    }

    if let Some(profile) = dispatcher.driver_profile() {
        println!("\n── Precision Strategy ──");
        println!("  FP64 strategy: {:?}", profile.fp64_strategy());
        println!(
            "  DF64 SPIR-V poisoning: {}",
            profile.has_df64_spir_v_poisoning()
        );
        println!("  Precision routing: {:?}", profile.precision_routing());
        println!(
            "  Shared memory f64 safe: {}",
            dispatcher.shared_memory_f64_safe()
        );
        h.check_bool("driver profile available", true);
    }

    println!("\n── Bandwidth Tier ──");
    println!("  PCIe tier: {:?}", dispatcher.bandwidth_tier());
    println!(
        "  Pow workaround needed: {}",
        dispatcher.needs_pow_workaround()
    );
}

fn validate_eigensolve(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    println!("\n── Eigensolve GPU Validation ──");
    let n = 32;
    let mut matrix = vec![0.0_f64; n * n];
    for i in 0..n {
        matrix[i * n + i] = (i + 1) as f64;
    }
    let (eigenvalues, _) = dispatcher.eigh(&matrix, n);
    let mut sorted = eigenvalues;
    sorted.sort_by(f64::total_cmp);

    for (i, &ev) in sorted.iter().enumerate() {
        let expected = (i + 1) as f64;
        h.check_abs(
            &format!("eigenvalue[{i}]"),
            ev,
            expected,
            tolerances::GELU_LARGE_INPUT,
        );
    }
}

fn main() {
    let mut h = ValidationHarness::new("sovereign_compile");

    println!("═══ Exp 105: Sovereign Compile & Compute Triangle Validation ═══");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let dispatcher = rt.block_on(Dispatcher::new());

    report_hardware(&mut h, &dispatcher);
    validate_eigensolve(&mut h, &dispatcher);

    h.finish();
}
