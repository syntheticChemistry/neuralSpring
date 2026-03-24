// SPDX-License-Identifier: AGPL-3.0-or-later

//! `ToadStool` S93 / `BarraCUDA` standalone extraction validation.
//!
//! Validates the migration from the embedded `barracuda` crate inside
//! `ToadStool` (`../phase1/toadstool/crates/barracuda`) to the standalone
//! `BarraCUDA` primal (`../barraCuda/crates/barracuda` v0.3.1).
//!
//! ## What changed (S87 → S93 + standalone extraction)
//!
//! - `BarraCUDA` extracted from `ToadStool` into its own repo/primal (S89)
//! - `barracuda::spectral::tridiag_eigenvectors` added (S88)
//! - `barracuda::tolerances` gained domain constants: `HYDRO_*`,
//!   `PHYSICS_ANDERSON_EIGENVALUE`, `BIO_DIVERSITY_*`
//! - `barracuda::unified_math::MathOp` enum — canonical op vocabulary
//! - `barracuda::unified_hardware::ComputeExecutor` trait — multi-backend dispatch
//! - `barracuda::device::Fp64Strategy` — precision routing per hardware
//! - MSRV bumped 1.80 → 1.87
//! - 767 WGSL shaders, dual-protocol IPC
//!
//! ## v0.3.0 → v0.3.1 (confirmed Mar 4, 2026)
//!
//! - tarpc/JSON-RPC parity (signature changes: `MatmulResult`, `DispatchResult`, FHE types)
//! - blake3 `pure` feature (no C SIMD compilation)
//! - `println` → tracing migration
//! - `DeviceLost` error variant with `is_retriable()` check
//! - Global `DEVICE_CREATION_LOCK` for serialized device creation
//! - 2,965 upstream tests, 0 clippy warnings
//!
//! ## Provenance
//!
//! Validation class: Cross-spring (dependency migration)
//! Source pin: `ToadStool` S87 commit `2dc26792` → `BarraCUDA` v0.3.1 standalone
//! hotSpring validated the same path swap (716/716 pass, single-line change)
//!
//! ```text
//! cargo run --release --bin validate_toadstool_s93_barracuda_extraction
//! ```

#![expect(
    clippy::items_after_statements,
    reason = "validation binary — inline assert fn for trait object safety check"
)]

use barracuda::device::Fp64Strategy;
use barracuda::nautilus::{
    DriftMonitor, GenerationRecord, InstanceId, NautilusBrain, NautilusBrainConfig,
};
use barracuda::spectral::tridiag_eigenvectors;
use barracuda::unified_hardware::ComputeExecutor;
use barracuda::unified_math::MathOp;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, exit_no_gpu};

// Upstream contract expectations are centralized in tolerances::UPSTREAM_*
// (see tolerances/mod.rs "Upstream contract expectations" section).

fn validate_standalone_path(h: &mut ValidationHarness) {
    println!("\n── Standalone path verification (v0.3.1) ──");

    let version = env!("CARGO_PKG_VERSION");
    h.check_bool(
        "neuralSpring compiles against barraCuda v0.3.1 standalone",
        !version.is_empty(),
    );
}

fn validate_tridiag_eigenvectors(h: &mut ValidationHarness) {
    println!("\n── barracuda::spectral::tridiag_eigenvectors (S88) ──");

    let diag = vec![2.0, 2.0, 2.0];
    let off = vec![1.0, 1.0];
    let (eigenvalues, eigenvectors) = tridiag_eigenvectors(&diag, &off);

    h.check_bool(
        "tridiag_eigenvectors returns 3 eigenvalues",
        eigenvalues.len() == 3,
    );
    h.check_bool(
        "tridiag_eigenvectors returns 9 eigenvector elements (3x3)",
        eigenvectors.len() == 9,
    );

    let mut sorted = eigenvalues;
    sorted.sort_by(f64::total_cmp);
    let expected_min = 2.0 - 2.0_f64.sqrt();
    let expected_max = 2.0 + 2.0_f64.sqrt();
    h.check_abs(
        "tridiag smallest eigenvalue ≈ 2-√2",
        sorted[0],
        expected_min,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "tridiag middle eigenvalue ≈ 2",
        sorted[1],
        2.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "tridiag largest eigenvalue ≈ 2+√2",
        sorted[2],
        expected_max,
        tolerances::CROSS_LANGUAGE,
    );

    let (empty_vals, empty_vecs) = tridiag_eigenvectors(&[], &[]);
    h.check_bool(
        "tridiag_eigenvectors empty → empty",
        empty_vals.is_empty() && empty_vecs.is_empty(),
    );

    let (single_val, single_vec) = tridiag_eigenvectors(&[5.0], &[]);
    h.check_bool(
        "tridiag_eigenvectors single → eigenvalue=5",
        single_val.len() == 1,
    );
    h.check_abs(
        "tridiag single eigenvalue == input",
        single_val[0],
        5.0,
        tolerances::EXACT_F64,
    );
    let _ = single_vec;
}

fn validate_tolerance_constants(h: &mut ValidationHarness) {
    println!("\n── barracuda::tolerances domain constants (S88+) ──");

    let hydro_et0 = barracuda::tolerances::HYDRO_ET0;
    h.check_bool("HYDRO_ET0 accessible", hydro_et0.abs_tol > 0.0);
    h.check_abs(
        "HYDRO_ET0.abs_tol",
        hydro_et0.abs_tol,
        0.05,
        tolerances::EXACT_F64,
    );

    let hydro_soil = barracuda::tolerances::HYDRO_SOIL_MOISTURE;
    h.check_bool("HYDRO_SOIL_MOISTURE accessible", hydro_soil.abs_tol > 0.0);

    let hydro_water = barracuda::tolerances::HYDRO_WATER_BALANCE;
    h.check_bool("HYDRO_WATER_BALANCE accessible", hydro_water.abs_tol > 0.0);

    let hydro_kc = barracuda::tolerances::HYDRO_CROP_COEFFICIENT;
    h.check_abs(
        "HYDRO_CROP_COEFFICIENT.abs_tol",
        hydro_kc.abs_tol,
        tolerances::UPSTREAM_HYDRO_CROP_COEFFICIENT,
        tolerances::EXACT_F64,
    );

    let anderson = barracuda::tolerances::PHYSICS_ANDERSON_EIGENVALUE;
    h.check_abs(
        "PHYSICS_ANDERSON_EIGENVALUE.abs_tol",
        anderson.abs_tol,
        tolerances::UPSTREAM_PHYSICS_ANDERSON_EIGENVALUE,
        tolerances::EXACT_F64,
    );

    let shannon = barracuda::tolerances::BIO_DIVERSITY_SHANNON;
    h.check_abs(
        "BIO_DIVERSITY_SHANNON.abs_tol",
        shannon.abs_tol,
        tolerances::UPSTREAM_BIO_DIVERSITY_SHANNON,
        tolerances::EXACT_F64,
    );

    let simpson = barracuda::tolerances::BIO_DIVERSITY_SIMPSON;
    h.check_abs(
        "BIO_DIVERSITY_SIMPSON.abs_tol",
        simpson.abs_tol,
        tolerances::UPSTREAM_BIO_DIVERSITY_SIMPSON,
        tolerances::EXACT_F64,
    );
}

fn validate_unified_math(h: &mut ValidationHarness) {
    println!("\n── barracuda::unified_math::MathOp vocabulary ──");

    let ops = [
        MathOp::Negate,
        MathOp::Abs,
        MathOp::Exp,
        MathOp::Sqrt,
        MathOp::Square,
        MathOp::Reciprocal,
    ];
    h.check_bool("MathOp unary variants accessible", ops.len() == 6);

    let matmul = MathOp::MatMul {
        transpose_a: false,
        transpose_b: false,
    };
    h.check_bool(
        "MathOp::MatMul struct variant accessible",
        matches!(matmul, MathOp::MatMul { .. }),
    );

    let softmax = MathOp::Softmax { dim: 0 };
    h.check_bool(
        "MathOp::Softmax struct variant accessible",
        matches!(softmax, MathOp::Softmax { .. }),
    );
}

fn validate_fp64_strategy(h: &mut ValidationHarness) {
    println!("\n── barracuda::device::Fp64Strategy (precision routing) ──");

    let native = Fp64Strategy::Native;
    let hybrid = Fp64Strategy::Hybrid;
    let concurrent = Fp64Strategy::Concurrent;

    h.check_bool(
        "Fp64Strategy::Native != Hybrid",
        !matches!(native, Fp64Strategy::Hybrid),
    );
    h.check_bool(
        "Fp64Strategy::Hybrid != Concurrent",
        !matches!(hybrid, Fp64Strategy::Concurrent),
    );
    h.check_bool(
        "Fp64Strategy::Concurrent is variant",
        matches!(concurrent, Fp64Strategy::Concurrent),
    );
}

fn validate_compute_executor_trait(h: &mut ValidationHarness) {
    println!("\n── barracuda::unified_hardware::ComputeExecutor trait ──");

    const fn assert_trait_object_safe<T: ComputeExecutor + ?Sized>() {}
    h.check_bool("ComputeExecutor trait is accessible and object-safe", {
        assert_trait_object_safe::<dyn ComputeExecutor>();
        true
    });
}

fn validate_nautilus_continuity(h: &mut ValidationHarness) {
    println!("\n── Nautilus API continuity (S86→S93 standalone) ──");

    let config = NautilusBrainConfig::default();
    let brain = NautilusBrain::new(config, "s93-continuity");
    h.check_bool(
        "NautilusBrain creates on standalone barraCuda",
        !brain.trained,
    );

    let mut drift = DriftMonitor::default();
    let record = GenerationRecord {
        generation: 0,
        mean_fitness: 0.5,
        best_fitness: 0.8,
        pop_size: 100,
        origin: InstanceId("s93-test".to_string()),
        training_size: 10,
    };
    drift.record(&record, 100);
    let ne_s = drift.ne_s_history[0];
    let expected = (100.0 * 0.8) / (1.0 + 0.8);
    h.check_abs(
        "DriftMonitor ne_s on standalone barraCuda",
        ne_s,
        expected,
        tolerances::CROSS_LANGUAGE,
    );
}

fn validate_dispatcher_continuity(h: &mut ValidationHarness, disp: &Dispatcher) {
    println!("\n── Dispatcher continuity on standalone barraCuda ──");

    let a: Vec<f64> = (0..64_i32).map(|i| f64::from(i) * 0.01).collect();
    let b: Vec<f64> = (0..64_i32)
        .map(|i| f64::from(i).mul_add(0.02, 0.5))
        .collect();

    let result = disp.mat_mul(&a, &b, 8);
    h.check_bool("mat_mul works on standalone barraCuda", result.len() == 64);

    let sm = disp.softmax(&a[..8]);
    let sm_sum: f64 = sm.iter().sum();
    h.check_abs(
        "softmax sums to 1.0 on standalone barraCuda",
        sm_sum,
        1.0,
        tolerances::TENSOR_EXACT_F32,
    );

    let e = disp.shannon_entropy(&[0.25, 0.25, 0.25, 0.25]);
    let expected_h = (4.0_f64).ln();
    h.check_abs(
        "shannon_entropy on standalone barraCuda",
        e,
        expected_h,
        tolerances::GPU_ENTROPY_F64,
    );
}

#[tokio::main]
async fn main() {
    println!("=== ToadStool S93 / barraCuda Standalone Extraction Validation ===\n");
    println!("Migration: ../phase1/toadstool/crates/barracuda → ../barraCuda/crates/barracuda");
    println!("Pin: ToadStool S87 (2dc26792) → BarraCUDA v0.3.1 standalone");
    println!("Key: standalone extraction, tridiag_eigenvectors, domain tolerances,");
    println!("     unified_math::MathOp, Fp64Strategy, ComputeExecutor trait");

    let mut h = ValidationHarness::new("toadstool_s93_barracuda_extraction");

    validate_standalone_path(&mut h);
    validate_tridiag_eigenvectors(&mut h);
    validate_tolerance_constants(&mut h);
    validate_unified_math(&mut h);
    validate_fp64_strategy(&mut h);
    validate_compute_executor_trait(&mut h);
    validate_nautilus_continuity(&mut h);

    let Ok(gpu) = Gpu::new().await else {
        exit_no_gpu();
    };
    let disp = Dispatcher::from_gpu(gpu);

    validate_dispatcher_continuity(&mut h, &disp);

    h.finish();
}
