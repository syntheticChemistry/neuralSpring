// SPDX-License-Identifier: AGPL-3.0-or-later

//! `ToadStool` S87 sync validation — deep debt evolution, CPU module ungating,
//! error type evolution, `gpu_helpers` refactor.
//!
//! Validates that `neuralSpring` works correctly with `ToadStool` S87
//! (`2dc26792`), which includes:
//!
//! - CPU modules properly ungated from `#[cfg(feature = "gpu")]`
//! - `gpu_helpers` split into `buffers`, `bind_group_layouts`, `pipelines`
//! - `BarracudaError::is_device_lost()` + `gpu_ctx()` additions
//! - `async-trait` reclassified from TODO→NOTE (architectural choice)
//! - FHE shader fixes (NTT/INTT/`pointwise_mul` `u64_mod_simple`)
//! - Unsafe audit: 60+ sites documented with `// SAFETY:`
//!
//! ```text
//! cargo run --release --bin validate_toadstool_s87_sync
//! ```

#![expect(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "validation binary"
)]

use barracuda::error::BarracudaError;
use barracuda::nautilus::{
    DriftMonitor, GenerationRecord, InstanceId, NautilusBrain, NautilusBrainConfig,
};
use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::tolerances;
use neural_spring::validation::{exit_no_gpu, ValidationHarness};

fn validate_cpu_modules_accessible(h: &mut ValidationHarness) {
    eprintln!("\n── CPU module accessibility (S87 ungating fix) ──");

    let v = barracuda::stats::correlation::variance(&[1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("variance should succeed");
    h.check_abs(
        "stats::correlation::variance accessible (ungated)",
        v,
        2.5,
        tolerances::EXACT_F64,
    );

    let pc = barracuda::stats::pearson_correlation(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0])
        .expect("pearson should succeed");
    h.check_abs(
        "stats::pearson_correlation accessible (ungated)",
        pc,
        1.0,
        tolerances::EXACT_F64,
    );

    let counts = [10.0, 20.0, 30.0, 40.0];
    let shannon = barracuda::stats::shannon(&counts);
    h.check_bool(
        "stats::shannon accessible (ungated)",
        shannon.is_finite() && shannon > 0.0,
    );

    let simpson = barracuda::stats::simpson(&counts);
    h.check_bool(
        "stats::simpson accessible (ungated)",
        simpson.is_finite() && simpson > 0.0,
    );
}

fn validate_error_type_evolution(h: &mut ValidationHarness) {
    eprintln!("\n── BarracudaError S87 evolution ──");

    let err = BarracudaError::Gpu("test device error".into());
    h.check_bool(
        "BarracudaError::Gpu constructs",
        matches!(&err, BarracudaError::Gpu(_)),
    );

    h.check_bool(
        "BarracudaError::is_device_lost() exists and returns bool",
        !err.is_device_lost(),
    );

    let io_err = BarracudaError::io(
        "test context",
        std::io::Error::new(std::io::ErrorKind::NotFound, "test"),
    );
    h.check_bool(
        "BarracudaError::io() constructs Io variant",
        matches!(&io_err, BarracudaError::Io { .. }),
    );
}

fn validate_nautilus_s87_compat(h: &mut ValidationHarness) {
    eprintln!("\n── Nautilus S87 compatibility ──");

    let config = NautilusBrainConfig::default();
    let mut brain = NautilusBrain::new(config, "s87-compat-test");
    h.check_bool("NautilusBrain creates on S87", true);

    let obs = barracuda::nautilus::BetaObservation {
        beta: 5.5,
        plaquette: 0.58,
        cg_iters: 120.0,
        acceptance: 0.75,
        delta_h_abs: 0.01,
        quenched_plaq: None,
        quenched_plaq_var: None,
        anderson_r: None,
        anderson_lambda_min: None,
    };
    brain.observe(obs);
    h.check_bool(
        "NautilusBrain::observe works on S87",
        brain.observations.len() == 1,
    );

    let mut drift = DriftMonitor::default();
    let gen = GenerationRecord {
        generation: 0,
        mean_fitness: 0.5,
        best_fitness: 0.8,
        pop_size: 100,
        origin: InstanceId("s87-test".to_string()),
        training_size: 10,
    };
    drift.record(&gen, 100);
    h.check_bool("DriftMonitor records on S87", drift.ne_s_history.len() == 1);
}

fn validate_dispatcher_s87(h: &mut ValidationHarness, disp: &Dispatcher) {
    eprintln!("\n── Dispatcher core ops on S87 ──");

    let a: Vec<f64> = (0..64_i32).map(|i| f64::from(i) * 0.01).collect();
    let b: Vec<f64> = (0..64_i32)
        .map(|i| f64::from(i).mul_add(0.02, 0.5))
        .collect();

    let result = disp.mat_mul(&a, &b, 8);
    h.check_bool("Dispatcher::mat_mul works on S87", result.len() == 64);

    let transposed = disp.transpose(&a, 8);
    h.check_bool("Dispatcher::transpose works on S87", transposed.len() == 64);

    let sm = disp.softmax(&a[..8]);
    let sm_sum: f64 = sm.iter().sum();
    h.check_abs(
        "Dispatcher::softmax sums to 1.0 on S87",
        sm_sum,
        1.0,
        tolerances::TENSOR_EXACT_F32,
    );

    let v = disp.variance(&a);
    h.check_bool("Dispatcher::variance is finite on S87", v.is_finite());

    let m = disp.mean(&a);
    h.check_bool("Dispatcher::mean is finite on S87", m.is_finite());

    let e = disp.shannon_entropy(&[0.25, 0.25, 0.25, 0.25]);
    let expected_h = (4.0_f64).ln();
    h.check_abs(
        "Dispatcher::shannon_entropy on S87",
        e,
        expected_h,
        tolerances::GPU_ENTROPY_F64,
    );
}

fn validate_dispatch_bridge_s87(h: &mut ValidationHarness, disp: &Dispatcher) {
    eprintln!("\n── barracuda::dispatch bridge on S87 ──");

    let n = 8_usize;
    let data: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.1).collect();

    let disp_result = disp.frobenius_norm(&data);
    let bridge_result = barracuda::dispatch::frobenius_norm_dispatch(&data, disp.wgpu_device())
        .expect("frobenius_norm_dispatch");
    h.check_abs(
        "frobenius_norm: Dispatcher == barracuda::dispatch on S87",
        (disp_result - bridge_result).abs(),
        0.0,
        tolerances::EXACT_F64,
    );

    let disp_mean = disp.mean(&data);
    let bridge_mean =
        barracuda::dispatch::mean_dispatch(&data, disp.wgpu_device()).expect("mean_dispatch");
    h.check_abs(
        "mean: Dispatcher == barracuda::dispatch on S87",
        (disp_mean - bridge_mean).abs(),
        0.0,
        tolerances::EXACT_F64,
    );
}

#[tokio::main]
async fn main() {
    eprintln!("=== ToadStool S87 Sync Validation ===\n");
    eprintln!("Pin: 2fee1969 → 2dc26792 (S86→S87, 2 commits)");
    eprintln!("Key: deep debt evolution, CPU ungating, gpu_helpers refactor");
    eprintln!("FHE: NTT/INTT u64_mod_simple fix, pointwise_mul correction");
    eprintln!("Unsafe: 60+ sites documented with // SAFETY:");

    let mut h = ValidationHarness::new("toadstool_s87_sync");

    validate_cpu_modules_accessible(&mut h);
    validate_error_type_evolution(&mut h);
    validate_nautilus_s87_compat(&mut h);

    let Ok(gpu) = Gpu::new().await else {
        exit_no_gpu();
    };
    let disp = Dispatcher::from_gpu(gpu);

    validate_dispatcher_s87(&mut h, &disp);
    validate_dispatch_bridge_s87(&mut h, &disp);

    h.finish();
}
