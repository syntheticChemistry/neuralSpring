// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(clippy::pedantic, clippy::expect_used, reason = "validation binary")]

//! GPU-batched ODE integration validator (Phase B gap closure).
//!
//! Validates `integrate_ode_batch` against CPU reference for the Hill ODE.
//! Uses encoder batching: N systems × T timesteps in one dispatch, final state only.
//!
//! For deterministic validation we use the generic Hill ODE (rk4_parallel.wgsl).
//! The signal_integration vpsT ODE uses stochastic noise; with noise_scale=0
//! the deterministic part could be validated via a dedicated vpsT shader (future).
//!
//! ## Papers validated
//!
//! - Paper 020: Regulatory Network (Mhatre et al., 2020)
//! - Paper 021: Signal Integration (Srivastava et al., 2011)

#![expect(clippy::cast_precision_loss, reason = "validation binary")]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::cpu_fallback;
use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::signal_integration::{integrate_ode, OdeParams, OdeState};
use neural_spring::tolerances;
use neural_spring::validation::{exit_no_gpu, ValidationHarness};

#[tokio::main]
async fn main() {
    let dispatcher = match Gpu::new().await {
        Ok(gpu) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                gpu.adapter_name, gpu.device_type, gpu.backend
            );
            Dispatcher::from_gpu(gpu)
        }
        Err(_) => exit_no_gpu(),
    };

    let mut h = ValidationHarness::new("gpu_ode_batch");

    validate_batch_vs_cpu(&mut h, &dispatcher);
    validate_signal_integration_deterministic(&mut h);
    validate_batch_multi_system(&mut h, &dispatcher);

    h.finish();
}

/// Validate GPU batch ODE vs CPU reference (Hill ODE).
fn validate_batch_vs_cpu(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    let dim = 4_usize;
    let n_systems = 8_usize;
    let n_steps = 100_usize;
    let dt = 0.01_f64;

    // Hill ODE coeffs: [prod, deg, act_idx] per dimension
    let coeffs_template: Vec<f64> = vec![
        1.0, 0.5, 0.0, // y0: prod=1, deg=0.5, activator=y0
        0.5, 0.3, 0.0, // y1: prod=0.5, deg=0.3, activator=y0
        0.3, 0.2, 1.0, // y2: prod=0.3, deg=0.2, activator=y1
        0.2, 0.1, 2.0, // y3: prod=0.2, deg=0.1, activator=y2
    ];
    let initial: Vec<f64> = vec![1.0, 0.5, 0.0, 0.0];

    let mut states: Vec<f64> = Vec::new();
    let mut coeffs: Vec<f64> = Vec::new();
    for _ in 0..n_systems {
        states.extend_from_slice(&initial);
        coeffs.extend_from_slice(&coeffs_template);
    }

    let cpu_result =
        cpu_fallback::cpu_ode_batch_hill(&states, &coeffs, n_systems, dim, n_steps, dt);
    let gpu_result = dispatcher.integrate_ode_batch(&states, &coeffs, n_systems, dim, n_steps, dt);

    let mut max_diff = 0.0_f64;
    for sys in 0..n_systems {
        for d in 0..dim {
            let g = gpu_result[sys * dim + d];
            let c = cpu_result[sys * dim + d];
            max_diff = max_diff.max((g - c).abs());
        }
    }

    h.check_upper(
        &format!("batch ODE: max diff {max_diff:.2e} across {n_systems} systems"),
        max_diff,
        tolerances::GPU_RK4_F32,
    );

    let all_finite = gpu_result.iter().all(|&v| v.is_finite());
    h.check_bool("batch ODE: all outputs finite", all_finite);
}

/// Validate signal_integration::integrate_ode with zero noise (deterministic).
///
/// This tests the CPU reference path. Full GPU validation of vpsT ODE
/// would require a dedicated shader (two_input_hill RHS).
fn validate_signal_integration_deterministic(h: &mut ValidationHarness) {
    let y0 = OdeState {
        cdg: 0.1,
        ai: 0.1,
        vps_t: 0.0,
        biofilm: 0.0,
    };
    let params = OdeParams {
        noise_scale: 0.0,
        ..OdeParams::default()
    };

    let trace = integrate_ode(1.0, 0.01, &y0, &params);
    let final_state = trace.last().copied().expect("ODE trace must be non-empty");

    h.check_bool(
        "signal_integration (noise=0): trace non-empty",
        !trace.is_empty(),
    );
    h.check_bool(
        "signal_integration (noise=0): final state finite",
        final_state.cdg.is_finite()
            && final_state.ai.is_finite()
            && final_state.vps_t.is_finite()
            && final_state.biofilm.is_finite(),
    );
    h.check_bool(
        "signal_integration (noise=0): final state non-negative",
        final_state.cdg >= 0.0
            && final_state.ai >= 0.0
            && final_state.vps_t >= 0.0
            && final_state.biofilm >= 0.0,
    );
}

/// Validate multi-system batch produces correct output shape.
fn validate_batch_multi_system(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    let dim = 2_usize;
    let n_systems = 4_usize;
    let n_steps = 50_usize;
    let dt = 0.01_f64;

    let coeffs_template: Vec<f64> = vec![0.5, 0.1, 1.0, 0.3, 0.2, 0.0];
    let mut states: Vec<f64> = Vec::new();
    let mut coeffs: Vec<f64> = Vec::new();
    for i in 0..n_systems {
        let fi = i as f64;
        states.push(0.25_f64.mul_add(fi, 0.5));
        states.push(0.1_f64.mul_add(fi, 0.3));
        coeffs.extend_from_slice(&coeffs_template);
    }

    let result = dispatcher.integrate_ode_batch(&states, &coeffs, n_systems, dim, n_steps, dt);

    h.check_bool(
        &format!("multi-system: output count {}", result.len()),
        result.len() == n_systems * dim,
    );

    let all_finite = result.iter().all(|&v| v.is_finite());
    h.check_bool("multi-system: all finite", all_finite);
}
