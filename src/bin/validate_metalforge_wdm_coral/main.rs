// SPDX-License-Identifier: AGPL-3.0-or-later

//! metalForge mixed-hardware validation for WDM + coralForge domain workloads.
//!
//! Proves that WDM surrogate inference and coralForge Evoformer operations
//! route correctly through the metalForge mixed-hardware substrate model,
//! with NUCLEUS atomic coordination (tower discovery, node compute, nest
//! provenance) and PCIe bypass cost modelling for GPU→NPU→CPU paths.
//!
//! ## Structure
//!
//! Tower discovery and node WDM validators live here. coralForge dispatch,
//! nest provenance, mixed routing, and PCIe bypass live in `coral_mixed.rs`.
//!
//! ## NUCLEUS atomics exercised
//!
//! - **Tower**: substrate discovery, capability enumeration
//! - **Node**: GPU compute dispatch for WDM MLP + coralForge attention
//! - **Nest**: result provenance transfer, entropy tracking
//!
//! ## Provenance
//!
//! Validation class: GPU cross-dispatch.
//! CPU reference: neuralSpring lib (Rust CPU).
//! GPU path: metalForge mixed-hardware (WDM + coralForge via `DispatchConfig` routing).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::suspicious_operation_groupings,
    reason = "validation binary"
)]

mod coral_mixed;

use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, exit_no_gpu};
use neural_spring_forge::inventory;
use neural_spring_forge::mixed::MixedSubstrate;
use neural_spring_forge::substrate::SubstrateKind;

use coral_mixed::rect_matmul_cpu;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("metalforge_wdm_coral");

    let Ok(gpu) = Gpu::new().await else {
        exit_no_gpu();
    };
    let disp = Dispatcher::from_gpu(gpu);

    let mut rng = Rng::new(42);

    // Tower: substrate discovery for WDM/coralForge workloads
    validate_tower_wdm_coral(&mut h);

    // Node: WDM surrogate compute dispatch
    validate_node_wdm_transport(&mut h, &disp, &mut rng);
    validate_node_wdm_eos(&mut h, &disp, &mut rng);
    validate_node_wdm_sqw_lstm(&mut h, &disp, &mut rng);
    validate_node_wdm_transfer_mlp(&mut h, &disp, &mut rng);
    validate_node_wdm_esn_spectral(&mut h, &disp, &mut rng);

    // Node: coralForge Evoformer compute dispatch
    coral_mixed::validate_node_coral_attention(&mut h, &disp, &mut rng);
    coral_mixed::validate_node_coral_trimul(&mut h, &disp, &mut rng);
    coral_mixed::validate_node_coral_confidence(&mut h, &disp, &mut rng);

    // Nest: provenance and entropy tracking
    coral_mixed::validate_nest_wdm_provenance(&mut h, &disp, &mut rng);

    // Mixed routing: heterogeneous domain pipelines
    coral_mixed::validate_mixed_wdm_routing(&mut h, &disp);
    coral_mixed::validate_mixed_coral_routing(&mut h, &disp);
    coral_mixed::validate_mixed_heterogeneous_pipeline(&mut h, &disp, &mut rng);

    // PCIe bypass: GPU→NPU direct transfer for folding workloads
    coral_mixed::validate_pcie_folding_bypass(&mut h);

    h.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Tower: substrate discovery for WDM/coralForge domain workloads
// ═══════════════════════════════════════════════════════════════════

fn validate_tower_wdm_coral(h: &mut ValidationHarness) {
    let substrates = inventory::discover();

    let has_gpu = substrates.iter().any(|s| s.kind == SubstrateKind::Gpu);
    let has_cpu = substrates.iter().any(|s| s.kind == SubstrateKind::Cpu);

    h.check_bool("Tower: GPU discovered for WDM/coral", has_gpu);
    h.check_bool("Tower: CPU discovered for fallback", has_cpu);
    h.check_bool("Tower: multi-substrate available", substrates.len() >= 2);

    for s in &substrates {
        h.check_bool(
            &format!("Tower: '{}' capabilities non-empty", s.identity.name),
            !s.capabilities.is_empty() || s.kind == SubstrateKind::Cpu,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// Node: WDM transport MLP inference via mixed dispatch
// ═══════════════════════════════════════════════════════════════════

fn validate_node_wdm_transport(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let batch = 8;
    let d_in = 2;
    let d_h = 16;

    let input: Vec<f64> = (0..batch * d_in).map(|_| rng.normal()).collect();
    let w: Vec<f64> = (0..d_in * d_h).map(|_| rng.normal() * 0.3).collect();

    let wl = MixedWorkload {
        op: "wdm_transport_mlp_layer1",
        compute_us: 200_000.0,
        data_bytes: ((batch * d_in + d_in * d_h) * 8) as u64,
        npu_available: false,
        needs_realtime: false,
    };

    let (result, substrate) = disp.mixed_dispatch(
        &wl,
        |dev| {
            barracuda::dispatch::matmul_dispatch(&input, &w, batch, d_in, d_h, Some(dev))
                .map_err(|e| format!("{e}"))
        },
        || rect_matmul_cpu(&input, &w, batch, d_in, d_h),
    );

    h.check_bool(
        "Node WDM transport: result correct shape",
        result.len() == batch * d_h,
    );
    h.check_bool(
        "Node WDM transport: all finite",
        result.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "Node WDM transport: routed to GPU (large compute)",
        substrate == MixedSubstrate::GpuOnly,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Node: WDM EOS via mixed dispatch
// ═══════════════════════════════════════════════════════════════════

fn validate_node_wdm_eos(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let n = 4;
    let input: Vec<f64> = (0..n).map(|_| rng.normal()).collect();

    let wl_small = MixedWorkload {
        op: "wdm_eos_single_point",
        compute_us: 50.0,
        data_bytes: (n * 8) as u64,
        npu_available: false,
        needs_realtime: false,
    };

    let cpu_result = barracuda::stats::mean(&input);
    let (mixed_result, substrate) = disp.mixed_dispatch(
        &wl_small,
        |dev| {
            let f32_data: Vec<f32> = input.iter().map(|&v| v as f32).collect();
            let t = barracuda::tensor::Tensor::from_data(&f32_data, vec![n], dev.clone())
                .map_err(|e| format!("{e}"))?;
            let mean = t.mean().map_err(|e| format!("{e}"))?;
            let v = mean.to_vec().map_err(|e| format!("{e}"))?;
            Ok(f64::from(v[0]))
        },
        || cpu_result,
    );

    h.check_abs(
        "Node WDM EOS: small → correct result",
        mixed_result,
        cpu_result,
        0.01,
    );
    h.check_bool(
        "Node WDM EOS: small → CPU (overhead dominates)",
        substrate == MixedSubstrate::CpuOnly,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Node: WDM ESN spectral analysis via eigensolve dispatch
// ═══════════════════════════════════════════════════════════════════

fn validate_node_wdm_esn_spectral(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let n = 12;
    let raw: Vec<f64> = (0..n * n).map(|_| rng.normal() * 0.3).collect();
    let sym: Vec<f64> = {
        let mut s = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let v = 0.5 * (raw[i * n + j] + raw[j * n + i]);
                s[i * n + j] = v;
                s[j * n + i] = v;
            }
        }
        s
    };

    let (evals, _) = disp.eigh(&sym, n);

    h.check_bool(
        "Node ESN spectral: correct eigenvalue count",
        evals.len() == n,
    );
    h.check_bool(
        "Node ESN spectral: all finite",
        evals.iter().all(|v| v.is_finite()),
    );

    let spectral_radius = evals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    h.check_bool("Node ESN spectral: radius > 0", spectral_radius > 0.0);

    let eval_var = disp.variance(&evals);
    h.check_bool("Node ESN spectral: eigenvalue variance > 0", eval_var > 0.0);
}

// ═══════════════════════════════════════════════════════════════════
// Node: WDM SQW LSTM gate computation via mixed dispatch (nW-03)
// ═══════════════════════════════════════════════════════════════════

fn validate_node_wdm_sqw_lstm(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let hs = 4;
    let input_size = 1;

    let w_i: Vec<f64> = (0..4 * hs * input_size)
        .map(|_| rng.normal() * 0.2)
        .collect();
    let w_h: Vec<f64> = (0..4 * hs * hs).map(|_| rng.normal() * 0.1).collect();
    let b: Vec<f64> = (0..4 * hs).map(|_| rng.normal() * 0.05).collect();

    let x_input = vec![0.7_f64; input_size];
    let h_prev = vec![0.0_f64; hs];

    let wl = MixedWorkload {
        op: "wdm_sqw_lstm_gate",
        compute_us: 200_000.0,
        data_bytes: ((4 * hs * input_size + 4 * hs * hs + hs) * 8) as u64,
        npu_available: false,
        needs_realtime: false,
    };

    let cpu_input_proj = rect_matmul_cpu(&w_i, &x_input, 4 * hs, input_size, 1);
    let cpu_hidden_proj = rect_matmul_cpu(&w_h, &h_prev, 4 * hs, hs, 1);
    let cpu_gates: Vec<f64> = cpu_input_proj
        .iter()
        .zip(cpu_hidden_proj.iter())
        .zip(b.iter())
        .map(|((i, h), b)| i + h + b)
        .collect();

    let (mixed_input_proj, substrate) = disp.mixed_dispatch(
        &wl,
        |dev| {
            barracuda::dispatch::matmul_dispatch(&w_i, &x_input, 4 * hs, input_size, 1, Some(dev))
                .map_err(|e| format!("{e}"))
        },
        || cpu_input_proj.clone(),
    );

    h.check_bool(
        "Node WDM SQW: LSTM gate shape",
        mixed_input_proj.len() == 4 * hs,
    );
    h.check_bool(
        "Node WDM SQW: GPU routing (LSTM compute)",
        substrate == MixedSubstrate::GpuOnly,
    );

    let max_diff = cpu_gates
        .iter()
        .zip(
            mixed_input_proj
                .iter()
                .zip(cpu_hidden_proj.iter())
                .zip(b.iter())
                .map(|((i, h), b)| i + h + b),
        )
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        "Node WDM SQW: gate pre-activations parity",
        max_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Node: WDM Transfer classical→WDM MLP via mixed dispatch (nW-04)
// ═══════════════════════════════════════════════════════════════════

fn validate_node_wdm_transfer_mlp(h: &mut ValidationHarness, disp: &Dispatcher, rng: &mut Rng) {
    let classical_dim = 3;
    let hid = 6;
    let wdm_dim = 2;

    let w1: Vec<f64> = (0..hid * classical_dim)
        .map(|_| rng.normal() * 0.3)
        .collect();
    let b1: Vec<f64> = (0..hid).map(|_| rng.normal() * 0.1).collect();
    let w2: Vec<f64> = (0..wdm_dim * hid).map(|_| rng.normal() * 0.3).collect();
    let b2: Vec<f64> = (0..wdm_dim).map(|_| rng.normal() * 0.1).collect();
    let x: Vec<f64> = (0..classical_dim).map(|_| rng.normal()).collect();

    let wl = MixedWorkload {
        op: "wdm_transfer_mlp_forward",
        compute_us: 200_000.0,
        data_bytes: ((hid * classical_dim + wdm_dim * hid) * 8) as u64,
        npu_available: false,
        needs_realtime: false,
    };

    let cpu_hidden = rect_matmul_cpu(&w1, &x, hid, classical_dim, 1);
    let cpu_act: Vec<f64> = cpu_hidden
        .iter()
        .zip(b1.iter())
        .map(|(h, b)| (h + b).max(0.0))
        .collect();
    let cpu_out: Vec<f64> = {
        let o = rect_matmul_cpu(&w2, &cpu_act, wdm_dim, hid, 1);
        o.iter().zip(b2.iter()).map(|(o, b)| o + b).collect()
    };

    let (mixed_hidden, substrate) = disp.mixed_dispatch(
        &wl,
        |dev| {
            barracuda::dispatch::matmul_dispatch(&w1, &x, hid, classical_dim, 1, Some(dev))
                .map_err(|e| format!("{e}"))
        },
        || cpu_hidden.clone(),
    );

    let mixed_act: Vec<f64> = mixed_hidden
        .iter()
        .zip(b1.iter())
        .map(|(h, b)| (h + b).max(0.0))
        .collect();
    let mixed_out: Vec<f64> = {
        let o = rect_matmul_cpu(&w2, &mixed_act, wdm_dim, hid, 1);
        o.iter().zip(b2.iter()).map(|(o, b)| o + b).collect()
    };

    let out_diff = cpu_out
        .iter()
        .zip(mixed_out.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        "Node WDM Transfer: MLP output parity",
        out_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
    h.check_bool(
        "Node WDM Transfer: GPU routing (large compute)",
        substrate == MixedSubstrate::GpuOnly,
    );
    h.check_bool(
        "Node WDM Transfer: all outputs finite",
        mixed_out.iter().all(|v| v.is_finite()),
    );
}
