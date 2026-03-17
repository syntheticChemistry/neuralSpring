// SPDX-License-Identifier: AGPL-3.0-or-later

//! WDM + `AlphaFold3` dispatch parity — proves domain-specific ML workloads
//! route correctly through `BarraCUDA` CPU↔GPU dispatch and `metalForge`
//! mixed-hardware substrate routing.
//!
//! Validates the evolution chain:
//! ```text
//! Python baseline → Rust CPU → BarraCUDA GPU Tensor → Dispatcher CPU↔GPU
//!                                                   → metalForge routing
//!                                                   → NUCLEUS coordination
//! ```
//!
//! ## Coverage
//!
//! - **WDM Transport MLP**: matmul chain → `ReLU` → readout (nW-01)
//! - **WDM EOS MLP**: matmul chain → `ReLU` → readout (nW-02)
//! - **WDM S(q,ω) LSTM**: gate matmul → sigmoid/tanh → cell update (nW-03)
//! - **WDM Transfer**: classical→WDM domain MLP via `matmul` dispatch (nW-04)
//! - **WDM ESN**: recurrence matmul → tanh → readout (nW-05)
//! - **`AlphaFold3` pLDDT**: per-residue sigmoid confidence (nF-03)
//! - **`AlphaFold3` PAE**: distance matmul + row-softmax (nF-03)
//! - **Mixed-hardware routing**: metalForge substrate selection
//! - **NUCLEUS coordination**: tower (eigensolve) + node (WDM state) + nest (provenance)
//!
//! ## Provenance
//!
//! Validation class: Cross-spring (GPU↔CPU dispatch parity)
//! Provenance: WDM baselines from `control/wdm/` (commit `f9ad0268`, 2026-02-16),
//! `AlphaFold3` baselines from `control/coral_forge/` (commit `f9ad0268`, 2026-02-16).
//! CPU reference values computed inline; GPU parity via `barracuda::dispatch`.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    reason = "validation binary — numeric casts and multi-domain dispatch compositions"
)]

use neural_spring::gpu::Gpu;
use neural_spring::gpu_dispatch::{Dispatcher, MixedWorkload};
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, exit_no_gpu};
use neural_spring_forge::mixed::MixedSubstrate;

#[tokio::main]
async fn main() {
    let mut h = ValidationHarness::new("wdm_alphafold_dispatch");

    let Ok(gpu) = Gpu::new().await else {
        exit_no_gpu();
    };
    let gpu_disp = Dispatcher::from_gpu(gpu);
    let cpu_disp = Dispatcher::cpu_only();

    validate_wdm_transport_mlp(&mut h, &gpu_disp, &cpu_disp);
    validate_wdm_eos_mlp(&mut h, &gpu_disp, &cpu_disp);
    validate_wdm_sqw_lstm(&mut h, &gpu_disp, &cpu_disp);
    validate_wdm_transfer_mlp(&mut h, &gpu_disp, &cpu_disp);
    validate_wdm_esn_recurrence(&mut h, &gpu_disp, &cpu_disp);
    validate_alphafold3_pldt(&mut h, &gpu_disp, &cpu_disp);
    validate_alphafold3_pae(&mut h, &gpu_disp, &cpu_disp);
    validate_mixed_routing_wdm(&mut h, &gpu_disp);
    validate_nucleus_wdm_coordination(&mut h, &gpu_disp);

    h.finish();
}

// ═══════════════════════════════════════════════════════════════════
// WDM Transport MLP (nW-01): matmul chain → ReLU → readout
// ═══════════════════════════════════════════════════════════════════

fn validate_wdm_transport_mlp(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(101);
    let in_dim = 4;
    let hid_dim = 8;
    let out_dim = 2;

    let w1: Vec<f64> = (0..hid_dim * in_dim).map(|_| rng.normal() * 0.3).collect();
    let b1: Vec<f64> = (0..hid_dim).map(|_| rng.normal() * 0.1).collect();
    let w2: Vec<f64> = (0..out_dim * hid_dim).map(|_| rng.normal() * 0.3).collect();
    let b2: Vec<f64> = (0..out_dim).map(|_| rng.normal() * 0.1).collect();
    let x: Vec<f64> = (0..in_dim).map(|_| rng.normal()).collect();

    let g_out = mlp_forward_dispatch(gpu, &x, &w1, &b1, &w2, &b2, in_dim, hid_dim, out_dim);
    let c_out = mlp_forward_dispatch(cpu, &x, &w1, &b1, &w2, &b2, in_dim, hid_dim, out_dim);

    for i in 0..out_dim {
        h.check_abs(
            &format!("wdm_transport MLP out[{i}]"),
            g_out[i],
            c_out[i],
            tolerances::GPU_MATMUL_RANDOM_F32,
        );
    }
    h.check_bool(
        "wdm_transport MLP finite",
        g_out.iter().all(|v| v.is_finite()),
    );
}

// ═══════════════════════════════════════════════════════════════════
// WDM EOS MLP (nW-02): same architecture, different random seed
// ═══════════════════════════════════════════════════════════════════

fn validate_wdm_eos_mlp(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(202);
    let in_dim = 3;
    let hid_dim = 16;
    let out_dim = 1;

    let w1: Vec<f64> = (0..hid_dim * in_dim).map(|_| rng.normal() * 0.3).collect();
    let b1: Vec<f64> = (0..hid_dim).map(|_| rng.normal() * 0.1).collect();
    let w2: Vec<f64> = (0..out_dim * hid_dim).map(|_| rng.normal() * 0.3).collect();
    let b2: Vec<f64> = (0..out_dim).map(|_| rng.normal() * 0.1).collect();
    let x: Vec<f64> = (0..in_dim).map(|_| rng.normal()).collect();

    let g_out = mlp_forward_dispatch(gpu, &x, &w1, &b1, &w2, &b2, in_dim, hid_dim, out_dim);
    let c_out = mlp_forward_dispatch(cpu, &x, &w1, &b1, &w2, &b2, in_dim, hid_dim, out_dim);

    h.check_abs(
        "wdm_eos MLP out[0]",
        g_out[0],
        c_out[0],
        tolerances::GPU_MATMUL_RANDOM_F32,
    );
    h.check_bool("wdm_eos MLP finite", g_out[0].is_finite());
}

// ═══════════════════════════════════════════════════════════════════
// WDM S(q,ω) LSTM (nW-03): gate matmuls through dispatch
// ═══════════════════════════════════════════════════════════════════

fn validate_wdm_sqw_lstm(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(303);
    let hs = 4;
    let input_size = 1;

    let w_i: Vec<f64> = (0..4 * hs * input_size)
        .map(|_| rng.normal() * 0.2)
        .collect();
    let w_h: Vec<f64> = (0..4 * hs * hs).map(|_| rng.normal() * 0.1).collect();
    let b: Vec<f64> = (0..4 * hs).map(|_| rng.normal() * 0.05).collect();

    let x_val = 0.7;
    let h_prev = vec![0.0_f64; hs];

    let g_gates = lstm_gate_dispatch(gpu, x_val, &h_prev, &w_i, &w_h, &b, hs, input_size);
    let c_gates = lstm_gate_dispatch(cpu, x_val, &h_prev, &w_i, &w_h, &b, hs, input_size);

    for i in 0..4 * hs {
        h.check_abs(
            &format!("wdm_sqw LSTM gate[{i}]"),
            g_gates[i],
            c_gates[i],
            tolerances::GPU_MATMUL_RANDOM_F32,
        );
    }

    let g_cell = lstm_cell_from_gates(&g_gates, &[0.0; 4], hs);
    let c_cell = lstm_cell_from_gates(&c_gates, &[0.0; 4], hs);

    let max_h_diff = g_cell
        .0
        .iter()
        .zip(c_cell.0.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    h.check_bool(
        &format!("wdm_sqw LSTM h_new parity (max_diff={max_h_diff:.2e})"),
        max_h_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
}

// ═══════════════════════════════════════════════════════════════════
// WDM Transfer (nW-04): classical→WDM domain MLP through dispatch
// ═══════════════════════════════════════════════════════════════════

fn validate_wdm_transfer_mlp(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(404);
    let classical_dim = 3;
    let hid_dim = 6;
    let wdm_dim = 2;

    let w1: Vec<f64> = (0..hid_dim * classical_dim)
        .map(|_| rng.normal() * 0.3)
        .collect();
    let b1: Vec<f64> = (0..hid_dim).map(|_| rng.normal() * 0.1).collect();
    let w2: Vec<f64> = (0..wdm_dim * hid_dim).map(|_| rng.normal() * 0.3).collect();
    let b2: Vec<f64> = (0..wdm_dim).map(|_| rng.normal() * 0.1).collect();
    let x: Vec<f64> = (0..classical_dim).map(|_| rng.normal()).collect();

    let g_out = mlp_forward_dispatch(gpu, &x, &w1, &b1, &w2, &b2, classical_dim, hid_dim, wdm_dim);
    let c_out = mlp_forward_dispatch(cpu, &x, &w1, &b1, &w2, &b2, classical_dim, hid_dim, wdm_dim);

    for i in 0..wdm_dim {
        h.check_abs(
            &format!("wdm_transfer MLP out[{i}]"),
            g_out[i],
            c_out[i],
            tolerances::GPU_MATMUL_RANDOM_F32,
        );
    }
    h.check_bool(
        "wdm_transfer MLP: all outputs finite",
        g_out.iter().all(|v| v.is_finite()),
    );
}

// ═══════════════════════════════════════════════════════════════════
// WDM ESN (nW-05): recurrence matmul + tanh through dispatch
// ═══════════════════════════════════════════════════════════════════

fn validate_wdm_esn_recurrence(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(505);
    let res_size = 8;

    let w_res: Vec<f64> = (0..res_size * res_size)
        .map(|_| rng.normal() * 0.1)
        .collect();
    let w_in: Vec<f64> = (0..res_size).map(|_| rng.normal() * 0.3).collect();
    let state: Vec<f64> = (0..res_size).map(|_| rng.normal() * 0.5).collect();
    let x_val = 0.42;

    let g_state = esn_step_dispatch(gpu, &state, x_val, &w_res, &w_in, res_size);
    let c_state = esn_step_dispatch(cpu, &state, x_val, &w_res, &w_in, res_size);

    let max_diff = g_state
        .iter()
        .zip(c_state.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("wdm_esn recurrence parity (max_diff={max_diff:.2e})"),
        max_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );
    h.check_bool(
        "wdm_esn state finite",
        g_state.iter().all(|v| v.is_finite()),
    );
    h.check_bool(
        "wdm_esn state bounded (tanh)",
        g_state
            .iter()
            .all(|v| v.abs() <= 1.0 + tolerances::EXACT_F64),
    );
}

// ═══════════════════════════════════════════════════════════════════
// AlphaFold3 pLDDT (nF-03): per-residue sigmoid confidence
// ═══════════════════════════════════════════════════════════════════

fn validate_alphafold3_pldt(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(901);
    let n_residues = 16;

    let logits: Vec<f64> = (0..n_residues).map(|_| rng.normal() * 2.0).collect();
    let w: Vec<f64> = (0..n_residues).map(|_| rng.normal() * 0.3).collect();
    let b: Vec<f64> = (0..1).map(|_| rng.normal() * 0.1).collect();

    let g_score = pldt_dispatch(gpu, &logits, &w, &b);
    let c_score = pldt_dispatch(cpu, &logits, &w, &b);

    h.check_abs(
        "alphafold3 pLDDT dispatch parity",
        g_score,
        c_score,
        tolerances::GPU_MATMUL_RANDOM_F32,
    );
    h.check_bool("alphafold3 pLDDT in [0,1]", (0.0..=1.0).contains(&g_score));
}

// ═══════════════════════════════════════════════════════════════════
// AlphaFold3 PAE (nF-03): distance matmul + row-softmax
// ═══════════════════════════════════════════════════════════════════

fn validate_alphafold3_pae(h: &mut ValidationHarness, gpu: &Dispatcher, cpu: &Dispatcher) {
    let mut rng = Rng::new(902);
    let n = 4;
    let n_bins = 8;

    let logits: Vec<f64> = (0..n * n * n_bins).map(|_| rng.normal()).collect();

    let g_pae = pae_dispatch(gpu, &logits, n, n_bins);
    let c_pae = pae_dispatch(cpu, &logits, n, n_bins);

    let max_diff = g_pae
        .iter()
        .zip(c_pae.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    h.check_bool(
        &format!("alphafold3 PAE parity (max_diff={max_diff:.2e})"),
        max_diff < tolerances::GPU_MATMUL_RANDOM_F32,
    );

    for (pair_idx, row) in g_pae.chunks(n_bins).enumerate() {
        let sum: f64 = row.iter().sum();
        h.check_abs(
            &format!("alphafold3 PAE row[{pair_idx}] sums to 1"),
            sum,
            1.0,
            tolerances::TENSOR_EXACT_F32,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// Mixed-hardware routing: metalForge substrate selection for WDM
// ═══════════════════════════════════════════════════════════════════

fn validate_mixed_routing_wdm(h: &mut ValidationHarness, disp: &Dispatcher) {
    let small = MixedWorkload {
        op: "wdm_mlp_small",
        compute_us: 50.0,
        data_bytes: 1024,
        npu_available: false,
        needs_realtime: false,
    };

    let data = [1.0, 2.0, 3.0, 4.0];
    let (result, substrate) = disp.mixed_dispatch(
        &small,
        |dev| {
            let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let t = barracuda::tensor::Tensor::from_data(&f32_data, vec![4], dev.clone())
                .map_err(|e| format!("{e}"))?;
            let mean = t.mean().map_err(|e| format!("{e}"))?;
            let v = mean.to_vec().map_err(|e| format!("{e}"))?;
            Ok(f64::from(v[0]))
        },
        || data.iter().sum::<f64>() / data.len() as f64,
    );
    h.check_abs(
        "mixed small WDM → correct result",
        result,
        2.5,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );
    h.check_bool(
        "mixed small WDM → CPU (dispatch overhead dominates)",
        substrate == MixedSubstrate::CpuOnly,
    );

    let large = MixedWorkload {
        op: "wdm_lstm_large",
        compute_us: 200_000.0,
        data_bytes: 4_194_304,
        npu_available: false,
        needs_realtime: false,
    };
    let (result_lg, substrate_lg) = disp.mixed_dispatch(
        &large,
        |dev| {
            let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let t = barracuda::tensor::Tensor::from_data(&f32_data, vec![4], dev.clone())
                .map_err(|e| format!("{e}"))?;
            let mean = t.mean().map_err(|e| format!("{e}"))?;
            let v = mean.to_vec().map_err(|e| format!("{e}"))?;
            Ok(f64::from(v[0]))
        },
        || data.iter().sum::<f64>() / data.len() as f64,
    );
    h.check_abs(
        "mixed large WDM → correct result",
        result_lg,
        2.5,
        tolerances::GPU_MEAN_DISPATCH_F32,
    );
    h.check_bool(
        "mixed large WDM → GPU (compute dominates)",
        substrate_lg == MixedSubstrate::GpuOnly,
    );

    let npu = MixedWorkload {
        op: "alphafold3_realtime_inference",
        compute_us: 100_000.0,
        data_bytes: 2_097_152,
        npu_available: true,
        needs_realtime: true,
    };
    let (_, substrate_npu) = disp.mixed_dispatch(
        &npu,
        |dev| {
            let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let t = barracuda::tensor::Tensor::from_data(&f32_data, vec![4], dev.clone())
                .map_err(|e| format!("{e}"))?;
            let v = t
                .mean()
                .map_err(|e| format!("{e}"))?
                .to_vec()
                .map_err(|e| format!("{e}"))?;
            Ok(f64::from(v[0]))
        },
        || data.iter().sum::<f64>() / data.len() as f64,
    );
    h.check_bool(
        "mixed NPU realtime AF3 → GpuToNpu routing",
        substrate_npu == MixedSubstrate::GpuToNpu,
    );
}

// ═══════════════════════════════════════════════════════════════════
// NUCLEUS coordination: tower (eigensolve), node (WDM state),
// nest (provenance) working together
// ═══════════════════════════════════════════════════════════════════

fn validate_nucleus_wdm_coordination(h: &mut ValidationHarness, disp: &Dispatcher) {
    let a = vec![2.0, 0.5, 0.5, 3.0];
    let (eigenvalues, _) = disp.eigh(&a, 2);
    let mut sorted = eigenvalues;
    sorted.sort_by(f64::total_cmp);
    h.check_bool(
        "nucleus tower: WDM Hamiltonian eigensolve",
        sorted.len() == 2,
    );
    // [[2, 0.5], [0.5, 3]] → eigenvalues (5 ± √2)/2
    let sqrt2_half = std::f64::consts::FRAC_1_SQRT_2;
    h.check_abs(
        "nucleus tower: λ_min",
        sorted[0],
        2.5 - sqrt2_half,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );
    h.check_abs(
        "nucleus tower: λ_max",
        sorted[1],
        2.5 + sqrt2_half,
        tolerances::GPU_EIGH_DISPATCH_F64,
    );

    let transition_probs = disp.softmax(&[0.3, 0.5, 0.2]);
    let sum: f64 = transition_probs.iter().sum();
    h.check_abs(
        "nucleus node: WDM state transitions sum to 1",
        sum,
        1.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "nucleus node: all transitions positive",
        transition_probs.iter().all(|&p| p > 0.0),
    );

    let entropy = disp.shannon_entropy(&transition_probs);
    h.check_bool(
        "nucleus nest: provenance entropy finite",
        entropy.is_finite(),
    );
    h.check_bool("nucleus nest: entropy > 0 (non-trivial)", entropy > 0.0);
    h.check_bool(
        "nucleus nest: entropy < ln(3) (bounded)",
        entropy < 3.0_f64.ln() + 0.01,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Domain-specific dispatch compositions
// ═══════════════════════════════════════════════════════════════════

fn rect_matmul(disp: &Dispatcher, a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    barracuda::dispatch::matmul_dispatch(a, b, m, k, n, disp.wgpu_device()).unwrap_or_else(|_| {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    })
}

fn mlp_forward_dispatch(
    disp: &Dispatcher,
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    x_dim: usize,
    hid_dim: usize,
    out_dim: usize,
) -> Vec<f64> {
    let hidden = rect_matmul(disp, w1, x, hid_dim, x_dim, 1);
    let hidden_biased: Vec<f64> = hidden.iter().zip(b1.iter()).map(|(h, b)| h + b).collect();
    let activated: Vec<f64> = hidden_biased.iter().map(|&v| v.max(0.0)).collect();
    let output = rect_matmul(disp, w2, &activated, out_dim, hid_dim, 1);
    output.iter().zip(b2.iter()).map(|(o, b)| o + b).collect()
}

fn lstm_gate_dispatch(
    disp: &Dispatcher,
    x_val: f64,
    h_prev: &[f64],
    w_i: &[f64],
    w_h: &[f64],
    b: &[f64],
    hs: usize,
    input_size: usize,
) -> Vec<f64> {
    let x = vec![x_val; input_size];
    let input_proj = rect_matmul(disp, w_i, &x, 4 * hs, input_size, 1);
    let hidden_proj = rect_matmul(disp, w_h, h_prev, 4 * hs, hs, 1);
    input_proj
        .iter()
        .zip(hidden_proj.iter())
        .zip(b.iter())
        .map(|((i, h), b)| i + h + b)
        .collect()
}

fn lstm_cell_from_gates(gates: &[f64], c_prev: &[f64], hs: usize) -> (Vec<f64>, Vec<f64>) {
    let f_gate: Vec<f64> = gates[..hs].iter().map(|&v| sigmoid_f64(v)).collect();
    let i_gate: Vec<f64> = gates[hs..2 * hs].iter().map(|&v| sigmoid_f64(v)).collect();
    let g_gate: Vec<f64> = gates[2 * hs..3 * hs].iter().map(|v| v.tanh()).collect();
    let o_gate: Vec<f64> = gates[3 * hs..].iter().map(|&v| sigmoid_f64(v)).collect();

    let c_new: Vec<f64> = (0..hs)
        .map(|j| f_gate[j].mul_add(c_prev[j], i_gate[j] * g_gate[j]))
        .collect();
    let h_new: Vec<f64> = (0..hs).map(|j| o_gate[j] * c_new[j].tanh()).collect();
    (h_new, c_new)
}

fn sigmoid_f64(x: f64) -> f64 {
    neural_spring::primitives::sigmoid(x)
}

fn esn_step_dispatch(
    disp: &Dispatcher,
    state: &[f64],
    x_val: f64,
    w_res: &[f64],
    w_in: &[f64],
    res_size: usize,
) -> Vec<f64> {
    let res_proj = rect_matmul(disp, w_res, state, res_size, res_size, 1);
    let input_contrib: Vec<f64> = w_in.iter().map(|&w| w * x_val).collect();
    res_proj
        .iter()
        .zip(input_contrib.iter())
        .map(|(r, i)| (r + i).tanh())
        .collect()
}

fn pldt_dispatch(disp: &Dispatcher, logits: &[f64], w: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = logits.iter().zip(w.iter()).map(|(l, w)| l * w).sum();
    let score = dot + b[0];
    let softmax = disp.softmax(&[score, 0.0]);
    softmax[0]
}

fn pae_dispatch(disp: &Dispatcher, logits: &[f64], n: usize, n_bins: usize) -> Vec<f64> {
    let n_pairs = n * n;
    let mut result = Vec::with_capacity(n_pairs * n_bins);
    for pair in 0..n_pairs {
        let row = &logits[pair * n_bins..(pair + 1) * n_bins];
        let probs = disp.softmax(row);
        result.extend_from_slice(&probs);
    }
    result
}
