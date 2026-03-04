// SPDX-License-Identifier: AGPL-3.0-or-later

//! `barraCuda` v0.3.1 rewire validation — Session 121.
//!
//! Validates two major rewires from local implementations to upstream
//! `barraCuda` primitives:
//!
//! 1. **WDM surrogates → `SimpleMlp`**: `EosSurrogate` and `TransportSurrogate`
//!    now delegate MLP inference to `barracuda::nn::SimpleMlp` instead of
//!    hand-rolled matmul loops. JSON weight loading converts flat row-major
//!    weights to `DenseLayer` 2D format. ~400 LOC eliminated.
//!
//! 2. **HMM Viterbi chain → f64 `ComputeDispatch`**: `hmm_viterbi_chain_gpu`
//!    replaced per-step f32 `Tensor` matmul loop with single-dispatch f64
//!    `hmm_viterbi_f64.wgsl` shader from upstream.
//!
//! ## Cross-spring evolution provenance
//!
//! | Component | Origin | Absorption Path |
//! |-----------|--------|-----------------|
//! | `SimpleMlp` | neuralSpring nW-01/02 → `ToadStool` S83 | MLP concept → standalone API |
//! | `hmm_viterbi_f64.wgsl` | wetSpring bio → `ToadStool` S69 | per-step Tensor → f64 shader |
//! | `HmmBatchForwardF64` | wetSpring bio → `ToadStool` S52 | CPU Hmm → GPU batch forward |
//!
//! ```text
//! cargo run --release --bin validate_barracuda_s121_rewire
//! ```

#![expect(clippy::expect_used, reason = "validation binary")]

use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

const EOS_JSON: &str = include_str!("../../control/wdm/eos_surrogate_baseline.json");
const TRANSPORT_JSON: &str = include_str!("../../control/wdm/transport_surrogate_baseline.json");

fn validate_simplemlp_eos(h: &mut ValidationHarness) {
    eprintln!("\n── WDM EOS → SimpleMlp rewire ──");

    for element in ["H", "He", "C"] {
        let surr = neural_spring::wdm_surrogate::load_surrogate_from_json(EOS_JSON, element)
            .expect("JSON load should succeed");

        h.check_bool(
            &format!("{element}: SimpleMlp layers non-empty"),
            !surr.mlp.layers.is_empty(),
        );
        h.check_bool(
            &format!("{element}: SimpleMlp input_size == 2"),
            surr.mlp.input_size() == Some(2),
        );
        h.check_bool(
            &format!("{element}: SimpleMlp output_size == 2"),
            surr.mlp.output_size() == Some(2),
        );

        let (p, e) = surr.predict(1.0, 100_000.0);
        h.check_bool(&format!("{element}: P finite"), p.is_finite());
        h.check_bool(&format!("{element}: E finite"), e.is_finite());

        let (p2, e2) = surr.predict(1.0, 100_000.0);
        h.check_abs(
            &format!("{element}: P determinism"),
            p,
            p2,
            tolerances::GPU_F64_EXACT,
        );
        h.check_abs(
            &format!("{element}: E determinism"),
            e,
            e2,
            tolerances::GPU_F64_EXACT,
        );

        let json = surr.mlp.to_json().expect("SimpleMlp serialization");
        let roundtrip =
            barracuda::nn::SimpleMlp::from_json(&json).expect("SimpleMlp deserialization");
        let rt_out = roundtrip.forward(&[0.5, -0.3]);
        let orig_out = surr.mlp.forward(&[0.5, -0.3]);
        h.check_abs(
            &format!("{element}: JSON roundtrip fidelity [0]"),
            rt_out[0],
            orig_out[0],
            tolerances::GPU_F64_EXACT,
        );
        h.check_abs(
            &format!("{element}: JSON roundtrip fidelity [1]"),
            rt_out[1],
            orig_out[1],
            tolerances::GPU_F64_EXACT,
        );
    }
}

fn validate_simplemlp_transport(h: &mut ValidationHarness) {
    eprintln!("\n── WDM Transport → SimpleMlp rewire ──");

    let surr = neural_spring::wdm_transport::load_transport_from_json(TRANSPORT_JSON)
        .expect("transport JSON load");

    h.check_bool(
        "transport: SimpleMlp input_size == 3",
        surr.mlp.input_size() == Some(3),
    );
    h.check_bool(
        "transport: SimpleMlp output_size == 3",
        surr.mlp.output_size() == Some(3),
    );

    let test_points: &[(f64, f64, f64)] = &[
        (-1.0, 4.0, 1.0),
        (0.0, 5.0, 3.0),
        (0.5, 6.0, 6.0),
        (1.0, 7.0, 10.0),
    ];

    for &(lr, lt, z) in test_points {
        let (d, eta, lam) = surr.predict(lr, lt, z);
        h.check_bool(&format!("D*({lr},{lt},{z}) finite"), d.is_finite());
        h.check_bool(&format!("η*({lr},{lt},{z}) finite"), eta.is_finite());
        h.check_bool(&format!("λ*({lr},{lt},{z}) finite"), lam.is_finite());
        h.check_bool(&format!("D*({lr},{lt},{z}) > 0"), d > 0.0);
        h.check_bool(&format!("η*({lr},{lt},{z}) > 0"), eta > 0.0);
        h.check_bool(&format!("λ*({lr},{lt},{z}) > 0"), lam > 0.0);
    }

    let (d1, e1, l1) = surr.predict(0.5, 6.0, 6.0);
    let (d2, e2, l2) = surr.predict(0.5, 6.0, 6.0);
    h.check_abs("D* determinism", d1, d2, tolerances::GPU_F64_EXACT);
    h.check_abs("η* determinism", e1, e2, tolerances::GPU_F64_EXACT);
    h.check_abs("λ* determinism", l1, l2, tolerances::GPU_F64_EXACT);

    let json = surr
        .mlp
        .to_json()
        .expect("transport SimpleMlp serialization");
    let roundtrip =
        barracuda::nn::SimpleMlp::from_json(&json).expect("transport SimpleMlp deserialization");
    let rt_out = roundtrip.forward(&[0.1, 0.2, 0.3]);
    let orig_out = surr.mlp.forward(&[0.1, 0.2, 0.3]);
    for i in 0..3 {
        h.check_abs(
            &format!("transport JSON roundtrip [{i}]"),
            rt_out[i],
            orig_out[i],
            tolerances::GPU_F64_EXACT,
        );
    }
}

fn validate_hmm_viterbi_f64(h: &mut ValidationHarness) {
    eprintln!("\n── HMM Viterbi chain → f64 ComputeDispatch ──");

    let transition = vec![0.7, 0.3, 0.4, 0.6];
    let emission = vec![0.1, 0.4, 0.5, 0.6, 0.3, 0.1];
    let initial = vec![0.6, 0.4];
    let observations = vec![0, 1, 2, 0, 1, 2, 0, 1];
    let n_states = 2;
    let n_obs = 3;

    let hmm = neural_spring::hmm::Hmm::from_flat(
        transition.clone(),
        emission.clone(),
        initial.clone(),
        n_states,
        n_obs,
    );
    let (cpu_path, cpu_log_prob) = hmm.viterbi(&observations);

    h.check_bool(
        "CPU Viterbi path length",
        cpu_path.len() == observations.len(),
    );
    h.check_bool("CPU Viterbi log_prob finite", cpu_log_prob.is_finite());
    h.check_bool("CPU Viterbi log_prob negative", cpu_log_prob < 0.0);

    for &s in &cpu_path {
        h.check_bool(&format!("CPU Viterbi state {s} < n_states"), s < n_states);
    }

    let (cpu2_path, cpu2_prob) = hmm.viterbi(&observations);
    h.check_bool("CPU Viterbi deterministic (path)", cpu_path == cpu2_path);
    h.check_abs(
        "CPU Viterbi deterministic (prob)",
        cpu_log_prob,
        cpu2_prob,
        tolerances::GPU_F64_EXACT,
    );

    let dispatcher = neural_spring::gpu_dispatch::Dispatcher::cpu_only();
    let (disp_path, disp_prob): (Vec<usize>, f64) = dispatcher.hmm_viterbi_chain(
        &initial,
        &transition,
        &emission,
        &observations,
        n_states,
        n_obs,
    );

    h.check_bool(
        "Dispatcher Viterbi path length",
        disp_path.len() == observations.len(),
    );
    h.check_bool("Dispatcher Viterbi log_prob finite", disp_prob.is_finite());

    h.check_bool("Dispatcher vs CPU path agreement", disp_path == cpu_path);

    let log_prob_diff = (disp_prob - cpu_log_prob).abs();
    h.check_bool(
        &format!("Dispatcher vs CPU log_prob within tolerance (diff={log_prob_diff:.6e})"),
        log_prob_diff < tolerances::GPU_HMM_VITERBI_LOGPROB_F64,
    );
}

fn validate_hmm_forward_chain(h: &mut ValidationHarness) {
    eprintln!("\n── HMM forward chain (dispatcher) ──");

    let transition = vec![0.7, 0.3, 0.4, 0.6];
    let emission = vec![0.1, 0.4, 0.5, 0.6, 0.3, 0.1];
    let initial = vec![0.6, 0.4];
    let observations = vec![0, 1, 2, 0];

    let hmm = neural_spring::hmm::Hmm::from_flat(
        transition.clone(),
        emission.clone(),
        initial.clone(),
        2,
        3,
    );
    let (_, cpu_ll) = hmm.forward(&observations);

    let dispatcher = neural_spring::gpu_dispatch::Dispatcher::cpu_only();
    let disp_ll =
        dispatcher.hmm_forward_chain(&initial, &transition, &emission, &observations, 2, 3);

    h.check_bool("CPU forward ll finite", cpu_ll.is_finite());
    h.check_bool("Dispatcher forward ll finite", disp_ll.is_finite());
    h.check_bool("CPU forward ll negative", cpu_ll < 0.0);

    let ll_diff = (disp_ll - cpu_ll).abs();
    h.check_bool(
        &format!("forward ll within tolerance (diff={ll_diff:.6e})"),
        ll_diff < tolerances::GPU_HMM_LOG_LIKELIHOOD_F32,
    );
}

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  barraCuda v0.3.1 S121 Rewire Validation                   ║");
    eprintln!("║  SimpleMlp + HMM f64 ComputeDispatch                       ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  WDM surrogates: local MLP → barracuda::nn::SimpleMlp");
    eprintln!("  HMM Viterbi:    per-step f32 Tensor → f64 hmm_viterbi_f64.wgsl");
    eprintln!();

    let mut h = ValidationHarness::new("barracuda_s121_rewire");

    validate_simplemlp_eos(&mut h);
    validate_simplemlp_transport(&mut h);
    validate_hmm_viterbi_f64(&mut h);
    validate_hmm_forward_chain(&mut h);

    h.finish();
}
