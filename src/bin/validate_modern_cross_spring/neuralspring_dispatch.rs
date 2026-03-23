// SPDX-License-Identifier: AGPL-3.0-or-later

// neuralSpring provenance: Dispatcher softmax/GELU/variance/HMM + graph Laplacian.

use neural_spring::gpu_dispatch::Dispatcher;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, bench_once};

pub fn validate_neuralspring_dispatch(h: &mut ValidationHarness, dispatcher: &Dispatcher) {
    println!("\n─── neuralSpring provenance: ML + dispatch ───\n");

    // Dispatcher: softmax (nS S58 → BarraCUDA domain_ops)
    let logits = [1.0, 2.0, 3.0, 4.0];
    let (sm, _) = bench_once("softmax (nS→BarraCUDA)", || dispatcher.softmax(&logits));
    let sm_sum: f64 = sm.iter().sum();
    h.check_abs(
        "nS→dispatch: softmax sums to 1",
        sm_sum,
        1.0,
        tolerances::CROSS_LANGUAGE,
    );

    // Dispatcher: GELU (nS S59 → BarraCUDA domain_ops)
    let vals = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let (gelu_out, _) = bench_once("gelu (nS→BarraCUDA)", || dispatcher.gelu(&vals));
    h.check_abs(
        "nS→dispatch: GELU(0) = 0",
        gelu_out[2],
        0.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool("nS→dispatch: GELU monotonic", gelu_out[3] < gelu_out[4]);

    // Dispatcher: variance (nS → BarraCUDA domain_ops)
    let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let (var, _) = bench_once("variance (nS→BarraCUDA)", || dispatcher.variance(&data));
    let cpu = Dispatcher::cpu_only();
    let var_cpu = cpu.variance(&data);
    h.check_abs(
        "nS→dispatch: variance GPU≈CPU",
        var,
        var_cpu,
        tolerances::CROSS_LANGUAGE,
    );

    // Dispatcher: HMM forward (nS+wS → BarraCUDA)
    let alpha = [0.5, 0.5];
    let trans = [0.7, 0.3, 0.4, 0.6];
    let emis = [0.9, 0.2];
    let (hmm, _) = bench_once("hmm_forward (nS+wS→BarraCUDA)", || {
        dispatcher.hmm_forward_step(&alpha, &trans, &emis, 2)
    });
    h.check_bool(
        "nS+wS→dispatch: HMM alpha finite",
        hmm.0.iter().all(|v| v.is_finite()),
    );
    h.check_bool("nS+wS→dispatch: HMM scale > 0", hmm.1 > 0.0);

    // Graph operations (nS baseCamp → BarraCUDA)
    let adjacency = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let (laplacian, _) = bench_once("graph_laplacian (nS→BarraCUDA)", || {
        barracuda::linalg::graph_laplacian(&adjacency, 3)
    });
    h.check_abs(
        "nS→graph: L[0,0] = degree(0) = 2",
        laplacian[0],
        2.0,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "nS→graph: L[0,1] = -adj[0,1] = -1",
        laplacian[1],
        -1.0,
        tolerances::CROSS_LANGUAGE,
    );

    // Effective rank (nS baseCamp → BarraCUDA)
    let eigs = [10.0, 5.0, 1.0, 0.1, 0.01];
    let (eff_rank, _) = bench_once("effective_rank (nS→BarraCUDA)", || {
        barracuda::linalg::effective_rank(&eigs)
    });
    h.check_bool("nS→graph: effective_rank > 1", eff_rank > 1.0);
    h.check_bool("nS→graph: effective_rank ≤ 5", eff_rank <= 5.0);
}
