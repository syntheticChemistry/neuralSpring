// SPDX-License-Identifier: AGPL-3.0-or-later
//
// HMM Forward Pass — Log-Domain GEMM via WGSL
//
// Computes the forward algorithm for Hidden Markov Models entirely on GPU
// using log-domain arithmetic to avoid underflow. This is the GPU equivalent
// of the sequential matrix-multiply chain in Papers 016–018.
//
// Math:
//   α_t(j) = log[ Σ_i exp(α_{t-1}(i) + log(A[i,j])) ] + log(B[j, o_t])
//
// The logsumexp reduction uses the max-subtract trick for numerical stability:
//   logsumexp(x) = max(x) + log(Σ exp(x - max(x)))
//
// Absorption target: barracuda::ops::hmm or StatefulPipeline extension
// Validates against: neuralSpring src/hmm.rs (Papers 016–018)
// Reference: Rabiner (1989), "A Tutorial on HMM", Proc IEEE 77:257

// Per-state forward value from previous timestep
@group(0) @binding(0) var<storage, read> alpha_prev: array<f32>;

// Log transition matrix A[i*N + j] = log P(state_j | state_i), row-major
@group(0) @binding(1) var<storage, read> log_trans: array<f32>;

// Log emission for current observation: log_emit[j] = log P(o_t | state_j)
@group(0) @binding(2) var<storage, read> log_emit: array<f32>;

// Output: forward values for current timestep
@group(0) @binding(3) var<storage, read_write> alpha_curr: array<f32>;

// Uniform params
struct HmmParams {
    n_states: u32,
}
@group(0) @binding(4) var<uniform> params: HmmParams;

// Each workgroup thread handles one destination state j.
// For each j, we compute logsumexp over all source states i.
@compute @workgroup_size(256)
fn hmm_forward_log(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let n = params.n_states;
    if j >= n {
        return;
    }

    // Pass 1: find max for numerical stability
    var max_val: f32 = -3.4028235e+38; // -FLT_MAX
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let v = alpha_prev[i] + log_trans[i * n + j];
        max_val = max(max_val, v);
    }

    // Pass 2: accumulate exp(x - max)
    var sum_exp: f32 = 0.0;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let v = alpha_prev[i] + log_trans[i * n + j];
        sum_exp = sum_exp + exp(v - max_val);
    }

    // logsumexp + emission
    alpha_curr[j] = max_val + log(sum_exp) + log_emit[j];
}
