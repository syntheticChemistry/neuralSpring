// SPDX-License-Identifier: AGPL-3.0-or-later
//
// triangle_attention_f64.wgsl — Triangle self-attention with pair bias
//
// From Jumper et al. 2021 (AlphaFold2), Algorithms 13-14.
// Performs row-wise self-attention on pair representation z[i,j,c]
// with additive pair bias b[j,k]:
//
//   For each row i:
//     Q[j,h] = linear_q(z[i,j])  (query projection)
//     K[k,h] = linear_k(z[i,k])  (key projection)
//     V[k,h] = linear_v(z[i,k])  (value projection)
//     logit[j,k] = Q[j,h] * K[k,h] / sqrt(c_h) + bias[j,k]
//     w[j,k] = softmax_k(logit[j,k]) * gate[j]
//     z[i,j] += sum_k w[j,k] * V[k,h]
//
// This shader computes the biased attention scores (pass 1).
// Softmax (pass 2) uses softmax_f64.wgsl.
// Value application (pass 3) uses attention_apply_f64.wgsl.
//
// For column-wise attention (Algorithm 14), transpose the pair
// representation before dispatch: z[j,i,c] instead of z[i,j,c].
//
// Absorption target: barracuda::ops::triangle_attention_f64
// Requires: df64_core.wgsl (auto-injected by compile_shader_df64)

struct Params {
    n_rows:   u32,
    n_res:    u32,
    n_heads:  u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read>       query: array<f32>;
@group(0) @binding(1) var<storage, read>       key:   array<f32>;
@group(0) @binding(2) var<storage, read>       bias:  array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;
@group(0) @binding(4) var<uniform>             params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.n_res;
    let H = params.n_heads;
    let D = params.head_dim;
    let R = params.n_rows;

    let total = R * H * N * N;
    let idx = gid.x;
    if idx >= total { return; }

    let rh  = idx / (N * N);
    let rem = idx % (N * N);
    let j   = rem / N;
    let k   = rem % N;
    let row = rh / H;
    let h   = rh % H;

    if row >= R { return; }

    let q_base = (row * N * H + j * H + h) * D;
    let k_base = (row * N * H + k * H + h) * D;

    var acc = df64_zero();
    for (var d = 0u; d < D; d++) {
        let prod = two_prod(query[q_base + d], key[k_base + d]);
        acc = df64_add(acc, prod);
    }

    let scale = sqrt(f32(D));
    let score = acc.hi / scale;

    let bias_idx = h * N * N + j * N + k;
    let biased_score = score + bias[bias_idx];

    let out_idx = row * H * N * N + h * N * N + j * N + k;
    scores[out_idx] = biased_score;
}
