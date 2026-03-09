// SPDX-License-Identifier: AGPL-3.0-or-later
//
// gelu_f64.wgsl — GELU activation (f64 approximation)
//
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Uses f32 arithmetic with f64 I/O for cross-driver compatibility.
// The f32 compute path avoids df64 transcendental issues on some GPUs
// (exp_df64 regression on Ada Lovelace). Precision is ~7 digits which
// suffices for GELU (activation function, not accumulation).
//
// Cross-spring: transformer building block for neuralSpring, folding
// (Evoformer FFN), baseCamp (Sub-02 attention FFN).
//
// Absorption target: barracuda::ops::gelu_f64

struct Params {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }

    let x = f32(input[idx]);
    let x3 = x * x * x;
    let inner = 0.7978845608 * (x + 0.044715 * x3);
    let t = tanh(inner);
    let result = 0.5 * x * (1.0 + t);
    output[idx] = f64(result);
}
