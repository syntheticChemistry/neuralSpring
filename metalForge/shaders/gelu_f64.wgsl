// SPDX-License-Identifier: AGPL-3.0-or-later
//
// gelu_f64.wgsl — f64-precision GELU activation
//
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Uses FMA for the cubic term to preserve precision.
// No df64 accumulation needed (pointwise op), but benefits from
// compile_shader_df64 when used alongside df64 shaders.
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

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_COEFF: f32 = 0.044715;

@compute @workgroup_size(256)
fn gelu_f64(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }

    let x = input[idx];
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * fma(GELU_COEFF, x3, x);
    let t = tanh(inner);
    output[idx] = 0.5 * x * (1.0 + t);
}
