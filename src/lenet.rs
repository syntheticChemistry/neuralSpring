// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure-math primitives for `LeNet-5` validation (Study 003).
//!
//! Provides portable `Conv2d`, `MaxPool2d`, and FC-chain operations for
//! validating `BarraCUDA`'s vision primitive stack (`conv2d.wgsl`,
//! `max_pool2d.wgsl`, `nn::ReLU`) against hand-computed known values.
//!
//! Reference: `LeCun`, Bottou, Bengio, Haffner (1998) *Proc IEEE* 86:2278-2324.

#![allow(clippy::too_many_arguments)]

/// Single-channel 2D convolution parameters.
pub struct Conv2dParams<'a> {
    pub input: &'a [f64],
    pub h: usize,
    pub w: usize,
    pub kernel: &'a [f64],
    pub kh: usize,
    pub kw: usize,
    pub bias: f64,
    pub pad: usize,
}

/// 2D convolution with zero-padding (single input channel, single output channel).
///
/// Returns output of size `(h + 2*pad - kh + 1, w + 2*pad - kw + 1)`.
#[must_use]
pub fn conv2d(p: &Conv2dParams<'_>) -> Vec<f64> {
    let oh = p.h + 2 * p.pad - p.kh + 1;
    let ow = p.w + 2 * p.pad - p.kw + 1;
    let mut out = vec![0.0; oh * ow];
    for oy in 0..oh {
        for ox in 0..ow {
            let mut sum = p.bias;
            for ky in 0..p.kh {
                for kx in 0..p.kw {
                    let iy = oy + ky;
                    let ix = ox + kx;
                    let in_bounds =
                        iy >= p.pad && iy < p.h + p.pad && ix >= p.pad && ix < p.w + p.pad;
                    if in_bounds {
                        sum +=
                            p.input[(iy - p.pad) * p.w + (ix - p.pad)] * p.kernel[ky * p.kw + kx];
                    }
                }
            }
            out[oy * ow + ox] = sum;
        }
    }
    out
}

/// Multi-channel `Conv2d` parameters.
pub struct Conv2dMultiParams<'a> {
    pub input: &'a [f64],
    pub c_in: usize,
    pub h: usize,
    pub w: usize,
    pub kernel: &'a [f64],
    pub c_out: usize,
    pub kh: usize,
    pub kw: usize,
    pub bias: &'a [f64],
    pub pad: usize,
}

/// Multi-channel `Conv2d`: `(c_in, h, w)` input, `(c_out, c_in, kh, kw)` kernel.
///
/// Returns `(c_out, oh, ow)` output.
#[must_use]
pub fn conv2d_multi(p: &Conv2dMultiParams<'_>) -> Vec<f64> {
    let oh = p.h + 2 * p.pad - p.kh + 1;
    let ow = p.w + 2 * p.pad - p.kw + 1;
    let mut out = vec![0.0; p.c_out * oh * ow];
    let kernel_size = p.c_in * p.kh * p.kw;
    for co in 0..p.c_out {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut sum = p.bias[co];
                for ci in 0..p.c_in {
                    for ky in 0..p.kh {
                        for kx in 0..p.kw {
                            let iy = oy + ky;
                            let ix = ox + kx;
                            let in_bounds =
                                iy >= p.pad && iy < p.h + p.pad && ix >= p.pad && ix < p.w + p.pad;
                            if in_bounds {
                                let k_idx = co * kernel_size + ci * p.kh * p.kw + ky * p.kw + kx;
                                sum += p.input[ci * p.h * p.w + (iy - p.pad) * p.w + (ix - p.pad)]
                                    * p.kernel[k_idx];
                            }
                        }
                    }
                }
                out[co * oh * ow + oy * ow + ox] = sum;
            }
        }
    }
    out
}

/// 2x2 max pooling with stride 2.
///
/// `input` is `(h, w)` row-major where `h` and `w` are even.
/// Returns `(h/2, w/2)` output.
#[must_use]
pub fn max_pool2d(input: &[f64], h: usize, w: usize) -> Vec<f64> {
    let oh = h / 2;
    let ow = w / 2;
    let mut out = vec![f64::NEG_INFINITY; oh * ow];
    for oy in 0..oh {
        for ox in 0..ow {
            let mut m = f64::NEG_INFINITY;
            for dy in 0..2 {
                for dx in 0..2 {
                    let v = input[(2 * oy + dy) * w + (2 * ox + dx)];
                    if v > m {
                        m = v;
                    }
                }
            }
            out[oy * ow + ox] = m;
        }
    }
    out
}

/// `ReLU` activation (element-wise).
#[must_use]
pub fn relu(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// Fully-connected layer: `output = W * input + b`.
///
/// `weights` is `(out_dim, in_dim)` row-major.
#[must_use]
pub fn fc_forward(input: &[f64], weights: &[f64], bias: &[f64], out_dim: usize) -> Vec<f64> {
    let in_dim = input.len();
    let mut out = vec![0.0; out_dim];
    for i in 0..out_dim {
        let mut sum = bias[i];
        for j in 0..in_dim {
            sum = input[j].mul_add(weights[i * in_dim + j], sum);
        }
        out[i] = sum;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv2d_identity_kernel() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = conv2d(&Conv2dParams {
            input: &input,
            h: 2,
            w: 2,
            kernel: &[1.0],
            kh: 1,
            kw: 1,
            bias: 0.0,
            pad: 0,
        });
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn conv2d_sum_kernel_3x3() {
        let input = vec![1.0; 9];
        let out = conv2d(&Conv2dParams {
            input: &input,
            h: 3,
            w: 3,
            kernel: &[1.0; 9],
            kh: 3,
            kw: 3,
            bias: 0.0,
            pad: 0,
        });
        assert_eq!(out.len(), 1);
        assert!((out[0] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn conv2d_with_padding() {
        // 2×2 input [1,2; 3,4], 3×3 all-ones kernel, pad=1
        // Padded 4×4: [0,0,0,0; 0,1,2,0; 0,3,4,0; 0,0,0,0]
        // Output (0,0) sees top-left 3×3: [0,0,0; 0,1,2; 0,3,4] = 10
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = conv2d(&Conv2dParams {
            input: &input,
            h: 2,
            w: 2,
            kernel: &[1.0; 9],
            kh: 3,
            kw: 3,
            bias: 0.0,
            pad: 1,
        });
        assert_eq!(out.len(), 4);
        assert!((out[0] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn max_pool2d_basic() {
        let input = vec![1.0, 3.0, 2.0, 4.0];
        let out = max_pool2d(&input, 2, 2);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn relu_positive_and_negative() {
        let out = relu(&[-2.0, 0.0, 3.0, -1.0]);
        assert_eq!(out, vec![0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn fc_identity() {
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![0.0, 0.0];
        let out = fc_forward(&[3.0, 7.0], &w, &b, 2);
        assert!((out[0] - 3.0).abs() < 1e-12);
        assert!((out[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn fc_with_bias() {
        let w = vec![2.0];
        let b = vec![1.0];
        let out = fc_forward(&[3.0], &w, &b, 1);
        assert!((out[0] - 7.0).abs() < 1e-12);
    }
}
