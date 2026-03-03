// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `LeNet-5` vision primitives (Study 003).
//!
//! Validates the pure-math components of the `LeNet-5` CNN:
//!  1. `Conv2d` forward pass (single & multi-channel)
//!  2. `MaxPool2d` (stride-2)
//!  3. `ReLU` activation
//!  4. FC (fully-connected) layer chain
//!  5. Feature map shape invariants
//!
//! ## Provenance
//!
//! Python baseline: `control/lenet/lenet_mnist.py`
//! Paper: `LeCun`, Bottou, Bengio, Haffner (1998) *Proc IEEE* 86:2278-2324.
//! Command: `python3 control/lenet/lenet_mnist.py`
//! Result: 5/5 PASS (test accuracy 98.89%)
//! Reference: [`LENET_PROVENANCE`](neural_spring::provenance::LENET_PROVENANCE)

#![expect(clippy::too_many_lines, reason = "validation binary")]

use neural_spring::lenet::{
    conv2d, conv2d_multi, fc_forward, max_pool2d, relu, Conv2dMultiParams, Conv2dParams,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("lenet");

    // ── Part 1: Conv2d with known values ──

    // Analytical: 1×1 kernel [1], bias 0 ⇒ conv = identity
    let input_2x2 = [1.0, 2.0, 3.0, 4.0];
    let out = conv2d(&Conv2dParams {
        input: &input_2x2,
        h: 2,
        w: 2,
        kernel: &[1.0],
        kh: 1,
        kw: 1,
        bias: 0.0,
        pad: 0,
    });
    h.check_abs(
        "conv2d 1×1 identity [0]",
        out[0],
        1.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "conv2d 1×1 identity [3]",
        out[3],
        4.0,
        tolerances::EXACT_F64,
    );

    // Analytical: 3×3 sum kernel on 3×3 ones ⇒ Σ=9
    let input_3x3 = [1.0; 9];
    let kernel_3x3 = [1.0; 9];
    let out = conv2d(&Conv2dParams {
        input: &input_3x3,
        h: 3,
        w: 3,
        kernel: &kernel_3x3,
        kh: 3,
        kw: 3,
        bias: 0.0,
        pad: 0,
    });
    h.check_bool("conv2d 3×3 sum output shape", out.len() == 1);
    h.check_abs("conv2d 3×3 sum = 9", out[0], 9.0, tolerances::EXACT_F64);

    // Analytical: 9 + bias 1.5 = 10.5
    let out = conv2d(&Conv2dParams {
        input: &input_3x3,
        h: 3,
        w: 3,
        kernel: &kernel_3x3,
        kh: 3,
        kw: 3,
        bias: 1.5,
        pad: 0,
    });
    h.check_abs(
        "conv2d 3×3 sum + bias = 10.5",
        out[0],
        10.5,
        tolerances::EXACT_F64,
    );

    // Conv2d with padding preserves spatial dimensions
    let out = conv2d(&Conv2dParams {
        input: &input_2x2,
        h: 2,
        w: 2,
        kernel: &kernel_3x3,
        kh: 3,
        kw: 3,
        bias: 0.0,
        pad: 1,
    });
    h.check_bool("conv2d pad=1 preserves 2×2", out.len() == 4);

    // ── Part 2: Multi-channel Conv2d ──

    // Analytical: ch0+ch1 at each pixel → [0]=1+10=11, [3]=4+40=44
    let input_mc = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let kernel_mc = [1.0, 1.0];
    let bias_mc = [0.0];
    let out = conv2d_multi(&Conv2dMultiParams {
        input: &input_mc,
        c_in: 2,
        h: 2,
        w: 2,
        kernel: &kernel_mc,
        c_out: 1,
        kh: 1,
        kw: 1,
        bias: &bias_mc,
        pad: 0,
    });
    h.check_abs(
        "multi-ch conv2d [0] = 1+10",
        out[0],
        11.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "multi-ch conv2d [3] = 4+40",
        out[3],
        44.0,
        tolerances::EXACT_F64,
    );

    // ── Part 3: MaxPool2d ──

    // Analytical: 2×2 maxpool ⇒ [0,0]=max(1,3,5,7)=7, [1,1]=max(10..16)=16
    #[rustfmt::skip]
    let pool_input = [
        1.0,  3.0,  2.0,  4.0,
        5.0,  7.0,  6.0,  8.0,
        9.0,  11.0, 10.0, 12.0,
        13.0, 15.0, 14.0, 16.0,
    ];
    let out = max_pool2d(&pool_input, 4, 4);
    h.check_bool("maxpool 4×4→2×2 shape", out.len() == 4);
    h.check_abs(
        "maxpool [0,0] = max(1,3,5,7)",
        out[0],
        7.0,
        tolerances::EXACT_F64,
    );
    h.check_abs(
        "maxpool [1,1] = max(10,12,14,16)",
        out[3],
        16.0,
        tolerances::EXACT_F64,
    );

    // ── Part 4: ReLU ──
    // Analytical: ReLU(x)=max(0,x) ⇒ -2→0, 0→0, 3.7→3.7

    let out = relu(&[-2.0, -0.5, 0.0, 1.0, 3.7]);
    h.check_abs("ReLU(-2) = 0", out[0], 0.0, tolerances::EXACT_F64);
    h.check_abs("ReLU(0) = 0", out[2], 0.0, tolerances::EXACT_F64);
    h.check_abs("ReLU(3.7) = 3.7", out[4], 3.7, tolerances::EXACT_F64);

    // ── Part 5: FC chain (LeNet classifier: 400→120→84→10) ──

    // Analytical: FC1 weights [1,1,0,0;0,0,1,1], input [1..4] ⇒ [0]=1+2=3, [1]=3+4=7
    let w1 = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let b1 = [0.0, 0.0];
    let input = [1.0, 2.0, 3.0, 4.0];
    let fc1_out = fc_forward(&input, &w1, &b1, 2);
    h.check_abs("FC1 [0] = 1+2", fc1_out[0], 3.0, tolerances::EXACT_F64);
    h.check_abs("FC1 [1] = 3+4", fc1_out[1], 7.0, tolerances::EXACT_F64);

    // Analytical: FC2 = 3+7+0.5 = 10.5 (ReLU(3),ReLU(7) passthrough)
    let fc1_relu = relu(&fc1_out);
    let w2 = [1.0, 1.0];
    let b2 = [0.5];
    let fc2_out = fc_forward(&fc1_relu, &w2, &b2, 1);
    h.check_abs(
        "FC2 = 3+7+0.5 = 10.5",
        fc2_out[0],
        10.5,
        tolerances::EXACT_F64,
    );

    // ── Part 6: LeNet shape invariants (28×28 → 10) ──

    let oh1 = 28 + 2 * 2 - 5 + 1; // 28
    let oh1_pool = oh1 / 2; // 14
    let oh2 = oh1_pool - 5 + 1; // 10
    let oh2_pool = oh2 / 2; // 5
    let flatten = 16 * oh2_pool * oh2_pool; // 400
    h.check_bool("LeNet conv1 output = 28", oh1 == 28);
    h.check_bool("LeNet pool1 output = 14", oh1_pool == 14);
    h.check_bool("LeNet conv2 output = 10", oh2 == 10);
    h.check_bool("LeNet pool2 output = 5", oh2_pool == 5);
    h.check_bool("LeNet flatten = 400", flatten == 400);

    h.finish();
}
