// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `barracuda::shaders::quantized` CPU primitives.
//!
//! Validates `dequant_q8_cpu`, `dequant_q4_cpu`, and `gemv_quantized_cpu`
//! against hand-constructed quantized blocks with known expected outputs.
//!
//! Maps to Study 005 (Quantized Inference): INT8/INT4 accuracy validation.
//!
//! ## Provenance
//!
//! Expected values: manually constructed Q8/Q4 blocks with scale=1.0.
//! Format follows llama.cpp `Q4_0` and `Q8_0` block layout.

use barracuda::shaders::quantized;
use neural_spring::validation::ValidationHarness;

fn main() {
    let mut h = ValidationHarness::new("barracuda_quantized");

    validate_q8_dequant(&mut h);
    validate_q4_dequant(&mut h);
    validate_quantized_gemv(&mut h);
    validate_quant_properties(&mut h);

    h.finish();
}

fn validate_q8_dequant(h: &mut ValidationHarness) {
    // Q8_0 block: 2 bytes scale (f16) + 32 bytes data (i8)
    // Scale = 1.0 as f16 = 0x3C00 (little-endian: [0x00, 0x3C])
    let mut data = vec![0_u8; quantized::Q8_BYTES_PER_BLOCK];
    data[0] = 0x00;
    data[1] = 0x3C;

    // Set first few quantized values
    data[2] = 10_u8; // +10
    data[3] = 246_u8; // -10 as u8 (i8 reinterpret)
    data[4] = 0_u8; // 0
    data[5] = 127_u8; // +127

    let result = quantized::dequant_q8_cpu(&data, 4);
    h.check_abs("Q8 dequant[0] = 10", f64::from(result[0]), 10.0, 0.5);
    h.check_abs("Q8 dequant[1] = -10", f64::from(result[1]), -10.0, 0.5);
    h.check_abs("Q8 dequant[2] = 0", f64::from(result[2]), 0.0, 0.5);

    // Length check
    h.check_bool("Q8 dequant length == 4", result.len() == 4);
}

fn validate_q4_dequant(h: &mut ValidationHarness) {
    // Q4_0 block: 2 bytes scale (f16) + 16 bytes data (4-bit pairs)
    // Scale = 1.0 as f16
    let mut data = vec![0_u8; quantized::Q4_BYTES_PER_BLOCK];
    data[0] = 0x00;
    data[1] = 0x3C;

    // Each byte encodes two 4-bit values: low nibble, high nibble
    // Value = (nibble - 8) * scale
    // nibble=8 → 0, nibble=15 → +7, nibble=0 → -8
    data[2] = 0x88; // low=8→0, high=8→0

    let result = quantized::dequant_q4_cpu(&data, 2);
    h.check_abs("Q4 dequant zero pair", f64::from(result[0]), 0.0, 0.5);
    h.check_abs("Q4 dequant zero pair[1]", f64::from(result[1]), 0.0, 0.5);

    // Length check
    h.check_bool("Q4 dequant length == 2", result.len() == 2);
}

fn validate_quantized_gemv(h: &mut ValidationHarness) {
    // Build a 1×32 Q8 matrix (single row, one block) with scale=1.0
    // and all quant values = 1 (so dequantized row is all 1.0s)
    let cols = 32_usize;
    let rows = 1_usize;

    let mut a_quant = vec![0_u8; quantized::Q8_BYTES_PER_BLOCK];
    a_quant[0] = 0x00;
    a_quant[1] = 0x3C; // scale = 1.0

    for i in 0..32 {
        a_quant[2 + i] = 1_u8; // all +1
    }

    let input: Vec<f32> = vec![1.0; cols];

    let output =
        quantized::gemv_quantized_cpu(&a_quant, &input, rows, cols, quantized::QuantType::Q8_0);

    // output[0] = sum(1.0 * 1.0 for 32 elements) = 32.0 (approximately, due to f16 scale)
    h.check_abs("Q8 GEMV ones·ones ≈ 32", f64::from(output[0]), 32.0, 1.0);

    h.check_bool("Q8 GEMV output length == 1", output.len() == rows);
}

fn validate_quant_properties(h: &mut ValidationHarness) {
    // Block size
    h.check_bool(
        "Q4 block_size == 32",
        quantized::QuantType::Q4_0.block_size() == 32,
    );
    h.check_bool(
        "Q8 block_size == 32",
        quantized::QuantType::Q8_0.block_size() == 32,
    );

    // Bytes per block
    h.check_bool(
        "Q4 bytes_per_block == 18",
        quantized::QuantType::Q4_0.bytes_per_block() == 18,
    );
    h.check_bool(
        "Q8 bytes_per_block == 34",
        quantized::QuantType::Q8_0.bytes_per_block() == 34,
    );

    // Compression ratio: Q4 ≈ 7.1x, Q8 ≈ 3.8x
    let q4_ratio = quantized::QuantType::Q4_0.compression_ratio();
    h.check_bool("Q4 compression > 7x", q4_ratio > 7.0);

    let q8_ratio = quantized::QuantType::Q8_0.compression_ratio();
    h.check_bool("Q8 compression > 3.5x", q8_ratio > 3.5);
}
