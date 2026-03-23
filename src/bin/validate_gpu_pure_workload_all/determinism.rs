// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-domain determinism check (batch fitness twice).

use barracuda::ops::bio::BatchFitnessGpu;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::validation::{ValidationHarness, output_buf, storage_buf};
use std::sync::Arc;

pub fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop = 16_usize;
    let glen = 8_usize;
    let mut rng = Rng::new(42);
    let genotypes: Vec<f64> = (0..pop * glen).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..glen).map(|_| rng.uniform()).collect();
    let op = BatchFitnessGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();

    let run = || -> Result<f64, String> {
        let g = storage_buf(device, "det_g", bytemuck::cast_slice(&genotypes));
        let w = storage_buf(device, "det_w", bytemuck::cast_slice(&weights));
        let o = output_buf(device, "det_o", (pop * 8) as u64);
        op.dispatch(&g, &w, &o, pop as u32, glen as u32);
        let f = gpu.read_buffer_f64(&o, pop)?;
        Ok(f.iter().sum::<f64>() / f.len() as f64)
    };

    match (run(), run()) {
        (Ok(a), Ok(b)) => {
            h.check_bool(
                &format!("determinism: run1={a:.10} == run2={b:.10}"),
                (a - b).abs() < f64::EPSILON,
            );
        }
        _ => h.check_bool("determinism: dispatch failed", false),
    }
}
