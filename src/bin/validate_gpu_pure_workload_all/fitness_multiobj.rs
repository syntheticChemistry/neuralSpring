// SPDX-License-Identifier: AGPL-3.0-or-later

//! Batch fitness and multi-objective fitness GPU validation (papers 011–014).

use barracuda::ops::bio::{BatchFitnessGpu, MultiObjFitnessGpu};
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, output_buf, storage_buf};
use std::sync::Arc;

pub fn validate_fitness(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop_size = 64_usize;
    let genome_len = 16_usize;
    let mut rng = Rng::new(42);
    let genotypes: Vec<f64> = (0..pop_size * genome_len).map(|_| rng.uniform()).collect();
    let weights: Vec<f64> = (0..genome_len).map(|_| rng.uniform()).collect();

    let cpu_mean = {
        let total: f64 = (0..pop_size)
            .map(|i| {
                let base = i * genome_len;
                (0..genome_len)
                    .map(|g| genotypes[base + g] * weights[g])
                    .sum::<f64>()
            })
            .sum();
        total / pop_size as f64
    };

    let op = BatchFitnessGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let geno_buf = storage_buf(device, "fit_geno", bytemuck::cast_slice(&genotypes));
    let weight_buf = storage_buf(device, "fit_w", bytemuck::cast_slice(&weights));
    let out_buf = output_buf(device, "fit_out", (pop_size * 8) as u64);

    op.dispatch(
        &geno_buf,
        &weight_buf,
        &out_buf,
        pop_size as u32,
        genome_len as u32,
    );

    match gpu.read_buffer_f64(&out_buf, pop_size) {
        Ok(fitness) => {
            let gpu_mean = fitness.iter().sum::<f64>() / fitness.len() as f64;
            h.check_abs(
                &format!("fitness 64×16: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_FITNESS_F32,
            );
        }
        Err(e) => h.check_bool(&format!("fitness: {e}"), false),
    }
}

pub fn validate_multi_obj(h: &mut ValidationHarness, gpu: &Gpu) {
    let pop = 32_usize;
    let glen = 12_usize;
    let n_obj = 3_usize;
    let mut rng = Rng::new(77);
    let genotypes: Vec<f64> = (0..pop * glen).map(|_| rng.uniform()).collect();

    let cpu_mean = {
        let mut all_fitness = Vec::with_capacity(pop * n_obj);
        for i in 0..pop {
            let individual = &genotypes[i * glen..(i + 1) * glen];
            let f = neural_spring::directed_evolution::multi_objective_fitness(individual, n_obj);
            all_fitness.extend_from_slice(&f);
        }
        all_fitness.iter().sum::<f64>() / all_fitness.len() as f64
    };

    let op = MultiObjFitnessGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let geno_buf = storage_buf(device, "mof_g", bytemuck::cast_slice(&genotypes));
    let out_buf = output_buf(device, "mof_out", (pop * n_obj * 8) as u64);

    op.dispatch(&geno_buf, &out_buf, pop as u32, glen as u32, n_obj as u32);

    match gpu.read_buffer_f64(&out_buf, pop * n_obj) {
        Ok(gpu_f) => {
            let gpu_mean = gpu_f.iter().sum::<f64>() / gpu_f.len() as f64;
            h.check_abs(
                &format!("multi_obj 32×12×3: GPU={gpu_mean:.4} vs CPU={cpu_mean:.4}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_MULTI_OBJ_FITNESS_F64,
            );
        }
        Err(e) => h.check_bool(&format!("multi_obj: {e}"), false),
    }
}
