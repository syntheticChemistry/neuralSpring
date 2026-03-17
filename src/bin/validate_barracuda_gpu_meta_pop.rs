// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: Meta-population dynamics (Paper 025).
//!
//! Validates GPU `Tensor::matmul` for meta-population dynamics: migration
//! matrix computation, allele frequency update, covariance (pop × pop^T),
//! FST-related structure.
//!
//! ## S-14 workaround
//!
//! Uses `freq × migration^T` and `pop × pop^T` patterns (transpose second operand).
//!
//! ## S-15 workaround
//!
//! All data uses `rng.uniform()` ([0, 1)) to avoid matmul hang on RTX 4070.
//!
//! ## Provenance
//!
//! Python baseline: `control/meta_population/meta_population.py`

#![expect(clippy::cast_possible_truncation, reason = "validation binary")]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_tensor;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{gpu_readback, max_abs_diff_gpu_vs_cpu, ValidationHarness};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            println!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("barracuda_gpu_meta_pop");

    validate_migration_matmul(&mut h, &device);
    validate_allele_frequency_update(&mut h, &device);
    validate_covariance_matmul(&mut h, &device);
    validate_fst_structure(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

/// CPU reference: A (M×K) × B^T (N×K)^T → (M×N).
fn cpu_matmul_a_bt(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b.len();
    let depth = a[0].len();
    let mut out = vec![vec![0.0_f64; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            for k in 0..depth {
                out[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    out
}

/// Check 1: Migration matrix computation via matmul.
/// `new_freq` = migration (P×P) × freq (P×L). Use freq stored (L×P), migration.matmul(freq.transpose()).
fn validate_migration_matmul(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(42);
    let n_pops = 6_usize;
    let n_loci = 10_usize;

    let migration: Vec<Vec<f64>> = (0..n_pops)
        .map(|_| (0..n_pops).map(|_| rng.uniform()).collect())
        .collect();
    let freq: Vec<Vec<f64>> = (0..n_pops)
        .map(|_| (0..n_loci).map(|_| rng.uniform()).collect())
        .collect();

    let mut cpu_out = vec![0.0_f64; n_pops * n_loci];
    for i in 0..n_pops {
        for l in 0..n_loci {
            for q in 0..n_pops {
                cpu_out[i * n_loci + l] += migration[i][q] * freq[q][l];
            }
        }
    }

    let migration_flat: Vec<f32> = migration
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let freq_flat: Vec<f32> = freq
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let migration_t = gpu_tensor!(h, &migration_flat, &[n_pops, n_pops], device);
    let freq_t = gpu_tensor!(h, &freq_flat, &[n_pops, n_loci], device);

    let out_t = match migration_t.matmul(&freq_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("migration × freq: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };
    let max_diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_out);
    h.check_upper(
        &format!("migration matmul: max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// Check 2: Allele frequency update (same migration × freq).
fn validate_allele_frequency_update(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(43);
    let n_pops = 6_usize;
    let n_loci = 10_usize;

    let migration: Vec<Vec<f64>> = (0..n_pops)
        .map(|_| (0..n_pops).map(|_| rng.uniform()).collect())
        .collect();
    let freq: Vec<Vec<f64>> = (0..n_pops)
        .map(|_| (0..n_loci).map(|_| rng.uniform()).collect())
        .collect();

    let mut cpu_out = vec![0.0_f64; n_pops * n_loci];
    for i in 0..n_pops {
        for l in 0..n_loci {
            for q in 0..n_pops {
                cpu_out[i * n_loci + l] += migration[i][q] * freq[q][l];
            }
        }
    }

    let migration_flat: Vec<f32> = migration
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();
    let freq_flat: Vec<f32> = freq
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let migration_t = gpu_tensor!(h, &migration_flat, &[n_pops, n_pops], device);
    let freq_t = gpu_tensor!(h, &freq_flat, &[n_pops, n_loci], device);

    let out_t = match migration_t.matmul(&freq_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("allele freq update matmul: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };
    let max_diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_out);
    h.check_upper(
        &format!("allele frequency update: max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// Check 3: Covariance via matmul (pop × pop^T). pop (`n_pops` × `n_loci`) × pop^T.
fn validate_covariance_matmul(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(44);
    let n_pops = 6_usize;
    let n_loci = 10_usize;

    let pop: Vec<Vec<f64>> = (0..n_pops)
        .map(|_| (0..n_loci).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_gram = cpu_matmul_a_bt(&pop, &pop);
    let cpu_flat: Vec<f64> = cpu_gram.iter().flat_map(|r| r.iter().copied()).collect();

    let pop_flat: Vec<f32> = pop
        .iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect();

    let pop_t1 = gpu_tensor!(h, &pop_flat, &[n_pops, n_loci], device);
    let pop_t2 = gpu_tensor!(h, &pop_flat, &[n_pops, n_loci], device);
    let pop_t2_t = match pop_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let gram_t = match pop_t1.matmul(&pop_t2_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("pop × pop^T: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &gram_t) else {
        return;
    };
    let max_diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_flat);
    h.check_upper(
        &format!("covariance pop × pop^T: max diff ({max_diff:.2e})"),
        max_diff,
        tolerances::BARRACUDA_GPU_ECO_F32,
    );
}

/// Check 4: FST-related structure — covariance diagonal non-negative.
fn validate_fst_structure(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(45);
    let n_pops = 6_usize;
    let n_loci = 10_usize;

    let pop_flat: Vec<f32> = (0..n_pops * n_loci).map(|_| rng.uniform() as f32).collect();

    let pop_t1 = gpu_tensor!(h, &pop_flat, &[n_pops, n_loci], device);
    let pop_t2 = gpu_tensor!(h, &pop_flat, &[n_pops, n_loci], device);
    let pop_t2_t = match pop_t2.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let gram_t = match pop_t1.matmul(&pop_t2_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("pop × pop^T: {e}"), false);
            return;
        }
    };

    let Some(gram) = gpu_readback(h, &gram_t) else {
        return;
    };
    let min_diag = (0..n_pops)
        .map(|i| f64::from(gram[i * n_pops + i]))
        .fold(f64::INFINITY, f64::min);

    h.check_lower(
        &format!("FST structure: covariance diagonal non-negative ({min_diag:.2e})"),
        min_diag,
        tolerances::VARIANCE_FLOOR,
    );
}

/// Check 5: Determinism.
fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(46);
    let n_pops = 6_usize;
    let n_loci = 10_usize;

    let migration_flat: Vec<f32> = (0..n_pops * n_pops).map(|_| rng.uniform() as f32).collect();
    let freq_flat: Vec<f32> = (0..n_pops * n_loci).map(|_| rng.uniform() as f32).collect();

    let run = |_: u32| -> Option<Vec<f32>> {
        let m = Tensor::from_data(&migration_flat, vec![n_pops, n_pops], device.clone()).ok()?;
        let f = Tensor::from_data(&freq_flat, vec![n_pops, n_loci], device.clone()).ok()?;
        let out = m.matmul(&f).ok()?;
        out.to_vec().ok()
    };

    let Some(r1) = run(1) else {
        h.check_bool("determinism run1 failed", false);
        return;
    };
    let Some(r2) = run(2) else {
        h.check_bool("determinism run2 failed", false);
        return;
    };

    let identical = r1
        .iter()
        .zip(r2.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    h.check_bool("determinism: two GPU runs bit-identical", identical);
}
