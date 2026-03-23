// SPDX-License-Identifier: AGPL-3.0-or-later

//! HMM batch forward (papers 016–018) and spatial payoff game theory (019).

use barracuda::ops::bio::{HmmBatchForwardF64, SpatialPayoffGpu};
use neural_spring::gpu::Gpu;
use neural_spring::hmm::Hmm;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, output_buf, storage_buf};
use std::sync::Arc;

pub fn validate_hmm(h: &mut ValidationHarness, gpu: &Gpu) {
    let hmm = Hmm::new(
        vec![
            vec![0.7, 0.2, 0.1],
            vec![0.2, 0.6, 0.2],
            vec![0.1, 0.2, 0.7],
        ],
        vec![
            vec![0.4, 0.3, 0.3],
            vec![0.2, 0.5, 0.3],
            vec![0.3, 0.3, 0.4],
        ],
        vec![0.33, 0.34, 0.33],
    );

    let mut rng = Rng::new(42);
    let n_seqs = 8_usize;
    let seq_len = 20_usize;
    let mut obs_batch = Vec::with_capacity(n_seqs);
    for _ in 0..n_seqs {
        let (_, obs) = hmm.generate_sequence(seq_len, &mut rng);
        obs_batch.push(obs);
    }

    let cpu_mean = {
        let mut sum = 0.0_f64;
        for obs in &obs_batch {
            let (_, ll) = hmm.forward(obs);
            sum += ll;
        }
        sum / n_seqs as f64
    };

    let dev = Arc::clone(gpu.wgpu_device());
    let op = match HmmBatchForwardF64::new(dev) {
        Ok(o) => o,
        Err(e) => {
            h.check_bool(&format!("HMM create: {e}"), false);
            return;
        }
    };

    let n_states = hmm.num_states() as u32;
    let n_symbols = hmm.num_symbols() as u32;
    let log_trans: Vec<f64> = hmm.transition.iter().map(|&p| p.ln()).collect();
    let log_emit: Vec<f64> = hmm.emission.iter().map(|&p| p.ln()).collect();
    let log_pi: Vec<f64> = hmm.initial.iter().map(|&p| p.ln()).collect();

    let mut obs_flat: Vec<u32> = Vec::with_capacity(n_seqs * seq_len);
    for seq in &obs_batch {
        for &o in seq {
            obs_flat.push(o as u32);
        }
        obs_flat.extend(std::iter::repeat_n(0u32, seq_len.saturating_sub(seq.len())));
    }

    let device = gpu.device();
    let lt_buf = storage_buf(device, "hmm_lt", bytemuck::cast_slice(&log_trans));
    let le_buf = storage_buf(device, "hmm_le", bytemuck::cast_slice(&log_emit));
    let lp_buf = storage_buf(device, "hmm_lp", bytemuck::cast_slice(&log_pi));
    let obs_buf = storage_buf(device, "hmm_obs", bytemuck::cast_slice(&obs_flat));
    let alpha_size = (n_seqs * seq_len * n_states as usize * 8) as u64;
    let alpha_buf = output_buf(device, "hmm_a", alpha_size);
    let ll_buf = output_buf(device, "hmm_ll", (n_seqs * 8) as u64);

    if let Err(e) = op.dispatch(&barracuda::ops::bio::hmm::HmmForwardArgs {
        n_states,
        n_symbols,
        n_steps: seq_len as u32,
        n_seqs: n_seqs as u32,
        log_trans: &lt_buf,
        log_emit: &le_buf,
        log_pi: &lp_buf,
        observations: &obs_buf,
        log_alpha_out: &alpha_buf,
        log_lik_out: &ll_buf,
    }) {
        h.check_bool(&format!("HMM dispatch: {e}"), false);
        return;
    }

    match gpu.read_buffer_f64(&ll_buf, n_seqs) {
        Ok(ll) => {
            let gpu_mean = ll.iter().sum::<f64>() / ll.len() as f64;
            h.check_abs(
                &format!("HMM 3×3, 8seq: GPU={gpu_mean:.4} vs CPU={cpu_mean:.4}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_HMM_ALPHA_F32,
            );
        }
        Err(e) => h.check_bool(&format!("HMM readback: {e}"), false),
    }
}

pub fn validate_spatial_payoff(h: &mut ValidationHarness, gpu: &Gpu) {
    let grid_size = 16_u32;
    let gs = grid_size as usize;
    let n = gs * gs;
    let b = 1.5_f32;
    let c = 1.0_f32;
    let mut rng = Rng::new(99);
    let grid: Vec<u32> = (0..n).map(|_| u32::from(rng.uniform() > 0.5)).collect();

    let cpu_mean = {
        let neighbors: [(i32, i32); 8] = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ];
        let gn = gs as i32;
        let mut total = 0.0_f32;
        for i in 0..gs {
            for j in 0..gs {
                let me = grid[i * gs + j];
                for (di, dj) in &neighbors {
                    let ni = ((i as i32 + di).rem_euclid(gn)) as usize;
                    let nj = ((j as i32 + dj).rem_euclid(gn)) as usize;
                    let other = grid[ni * gs + nj];
                    total += match (me, other) {
                        (1, 1) => b - c,
                        (1, 0) => -c,
                        (0, 1) => b,
                        _ => 0.0,
                    };
                }
            }
        }
        total / n as f32
    };

    let op = SpatialPayoffGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let grid_buf = storage_buf(device, "sp_grid", bytemuck::cast_slice(&grid));
    let out_buf = output_buf(device, "sp_out", (n * 4) as u64);

    op.dispatch(&grid_buf, &out_buf, grid_size, b, c);

    match gpu.read_buffer_f32(&out_buf, n) {
        Ok(gpu_p) => {
            let gpu_mean = gpu_p.iter().sum::<f32>() / gpu_p.len() as f32;
            h.check_abs(
                &format!("spatial 16×16: GPU={gpu_mean:.4} vs CPU={cpu_mean:.4}"),
                f64::from(gpu_mean),
                f64::from(cpu_mean),
                tolerances::GPU_SPATIAL_PAYOFF_F32,
            );
        }
        Err(e) => h.check_bool(&format!("spatial: {e}"), false),
    }
}
