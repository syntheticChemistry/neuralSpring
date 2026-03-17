// SPDX-License-Identifier: AGPL-3.0-or-later

//! `BarraCUDA` GPU validation: swarm robotics heterogeneous controllers (Paper 015).
//!
//! Validates that `BarraCUDA` `Tensor` matmul + tanh + add on GPU correctly
//! compute NN forward passes for heterogeneous swarm robot controllers (small MLPs).
//! output = tanh(input × weights^T + bias).
//!
//! ## S-14 workaround
//!
//! All matmul operations use A × B^T (transpose second operand).
//!
//! ## S-15 workaround
//!
//! All data uses `rng.uniform()` ([0, 1)) to avoid matmul hang.
//!
//! ## Provenance
//!
//! CPU baseline: `validate_barracuda_swarm`, `validate_swarm_robotics`

#![expect(
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "validation binary"
)]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::gpu_tensor;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::{gpu_readback, max_abs_diff_gpu_vs_cpu, ValidationHarness};
use std::sync::Arc;

fn cpu_a_bt(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b.len();
    let depth = a[0].len();
    let mut out = vec![vec![0.0_f64; cols]; rows];
    for row_idx in 0..rows {
        for col_idx in 0..cols {
            for inner_idx in 0..depth {
                out[row_idx][col_idx] += a[row_idx][inner_idx] * b[col_idx][inner_idx];
            }
        }
    }
    out
}

fn flatten_f32(data: &[Vec<f64>]) -> Vec<f32> {
    data.iter()
        .flat_map(|r| r.iter().map(|&x| x as f32))
        .collect()
}

fn flatten_f64(data: &[Vec<f64>]) -> Vec<f64> {
    data.iter().flat_map(|r| r.iter().copied()).collect()
}

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
    let mut h = ValidationHarness::new("barracuda_gpu_swarm");

    validate_nn_forward_pass(&mut h, &device);
    validate_heterogeneous_outputs_differ(&mut h, &device);
    validate_tanh_activation(&mut h, &device);
    validate_output_finite(&mut h, &device);
    validate_determinism(&mut h, &device);

    h.finish();
}

fn validate_nn_forward_pass(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(42);
    let n_robots = 10_usize;
    let input_dim = 6_usize;
    let hidden_dim = 8_usize;
    let output_dim = 3_usize;

    let input: Vec<Vec<f64>> = (0..n_robots)
        .map(|_| (0..input_dim).map(|_| rng.uniform()).collect())
        .collect();
    let w1: Vec<Vec<f64>> = (0..hidden_dim)
        .map(|_| (0..input_dim).map(|_| rng.uniform()).collect())
        .collect();
    let w2: Vec<Vec<f64>> = (0..output_dim)
        .map(|_| (0..hidden_dim).map(|_| rng.uniform()).collect())
        .collect();
    let bias1: Vec<f64> = (0..hidden_dim).map(|_| rng.uniform()).collect();
    let bias2: Vec<f64> = (0..output_dim).map(|_| rng.uniform()).collect();

    let h1_linear = cpu_a_bt(&input, &w1);
    let h1_biased: Vec<Vec<f64>> = h1_linear
        .iter()
        .map(|row| row.iter().zip(bias1.iter()).map(|(&v, &b)| v + b).collect())
        .collect();
    let h1_tanh: Vec<Vec<f64>> = h1_biased
        .iter()
        .map(|row| row.iter().map(|&x| x.tanh()).collect())
        .collect();
    let out_linear = cpu_a_bt(&h1_tanh, &w2);
    let cpu_out: Vec<f64> = out_linear
        .iter()
        .flat_map(|row| row.iter().zip(bias2.iter()).map(|(&v, &b)| (v + b).tanh()))
        .collect();

    let inp_t = gpu_tensor!(h, &flatten_f32(&input), &[n_robots, input_dim], device);
    let w1_t = gpu_tensor!(h, &flatten_f32(&w1), &[hidden_dim, input_dim], device);
    let w1_t_t = match w1_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose w1: {e}"), false);
            return;
        }
    };
    let w2_t = gpu_tensor!(h, &flatten_f32(&w2), &[output_dim, hidden_dim], device);
    let w2_t_t = match w2_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose w2: {e}"), false);
            return;
        }
    };

    let bias1_broadcast: Vec<f32> = bias1
        .iter()
        .cycle()
        .take(n_robots * hidden_dim)
        .map(|&x| x as f32)
        .collect();
    let bias2_broadcast: Vec<f32> = bias2
        .iter()
        .cycle()
        .take(n_robots * output_dim)
        .map(|&x| x as f32)
        .collect();

    let h1_t = match inp_t.matmul(&w1_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul layer1: {e}"), false);
            return;
        }
    };
    let h1_biased_t = match h1_t.add(&gpu_tensor!(
        h,
        &bias1_broadcast,
        &[n_robots, hidden_dim],
        device
    )) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("add bias1: {e}"), false);
            return;
        }
    };
    let h1_act_t = match h1_biased_t.tanh() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("tanh layer1: {e}"), false);
            return;
        }
    };
    let out_linear_t = match h1_act_t.matmul(&w2_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul layer2: {e}"), false);
            return;
        }
    };
    let out_t = match out_linear_t.add(&gpu_tensor!(
        h,
        &bias2_broadcast,
        &[n_robots, output_dim],
        device
    )) {
        Ok(t) => match t.tanh() {
            Ok(a) => a,
            Err(e) => {
                h.check_bool(&format!("tanh layer2: {e}"), false);
                return;
            }
        },
        Err(e) => {
            h.check_bool(&format!("add bias2: {e}"), false);
            return;
        }
    };

    let Some(out_gpu) = gpu_readback(h, &out_t) else {
        return;
    };

    let diff = max_abs_diff_gpu_vs_cpu(&out_gpu, &cpu_out);
    h.check_upper(
        &format!("NN forward pass: max diff ({diff:.2e})"),
        diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );
}

fn validate_heterogeneous_outputs_differ(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(123);
    let n_robots = 10_usize;
    let input_dim = 6_usize;
    let hidden_dim = 8_usize;
    let output_dim = 3_usize;

    let mut robot_outputs: Vec<Vec<f32>> = Vec::with_capacity(n_robots);
    for _robot in 0..n_robots {
        let inp: Vec<f32> = (0..input_dim).map(|_| rng.uniform() as f32).collect();
        let w1: Vec<f32> = (0..hidden_dim * input_dim)
            .map(|_| rng.uniform() as f32)
            .collect();
        let w2: Vec<f32> = (0..output_dim * hidden_dim)
            .map(|_| rng.uniform() as f32)
            .collect();

        let inp_t = gpu_tensor!(h, &inp, &[1, input_dim], device);
        let w1_t = gpu_tensor!(h, &w1, &[hidden_dim, input_dim], device);
        let w1_t_t = match w1_t.transpose() {
            Ok(t) => t,
            Err(e) => {
                h.check_bool(&format!("transpose: {e}"), false);
                return;
            }
        };
        let h1_t = match inp_t.matmul(&w1_t_t) {
            Ok(t) => match t.tanh() {
                Ok(a) => a,
                Err(e) => {
                    h.check_bool(&format!("tanh: {e}"), false);
                    return;
                }
            },
            Err(e) => {
                h.check_bool(&format!("matmul: {e}"), false);
                return;
            }
        };
        let w2_t = gpu_tensor!(h, &w2, &[output_dim, hidden_dim], device);
        let w2_t_t = match w2_t.transpose() {
            Ok(t) => t,
            Err(e) => {
                h.check_bool(&format!("transpose: {e}"), false);
                return;
            }
        };
        let out_t = match h1_t.matmul(&w2_t_t) {
            Ok(t) => match t.tanh() {
                Ok(a) => a,
                Err(e) => {
                    h.check_bool(&format!("tanh: {e}"), false);
                    return;
                }
            },
            Err(e) => {
                h.check_bool(&format!("matmul: {e}"), false);
                return;
            }
        };
        let Some(out) = gpu_readback(h, &out_t) else {
            return;
        };
        robot_outputs.push(out);
    }

    let mut at_least_one_pair_differs = false;
    for i in 0..n_robots {
        for j in (i + 1)..n_robots {
            if robot_outputs[i] != robot_outputs[j] {
                at_least_one_pair_differs = true;
                break;
            }
        }
        if at_least_one_pair_differs {
            break;
        }
    }

    h.check_bool(
        "heterogeneous controller outputs differ",
        at_least_one_pair_differs,
    );
}

fn validate_tanh_activation(
    h: &mut ValidationHarness,
    device: &Arc<barracuda::device::WgpuDevice>,
) {
    let mut rng = Rng::new(99);
    let n_robots = 10_usize;
    let input_dim = 6_usize;
    let hidden_dim = 8_usize;

    let input: Vec<Vec<f64>> = (0..n_robots)
        .map(|_| (0..input_dim).map(|_| rng.uniform()).collect())
        .collect();
    let w1: Vec<Vec<f64>> = (0..hidden_dim)
        .map(|_| (0..input_dim).map(|_| rng.uniform()).collect())
        .collect();

    let cpu_linear = cpu_a_bt(&input, &w1);
    let cpu_tanh: Vec<f64> = flatten_f64(&cpu_linear).iter().map(|&x| x.tanh()).collect();

    let inp_t = gpu_tensor!(h, &flatten_f32(&input), &[n_robots, input_dim], device);
    let w1_t = gpu_tensor!(h, &flatten_f32(&w1), &[hidden_dim, input_dim], device);
    let w1_t_t = match w1_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };
    let linear_t = match inp_t.matmul(&w1_t_t) {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };
    let act_t = match linear_t.tanh() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("tanh: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &act_t) else {
        return;
    };

    let diff = max_abs_diff_gpu_vs_cpu(&out, &cpu_tanh);
    h.check_upper(
        &format!("tanh activation: max diff ({diff:.2e})"),
        diff,
        tolerances::TENSOR_TRANSCENDENTAL_F32,
    );

    let in_range = out.iter().all(|&x| x > -1.01 && x < 1.01);
    h.check_bool("tanh output in (-1, 1)", in_range);
}

fn validate_output_finite(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(77);
    let n_robots = 10_usize;
    let input_dim = 6_usize;
    let hidden_dim = 8_usize;
    let output_dim = 3_usize;

    let input: Vec<Vec<f64>> = (0..n_robots)
        .map(|_| (0..input_dim).map(|_| rng.uniform()).collect())
        .collect();
    let w1: Vec<Vec<f64>> = (0..hidden_dim)
        .map(|_| (0..input_dim).map(|_| rng.uniform()).collect())
        .collect();
    let w2: Vec<Vec<f64>> = (0..output_dim)
        .map(|_| (0..hidden_dim).map(|_| rng.uniform()).collect())
        .collect();

    let inp_t = gpu_tensor!(h, &flatten_f32(&input), &[n_robots, input_dim], device);
    let w1_t = gpu_tensor!(h, &flatten_f32(&w1), &[hidden_dim, input_dim], device);
    let w1_t_t = match w1_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };
    let w2_t = gpu_tensor!(h, &flatten_f32(&w2), &[output_dim, hidden_dim], device);
    let w2_t_t = match w2_t.transpose() {
        Ok(t) => t,
        Err(e) => {
            h.check_bool(&format!("transpose: {e}"), false);
            return;
        }
    };

    let h1_t = match inp_t.matmul(&w1_t_t) {
        Ok(t) => match t.tanh() {
            Ok(a) => a,
            Err(e) => {
                h.check_bool(&format!("tanh: {e}"), false);
                return;
            }
        },
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };
    let out_t = match h1_t.matmul(&w2_t_t) {
        Ok(t) => match t.tanh() {
            Ok(a) => a,
            Err(e) => {
                h.check_bool(&format!("tanh: {e}"), false);
                return;
            }
        },
        Err(e) => {
            h.check_bool(&format!("matmul: {e}"), false);
            return;
        }
    };

    let Some(out) = gpu_readback(h, &out_t) else {
        return;
    };

    h.check_bool(
        "output finite: all NN outputs finite",
        out.iter().all(|x| x.is_finite()),
    );
}

fn validate_determinism(h: &mut ValidationHarness, device: &Arc<barracuda::device::WgpuDevice>) {
    let mut rng = Rng::new(42);
    let n_robots = 10_usize;
    let input_dim = 6_usize;
    let hidden_dim = 8_usize;
    let output_dim = 3_usize;

    let inp: Vec<f32> = (0..n_robots * input_dim)
        .map(|_| rng.uniform() as f32)
        .collect();
    let w1: Vec<f32> = (0..hidden_dim * input_dim)
        .map(|_| rng.uniform() as f32)
        .collect();
    let w2: Vec<f32> = (0..output_dim * hidden_dim)
        .map(|_| rng.uniform() as f32)
        .collect();

    let run = |_: u32| -> Option<Vec<f32>> {
        let i = Tensor::from_data(&inp, vec![n_robots, input_dim], device.clone()).ok()?;
        let w1_t = Tensor::from_data(&w1, vec![hidden_dim, input_dim], device.clone()).ok()?;
        let wt1 = w1_t.transpose().ok()?;
        let h1 = i.matmul(&wt1).ok()?.tanh().ok()?;
        let w2_t = Tensor::from_data(&w2, vec![output_dim, hidden_dim], device.clone()).ok()?;
        let wt2 = w2_t.transpose().ok()?;
        let out = h1.matmul(&wt2).ok()?.tanh().ok()?;
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
