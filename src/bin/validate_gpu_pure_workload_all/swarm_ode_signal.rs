// SPDX-License-Identifier: AGPL-3.0-or-later

//! Swarm NN (paper 015), RK45 regulatory ODE (020), and Hill gate signal integration (021).

use barracuda::ops::bio::SwarmNnGpu;
use barracuda::ops::bio::hill_gate::{HillGateGpu, HillGateParams};
use barracuda::ops::bio::swarm_nn::SwarmNnParams;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::signal_integration::two_input_hill;
use neural_spring::swarm_robotics::{ControllerType, create_controller, neural_forward};
use neural_spring::tolerances;
use neural_spring::validation::{ValidationHarness, output_buf, storage_buf};
use std::sync::Arc;

pub fn validate_swarm_nn(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_ctrl = 4_u32;
    let n_eval = 8_u32;
    let input_dim = 1_u32;
    let hidden_dim = 4_u32;
    let output_dim = 5_u32;

    let mut rng = Rng::new(55);
    let controllers: Vec<_> = (0..n_ctrl)
        .map(|_| create_controller(ControllerType::NeuralNet, &mut rng))
        .collect();

    let all_weights: Vec<f64> = controllers
        .iter()
        .flat_map(|c| c.params.iter().copied())
        .collect();
    let sense: Vec<f64> = (0..n_eval).map(|i| (i as f64) * 0.1).collect();

    let cpu_actions: Vec<u32> = controllers
        .iter()
        .flat_map(|c| (0..n_eval).map(|i| neural_forward(&c.params, (i as f64) * 0.1) as u32))
        .collect();
    let cpu_mean = cpu_actions.iter().sum::<u32>() as f64 / cpu_actions.len() as f64;

    let op = SwarmNnGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let inputs_f64: Vec<f64> = (0..n_ctrl).flat_map(|_| sense.iter().copied()).collect();
    let w_buf = storage_buf(device, "swarm_w", bytemuck::cast_slice(&all_weights));
    let in_buf = storage_buf(device, "swarm_in", bytemuck::cast_slice(&inputs_f64));
    let n_actions = (n_ctrl * n_eval) as usize;
    let act_buf = output_buf(device, "swarm_act", (n_actions * 4) as u64);

    op.dispatch(
        &w_buf,
        &in_buf,
        &act_buf,
        &SwarmNnParams {
            n_controllers: n_ctrl,
            n_evals: n_eval,
            input_dim,
            hidden_dim,
            output_dim,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        },
    );

    match gpu.read_buffer_u32(&act_buf, n_actions) {
        Ok(gpu_actions) => {
            let gpu_mean = gpu_actions.iter().sum::<u32>() as f64 / gpu_actions.len() as f64;
            h.check_bool(
                &format!(
                    "swarm_nn {n_ctrl}×{n_eval}: GPU mean action={gpu_mean:.2} (CPU={cpu_mean:.2}), all in [0,{output_dim})"
                ),
                gpu_actions.iter().all(|&a| a < output_dim),
            );
        }
        Err(e) => h.check_bool(&format!("swarm_nn: {e}"), false),
    }
}

pub fn validate_rk45_regulatory(h: &mut ValidationHarness, gpu: &Gpu) {
    use barracuda::ops::rk45_adaptive::Rk45AdaptiveGpu;

    let dim = 4_u32;
    let n_systems = 4_u32;
    let n_coeffs = dim * 3;
    let dt = 0.01_f64;

    let state: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4]
        .into_iter()
        .cycle()
        .take((dim * n_systems) as usize)
        .collect();
    let coeffs: Vec<f64> = (0..n_systems)
        .flat_map(|_| (0..dim).flat_map(|d| vec![1.0, 0.5, ((d + 1) % dim) as f64]))
        .collect();

    let total = (dim * n_systems) as usize;
    let device = gpu.device();

    let op = Rk45AdaptiveGpu::new(Arc::clone(gpu.wgpu_device()));
    let state_buf = storage_buf(device, "rk_state", bytemuck::cast_slice(&state));
    let coeff_buf = storage_buf(device, "rk_coeff", bytemuck::cast_slice(&coeffs));
    let out_buf = output_buf(device, "rk_out", (total * 8) as u64);
    let err_buf = output_buf(device, "rk_err", (total * 8) as u64);
    let scratch_buf = output_buf(device, "rk_scratch", (total * 8 * 8) as u64);

    op.dispatch(&barracuda::ops::rk45_adaptive::Rk45DispatchArgs {
        buffers: barracuda::ops::rk45_adaptive::Rk45Buffers {
            state_buf: &state_buf,
            coeffs_buf: &coeff_buf,
            new_state_buf: &out_buf,
            error_buf: &err_buf,
            scratch_buf: &scratch_buf,
        },
        params: barracuda::ops::rk45_adaptive::Rk45DispatchParams {
            n_systems,
            dim,
            n_coeffs,
            dt,
        },
    });

    match gpu.read_buffer_f64(&out_buf, total) {
        Ok(gpu_v) => {
            let gpu_mean = gpu_v.iter().sum::<f64>() / gpu_v.len() as f64;
            h.check_bool(
                &format!("rk45 {n_systems}×{dim}: GPU mean={gpu_mean:.6}, all finite"),
                gpu_v.iter().all(|v| v.is_finite()) && gpu_mean > 0.0,
            );
        }
        Err(e) => h.check_bool(&format!("rk45: {e}"), false),
    }
}

pub fn validate_hill_gate_signal(h: &mut ValidationHarness, gpu: &Gpu) {
    let n_a = 8_u32;
    let n_b = 8_u32;
    let n_out = (n_a * n_b) as usize;

    let a_vals: Vec<f64> = (0..n_a).map(|i| (i as f64) * 0.15).collect();
    let b_vals: Vec<f64> = (0..n_b).map(|i| (i as f64) * 0.12 + 0.05).collect();

    let cpu_mean = {
        let mut sum = 0.0_f64;
        for &a in &a_vals {
            for &b in &b_vals {
                sum += two_input_hill(a, b, 1.0, 0.5, 0.5, 2.0, 2.0);
            }
        }
        sum / n_out as f64
    };

    let op = HillGateGpu::new(Arc::clone(gpu.wgpu_device()));
    let device = gpu.device();
    let a_buf = storage_buf(device, "hill_a", bytemuck::cast_slice(&a_vals));
    let b_buf = storage_buf(device, "hill_b", bytemuck::cast_slice(&b_vals));
    let out_buf = output_buf(device, "hill_out", (n_out * 8) as u64);

    let params = HillGateParams {
        n_a,
        n_b,
        mode: 1,
        _pad: 0,
        k_a: 0.5,
        k_b: 0.5,
        n_a_exp: 2.0,
        n_b_exp: 2.0,
        vmax: 1.0,
        _pad2: 0.0,
    };
    op.dispatch(&a_buf, &b_buf, &out_buf, &params);

    match gpu.read_buffer_f64(&out_buf, n_out) {
        Ok(gpu_v) => {
            let gpu_mean = gpu_v.iter().sum::<f64>() / gpu_v.len() as f64;
            h.check_abs(
                &format!("hill_gate 8×8 grid: GPU={gpu_mean:.6} vs CPU={cpu_mean:.6}"),
                gpu_mean,
                cpu_mean,
                tolerances::GPU_HILL_GATE_F64,
            );
        }
        Err(e) => h.check_bool(&format!("hill_gate: {e}"), false),
    }
}
