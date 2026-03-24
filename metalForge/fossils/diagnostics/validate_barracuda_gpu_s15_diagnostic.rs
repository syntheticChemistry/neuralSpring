// SPDX-License-Identifier: AGPL-3.0-or-later

//! S-15 diagnostic v4: test with warmup call first, then binary search fill.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    reason = "fossil diagnostic — GPU memory probe with intentional cast patterns"
)]

use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::validation::ValidationHarness;
use std::time::Instant;

fn make_tridiag(n: usize, fill: f32) -> Vec<f32> {
    let mut data = vec![fill; n * n];
    for i in 0..n {
        data[i * n + i] = 2.0 + (i as f32) * 0.3;
        if i + 1 < n {
            data[i * n + (i + 1)] = 1.0;
            data[(i + 1) * n + i] = 1.0;
        }
    }
    data
}

fn make_dense(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|i| ((i % 7) as f32 + 1.0) * 0.1).collect()
}

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!("  adapter: {} ({:?}, {:?})", g.adapter_name, g.device_type, g.backend);
            g
        }
        Err(e) => {
            eprintln!("  SKIP: {e}");
            std::process::exit(0);
        }
    };
    let device = gpu.wgpu_device().clone();
    let mut h = ValidationHarness::new("s15_diagnostic");

    let n = 64_usize;
    let dense_a = make_dense(n, n);

    // Warmup: dense × dense (Tiled16 tier, should always work)
    eprint!("  warmup dense [64×64]×[64×64] ... ");
    let dense_b = make_dense(n, n);
    let wa = Tensor::from_data(&dense_a, vec![n, n], device.clone()).unwrap();
    let wb = Tensor::from_data(&dense_b, vec![n, n], device.clone()).unwrap();
    let wout = wa.matmul(&wb).unwrap();
    let _ = wout.to_vec().unwrap();
    eprintln!("OK");

    // Now test with tridiagonal data at different fill levels
    for &fill in &[0.5, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 0.0] {
        let label = if fill == 0.0 {
            "fill=0".to_string()
        } else {
            format!("fill={fill}")
        };
        eprint!("  {label} ... ");

        let b = make_tridiag(n, fill);
        let start = Instant::now();

        let a_t = Tensor::from_data(&dense_a, vec![n, n], device.clone()).unwrap();
        let b_t = Tensor::from_data(&b, vec![n, n], device.clone()).unwrap();
        let out_t = match a_t.matmul(&b_t) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("matmul FAIL: {e}");
                h.check_bool(&format!("{label}"), false);
                continue;
            }
        };

        match out_t.to_vec() {
            Ok(v) => {
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                let finite = v.iter().all(|x| x.is_finite());
                eprintln!("OK ({ms:.1}ms, finite={finite})");
                h.check_bool(&format!("{label}: {ms:.1}ms"), finite);
            }
            Err(e) => {
                eprintln!("readback FAIL: {e}");
                h.check_bool(&format!("{label}"), false);
            }
        }
    }

    h.finish();
}
