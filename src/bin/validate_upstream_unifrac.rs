// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: `UniFrac` tree propagation via [`barracuda::ops::bio::UniFracPropagateGpu`].
//!
//! Validates upstream GPU `UniFrac` leaf initialization against CPU reference.
//! Used for wetSpring metagenomics/phylogenetics parity.
//!
//! ## Provenance
//!
//! Upstream: [`barracuda::ops::bio::unifrac_propagate::UniFracPropagateGpu`]

use barracuda::ops::bio::unifrac_propagate::{UniFracConfig, UniFracPropagateGpu};
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

/// CPU reference for leaf initialization: `prop[leaf,s] = sample_mat[leaf,s]`.
/// The shader copies `sample_mat` into `node_sums` for leaf slots (no `branch_len` in `leaf_init`).
fn cpu_leaf_init(sample_mat: &[f64], n_leaves: usize, n_samples: usize) -> Vec<f64> {
    let mut prop = vec![0.0_f64; n_leaves * n_samples];
    for leaf in 0..n_leaves {
        for s in 0..n_samples {
            prop[leaf * n_samples + s] = sample_mat[leaf * n_samples + s];
        }
    }
    prop
}

fn read_buffer_f64(gpu: &Gpu, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f64>, String> {
    let device = gpu.device();
    let size = (count * 8) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    gpu.queue().submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv()
        .map_err(|e| format!("recv: {e}"))?
        .map_err(|e| format!("map: {e:?}"))?;
    let data = slice.get_mapped_range();
    let result: Vec<f64> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(result)
}

fn gpu_leaf_init(
    gpu: &Gpu,
    op: &UniFracPropagateGpu,
    config: &UniFracConfig,
    parent: &[i32],
    branch_len: &[f64],
    sample_mat: &[f64],
) -> Result<Vec<f64>, String> {
    let device = gpu.device();
    let n_nodes = config.n_nodes as usize;
    let n_leaves = config.n_leaves as usize;
    let n_samples = config.n_samples as usize;
    let total_slots = n_nodes * n_samples;

    let parent_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("parent"),
        contents: bytemuck::cast_slice(parent),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let branch_len_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("branch_len"),
        contents: bytemuck::cast_slice(branch_len),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let sample_mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sample_mat"),
        contents: bytemuck::cast_slice(sample_mat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let node_sums_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("node_sums"),
        contents: bytemuck::cast_slice(&vec![0.0_f64; total_slots]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    op.dispatch_leaf_init(
        config,
        &parent_buf,
        &branch_len_buf,
        &sample_mat_buf,
        &node_sums_buf,
    );

    let full = read_buffer_f64(gpu, &node_sums_buf, total_slots)?;
    // Return only leaf portion for comparison
    Ok(full[..n_leaves * n_samples].to_vec())
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

    if !gpu.capabilities.supports_f64 {
        println!("  SKIP: f64 shader support required for UniFracPropagateGpu");
        println!("  0/0 checks — skipping gracefully");
        std::process::exit(0);
    }

    let device = gpu.wgpu_device().clone();
    let op = UniFracPropagateGpu::new(device);

    let mut h = ValidationHarness::new("upstream_unifrac");

    validate_leaf_init(&mut h, &gpu, &op);
    validate_larger_tree(&mut h, &gpu, &op);

    h.finish();
}

fn validate_leaf_init(h: &mut ValidationHarness, gpu: &Gpu, op: &UniFracPropagateGpu) {
    // 3 leaves, 1 internal (root), 2 samples
    let n_nodes = 4_u32;
    let n_leaves = 3_u32;
    let n_samples = 2_u32;
    let config = UniFracConfig {
        n_nodes,
        n_samples,
        n_leaves,
        _pad: 0,
    };

    // parent[i] = parent index; root (index 3) points to self as -1 in some conventions,
    // but barracuda uses i32 and -1 = root. All leaves point to root (index 3).
    let parent: Vec<i32> = vec![3, 3, 3, -1]; // leaves 0,1,2 → node 3; root 3 → -1
    let branch_len: Vec<f64> = vec![0.1, 0.2, 0.3, 0.0];
    // sample_mat: leaf 0 in sample 0, leaf 1 in sample 1, leaf 2 in both
    let sample_mat: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let cpu_prop = cpu_leaf_init(&sample_mat, n_leaves as usize, n_samples as usize);

    match gpu_leaf_init(gpu, op, &config, &parent, &branch_len, &sample_mat) {
        Ok(gpu_prop) => {
            let max_err = cpu_prop
                .iter()
                .zip(gpu_prop.iter())
                .map(|(&c, &g)| (c - g).abs())
                .fold(0.0_f64, f64::max);
            h.check_abs(
                "leaf_init 3 leaves × 2 samples: GPU leaf portion matches CPU",
                max_err,
                0.0,
                tolerances::EXACT_F64,
            );
        }
        Err(e) => {
            h.check_bool(&format!("leaf_init: dispatch failed — {e}"), false);
        }
    }
}

fn validate_larger_tree(h: &mut ValidationHarness, gpu: &Gpu, op: &UniFracPropagateGpu) {
    let n_leaves = 10_u32;
    let n_samples = 5_u32;
    // Minimal tree: 10 leaves + internal nodes. Use a simple star: all leaves → root.
    // n_nodes = n_leaves + 1 for star topology
    let n_nodes = n_leaves + 1;

    let config = UniFracConfig {
        n_nodes,
        n_samples,
        n_leaves,
        _pad: 0,
    };

    #[expect(clippy::cast_possible_wrap, reason = "validation binary")]
    let n_leaves_i32 = n_leaves as i32;
    let mut parent: Vec<i32> = vec![n_leaves_i32; n_leaves as usize];
    parent.push(-1); // root
    let branch_len: Vec<f64> = (0..n_nodes).map(|i| 0.1 * (f64::from(i) + 1.0)).collect();

    let mut rng = Rng::new(77);
    let sample_mat: Vec<f64> = (0..n_leaves as usize * n_samples as usize)
        .map(|_| if rng.uniform() > 0.5 { 1.0 } else { 0.0 })
        .collect();

    let cpu_prop = cpu_leaf_init(&sample_mat, n_leaves as usize, n_samples as usize);

    match gpu_leaf_init(gpu, op, &config, &parent, &branch_len, &sample_mat) {
        Ok(gpu_prop) => {
            let max_err = cpu_prop
                .iter()
                .zip(gpu_prop.iter())
                .map(|(&c, &g)| (c - g).abs())
                .fold(0.0_f64, f64::max);
            h.check_abs(
                "leaf_init 10×5: GPU leaf portion matches CPU",
                max_err,
                0.0,
                tolerances::EXACT_F64,
            );
        }
        Err(e) => {
            h.check_bool(&format!("larger tree: dispatch failed — {e}"), false);
        }
    }
}
