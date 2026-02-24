// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU validation: k-mer histogram via `barracuda::ops::bio::KmerHistogramGpu`.
//!
//! Validates upstream GPU k-mer histogram against CPU reference.
//! Used for wetSpring metagenomics parity.
//!
//! ## Provenance
//!
//! Upstream: `barracuda::ops::bio::kmer_histogram::KmerHistogramGpu`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use barracuda::ops::bio::kmer_histogram::KmerHistogramGpu;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;
use neural_spring::validation::ValidationHarness;
use wgpu::util::DeviceExt;

fn cpu_kmer_histogram(kmers: &[u32], k: u32) -> Vec<u32> {
    let bins = 4_u32.pow(k) as usize;
    let mut hist = vec![0u32; bins];
    for &kmer in kmers {
        if (kmer as usize) < bins {
            hist[kmer as usize] += 1;
        }
    }
    hist
}

fn read_buffer_u32(gpu: &Gpu, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<u32>, String> {
    let device = gpu.device();
    let size = (count * 4) as u64;
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
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("recv: {e}"))?
        .map_err(|e| format!("map: {e:?}"))?;
    let data = slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(result)
}

fn gpu_kmer_histogram(
    gpu: &Gpu,
    op: &KmerHistogramGpu,
    kmers: &[u32],
    k: u32,
) -> Result<Vec<u32>, String> {
    let device = gpu.device();
    let bins = 4_usize.pow(k);

    let kmers_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("kmers"),
        contents: bytemuck::cast_slice(kmers),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let histogram_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("histogram"),
        contents: bytemuck::cast_slice(&vec![0u32; bins]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    op.dispatch(&kmers_buf, &histogram_buf, kmers.len() as u32, k);

    read_buffer_u32(gpu, &histogram_buf, bins)
}

#[tokio::main]
async fn main() {
    let gpu = match Gpu::new().await {
        Ok(g) => {
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                g.adapter_name, g.device_type, g.backend
            );
            g
        }
        Err(_) => neural_spring::validation::exit_no_gpu(),
    };

    let device = gpu.wgpu_device().clone();
    let op = KmerHistogramGpu::new(device);

    let mut h = ValidationHarness::new("upstream_kmer");

    validate_small_histogram(&mut h, &gpu, &op);
    validate_known_counts(&mut h, &gpu, &op);
    validate_determinism(&mut h, &gpu, &op);

    h.finish();
}

fn validate_small_histogram(h: &mut ValidationHarness, gpu: &Gpu, op: &KmerHistogramGpu) {
    let k = 2_u32;
    let bins = 4_usize.pow(k);
    let mut rng = Rng::new(42);
    let kmers: Vec<u32> = (0..100)
        .map(|_| (rng.uniform() * bins as f64).round() as u32)
        .collect();

    let cpu_hist = cpu_kmer_histogram(&kmers, k);

    match gpu_kmer_histogram(gpu, op, &kmers, k) {
        Ok(gpu_hist) => {
            let exact_match = cpu_hist.iter().zip(gpu_hist.iter()).all(|(c, g)| *c == *g);
            h.check_bool(
                "small histogram k=2: GPU vs CPU u32 histograms match exactly",
                exact_match,
            );
        }
        Err(e) => {
            h.check_bool(&format!("small histogram: dispatch failed — {e}"), false);
        }
    }
}

fn validate_known_counts(h: &mut ValidationHarness, gpu: &Gpu, op: &KmerHistogramGpu) {
    let k = 1_u32;
    let kmers: Vec<u32> = vec![0, 0, 0, 1, 1, 2, 3, 3, 3, 3];
    let expected: Vec<u32> = vec![3, 2, 1, 4];

    match gpu_kmer_histogram(gpu, op, &kmers, k) {
        Ok(gpu_hist) => {
            let exact_match = expected.iter().zip(gpu_hist.iter()).all(|(e, g)| *e == *g);
            h.check_bool(
                "known counts k=1: [0,0,0,1,1,2,3,3,3,3] → [3,2,1,4]",
                exact_match,
            );
        }
        Err(e) => {
            h.check_bool(&format!("known counts: dispatch failed — {e}"), false);
        }
    }
}

fn validate_determinism(h: &mut ValidationHarness, gpu: &Gpu, op: &KmerHistogramGpu) {
    let k = 2_u32;
    let mut rng = Rng::new(123);
    let kmers: Vec<u32> = (0..50)
        .map(|_| (rng.uniform() * 16.0).round() as u32)
        .collect();

    let r1 = gpu_kmer_histogram(gpu, op, &kmers, k);
    let r2 = gpu_kmer_histogram(gpu, op, &kmers, k);

    match (r1, r2) {
        (Ok(s1), Ok(s2)) => {
            let identical = s1.iter().zip(s2.iter()).all(|(a, b)| *a == *b);
            h.check_bool("determinism: two runs identical", identical);
        }
        _ => {
            h.check_bool("determinism: dispatch failed", false);
        }
    }
}
