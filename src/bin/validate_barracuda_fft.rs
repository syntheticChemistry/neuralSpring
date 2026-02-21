// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary: `BarraCUDA` FFT (Cooley-Tukey radix-2 via WGSL).
//!
//! Proves that `barracuda::ops::fft` produces correct results on GPU/CPU by
//! checking properties that **any** correct FFT implementation must satisfy:
//!
//! 1. Inverse round-trip: `IFFT(FFT(x)) == x`
//! 2. Parseval's theorem: `||x||² == ||FFT(x)||² / N`
//! 3. Known DFT pairs: delta → constant, constant → delta
//! 4. Cosine energy concentration
//!
//! ## Provenance
//!
//! All expected values are analytical (DFT definition). No Python baseline
//! needed. Reference: Cooley & Tukey (1965), FFTW documentation, NIST DLMF.
//!
//! ## Backend selection
//!
//! Same as `validate_barracuda_tensor`: set `NEURALSPRING_BACKEND=cpu|gpu|auto`.

#![allow(clippy::cast_precision_loss)]

use barracuda::device::WgpuDevice;
use barracuda::ops::fft::{Fft1D, Fft1DF64, Ifft1D, Rfft};
use barracuda::tensor::Tensor;
use neural_spring::fft::{
    complex_energy, complex_energy_f64, constant_signal, constant_signal_f64, cosine_signal,
    cosine_signal_f64, delta_signal, delta_signal_f64, max_abs_diff, max_abs_diff_f64,
};
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;
use std::sync::Arc;

async fn create_device_relaxed(selector: &str) -> barracuda::error::Result<WgpuDevice> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all());

    let adapter = if selector == "cpu" {
        adapters
            .into_iter()
            .find(|a| a.get_info().device_type == wgpu::DeviceType::Cpu)
    } else if let Ok(idx) = selector.parse::<usize>() {
        adapters.into_iter().nth(idx)
    } else {
        let sel = selector.to_ascii_lowercase();
        adapters
            .into_iter()
            .find(|a| a.get_info().name.to_ascii_lowercase().contains(&sel))
    }
    .ok_or_else(|| {
        barracuda::error::BarracudaError::device(format!("No adapter matches '{selector}'"))
    })?;

    let info = adapter.get_info();
    let mut features = wgpu::Features::empty();
    let adapter_features = adapter.features();
    if adapter_features.contains(wgpu::Features::SHADER_F64) {
        features |= wgpu::Features::SHADER_F64;
    }

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("neuralSpring FFT validation (relaxed)"),
                required_features: features,
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )
        .await
        .map_err(|e| barracuda::error::BarracudaError::device(format!("Device creation: {e}")))?;

    Ok(WgpuDevice::from_existing(
        Arc::new(device),
        Arc::new(queue),
        info,
    ))
}

async fn resolve_device() -> Option<(Arc<WgpuDevice>, String)> {
    let selector = std::env::var("NEURALSPRING_BACKEND")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();

    if selector == "list" {
        let adapters = WgpuDevice::enumerate_adapters();
        eprintln!("Available wgpu adapters:");
        for (i, info) in adapters.iter().enumerate() {
            eprintln!(
                "  [{i}] {name} ({ty:?}, {backend:?})",
                name = info.name,
                ty = info.device_type,
                backend = info.backend,
            );
        }
        std::process::exit(0);
    }

    let (result, label) = match selector.as_str() {
        "gpu" => (WgpuDevice::new_gpu().await, "gpu".to_owned()),
        "" | "auto" => (WgpuDevice::new().await, "auto".to_owned()),
        other => {
            let result = create_device_relaxed(other).await;
            (result, other.to_owned())
        }
    };

    match result {
        Ok(dev) => {
            let info = dev.adapter_info();
            eprintln!(
                "  adapter: {} ({:?}, {:?})",
                info.name, info.device_type, info.backend,
            );
            Some((Arc::new(dev), label))
        }
        Err(err) => {
            eprintln!("SKIP [{label}]: {err}");
            None
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Validation checks
// ═════════════════════════════════════════════════════════════════════

#[allow(clippy::expect_used)]
fn validate_inverse_roundtrip(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 16_u32;
    let input_data: Vec<f32> = (0..n * 2).map(|i| ((i as f32) * 0.1).sin()).collect();

    let tensor = Tensor::from_data(&input_data, vec![n as usize, 2], device.clone())
        .expect("FFT input tensor creation");

    let fft = Fft1D::new(tensor, n).expect("FFT creation");
    let spectrum = fft.execute().expect("FFT execution");

    let ifft = Ifft1D::new(spectrum, n).expect("IFFT creation");
    let reconstructed = ifft.execute().expect("IFFT execution");
    let result = reconstructed.to_vec().expect("readback");

    let diff = max_abs_diff(&input_data, &result);
    h.check_upper(
        "FFT inverse round-trip: IFFT(FFT(x)) == x",
        diff,
        tolerances::FFT_INVERSE_F32,
    );
}

#[allow(clippy::expect_used)]
fn validate_parseval(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 32_u32;
    let input_data = cosine_signal(n as usize, 3);
    let time_energy = complex_energy(&input_data);

    let tensor = Tensor::from_data(&input_data, vec![n as usize, 2], device.clone())
        .expect("Parseval input tensor");

    let fft = Fft1D::new(tensor, n).expect("Parseval FFT");
    let spectrum = fft.execute().expect("Parseval FFT execute");
    let spec_data = spectrum.to_vec().expect("Parseval readback");
    let freq_energy = complex_energy(&spec_data) / f64::from(n);

    let ratio = if time_energy > 1e-14 {
        freq_energy / time_energy
    } else {
        1.0
    };

    h.check_abs(
        "Parseval's theorem: ||x||² == ||X||²/N",
        ratio,
        1.0,
        tolerances::FFT_PARSEVAL_F32,
    );
}

#[allow(clippy::expect_used)]
fn validate_delta_to_constant(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 8_u32;
    let input_data = delta_signal(n as usize);

    let tensor = Tensor::from_data(&input_data, vec![n as usize, 2], device.clone())
        .expect("delta input tensor");

    let fft = Fft1D::new(tensor, n).expect("delta FFT");
    let spectrum = fft.execute().expect("delta FFT execute");
    let spec = spectrum.to_vec().expect("delta readback");

    let mut max_real_err = 0.0f64;
    let mut max_imag_err = 0.0f64;
    for k in 0..n as usize {
        max_real_err = max_real_err.max(f64::from((spec[k * 2] - 1.0).abs()));
        max_imag_err = max_imag_err.max(f64::from(spec[k * 2 + 1].abs()));
    }

    h.check_upper(
        "delta → constant: all real parts == 1.0",
        max_real_err,
        tolerances::FFT_KNOWN_PAIR_F32,
    );
    h.check_upper(
        "delta → constant: all imag parts == 0.0",
        max_imag_err,
        tolerances::FFT_KNOWN_PAIR_F32,
    );
}

#[allow(clippy::expect_used)]
fn validate_constant_to_delta(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 8_u32;
    let input_data = constant_signal(n as usize);

    let tensor = Tensor::from_data(&input_data, vec![n as usize, 2], device.clone())
        .expect("constant input tensor");

    let fft = Fft1D::new(tensor, n).expect("constant FFT");
    let spectrum = fft.execute().expect("constant FFT execute");
    let spec = spectrum.to_vec().expect("constant readback");

    h.check_abs(
        "constant → delta: X[0].re == N",
        f64::from(spec[0]),
        f64::from(n),
        tolerances::FFT_KNOWN_PAIR_F32,
    );
    h.check_upper(
        "constant → delta: X[0].im == 0",
        f64::from(spec[1].abs()),
        tolerances::FFT_KNOWN_PAIR_F32,
    );

    let mut off_peak_energy = 0.0f64;
    for k in 1..n as usize {
        off_peak_energy +=
            f64::from(spec[k * 2].mul_add(spec[k * 2], spec[k * 2 + 1] * spec[k * 2 + 1]));
    }

    h.check_upper(
        "constant → delta: off-peak energy ≈ 0",
        off_peak_energy,
        tolerances::FFT_KNOWN_PAIR_F32,
    );
}

#[allow(clippy::expect_used)]
fn validate_cosine_concentration(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 32_u32;
    let freq = 5_usize;
    let input_data = cosine_signal(n as usize, freq);

    let tensor = Tensor::from_data(&input_data, vec![n as usize, 2], device.clone())
        .expect("cosine input tensor");

    let fft = Fft1D::new(tensor, n).expect("cosine FFT");
    let spectrum = fft.execute().expect("cosine FFT execute");
    let spec = spectrum.to_vec().expect("cosine readback");

    let total_energy = complex_energy(&spec);

    let peak_energy = {
        let re_pos = spec[freq * 2];
        let im_pos = spec[freq * 2 + 1];
        let re_neg = spec[(n as usize - freq) * 2];
        let im_neg = spec[(n as usize - freq) * 2 + 1];
        f64::from(re_pos.mul_add(re_pos, im_pos * im_pos))
            + f64::from(re_neg.mul_add(re_neg, im_neg * im_neg))
    };

    let leakage_fraction = if total_energy > 1e-14 {
        1.0 - peak_energy / total_energy
    } else {
        0.0
    };

    h.check_upper(
        "cosine: energy concentrated at freq and N-freq",
        leakage_fraction,
        tolerances::FFT_SPECTRAL_LEAKAGE_F32,
    );
}

#[allow(clippy::expect_used)]
fn validate_larger_roundtrip(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 256_u32;
    let input_data: Vec<f32> = (0..n * 2)
        .map(|i| {
            let t = (i as f32) / (n as f32);
            (2.0 * std::f32::consts::PI * 7.0 * t)
                .cos()
                .mul_add(1.0, 0.5 * (2.0 * std::f32::consts::PI * 23.0 * t).sin())
        })
        .collect();

    let tensor = Tensor::from_data(&input_data, vec![n as usize, 2], device.clone())
        .expect("N=256 input tensor");

    let fft = Fft1D::new(tensor, n).expect("N=256 FFT");
    let spectrum = fft.execute().expect("N=256 FFT execute");

    let ifft = Ifft1D::new(spectrum, n).expect("N=256 IFFT");
    let reconstructed = ifft.execute().expect("N=256 IFFT execute");
    let result = reconstructed.to_vec().expect("N=256 readback");

    let diff = max_abs_diff(&input_data, &result);
    h.check_upper(
        "FFT N=256 round-trip: IFFT(FFT(x)) == x",
        diff,
        tolerances::FFT_INVERSE_F32,
    );
}

#[allow(clippy::expect_used)]
fn validate_multi_frequency(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 64_u32;
    let mut input_data = vec![0.0f32; n as usize * 2];
    let two_pi = 2.0 * std::f64::consts::PI;
    for k in 0..n as usize {
        #[allow(clippy::cast_possible_truncation)]
        let t = f64::from(k as u32) / f64::from(n);
        #[allow(clippy::cast_possible_truncation)]
        {
            input_data[k * 2] = (two_pi * 4.0 * t)
                .cos()
                .mul_add(1.0, 0.5 * (two_pi * 12.0 * t).cos())
                as f32;
        }
    }

    let tensor = Tensor::from_data(&input_data, vec![n as usize, 2], device.clone())
        .expect("multi-freq input tensor");

    let fft = Fft1D::new(tensor, n).expect("multi-freq FFT");
    let spectrum = fft.execute().expect("multi-freq FFT execute");
    let spec = spectrum.to_vec().expect("multi-freq readback");

    let bin4_energy =
        f64::from(spec[4 * 2].mul_add(spec[4 * 2], spec[4 * 2 + 1] * spec[4 * 2 + 1]));
    let bin12_energy =
        f64::from(spec[12 * 2].mul_add(spec[12 * 2], spec[12 * 2 + 1] * spec[12 * 2 + 1]));

    h.check_bool(
        "multi-freq: bin 4 has significant energy",
        bin4_energy > 1.0,
    );
    h.check_bool(
        "multi-freq: bin 12 has significant energy",
        bin12_energy > 0.1,
    );
    h.check_bool(
        "multi-freq: bin 4 > bin 12 (amplitude 1.0 vs 0.5)",
        bin4_energy > bin12_energy,
    );
}

// ═════════════════════════════════════════════════════════════════════
// f64 FFT validation (requires SHADER_F64)
// ═════════════════════════════════════════════════════════════════════

#[allow(clippy::expect_used)]
async fn validate_f64_inverse_roundtrip(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 16_u32;
    let pi = std::f64::consts::PI;
    let mut input_data = vec![0.0f64; (n * 2) as usize];
    for i in 0..n as usize {
        let t = (i as f64) / f64::from(n);
        input_data[i * 2] = 0.5f64.mul_add((4.0 * pi * t).cos(), (2.0 * pi * t).sin());
    }
    let original = input_data.clone();

    let tensor = Tensor::from_f64_data(&input_data, vec![n as usize, 2], device.clone())
        .expect("f64 FFT input tensor");
    let fft = Fft1DF64::new(tensor, n).expect("f64 FFT creation");
    let spectrum = fft.execute().await.expect("f64 FFT execute");

    let spec_data = spectrum.to_f64_vec().expect("f64 spectrum readback");
    let inv_tensor = Tensor::from_f64_data(&spec_data, vec![n as usize, 2], device.clone())
        .expect("f64 IFFT input");
    let ifft = Fft1DF64::new(inv_tensor, n).expect("f64 IFFT creation");
    let recovered_raw = ifft.execute_inverse().await.expect("f64 IFFT execute");
    let recovered = recovered_raw.to_f64_vec().expect("f64 IFFT readback");

    let scaled: Vec<f64> = recovered.iter().map(|&v| v / f64::from(n)).collect();
    let diff = max_abs_diff_f64(&original, &scaled);
    h.check_upper(
        "f64 FFT inverse round-trip: IFFT(FFT(x))/N == x",
        diff,
        tolerances::FFT_INVERSE_F64,
    );
}

#[allow(clippy::expect_used)]
async fn validate_f64_parseval(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 32_u32;
    let input_data = cosine_signal_f64(n as usize, 3);
    let time_energy = complex_energy_f64(&input_data);

    let tensor = Tensor::from_f64_data(&input_data, vec![n as usize, 2], device.clone())
        .expect("f64 Parseval input");
    let fft = Fft1DF64::new(tensor, n).expect("f64 Parseval FFT");
    let spectrum = fft.execute().await.expect("f64 Parseval execute");
    let spec_data = spectrum.to_f64_vec().expect("f64 Parseval readback");
    let freq_energy = complex_energy_f64(&spec_data) / f64::from(n);

    let ratio = if time_energy > 1e-14 {
        freq_energy / time_energy
    } else {
        1.0
    };

    h.check_abs(
        "f64 Parseval: ||x||² == ||X||²/N",
        ratio,
        1.0,
        tolerances::FFT_PARSEVAL_F64,
    );
}

#[allow(clippy::expect_used)]
async fn validate_f64_delta_to_constant(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 8_u32;
    let input_data = delta_signal_f64(n as usize);

    let tensor = Tensor::from_f64_data(&input_data, vec![n as usize, 2], device.clone())
        .expect("f64 delta input");
    let fft = Fft1DF64::new(tensor, n).expect("f64 delta FFT");
    let spectrum = fft.execute().await.expect("f64 delta execute");
    let spec = spectrum.to_f64_vec().expect("f64 delta readback");

    let mut max_real_err = 0.0f64;
    let mut max_imag_err = 0.0f64;
    for k in 0..n as usize {
        max_real_err = max_real_err.max((spec[k * 2] - 1.0).abs());
        max_imag_err = max_imag_err.max(spec[k * 2 + 1].abs());
    }

    h.check_upper(
        "f64 delta → constant: all real == 1.0",
        max_real_err,
        tolerances::FFT_KNOWN_PAIR_F64,
    );
    h.check_upper(
        "f64 delta → constant: all imag == 0.0",
        max_imag_err,
        tolerances::FFT_KNOWN_PAIR_F64,
    );
}

#[allow(clippy::expect_used)]
async fn validate_f64_constant_to_delta(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 8_u32;
    let input_data = constant_signal_f64(n as usize);

    let tensor = Tensor::from_f64_data(&input_data, vec![n as usize, 2], device.clone())
        .expect("f64 constant input");
    let fft = Fft1DF64::new(tensor, n).expect("f64 constant FFT");
    let spectrum = fft.execute().await.expect("f64 constant execute");
    let spec = spectrum.to_f64_vec().expect("f64 constant readback");

    h.check_abs(
        "f64 constant → delta: X[0].re == N",
        spec[0],
        f64::from(n),
        tolerances::FFT_KNOWN_PAIR_F64,
    );
    h.check_upper(
        "f64 constant → delta: X[0].im == 0",
        spec[1].abs(),
        tolerances::FFT_KNOWN_PAIR_F64,
    );

    let mut off_peak_energy = 0.0f64;
    for k in 1..n as usize {
        off_peak_energy += spec[k * 2].mul_add(spec[k * 2], spec[k * 2 + 1] * spec[k * 2 + 1]);
    }

    h.check_upper(
        "f64 constant → delta: off-peak energy ≈ 0",
        off_peak_energy,
        tolerances::FFT_KNOWN_PAIR_F64,
    );
}

#[allow(clippy::expect_used)]
async fn validate_f64_cosine_concentration(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 32_u32;
    let freq = 5_usize;
    let input_data = cosine_signal_f64(n as usize, freq);

    let tensor = Tensor::from_f64_data(&input_data, vec![n as usize, 2], device.clone())
        .expect("f64 cosine input");
    let fft = Fft1DF64::new(tensor, n).expect("f64 cosine FFT");
    let spectrum = fft.execute().await.expect("f64 cosine execute");
    let spec = spectrum.to_f64_vec().expect("f64 cosine readback");

    let total_energy = complex_energy_f64(&spec);
    let peak_energy = {
        let re_pos = spec[freq * 2];
        let im_pos = spec[freq * 2 + 1];
        let re_neg = spec[(n as usize - freq) * 2];
        let im_neg = spec[(n as usize - freq) * 2 + 1];
        re_pos.mul_add(re_pos, im_pos * im_pos) + re_neg.mul_add(re_neg, im_neg * im_neg)
    };

    let leakage_fraction = if total_energy > 1e-14 {
        1.0 - peak_energy / total_energy
    } else {
        0.0
    };

    h.check_upper(
        "f64 cosine: energy concentrated at freq and N-freq",
        leakage_fraction,
        tolerances::FFT_SPECTRAL_LEAKAGE_F64,
    );
}

// ═════════════════════════════════════════════════════════════════════
// Rfft validation (real-to-complex, f32)
// ═════════════════════════════════════════════════════════════════════

#[allow(clippy::expect_used)]
fn validate_rfft_shape(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 16_u32;
    let data: Vec<f32> = (0..n)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * f64::from(k) / f64::from(n);
            #[allow(clippy::cast_possible_truncation)]
            {
                angle.sin() as f32
            }
        })
        .collect();

    let tensor =
        Tensor::from_data(&data, vec![n as usize], device.clone()).expect("Rfft input tensor");
    let rfft = Rfft::new(tensor, n).expect("Rfft creation");
    let spectrum = rfft.execute().expect("Rfft execute");

    let expected_points = (n as usize / 2) + 1;
    h.check_bool(
        &format!("Rfft output shape: [{expected_points}, 2]"),
        spectrum.shape() == [expected_points, 2],
    );
}

#[allow(clippy::expect_used)]
fn validate_rfft_dc_component(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 16_u32;
    let data = vec![1.0f32; n as usize];

    let tensor = Tensor::from_data(&data, vec![n as usize], device.clone()).expect("Rfft DC input");
    let rfft = Rfft::new(tensor, n).expect("Rfft DC creation");
    let spectrum = rfft.execute().expect("Rfft DC execute");
    let spec = spectrum.to_vec().expect("Rfft DC readback");

    h.check_abs(
        "Rfft DC component: X[0].re == N for constant signal",
        f64::from(spec[0]),
        f64::from(n),
        tolerances::RFFT_DC_COMPONENT_F32,
    );

    let mut off_peak_energy = 0.0f64;
    let unique = (n as usize / 2) + 1;
    for k in 1..unique {
        off_peak_energy +=
            f64::from(spec[k * 2].mul_add(spec[k * 2], spec[k * 2 + 1] * spec[k * 2 + 1]));
    }
    h.check_upper(
        "Rfft DC: off-peak energy ≈ 0",
        off_peak_energy,
        tolerances::RFFT_DC_COMPONENT_F32,
    );
}

#[allow(clippy::expect_used, clippy::cast_precision_loss)]
fn validate_rfft_cosine_energy(h: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    let n = 32_u32;
    let freq = 5_usize;
    let data: Vec<f32> = (0..n)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * (freq as f64) * f64::from(k) / f64::from(n);
            #[allow(clippy::cast_possible_truncation)]
            {
                angle.cos() as f32
            }
        })
        .collect();

    let tensor =
        Tensor::from_data(&data, vec![n as usize], device.clone()).expect("Rfft cosine input");
    let rfft = Rfft::new(tensor, n).expect("Rfft cosine creation");
    let spectrum = rfft.execute().expect("Rfft cosine execute");
    let spec = spectrum.to_vec().expect("Rfft cosine readback");

    let bin_energy =
        f64::from(spec[freq * 2].mul_add(spec[freq * 2], spec[freq * 2 + 1] * spec[freq * 2 + 1]));
    h.check_bool(
        &format!("Rfft cosine: bin {freq} has significant energy ({bin_energy:.2})"),
        bin_energy > 1.0,
    );
}

// ═════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() {
    let Some((device, label)) = resolve_device().await else {
        eprintln!("  0/0 checks — skipping gracefully (no GPU/CPU adapter)");
        std::process::exit(0);
    };

    let harness_name = format!("barracuda_fft[{label}]");
    let mut h = ValidationHarness::new(&harness_name);

    // f32 FFT (Fft1D / Ifft1D) — 12 checks
    validate_inverse_roundtrip(&mut h, &device);
    validate_parseval(&mut h, &device);
    validate_delta_to_constant(&mut h, &device);
    validate_constant_to_delta(&mut h, &device);
    validate_cosine_concentration(&mut h, &device);
    validate_larger_roundtrip(&mut h, &device);
    validate_multi_frequency(&mut h, &device);

    // Rfft (real-to-complex, f32) — 4 checks
    validate_rfft_shape(&mut h, &device);
    validate_rfft_dc_component(&mut h, &device);
    validate_rfft_cosine_energy(&mut h, &device);

    // f64 FFT (Fft1DF64) — requires SHADER_F64, 8 checks when available
    if device.has_f64_shaders() {
        eprintln!("  SHADER_F64 available — running f64 FFT validation");
        validate_f64_inverse_roundtrip(&mut h, &device).await;
        validate_f64_parseval(&mut h, &device).await;
        validate_f64_delta_to_constant(&mut h, &device).await;
        validate_f64_constant_to_delta(&mut h, &device).await;
        validate_f64_cosine_concentration(&mut h, &device).await;
    } else {
        eprintln!("  SHADER_F64 not available — skipping f64 FFT tests");
    }

    h.finish();
}
