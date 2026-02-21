// SPDX-License-Identifier: AGPL-3.0-or-later

//! FFT validation helpers for `BarraCUDA`'s Cooley-Tukey radix-2 implementation.
//!
//! Provides analytical reference values and numerical checks that any correct
//! FFT must satisfy, independent of implementation details:
//!
//! - **Parseval's theorem**: `||x||² == ||FFT(x)||² / N`
//! - **Inverse round-trip**: `IFFT(FFT(x)) == x`
//! - **Known DFT pairs**: delta → constant, constant → delta, sine → impulse pair
//! - **Linearity**: `FFT(a·x + b·y) == a·FFT(x) + b·FFT(y)`
//!
//! These checks validate the GPU shader without depending on any specific
//! Python baseline — the reference values are analytical (IEEE 754 exact or
//! NIST DLMF).

/// Compute the squared magnitude of a complex vector stored as interleaved
/// `[re0, im0, re1, im1, ...]`.
///
/// # Panics
///
/// Panics if `data.len()` is odd (interleaved complex requires even length).
#[must_use]
pub fn complex_energy(data: &[f32]) -> f64 {
    assert!(
        data.len().is_multiple_of(2),
        "interleaved complex: even length required"
    );
    data.chunks_exact(2)
        .map(|c| f64::from(c[0]).mul_add(f64::from(c[0]), f64::from(c[1]) * f64::from(c[1])))
        .sum()
}

/// Compute the squared magnitude of a complex f64 vector stored as interleaved
/// `[re0, im0, re1, im1, ...]`.
///
/// # Panics
///
/// Panics if `data.len()` is odd.
#[must_use]
pub fn complex_energy_f64(data: &[f64]) -> f64 {
    assert!(
        data.len().is_multiple_of(2),
        "interleaved complex: even length required"
    );
    data.chunks_exact(2)
        .map(|c| c[0].mul_add(c[0], c[1] * c[1]))
        .sum()
}

/// Build a complex delta signal: `x[0] = (1, 0)`, rest zero.
///
/// FFT of a delta should be a constant: `X[k] = (1, 0)` for all k.
#[must_use]
pub fn delta_signal(n: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; n * 2];
    data[0] = 1.0;
    data
}

/// Build a constant complex signal: `x[k] = (1, 0)` for all k.
///
/// FFT of a constant should be a delta scaled by N: `X[0] = (N, 0)`, rest zero.
#[must_use]
pub fn constant_signal(n: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; n * 2];
    for k in 0..n {
        data[k * 2] = 1.0;
    }
    data
}

/// Build a pure cosine signal: `x[k] = cos(2π·freq·k/N)` (real part only).
///
/// FFT should have energy concentrated at bins `freq` and `N - freq`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn cosine_signal(n: usize, freq: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; n * 2];
    let two_pi = 2.0 * std::f64::consts::PI;
    for k in 0..n {
        let angle = two_pi * (freq as f64) * (k as f64) / (n as f64);
        #[allow(clippy::cast_possible_truncation)]
        {
            data[k * 2] = angle.cos() as f32;
        }
    }
    data
}

/// Max absolute element-wise difference between two f32 slices.
///
/// # Panics
///
/// Panics if the slices have different lengths.
#[must_use]
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| f64::from((x - y).abs()))
        .fold(0.0f64, f64::max)
}

/// Max absolute element-wise difference between two f64 slices.
///
/// # Panics
///
/// Panics if the slices have different lengths.
#[must_use]
pub fn max_abs_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// Build a complex delta signal (f64): `x[0] = (1, 0)`, rest zero.
#[must_use]
pub fn delta_signal_f64(n: usize) -> Vec<f64> {
    let mut data = vec![0.0f64; n * 2];
    data[0] = 1.0;
    data
}

/// Build a constant complex signal (f64): `x[k] = (1, 0)` for all k.
#[must_use]
pub fn constant_signal_f64(n: usize) -> Vec<f64> {
    let mut data = vec![0.0f64; n * 2];
    for k in 0..n {
        data[k * 2] = 1.0;
    }
    data
}

/// Build a pure cosine signal (f64): `x[k] = cos(2π·freq·k/N)` (real part only).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn cosine_signal_f64(n: usize, freq: usize) -> Vec<f64> {
    let mut data = vec![0.0f64; n * 2];
    let two_pi = 2.0 * std::f64::consts::PI;
    for k in 0..n {
        let angle = two_pi * (freq as f64) * (k as f64) / (n as f64);
        data[k * 2] = angle.cos();
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_signal_structure() {
        let d = delta_signal(8);
        assert_eq!(d.len(), 16);
        assert!((d[0] - 1.0).abs() < f32::EPSILON);
        assert!(d[1..].iter().all(|&x| x.abs() < f32::EPSILON));
    }

    #[test]
    fn constant_signal_structure() {
        let c = constant_signal(4);
        assert_eq!(c.len(), 8);
        for k in 0..4 {
            assert!((c[k * 2] - 1.0).abs() < f32::EPSILON);
            assert!(c[k * 2 + 1].abs() < f32::EPSILON);
        }
    }

    #[test]
    fn cosine_signal_energy() {
        let c = cosine_signal(8, 1);
        let e = complex_energy(&c);
        assert!((e - 4.0).abs() < 1e-6, "cosine energy should be N/2 = 4.0");
    }

    #[test]
    fn delta_energy_is_one() {
        let d = delta_signal(16);
        let e = complex_energy(&d);
        assert!((e - 1.0).abs() < 1e-12);
    }

    #[test]
    fn max_abs_diff_identical() {
        let a = vec![1.0f32, 2.0, 3.0];
        assert!(max_abs_diff(&a, &a) < 1e-12);
    }
}
