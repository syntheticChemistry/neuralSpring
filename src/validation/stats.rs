// SPDX-License-Identifier: AGPL-3.0-or-later

//! Slice statistics helpers (shared across validation binaries).

/// Mean of the last `n` elements of a `f64` slice.
///
/// If the slice has fewer than `n` elements, averages the entire slice.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "slice length → f64 for mean computation"
)]
pub fn mean_last_n(v: &[f64], n: usize) -> f64 {
    let start = v.len().saturating_sub(n);
    let slice = &v[start..];
    slice.iter().sum::<f64>() / slice.len() as f64
}

/// Mean of the first `n` elements of a `f64` slice.
///
/// If the slice has fewer than `n` elements, averages the entire slice.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "slice length → f64 for mean computation"
)]
pub fn mean_first_n(v: &[f64], n: usize) -> f64 {
    let end = n.min(v.len());
    let slice = &v[..end];
    slice.iter().sum::<f64>() / slice.len() as f64
}

/// Mean of the last `n` elements of a `usize` slice, returned as `f64`.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "usize sum and length → f64 for mean computation"
)]
pub fn mean_last_n_usize(v: &[usize], n: usize) -> f64 {
    let start = v.len().saturating_sub(n);
    let slice = &v[start..];
    slice.iter().sum::<usize>() as f64 / slice.len() as f64
}

/// Population variance of the last `n` elements of a `f64` slice (ddof=0).
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "slice length → f64 for variance denominator"
)]
pub fn variance_last_n(v: &[f64], n: usize) -> f64 {
    let mean = mean_last_n(v, n);
    let start = v.len().saturating_sub(n);
    let slice = &v[start..];
    slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64
}

/// Run a closure once and return its result alongside elapsed microseconds.
///
/// Replaces the `bench(label, f) -> (T, f64)` helper duplicated across
/// 10+ validation/benchmark binaries.
pub fn bench_once<F: FnOnce() -> T, T>(label: &str, f: F) -> (T, f64) {
    let start = std::time::Instant::now();
    let result = f();
    let us = start.elapsed().as_secs_f64() * 1e6;
    log::info!("  {label}: {us:.1} µs");
    (result, us)
}

/// Run a closure multiple times after warm-up, returning the median elapsed
/// microseconds.
///
/// Replaces `bench_rust` / `bench` helpers duplicated across 8+ benchmark
/// binaries.
pub fn bench_median<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = std::time::Instant::now();
        f();
        times.push(start.elapsed());
    }
    median_duration_us(&mut times)
}

/// Compute the median of a `Duration` slice, returned in microseconds.
///
/// Replaces `median` / `median_us` helpers duplicated across 6–8
/// benchmark binaries. Sorts the slice in place.
pub fn median_duration_us(times: &mut [std::time::Duration]) -> f64 {
    times.sort();
    let n = times.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        times[n / 2].as_secs_f64() * 1e6
    } else {
        (times[n / 2 - 1].as_secs_f64() + times[n / 2].as_secs_f64()) * 0.5e6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_last_n_basic() {
        // Analytical: mean([4,5]) = 4.5
        assert!((mean_last_n(&[1.0, 2.0, 3.0, 4.0, 5.0], 2) - 4.5).abs() < 1e-14);
    }

    #[test]
    fn mean_last_n_exceeds_length() {
        // Analytical: mean([1,2,3]) = 2.0
        assert!((mean_last_n(&[1.0, 2.0, 3.0], 100) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn mean_first_n_basic() {
        // Analytical: mean([1,2]) = 1.5
        assert!((mean_first_n(&[1.0, 2.0, 3.0, 4.0], 2) - 1.5).abs() < 1e-14);
    }

    #[test]
    fn mean_first_n_exceeds_length() {
        // Analytical: mean([10,20]) = 15.0
        assert!((mean_first_n(&[10.0, 20.0], 100) - 15.0).abs() < 1e-14);
    }

    #[test]
    fn mean_last_n_usize_basic() {
        // Analytical: mean([8,10]) = 9.0
        assert!((mean_last_n_usize(&[2, 4, 6, 8, 10], 2) - 9.0).abs() < 1e-14);
    }

    #[test]
    fn mean_last_n_usize_all() {
        // Analytical: mean([1,2,3]) = 2.0
        assert!((mean_last_n_usize(&[1, 2, 3], 10) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn variance_last_n_basic() {
        // Analytical: var([4,6]) = ((4-5)^2 + (6-5)^2)/2 = 1.0
        assert!((variance_last_n(&[1.0, 2.0, 4.0, 6.0], 2) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn variance_last_n_constant() {
        // Analytical: var([3,3,3]) = 0
        assert!((variance_last_n(&[3.0, 3.0, 3.0], 3) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn bench_once_returns_value() {
        let (val, us) = bench_once("test_add", || 2 + 3);
        assert_eq!(val, 5);
        assert!(us >= 0.0);
    }

    #[test]
    fn bench_median_basic() {
        let us = bench_median(1, 5, || {
            std::hint::black_box(42);
        });
        assert!(us >= 0.0);
    }

    #[test]
    fn median_duration_us_odd() {
        use std::time::Duration;
        let mut times = vec![
            Duration::from_micros(10),
            Duration::from_micros(30),
            Duration::from_micros(20),
        ];
        let med = median_duration_us(&mut times);
        assert!((med - 20.0).abs() < 1.0);
    }

    #[test]
    fn median_duration_us_even() {
        use std::time::Duration;
        let mut times = vec![
            Duration::from_micros(10),
            Duration::from_micros(30),
            Duration::from_micros(20),
            Duration::from_micros(40),
        ];
        let med = median_duration_us(&mut times);
        assert!((med - 25.0).abs() < 1.0);
    }

    #[test]
    fn median_duration_us_empty() {
        let mut times = vec![];
        assert!((median_duration_us(&mut times) - 0.0).abs() < 1e-15);
    }
}
