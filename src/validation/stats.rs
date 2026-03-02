// SPDX-License-Identifier: AGPL-3.0-or-later

//! Slice statistics helpers (shared across validation binaries).

/// Mean of the last `n` elements of a `f64` slice.
///
/// If the slice has fewer than `n` elements, averages the entire slice.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mean_last_n(v: &[f64], n: usize) -> f64 {
    let start = v.len().saturating_sub(n);
    let slice = &v[start..];
    slice.iter().sum::<f64>() / slice.len() as f64
}

/// Mean of the first `n` elements of a `f64` slice.
///
/// If the slice has fewer than `n` elements, averages the entire slice.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mean_first_n(v: &[f64], n: usize) -> f64 {
    let end = n.min(v.len());
    let slice = &v[..end];
    slice.iter().sum::<f64>() / slice.len() as f64
}

/// Mean of the last `n` elements of a `usize` slice, returned as `f64`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mean_last_n_usize(v: &[usize], n: usize) -> f64 {
    let start = v.len().saturating_sub(n);
    let slice = &v[start..];
    slice.iter().sum::<usize>() as f64 / slice.len() as f64
}

/// Population variance of the last `n` elements of a `f64` slice (ddof=0).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn variance_last_n(v: &[f64], n: usize) -> f64 {
    let mean = mean_last_n(v, n);
    let start = v.len().saturating_sub(n);
    let slice = &v[start..];
    slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64
}
