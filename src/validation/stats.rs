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
}
