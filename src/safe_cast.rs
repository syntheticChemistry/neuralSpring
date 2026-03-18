// SPDX-License-Identifier: AGPL-3.0-or-later

//! Checked numeric conversions (groundSpring V112 pattern).
//!
//! Replaces bare `as` casts with functions that either return `Result` or
//! panic with a meaningful message. GPU dispatch parameters (usize → u32)
//! and buffer size calculations (usize → u64) are the primary consumers.

/// `usize` → `u32`, returning an error on overflow.
///
/// GPU dispatch parameters (workgroup counts, dimension sizes) must fit in
/// `u32`. This makes the check explicit rather than silently truncating.
///
/// # Errors
///
/// Returns an error string if the value exceeds `u32::MAX`.
pub fn usize_u32(value: usize, label: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{label}: {value} exceeds u32::MAX"))
}

/// `usize` → `u64`, infallible on 64-bit platforms.
///
/// On 32-bit platforms `usize` ≤ `u64` so this is always safe, but using
/// a named function documents intent and avoids bare `as` casts.
#[must_use]
pub const fn usize_u64(value: usize) -> u64 {
    value as u64
}

/// `usize` → `f64`, lossy for values > 2^53.
///
/// Scientific computing routinely converts counts to floats for
/// normalization / averaging. Values above 2^53 lose precision.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "intentional: scientific counts fit well within f64 mantissa"
)]
pub const fn usize_f64(value: usize) -> f64 {
    value as f64
}

/// `f64` → `f32`, intentionally lossy for GPU shader inputs.
///
/// Most WGSL shaders operate on f32. This wrapper documents the
/// intentional precision loss and avoids lint noise from bare casts.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "intentional: GPU shaders require f32"
)]
pub const fn f64_f32(value: f64) -> f32 {
    value as f32
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test inputs are small known constants — unwrap documents expected success"
)]
mod tests {
    use super::*;

    #[test]
    fn usize_u32_in_range() {
        assert_eq!(usize_u32(42, "n").unwrap(), 42);
        assert_eq!(usize_u32(0, "zero").unwrap(), 0);
        assert_eq!(usize_u32(u32::MAX as usize, "max").unwrap(), u32::MAX);
    }

    #[test]
    fn usize_u32_overflow() {
        let big = u32::MAX as usize + 1;
        assert!(usize_u32(big, "too_big").is_err());
    }

    #[test]
    fn usize_u64_identity() {
        assert_eq!(usize_u64(0), 0);
        assert_eq!(usize_u64(1_000_000), 1_000_000);
    }

    #[test]
    fn usize_f64_small_exact() {
        assert!((usize_f64(42) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn f64_f32_basic() {
        let v: f32 = f64_f32(1.234_567_8);
        assert!((f64::from(v) - 1.234_567_8).abs() < 1e-5);
    }
}
