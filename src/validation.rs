// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation harness for neuralSpring binaries.
//!
//! Imitates the hotSpring pattern:
//!   - Hardcoded expected values with provenance
//!   - Explicit pass/fail checks against documented tolerances
//!   - Exit code 0 (all checks pass) or 1 (any check fails)
//!   - Machine-readable summary on stdout
//!
//! This module provides the shared infrastructure. Every validation
//! binary (`validate_*`) uses [`ValidationHarness`] to accumulate
//! checks and produce a deterministic exit code.

use crate::tolerances;
use std::path::PathBuf;
use std::process;

/// Unwrap a `Result` or record failure and early-return from the caller.
///
/// Replaces `.expect()` in validation binaries with graceful failure
/// recording. On error, records a FAIL check in the harness and returns
/// from the enclosing function.
///
/// # Usage
///
/// ```ignore
/// let tensor = require!(harness, Tensor::from_data(&data, shape, dev), "create tensor");
/// ```
#[macro_export]
macro_rules! require {
    ($harness:expr, $result:expr, $label:expr) => {
        match $result {
            Ok(v) => v,
            Err(e) => {
                $harness.check_bool(&format!("{}: {}", $label, e), false);
                return;
            }
        }
    };
}

/// How a tolerance threshold is applied.
#[derive(Debug, Clone, Copy)]
pub enum ToleranceMode {
    /// |observed - expected| < tolerance
    Absolute,
    /// |observed - expected| / |expected| < tolerance
    Relative,
    /// observed < threshold (upper bound only)
    UpperBound,
    /// observed > threshold (lower bound only)
    LowerBound,
}

impl std::fmt::Display for ToleranceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Absolute => write!(f, "abs"),
            Self::Relative => write!(f, "rel"),
            Self::UpperBound => write!(f, "<"),
            Self::LowerBound => write!(f, ">"),
        }
    }
}

/// A single validation check with result tracking.
#[derive(Debug, Clone)]
pub struct Check {
    pub label: String,
    pub passed: bool,
    pub observed: f64,
    pub expected: f64,
    pub tolerance: f64,
    pub mode: ToleranceMode,
}

/// Accumulates validation checks and produces a summary with exit code.
#[derive(Debug, Default)]
pub struct ValidationHarness {
    pub name: String,
    pub checks: Vec<Check>,
}

impl ValidationHarness {
    /// Create a new harness for a named validation binary.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            checks: Vec::new(),
        }
    }

    /// Absolute tolerance check: |observed - expected| < tolerance
    pub fn check_abs(&mut self, label: &str, observed: f64, expected: f64, tolerance: f64) {
        let passed = (observed - expected).abs() < tolerance;
        self.checks.push(Check {
            label: label.to_string(),
            passed,
            observed,
            expected,
            tolerance,
            mode: ToleranceMode::Absolute,
        });
    }

    /// Relative tolerance check: |observed - expected| / |expected| < tolerance
    pub fn check_rel(&mut self, label: &str, observed: f64, expected: f64, tolerance: f64) {
        let passed = if expected.abs() > f64::EPSILON {
            ((observed - expected) / expected).abs() < tolerance
        } else {
            observed.abs() < tolerance
        };
        self.checks.push(Check {
            label: label.to_string(),
            passed,
            observed,
            expected,
            tolerance,
            mode: ToleranceMode::Relative,
        });
    }

    /// Combined absolute-or-relative check (matches hotSpring convention).
    pub fn check_abs_or_rel(&mut self, label: &str, observed: f64, expected: f64, tolerance: f64) {
        let abs_err = (observed - expected).abs();
        let rel_err = if expected.abs() > tolerances::ZERO_DETECTION {
            abs_err / expected.abs()
        } else {
            abs_err
        };
        let passed = abs_err < tolerance || rel_err < tolerance;
        self.checks.push(Check {
            label: label.to_string(),
            passed,
            observed,
            expected,
            tolerance,
            mode: ToleranceMode::Absolute,
        });
    }

    /// Upper-bound check: observed < threshold
    pub fn check_upper(&mut self, label: &str, observed: f64, threshold: f64) {
        self.checks.push(Check {
            label: label.to_string(),
            passed: observed < threshold,
            observed,
            expected: threshold,
            tolerance: threshold,
            mode: ToleranceMode::UpperBound,
        });
    }

    /// Lower-bound check: observed > threshold
    pub fn check_lower(&mut self, label: &str, observed: f64, threshold: f64) {
        self.checks.push(Check {
            label: label.to_string(),
            passed: observed > threshold,
            observed,
            expected: threshold,
            tolerance: threshold,
            mode: ToleranceMode::LowerBound,
        });
    }

    /// Boolean pass/fail check.
    pub fn check_bool(&mut self, label: &str, passed: bool) {
        self.checks.push(Check {
            label: label.to_string(),
            passed,
            observed: f64::from(u8::from(passed)),
            expected: 1.0,
            tolerance: 0.0,
            mode: ToleranceMode::Absolute,
        });
    }

    /// Try to unwrap a `Result`, recording a FAIL check if it errors.
    ///
    /// Returns `Some(value)` on success, `None` on failure (after recording
    /// the error in the harness). Callers should early-return on `None`.
    ///
    /// This replaces `.expect()` in validation binaries — GPU/tensor
    /// operations that fail are recorded as check failures rather than
    /// panicking, so the harness can continue and report all failures.
    pub fn require<T, E: std::fmt::Display>(
        &mut self,
        label: &str,
        result: Result<T, E>,
    ) -> Option<T> {
        match result {
            Ok(v) => Some(v),
            Err(e) => {
                self.check_bool(&format!("{label}: {e}"), false);
                None
            }
        }
    }

    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.checks.iter().filter(|c| c.passed).count()
    }

    #[must_use]
    pub const fn total_count(&self) -> usize {
        self.checks.len()
    }

    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    /// Print summary and exit with appropriate code.
    pub fn finish(&self) -> ! {
        println!();
        for check in &self.checks {
            let icon = if check.passed { "PASS" } else { "FAIL" };
            println!(
                "  [{icon}] {}: observed={:.10e}, expected={:.10e}, tol={:.2e} ({})",
                check.label, check.observed, check.expected, check.tolerance, check.mode
            );
        }

        println!();
        println!(
            "=== {}: {}/{} PASS, {} FAIL ===",
            self.name,
            self.passed_count(),
            self.total_count(),
            self.total_count() - self.passed_count(),
        );

        if self.all_passed() {
            process::exit(0);
        } else {
            let failed: Vec<&str> = self
                .checks
                .iter()
                .filter(|c| !c.passed)
                .map(|c| c.label.as_str())
                .collect();
            println!("FAILED: {}", failed.join(", "));
            process::exit(1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Environment-driven runtime policy
// ═══════════════════════════════════════════════════════════════════

/// Whether `NEURALSPRING_REQUIRE_GPU=1` is set.
///
/// When `true`, validation binaries that cannot obtain a GPU adapter
/// **must** exit 1 instead of silently skipping.  This is intended for
/// CI pipelines that have a known-good GPU and want to catch adapter
/// regressions.
///
/// Default behaviour (variable unset or `0`): binaries skip gracefully
/// with exit 0 when no GPU is available, which is appropriate for
/// headless / CPU-only build environments.
#[must_use]
pub fn gpu_required() -> bool {
    std::env::var("NEURALSPRING_REQUIRE_GPU")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Handle the absence of a GPU adapter in a validation binary.
///
/// If `NEURALSPRING_REQUIRE_GPU=1`, prints an error and exits 1.
/// Otherwise, prints a skip message and exits 0.
///
/// Replaces the duplicated `let Ok(gpu) = Gpu::new().await else { … }`
/// pattern across all GPU validation binaries.
pub fn exit_no_gpu() -> ! {
    if gpu_required() {
        eprintln!("  FAIL: no GPU adapter (NEURALSPRING_REQUIRE_GPU=1)");
        process::exit(1);
    }
    eprintln!("  0/0 checks — skipping gracefully (no GPU adapter)");
    process::exit(0);
}

// ═══════════════════════════════════════════════════════════════════
// Slice statistics helpers (shared across validation binaries)
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
// Data path resolution
// ═══════════════════════════════════════════════════════════════════

/// Resolve a workspace-relative path to an absolute path.
///
/// Uses `CARGO_MANIFEST_DIR` at compile time — this is the standard Rust
/// mechanism for finding workspace resources and avoids runtime path
/// assumptions.
///
/// # Example
///
/// ```ignore
/// let p = baseline_path("control/ml_inference/mlp_baseline.json");
/// let file = std::fs::File::open(&p)?;
/// ```
#[must_use]
pub fn baseline_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

// ═══════════════════════════════════════════════════════════════════
// WGSL shader helpers
// ═══════════════════════════════════════════════════════════════════

/// Replace native `pow(` with `pow_f64(` in WGSL shader source.
///
/// Works around NVVM/NAK failure on `pow(f64, f64)` — the injected
/// `pow_f64` polyfill uses `exp_f64(exponent * log_f64(base))` instead.
/// Preserves comments (lines starting with `//` are left untouched;
/// inline `// …` suffixes are preserved).
///
/// Consolidates 4 formerly-duplicated copies across validation binaries.
/// Upstream `BarraCUDA` S59 now exposes `GpuDriverProfile::needs_pow_f64_workaround()`.
#[must_use]
pub fn patch_pow_to_polyfill(shader: &str) -> String {
    shader
        .lines()
        .map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                return line.to_string();
            }
            line.find("//").map_or_else(
                || line.replace("pow(", "pow_f64("),
                |pos| {
                    let code = &line[..pos];
                    let comment = &line[pos..];
                    format!("{}{comment}", code.replace("pow(", "pow_f64("))
                },
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ═══════════════════════════════════════════════════════════════════
// GPU tensor validation helpers (shared across 24+ validation binaries)
// ═══════════════════════════════════════════════════════════════════

/// Attempt GPU tensor readback, recording a FAIL check on error.
///
/// Returns `Some(data)` on success, `None` on failure.
/// The caller should early-return on `None`.
pub fn gpu_readback(
    h: &mut ValidationHarness,
    tensor: &barracuda::tensor::Tensor,
) -> Option<Vec<f32>> {
    match tensor.to_vec() {
        Ok(data) => Some(data),
        Err(e) => {
            h.check_bool(&format!("GPU readback: {e}"), false);
            None
        }
    }
}

/// Compute the maximum absolute difference between two f32 slices.
///
/// Used by GPU validation binaries to compare GPU output against
/// GPU output (both f32).
#[must_use]
pub fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0.0_f64, f64::max)
}

/// Compute the maximum absolute difference between GPU f32 output
/// and CPU f64 reference values.
///
/// The promotion to f64 before comparison avoids f32 rounding
/// masking real errors.
#[must_use]
pub fn max_abs_diff_gpu_vs_cpu(gpu: &[f32], cpu: &[f64]) -> f64 {
    gpu.iter()
        .zip(cpu.iter())
        .map(|(&g, &c)| (f64::from(g) - c).abs())
        .fold(0.0_f64, f64::max)
}

/// Batch-check readback values against expected (label, index, expected, tolerance).
///
/// Shared helper that replaces `check_points` duplicated across tensor
/// validation binaries.
pub fn check_gpu_points(
    h: &mut ValidationHarness,
    data: &[f32],
    checks: &[(&str, usize, f64, f64)],
) {
    for &(label, idx, expected, tol) in checks {
        h.check_abs(label, f64::from(data[idx]), expected, tol);
    }
}

/// Create a tensor from data, recording a FAIL check on error.
///
/// Returns `Some(tensor)` on success, `None` on failure.
pub fn gpu_tensor(
    h: &mut ValidationHarness,
    data: &[f32],
    shape: &[usize],
    device: &std::sync::Arc<barracuda::device::WgpuDevice>,
) -> Option<barracuda::tensor::Tensor> {
    match barracuda::tensor::Tensor::from_data(data, shape.to_vec(), device.clone()) {
        Ok(t) => Some(t),
        Err(e) => {
            h.check_bool(&format!("tensor create: {e}"), false);
            None
        }
    }
}

/// Create a GPU tensor, or record FAIL and return from the enclosing function.
///
/// This macro replaces the `tensor!` macro duplicated across 14+ GPU
/// validation binaries.
///
/// # Usage
///
/// ```ignore
/// let t = gpu_tensor!(harness, &data, &[rows, cols], device);
/// ```
#[macro_export]
macro_rules! gpu_tensor {
    ($harness:expr, $data:expr, $shape:expr, $device:expr) => {
        match barracuda::tensor::Tensor::from_data($data, $shape.to_vec(), $device.clone()) {
            Ok(t) => t,
            Err(e) => {
                $harness.check_bool(&format!("tensor create: {}", e), false);
                return;
            }
        }
    };
}

/// Validate a unary tensor operation against expected point checks.
///
/// Handles the common pattern:
///   create tensor → apply op → readback → `check_gpu_points`
/// with graceful error recording on failure at any step.
pub fn validate_tensor_unary(
    h: &mut ValidationHarness,
    device: &std::sync::Arc<barracuda::device::WgpuDevice>,
    data: &[f32],
    shape: &[usize],
    op: impl FnOnce(
        &barracuda::tensor::Tensor,
    ) -> Result<barracuda::tensor::Tensor, barracuda::error::BarracudaError>,
    op_name: &str,
    checks: &[(&str, usize, f64, f64)],
) {
    let Some(input) = gpu_tensor(h, data, shape, device) else {
        return;
    };
    match op(&input) {
        Ok(out) => {
            let Some(v) = gpu_readback(h, &out) else {
                return;
            };
            check_gpu_points(h, &v, checks);
        }
        Err(e) => h.check_bool(&format!("{op_name} [ERROR: {e}]"), false),
    }
}

/// Validate a scalar reduction (sum, mean, max, etc.) against an expected value.
///
/// Handles: create tensor → apply reduction op → readback scalar → `check_abs`.
#[allow(clippy::too_many_arguments)]
pub fn validate_tensor_reduction(
    h: &mut ValidationHarness,
    device: &std::sync::Arc<barracuda::device::WgpuDevice>,
    data: &[f32],
    shape: &[usize],
    op: impl FnOnce(
        &barracuda::tensor::Tensor,
    ) -> Result<barracuda::tensor::Tensor, barracuda::error::BarracudaError>,
    label: &str,
    expected: f64,
    tolerance: f64,
) {
    let Some(input) = gpu_tensor(h, data, shape, device) else {
        return;
    };
    match op(&input) {
        Ok(out) => {
            let Some(v) = gpu_readback(h, &out) else {
                return;
            };
            h.check_abs(label, f64::from(v[0]), expected, tolerance);
        }
        Err(e) => h.check_bool(&format!("{label} [ERROR: {e}]"), false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harness_tracks_pass_fail() {
        let mut h = ValidationHarness::new("test");
        h.check_abs("exact", 1.0, 1.0, 1e-10);
        h.check_abs("close", 1.0001, 1.0, 1e-3);
        h.check_abs("far", 2.0, 1.0, 1e-3);
        assert_eq!(h.passed_count(), 2);
        assert_eq!(h.total_count(), 3);
        assert!(!h.all_passed());
    }

    #[test]
    fn harness_all_pass() {
        let mut h = ValidationHarness::new("test");
        h.check_abs("a", 1.0, 1.0, 1e-10);
        h.check_upper("b", 0.5, 1.0);
        h.check_bool("c", true);
        assert!(h.all_passed());
    }

    #[test]
    fn relative_check_handles_zero() {
        let mut h = ValidationHarness::new("test");
        h.check_rel("near_zero", 1e-15, 0.0, 1e-10);
        assert!(h.checks[0].passed);
    }

    #[test]
    fn check_bool_false() {
        let mut h = ValidationHarness::new("test");
        h.check_bool("fail", false);
        assert!(!h.checks[0].passed);
    }

    #[test]
    fn harness_zero_checks() {
        let h = ValidationHarness::new("empty");
        assert_eq!(h.passed_count(), 0);
        assert_eq!(h.total_count(), 0);
        assert!(h.all_passed());
    }

    #[test]
    fn check_lower_pass_and_fail() {
        let mut h = ValidationHarness::new("test");
        h.check_lower("above", 5.0, 3.0);
        assert!(h.checks[0].passed);
        h.check_lower("below", 1.0, 3.0);
        assert!(!h.checks[1].passed);
    }

    #[test]
    fn check_upper_pass_and_fail() {
        let mut h = ValidationHarness::new("test");
        h.check_upper("below", 1.0, 3.0);
        assert!(h.checks[0].passed);
        h.check_upper("above", 5.0, 3.0);
        assert!(!h.checks[1].passed);
    }

    #[test]
    fn check_abs_or_rel_near_zero() {
        let mut h = ValidationHarness::new("test");
        h.check_abs_or_rel("near_zero_pass", 1e-16, 0.0, 1e-10);
        assert!(h.checks[0].passed);
        h.check_abs_or_rel("near_zero_fail", 1.0, 0.0, 1e-10);
        assert!(!h.checks[1].passed);
    }

    #[test]
    fn check_abs_or_rel_relative_mode() {
        let mut h = ValidationHarness::new("test");
        h.check_abs_or_rel("rel_pass", 100.000_001, 100.0, 1e-6);
        assert!(h.checks[0].passed);
    }

    #[test]
    fn check_rel_pass_and_fail() {
        let mut h = ValidationHarness::new("test");
        h.check_rel("close", 1.001, 1.0, 0.01);
        assert!(h.checks[0].passed);
        h.check_rel("far", 2.0, 1.0, 0.01);
        assert!(!h.checks[1].passed);
    }

    #[test]
    fn mixed_pass_fail_counts() {
        let mut h = ValidationHarness::new("test");
        h.check_bool("ok", true);
        h.check_bool("fail", false);
        h.check_abs("ok2", 1.0, 1.0, 1e-10);
        assert_eq!(h.passed_count(), 2);
        assert_eq!(h.total_count(), 3);
        assert!(!h.all_passed());
    }

    // ── require method ──────────────────────────────────────────

    #[test]
    fn require_ok_returns_value() {
        let mut h = ValidationHarness::new("test");
        let val: Option<i32> = h.require("op", Ok::<i32, String>(42));
        assert_eq!(val, Some(42));
        assert_eq!(h.total_count(), 0, "success records no check");
    }

    #[test]
    fn require_err_records_fail() {
        let mut h = ValidationHarness::new("test");
        let val: Option<i32> = h.require("op", Err::<i32, &str>("boom"));
        assert_eq!(val, None);
        assert_eq!(h.total_count(), 1);
        assert!(!h.checks[0].passed);
        assert!(h.checks[0].label.contains("boom"));
    }

    // ── f32 diff helpers ────────────────────────────────────────

    #[test]
    fn max_abs_diff_f32_exact() {
        assert!((max_abs_diff_f32(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn max_abs_diff_f32_nonzero() {
        let diff = max_abs_diff_f32(&[1.0_f32, 5.0], &[1.0, 2.0]);
        assert!((diff - 3.0).abs() < 1e-6);
    }

    #[test]
    fn max_abs_diff_gpu_vs_cpu_promotion() {
        let gpu = vec![1.0_f32, 2.0];
        let cpu = vec![1.0_f64, 2.001];
        let diff = max_abs_diff_gpu_vs_cpu(&gpu, &cpu);
        assert!((diff - 0.001).abs() < 1e-5);
    }

    // ── check_gpu_points ────────────────────────────────────────

    #[test]
    fn check_gpu_points_pass_and_fail() {
        let mut h = ValidationHarness::new("test");
        let data = vec![1.0_f32, 2.5, 3.0];
        let checks: Vec<(&str, usize, f64, f64)> = vec![
            ("val0", 0, 1.0, 0.1),
            ("val1", 1, 2.5, 0.01),
            ("val2", 2, 999.0, 0.01),
        ];
        check_gpu_points(&mut h, &data, &checks);
        assert_eq!(h.passed_count(), 2);
        assert_eq!(h.total_count(), 3);
    }

    // ── ToleranceMode display ───────────────────────────────────

    #[test]
    fn tolerance_mode_display() {
        assert_eq!(format!("{}", ToleranceMode::Absolute), "abs");
        assert_eq!(format!("{}", ToleranceMode::Relative), "rel");
        assert_eq!(format!("{}", ToleranceMode::UpperBound), "<");
        assert_eq!(format!("{}", ToleranceMode::LowerBound), ">");
    }

    // ── Edge cases ──────────────────────────────────────────────

    #[test]
    fn check_abs_or_rel_abs_wins() {
        let mut h = ValidationHarness::new("test");
        h.check_abs_or_rel("abs_wins", 0.0001, 0.0, 0.001);
        assert!(h.checks[0].passed, "abs diff < tolerance → pass");
    }

    #[test]
    fn check_rel_large_expected() {
        let mut h = ValidationHarness::new("test");
        h.check_rel("large", 1_000_001.0, 1_000_000.0, 1e-4);
        assert!(h.checks[0].passed, "1 part per million < 1e-4 rel");
    }

    #[test]
    fn harness_name_preserved() {
        let h = ValidationHarness::new("my_validation_binary");
        assert_eq!(h.name, "my_validation_binary");
    }

    // ── baseline_path resolution ─────────────────────────────────

    #[test]
    fn baseline_path_is_absolute() {
        let p = baseline_path("control/ml_inference/mlp_baseline.json");
        assert!(p.is_absolute(), "baseline_path should return absolute path");
        assert!(
            p.ends_with("control/ml_inference/mlp_baseline.json"),
            "path should end with relative component"
        );
    }

    #[test]
    fn baseline_path_different_inputs() {
        let a = baseline_path("a.json");
        let b = baseline_path("b.json");
        assert_ne!(a, b, "different inputs → different paths");
        assert_eq!(a.parent(), b.parent(), "same parent directory");
    }

    // ── gpu_required env detection ───────────────────────────────

    #[test]
    fn gpu_required_respects_env() {
        let original = std::env::var("NEURALSPRING_REQUIRE_GPU").ok();
        std::env::set_var("NEURALSPRING_REQUIRE_GPU", "0");
        assert!(!gpu_required(), "0 → false");

        std::env::set_var("NEURALSPRING_REQUIRE_GPU", "1");
        assert!(gpu_required(), "1 → true");

        std::env::set_var("NEURALSPRING_REQUIRE_GPU", "true");
        assert!(gpu_required(), "true → true");

        std::env::set_var("NEURALSPRING_REQUIRE_GPU", "TRUE");
        assert!(gpu_required(), "TRUE → true");

        std::env::remove_var("NEURALSPRING_REQUIRE_GPU");
        assert!(!gpu_required(), "unset → false");

        if let Some(v) = original {
            std::env::set_var("NEURALSPRING_REQUIRE_GPU", v);
        }
    }

    // ── require method edge cases ────────────────────────────────

    #[test]
    fn require_err_message_includes_error() {
        let mut h = ValidationHarness::new("test");
        let _: Option<i32> = h.require("create tensor", Err::<i32, &str>("device lost"));
        assert!(
            h.checks[0].label.contains("create tensor"),
            "label should contain operation name"
        );
        assert!(
            h.checks[0].label.contains("device lost"),
            "label should contain error message"
        );
    }

    #[test]
    fn require_multiple_failures_accumulated() {
        let mut h = ValidationHarness::new("test");
        let _: Option<i32> = h.require("op1", Err::<i32, &str>("e1"));
        let _: Option<i32> = h.require("op2", Err::<i32, &str>("e2"));
        let _: Option<i32> = h.require("op3", Ok::<i32, &str>(1));
        assert_eq!(h.total_count(), 2, "only failures recorded");
        assert_eq!(h.passed_count(), 0);
    }

    // ── max_abs_diff edge cases ──────────────────────────────────

    #[test]
    fn max_abs_diff_f32_empty() {
        let diff = max_abs_diff_f32(&[], &[]);
        assert!((diff - 0.0).abs() < 1e-15, "empty → 0");
    }

    #[test]
    fn max_abs_diff_gpu_vs_cpu_empty() {
        let diff = max_abs_diff_gpu_vs_cpu(&[], &[]);
        assert!((diff - 0.0).abs() < 1e-15, "empty → 0");
    }

    #[test]
    fn max_abs_diff_gpu_vs_cpu_precision() {
        let gpu = vec![0.1_f32];
        let cpu = vec![0.1_f64];
        let diff = max_abs_diff_gpu_vs_cpu(&gpu, &cpu);
        assert!(
            diff < 1e-6,
            "f32 0.1 should be close to f64 0.1, got {diff}"
        );
    }

    // ── check_abs boundary ───────────────────────────────────────

    #[test]
    fn check_abs_exact_boundary() {
        let mut h = ValidationHarness::new("test");
        h.check_abs("at_boundary", 1.0 + 1e-10, 1.0, 1e-10);
        assert!(
            !h.checks[0].passed,
            "exactly at tolerance → fail (strict <)"
        );
    }

    // ── check_abs_or_rel rel path ────────────────────────────────

    #[test]
    fn check_abs_or_rel_rel_path_only() {
        let mut h = ValidationHarness::new("test");
        h.check_abs_or_rel("rel_only", 1000.5, 1000.0, 1e-3);
        assert!(h.checks[0].passed, "0.5/1000 = 5e-4 < 1e-3 → pass via rel");
    }

    // ── patch_pow_to_polyfill ─────────────────────────────────────

    #[test]
    fn patch_pow_basic_replacement() {
        let input = "let x = pow(a, b);";
        let out = patch_pow_to_polyfill(input);
        assert_eq!(out, "let x = pow_f64(a, b);");
    }

    #[test]
    fn patch_pow_preserves_comment_lines() {
        let input = "// pow(a, b) should NOT be replaced";
        let out = patch_pow_to_polyfill(input);
        assert_eq!(out, input);
    }

    #[test]
    fn patch_pow_inline_comment_preserved() {
        let input = "let x = pow(a, 2.0); // compute a^2 with pow(x,n)";
        let out = patch_pow_to_polyfill(input);
        assert_eq!(out, "let x = pow_f64(a, 2.0); // compute a^2 with pow(x,n)");
    }

    #[test]
    fn patch_pow_multiline() {
        let input = "let a = pow(x, y);\n// comment\nlet b = pow(z, w);";
        let out = patch_pow_to_polyfill(input);
        assert_eq!(
            out,
            "let a = pow_f64(x, y);\n// comment\nlet b = pow_f64(z, w);"
        );
    }

    #[test]
    fn patch_pow_no_pow_unchanged() {
        let input = "let x = sin(a) + cos(b);";
        let out = patch_pow_to_polyfill(input);
        assert_eq!(out, input);
    }

    #[test]
    fn patch_pow_indented_comment() {
        let input = "    // pow(x, y) in a comment";
        let out = patch_pow_to_polyfill(input);
        assert_eq!(out, input, "indented comment lines preserved");
    }
}
