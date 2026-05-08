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

/// Composition validation for NUCLEUS proto-nucleate patterns.
pub mod composition;
/// CPU-side benchmark validation helpers shared by `validate_*` binaries.
pub mod cpu_bench;
mod env;
#[cfg(feature = "barracuda")]
mod gpu;
/// Output sinks for machine-readable validation (JSON, NDJSON, collecting).
pub mod sink;
mod stats;

pub use env::*;
#[cfg(feature = "barracuda")]
pub use gpu::*;
pub use sink::{CollectingSink, JsonSink, NdjsonSink, SilentSink, StdoutSink, ValidationSink};
pub use stats::*;

use crate::tolerances;
use std::borrow::Cow;
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

/// Panic-free unwrap for validation binaries (wetSpring V123 pattern).
///
/// Replaces `.expect()` and `.unwrap()` with clean stderr output and
/// `process::exit(1)` — no stack trace, no panic handler overhead.
/// Use in `main()` setup code where [`require!`] (which returns from
/// the enclosing function) cannot be used.
///
/// ```ignore
/// let gpu = Gpu::new().await.or_exit("GPU init");
/// let rt = Runtime::new().or_exit("tokio runtime");
/// let data = serde_json::from_str(json).or_exit("baseline JSON");
/// ```
pub trait OrExit<T> {
    /// Unwrap the value or log `context` and terminate the process with code 1.
    fn or_exit(self, context: &str) -> T;
}

impl<T, E: std::fmt::Display> OrExit<T> for Result<T, E> {
    fn or_exit(self, context: &str) -> T {
        self.unwrap_or_else(|e| {
            log::error!("FATAL: {context}: {e}");
            process::exit(1)
        })
    }
}

impl<T> OrExit<T> for Option<T> {
    fn or_exit(self, context: &str) -> T {
        self.unwrap_or_else(|| {
            log::error!("FATAL: {context}");
            process::exit(1)
        })
    }
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
///
/// Uses `Cow<'static, str>` for labels: zero-copy for string literals,
/// owned for runtime-formatted labels.
#[derive(Debug, Clone)]
pub struct Check {
    /// Human-readable name or description of this validation check.
    pub label: Cow<'static, str>,
    /// Whether the check passed given `mode`, `tolerance`, and expected value.
    pub passed: bool,
    /// Value measured during validation.
    pub observed: f64,
    /// Reference or threshold value (meaning depends on `mode`).
    pub expected: f64,
    /// Tolerance or bound width used by `mode` (see [`ToleranceMode`]).
    pub tolerance: f64,
    /// How `observed` is compared to `expected` / `tolerance`.
    pub mode: ToleranceMode,
}

/// Accumulates validation checks and produces a summary with exit code.
#[derive(Debug, Default)]
pub struct ValidationHarness {
    /// Display name of the validation binary or suite (printed in summaries).
    pub name: String,
    /// Recorded checks in submission order.
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
            label: Cow::Owned(label.to_owned()),
            passed,
            observed,
            expected,
            tolerance,
            mode: ToleranceMode::Absolute,
        });
    }

    /// Relative tolerance check: |observed - expected| / |expected| < tolerance
    pub fn check_rel(&mut self, label: &str, observed: f64, expected: f64, tolerance: f64) {
        let passed = if expected.abs() > tolerances::ZERO_DETECTION {
            ((observed - expected) / expected).abs() < tolerance
        } else {
            observed.abs() < tolerance
        };
        self.checks.push(Check {
            label: Cow::Owned(label.to_owned()),
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
        let abs_pass = abs_err < tolerance;
        let rel_pass = rel_err < tolerance;
        let passed = abs_pass || rel_pass;
        let mode = if abs_pass {
            ToleranceMode::Absolute
        } else {
            ToleranceMode::Relative
        };
        self.checks.push(Check {
            label: Cow::Owned(label.to_owned()),
            passed,
            observed,
            expected,
            tolerance,
            mode,
        });
    }

    /// Upper-bound check: observed < threshold
    pub fn check_upper(&mut self, label: &str, observed: f64, threshold: f64) {
        self.checks.push(Check {
            label: Cow::Owned(label.to_owned()),
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
            label: Cow::Owned(label.to_owned()),
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
            label: Cow::Owned(label.to_owned()),
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

    /// Count of checks that passed so far.
    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.checks.iter().filter(|c| c.passed).count()
    }

    /// Total number of checks recorded in this harness.
    #[must_use]
    pub const fn total_count(&self) -> usize {
        self.checks.len()
    }

    /// Returns true when every recorded check passed (or when there are zero checks).
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    /// Emit all check results to a [`ValidationSink`], then emit the finish event.
    ///
    /// Does **not** exit the process — call [`Self::finish`] afterward if you
    /// need the standard exit behavior, or use the return value for CI.
    pub fn emit_to_sink(&self, sink: &mut dyn ValidationSink) {
        for check in &self.checks {
            sink.on_check(check);
        }
        sink.on_finish(
            &self.name,
            self.passed_count(),
            self.total_count(),
            self.all_passed(),
        );
    }

    /// Emit a JSON report of all checks to the given writer.
    ///
    /// Convenience wrapper around [`JsonSink`]. Call before [`Self::finish`].
    pub fn emit_json<W: std::io::Write>(&self, writer: W) {
        let mut json = JsonSink::new(writer);
        json.emit(&self.name, &self.checks);
    }

    /// Print summary and exit with appropriate code.
    pub fn finish(&self) -> ! {
        let mut stdout_sink = StdoutSink;
        log::info!("");
        self.emit_to_sink(&mut stdout_sink);

        if self.all_passed() {
            process::exit(0);
        } else {
            let failed: Vec<&str> = self
                .checks
                .iter()
                .filter(|c| !c.passed)
                .map(|c| c.label.as_ref())
                .collect();
            log::error!("FAILED: {}", failed.join(", "));
            process::exit(1);
        }
    }
}

/// Expected value and tolerance for a scalar reduction check.
pub struct ReductionExpected<'a> {
    /// Label identifying this reduction in logs and reports.
    pub label: &'a str,
    /// Expected scalar value from the reference implementation.
    pub value: f64,
    /// Maximum allowed absolute deviation from `value`.
    pub tolerance: f64,
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

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test assertions on known-populated test fixtures"
)]
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

    #[test]
    fn tolerance_mode_display() {
        assert_eq!(format!("{}", ToleranceMode::Absolute), "abs");
        assert_eq!(format!("{}", ToleranceMode::Relative), "rel");
        assert_eq!(format!("{}", ToleranceMode::UpperBound), "<");
        assert_eq!(format!("{}", ToleranceMode::LowerBound), ">");
    }

    #[test]
    fn check_abs_or_rel_abs_wins() {
        let mut h = ValidationHarness::new("test");
        h.check_abs_or_rel("abs_wins", 0.0001, 0.0, 0.001);
        assert!(h.checks[0].passed, "abs diff < tolerance → pass");
    }

    #[test]
    fn emit_to_collecting_sink() {
        let mut h = ValidationHarness::new("sink_test");
        h.check_abs("ok", 1.0, 1.0, 1e-10);
        h.check_abs("fail", 2.0, 1.0, 1e-10);
        let mut sink = CollectingSink::new();
        h.emit_to_sink(&mut sink);
        assert_eq!(sink.checks.len(), 2);
        assert!(sink.checks[0].passed);
        assert!(!sink.checks[1].passed);
        let r = sink.result.as_ref().unwrap();
        assert_eq!(r.passed, 1);
        assert_eq!(r.total, 2);
        assert!(!r.success);
    }

    #[test]
    fn emit_json_produces_output() {
        let mut h = ValidationHarness::new("json_test");
        h.check_abs("ok", 1.0, 1.0, 1e-10);
        let mut buf = Vec::new();
        h.emit_json(&mut buf);
        let json = String::from_utf8(buf).unwrap();
        assert!(json.contains("\"suite\":\"json_test\""));
        assert!(json.contains("\"all_passed\":true"));
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

    #[test]
    fn check_abs_exact_boundary() {
        let mut h = ValidationHarness::new("test");
        h.check_abs("at_boundary", 1.0 + 1e-10, 1.0, 1e-10);
        assert!(
            !h.checks[0].passed,
            "exactly at tolerance → fail (strict <)"
        );
    }

    #[test]
    fn check_abs_or_rel_rel_path_only() {
        let mut h = ValidationHarness::new("test");
        h.check_abs_or_rel("rel_only", 1000.5, 1000.0, 1e-3);
        assert!(h.checks[0].passed, "0.5/1000 = 5e-4 < 1e-3 → pass via rel");
    }
}
