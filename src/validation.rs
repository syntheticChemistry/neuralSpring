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
}
