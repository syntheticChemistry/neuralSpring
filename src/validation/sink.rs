// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation output sinks for machine-readable CI integration.
//!
//! Three implementations:
//! - [`StdoutSink`] — human-readable `[PASS]`/`[FAIL]` lines (default)
//! - [`JsonSink`] — single JSON object emitted on finish (for CI parsers)
//! - [`NdjsonSink`] — one JSON line per check (streaming, for pipelines)
//!
//! Absorbed from ecosystem convergence: wetSpring V134, airSpring V010,
//! groundSpring V121.

use super::Check;
use std::io::Write;

/// Callback interface for validation check results.
///
/// Implementations receive structured events as checks are recorded,
/// enabling machine-readable output without changing binary logic.
pub trait ValidationSink: Send {
    /// A single check completed.
    fn on_check(&mut self, check: &Check);

    /// Validation run finished.
    fn on_finish(&mut self, name: &str, passed: usize, total: usize, success: bool);
}

/// Human-readable sink writing `[PASS]`/`[FAIL]` lines via `log::info!`.
///
/// This is the default behavior matching the original `finish()` output.
pub struct StdoutSink;

impl ValidationSink for StdoutSink {
    fn on_check(&mut self, check: &Check) {
        let icon = if check.passed { "PASS" } else { "FAIL" };
        log::info!(
            "  [{icon}] {}: observed={:.10e}, expected={:.10e}, tol={:.2e} ({})",
            check.label,
            check.observed,
            check.expected,
            check.tolerance,
            check.mode,
        );
    }

    fn on_finish(&mut self, name: &str, passed: usize, total: usize, _success: bool) {
        log::info!("");
        log::info!(
            "=== {name}: {passed}/{total} PASS, {} FAIL ===",
            total - passed
        );
    }
}

/// Silent sink that discards all output. Used in tests.
pub struct SilentSink;

impl ValidationSink for SilentSink {
    fn on_check(&mut self, _check: &Check) {}
    fn on_finish(&mut self, _name: &str, _passed: usize, _total: usize, _success: bool) {}
}

/// Collects checks for programmatic inspection in tests.
pub struct CollectingSink {
    /// All checks received.
    pub checks: Vec<CheckRecord>,
    /// Final result, if `on_finish` was called.
    pub result: Option<FinishRecord>,
}

/// Record of a single check as received by [`CollectingSink`].
pub struct CheckRecord {
    /// Check label.
    pub label: String,
    /// Whether the check passed.
    pub passed: bool,
    /// Tolerance mode used.
    pub mode: String,
}

/// Record of the finish event as received by [`CollectingSink`].
pub struct FinishRecord {
    /// Suite name.
    pub name: String,
    /// Number of passing checks.
    pub passed: usize,
    /// Total number of checks.
    pub total: usize,
    /// Whether the overall run succeeded.
    pub success: bool,
}

impl CollectingSink {
    /// Create an empty collecting sink.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            checks: Vec::new(),
            result: None,
        }
    }
}

impl Default for CollectingSink {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationSink for CollectingSink {
    fn on_check(&mut self, check: &Check) {
        self.checks.push(CheckRecord {
            label: check.label.to_string(),
            passed: check.passed,
            mode: check.mode.to_string(),
        });
    }

    fn on_finish(&mut self, name: &str, passed: usize, total: usize, success: bool) {
        self.result = Some(FinishRecord {
            name: name.to_string(),
            passed,
            total,
            success,
        });
    }
}

/// Emits a single JSON object on finish with all check results.
///
/// Call [`JsonSink::emit`] after all checks are recorded to write
/// the JSON report. Compatible with CI JSON artifact collection.
pub struct JsonSink<W: Write> {
    writer: W,
    pretty: bool,
}

impl JsonSink<std::io::Stdout> {
    /// Create a `JsonSink` writing to stdout.
    #[must_use]
    pub fn stdout() -> Self {
        Self {
            writer: std::io::stdout(),
            pretty: false,
        }
    }

    /// Create a pretty-printing `JsonSink` writing to stdout.
    #[must_use]
    pub fn stdout_pretty() -> Self {
        Self {
            writer: std::io::stdout(),
            pretty: true,
        }
    }
}

impl<W: Write> JsonSink<W> {
    /// Create a `JsonSink` writing to an arbitrary writer.
    pub const fn new(writer: W) -> Self {
        Self {
            writer,
            pretty: false,
        }
    }

    /// Emit the full JSON report for a completed harness.
    pub fn emit(&mut self, name: &str, checks: &[Check]) {
        let passed = checks.iter().filter(|c| c.passed).count();
        let total = checks.len();
        let all_passed = checks.iter().all(|c| c.passed);

        let checks_json: Vec<String> = checks
            .iter()
            .map(|c| {
                format!(
                    r#"{{"label":"{}","passed":{},"observed":{:.15e},"expected":{:.15e},"tolerance":{:.6e},"mode":"{}"}}"#,
                    escape_json(&c.label),
                    c.passed,
                    c.observed,
                    c.expected,
                    c.tolerance,
                    c.mode,
                )
            })
            .collect();

        let json = if self.pretty {
            let checks_str = checks_json.join(",\n    ");
            format!(
                "{{\n  \"suite\": \"{}\",\n  \"passed\": {passed},\n  \"total\": {total},\n  \"all_passed\": {all_passed},\n  \"checks\": [\n    {checks_str}\n  ]\n}}",
                escape_json(name),
            )
        } else {
            let checks_str = checks_json.join(",");
            format!(
                r#"{{"suite":"{}","passed":{passed},"total":{total},"all_passed":{all_passed},"checks":[{checks_str}]}}"#,
                escape_json(name),
            )
        };

        let _ = writeln!(self.writer, "{json}");
    }
}

impl<W: Write + Send> ValidationSink for JsonSink<W> {
    fn on_check(&mut self, _check: &Check) {}

    fn on_finish(&mut self, _name: &str, _passed: usize, _total: usize, _success: bool) {}
}

/// Emits one NDJSON line per check as it happens (streaming).
///
/// Each line is a self-contained JSON object. Suitable for pipeline
/// consumption and cross-spring aggregation.
pub struct NdjsonSink<W: Write> {
    writer: W,
}

impl<W: Write> NdjsonSink<W> {
    /// Create an NDJSON sink writing to the given writer.
    pub const fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl NdjsonSink<std::io::Stdout> {
    /// Create an NDJSON sink writing to stdout.
    #[must_use]
    pub fn stdout() -> Self {
        Self {
            writer: std::io::stdout(),
        }
    }
}

impl<W: Write + Send> ValidationSink for NdjsonSink<W> {
    fn on_check(&mut self, check: &Check) {
        let status = if check.passed { "pass" } else { "fail" };
        let _ = writeln!(
            self.writer,
            r#"{{"type":"check","status":"{status}","label":"{}","observed":{:.15e},"expected":{:.15e},"tolerance":{:.6e},"mode":"{}"}}"#,
            escape_json(&check.label),
            check.observed,
            check.expected,
            check.tolerance,
            check.mode,
        );
    }

    fn on_finish(&mut self, name: &str, passed: usize, total: usize, success: bool) {
        let _ = writeln!(
            self.writer,
            r#"{{"type":"summary","suite":"{}","passed":{passed},"total":{total},"success":{success}}}"#,
            escape_json(name),
        );
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::ToleranceMode;
    use std::borrow::Cow;

    fn sample_check(label: &str, passed: bool) -> Check {
        Check {
            label: Cow::Owned(label.to_string()),
            passed,
            observed: 1.0,
            expected: 1.0,
            tolerance: 1e-10,
            mode: ToleranceMode::Absolute,
        }
    }

    #[test]
    fn collecting_sink_records_checks() {
        let mut sink = CollectingSink::new();
        sink.on_check(&sample_check("a", true));
        sink.on_check(&sample_check("b", false));
        assert_eq!(sink.checks.len(), 2);
        assert!(sink.checks[0].passed);
        assert!(!sink.checks[1].passed);
    }

    #[test]
    fn collecting_sink_records_finish() {
        let mut sink = CollectingSink::new();
        sink.on_finish("suite", 5, 6, false);
        let r = sink.result.as_ref().unwrap();
        assert_eq!(r.name, "suite");
        assert_eq!(r.passed, 5);
        assert_eq!(r.total, 6);
        assert!(!r.success);
    }

    #[test]
    fn json_sink_emits_valid_json() {
        let mut buf = Vec::new();
        let mut sink = JsonSink::new(&mut buf);
        let checks = vec![sample_check("alpha", true), sample_check("beta", false)];
        sink.emit("test_suite", &checks);
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains(r#""suite":"test_suite""#));
        assert!(output.contains(r#""passed":1"#));
        assert!(output.contains(r#""total":2"#));
        assert!(output.contains(r#""all_passed":false"#));
    }

    #[test]
    fn ndjson_sink_streams_lines() {
        let mut buf = Vec::new();
        {
            let mut sink = NdjsonSink::new(&mut buf);
            sink.on_check(&sample_check("a", true));
            sink.on_check(&sample_check("b", false));
            sink.on_finish("suite", 1, 2, false);
        }
        let output = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains(r#""status":"pass""#));
        assert!(lines[1].contains(r#""status":"fail""#));
        assert!(lines[2].contains(r#""type":"summary""#));
    }

    #[test]
    fn escape_json_handles_special_chars() {
        assert_eq!(escape_json(r#"a"b"#), r#"a\"b"#);
        assert_eq!(escape_json("a\\b"), "a\\\\b");
        assert_eq!(escape_json("a\nb"), "a\\nb");
    }

    #[test]
    fn silent_sink_is_noop() {
        let mut sink = SilentSink;
        sink.on_check(&sample_check("x", true));
        sink.on_finish("x", 1, 1, true);
    }
}
