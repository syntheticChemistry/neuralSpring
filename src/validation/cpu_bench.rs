// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU benchmark infrastructure for Python vs Rust parity validation.
//!
//! Shared helpers for running Python benchmark subprocesses, measuring
//! Rust counterparts, and reporting speedup results. Used by
//! `validate_barracuda_cpu_bench` and any future CPU parity validators.

use super::env::baseline_path;
use std::process::Command;

/// Result of a single Python-vs-Rust CPU benchmark domain.
pub struct CpuBenchResult {
    pub domain: &'static str,
    pub papers: &'static str,
    pub python_us: Option<f64>,
    pub rust_us: f64,
    pub speedup: Option<f64>,
    pub parity_ok: bool,
}

/// Run a Python benchmark script and extract its median µs timing.
///
/// Scripts must print `*_US=<value>` on stdout to report their timing.
/// Returns `None` if the script is missing, fails, or produces no timing.
///
/// Environment variables `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
/// `OMP_NUM_THREADS` are set to `1` to ensure single-threaded parity.
#[must_use]
pub fn run_python_bench(script_rel: &str) -> Option<f64> {
    let script = baseline_path(script_rel);
    if !script.exists() {
        println!("    [skip] Python script not found: {}", script.display());
        return None;
    }
    let output = Command::new("python3")
        .arg(&script)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .output()
        .ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!(
            "    [skip] Python script failed: {}",
            stderr.lines().next().unwrap_or("")
        );
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(idx) = line.find("_US=") {
            let val_str = &line[idx + 4..];
            if let Ok(v) = val_str.trim().parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

/// Record a benchmark result: report speedup and accumulate.
///
/// Replaces the 15-line boilerplate repeated per domain in CPU bench binaries.
pub fn record_domain(
    results: &mut Vec<CpuBenchResult>,
    domain: &'static str,
    papers: &'static str,
    python_us: Option<f64>,
    rust_us: f64,
    parity_ok: bool,
) {
    let speedup = python_us.map(|p| p / rust_us);
    if let (Some(s), Some(p)) = (speedup, python_us) {
        println!("    Python: {p:.1}µs, Rust: {rust_us:.1}µs, Speedup: {s:.1}×");
    } else {
        println!("    Rust: {rust_us:.1}µs (Python unavailable)");
    }
    results.push(CpuBenchResult {
        domain,
        papers,
        python_us,
        rust_us,
        speedup,
        parity_ok,
    });
}

/// Print the summary table for CPU benchmark results.
///
/// Reports per-domain Python µs, Rust µs, speedup, and parity status,
/// then computes the geometric mean speedup across all domains with
/// Python timings.
pub fn print_cpu_summary(h: &mut super::ValidationHarness, results: &[CpuBenchResult]) {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Summary: BarraCUDA CPU vs Python/NumPy                                    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  {:<25} {:>6} {:>10} {:>10} {:>8} {:>7}",
        "Domain", "Papers", "Python µs", "Rust µs", "Speedup", "Parity"
    );
    println!("  {}", "─".repeat(70));

    let mut speedup_count = 0_u32;
    let mut all_parity = true;

    for r in results {
        let py_str = r
            .python_us
            .map_or_else(|| "—".to_string(), |p| format!("{p:.1}"));
        let sp_str = r
            .speedup
            .map_or_else(|| "—".to_string(), |s| format!("{s:.1}×"));
        let par_str = if r.parity_ok { "✓" } else { "✗" };
        println!(
            "  {:<25} {:>6} {:>10} {:>10.1} {:>8} {:>7}",
            r.domain, r.papers, py_str, r.rust_us, sp_str, par_str
        );
        if r.speedup.is_some() {
            speedup_count += 1;
        }
        if !r.parity_ok {
            all_parity = false;
        }
    }

    println!("  {}", "─".repeat(70));
    if speedup_count > 0 {
        let geomean = (results
            .iter()
            .filter_map(|r| r.speedup)
            .map(f64::ln)
            .sum::<f64>()
            / f64::from(speedup_count))
        .exp();
        println!("  Geometric mean speedup: {geomean:.1}× (across {speedup_count} domains)");
        h.check_bool(
            &format!("Geometric mean speedup > 1.0× ({geomean:.1}×)"),
            geomean > 1.0,
        );
    }
    h.check_bool("All parity checks passed", all_parity);

    println!(
        "\n  Portability chain: Python/NumPy \u{2192} BarraCUDA CPU (pure Rust) \u{2192} BarraCUDA GPU\n"
    );
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn record_domain_with_python() {
        let mut results = Vec::new();
        record_domain(&mut results, "test_domain", "001", Some(100.0), 10.0, true);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].domain, "test_domain");
        assert!((results[0].speedup.expect("has Python timing") - 10.0).abs() < 1e-10);
        assert!(results[0].parity_ok);
    }

    #[test]
    fn record_domain_without_python() {
        let mut results = Vec::new();
        record_domain(&mut results, "test", "002", None, 5.0, true);
        assert_eq!(results.len(), 1);
        assert!(results[0].speedup.is_none());
    }

    #[test]
    fn print_cpu_summary_no_panic() {
        let mut h = super::super::ValidationHarness::new("test");
        let results = vec![
            CpuBenchResult {
                domain: "matmul",
                papers: "001",
                python_us: Some(100.0),
                rust_us: 10.0,
                speedup: Some(10.0),
                parity_ok: true,
            },
            CpuBenchResult {
                domain: "fft",
                papers: "002",
                python_us: None,
                rust_us: 5.0,
                speedup: None,
                parity_ok: true,
            },
        ];
        print_cpu_summary(&mut h, &results);
        assert!(h.passed_count() >= 1);
    }

    #[test]
    fn print_cpu_summary_failing_parity() {
        let mut h = super::super::ValidationHarness::new("test");
        let results = vec![CpuBenchResult {
            domain: "broken",
            papers: "003",
            python_us: Some(50.0),
            rust_us: 10.0,
            speedup: Some(5.0),
            parity_ok: false,
        }];
        print_cpu_summary(&mut h, &results);
        assert!(!h.all_passed());
    }

    #[test]
    fn run_python_bench_missing_script() {
        let result = run_python_bench("nonexistent/bench_does_not_exist.py");
        assert!(result.is_none());
    }
}
