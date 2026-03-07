// SPDX-License-Identifier: AGPL-3.0-or-later

//! Environment-driven runtime policy and path resolution.

use std::path::PathBuf;
use std::process;

/// Whether `REQUIRE_GPU=1` is set.
///
/// Capability-based: reads `REQUIRE_GPU` first, falls back to
/// `NEURALSPRING_REQUIRE_GPU` for backward compatibility.
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
    std::env::var("REQUIRE_GPU")
        .or_else(|_| std::env::var("NEURALSPRING_REQUIRE_GPU"))
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Handle the absence of a GPU adapter in a validation binary.
///
/// If `REQUIRE_GPU=1`, prints an error and exits 1.
/// Otherwise, prints a skip message and exits 0.
///
/// Replaces the duplicated `let Ok(gpu) = Gpu::new().await else { … }`
/// pattern across all GPU validation binaries.
pub fn exit_no_gpu() -> ! {
    if gpu_required() {
        log::info!("FAIL: no GPU adapter (REQUIRE_GPU=1)");
        process::exit(1);
    }
    log::info!("0/0 checks — skipping gracefully (no GPU adapter)");
    process::exit(0);
}

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

/// Whether the current GPU adapter is a software rasterizer (e.g. llvmpipe).
///
/// Fused GPU shaders (`VarianceF64`, `CorrelationF64`, `HmmBatchForwardF64`)
/// produce incorrect results on software Vulkan due to a wgpu 28 interaction
/// (upstream `BarraCUDA` `Fp64Strategy` regression — all Springs affected).
/// Tests gated behind this check skip gracefully on software backends and
/// run on real hardware.
#[must_use]
pub fn is_software_adapter(adapter_name: &str) -> bool {
    let lower = adapter_name.to_lowercase();
    lower.contains("llvmpipe")
        || lower.contains("swiftshader")
        || lower.contains("lavapipe")
        || lower.contains("software")
}

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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn gpu_required_respects_env() {
        std::env::remove_var("GPU_BACKEND");
        let original = std::env::var("REQUIRE_GPU").ok();
        let legacy = std::env::var("NEURALSPRING_REQUIRE_GPU").ok();

        std::env::remove_var("NEURALSPRING_REQUIRE_GPU");

        std::env::set_var("REQUIRE_GPU", "0");
        assert!(!gpu_required(), "0 → false");

        std::env::set_var("REQUIRE_GPU", "1");
        assert!(gpu_required(), "1 → true");

        std::env::set_var("REQUIRE_GPU", "true");
        assert!(gpu_required(), "true → true");

        std::env::set_var("REQUIRE_GPU", "TRUE");
        assert!(gpu_required(), "TRUE → true");

        // Fallback to legacy env var
        std::env::remove_var("REQUIRE_GPU");
        std::env::set_var("NEURALSPRING_REQUIRE_GPU", "1");
        assert!(gpu_required(), "legacy fallback → true");

        std::env::remove_var("REQUIRE_GPU");
        std::env::remove_var("NEURALSPRING_REQUIRE_GPU");
        assert!(!gpu_required(), "unset → false");

        if let Some(v) = original {
            std::env::set_var("REQUIRE_GPU", v);
        }
        if let Some(v) = legacy {
            std::env::set_var("NEURALSPRING_REQUIRE_GPU", v);
        }
    }

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
