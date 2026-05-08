// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed error hierarchy for neuralSpring library code.
//!
//! Replaces stringly-typed `Result<T, String>` with structured variants
//! that preserve context without losing ergonomics.  Validation binaries
//! may still use `.to_string()` at the boundary — library code returns
//! typed errors.

use std::fmt;

/// Errors originating from GPU device or shader operations.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// GPU device creation or adapter selection failed.
    #[error("gpu device: {reason}")]
    Device {
        /// What went wrong during device creation.
        reason: String,
    },

    /// Shader compilation failed.
    #[error("shader compile ({label}): {reason}")]
    ShaderCompile {
        /// Shader module label.
        label: String,
        /// Compiler diagnostic.
        reason: String,
    },

    /// Buffer creation, upload, or mapping failed.
    #[error("gpu buffer ({op}): {reason}")]
    Buffer {
        /// Buffer operation that failed (e.g. "create", "upload", "map").
        op: &'static str,
        /// Driver or validation-layer message.
        reason: String,
    },

    /// Readback from GPU to CPU failed.
    #[error("gpu readback: {reason}")]
    Readback {
        /// What went wrong during readback.
        reason: String,
    },

    /// Compute dispatch or pipeline submission failed.
    #[error("gpu dispatch: {reason}")]
    Dispatch {
        /// Dispatch failure description.
        reason: String,
    },
}

/// Errors from tensor creation or arithmetic.
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    /// Tensor creation from data failed (shape or device mismatch).
    #[error("tensor create ({context}): {reason}")]
    Create {
        /// Which tensor was being created.
        context: String,
        /// Why creation failed.
        reason: String,
    },

    /// A tensor operation (add, sub, matmul, etc.) failed.
    #[error("tensor op ({op}): {reason}")]
    Operation {
        /// Operation name (e.g. "matmul", "softmax").
        op: &'static str,
        /// Failure description.
        reason: String,
    },

    /// Readback to host memory failed.
    #[error("tensor readback ({context}): {reason}")]
    Readback {
        /// Which readback was attempted.
        context: String,
        /// Why readback failed.
        reason: String,
    },

    /// Shape mismatch between operands.
    #[error("tensor shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// Shape that was expected.
        expected: String,
        /// Shape that was found.
        actual: String,
    },
}

/// Errors from streaming I/O parsers (FASTA, FASTQ, VCF).
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    /// Invalid record format in a bioinformatics file.
    #[error("{format} parse error at record {record}: {reason}")]
    InvalidRecord {
        /// File format being parsed (e.g. "FASTA", "VCF").
        format: &'static str,
        /// Zero-based record index where the error occurred.
        record: usize,
        /// What was wrong with the record.
        reason: String,
    },

    /// Underlying I/O failure.
    #[error("{format} I/O: {source}")]
    Io {
        /// File format being parsed.
        format: &'static str,
        /// The I/O error from the underlying reader.
        source: std::io::Error,
    },
}

/// Top-level library error type.
///
/// Composes domain-specific error types into a single `Result<T, Error>`
/// return type for public API boundaries.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// GPU device or shader operation failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),

    /// Tensor operation failed.
    #[error(transparent)]
    Tensor(#[from] TensorError),

    /// Streaming parser error.
    #[error(transparent)]
    Parse(#[from] ParseError),

    /// Contextual error with a descriptive message.
    ///
    /// Escape hatch for one-off errors that don't warrant a new variant.
    /// Prefer adding a typed variant when a pattern recurs.
    #[error("{0}")]
    Context(String),
}

impl Error {
    /// Create a contextual error from any displayable value.
    pub fn context(msg: impl fmt::Display) -> Self {
        Self::Context(msg.to_string())
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Self::Context(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Self::Context(s.to_owned())
    }
}

impl From<GpuError> for String {
    fn from(e: GpuError) -> Self {
        e.to_string()
    }
}

impl From<TensorError> for String {
    fn from(e: TensorError) -> Self {
        e.to_string()
    }
}

/// Convenience alias for library functions.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_error_display_device() {
        let e = GpuError::Device {
            reason: "no adapter".into(),
        };
        assert_eq!(e.to_string(), "gpu device: no adapter");
    }

    #[test]
    fn gpu_error_display_shader() {
        let e = GpuError::ShaderCompile {
            label: "relu.wgsl".into(),
            reason: "syntax error at line 3".into(),
        };
        assert!(e.to_string().contains("relu.wgsl"));
        assert!(e.to_string().contains("syntax error"));
    }

    #[test]
    fn gpu_error_display_buffer() {
        let e = GpuError::Buffer {
            op: "create",
            reason: "out of memory".into(),
        };
        assert_eq!(e.to_string(), "gpu buffer (create): out of memory");
    }

    #[test]
    fn tensor_error_display_shape_mismatch() {
        let e = TensorError::ShapeMismatch {
            expected: "[4, 8]".into(),
            actual: "[4, 16]".into(),
        };
        assert!(e.to_string().contains("[4, 8]"));
        assert!(e.to_string().contains("[4, 16]"));
    }

    #[test]
    fn tensor_error_display_operation() {
        let e = TensorError::Operation {
            op: "matmul",
            reason: "dimension mismatch".into(),
        };
        assert!(e.to_string().contains("matmul"));
    }

    #[test]
    fn parse_error_display_invalid_record() {
        let e = ParseError::InvalidRecord {
            format: "FASTA",
            record: 42,
            reason: "missing > header".into(),
        };
        let s = e.to_string();
        assert!(s.contains("FASTA"));
        assert!(s.contains("42"));
    }

    #[test]
    fn top_level_error_from_gpu() {
        let gpu = GpuError::Readback {
            reason: "timeout".into(),
        };
        let e: Error = gpu.into();
        assert!(e.to_string().contains("timeout"));
    }

    #[test]
    fn top_level_error_from_tensor() {
        let tensor = TensorError::Create {
            context: "weights".into(),
            reason: "bad shape".into(),
        };
        let e: Error = tensor.into();
        assert!(e.to_string().contains("weights"));
    }

    #[test]
    fn top_level_error_from_parse() {
        let parse = ParseError::InvalidRecord {
            format: "VCF",
            record: 0,
            reason: "missing header".into(),
        };
        let e: Error = parse.into();
        assert!(e.to_string().contains("VCF"));
    }

    #[test]
    fn context_error_from_string() {
        let e: Error = "something went wrong".into();
        assert_eq!(e.to_string(), "something went wrong");
    }

    #[test]
    fn context_error_from_owned_string() {
        let e: Error = String::from("owned error").into();
        assert_eq!(e.to_string(), "owned error");
    }

    #[test]
    fn context_helper() {
        let e = Error::context("helper message");
        assert_eq!(e.to_string(), "helper message");
    }

    #[test]
    fn gpu_error_to_string_conversion() {
        let gpu = GpuError::Dispatch {
            reason: "queue full".into(),
        };
        let s: String = gpu.into();
        assert!(s.contains("queue full"));
    }

    #[test]
    fn tensor_error_to_string_conversion() {
        let tensor = TensorError::Readback {
            context: "output".into(),
            reason: "device lost".into(),
        };
        let s: String = tensor.into();
        assert!(s.contains("device lost"));
    }
}
