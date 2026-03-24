// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(missing_docs)]

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
    Device { reason: String },

    /// Shader compilation failed.
    #[error("shader compile ({label}): {reason}")]
    ShaderCompile { label: String, reason: String },

    /// Buffer creation, upload, or mapping failed.
    #[error("gpu buffer ({op}): {reason}")]
    Buffer { op: &'static str, reason: String },

    /// Readback from GPU to CPU failed.
    #[error("gpu readback: {reason}")]
    Readback { reason: String },

    /// Compute dispatch or pipeline submission failed.
    #[error("gpu dispatch: {reason}")]
    Dispatch { reason: String },
}

/// Errors from tensor creation or arithmetic.
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    /// Tensor creation from data failed (shape or device mismatch).
    #[error("tensor create ({context}): {reason}")]
    Create { context: String, reason: String },

    /// A tensor operation (add, sub, matmul, etc.) failed.
    #[error("tensor op ({op}): {reason}")]
    Operation { op: &'static str, reason: String },

    /// Readback to host memory failed.
    #[error("tensor readback ({context}): {reason}")]
    Readback { context: String, reason: String },

    /// Shape mismatch between operands.
    #[error("tensor shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },
}

/// Errors from streaming I/O parsers (FASTA, FASTQ, VCF).
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    /// Invalid record format in a bioinformatics file.
    #[error("{format} parse error at record {record}: {reason}")]
    InvalidRecord {
        format: &'static str,
        record: usize,
        reason: String,
    },

    /// Underlying I/O failure.
    #[error("{format} I/O: {source}")]
    Io {
        format: &'static str,
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
