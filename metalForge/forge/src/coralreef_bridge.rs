// SPDX-License-Identifier: AGPL-3.0-or-later

//! coralReef GPU compiler bridge — sovereign shader compilation.
//!
//! Two integration paths following sovereignty principles:
//!
//! 1. **Compile-time** (`coralreef` feature): Links `coral-reef` directly for
//!    native binary compilation. Produces GPU-native code (PTX/SASS for NVIDIA,
//!    ISA for AMD) from WGSL source.
//!
//! 2. **Runtime discovery** (always available): Discovers a running coralReef
//!    primal via Unix socket at `$XDG_RUNTIME_DIR/biomeos/coralreef.sock`
//!    or capability manifest at `$XDG_RUNTIME_DIR/ecoPrimals/*.json`
//!    (with `shader.compile` capability), then calls `shader.compile.wgsl`
//!    over JSON-RPC. This path requires no compile-time dependency and
//!    follows the primal self-knowledge pattern.
//!
//! ## Absorption target
//!
//! `barracuda::device::coral_compiler` already has `spawn_coral_compile` for
//! async native compilation. This bridge completes the neuralSpring side:
//! shader catalog → coralReef compilation → parity validation against wgpu.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use neural_spring_forge::coralreef_bridge::{CoralCompiler, CoralResult};
//!
//! // Compile-time path (requires `coralreef` feature)
//! let compiler = CoralCompiler::auto()?;
//! let binary = compiler.compile_wgsl(GELU_F64_WGSL)?;
//!
//! // Runtime discovery (always available)
//! if let Some(socket) = CoralCompiler::discover_socket() {
//!     // IPC compile via discovered coralReef primal
//! }
//! ```

use std::path::PathBuf;

/// Result type for coralReef operations.
pub type CoralResult<T> = Result<T, CoralError>;

/// Errors from coralReef compilation or discovery.
#[derive(Debug)]
pub enum CoralError {
    /// coralReef feature not enabled at compile time.
    NotAvailable,
    /// Compilation failed.
    CompileFailed(String),
    /// Socket discovery failed.
    SocketNotFound,
}

impl std::fmt::Display for CoralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAvailable => write!(f, "coralReef feature not enabled"),
            Self::CompileFailed(e) => write!(f, "coralReef compile: {e}"),
            Self::SocketNotFound => write!(f, "coralReef socket not found"),
        }
    }
}

impl std::error::Error for CoralError {}

/// Compiled shader binary from coralReef.
#[derive(Debug)]
pub struct CompiledShader {
    /// Native GPU binary (PTX, SASS, AMD ISA, etc.).
    pub binary: Vec<u8>,
    /// Target architecture string.
    pub arch: String,
}

/// coralReef compiler bridge.
///
/// Wraps the coralReef compilation API behind a feature-gated interface.
/// When the `coralreef` feature is disabled, all compile methods return
/// [`CoralError::NotAvailable`].
pub struct CoralCompiler {
    #[cfg(feature = "coralreef")]
    options: coral_reef::CompileOptions,
    /// Whether compile-time support is available.
    available: bool,
}

impl CoralCompiler {
    /// Auto-detect GPU target and create a compiler.
    ///
    /// With `coralreef` feature: probes GPU via adapter info.
    /// Without: returns a stub that reports `NotAvailable`.
    #[must_use]
    #[cfg(feature = "coralreef")]
    pub fn auto() -> Self {
        Self {
            options: coral_reef::CompileOptions::default(),
            available: true,
        }
    }

    #[cfg(not(feature = "coralreef"))]
    #[must_use]
    pub const fn auto() -> Self {
        Self { available: false }
    }

    /// Whether compile-time coralReef support is available.
    #[must_use]
    pub const fn is_available(&self) -> bool {
        self.available
    }

    /// Compile WGSL source to native GPU binary.
    ///
    /// # Errors
    ///
    /// Returns [`CoralError::NotAvailable`] if the `coralreef` feature is
    /// disabled, or [`CoralError::CompileFailed`] if compilation fails.
    #[cfg(feature = "coralreef")]
    pub fn compile_wgsl(&self, wgsl: &str) -> CoralResult<CompiledShader> {
        let opts = &self.options;
        match coral_reef::compile_wgsl(wgsl, opts) {
            Ok(binary) => Ok(CompiledShader {
                binary,
                arch: format!("{:?}", opts.target),
            }),
            Err(e) => Err(CoralError::CompileFailed(format!("{e:?}"))),
        }
    }

    /// Compile WGSL source to native GPU binary.
    ///
    /// # Errors
    ///
    /// Always returns [`CoralError::NotAvailable`] when the `coralreef`
    /// feature is disabled.
    #[cfg(not(feature = "coralreef"))]
    pub const fn compile_wgsl(&self, _wgsl: &str) -> CoralResult<CompiledShader> {
        Err(CoralError::NotAvailable)
    }

    /// Discover a running coralReef primal's IPC socket.
    ///
    /// Checks two paths following the ecosystem convention:
    /// 1. Unix socket: `$XDG_RUNTIME_DIR/biomeos/coralreef.sock`
    /// 2. Capability manifests: `$XDG_RUNTIME_DIR/ecoPrimals/*.json`
    ///    (looks for `shader.compile` or `shader_compiler` capability)
    ///
    /// Returns the socket path if found, `None` otherwise.
    #[must_use]
    pub fn discover_socket() -> Option<PathBuf> {
        let xdg = std::env::var("XDG_RUNTIME_DIR").ok()?;
        let base = PathBuf::from(xdg);

        // Primary: Unix socket (coralReef Iteration 6+)
        let sock = base.join("biomeos").join("coralreef.sock");
        if sock.exists() {
            return Some(sock);
        }

        // Fallback: capability manifest scan (dual-write discovery)
        let manifest_dir = base.join("ecoPrimals");
        if let Ok(entries) = std::fs::read_dir(&manifest_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "json") {
                    if let Ok(contents) = std::fs::read_to_string(&path) {
                        if contents.contains("shader.compile")
                            || contents.contains("shader_compiler")
                        {
                            return Some(path);
                        }
                    }
                }
            }
        }

        None
    }

    /// Check if coralReef compilation is available either via compile-time
    /// dependency or runtime socket discovery.
    #[must_use]
    pub fn is_reachable(&self) -> bool {
        self.available || Self::discover_socket().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiler_creation() {
        let compiler = CoralCompiler::auto();
        #[cfg(feature = "coralreef")]
        assert!(compiler.is_available());
        #[cfg(not(feature = "coralreef"))]
        assert!(!compiler.is_available());
    }

    #[test]
    fn compile_without_feature_returns_not_available() {
        #[cfg(not(feature = "coralreef"))]
        {
            let compiler = CoralCompiler::auto();
            let result = compiler.compile_wgsl("@compute @workgroup_size(1) fn main() {}");
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), CoralError::NotAvailable));
        }
    }

    #[test]
    fn socket_discovery_does_not_panic() {
        let _ = CoralCompiler::discover_socket();
    }

    #[test]
    fn reachable_check() {
        let compiler = CoralCompiler::auto();
        let _ = compiler.is_reachable();
    }
}
