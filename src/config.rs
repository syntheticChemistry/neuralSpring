// SPDX-License-Identifier: AGPL-3.0-or-later

//! Centralized configuration constants for neuralSpring.
//!
//! Gathers environment variable names, primal identity strings, and
//! runtime defaults in one place. No ad-hoc strings scattered across
//! modules — if a value is used in more than one file, it belongs here.

// ═══════════════════════════════════════════════════════════════════
// Primal identity
// ═══════════════════════════════════════════════════════════════════

/// This primal's family name, used in visualization scenarios and
/// capability announcements. Derived from `CARGO_PKG_NAME` at build
/// time for single-source-of-truth.
pub const PRIMAL_FAMILY: &str = env!("CARGO_PKG_NAME");

/// Human-readable display name (capitalized).
pub const PRIMAL_DISPLAY_NAME: &str = "neuralSpring";

/// petalTongue domain palette key (triggers electric blue/magenta).
pub const PETALTONGUE_DOMAIN: &str = "neural";

/// petalTongue UI theme.
pub const PETALTONGUE_THEME: &str = "neural-dark";

// ═══════════════════════════════════════════════════════════════════
// Environment variable names
// ═══════════════════════════════════════════════════════════════════

/// Override path to petalTongue Unix socket.
pub const ENV_PETALTONGUE_SOCKET: &str = "PETALTONGUE_SOCKET";

/// XDG runtime directory (standard freedesktop.org).
pub const ENV_XDG_RUNTIME_DIR: &str = "XDG_RUNTIME_DIR";

/// Require GPU for validation binaries (exit 0 if absent when unset).
pub const ENV_REQUIRE_GPU: &str = "NEURALSPRING_REQUIRE_GPU";

/// Override GPU backend selection (`vulkan`, `metal`, `dx12`).
pub const ENV_GPU_BACKEND: &str = "NEURALSPRING_BACKEND";

// ═══════════════════════════════════════════════════════════════════
// Capability strings announced via Songbird
// ═══════════════════════════════════════════════════════════════════

/// Prefix for all neuralSpring capabilities.
pub const CAPABILITY_PREFIX: &str = "science";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primal_family_matches_package() {
        assert_eq!(PRIMAL_FAMILY, "neural-spring");
    }

    #[test]
    fn env_vars_are_nonempty() {
        assert!(!ENV_PETALTONGUE_SOCKET.is_empty());
        assert!(!ENV_XDG_RUNTIME_DIR.is_empty());
        assert!(!ENV_REQUIRE_GPU.is_empty());
        assert!(!ENV_GPU_BACKEND.is_empty());
    }

    #[test]
    fn domain_theme_consistency() {
        assert!(PETALTONGUE_THEME.contains(PETALTONGUE_DOMAIN));
    }
}
