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

/// Socket directory name for petalTongue discovery.
///
/// Used when probing `$XDG_RUNTIME_DIR` and `temp_dir()` for a
/// petalTongue instance. Delegates to `primal_names::PETALTONGUE`.
pub const PETALTONGUE_SOCKET_DIR: &str = crate::primal_names::PETALTONGUE;

/// Socket file prefix for petalTongue discovery.
///
/// Used when scanning `temp_dir()` for petalTongue socket files
/// matching the `{prefix}*.sock` pattern.
pub const PETALTONGUE_SOCKET_PREFIX: &str = crate::primal_names::PETALTONGUE;

// ═══════════════════════════════════════════════════════════════════
// biomeOS socket resolution
// ═══════════════════════════════════════════════════════════════════

/// biomeOS socket subdirectory name (under `$XDG_RUNTIME_DIR`).
pub const BIOMEOS_SOCKET_SUBDIR: &str = "biomeos";

/// biomeOS orchestrator socket filename.
pub const BIOMEOS_ORCHESTRATOR_SOCKET: &str = "biomeos.sock";

/// biomeOS orchestrator socket env var override.
pub const ENV_BIOMEOS_ORCHESTRATOR: &str = "BIOMEOS_ORCHESTRATOR_SOCKET";

// ═══════════════════════════════════════════════════════════════════
// Primal name hints (re-exported from primal_names for backwards compat)
// ═══════════════════════════════════════════════════════════════════

/// `ToadStool` name hint for fallback discovery.
pub const TOADSTOOL_NAME_HINT: &str = crate::primal_names::TOADSTOOL;

/// coralReef name hint for fallback discovery.
pub const CORALREEF_NAME_HINT: &str = crate::primal_names::CORALREEF;

/// Squirrel name hint for fallback discovery.
pub const SQUIRREL_NAME_HINT: &str = crate::primal_names::SQUIRREL;

// ═══════════════════════════════════════════════════════════════════
// Validation / GPU env vars
// ═══════════════════════════════════════════════════════════════════

/// Require GPU for validation binaries (exit 0 if absent when unset).
pub const ENV_REQUIRE_GPU: &str = "NEURALSPRING_REQUIRE_GPU";

/// Override GPU backend selection (`vulkan`, `metal`, `dx12`).
pub const ENV_GPU_BACKEND: &str = "NEURALSPRING_BACKEND";

// ═══════════════════════════════════════════════════════════════════
// Capability strings announced via Songbird
// ═══════════════════════════════════════════════════════════════════

/// Prefix for all neuralSpring capabilities.
pub const CAPABILITY_PREFIX: &str = "science";

/// Complete capability set advertised by neuralSpring via `capability.list`.
///
/// Single source of truth — imported by `neuralspring_primal` and
/// `playGround::mcp_tools` instead of maintaining duplicate lists.
pub const ALL_CAPABILITIES: &[&str] = &[
    "science.spectral_analysis",
    "science.anderson_localization",
    "science.hessian_eigen",
    "science.agent_coordination",
    "science.ipr",
    "science.disorder_sweep",
    "science.training_trajectory",
    "science.evoformer_block",
    "science.structure_module",
    "science.folding_health",
    "science.gpu_dispatch",
    "science.cross_spring_provenance",
    "science.cross_spring_benchmark",
    "science.precision_routing",
];

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
