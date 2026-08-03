// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tower Atomic discovery and BTSP readiness probes.
//!
//! Discovers Tower primals via capability-based socket probing. The
//! security primal is found by probing for `crypto.btsp_handshake` (with
//! name hint fallback), and the mesh primal by probing for
//! `discovery.peers`. Does not block startup — neuralSpring runs
//! standalone when Tower primals are absent.
//!
//! BTSP session establishment is deferred until `BearDog` exposes the
//! `crypto.btsp_handshake` wire (tracked in `docs/PRIMAL_GAPS.md` §6).

use neural_spring::primal_names;
use neural_spring::validation::composition::{self, DiscoveryResult};
use std::time::Duration;

const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Required capabilities for each Tower primal.
const SECURITY_CAPABILITIES: &[&str] = &["crypto.btsp_handshake", "security.audit_log"];
const MESH_CAPABILITIES: &[&str] = &["discovery.peers", "mesh.init"];

/// Probe Tower Atomic primals at startup.
///
/// Discovers primals by probing sockets for capability advertisements.
/// Falls back to name-based discovery if capability probing finds nothing.
/// Returns the number of Tower primals discovered (0, 1, or 2).
/// Non-blocking — failures are logged as warnings, not errors.
pub fn probe_tower_atomic() -> usize {
    let mut found = 0;

    if probe_tower_primal(
        primal_names::BEARDOG,
        "BearDog",
        SECURITY_CAPABILITIES,
        "BTSP unavailable, standalone mode",
    ) {
        found += 1;
    }

    if probe_tower_primal(
        primal_names::SONGBIRD,
        "Songbird",
        MESH_CAPABILITIES,
        "mesh discovery unavailable",
    ) {
        found += 1;
    }

    match found {
        2 => log::info!("Tower Atomic: complete (security + mesh)"),
        1 => log::info!("Tower Atomic: partial (1/2 primals)"),
        _ => log::info!("Tower Atomic: not available (standalone mode)"),
    }

    found
}

/// Discover and probe a single Tower primal.
///
/// Tries capability-based discovery first: scans live sockets for one
/// that advertises the required capabilities. Falls back to name-based
/// socket lookup if no socket advertises the capability.
fn probe_tower_primal(
    name_hint: &str,
    display_name: &str,
    required_capabilities: &[&str],
    absent_reason: &str,
) -> bool {
    if let Some(path) = discover_by_capability(required_capabilities) {
        log::info!(
            "Tower: {display_name} discovered via capability at {}",
            path.display()
        );
        return probe_liveness_and_log(&path, display_name);
    }

    match composition::discover_primal_socket(name_hint) {
        DiscoveryResult::Found(path) => {
            log::info!(
                "Tower: {display_name} discovered via name at {}",
                path.display()
            );
            probe_liveness_and_log(&path, display_name)
        }
        DiscoveryResult::NotFound { .. } => {
            log::debug!("Tower: {display_name} not running ({absent_reason})");
            false
        }
    }
}

/// Scan socket directory for a primal advertising any of the given capabilities.
fn discover_by_capability(capabilities: &[&str]) -> Option<std::path::PathBuf> {
    let socket_dir = neural_spring::config::resolve_biomeos_socket_dir();
    let entries = std::fs::read_dir(&socket_dir).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.ends_with(".sock") {
            continue;
        }

        if let Ok(caps) = composition::probe_capabilities(&path, PROBE_TIMEOUT) {
            for required in capabilities {
                if caps.iter().any(|c| c == required) {
                    return Some(path);
                }
            }
        }
    }

    None
}

fn probe_liveness_and_log(path: &std::path::Path, display_name: &str) -> bool {
    match composition::probe_liveness(path, PROBE_TIMEOUT) {
        Ok(()) => {
            log::info!("Tower: {display_name} liveness OK");
            true
        }
        Err(e) => {
            log::warn!("Tower: {display_name} liveness failed: {e}");
            false
        }
    }
}
