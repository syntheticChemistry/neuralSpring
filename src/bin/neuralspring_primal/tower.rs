// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tower Atomic discovery and BTSP readiness probes.
//!
//! Discovers `BearDog` (security) and `Songbird` (discovery mesh) via
//! capability-based socket discovery. When found, probes liveness
//! and logs Tower status. Does not block startup — neuralSpring runs
//! standalone when Tower primals are absent.
//!
//! BTSP session establishment is deferred until `BearDog` exposes the
//! `crypto.btsp_handshake` wire (tracked in `docs/PRIMAL_GAPS.md` §6).

use neural_spring::primal_names;
use neural_spring::validation::composition::{self, DiscoveryResult};
use std::time::Duration;

const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Probe Tower Atomic primals at startup.
///
/// Logs discovery status for `BearDog` and `Songbird`. Returns the number
/// of Tower primals discovered (0, 1, or 2). Non-blocking — failures
/// are logged as warnings, not errors.
pub fn probe_tower_atomic() -> usize {
    let mut found = 0;

    match composition::discover_primal_socket(primal_names::BEARDOG) {
        DiscoveryResult::Found(path) => {
            found += 1;
            log::info!("Tower: BearDog discovered at {}", path.display());
            match composition::probe_liveness(&path, PROBE_TIMEOUT) {
                Ok(()) => log::info!("Tower: BearDog liveness OK"),
                Err(e) => log::warn!("Tower: BearDog liveness failed: {e}"),
            }
        }
        DiscoveryResult::NotFound { .. } => {
            log::debug!("Tower: BearDog not running (BTSP unavailable, standalone mode)");
        }
    }

    match composition::discover_primal_socket(primal_names::SONGBIRD) {
        DiscoveryResult::Found(path) => {
            found += 1;
            log::info!("Tower: Songbird discovered at {}", path.display());
            match composition::probe_liveness(&path, PROBE_TIMEOUT) {
                Ok(()) => log::info!("Tower: Songbird liveness OK"),
                Err(e) => log::warn!("Tower: Songbird liveness failed: {e}"),
            }
        }
        DiscoveryResult::NotFound { .. } => {
            log::debug!("Tower: Songbird not running (mesh discovery unavailable)");
        }
    }

    if found == 2 {
        log::info!("Tower Atomic: complete (BearDog + Songbird)");
    } else if found > 0 {
        log::info!("Tower Atomic: partial ({found}/2 primals)");
    } else {
        log::info!("Tower Atomic: not available (standalone mode)");
    }

    found
}
