// SPDX-License-Identifier: AGPL-3.0-or-later

//! Health probing for discovered primals.
//!
//! [`IpcLivenessReport`] and [`PrimalSlot`] track per-primal liveness
//! status returned by [`probe_all`].

use std::time::Duration;

use crate::capabilities;
use crate::validation::composition::probe_liveness;

use super::router::CapabilityRouter;

/// Liveness status for all IPC-relevant primals.
///
/// Indexed by [`PrimalSlot`] to avoid a flat struct with many bools.
pub struct IpcLivenessReport {
    pub(crate) alive: [bool; 7],
}

/// Index into [`IpcLivenessReport`].
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum PrimalSlot {
    /// barraCuda.
    Barracuda = 0,
    /// toadStool.
    Toadstool = 1,
    /// `BearDog`.
    Beardog = 2,
    /// Squirrel.
    Squirrel = 3,
    /// coralReef.
    Coralreef = 4,
    /// skunkBat.
    Skunkbat = 5,
    /// `NestGate`.
    Nestgate = 6,
}

impl IpcLivenessReport {
    /// Whether a specific primal is alive.
    #[must_use]
    pub const fn is_alive(&self, slot: PrimalSlot) -> bool {
        self.alive[slot as usize]
    }

    /// How many primals are alive.
    #[must_use]
    pub fn alive_count(&self) -> usize {
        self.alive.iter().filter(|&&v| v).count()
    }
}

/// Probe liveness of all discovered primals via the capability router.
#[must_use]
pub(crate) fn probe_all(router: &CapabilityRouter, timeout: Duration) -> IpcLivenessReport {
    let cap_check = |cap: &str| {
        router
            .get(cap)
            .is_some_and(|p| probe_liveness(p, timeout).is_ok())
    };
    IpcLivenessReport {
        alive: [
            cap_check(capabilities::STATS_MEAN),
            cap_check(capabilities::COMPUTE_DISPATCH),
            cap_check(capabilities::CRYPTO_HASH),
            cap_check(capabilities::INFERENCE_COMPLETE),
            cap_check(capabilities::SHADER_COMPILE_WGSL),
            cap_check(capabilities::SECURITY_AUDIT_LOG),
            cap_check(capabilities::CONTENT_PUT),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primal_slot_values() {
        assert_eq!(PrimalSlot::Barracuda as usize, 0);
        assert_eq!(PrimalSlot::Toadstool as usize, 1);
        assert_eq!(PrimalSlot::Beardog as usize, 2);
        assert_eq!(PrimalSlot::Squirrel as usize, 3);
        assert_eq!(PrimalSlot::Coralreef as usize, 4);
        assert_eq!(PrimalSlot::Skunkbat as usize, 5);
        assert_eq!(PrimalSlot::Nestgate as usize, 6);
    }

    #[test]
    fn liveness_report_zero_on_no_primals() {
        let report = IpcLivenessReport { alive: [false; 7] };
        assert_eq!(report.alive_count(), 0);
        for slot in [
            PrimalSlot::Barracuda,
            PrimalSlot::Toadstool,
            PrimalSlot::Beardog,
            PrimalSlot::Squirrel,
            PrimalSlot::Coralreef,
            PrimalSlot::Skunkbat,
            PrimalSlot::Nestgate,
        ] {
            assert!(!report.is_alive(slot));
        }
    }

    #[test]
    fn liveness_report_partial() {
        let report = IpcLivenessReport {
            alive: [true, false, true, false, false, false, false],
        };
        assert_eq!(report.alive_count(), 2);
        assert!(report.is_alive(PrimalSlot::Barracuda));
        assert!(!report.is_alive(PrimalSlot::Toadstool));
        assert!(report.is_alive(PrimalSlot::Beardog));
        assert!(!report.is_alive(PrimalSlot::Skunkbat));
        assert!(!report.is_alive(PrimalSlot::Nestgate));
    }
}
