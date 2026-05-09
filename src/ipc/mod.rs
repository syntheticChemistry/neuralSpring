// SPDX-License-Identifier: AGPL-3.0-or-later

//! Per-primal IPC modules — eukaryotic cell membrane.
//!
//! Graduated from the monolithic `ipc_dispatch` module during the
//! interstadial eukaryotic evolution (May 2026). Each submodule owns
//! the JSON-RPC surface for a single primal:
//!
//! | Module       | Primal       | Capabilities |
//! |--------------|--------------|--------------|
//! | [`barracuda`] | barraCuda   | `stats.*`, `tensor.*` |
//! | [`toadstool`] | toadStool   | `compute.dispatch` |
//! | [`beardog`]   | BearDog     | `crypto.hash` |
//! | [`squirrel`]  | Squirrel    | `inference.*` |
//! | [`coralreef`] | coralReef   | `shader.compile.*` |
//!
//! The [`IpcMathClient`] facade provides unified discovery and
//! delegates to per-primal functions.

pub mod barracuda;
pub mod beardog;
pub mod coralreef;
pub mod squirrel;
pub mod toadstool;

use std::path::PathBuf;
use std::time::Duration;

use crate::primal_names;
use crate::validation::composition::{DiscoveryResult, discover_primal_socket, probe_liveness};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Discovered primal socket paths for IPC dispatch.
///
/// Unified facade over per-primal IPC modules. Discovers all
/// math-relevant primals at construction time; missing primals
/// produce `Err` at call time with an honest discovery message.
pub struct IpcMathClient {
    barracuda_socket: Option<PathBuf>,
    toadstool_socket: Option<PathBuf>,
    beardog_socket: Option<PathBuf>,
    squirrel_socket: Option<PathBuf>,
    coralreef_socket: Option<PathBuf>,
    timeout: Duration,
}

impl IpcMathClient {
    /// Discover all math-relevant primals and return a connected client.
    ///
    /// Missing primals are recorded as `None` — calls to their methods will
    /// return `Err` with an honest "not discovered" message.
    #[must_use]
    pub fn discover() -> Self {
        Self {
            barracuda_socket: resolve(primal_names::BARRACUDA),
            toadstool_socket: resolve(primal_names::TOADSTOOL),
            beardog_socket: resolve(primal_names::BEARDOG),
            squirrel_socket: resolve(primal_names::SQUIRREL),
            coralreef_socket: resolve(primal_names::CORALREEF),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Override the default IPC timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Whether the barraCuda primal was discovered.
    #[must_use]
    pub const fn has_barracuda(&self) -> bool {
        self.barracuda_socket.is_some()
    }

    /// Whether the toadStool primal was discovered.
    #[must_use]
    pub const fn has_toadstool(&self) -> bool {
        self.toadstool_socket.is_some()
    }

    /// Whether the BearDog primal was discovered.
    #[must_use]
    pub const fn has_beardog(&self) -> bool {
        self.beardog_socket.is_some()
    }

    /// Whether the coralReef primal was discovered.
    #[must_use]
    pub const fn has_coralreef(&self) -> bool {
        self.coralreef_socket.is_some()
    }

    /// Probe liveness of all discovered primals.
    #[must_use]
    pub fn probe_all(&self) -> IpcLivenessReport {
        let check = |socket: &Option<PathBuf>| {
            socket
                .as_ref()
                .is_some_and(|p| probe_liveness(p, self.timeout).is_ok())
        };
        IpcLivenessReport {
            alive: [
                check(&self.barracuda_socket),
                check(&self.toadstool_socket),
                check(&self.beardog_socket),
                check(&self.squirrel_socket),
                check(&self.coralreef_socket),
            ],
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // barraCuda surface
    // ═══════════════════════════════════════════════════════════════

    /// `stats.mean` via barraCuda IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if barraCuda is not discovered or the IPC call fails.
    pub fn stats_mean(&self, data: &[f64]) -> Result<f64, String> {
        barracuda::stats_mean(self.require_barracuda()?, data, self.timeout)
    }

    /// `stats.std_dev` via barraCuda IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if barraCuda is not discovered or the IPC call fails.
    pub fn stats_std_dev(&self, data: &[f64]) -> Result<f64, String> {
        barracuda::stats_std_dev(self.require_barracuda()?, data, self.timeout)
    }

    /// `stats.weighted_mean` via barraCuda IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if barraCuda is not discovered or the IPC call fails.
    pub fn stats_weighted_mean(&self, data: &[f64], weights: &[f64]) -> Result<f64, String> {
        barracuda::stats_weighted_mean(self.require_barracuda()?, data, weights, self.timeout)
    }

    /// `tensor.matmul` via barraCuda IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if barraCuda is not discovered or the IPC call fails.
    pub fn tensor_matmul(
        &self,
        a: &[f64],
        b: &[f64],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Result<Vec<f64>, String> {
        barracuda::tensor_matmul(
            self.require_barracuda()?,
            a,
            b,
            rows_a,
            cols_a,
            cols_b,
            self.timeout,
        )
    }

    /// `tensor.create` via barraCuda IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if barraCuda is not discovered or the IPC call fails.
    pub fn tensor_create(&self, shape: &[usize], fill: &str) -> Result<serde_json::Value, String> {
        barracuda::tensor_create(self.require_barracuda()?, shape, fill, self.timeout)
    }

    // ═══════════════════════════════════════════════════════════════
    // toadStool surface
    // ═══════════════════════════════════════════════════════════════

    /// `compute.dispatch` via toadStool IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if toadStool is not discovered or the IPC call fails.
    pub fn compute_dispatch(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        toadstool::compute_dispatch(self.require_toadstool()?, params, self.timeout)
    }

    // ═══════════════════════════════════════════════════════════════
    // BearDog surface
    // ═══════════════════════════════════════════════════════════════

    /// `crypto.hash` via BearDog IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if BearDog is not discovered or the IPC call fails.
    pub fn crypto_hash(&self, algorithm: &str, data: &str) -> Result<String, String> {
        beardog::crypto_hash(self.require_beardog()?, algorithm, data, self.timeout)
    }

    // ═══════════════════════════════════════════════════════════════
    // Squirrel surface
    // ═══════════════════════════════════════════════════════════════

    /// `inference.complete` via Squirrel IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if Squirrel is not discovered or the IPC call fails.
    pub fn inference_complete(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        squirrel::inference_complete(self.require_squirrel()?, params, self.timeout)
    }

    /// `inference.embed` via Squirrel IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if Squirrel is not discovered or the IPC call fails.
    pub fn inference_embed(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
        squirrel::inference_embed(self.require_squirrel()?, params, self.timeout)
    }

    // ═══════════════════════════════════════════════════════════════
    // coralReef surface
    // ═══════════════════════════════════════════════════════════════

    /// `shader.compile.wgsl` via coralReef IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if coralReef is not discovered or the IPC call fails.
    pub fn shader_compile_wgsl(
        &self,
        source: &str,
        label: &str,
    ) -> Result<serde_json::Value, String> {
        coralreef::shader_compile_wgsl(self.require_coralreef()?, source, label, self.timeout)
    }

    /// `shader.compile.capabilities` via coralReef IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if coralReef is not discovered or the IPC call fails.
    pub fn shader_capabilities(&self) -> Result<serde_json::Value, String> {
        coralreef::shader_capabilities(self.require_coralreef()?, self.timeout)
    }

    // ═══════════════════════════════════════════════════════════════
    // Internal helpers
    // ═══════════════════════════════════════════════════════════════

    fn require_barracuda(&self) -> Result<&PathBuf, String> {
        self.barracuda_socket
            .as_ref()
            .ok_or_else(|| "barraCuda not discovered — is it running?".to_string())
    }

    fn require_toadstool(&self) -> Result<&PathBuf, String> {
        self.toadstool_socket
            .as_ref()
            .ok_or_else(|| "toadStool not discovered — is it running?".to_string())
    }

    fn require_beardog(&self) -> Result<&PathBuf, String> {
        self.beardog_socket
            .as_ref()
            .ok_or_else(|| "BearDog not discovered — is it running?".to_string())
    }

    fn require_squirrel(&self) -> Result<&PathBuf, String> {
        self.squirrel_socket
            .as_ref()
            .ok_or_else(|| "Squirrel not discovered — is it running?".to_string())
    }

    fn require_coralreef(&self) -> Result<&PathBuf, String> {
        self.coralreef_socket
            .as_ref()
            .ok_or_else(|| "coralReef not discovered — is it running?".to_string())
    }
}

/// Liveness status for all math-relevant primals.
///
/// Indexed by [`PrimalSlot`] to avoid a flat struct with > 3 bools.
pub struct IpcLivenessReport {
    alive: [bool; 5],
}

/// Index into [`IpcLivenessReport`].
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum PrimalSlot {
    /// barraCuda.
    Barracuda = 0,
    /// toadStool.
    Toadstool = 1,
    /// BearDog.
    Beardog = 2,
    /// Squirrel.
    Squirrel = 3,
    /// coralReef.
    Coralreef = 4,
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

fn resolve(primal: &str) -> Option<PathBuf> {
    match discover_primal_socket(primal) {
        DiscoveryResult::Found(path) => Some(path),
        DiscoveryResult::NotFound { .. } => None,
    }
}

pub(crate) fn extract_f64(val: &serde_json::Value, keys: &[&str]) -> Option<f64> {
    for key in keys {
        if let Some(v) = val.get(*key).and_then(serde_json::Value::as_f64) {
            return Some(v);
        }
    }
    val.as_f64()
}

pub(crate) fn extract_f64_array(val: &serde_json::Value, keys: &[&str]) -> Option<Vec<f64>> {
    for key in keys {
        if let Some(arr) = val.get(*key).and_then(|v| v.as_array()) {
            let floats: Vec<f64> = arr.iter().filter_map(serde_json::Value::as_f64).collect();
            if !floats.is_empty() {
                return Some(floats);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_returns_none_for_absent_primals() {
        let client = IpcMathClient::discover();
        let report = client.probe_all();
        assert_eq!(report.alive_count(), 0);
        assert!(!report.is_alive(PrimalSlot::Barracuda));
    }

    #[test]
    fn extract_f64_from_nested() {
        let v = serde_json::json!({"mean": 3.0});
        assert_eq!(extract_f64(&v, &["mean", "result"]), Some(3.0));
    }

    #[test]
    fn extract_f64_fallback_to_root() {
        let v = serde_json::json!(42.0);
        assert_eq!(extract_f64(&v, &["missing"]), Some(42.0));
    }

    #[test]
    fn extract_f64_array_from_result() {
        let v = serde_json::json!({"data": [1.0, 2.0, 3.0]});
        let arr = extract_f64_array(&v, &["data", "result"]);
        assert_eq!(arr, Some(vec![1.0, 2.0, 3.0]));
    }
}
