// SPDX-License-Identifier: AGPL-3.0-or-later

//! IPC-based math dispatch — calls `barraCuda`, `toadStool`, and `BearDog` over
//! JSON-RPC instead of linking them as Rust library dependencies.
//!
//! This module is the Level 5 (primal proof) counterpart to
//! [`crate::gpu_dispatch::Dispatcher`] (Level 2 Rust proof). Both provide
//! the same math surface; the `Dispatcher` calls `barracuda::` in-process while
//! [`IpcMathClient`] calls primal JSON-RPC methods over Unix sockets.
//!
//! ## Usage
//!
//! ```ignore
//! let client = IpcMathClient::discover()?;
//! let result = client.stats_mean(&[1.0, 2.0, 3.0, 4.0, 5.0])?;
//! assert!((result - 3.0).abs() < 1e-10);
//! ```
//!
//! ## Capability → Primal routing
//!
//! | Method           | Owning primal  |
//! |------------------|----------------|
//! | `tensor.matmul`  | `barraCuda`    |
//! | `tensor.create`  | `barraCuda`    |
//! | `stats.mean`     | `barraCuda`    |
//! | `compute.dispatch` | `toadStool`  |
//! | `crypto.hash`    | `BearDog`      |
//! | `inference.*`    | `Squirrel`     |

use std::path::PathBuf;
use std::time::Duration;

use crate::primal_names;
use crate::validation::composition::{
    DiscoveryResult, call_capability, discover_primal_socket, probe_liveness,
};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Discovered primal socket paths for IPC dispatch.
pub struct IpcMathClient {
    barracuda: Option<PathBuf>,
    toadstool: Option<PathBuf>,
    beardog: Option<PathBuf>,
    squirrel: Option<PathBuf>,
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
            barracuda: resolve(primal_names::BARRACUDA),
            toadstool: resolve(primal_names::TOADSTOOL),
            beardog: resolve(primal_names::BEARDOG),
            squirrel: resolve(primal_names::SQUIRREL),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Override the default IPC timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Whether the `barraCuda` primal was discovered.
    #[must_use]
    pub const fn has_barracuda(&self) -> bool {
        self.barracuda.is_some()
    }

    /// Whether the `toadStool` primal was discovered.
    #[must_use]
    pub const fn has_toadstool(&self) -> bool {
        self.toadstool.is_some()
    }

    /// Whether the `BearDog` primal was discovered.
    #[must_use]
    pub const fn has_beardog(&self) -> bool {
        self.beardog.is_some()
    }

    /// Probe liveness of all discovered primals.
    ///
    /// Returns a summary of which primals are alive.
    #[must_use]
    pub fn probe_all(&self) -> IpcLivenessReport {
        let check = |socket: &Option<PathBuf>| {
            socket
                .as_ref()
                .is_some_and(|p| probe_liveness(p, self.timeout).is_ok())
        };
        IpcLivenessReport {
            alive: [
                check(&self.barracuda),
                check(&self.toadstool),
                check(&self.beardog),
                check(&self.squirrel),
            ],
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // barraCuda surface (tensor.*, stats.*)
    // ═══════════════════════════════════════════════════════════════

    /// `stats.mean` via `barraCuda` IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if `barraCuda` is not discovered or the IPC call fails.
    pub fn stats_mean(&self, data: &[f64]) -> Result<f64, String> {
        let socket = self.require_barracuda()?;
        let result = call_capability(
            socket,
            "stats.mean",
            &serde_json::json!({ "data": data }),
            self.timeout,
        )?;
        extract_f64(&result, &["mean", "result", "value"])
            .ok_or_else(|| "stats.mean: response missing numeric result".to_string())
    }

    /// `stats.std_dev` via `barraCuda` IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if `barraCuda` is not discovered or the IPC call fails.
    pub fn stats_std_dev(&self, data: &[f64]) -> Result<f64, String> {
        let socket = self.require_barracuda()?;
        let result = call_capability(
            socket,
            "stats.std_dev",
            &serde_json::json!({ "data": data }),
            self.timeout,
        )?;
        extract_f64(&result, &["std_dev", "result", "value"])
            .ok_or_else(|| "stats.std_dev: response missing numeric result".to_string())
    }

    /// `stats.weighted_mean` via `barraCuda` IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if `barraCuda` is not discovered or the IPC call fails.
    pub fn stats_weighted_mean(&self, data: &[f64], weights: &[f64]) -> Result<f64, String> {
        let socket = self.require_barracuda()?;
        let result = call_capability(
            socket,
            "stats.weighted_mean",
            &serde_json::json!({ "data": data, "weights": weights }),
            self.timeout,
        )?;
        extract_f64(&result, &["weighted_mean", "result", "value"])
            .ok_or_else(|| "stats.weighted_mean: response missing numeric result".to_string())
    }

    /// `tensor.matmul` via `barraCuda` IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if `barraCuda` is not discovered or the IPC call fails.
    pub fn tensor_matmul(
        &self,
        a: &[f64],
        b: &[f64],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Result<Vec<f64>, String> {
        let socket = self.require_barracuda()?;
        let result = call_capability(
            socket,
            "tensor.matmul",
            &serde_json::json!({
                "a": a, "b": b,
                "rows_a": rows_a, "cols_a": cols_a, "cols_b": cols_b,
            }),
            self.timeout,
        )?;
        extract_f64_array(&result, &["data", "result"])
            .ok_or_else(|| "tensor.matmul: response missing data array".to_string())
    }

    /// `tensor.create` via `barraCuda` IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if `barraCuda` is not discovered or the IPC call fails.
    pub fn tensor_create(&self, shape: &[usize], fill: &str) -> Result<serde_json::Value, String> {
        let socket = self.require_barracuda()?;
        call_capability(
            socket,
            "tensor.create",
            &serde_json::json!({ "shape": shape, "fill": fill }),
            self.timeout,
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // toadStool surface (compute.dispatch)
    // ═══════════════════════════════════════════════════════════════

    /// `compute.dispatch` via `toadStool` IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if `toadStool` is not discovered or the IPC call fails.
    pub fn compute_dispatch(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let socket = self.require_toadstool()?;
        call_capability(socket, "compute.dispatch", params, self.timeout)
    }

    // ═══════════════════════════════════════════════════════════════
    // BearDog surface (crypto.hash)
    // ═══════════════════════════════════════════════════════════════

    /// `crypto.hash` via `BearDog` IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if `BearDog` is not discovered or the IPC call fails.
    pub fn crypto_hash(&self, algorithm: &str, data: &str) -> Result<String, String> {
        let socket = self.require_beardog()?;
        let result = call_capability(
            socket,
            "crypto.hash",
            &serde_json::json!({ "algorithm": algorithm, "data": data }),
            self.timeout,
        )?;
        result
            .get("hash")
            .or_else(|| result.get("digest"))
            .or_else(|| result.get("result"))
            .and_then(|v| v.as_str())
            .map(String::from)
            .ok_or_else(|| "crypto.hash: response missing hash string".to_string())
    }

    // ═══════════════════════════════════════════════════════════════
    // Squirrel surface (inference.*)
    // ═══════════════════════════════════════════════════════════════

    /// `inference.complete` via `Squirrel` IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if `Squirrel` is not discovered or the IPC call fails.
    pub fn inference_complete(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let socket = self.require_squirrel()?;
        call_capability(socket, "inference.complete", params, self.timeout)
    }

    /// `inference.embed` via `Squirrel` IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if `Squirrel` is not discovered or the IPC call fails.
    pub fn inference_embed(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
        let socket = self.require_squirrel()?;
        call_capability(socket, "inference.embed", params, self.timeout)
    }

    // ═══════════════════════════════════════════════════════════════
    // Internal helpers
    // ═══════════════════════════════════════════════════════════════

    fn require_barracuda(&self) -> Result<&PathBuf, String> {
        self.barracuda
            .as_ref()
            .ok_or_else(|| "barraCuda not discovered — is it running?".to_string())
    }

    fn require_toadstool(&self) -> Result<&PathBuf, String> {
        self.toadstool
            .as_ref()
            .ok_or_else(|| "toadStool not discovered — is it running?".to_string())
    }

    fn require_beardog(&self) -> Result<&PathBuf, String> {
        self.beardog
            .as_ref()
            .ok_or_else(|| "BearDog not discovered — is it running?".to_string())
    }

    fn require_squirrel(&self) -> Result<&PathBuf, String> {
        self.squirrel
            .as_ref()
            .ok_or_else(|| "Squirrel not discovered — is it running?".to_string())
    }
}

/// Liveness status for all math-relevant primals.
///
/// Indexed by [`PrimalSlot`] to avoid a flat struct with > 3 bools.
pub struct IpcLivenessReport {
    /// Per-primal liveness: `[barraCuda, toadStool, BearDog, Squirrel]`.
    alive: [bool; 4],
}

/// Index into [`IpcLivenessReport`].
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum PrimalSlot {
    /// `barraCuda`.
    Barracuda = 0,
    /// `toadStool`.
    Toadstool = 1,
    /// `BearDog`.
    Beardog = 2,
    /// `Squirrel`.
    Squirrel = 3,
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

fn extract_f64(val: &serde_json::Value, keys: &[&str]) -> Option<f64> {
    for key in keys {
        if let Some(v) = val.get(*key).and_then(serde_json::Value::as_f64) {
            return Some(v);
        }
    }
    val.as_f64()
}

fn extract_f64_array(val: &serde_json::Value, keys: &[&str]) -> Option<Vec<f64>> {
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
