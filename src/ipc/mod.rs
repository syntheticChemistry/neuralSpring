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
//! | [`beardog`]   | `BearDog`   | `crypto.hash` |
//! | [`squirrel`]  | Squirrel    | `inference.*` |
//! | [`coralreef`] | coralReef   | `shader.compile.*` |
//! | [`skunkbat`]  | skunkBat    | `security.audit_log` |
//! | [`nestgate`]  | NestGate    | `content.put`, `content.get` |
//!
//! ## Capability-Based Discovery
//!
//! [`IpcMathClient`] discovers primals by **capability hint** with primal
//! name as fallback. Each capability declares which primal is expected to
//! provide it, but the discovery mechanism resolves sockets by scanning
//! the biomeOS runtime directory — the client never hardcodes socket paths.
//!
//! This follows the ecoPrimals self-knowledge principle: a spring only
//! knows *what* it needs (a capability), not *where* to find it.

pub mod barracuda;
pub mod beardog;
pub mod coralreef;
pub mod nestgate;
pub mod skunkbat;
pub mod squirrel;
pub mod toadstool;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use crate::capabilities;
use crate::error::IpcError;
use crate::primal_names;
use crate::validation::composition::{DiscoveryResult, discover_primal_socket, probe_liveness};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Capability-to-primal hint: which primal is expected to provide each
/// capability. Used as fallback when runtime `capability.list` probing
/// is not available.
const CAPABILITY_HINTS: &[(&str, &str)] = &[
    (capabilities::STATS_MEAN, primal_names::BARRACUDA),
    (capabilities::STATS_STD_DEV, primal_names::BARRACUDA),
    (capabilities::STATS_WEIGHTED_MEAN, primal_names::BARRACUDA),
    (capabilities::TENSOR_MATMUL, primal_names::BARRACUDA),
    (capabilities::TENSOR_CREATE, primal_names::BARRACUDA),
    (capabilities::COMPUTE_DISPATCH, primal_names::TOADSTOOL),
    (capabilities::COMPUTE_OFFLOAD, primal_names::TOADSTOOL),
    (capabilities::CRYPTO_HASH, primal_names::BEARDOG),
    (capabilities::INFERENCE_COMPLETE, primal_names::SQUIRREL),
    (capabilities::INFERENCE_EMBED, primal_names::SQUIRREL),
    (capabilities::INFERENCE_MODELS, primal_names::SQUIRREL),
    (capabilities::SHADER_COMPILE_WGSL, primal_names::CORALREEF),
    (capabilities::SHADER_COMPILE_CAPABILITIES, primal_names::CORALREEF),
    (capabilities::SECURITY_AUDIT_LOG, primal_names::SKUNKBAT),
    (capabilities::CONTENT_PUT, primal_names::NESTGATE),
    (capabilities::CONTENT_GET, primal_names::NESTGATE),
    (capabilities::CONTENT_EXISTS, primal_names::NESTGATE),
];

/// Routes capability requests to discovered primal sockets.
///
/// Populated at construction time via name-based discovery with
/// capability hints. The router maps capability strings (e.g.
/// `"stats.mean"`) to socket paths, deduplicating when multiple
/// capabilities resolve to the same primal.
pub struct CapabilityRouter {
    routes: HashMap<&'static str, PathBuf>,
}

impl CapabilityRouter {
    /// Build a router by resolving each known capability hint.
    #[must_use]
    fn from_hints() -> Self {
        let mut primal_cache: HashMap<&str, Option<PathBuf>> = HashMap::new();
        let mut routes = HashMap::new();

        for &(capability, hint_primal) in CAPABILITY_HINTS {
            let socket = primal_cache
                .entry(hint_primal)
                .or_insert_with(|| resolve(hint_primal))
                .clone();
            if let Some(path) = socket {
                routes.insert(capability, path);
            }
        }

        Self { routes }
    }

    /// Look up the socket for a capability.
    fn get(&self, capability: &str) -> Option<&PathBuf> {
        self.routes.get(capability)
    }

    /// Require a socket for a capability, returning a typed error.
    fn require(&self, capability: &str) -> Result<&PathBuf, IpcError> {
        self.get(capability)
            .ok_or_else(|| {
                let primal = CAPABILITY_HINTS
                    .iter()
                    .find(|&&(c, _)| c == capability)
                    .map_or("unknown", |&(_, p)| p);
                IpcError::NotDiscovered { primal }
            })
    }

    /// All unique primal sockets discovered.
    fn discovered_primals(&self) -> Vec<&PathBuf> {
        let mut seen = Vec::new();
        for path in self.routes.values() {
            if !seen.iter().any(|p: &&PathBuf| *p == path) {
                seen.push(path);
            }
        }
        seen
    }
}

/// Discovered primal socket paths for IPC dispatch.
///
/// Unified facade over per-primal IPC modules. Uses capability-based
/// routing: each method call resolves through the [`CapabilityRouter`]
/// rather than a fixed primal name. Missing primals produce `Err` at
/// call time with an honest discovery message.
pub struct IpcMathClient {
    router: CapabilityRouter,
    timeout: Duration,
}

impl IpcMathClient {
    /// Discover all math-relevant primals and return a connected client.
    ///
    /// Missing primals are recorded in the capability router — calls to
    /// their methods will return `Err` with an honest "not discovered"
    /// message.
    #[must_use]
    pub fn discover() -> Self {
        Self {
            router: CapabilityRouter::from_hints(),
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
    pub fn has_barracuda(&self) -> bool {
        self.router.get(capabilities::STATS_MEAN).is_some()
    }

    /// Whether the toadStool primal was discovered.
    #[must_use]
    pub fn has_toadstool(&self) -> bool {
        self.router.get(capabilities::COMPUTE_DISPATCH).is_some()
    }

    /// Whether the `BearDog` primal was discovered.
    #[must_use]
    pub fn has_beardog(&self) -> bool {
        self.router.get(capabilities::CRYPTO_HASH).is_some()
    }

    /// Whether the coralReef primal was discovered.
    #[must_use]
    pub fn has_coralreef(&self) -> bool {
        self.router.get(capabilities::SHADER_COMPILE_WGSL).is_some()
    }

    /// Probe liveness of all discovered primals.
    #[must_use]
    pub fn probe_all(&self) -> IpcLivenessReport {
        let cap_check = |cap: &str| {
            self.router
                .get(cap)
                .is_some_and(|p| probe_liveness(p, self.timeout).is_ok())
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

    // ═══════════════════════════════════════════════════════════════
    // barraCuda surface (routed via stats.*/tensor.* capabilities)
    // ═══════════════════════════════════════════════════════════════

    /// `stats.mean` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides `stats.mean` or the call fails.
    pub fn stats_mean(&self, data: &[f64]) -> Result<f64, IpcError> {
        barracuda::stats_mean(self.router.require(capabilities::STATS_MEAN)?, data, self.timeout)
    }

    /// `stats.std_dev` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides `stats.std_dev` or the call fails.
    pub fn stats_std_dev(&self, data: &[f64]) -> Result<f64, IpcError> {
        barracuda::stats_std_dev(self.router.require(capabilities::STATS_STD_DEV)?, data, self.timeout)
    }

    /// `stats.weighted_mean` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn stats_weighted_mean(&self, data: &[f64], weights: &[f64]) -> Result<f64, IpcError> {
        barracuda::stats_weighted_mean(
            self.router.require(capabilities::STATS_WEIGHTED_MEAN)?,
            data, weights, self.timeout,
        )
    }

    /// `tensor.matmul` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn tensor_matmul(
        &self,
        a: &[f64],
        b: &[f64],
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Result<Vec<f64>, IpcError> {
        barracuda::tensor_matmul(
            self.router.require(capabilities::TENSOR_MATMUL)?,
            a, b, rows_a, cols_a, cols_b, self.timeout,
        )
    }

    /// `tensor.create` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn tensor_create(&self, shape: &[usize], fill: &str) -> Result<serde_json::Value, IpcError> {
        barracuda::tensor_create(self.router.require(capabilities::TENSOR_CREATE)?, shape, fill, self.timeout)
    }

    // ═══════════════════════════════════════════════════════════════
    // toadStool surface (routed via compute.dispatch)
    // ═══════════════════════════════════════════════════════════════

    /// `compute.dispatch` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn compute_dispatch(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, IpcError> {
        toadstool::compute_dispatch(self.router.require(capabilities::COMPUTE_DISPATCH)?, params, self.timeout)
    }

    // ═══════════════════════════════════════════════════════════════
    // BearDog surface (routed via crypto.hash)
    // ═══════════════════════════════════════════════════════════════

    /// `crypto.hash` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn crypto_hash(&self, algorithm: &str, data: &str) -> Result<String, IpcError> {
        beardog::crypto_hash(self.router.require(capabilities::CRYPTO_HASH)?, algorithm, data, self.timeout)
    }

    // ═══════════════════════════════════════════════════════════════
    // Squirrel surface (routed via inference.*)
    // ═══════════════════════════════════════════════════════════════

    /// `inference.complete` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn inference_complete(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, IpcError> {
        squirrel::inference_complete(self.router.require(capabilities::INFERENCE_COMPLETE)?, params, self.timeout)
    }

    /// `inference.embed` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn inference_embed(&self, params: &serde_json::Value) -> Result<serde_json::Value, IpcError> {
        squirrel::inference_embed(self.router.require(capabilities::INFERENCE_EMBED)?, params, self.timeout)
    }

    // ═══════════════════════════════════════════════════════════════
    // coralReef surface (routed via shader.compile.*)
    // ═══════════════════════════════════════════════════════════════

    /// `shader.compile.wgsl` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn shader_compile_wgsl(
        &self,
        source: &str,
        label: &str,
    ) -> Result<serde_json::Value, IpcError> {
        coralreef::shader_compile_wgsl(self.router.require(capabilities::SHADER_COMPILE_WGSL)?, source, label, self.timeout)
    }

    /// `shader.compile.capabilities` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn shader_capabilities(&self) -> Result<serde_json::Value, IpcError> {
        coralreef::shader_capabilities(self.router.require(capabilities::SHADER_COMPILE_CAPABILITIES)?, self.timeout)
    }

    // ═══════════════════════════════════════════════════════════════
    // skunkBat surface (routed via security.audit_log)
    // ═══════════════════════════════════════════════════════════════

    /// `security.audit_log` — routed to whichever primal provides it.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides this capability or the call fails.
    pub fn audit_log(
        &self,
        event_type: &str,
        source: &str,
        payload: &serde_json::Value,
    ) -> Result<serde_json::Value, IpcError> {
        skunkbat::audit_log(
            self.router.require(capabilities::SECURITY_AUDIT_LOG)?,
            event_type, source, payload, self.timeout,
        )
    }

    /// Whether the skunkBat primal was discovered.
    #[must_use]
    pub fn has_skunkbat(&self) -> bool {
        self.router.get(capabilities::SECURITY_AUDIT_LOG).is_some()
    }

    /// Whether the NestGate primal was discovered.
    #[must_use]
    pub fn has_nestgate(&self) -> bool {
        self.router.get(capabilities::CONTENT_PUT).is_some()
    }

    // ═══════════════════════════════════════════════════════════════
    // NestGate surface (routed via content.*)
    // ═══════════════════════════════════════════════════════════════

    /// `content.put` — store content-addressed data via NestGate.
    ///
    /// Returns the BLAKE3 hash and metadata on success.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides `content.put` or the call fails.
    pub fn content_put(
        &self,
        data_base64: &str,
        content_type: Option<&str>,
    ) -> Result<nestgate::ContentPutResult, IpcError> {
        nestgate::content_put(
            self.router.require(capabilities::CONTENT_PUT)?,
            data_base64,
            content_type,
            self.timeout,
        )
    }

    /// `content.get` — retrieve content-addressed data by BLAKE3 hash.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides `content.get` or the call fails.
    pub fn content_get(&self, hash: &str) -> Result<nestgate::ContentGetResult, IpcError> {
        nestgate::content_get(
            self.router.require(capabilities::CONTENT_GET)?,
            hash,
            self.timeout,
        )
    }

    /// `content.exists` — check whether content-addressed data exists.
    ///
    /// # Errors
    ///
    /// Returns an error if no primal provides `content.exists` or the call fails.
    pub fn content_exists(&self, hash: &str) -> Result<bool, IpcError> {
        nestgate::content_exists(
            self.router.require(capabilities::CONTENT_EXISTS)?,
            hash,
            self.timeout,
        )
    }

    /// Number of unique primals discovered.
    #[must_use]
    pub fn discovered_count(&self) -> usize {
        self.router.discovered_primals().len()
    }
}

/// Liveness status for all IPC-relevant primals.
///
/// Indexed by [`PrimalSlot`] to avoid a flat struct with many bools.
pub struct IpcLivenessReport {
    alive: [bool; 7],
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
    /// NestGate.
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

    #[test]
    fn extract_f64_none_for_non_numeric() {
        let v = serde_json::json!({"name": "test"});
        assert_eq!(extract_f64(&v, &["name", "result"]), None);
    }

    #[test]
    fn extract_f64_array_empty_on_non_array() {
        let v = serde_json::json!({"data": "not-an-array"});
        assert_eq!(extract_f64_array(&v, &["data"]), None);
    }

    #[test]
    fn extract_f64_array_skips_non_float_elements() {
        let v = serde_json::json!({"data": ["a", "b"]});
        assert_eq!(extract_f64_array(&v, &["data"]), None);
    }

    #[test]
    fn client_require_returns_err_for_missing_primals() {
        let client = IpcMathClient::discover();
        assert!(client.stats_mean(&[1.0]).is_err());
        assert!(client.crypto_hash("blake3", "test").is_err());
        assert!(client.compute_dispatch(&serde_json::json!({})).is_err());
        assert!(client.inference_complete(&serde_json::json!({})).is_err());
        assert!(client.shader_capabilities().is_err());
        assert!(
            client
                .audit_log("test", "neuralspring", &serde_json::json!({}))
                .is_err()
        );
        assert!(client.content_put("dGVzdA==", None).is_err());
        assert!(client.content_get("deadbeef").is_err());
        assert!(client.content_exists("deadbeef").is_err());
    }

    #[test]
    fn with_timeout_overrides_default() {
        let client = IpcMathClient::discover().with_timeout(Duration::from_millis(500));
        assert_eq!(client.timeout, Duration::from_millis(500));
    }

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
    fn has_coralreef_false_when_absent() {
        let client = IpcMathClient::discover();
        assert!(!client.has_coralreef());
    }

    #[test]
    fn has_nestgate_false_when_absent() {
        let client = IpcMathClient::discover();
        assert!(!client.has_nestgate());
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
