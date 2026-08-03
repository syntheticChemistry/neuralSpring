// SPDX-License-Identifier: AGPL-3.0-or-later

//! Capability-to-primal routing infrastructure.
//!
//! [`CapabilityRouter`] resolves capability strings to discovered primal
//! sockets using the compile-time [`CAPABILITY_HINTS`] table.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::capabilities;
use crate::error::IpcError;
use crate::primal_names;
use crate::validation::composition::{DiscoveryResult, discover_primal_socket};

/// Capability-to-primal hint: which primal is expected to provide each
/// capability. Used as fallback when runtime `capability.list` probing
/// is not available.
pub(crate) const CAPABILITY_HINTS: &[(&str, &str)] = &[
    (capabilities::STATS_MEAN, primal_names::BARRACUDA),
    (capabilities::STATS_STD_DEV, primal_names::BARRACUDA),
    (capabilities::STATS_WEIGHTED_MEAN, primal_names::BARRACUDA),
    (capabilities::TENSOR_MATMUL, primal_names::BARRACUDA),
    (capabilities::TENSOR_CREATE, primal_names::BARRACUDA),
    (capabilities::PRECISION_ROUTE, primal_names::BARRACUDA),
    (capabilities::COMPUTE_DISPATCH, primal_names::TOADSTOOL),
    (
        capabilities::COMPUTE_DISPATCH_SUBMIT,
        primal_names::TOADSTOOL,
    ),
    (capabilities::COMPUTE_OFFLOAD, primal_names::TOADSTOOL),
    (capabilities::TOADSTOOL_VALIDATE, primal_names::TOADSTOOL),
    (
        capabilities::TOADSTOOL_LIST_WORKLOADS,
        primal_names::TOADSTOOL,
    ),
    (capabilities::CRYPTO_HASH, primal_names::BEARDOG),
    (capabilities::INFERENCE_COMPLETE, primal_names::SQUIRREL),
    (capabilities::INFERENCE_EMBED, primal_names::SQUIRREL),
    (capabilities::INFERENCE_MODELS, primal_names::SQUIRREL),
    (
        capabilities::INFERENCE_REGISTER_PROVIDER,
        primal_names::SQUIRREL,
    ),
    (
        capabilities::INFERENCE_UNREGISTER_PROVIDER,
        primal_names::SQUIRREL,
    ),
    (capabilities::SHADER_COMPILE_WGSL, primal_names::CORALREEF),
    (
        capabilities::SHADER_COMPILE_CAPABILITIES,
        primal_names::CORALREEF,
    ),
    (capabilities::SECURITY_AUDIT_LOG, primal_names::SKUNKBAT),
    (capabilities::CONTENT_PUT, primal_names::NESTGATE),
    (capabilities::CONTENT_GET, primal_names::NESTGATE),
    (capabilities::CONTENT_EXISTS, primal_names::NESTGATE),
    (capabilities::ML_MLP_INFER, primal_names::BARRACUDA),
    (capabilities::DISCOVERY_PEERS, primal_names::SONGBIRD),
    (capabilities::MESH_INIT, primal_names::SONGBIRD),
    (capabilities::CRYPTO_BTSP_HANDSHAKE, primal_names::BEARDOG),
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
    pub(crate) fn from_hints() -> Self {
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
    pub(crate) fn get(&self, capability: &str) -> Option<&PathBuf> {
        self.routes.get(capability)
    }

    /// Require a socket for a capability, returning a typed error.
    pub(crate) fn require(&self, capability: &str) -> Result<&PathBuf, IpcError> {
        self.get(capability).ok_or_else(|| {
            let primal = CAPABILITY_HINTS
                .iter()
                .find(|&&(c, _)| c == capability)
                .map_or("unknown", |&(_, p)| p);
            IpcError::NotDiscovered { primal }
        })
    }

    /// All unique primal sockets discovered.
    pub(crate) fn discovered_primals(&self) -> Vec<&PathBuf> {
        let seen: std::collections::HashSet<_> = self.routes.values().collect();
        seen.into_iter().collect()
    }
}

pub(crate) fn resolve(primal: &str) -> Option<PathBuf> {
    match discover_primal_socket(primal) {
        DiscoveryResult::Found(path) => Some(path),
        DiscoveryResult::NotFound { .. } => None,
    }
}

/// Name-based hint for a capability (last-resort fallback when runtime probing
/// finds no socket advertising the capability).
#[must_use]
pub fn hint_primal_for_capability(capability: &str) -> Option<&'static str> {
    CAPABILITY_HINTS
        .iter()
        .find(|&&(cap, _)| cap == capability)
        .map(|&(_, primal)| primal)
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
    use crate::capabilities;

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
    fn extract_f64_prefers_first_matching_key() {
        let v = serde_json::json!({"result": 1.0, "mean": 2.0});
        assert_eq!(extract_f64(&v, &["mean", "result"]), Some(2.0));
    }

    #[test]
    fn extract_f64_array_uses_fallback_key() {
        let v = serde_json::json!({"output": [4.0, 5.0]});
        assert_eq!(
            extract_f64_array(&v, &["data", "output"]),
            Some(vec![4.0, 5.0])
        );
    }

    #[test]
    fn extract_f64_array_ignores_empty_after_filter() {
        let v = serde_json::json!({"data": [null, true]});
        assert_eq!(extract_f64_array(&v, &["data"]), None);
    }

    #[test]
    fn extract_f64_none_when_no_numeric_keys() {
        let v = serde_json::json!({"nested": {"text": "value"}});
        assert_eq!(extract_f64(&v, &["nested", "result"]), None);
    }

    #[test]
    fn capability_hints_table_is_nonempty() {
        assert!(CAPABILITY_HINTS.len() >= 20);
        assert!(
            CAPABILITY_HINTS
                .iter()
                .any(|(c, _)| *c == capabilities::STATS_MEAN)
        );
        assert!(
            CAPABILITY_HINTS
                .iter()
                .any(|(c, _)| *c == capabilities::CONTENT_PUT)
        );
    }
}
