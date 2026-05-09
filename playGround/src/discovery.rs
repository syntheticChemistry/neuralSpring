// SPDX-License-Identifier: AGPL-3.0-or-later

//! biomeOS 5-tier socket resolution and capability-based primal discovery.
//!
//! Implements the standard ecoPrimals discovery hierarchy:
//! 1. `$BIOMEOS_SOCKET_DIR` (explicit override)
//! 2. `$XDG_RUNTIME_DIR/biomeos/` (freedesktop standard)
//! 3. `/run/user/{uid}/biomeos/` (Linux fallback)
//! 4. `temp_dir()/biomeos/` (last resort)
//!
//! Primals only know *what* they need (a capability), not *who* provides it.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};

/// Resolve the biomeOS socket directory using the standard 4-tier fallback.
#[must_use]
pub fn resolve_socket_dir() -> PathBuf {
    neural_spring::config::resolve_biomeos_socket_dir()
}

fn get_family_id() -> String {
    neural_spring::config::resolve_family_id()
}

/// Discover a primal socket by name using the standard resolution order:
/// 1. `{name}-{family_id}.sock`
/// 2. `{name}.sock`
/// 3. Any `{name}*.sock` in the socket directory
pub fn discover_socket(primal_name: &str) -> Result<PathBuf> {
    let socket_dir = resolve_socket_dir();
    let family_id = get_family_id();

    let with_family = socket_dir.join(format!("{primal_name}-{family_id}.sock"));
    if with_family.exists() {
        return Ok(with_family);
    }

    let without_family = socket_dir.join(format!("{primal_name}.sock"));
    if without_family.exists() {
        return Ok(without_family);
    }

    if let Ok(entries) = std::fs::read_dir(&socket_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(primal_name) && name_str.ends_with(".sock") {
                return Ok(entry.path());
            }
        }
    }

    anyhow::bail!(
        "no socket found for primal '{primal_name}' in {}",
        socket_dir.display()
    )
}

/// Discover a primal socket by required capability.
///
/// Scans the biomeOS socket directory for any primal that advertises the
/// given capability via `capability.list`.  Falls back to `discover_socket`
/// with the `hint_name` if no capability probe succeeds.
///
/// Follows the ecoPrimals self-knowledge principle: a client only knows
/// *what* it needs (a capability), not *who* provides it.
#[deprecated(
    since = "0.2.0",
    note = "use CompositionContext::from_live_discovery_with_fallback() for capability-based discovery"
)]
pub fn discover_by_capability(required_capability: &str, hint_name: &str) -> Result<PathBuf> {
    let socket_dir = resolve_socket_dir();

    if let Ok(entries) = std::fs::read_dir(&socket_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.ends_with(".sock") {
                continue;
            }
            if let Ok(caps) = probe_capabilities(&path)
                && caps.iter().any(|c| c == required_capability)
            {
                return Ok(path);
            }
        }
    }

    discover_socket(hint_name).with_context(|| {
        format!(
            "no primal advertising '{required_capability}' found, \
             fallback name '{hint_name}' also failed"
        )
    })
}

/// Probe a primal's capabilities by sending `capability.list` over JSON-RPC.
fn probe_capabilities(socket_path: &Path) -> Result<Vec<String>> {
    let params = serde_json::json!({});
    let timeout = Duration::from_secs(2);

    let result = if let Ok(h) = tokio::runtime::Handle::try_current() {
        h.block_on(crate::ipc_client::call(
            socket_path,
            "capability.list",
            &params,
            timeout,
        ))
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        rt.block_on(crate::ipc_client::call(
            socket_path,
            "capability.list",
            &params,
            timeout,
        ))
    }?;

    Ok(parse_capability_list(&result))
}

/// Extract capability strings from any response format used across the ecosystem.
///
/// Handles all 4 formats (airSpring V0.8.7 pattern):
///   - **Flat**: `["cap.a", "cap.b"]`
///   - **Object array**: `[{"name": "cap.a"}, {"capability": "cap.b"}]`
///   - **Nested wrapper**: `{"capabilities": ["cap.a"]}`
///   - **Double-nested**: `{"capabilities": {"capabilities": ["cap.a"]}}`
///   - **Result wrapper**: `{"result": ["cap.a"]}`
///
/// Returns an empty vec (never errors) for unrecognized formats.
#[must_use]
pub fn parse_capability_list(value: &serde_json::Value) -> Vec<String> {
    if let serde_json::Value::Object(obj) = value {
        if let Some(inner) = obj.get("capabilities") {
            return parse_capability_list(inner);
        }
        if let Some(inner) = obj.get("result") {
            return parse_capability_list(inner);
        }
    }

    match value {
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| match v {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Object(obj) => obj
                    .get("name")
                    .or_else(|| obj.get("capability"))
                    .and_then(|n| n.as_str())
                    .map(str::to_owned),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Read the IPC timeout from environment, defaulting to 5 seconds.
#[must_use]
pub fn ipc_timeout() -> Duration {
    let secs: u64 = std::env::var("PRIMAL_IPC_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    Duration::from_secs(secs)
}

/// Generate the environment variable name for a primal's socket override.
///
/// Follows the ecosystem convention: `{UPPER_NAME}_SOCKET`.
#[must_use]
pub fn socket_env_var(primal_name: &str) -> String {
    format!("{}_SOCKET", primal_name.to_uppercase())
}

/// Generate the environment variable name for a primal's address (host:port).
#[must_use]
pub fn address_env_var(primal_name: &str) -> String {
    format!("{}_ADDRESS", primal_name.to_uppercase())
}

/// Discover a primal socket by name, checking the `{UPPER}_SOCKET` env var
/// first, then falling back to biomeOS socket directory resolution.
#[deprecated(
    since = "0.2.0",
    note = "use CompositionContext::from_live_discovery_with_fallback() instead of name-based discovery"
)]
pub fn discover_primal(primal_name: &str) -> Result<PathBuf> {
    let env_key = socket_env_var(primal_name);
    if let Ok(path) = std::env::var(&env_key) {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Ok(p);
        }
    }
    discover_socket(primal_name)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test temp paths — panic on failure is intended"
)]
mod tests {
    use super::*;

    #[test]
    fn ipc_timeout_default() {
        temp_env::with_var_unset("PRIMAL_IPC_TIMEOUT_SECS", || {
            assert_eq!(ipc_timeout(), Duration::from_secs(5));
        });
    }

    #[test]
    fn resolve_socket_dir_respects_env() {
        let test_dir = std::env::temp_dir().join("ns_test_biomeos");
        let test_str = test_dir.to_str().expect("temp_dir is valid UTF-8");
        temp_env::with_var("BIOMEOS_SOCKET_DIR", Some(test_str), || {
            assert_eq!(resolve_socket_dir(), test_dir);
        });
    }

    #[test]
    fn resolve_socket_dir_falls_through_tiers() {
        let xdg_dir = std::env::temp_dir().join("ns_xdg_test");
        let xdg_str = xdg_dir.to_str().expect("temp_dir is valid UTF-8");
        temp_env::with_vars(
            [
                ("BIOMEOS_SOCKET_DIR", None::<&str>),
                ("XDG_RUNTIME_DIR", Some(xdg_str)),
            ],
            || {
                assert_eq!(resolve_socket_dir(), xdg_dir.join("biomeos"));
            },
        );
    }

    #[test]
    fn get_family_id_default() {
        temp_env::with_vars(
            [("FAMILY_ID", None::<&str>), ("BIOMEOS_FAMILY_ID", None)],
            || {
                assert_eq!(get_family_id(), "default");
            },
        );
    }

    #[test]
    fn get_family_id_from_env() {
        temp_env::with_var("FAMILY_ID", Some("test_family"), || {
            assert_eq!(get_family_id(), "test_family");
        });
    }

    #[test]
    fn discover_socket_fails_when_dir_missing() {
        let missing = std::env::temp_dir().join("ns_nonexistent_biomeos_test_dir");
        let missing_str = missing.to_str().expect("temp_dir is valid UTF-8");
        temp_env::with_var("BIOMEOS_SOCKET_DIR", Some(missing_str), || {
            assert!(discover_socket("some_primal").is_err());
        });
    }

    #[test]
    fn parse_capability_list_flat_array() {
        let val = serde_json::json!(["compute.submit", "compute.probe"]);
        assert_eq!(
            parse_capability_list(&val),
            vec!["compute.submit", "compute.probe"]
        );
    }

    #[test]
    fn parse_capability_list_object_format() {
        let obj = serde_json::json!({
            "primal": "neuralspring",
            "capabilities": ["science.ipr", "science.spectral_analysis"]
        });
        assert_eq!(
            parse_capability_list(&obj),
            vec!["science.ipr", "science.spectral_analysis"]
        );
    }

    #[test]
    fn parse_capability_list_object_array_format() {
        let val = serde_json::json!([
            {"name": "health", "version": "1.0"},
            {"capability": "compute.dispatch"}
        ]);
        assert_eq!(
            parse_capability_list(&val),
            vec!["health", "compute.dispatch"]
        );
    }

    #[test]
    fn parse_capability_list_double_nested() {
        let val = serde_json::json!({
            "capabilities": {"capabilities": ["health", "compute.dispatch"]}
        });
        assert_eq!(
            parse_capability_list(&val),
            vec!["health", "compute.dispatch"]
        );
    }

    #[test]
    fn parse_capability_list_result_wrapper() {
        let val = serde_json::json!({"result": ["health", "data.weather"]});
        assert_eq!(parse_capability_list(&val), vec!["health", "data.weather"]);
    }

    #[test]
    fn parse_capability_list_empty_and_junk() {
        assert!(parse_capability_list(&serde_json::json!(null)).is_empty());
        assert!(parse_capability_list(&serde_json::json!(42)).is_empty());
        assert!(parse_capability_list(&serde_json::json!([])).is_empty());
    }

    #[test]
    fn socket_env_var_uppercases() {
        assert_eq!(
            socket_env_var(neural_spring::primal_names::TOADSTOOL),
            "TOADSTOOL_SOCKET"
        );
        assert_eq!(
            socket_env_var(neural_spring::primal_names::BIOMEOS),
            "BIOMEOS_SOCKET"
        );
    }

    #[test]
    fn address_env_var_uppercases() {
        assert_eq!(
            address_env_var(neural_spring::primal_names::NESTGATE),
            "NESTGATE_ADDRESS"
        );
    }

    #[test]
    fn discover_primal_falls_back_to_socket_dir() {
        let missing = std::env::temp_dir().join("ns_nonexistent_biomeos_test_dir");
        let missing_str = missing.to_str().expect("temp_dir is valid UTF-8");
        temp_env::with_vars(
            [
                ("TOADSTOOL_SOCKET", None::<&str>),
                ("BIOMEOS_SOCKET_DIR", Some(missing_str)),
            ],
            || {
                assert!(discover_primal(neural_spring::primal_names::TOADSTOOL).is_err());
            },
        );
    }

    #[test]
    fn discover_socket_finds_exact_match() {
        let dir = std::env::temp_dir().join("biomeos_discover_test");
        let _ = std::fs::create_dir_all(&dir);
        let sock = dir.join("testprimal.sock");
        std::fs::write(&sock, b"").unwrap();

        temp_env::with_vars(
            [
                ("BIOMEOS_SOCKET_DIR", Some(dir.to_str().unwrap())),
                ("FAMILY_ID", None),
                ("BIOMEOS_FAMILY_ID", None),
            ],
            || {
                let found = discover_socket("testprimal").unwrap();
                assert_eq!(found, sock);
            },
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_capability_list_never_panics_on_arbitrary_json() {
        let fuzz_values: &[serde_json::Value] = &[
            serde_json::json!(null),
            serde_json::json!(true),
            serde_json::json!(false),
            serde_json::json!(42),
            serde_json::json!(-1),
            serde_json::json!(7.89),
            serde_json::json!(f64::NAN),
            serde_json::json!(f64::INFINITY),
            serde_json::json!(""),
            serde_json::json!("hello"),
            serde_json::json!([]),
            serde_json::json!([null, true, 42, "cap"]),
            serde_json::json!({}),
            serde_json::json!({"capabilities": null}),
            serde_json::json!({"capabilities": 42}),
            serde_json::json!({"capabilities": "not_array"}),
            serde_json::json!({"capabilities": []}),
            serde_json::json!({"result": null}),
            serde_json::json!({"result": []}),
            serde_json::json!({"result": [null, 42]}),
            serde_json::json!({"capabilities": {"capabilities": null}}),
            serde_json::json!([{"name": null}, {"capability": 42}]),
            serde_json::json!([{"other_field": "val"}]),
            serde_json::json!({"nested": {"capabilities": ["a"]}}),
        ];
        for val in fuzz_values {
            let _ = parse_capability_list(val);
        }
    }

    #[test]
    fn parse_capability_list_flat_roundtrip_preserves_strings() {
        let caps = vec!["health.liveness", "compute.submit", "science.ipr"];
        let val = serde_json::json!(caps);
        let parsed = parse_capability_list(&val);
        assert_eq!(parsed, caps);
    }
}
