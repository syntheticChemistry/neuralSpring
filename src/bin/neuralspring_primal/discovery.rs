// SPDX-License-Identifier: AGPL-3.0-or-later

//! Socket resolution and capability-based primal discovery.
//!
//! Primals only have self-knowledge; others are discovered at runtime
//! via the biomeOS orchestrator or by probing live sockets with
//! `capabilities.list` (Semantic Method Naming Standard v2.1).
//!
//! ## biomeOS 5-Tier Discovery
//!
//! 1. `BIOMEOS_SOCKET_DIR` env override
//! 2. `$XDG_RUNTIME_DIR/biomeos/`
//! 3. `/run/user/{uid}/biomeos/` (identity socket)
//! 4. `temp_dir()/biomeos/`
//! 5. `socket-registry.json` in the resolved socket directory

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use neural_spring::config;
use neural_spring::ipc::router::hint_primal_for_capability;
use neural_spring::validation::composition::{self, DiscoveryResult};

use super::{PRIMAL_NAME, ipc_response_timeout_secs, orchestrator_socket};

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// ═══════════════════════════════════════════════════════════════════
// Socket resolution (capability-based, no hardcoded paths)
// ═══════════════════════════════════════════════════════════════════

pub fn resolve_socket_dir() -> PathBuf {
    config::resolve_biomeos_socket_dir()
}

pub fn resolve_socket_path(family_id: &str) -> PathBuf {
    resolve_socket_dir().join(format!("{PRIMAL_NAME}-{family_id}.sock"))
}

pub fn get_family_id() -> String {
    config::resolve_family_id()
}

pub fn discover_primal_socket(primal_name: &str) -> Result<PathBuf> {
    let socket_dir = resolve_socket_dir();
    let family_id = get_family_id();

    // Tier 1-4: direct socket file lookup
    let with_family = socket_dir.join(format!("{primal_name}-{family_id}.sock"));
    if with_family.exists() {
        return Ok(with_family);
    }

    let without_family = socket_dir.join(format!("{primal_name}.sock"));
    if without_family.exists() {
        return Ok(without_family);
    }

    // Tier 5: socket-registry.json (biomeOS v2.66+)
    if let Some(path) = lookup_socket_registry(&socket_dir, primal_name) {
        if path.exists() {
            return Ok(path);
        }
    }

    // Fallback: scan directory for prefix match
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
        "No socket found for primal '{primal_name}' in {}",
        socket_dir.display()
    )
}

/// Discover a primal socket by advertised capability.
///
/// Scans the biomeOS socket directory, probes each live socket with
/// `capabilities.list` / `capability.list`, and returns the first socket
/// that advertises `capability`. Falls back to the compile-time
/// [`hint_primal_for_capability`] name hint when no socket advertises it.
#[must_use]
pub fn discover_by_capability(capability: &str, probe_timeout: Duration) -> Option<PathBuf> {
    let socket_dir = resolve_socket_dir();

    if let Ok(entries) = std::fs::read_dir(&socket_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.ends_with(".sock") || name_str.starts_with(PRIMAL_NAME) {
                continue;
            }
            let socket_path = entry.path();
            if let Ok(caps) = composition::probe_capabilities(&socket_path, probe_timeout)
                && caps
                    .iter()
                    .any(|c| c == capability || capability.starts_with(c.as_str()))
            {
                return Some(socket_path);
            }
        }
    }

    hint_primal_for_capability(capability).and_then(
        |hint| match composition::discover_primal_socket(hint) {
            DiscoveryResult::Found(path) => Some(path),
            DiscoveryResult::NotFound { .. } => None,
        },
    )
}

/// Look up a primal's socket path from the biomeOS `socket-registry.json`.
///
/// The registry maps primal names to socket paths. Returns `None` if the
/// registry file does not exist or does not contain the requested primal.
fn lookup_socket_registry(socket_dir: &std::path::Path, primal_name: &str) -> Option<PathBuf> {
    let registry_path = socket_dir.join(neural_spring::config::SOCKET_REGISTRY_FILENAME);
    let contents = std::fs::read_to_string(&registry_path).ok()?;
    let registry: serde_json::Value = serde_json::from_str(&contents).ok()?;
    let socket_str = registry.get(primal_name)?.as_str()?;
    let path = PathBuf::from(socket_str);
    if path.is_absolute() {
        Some(path)
    } else {
        Some(socket_dir.join(path))
    }
}

// ═══════════════════════════════════════════════════════════════════
// IPC forwarding
// ═══════════════════════════════════════════════════════════════════

pub async fn forward_to_primal(
    primal_name: &str,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value> {
    let socket = discover_primal_socket(primal_name)?;
    let stream = UnixStream::connect(&socket)
        .await
        .with_context(|| format!("connecting to {primal_name} at {}", socket.display()))?;

    let (reader, mut writer) = stream.into_split();

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": next_id()
    });

    let mut payload = serde_json::to_vec(&request)?;
    payload.push(b'\n');
    writer.write_all(&payload).await?;
    writer.flush().await?;

    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    buf_reader.read_line(&mut line).await?;

    let resp: serde_json::Value = serde_json::from_str(line.trim())?;
    Ok(resp)
}

pub async fn forward_to_primal_raw(
    socket_path: &std::path::Path,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value> {
    let stream = UnixStream::connect(socket_path).await?;
    let (reader, mut writer) = stream.into_split();

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": next_id()
    });

    let mut payload = serde_json::to_vec(&request)?;
    payload.push(b'\n');
    writer.write_all(&payload).await?;
    writer.flush().await?;

    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(ipc_response_timeout_secs()),
        buf_reader.read_line(&mut line),
    )
    .await
    .with_context(|| "timeout waiting for biomeOS response")??;

    let resp: serde_json::Value = serde_json::from_str(line.trim())?;
    Ok(resp)
}

/// Discover which primal can handle `data.*` methods at runtime.
///
/// Uses capability-based discovery exclusively — no hardcoded primal names.
pub async fn discover_data_primal_and_forward(
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value> {
    let socket_dir = resolve_socket_dir();

    let biomeos_socket = socket_dir.join(orchestrator_socket());
    if biomeos_socket.exists() {
        let discovery = forward_to_primal_raw(
            &biomeos_socket,
            "capability.resolve",
            &serde_json::json!({ "capability": method }),
        )
        .await;
        if let Ok(resp) = discovery
            && let Some(primal_name) = resp
                .get("result")
                .and_then(|r| r.get("primal"))
                .and_then(|p| p.as_str())
        {
            return forward_to_primal(primal_name, method, params).await;
        }
    }

    if let Ok(entries) = std::fs::read_dir(&socket_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.ends_with(".sock") || name_str.starts_with(PRIMAL_NAME) {
                continue;
            }
            let socket_path = entry.path();
            let caps = probe_capabilities(&socket_path).await;
            if caps
                .iter()
                .any(|c| c == method || method.starts_with(c.as_str()))
            {
                let primal = name_str
                    .trim_end_matches(".sock")
                    .rsplit_once('-')
                    .map_or_else(|| name_str.trim_end_matches(".sock"), |(base, _)| base);
                if let Ok(resp) = forward_to_primal(primal, method, params).await {
                    return Ok(resp);
                }
            }
        }
    }

    anyhow::bail!(
        "No primal found with data capability for '{method}' in {}",
        socket_dir.display()
    )
}

/// Probe a primal socket for its advertised capabilities.
///
/// Tries canonical `capabilities.list` first (v2.1 standard), then falls
/// back to legacy `capability.list` for older primals.
async fn probe_capabilities(socket_path: &std::path::Path) -> Vec<String> {
    let empty = serde_json::json!({});

    if let Ok(v) = forward_to_primal_raw(socket_path, "capabilities.list", &empty).await {
        let caps = parse_capability_response(&v);
        if !caps.is_empty() {
            return caps;
        }
    }

    if let Ok(v) = forward_to_primal_raw(socket_path, "capability.list", &empty).await {
        return parse_capability_response(&v);
    }

    Vec::new()
}

/// Parse capabilities from any of the 4 standard response formats.
fn parse_capability_response(v: &serde_json::Value) -> Vec<String> {
    let Some(result) = v.get("result") else {
        return Vec::new();
    };

    // Format A: { "capabilities": ["cap1", "cap2"] }
    if let Some(arr) = result.get("capabilities").and_then(|c| c.as_array()) {
        return arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
    }

    // Format B: { "capabilities": [{"name": "cap1"}, ...] }
    if let Some(arr) = result.get("capabilities").and_then(|c| c.as_array()) {
        let names: Vec<String> = arr
            .iter()
            .filter_map(|v| v.get("name")?.as_str().map(String::from))
            .collect();
        if !names.is_empty() {
            return names;
        }
    }

    // Format C: result is directly an array
    if let Some(arr) = result.as_array() {
        return arr
            .iter()
            .filter_map(|v| {
                v.as_str()
                    .map(String::from)
                    .or_else(|| v.get("name")?.as_str().map(String::from))
            })
            .collect();
    }

    Vec::new()
}
