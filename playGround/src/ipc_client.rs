// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reusable JSON-RPC 2.0 client over Unix domain sockets.
//!
//! Implements the biomeOS 5-tier socket resolution and newline-delimited
//! JSON-RPC protocol used by all ecoPrimals primals.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// biomeOS socket subdirectory name within the XDG runtime directory.
const BIOMEOS_SOCKET_SUBDIR: &str = "biomeos";

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Serialize)]
struct JsonRpcRequest<'a> {
    jsonrpc: &'static str,
    method: &'a str,
    params: &'a serde_json::Value,
    id: u64,
}

#[derive(Debug, Deserialize)]
pub struct JsonRpcResponse {
    #[serde(default)]
    pub result: Option<serde_json::Value>,
    #[serde(default)]
    pub error: Option<JsonRpcError>,
    #[serde(default)]
    pub id: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

impl std::fmt::Display for JsonRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON-RPC error {}: {}", self.code, self.message)
    }
}

/// Send a single JSON-RPC 2.0 request to a Unix socket and return the result.
pub async fn call(
    socket_path: &Path,
    method: &str,
    params: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value> {
    let stream = UnixStream::connect(socket_path)
        .await
        .with_context(|| format!("connecting to {}", socket_path.display()))?;

    let (reader, mut writer) = stream.into_split();

    let request = JsonRpcRequest {
        jsonrpc: "2.0",
        method,
        params,
        id: next_id(),
    };

    let mut payload = serde_json::to_vec(&request)?;
    payload.push(b'\n');
    writer.write_all(&payload).await?;
    writer.flush().await?;

    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    tokio::time::timeout(timeout, buf_reader.read_line(&mut line))
        .await
        .with_context(|| format!("timeout after {timeout:?} waiting for {method}"))?
        .with_context(|| format!("reading response from {}", socket_path.display()))?;

    let resp: JsonRpcResponse =
        serde_json::from_str(line.trim()).context("parsing JSON-RPC response")?;

    if let Some(err) = resp.error {
        anyhow::bail!("{err}");
    }

    resp.result
        .ok_or_else(|| anyhow::anyhow!("JSON-RPC response has neither result nor error"))
}

// ═══════════════════════════════════════════════════════════════════
// biomeOS 5-tier socket resolution
// ═══════════════════════════════════════════════════════════════════

/// Resolve the biomeOS socket directory using the standard 5-tier fallback.
pub fn resolve_socket_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("BIOMEOS_SOCKET_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(xdg).join(BIOMEOS_SOCKET_SUBDIR);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(meta) = std::fs::metadata("/proc/self") {
            let uid = meta.uid();
            let dir = PathBuf::from(format!("/run/user/{uid}")).join(BIOMEOS_SOCKET_SUBDIR);
            if dir.parent().is_some_and(Path::exists) {
                return dir;
            }
        }
    }
    std::env::temp_dir().join(BIOMEOS_SOCKET_SUBDIR)
}

fn get_family_id() -> String {
    std::env::var("FAMILY_ID")
        .or_else(|_| std::env::var("BIOMEOS_FAMILY_ID"))
        .unwrap_or_else(|_| "default".to_string())
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
/// with the `hint_name` if no capability probe succeeds (e.g. primal is
/// not yet running).
///
/// This follows the ecoPrimals self-knowledge principle: a client only
/// knows *what* it needs (a capability), not *who* provides it.
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
            if let Ok(caps) = probe_capabilities(&path) {
                if caps.iter().any(|c| c == required_capability) {
                    return Ok(path);
                }
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
///
/// Returns a list of capability strings, or an error if the primal does
/// not respond within 2 seconds.
fn probe_capabilities(socket_path: &std::path::Path) -> Result<Vec<String>> {
    let params = serde_json::json!({});
    let timeout = Duration::from_secs(2);

    let rt = if let Ok(h) = tokio::runtime::Handle::try_current() {
        h.block_on(call(socket_path, "capability.list", &params, timeout))
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        rt.block_on(call(socket_path, "capability.list", &params, timeout))
    }?;

    let caps: Vec<String> = serde_json::from_value(rt)?;
    Ok(caps)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipc_timeout_default() {
        std::env::remove_var("PRIMAL_IPC_TIMEOUT_SECS");
        assert_eq!(ipc_timeout(), Duration::from_secs(5));
    }

    #[test]
    fn resolve_socket_dir_respects_env() {
        let prev = std::env::var("BIOMEOS_SOCKET_DIR").ok();
        std::env::set_var("BIOMEOS_SOCKET_DIR", "/tmp/test_biomeos");
        let dir = resolve_socket_dir();
        assert_eq!(dir, PathBuf::from("/tmp/test_biomeos"));
        match prev {
            Some(v) => std::env::set_var("BIOMEOS_SOCKET_DIR", v),
            None => std::env::remove_var("BIOMEOS_SOCKET_DIR"),
        }
    }

    #[test]
    fn resolve_socket_dir_falls_through_tiers() {
        let prev_bio = std::env::var("BIOMEOS_SOCKET_DIR").ok();
        let prev_xdg = std::env::var("XDG_RUNTIME_DIR").ok();

        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        std::env::set_var("XDG_RUNTIME_DIR", "/tmp/xdg_test");
        let dir = resolve_socket_dir();
        assert_eq!(dir, PathBuf::from("/tmp/xdg_test/biomeos"));

        match prev_bio {
            Some(v) => std::env::set_var("BIOMEOS_SOCKET_DIR", v),
            None => std::env::remove_var("BIOMEOS_SOCKET_DIR"),
        }
        match prev_xdg {
            Some(v) => std::env::set_var("XDG_RUNTIME_DIR", v),
            None => std::env::remove_var("XDG_RUNTIME_DIR"),
        }
    }

    #[test]
    fn get_family_id_default() {
        let prev_f = std::env::var("FAMILY_ID").ok();
        let prev_b = std::env::var("BIOMEOS_FAMILY_ID").ok();
        std::env::remove_var("FAMILY_ID");
        std::env::remove_var("BIOMEOS_FAMILY_ID");
        assert_eq!(get_family_id(), "default");
        if let Some(v) = prev_f {
            std::env::set_var("FAMILY_ID", v);
        }
        if let Some(v) = prev_b {
            std::env::set_var("BIOMEOS_FAMILY_ID", v);
        }
    }

    #[test]
    fn get_family_id_from_env() {
        let prev = std::env::var("FAMILY_ID").ok();
        std::env::set_var("FAMILY_ID", "test_family");
        assert_eq!(get_family_id(), "test_family");
        match prev {
            Some(v) => std::env::set_var("FAMILY_ID", v),
            None => std::env::remove_var("FAMILY_ID"),
        }
    }

    #[test]
    fn discover_socket_fails_when_dir_missing() {
        let prev = std::env::var("BIOMEOS_SOCKET_DIR").ok();
        std::env::set_var("BIOMEOS_SOCKET_DIR", "/tmp/nonexistent_biomeos_test_dir");
        let result = discover_socket("some_primal");
        assert!(result.is_err());
        match prev {
            Some(v) => std::env::set_var("BIOMEOS_SOCKET_DIR", v),
            None => std::env::remove_var("BIOMEOS_SOCKET_DIR"),
        }
    }

    #[test]
    fn discover_socket_finds_exact_match() {
        let dir = std::env::temp_dir().join("biomeos_discover_test");
        let _ = std::fs::create_dir_all(&dir);
        let sock = dir.join("testprimal.sock");
        std::fs::write(&sock, b"").unwrap();

        let prev = std::env::var("BIOMEOS_SOCKET_DIR").ok();
        let prev_f = std::env::var("FAMILY_ID").ok();
        std::env::set_var("BIOMEOS_SOCKET_DIR", dir.to_str().unwrap());
        std::env::remove_var("FAMILY_ID");
        std::env::remove_var("BIOMEOS_FAMILY_ID");

        let found = discover_socket("testprimal").unwrap();
        assert_eq!(found, sock);

        let _ = std::fs::remove_dir_all(&dir);
        match prev {
            Some(v) => std::env::set_var("BIOMEOS_SOCKET_DIR", v),
            None => std::env::remove_var("BIOMEOS_SOCKET_DIR"),
        }
        if let Some(v) = prev_f {
            std::env::set_var("FAMILY_ID", v);
        }
    }
}
