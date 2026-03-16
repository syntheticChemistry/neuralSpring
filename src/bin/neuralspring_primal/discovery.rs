// SPDX-License-Identifier: AGPL-3.0-or-later

//! Socket resolution and capability-based primal discovery.
//!
//! Primals only have self-knowledge; others are discovered at runtime
//! via the biomeOS orchestrator or by probing live sockets with
//! `capability.list`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use neural_spring::config::BIOMEOS_SOCKET_SUBDIR;

use super::{ipc_response_timeout_secs, orchestrator_socket, PRIMAL_NAME};

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// ═══════════════════════════════════════════════════════════════════
// Socket resolution (capability-based, no hardcoded paths)
// ═══════════════════════════════════════════════════════════════════

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
            if dir.parent().is_some_and(std::path::Path::exists) {
                return dir;
            }
        }
    }
    std::env::temp_dir().join(BIOMEOS_SOCKET_SUBDIR)
}

pub fn resolve_socket_path(family_id: &str) -> PathBuf {
    resolve_socket_dir().join(format!("{PRIMAL_NAME}-{family_id}.sock"))
}

pub fn get_family_id() -> String {
    if let Ok(id) = std::env::var("FAMILY_ID") {
        return id;
    }
    if let Ok(id) = std::env::var("BIOMEOS_FAMILY_ID") {
        return id;
    }
    "default".to_string()
}

pub fn discover_primal_socket(primal_name: &str) -> Result<PathBuf> {
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
        "No socket found for primal '{primal_name}' in {}",
        socket_dir.display()
    )
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
        if let Ok(resp) = discovery {
            if let Some(primal_name) = resp
                .get("result")
                .and_then(|r| r.get("primal"))
                .and_then(|p| p.as_str())
            {
                return forward_to_primal(primal_name, method, params).await;
            }
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

async fn probe_capabilities(socket_path: &std::path::Path) -> Vec<String> {
    let resp = forward_to_primal_raw(socket_path, "capability.list", &serde_json::json!({})).await;

    resp.map_or_else(
        |_| Vec::new(),
        |v| {
            v.get("result")
                .and_then(|r| r.get("capabilities"))
                .and_then(|c| c.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default()
        },
    )
}
