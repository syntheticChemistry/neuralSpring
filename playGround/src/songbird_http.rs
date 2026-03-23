// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tower Atomic HTTP client — routes HTTP through Songbird via IPC.
//!
//! Instead of pulling in reqwest/rustls (which depends on `ring`, a C assembly
//! crate), primals route external HTTP through the Tower Atomic stack:
//! `BearDog` (crypto) + Songbird (TLS/HTTP) = Pure Rust HTTPS.
//!
//! This module discovers the `http.request` capability at runtime and forwards
//! all HTTP operations through it. Zero compile-time coupling, zero C deps.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};

use neural_spring::primal_names;

use crate::ipc_client;

const HTTP_CAPABILITY: &str = "http.request";

/// Pure-Rust HTTP client that routes through the Songbird primal.
pub struct SongbirdHttp {
    socket: PathBuf,
    default_headers: HashMap<String, String>,
    timeout: Duration,
}

impl SongbirdHttp {
    /// Discover Songbird via capability-based resolution.
    pub fn discover() -> Result<Self> {
        let socket = ipc_client::discover_by_capability(HTTP_CAPABILITY, primal_names::SONGBIRD)
            .context("discovering Songbird (http.request capability)")?;
        Ok(Self {
            socket,
            default_headers: HashMap::new(),
            timeout: Duration::from_secs(120),
        })
    }

    /// Set a default header applied to all requests.
    pub fn set_header(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.default_headers.insert(key.into(), value.into());
    }

    /// Set the IPC timeout (default: 120s for large downloads).
    pub const fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Perform an HTTP GET and return the response body as a string.
    pub async fn get(&self, url: &str) -> Result<HttpResponse> {
        self.request("GET", url, None, None).await
    }

    /// Perform an HTTP GET and parse the response as JSON.
    pub async fn get_json<T: serde::de::DeserializeOwned>(&self, url: &str) -> Result<T> {
        let resp = self.get(url).await?;
        if resp.status != 200 {
            anyhow::bail!("HTTP {} for {url}: {}", resp.status, resp.body);
        }
        serde_json::from_str(&resp.body)
            .with_context(|| format!("parsing JSON response from {url}"))
    }

    /// Download a URL directly to a file path (Songbird writes the file).
    pub async fn download_to_file(&self, url: &str, dest: &Path) -> Result<u64> {
        let resp = self.request("GET", url, None, Some(dest)).await?;
        if resp.status != 200 {
            anyhow::bail!("HTTP {} downloading {url}", resp.status);
        }
        Ok(resp.content_length)
    }

    async fn request(
        &self,
        method: &str,
        url: &str,
        body: Option<&str>,
        save_to: Option<&Path>,
    ) -> Result<HttpResponse> {
        let mut headers = self.default_headers.clone();
        headers
            .entry("User-Agent".to_owned())
            .or_insert_with(|| "neuralSpring-playGround/0.1.0".to_owned());

        let mut params = serde_json::json!({
            "method": method,
            "url": url,
            "headers": headers,
        });

        if let Some(b) = body {
            params["body"] = serde_json::Value::String(b.to_owned());
        }
        if let Some(dest) = save_to {
            params["save_to"] = serde_json::Value::String(dest.to_string_lossy().into_owned());
        }

        let result = ipc_client::call(&self.socket, HTTP_CAPABILITY, &params, self.timeout)
            .await
            .with_context(|| format!("{method} {url} via Songbird"))?;

        #[expect(
            clippy::cast_possible_truncation,
            reason = "HTTP status codes are always ≤999; u64→u16 safe"
        )]
        let status = result
            .get("status")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u16;

        let body_str = result
            .get("body")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();

        let content_length = result
            .get("content_length")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);

        Ok(HttpResponse {
            status,
            body: body_str,
            content_length,
        })
    }
}

/// HTTP response from Songbird.
pub struct HttpResponse {
    /// HTTP status code from the upstream response.
    pub status: u16,
    /// Response body as UTF-8 text (Songbird IPC payload).
    pub body: String,
    /// Declared `Content-Length` when the remote provided it.
    pub content_length: u64,
}
