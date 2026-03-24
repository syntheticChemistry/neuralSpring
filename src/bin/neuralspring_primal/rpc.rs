// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON-RPC 2.0 types and error codes.

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[serde(rename = "jsonrpc")]
    pub jsonrpc_version: String,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
    pub id: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    pub id: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

impl JsonRpcResponse {
    pub const fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0",
            result: Some(result),
            error: None,
            id,
        }
    }

    pub const fn error(id: serde_json::Value, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0",
            result: None,
            error: Some(JsonRpcError { code, message }),
            id,
        }
    }
}

/// Normalize a JSON-RPC method name: accepts both `{domain}.{operation}`
/// (standard) and legacy `neuralspring.{domain}.{operation}` (backward-compatible).
///
/// Mirrors `barracuda-core::ipc::methods::normalize_method` per ecosystem
/// convention (wetSpring V132, loamSpine v0.9.8, barraCuda v0.3.7).
pub fn normalize_method(method: &str) -> &str {
    method
        .strip_prefix(super::PRIMAL_NAME)
        .and_then(|s| s.strip_prefix('.'))
        .unwrap_or(method)
}

/// JSON-RPC 2.0 standard error codes (§5.1).
pub mod error_code {
    pub const PARSE_ERROR: i32 = -32_700;
    pub const INVALID_REQUEST: i32 = -32_600;
    pub const METHOD_NOT_FOUND: i32 = -32_601;
    pub const INVALID_PARAMS: i32 = -32_602;
    pub const INTERNAL_ERROR: i32 = -32_603;
    /// JSON-RPC application-defined server error range floor (−32000..−32099).
    #[expect(
        dead_code,
        reason = "reserved range anchor; handlers use INTERNAL_ERROR"
    )]
    pub const SERVER_ERROR: i32 = -32_000;
}
