// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON-RPC 2.0 types and error codes.

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[serde(rename = "jsonrpc")]
    pub _jsonrpc: String,
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

/// JSON-RPC 2.0 standard error codes (§5.1).
pub mod error_code {
    pub const PARSE_ERROR: i32 = -32_700;
    #[expect(dead_code, reason = "protocol completeness")]
    pub const INVALID_REQUEST: i32 = -32_600;
    pub const METHOD_NOT_FOUND: i32 = -32_601;
    pub const INVALID_PARAMS: i32 = -32_602;
    #[expect(dead_code, reason = "protocol completeness")]
    pub const INTERNAL_ERROR: i32 = -32_603;
    pub const SERVER_ERROR: i32 = -32_000;
}
