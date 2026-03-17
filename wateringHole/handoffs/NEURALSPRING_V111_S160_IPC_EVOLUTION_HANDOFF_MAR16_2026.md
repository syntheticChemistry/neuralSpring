# neuralSpring V111 — S160 IPC Evolution Handoff

**Date**: March 16, 2026
**From**: neuralSpring S160
**Scope**: Structured IPC errors, compute.dispatch protocol, centralized RPC extraction

## Changes

### 1. Structured `IpcError` (healthSpring V31 / rhizoCrypt V13 pattern)

`playGround/src/ipc_client.rs` now exports a typed `IpcError` enum:

```rust
pub enum IpcError {
    Connect(std::io::Error),
    Write(std::io::Error),
    Read(std::io::Error),
    InvalidJson(serde_json::Error),
    NoResult,
    RpcError { code: i64, message: String },
    Timeout,
}
```

`is_recoverable()` returns `true` for `Connect` and `Timeout` — callers can use this
for targeted retry logic without pattern-matching.

### 2. `call_typed()` — Structured Error Path

New `call_typed()` returns `Result<Value, IpcError>` instead of `Result<Value>`. The
existing `call()` function is preserved for backward compatibility (wraps `call_typed()`
with anyhow conversion).

### 3. `extract_rpc_error()` — Centralized RPC Error Extraction

```rust
pub fn extract_rpc_error(response: &Value) -> Option<(i64, String)>
```

Replaces ad-hoc `response.get("error").is_some()` patterns (airSpring V0.8.6 pattern).

### 4. Typed `compute.dispatch` Protocol (wetSpring V124)

`ToadStoolClient` gains three new methods:

- `dispatch_submit(operation, input)` → `DispatchHandle`
- `dispatch_result(dispatch_id)` → `DispatchResult`
- `dispatch_capabilities()` → `Vec<String>`

Types: `DispatchHandle { dispatch_id, status }`, `DispatchResult { dispatch_id, status, output, elapsed_ms }`.

### 5. `JsonRpcError::code` i32 → i64

JSON-RPC 2.0 specifies integer codes without a fixed bit width. Evolved from `i32` to
`i64` for full spec compliance and alignment with ecosystem practice.

## Quality Gates

| Metric | Value |
|--------|-------|
| Library tests | 1128 |
| playGround tests | 61 |
| Forge tests | 73 |
| Clippy warnings | 0 (pedantic + nursery) |
| Unfulfilled expectations | 0 |
| `#[allow()]` | 0 |
| `unsafe` blocks | 0 (`forbid(unsafe_code)` on all 3 crates) |
| C dependencies | 0 |
| fmt diffs | 0 |

## Absorption Opportunities for Other Springs

- **`IpcError` pattern**: Any spring using `anyhow::bail!` for IPC failures can adopt
  structured errors for retry logic.
- **`extract_rpc_error()`**: Replaces duplicated error field extraction.
- **`compute.dispatch` client**: Ready for springs that delegate GPU work to ToadStool.
- **`call_typed()`**: New code should prefer this over `call()` for typed error handling.

## Remaining Evolution

- **rhizoCrypt NDJSON streaming**: `ipc_client` could support streaming line-delimited
  responses for long-running operations.
- **biomeOS SDK `CapabilityClient`**: When biomeos-primal-sdk matures, playGround can
  migrate `discover_by_capability` to the SDK's typed client.
- **Circuit breaker / retry**: `IpcError::is_recoverable()` enables future
  `CircuitBreaker + RetryPolicy` (rhizoCrypt pattern).
