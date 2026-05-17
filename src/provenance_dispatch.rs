// SPDX-License-Identifier: AGPL-3.0-or-later

//! Signal dispatch for provenance chains — `nest.store` and `nest.commit`.
//!
//! Extracted from `weight_loader` to separate signal dispatch concerns from
//! safetensors/JSON weight loading. These functions use `CompositionContext`
//! to dispatch composed signals through biomeOS, which manages the full
//! provenance graph (rhizoCrypt → bearDog → nestGate → loamSpine → sweetGrass).
//!
//! All functions are gated behind `#[cfg(feature = "primalspring")]`.

use crate::error::IpcError;

/// Store model weights via `nest.store` signal dispatch (Wave 17).
///
/// When running inside a biomeOS composition, this sends a single
/// `nest.store` signal that biomeOS decomposes into:
/// `NestGate.content.put → rhizoCrypt.dag.event.append → loamSpine.spine.seal → sweetGrass.braid.create`
///
/// Returns the composed result including provenance artifacts.
///
/// # Trio semantics
///
/// The provenance trio commit is **not atomic**. Partial completion
/// (e.g. DAG without braid) is valid. Callers MUST treat errors as
/// non-fatal — science logic must never fail due to provenance.
///
/// # Errors
///
/// Returns an error if the file cannot be read or signal dispatch fails.
/// Callers should handle this gracefully (log and continue).
#[cfg(feature = "primalspring")]
pub fn store_to_nestgate_signal(
    path: &std::path::Path,
    ctx: &mut primalspring::composition::CompositionContext,
    author: &str,
) -> Result<serde_json::Value, IpcError> {
    use base64::Engine;

    let raw = std::fs::read(path)
        .map_err(|e| IpcError::Other(format!("read {}: {e}", path.display())))?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&raw);

    ctx.dispatch(
        "nest.store",
        serde_json::json!({
            "content": encoded,
            "content_type": "application/x-safetensors",
            "author": author,
            "filename": path.file_name().and_then(|n| n.to_str()).unwrap_or("weights.safetensors"),
        }),
    )
    .map_err(|e| IpcError::Other(format!("nest.store dispatch: {e}")))
}

/// Commit a provenance session via `nest.commit` signal dispatch.
///
/// Finalizes a rhizoCrypt session: dehydrates the DAG, signs via bearDog,
/// optionally stores via NestGate, commits to loamSpine ledger, and
/// creates a sweetGrass attribution braid. biomeOS manages the graph:
/// `rhizoCrypt.event.append → bearDog.crypto.sign → nestGate.content.put → loamSpine.session.commit → sweetGrass.braid.create`
///
/// Use after one or more `nest.store` dispatches to seal a training or
/// experiment session into permanent provenance.
///
/// # Trio semantics
///
/// Not atomic — DAG sessions are append-only with no rollback.
/// Partial state (DAG without spine/braid) is valid provenance.
/// Callers MUST treat errors as non-fatal.
///
/// # Errors
///
/// Returns an error if signal dispatch fails or biomeOS is unavailable.
/// Callers should handle this gracefully (log and continue).
#[cfg(feature = "primalspring")]
pub fn commit_session_signal(
    ctx: &mut primalspring::composition::CompositionContext,
    session_id: &str,
) -> Result<serde_json::Value, IpcError> {
    ctx.dispatch(
        "nest.commit",
        serde_json::json!({ "session_id": session_id }),
    )
    .map_err(|e| IpcError::Other(format!("nest.commit dispatch: {e}")))
}

/// Store a science computation result with provenance via `nest.store`.
///
/// Wraps a JSON result from a science method (e.g. `science.spectral_analysis`)
/// in the full provenance chain. biomeOS decomposes into:
/// `NestGate.content.put → rhizoCrypt.dag.event.append → loamSpine.spine.seal → sweetGrass.braid.create`
///
/// # Trio semantics
///
/// Partial trio completion is valid — see `PROVENANCE_TRIO_INTEGRATION_GUIDE.md`.
/// Callers MUST treat errors as non-fatal; science results are valid
/// regardless of provenance recording success.
///
/// # Errors
///
/// Returns an error if signal dispatch fails or biomeOS is unavailable.
/// Callers should handle this gracefully (log and continue).
#[cfg(feature = "primalspring")]
pub fn store_science_result(
    ctx: &mut primalspring::composition::CompositionContext,
    method: &str,
    result: &serde_json::Value,
    author: &str,
) -> Result<serde_json::Value, IpcError> {
    let content = serde_json::to_string(result)
        .map_err(|e| IpcError::Other(format!("serialize science result: {e}")))?;

    ctx.dispatch(
        "nest.store",
        serde_json::json!({
            "content": content,
            "content_type": "application/json",
            "author": author,
            "metadata": {
                "method": method,
                "domain": "science",
                "spring": "neuralSpring",
            },
        }),
    )
    .map_err(|e| IpcError::Other(format!("nest.store science result: {e}")))
}
