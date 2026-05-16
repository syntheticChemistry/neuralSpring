// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical capability method names used across the IPC surface.
//!
//! These constants eliminate hardcoded string literals when constructing
//! JSON-RPC requests or registering capabilities. Each constant maps to
//! a single JSON-RPC method name as defined by the owning primal.
//!
//! neuralSpring only knows *what* it needs (a capability), not *who*
//! provides it. Discovery handles the routing.

// ─── barraCuda surface (stats + tensor + precision) ─────────────
/// Statistical mean.
pub const STATS_MEAN: &str = "stats.mean";
/// Standard deviation.
pub const STATS_STD_DEV: &str = "stats.std_dev";
/// Weighted mean.
pub const STATS_WEIGHTED_MEAN: &str = "stats.weighted_mean";
/// Matrix multiplication.
pub const TENSOR_MATMUL: &str = "tensor.matmul";
/// Tensor creation.
pub const TENSOR_CREATE: &str = "tensor.create";
/// Domain-specific precision routing (Tier 2 Science API).
pub const PRECISION_ROUTE: &str = "barracuda.precision.route";

// ─── toadStool surface (compute + workload) ─────────────────────
/// General compute dispatch.
pub const COMPUTE_DISPATCH: &str = "compute.dispatch";
/// Compute offload.
pub const COMPUTE_OFFLOAD: &str = "compute.offload";
/// Workload pre-flight validation (Tier 2 Science API).
pub const TOADSTOOL_VALIDATE: &str = "toadstool.validate";
/// List available workloads.
pub const TOADSTOOL_LIST_WORKLOADS: &str = "toadstool.list_workloads";

// ─── BearDog surface (crypto) ───────────────────────────────────
/// Cryptographic hashing.
pub const CRYPTO_HASH: &str = "crypto.hash";

// ─── Squirrel surface (inference) ───────────────────────────────
/// Text completion.
pub const INFERENCE_COMPLETE: &str = "inference.complete";
/// Text embedding.
pub const INFERENCE_EMBED: &str = "inference.embed";
/// List available models.
pub const INFERENCE_MODELS: &str = "inference.models";

// ─── coralReef surface (shader compilation) ─────────────────────
/// Compile WGSL shader source.
pub const SHADER_COMPILE_WGSL: &str = "shader.compile.wgsl";
/// Query shader compilation capabilities.
pub const SHADER_COMPILE_CAPABILITIES: &str = "shader.compile.capabilities";

// ─── skunkBat surface (security) ────────────────────────────────
/// Audit event logging.
pub const SECURITY_AUDIT_LOG: &str = "security.audit_log";

// ─── NestGate surface (storage + content) ───────────────────────
/// Store content-addressed data (BLAKE3 hash-as-key).
pub const CONTENT_PUT: &str = "content.put";
/// Retrieve content-addressed data by BLAKE3 hash.
pub const CONTENT_GET: &str = "content.get";
/// Check whether content-addressed data exists.
pub const CONTENT_EXISTS: &str = "content.exists";

// ─── petalTongue surface (visualization) ────────────────────────
/// Render a visualization frame.
pub const VISUALIZATION_RENDER: &str = "visualization.render";
/// Stream visualization data.
pub const VISUALIZATION_RENDER_STREAM: &str = "visualization.render.stream";
/// Query visualization capabilities.
pub const VISUALIZATION_CAPABILITIES: &str = "visualization.capabilities";

// ─── biomeOS composition ────────────────────────────────────────
/// Health liveness probe.
pub const HEALTH_LIVENESS: &str = "health.liveness";
/// Health readiness probe.
pub const HEALTH_READINESS: &str = "health.readiness";
/// Health check.
pub const HEALTH_CHECK: &str = "health.check";
/// List capabilities.
pub const CAPABILITY_LIST: &str = "capability.list";
/// Identity query.
pub const IDENTITY_GET: &str = "identity.get";
/// MCP tools listing.
pub const MCP_TOOLS_LIST: &str = "mcp.tools.list";
/// Composition status.
pub const COMPOSITION_STATUS: &str = "composition.status";
/// Method registration (legacy).
pub const METHOD_REGISTER: &str = "method.register";
/// Primal announcement (Wave 17 signal API — replaces multi-call registration).
pub const PRIMAL_ANNOUNCE: &str = "primal.announce";

// ─── Provenance trio ────────────────────────────────────────────
/// Begin provenance session.
pub const PROVENANCE_BEGIN: &str = "provenance.begin";
/// Record provenance step.
pub const PROVENANCE_RECORD: &str = "provenance.record";
/// Complete provenance session.
pub const PROVENANCE_COMPLETE: &str = "provenance.complete";
/// Provenance status query.
pub const PROVENANCE_STATUS: &str = "provenance.status";

// ─── Cross-primal ───────────────────────────────────────────────
/// Forward a request to another primal.
pub const PRIMAL_FORWARD: &str = "primal.forward";
/// Discover available primals.
pub const PRIMAL_DISCOVER: &str = "primal.discover";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_capabilities_are_dotted() {
        let caps = [
            STATS_MEAN, STATS_STD_DEV, STATS_WEIGHTED_MEAN,
            TENSOR_MATMUL, TENSOR_CREATE, PRECISION_ROUTE,
            COMPUTE_DISPATCH, COMPUTE_OFFLOAD, TOADSTOOL_VALIDATE, TOADSTOOL_LIST_WORKLOADS,
            CRYPTO_HASH,
            INFERENCE_COMPLETE, INFERENCE_EMBED, INFERENCE_MODELS,
            SHADER_COMPILE_WGSL, SHADER_COMPILE_CAPABILITIES,
            SECURITY_AUDIT_LOG,
            CONTENT_PUT, CONTENT_GET, CONTENT_EXISTS,
            VISUALIZATION_RENDER, VISUALIZATION_RENDER_STREAM, VISUALIZATION_CAPABILITIES,
            HEALTH_LIVENESS, HEALTH_READINESS, HEALTH_CHECK,
            CAPABILITY_LIST, IDENTITY_GET, MCP_TOOLS_LIST,
            COMPOSITION_STATUS, METHOD_REGISTER,
            PROVENANCE_BEGIN, PROVENANCE_RECORD, PROVENANCE_COMPLETE, PROVENANCE_STATUS,
            PRIMAL_FORWARD, PRIMAL_DISCOVER,
        ];
        for cap in caps {
            assert!(cap.contains('.'), "{cap} must use dotted notation");
        }
    }
}
