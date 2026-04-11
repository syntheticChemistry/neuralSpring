// SPDX-License-Identifier: AGPL-3.0-or-later

//! MCP tool definitions for the full neuralSpring capability surface.
//!
//! Each tool has a name, description, and JSON Schema for input parameters.
//! These definitions are registered with Squirrel via `capability.announce`
//! by the MCP adapter binary.  The set covers science, health, inference,
//! provenance, cross-primal routing, niche deployment, and compute offload.

use serde::Serialize;
use serde_json::json;

/// Metadata for one MCP tool exposed to Squirrel (`capability.announce` / MCP adapter).
#[derive(Debug, Clone, Serialize)]
pub struct McpToolDef {
    /// Stable tool id matching the neuralSpring JSON-RPC capability name.
    pub name: &'static str,
    /// Short description shown to MCP clients and operators.
    pub description: &'static str,
    /// Logical grouping (e.g. `science`, `health`) for UI and policy.
    pub domain: &'static str,
    /// JSON Schema for tool arguments, serialized for MCP registration.
    pub input_schema: serde_json::Value,
}

/// All neuralSpring capabilities as MCP tool definitions.
#[must_use]
#[expect(clippy::too_many_lines, reason = "27 tool definitions in one registry")]
pub fn tool_definitions() -> Vec<McpToolDef> {
    vec![
        McpToolDef {
            name: "science.spectral_analysis",
            description: "Spectral analysis of weight matrices: eigenvalue distribution, \
                          Marchenko-Pastur fit, bulk ratio, and localization metrics",
            domain: "science",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "matrix": { "type": "array", "description": "Flat weight matrix (row-major)" },
                    "rows": { "type": "integer", "description": "Number of rows" },
                    "cols": { "type": "integer", "description": "Number of columns" }
                },
                "required": ["matrix", "rows", "cols"]
            }),
        },
        McpToolDef {
            name: "science.anderson_localization",
            description: "Anderson localization analysis: disorder sweep, IPR, level spacing \
                          ratio, and localization length for 1D tight-binding Hamiltonians",
            domain: "science",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "size": { "type": "integer", "description": "Lattice size" },
                    "disorder": { "type": "number", "description": "Disorder strength W" },
                    "samples": { "type": "integer", "description": "Disorder realizations" }
                },
                "required": ["size", "disorder"]
            }),
        },
        McpToolDef {
            name: "science.hessian_eigen",
            description: "Hessian eigenanalysis: eigenvalue spectrum of the loss landscape \
                          Hessian for neural network training diagnostics",
            domain: "science",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "hessian": { "type": "array", "description": "Flat Hessian matrix" },
                    "size": { "type": "integer", "description": "Matrix dimension" }
                },
                "required": ["hessian", "size"]
            }),
        },
        McpToolDef {
            name: "science.agent_coordination",
            description: "Multi-agent coordination metrics: graph Laplacian, Fiedler value, \
                          cooperation index from interaction matrices",
            domain: "science",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "adjacency": { "type": "array", "description": "Flat adjacency matrix" },
                    "agents": { "type": "integer", "description": "Number of agents" }
                },
                "required": ["adjacency", "agents"]
            }),
        },
        McpToolDef {
            name: "science.ipr",
            description: "Inverse Participation Ratio for eigenvector localization measurement",
            domain: "science",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "eigenvector": { "type": "array", "description": "Eigenvector components" }
                },
                "required": ["eigenvector"]
            }),
        },
        McpToolDef {
            name: "science.disorder_sweep",
            description: "Sweep disorder strength and compute localization metrics at each point",
            domain: "science",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "size": { "type": "integer", "description": "Lattice size" },
                    "disorder_min": { "type": "number" },
                    "disorder_max": { "type": "number" },
                    "steps": { "type": "integer" }
                },
                "required": ["size", "disorder_min", "disorder_max", "steps"]
            }),
        },
        McpToolDef {
            name: "science.training_trajectory",
            description: "Analyze neural network training trajectory: loss curve, gradient norms, \
                          spectral evolution across epochs",
            domain: "science",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "trajectory": { "type": "array", "description": "Per-epoch metrics" }
                },
                "required": ["trajectory"]
            }),
        },
        McpToolDef {
            name: "science.evoformer_block",
            description: "Execute an Evoformer block (MSA attention + pair attention) for \
                          protein structure prediction validation",
            domain: "science",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "msa": { "type": "array", "description": "MSA representation" },
                    "pair": { "type": "array", "description": "Pair representation" }
                },
                "required": ["msa", "pair"]
            }),
        },
        McpToolDef {
            name: "science.structure_module",
            description: "Run the structure module for 3D coordinate prediction from \
                          single representation",
            domain: "science",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "single": { "type": "array", "description": "Single representation" }
                },
                "required": ["single"]
            }),
        },
        McpToolDef {
            name: "science.folding_health",
            description: "Health check for the protein folding pipeline: GPU availability, \
                          model readiness, and validation status",
            domain: "science",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        McpToolDef {
            name: "science.gpu_dispatch",
            description: "Route arbitrary GPU operations through the neuralSpring Dispatcher: \
                          matmul, eigensolve, reduction, etc.",
            domain: "science",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "op": { "type": "string", "description": "Operation name" },
                    "params": { "type": "object", "description": "Operation-specific parameters" }
                },
                "required": ["op"]
            }),
        },
        McpToolDef {
            name: "science.cross_spring_provenance",
            description: "Query cross-spring provenance: which Python baselines, barraCuda \
                          primitives, and validation binaries exist for each experiment",
            domain: "science",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        McpToolDef {
            name: "science.cross_spring_benchmark",
            description: "Run cross-spring benchmarks: Python vs Rust vs GPU performance \
                          comparison across domains",
            domain: "science",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "domain": { "type": "string", "description": "Benchmark domain filter" }
                }
            }),
        },
        McpToolDef {
            name: "science.precision_routing",
            description: "Query precision routing advice: f32/f64/df64 strategy for the \
                          current GPU hardware profile",
            domain: "science",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        McpToolDef {
            name: "health.liveness",
            description: "Liveness probe: returns immediately if the primal process is alive",
            domain: "health",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        McpToolDef {
            name: "health.readiness",
            description: "Readiness probe: reports subsystem status (dispatcher, GPU backend) \
                          and uptime",
            domain: "health",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        McpToolDef {
            name: "inference.complete",
            description: "Text completion via Squirrel → provider chain. Returns generated \
                          text, token count, and provider metadata.",
            domain: "inference",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "prompt": { "type": "string", "description": "Input text for completion" },
                    "max_tokens": { "type": "integer", "description": "Maximum tokens to generate (default 256)" },
                    "temperature": { "type": "number", "description": "Sampling temperature 0.0-1.0 (default 0.7)" },
                    "model": { "type": "string", "description": "Optional model identifier" }
                },
                "required": ["prompt"]
            }),
        },
        McpToolDef {
            name: "inference.embed",
            description: "Text embedding via Squirrel → provider chain. Returns a dense \
                          vector representation of the input text.",
            domain: "inference",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "Text to embed" },
                    "model": { "type": "string", "description": "Optional embedding model identifier" }
                },
                "required": ["text"]
            }),
        },
        McpToolDef {
            name: "inference.models",
            description: "List available inference models and their capabilities \
                          (complete, embed, context length, parameter count).",
            domain: "inference",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        // ── Provenance tracking (biomeOS trio composition) ───────────
        McpToolDef {
            name: "provenance.begin",
            description: "Begin a provenance session for an experiment or niche deploy. \
                          Acknowledges on this niche; full DAG lifecycle is composed via \
                          biomeOS graphs (rhizoCrypt → loamSpine → sweetGrass).",
            domain: "provenance",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "experiment_id": { "type": "string", "description": "Experiment or session identifier" },
                    "agent": { "type": "string", "description": "Initiating agent / primal name" }
                },
                "required": ["experiment_id"]
            }),
        },
        McpToolDef {
            name: "provenance.record",
            description: "Record a provenance event within an active session.",
            domain: "provenance",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string", "description": "Active session identifier" },
                    "event": { "type": "object", "description": "Event payload to record" }
                },
                "required": ["session_id", "event"]
            }),
        },
        McpToolDef {
            name: "provenance.complete",
            description: "Complete and seal a provenance session.",
            domain: "provenance",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string", "description": "Session to complete" }
                },
                "required": ["session_id"]
            }),
        },
        McpToolDef {
            name: "provenance.status",
            description: "Query the status of a provenance session or the provenance subsystem.",
            domain: "provenance",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string", "description": "Optional session to query" }
                }
            }),
        },
        // ── Cross-primal routing ─────────────────────────────────────
        McpToolDef {
            name: "primal.forward",
            description: "Forward a JSON-RPC request to another primal via biomeOS capability \
                          routing or direct socket discovery.",
            domain: "primal",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "primal": { "type": "string", "description": "Target primal name or capability" },
                    "method": { "type": "string", "description": "JSON-RPC method to call on the target" },
                    "params": { "type": "object", "description": "Parameters for the forwarded call" }
                },
                "required": ["primal", "method"]
            }),
        },
        McpToolDef {
            name: "primal.discover",
            description: "Advertise this niche's capability surface for cross-primal discovery. \
                          Returns primal name, niche, and full capability list.",
            domain: "primal",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        // ── Niche deployment surface ─────────────────────────────────
        McpToolDef {
            name: "capability.list",
            description: "List all capabilities advertised by this neuralSpring niche. \
                          Used by biomeOS, Songbird, and composition validators.",
            domain: "capability",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        McpToolDef {
            name: "compute.offload",
            description: "Node Atomic compute offload: reports GPU dispatcher readiness and \
                          routes compute workloads through the neuralSpring Dispatcher.",
            domain: "compute",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "op": { "type": "string", "description": "Compute operation to offload" },
                    "params": { "type": "object", "description": "Operation-specific parameters" }
                }
            }),
        },
    ]
}

pub use neural_spring::config::ALL_CAPABILITIES;

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test assertions — tool definitions are known-valid"
)]
mod tests {
    use super::*;

    #[test]
    fn tool_count_matches_capabilities() {
        let tools = tool_definitions();
        assert_eq!(
            tools.len(),
            ALL_CAPABILITIES.len(),
            "tool_definitions() and ALL_CAPABILITIES must have same count"
        );
        assert_eq!(tools.len(), 27);
    }

    #[test]
    fn tool_names_match_capabilities() {
        let tools = tool_definitions();
        let tool_names: Vec<&str> = tools.iter().map(|t| t.name).collect();
        for cap in ALL_CAPABILITIES {
            assert!(
                tool_names.contains(cap),
                "ALL_CAPABILITIES entry {cap} missing from tool_definitions()"
            );
        }
    }

    #[test]
    fn all_tools_have_valid_domain() {
        let valid_domains = [
            "science",
            "health",
            "inference",
            "provenance",
            "primal",
            "capability",
            "compute",
        ];
        for tool in tool_definitions() {
            assert!(
                valid_domains.contains(&tool.domain),
                "tool {} has unexpected domain '{}'",
                tool.name,
                tool.domain
            );
        }
    }

    #[test]
    fn all_tools_have_valid_schema() {
        for tool in tool_definitions() {
            assert!(!tool.name.is_empty(), "tool name must not be empty");
            assert!(
                !tool.description.is_empty(),
                "tool {} must have a description",
                tool.name
            );
            assert_eq!(
                tool.input_schema["type"], "object",
                "tool {} schema must be type: object",
                tool.name
            );
            assert!(
                tool.input_schema.get("properties").is_some(),
                "tool {} schema must have 'properties'",
                tool.name
            );
        }
    }

    #[test]
    fn tool_names_start_with_domain_prefix() {
        for tool in tool_definitions() {
            let expected_prefix = format!("{}.", tool.domain);
            assert!(
                tool.name.starts_with(&expected_prefix),
                "tool name '{}' must start with '{}'",
                tool.name,
                expected_prefix
            );
        }
    }

    #[test]
    fn tools_serialize_to_json() {
        for tool in tool_definitions() {
            let json = serde_json::to_value(&tool).expect("tool must serialize");
            assert!(json.is_object());
            assert!(json["name"].is_string());
            assert!(json["input_schema"].is_object());
        }
    }
}
