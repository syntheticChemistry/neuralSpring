// SPDX-License-Identifier: AGPL-3.0-or-later

//! tarpc service definition for neuralSpring.
//!
//! Complements the JSON-RPC 2.0 transport with a typed Rust-native RPC
//! interface, per wateringHole `UNIVERSAL_IPC_STANDARD_V3.md`.
//!
//! ## Protocol negotiation
//!
//! The primal binary listens on a Unix socket speaking newline-delimited
//! JSON-RPC.  When a client connects with the tarpc bincode framing, the
//! connection is upgraded to tarpc automatically (future work — currently
//! this module defines the service trait for use by Rust-native callers).
//!
//! ## Usage
//!
//! ```ignore
//! use neural_spring::rpc_service::NeuralSpringClient;
//!
//! let client = NeuralSpringClient::new(/* transport */);
//! let ipr = client.ipr(ctx, wavefunction).await?;
//! ```

/// Spectral analysis result from `science.spectral_analysis`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpectralAnalysisResult {
    /// Eigenvalues of the sampled Anderson Hamiltonian.
    pub eigenvalues: Vec<f64>,
    /// Mean inverse participation ratio across eigenstates.
    pub mean_ipr: f64,
    /// Ratio of mean adjacent level spacing to the mean level spacing.
    pub level_spacing_ratio: f64,
    /// Spectral bandwidth of the Hamiltonian.
    pub bandwidth: f64,
    /// Condition number of the eigenproblem (numerical stability).
    pub condition_number: f64,
    /// Labeled localization or spectral phase (human-readable).
    pub phase: String,
}

/// Disorder sweep result from `science.disorder_sweep`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DisorderSweepResult {
    /// Disorder strengths used in the sweep.
    pub disorder_values: Vec<f64>,
    /// Inverse participation ratio at each disorder strength.
    pub ipr_values: Vec<f64>,
}

/// Health status report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthStatus {
    /// Overall health label (e.g. ok or degraded).
    pub status: String,
    /// Name or identifier of the running primal binary.
    pub primal: String,
    /// Build or crate version string.
    pub version: String,
    /// Advertised capability identifiers.
    pub capabilities: Vec<String>,
    /// Total RPC requests served since startup.
    pub requests_served: u64,
    /// Process uptime in seconds.
    pub uptime_seconds: u64,
    /// Whether a GPU backend is available to the service.
    pub gpu_available: bool,
}

// ═══════════════════════════════════════════════════════════════════
// Inference capability wire types (proto-nucleate: inference.*)
// ═══════════════════════════════════════════════════════════════════

/// Request payload for `inference.complete`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceCompleteRequest {
    /// The prompt or input text for completion.
    pub prompt: String,
    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Sampling temperature (0.0 = deterministic, 1.0 = creative).
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Optional model identifier (routes to specific provider).
    pub model: Option<String>,
}

const fn default_max_tokens() -> u32 {
    256
}

const fn default_temperature() -> f64 {
    0.7
}

/// Response payload for `inference.complete`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceCompleteResponse {
    /// Generated text.
    pub text: String,
    /// Number of tokens in the completion.
    pub tokens_generated: u32,
    /// Model that served the request.
    pub model: String,
    /// Provider that fulfilled the request (e.g. "squirrel", "ollama", "native-wgsl").
    pub provider: String,
    /// Whether the response was truncated at `max_tokens`.
    pub truncated: bool,
}

/// Request payload for `inference.embed`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceEmbedRequest {
    /// Text to embed.
    pub text: String,
    /// Optional model identifier for the embedding model.
    pub model: Option<String>,
}

/// Response payload for `inference.embed`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceEmbedResponse {
    /// The embedding vector.
    pub embedding: Vec<f64>,
    /// Dimensionality of the embedding.
    pub dimensions: usize,
    /// Model that produced the embedding.
    pub model: String,
    /// Provider that fulfilled the request.
    pub provider: String,
}

/// Response payload for `inference.models`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceModelsResponse {
    /// Available models with metadata.
    pub models: Vec<ModelInfo>,
    /// The provider serving these models.
    pub provider: String,
}

/// Metadata for a single available model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelInfo {
    /// Model identifier (used in `model` field of requests).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Capabilities: "complete", "embed", or both.
    pub capabilities: Vec<String>,
    /// Parameter count (if known).
    pub parameters: Option<u64>,
    /// Context window size in tokens (if known).
    pub context_length: Option<u32>,
}

/// Typed tarpc surface for health, IPR, disorder sweeps, inference, and capabilities.
#[tarpc::service]
pub trait NeuralSpring {
    /// Health check — returns primal status and capability list.
    async fn health() -> HealthStatus;

    /// Inverse participation ratio of a wavefunction.
    async fn ipr(wavefunction: Vec<f64>) -> f64;

    /// Disorder sweep: IPR as a function of disorder strength.
    async fn disorder_sweep(
        lattice_size: usize,
        hopping: f64,
        disorder_values: Vec<f64>,
        seed: u64,
    ) -> DisorderSweepResult;

    /// Full spectral analysis of a random Anderson Hamiltonian.
    async fn spectral_analysis(dim: usize, disorder: f64, seed: u64) -> SpectralAnalysisResult;

    /// Text completion via Squirrel → provider chain.
    async fn inference_complete(request: InferenceCompleteRequest) -> InferenceCompleteResponse;

    /// Text embedding via Squirrel → provider chain.
    async fn inference_embed(request: InferenceEmbedRequest) -> InferenceEmbedResponse;

    /// List available inference models.
    async fn inference_models() -> InferenceModelsResponse;

    /// List advertised capabilities (sovereign discovery).
    async fn capability_list() -> Vec<String>;
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test serialization roundtrips")]
mod tests {
    use super::*;

    #[test]
    fn spectral_result_serde_roundtrip() {
        let result = SpectralAnalysisResult {
            eigenvalues: vec![-1.0, 0.0, 1.0],
            mean_ipr: 0.25,
            level_spacing_ratio: 0.53,
            bandwidth: 2.0,
            condition_number: 1.5,
            phase: "Extended".to_string(),
        };
        let json = serde_json::to_string(&result).expect("serialize");
        let deser: SpectralAnalysisResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.eigenvalues.len(), 3);
        assert_eq!(deser.phase, "Extended");
    }

    #[test]
    fn disorder_sweep_serde_roundtrip() {
        let sweep = DisorderSweepResult {
            disorder_values: vec![0.5, 1.0, 2.0],
            ipr_values: vec![0.1, 0.3, 0.8],
        };
        let json = serde_json::to_string(&sweep).expect("serialize");
        let deser: DisorderSweepResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.disorder_values.len(), 3);
    }

    #[test]
    fn health_status_serde_roundtrip() {
        let status = HealthStatus {
            status: "ok".to_string(),
            primal: "neuralspring".to_string(),
            version: "0.1.0".to_string(),
            capabilities: vec!["science.ipr".to_string()],
            requests_served: 42,
            uptime_seconds: 3600,
            gpu_available: false,
        };
        let json = serde_json::to_string(&status).expect("serialize");
        let deser: HealthStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.primal, "neuralspring");
        assert_eq!(deser.requests_served, 42);
    }

    #[test]
    fn inference_request_defaults() {
        let json = r#"{"prompt":"test"}"#;
        let req: InferenceCompleteRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.prompt, "test");
        assert_eq!(req.max_tokens, 256);
        assert!((req.temperature - 0.7).abs() < f64::EPSILON);
        assert!(req.model.is_none());
    }

    #[test]
    fn inference_response_serde() {
        let resp = InferenceCompleteResponse {
            text: "hello".to_string(),
            tokens_generated: 1,
            model: "test-model".to_string(),
            provider: "squirrel".to_string(),
            truncated: false,
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(json.contains("squirrel"));
    }

    #[test]
    fn embed_request_serde() {
        let req = InferenceEmbedRequest {
            text: "test text".to_string(),
            model: Some("e5".to_string()),
        };
        let json = serde_json::to_string(&req).expect("serialize");
        let deser: InferenceEmbedRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.model, Some("e5".to_string()));
    }

    #[test]
    fn model_info_optional_fields() {
        let json = r#"{"id":"m1","name":"Model 1","capabilities":["complete"]}"#;
        let info: ModelInfo = serde_json::from_str(json).expect("deserialize");
        assert!(info.parameters.is_none());
        assert!(info.context_length.is_none());
    }

    #[test]
    fn models_response_serde() {
        let resp = InferenceModelsResponse {
            models: vec![ModelInfo {
                id: "m1".to_string(),
                name: "Model 1".to_string(),
                capabilities: vec!["complete".to_string(), "embed".to_string()],
                parameters: Some(7_000_000_000),
                context_length: Some(4096),
            }],
            provider: "squirrel".to_string(),
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        let deser: InferenceModelsResponse = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.models.len(), 1);
        assert_eq!(deser.models[0].parameters, Some(7_000_000_000));
    }
}
