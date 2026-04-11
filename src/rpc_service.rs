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
