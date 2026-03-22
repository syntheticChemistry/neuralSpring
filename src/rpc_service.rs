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

/// Typed tarpc surface for health, IPR, disorder sweeps, and capabilities.
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

    /// List advertised capabilities (sovereign discovery).
    async fn capability_list() -> Vec<String>;
}
