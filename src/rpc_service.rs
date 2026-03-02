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
    pub eigenvalues: Vec<f64>,
    pub mean_ipr: f64,
    pub level_spacing_ratio: f64,
    pub bandwidth: f64,
    pub condition_number: f64,
    pub phase: String,
}

/// Disorder sweep result from `science.disorder_sweep`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DisorderSweepResult {
    pub disorder_values: Vec<f64>,
    pub ipr_values: Vec<f64>,
}

/// Health status report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub primal: String,
    pub version: String,
    pub capabilities: Vec<String>,
    pub requests_served: u64,
    pub uptime_seconds: u64,
    pub gpu_available: bool,
}

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
