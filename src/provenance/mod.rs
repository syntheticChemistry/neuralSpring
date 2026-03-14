// SPDX-License-Identifier: AGPL-3.0-or-later

//! Provenance metadata for all Python baseline values.
//!
//! Every hardcoded expected value in validation binaries traces back to a
//! specific Python control run. This module centralizes that metadata so
//! validation binaries carry machine-readable provenance.
//!
//! Imitates the hotSpring pattern:
//! ```text
//! Python script → commit → environment → command → output → Rust constant
//! ```
//!
//! ## Data Sources
//!
//! | Dataset | Source | License |
//! |---------|--------|---------|
//! | Benchmark functions | Analytical (Rastrigin, Rosenbrock, Ackley) | N/A |
//! | FAO-56 ET₀ | Allen et al. (1998) FAO Paper 56 | Public |
//! | MNIST | LeCun et al. (1998) via `torchvision` | CC BY-SA 3.0 |
//! | ERA5 weather | Open-Meteo Archive API (ECMWF Copernicus) | CC BY 4.0 |
//! | Burgers PDE | Raissi et al. (2019) JCP, DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045) | N/A |
//! | Antiderivative | Lu et al. (2021) NMI, DOI: [10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5) | N/A |

pub mod experiments;
pub mod references;

pub use experiments::*;
pub use references::*;

/// A single provenance record tying a Rust reference value to its Python origin.
#[derive(Debug, Clone)]
pub struct BaselineProvenance {
    pub label: &'static str,
    pub script: &'static str,
    pub commit: &'static str,
    pub date: &'static str,
    pub command: &'static str,
    pub environment: &'static str,
    pub value: f64,
    pub unit: &'static str,
}

// ═══════════════════════════════════════════════════════════════════
// Environment
// ═══════════════════════════════════════════════════════════════════

/// Python environment for all control runs (frozen at baseline time).
pub const ENVIRONMENT: &str = "Python 3.10.12, PyTorch 2.9.0+cu128, NumPy 2.2.6, SciPy 1.15.3";

/// Hardware for all control runs (frozen at baseline time).
pub const HARDWARE: &str = "Eastgate (i9-12900K, RTX 4070 12GB, Pop!_OS 22.04)";

/// Pinned commit for baseline results (Phase 0+: 75/75 PASS).
pub const BASELINE_COMMIT: &str = "f9ad0268917a335dce2b1175ea0d77add271b25b";

/// Pinned date for baseline results.
pub const BASELINE_DATE: &str = "2026-02-16";

/// Pinned date for publication experiment baselines (Exp-050/052/053).
pub const PUBLICATION_BASELINE_DATE: &str = "2026-02-26";

/// Pinned date for baseCamp nS-06 (immunological Anderson) baselines.
///
/// Extended experiments ran after the Phase 0 baseline freeze.
pub const NS06_BASELINE_DATE: &str = "2026-03-02";

/// Pinned commit for CPU parity references (generated post-baseline freeze).
///
/// `cpu_parity_references.json` was regenerated on 2026-03-05 at a later
/// commit than the Phase 0 baseline freeze.  `NumPy` 2.1.3 (vs 2.2.6 for
/// Phase 0).  Values are stable: cross-language tolerance (`CROSS_LANGUAGE`)
/// absorbs the version difference.
pub const CPU_PARITY_COMMIT: &str = "359fc1d815791e8269904484ffd76e3d10f2bba6";

/// Pinned date for CPU parity references.
pub const CPU_PARITY_DATE: &str = "2026-03-05";

/// Environment for CPU parity references.
pub const CPU_PARITY_ENVIRONMENT: &str = "Python 3.10.12, NumPy 2.1.3";

/// Pinned environment for publication experiment baselines.
pub const PUBLICATION_ENVIRONMENT: &str = "Python 3.12, PyTorch 2.9.0+cu128, NumPy, seed=42";

/// Runtime-detected execution environment.
///
/// Discovers Rust version, OS, and architecture at runtime rather than
/// relying on hardcoded strings. Each primal carries self-knowledge.
#[derive(Debug, Clone)]
pub struct RuntimeEnvironment {
    pub rust_version: String,
    pub os: String,
    pub arch: String,
    pub neuralspring_version: String,
}

impl RuntimeEnvironment {
    /// Discover the current execution environment.
    ///
    /// All fields are derived from compile-time or runtime introspection —
    /// no hardcoded strings.  Primal self-knowledge only.
    #[must_use]
    pub fn discover() -> Self {
        Self {
            rust_version: format!("rustc {}", env!("CARGO_PKG_RUST_VERSION", "unknown"),),
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            neuralspring_version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }

    /// Summary string for logging and provenance records.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} v{} | {} {} | {}",
            env!("CARGO_PKG_NAME"),
            self.neuralspring_version,
            self.os,
            self.arch,
            self.rust_version
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_reference_sums_near_one() {
        let sum: f64 = SOFTMAX_1_TO_5.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14);
    }

    #[test]
    #[expect(
        clippy::expect_used,
        reason = "test verifies known reference point exists"
    )]
    fn gelu_zero_is_zero() {
        let (_, y) = GELU_REFERENCE
            .iter()
            .find(|(x, _)| *x == 0.0)
            .expect("GELU_REFERENCE must contain x=0.0");
        assert!(y.abs() < 1e-15);
    }

    #[test]
    fn provenance_records_non_empty() {
        let records = [
            &SURROGATE_PROVENANCE,
            &TRANSFORMER_PROVENANCE,
            &SEQUENCE_PROVENANCE,
            &TRANSFER_PROVENANCE,
            &ISOMORPHIC_PROVENANCE,
            &PINN_PROVENANCE,
            &DEEPONET_PROVENANCE,
            &LENET_PROVENANCE,
            &LSTM_ERA5_PROVENANCE,
            &QUANTIZED_PROVENANCE,
            &COUNTERDIABATIC_PROVENANCE,
            &MODES_PROVENANCE,
            &ECO_DYNAMICS_PROVENANCE,
            &DIRECTED_EVOLUTION_PROVENANCE,
            &HMM_PROVENANCE,
            &GAME_THEORY_PROVENANCE,
            &SWARM_ROBOTICS_PROVENANCE,
            &SATE_ALIGNMENT_PROVENANCE,
            &INTROGRESSION_PROVENANCE,
            &REGULATORY_NETWORK_PROVENANCE,
            &SIGNAL_INTEGRATION_PROVENANCE,
            &SPECTRAL_COMMUTATIVITY_PROVENANCE,
            &ANDERSON_LOCALIZATION_PROVENANCE,
            &ML_INFERENCE_PROVENANCE,
            &WDM_TRANSPORT_PROVENANCE,
            &WDM_EOS_PROVENANCE,
            &WDM_SQW_PROVENANCE,
            &WDM_TRANSFER_PROVENANCE,
            &WDM_ESN_PROVENANCE,
            &PANGENOME_SELECTION_PROVENANCE,
            &META_POPULATION_PROVENANCE,
            &CORAL_FORGE_PROVENANCE,
            &ALPHAFOLD2_EVOFORMER_PROVENANCE,
            &ALPHAFOLD3_DIFFUSION_PROVENANCE,
            &ALPHAFOLD3_PAIRFORMER_PROVENANCE,
            &ALPHAFOLD3_CONFIDENCE_PROVENANCE,
            &TRAINING_TRAJECTORY_PROVENANCE,
            &HESSIAN_EIGENANALYSIS_PROVENANCE,
            &ANDERSON_MULTIAGENT_PROVENANCE,
            &CPU_PARITY_PROVENANCE,
            &IMMUNOLOGICAL_ANDERSON_PROVENANCE,
            &IMMUNOLOGICAL_ANDERSON_EXTENDED_PROVENANCE,
            &DIGESTION_PREDICTION_PROVENANCE,
            &DIGESTER_ANDERSON_PROVENANCE,
            &ISOMORPHIC_RESERVOIR_PROVENANCE,
            &WDM_ENSEMBLE_QS_PROVENANCE,
            &INTROGRESSION_NN_PROVENANCE,
            &ATTENTION_ANDERSON_PROVENANCE,
        ];
        let valid_commits = [BASELINE_COMMIT, CPU_PARITY_COMMIT];
        for p in records {
            assert!(!p.label.is_empty(), "empty label: {}", p.script);
            assert!(!p.script.is_empty());
            assert!(!p.date.is_empty());
            assert!(!p.command.is_empty());
            assert!(
                valid_commits.contains(&p.commit),
                "{}: commit {} not in valid set",
                p.label,
                p.commit,
            );
        }
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "global minima are exactly 0.0 by mathematical definition"
    )]
    fn benchmark_references_have_global_minima() {
        assert!(RASTRIGIN_REFERENCE
            .iter()
            .any(|(x, y, _)| *x == 1.0 && *y == 1.0));
        assert!(ROSENBROCK_REFERENCE.iter().any(|(_, _, f)| *f == 0.0));
    }

    #[test]
    fn runtime_environment_discovery() {
        let env = RuntimeEnvironment::discover();
        assert!(!env.os.is_empty());
        assert!(!env.arch.is_empty());
        assert!(!env.neuralspring_version.is_empty());
        let summary = env.summary();
        assert!(summary.contains(env!("CARGO_PKG_NAME")));
        assert!(summary.contains(&env.os));
    }
}
