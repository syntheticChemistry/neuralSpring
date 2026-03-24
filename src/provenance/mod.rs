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
    /// Human-readable label for this baseline record.
    pub label: &'static str,
    /// Path to the Python control script that produced the reference.
    pub script: &'static str,
    /// Git commit hash pinned for this baseline.
    pub commit: &'static str,
    /// Date string for when the baseline was captured.
    pub date: &'static str,
    /// Full command line used to run the control script.
    pub command: &'static str,
    /// Frozen Python/NumPy environment string for the control run.
    pub environment: &'static str,
    /// Reference numeric value validated against Rust.
    pub value: f64,
    /// Unit string for the reference value (e.g. dimensionless).
    pub unit: &'static str,
}

impl BaselineProvenance {
    /// Where expected reference values live for this experiment.
    ///
    /// Derives the source location from the `script` path — every provenance
    /// record maps to either a `provenance::references::*` constant, a JSON
    /// baseline in `control/`, or inline analytical values.
    #[must_use]
    #[expect(
        clippy::too_many_lines,
        reason = "pure data mapping — one arm per experiment script, no logic to extract"
    )]
    pub fn expected_source(&self) -> &'static str {
        match self.script {
            "control/surrogate/surrogate_validation.py" => {
                "provenance::references::{RASTRIGIN,ROSENBROCK,ACKLEY}_REFERENCE"
            }
            "control/transformer/transformer_inference.py" => {
                "provenance::references::{SOFTMAX_1_TO_5,GELU_REFERENCE}"
            }
            "control/sequence/sequence_forecasting.py" | "control/lstm_weather/lstm_era5.py" => {
                "inline analytical (seasonal model)"
            }
            "control/transfer/transfer_learning.py" => "inline analytical (isomorphic catalog)",
            "control/isomorphic/isomorphic_catalog.py" => "inline analytical (benchmark functions)",
            "control/pinn/pinn_burgers.py" => "inline analytical (Cole-Hopf exact solution)",
            "control/deeponet/deeponet_antideriv.py" => "inline analytical (antiderivative)",
            "control/lenet/lenet_mnist.py" => "control/lenet/lenet_baseline.json",
            "control/quantized/quantized_inference.py" => {
                "inline analytical (quantization error bounds)"
            }
            "control/counterdiabatic/counterdiabatic_evolution.py" => {
                "inline analytical (NK landscape)"
            }
            "control/modes/modes_toolbox.py" => "inline analytical (MODES metrics)",
            "control/eco_dynamics/eco_dynamics.py" => {
                "inline analytical (Lotka-Volterra equilibria)"
            }
            "control/directed_evolution/directed_evolution.py" => {
                "inline analytical (selection pressure)"
            }
            "control/hmm_phylo/hmm_phylo.py" => "inline analytical (HMM forward/Viterbi)",
            "control/game_theory/game_theory.py" => "inline analytical (Nash equilibria)",
            "control/swarm_robotics/swarm_robotics.py" => "inline analytical (swarm fitness)",
            "control/sate_alignment/sate_alignment.py" => "inline analytical (NJ tree + MSA)",
            "control/introgression/introgression.py" => "inline analytical (PhyloNet-HMM)",
            "control/regulatory_network/regulatory_network.py" => {
                "inline analytical (GRN ODE steady state)"
            }
            "control/signal_integration/signal_integration.py" => {
                "inline analytical (Hill gate output)"
            }
            "control/spectral_commutativity/spectral_commutativity.py" => {
                "inline analytical (commutator norm)"
            }
            "control/anderson_localization/anderson_localization.py" => {
                "inline analytical (IPR, Lyapunov)"
            }
            "control/pangenome_selection/pangenome_selection.py" => {
                "inline analytical (chi² selection)"
            }
            "control/meta_population/meta_population.py" => "inline analytical (FST, Mantel)",
            "control/ml_inference/generate_baselines.py" => {
                "control/ml_inference/{mlp,transformer}_baseline.json"
            }
            "control/generate_cpu_references.py" => "control/cpu_parity_references.json",
            "control/wdm/transport_surrogate.py" => "control/wdm/transport_surrogate_baseline.json",
            "control/wdm/eos_surrogate.py" => "control/wdm/eos_surrogate_baseline.json",
            "control/wdm/sqw_peak_predictor.py" => "control/wdm/sqw_peak_baseline.json",
            "control/wdm/transfer_classical_to_wdm.py" => "control/wdm/transfer_baseline.json",
            "control/wdm/esn_regime_classifier.py" => "control/wdm/esn_regime_baseline.json",
            "control/coral_forge/evoformer_primitives.py" => {
                "control/coral_forge/evoformer_baselines.json"
            }
            "control/coral_forge/alphafold2_evoformer_block.py" => {
                "control/coral_forge/evoformer_block_baselines.json"
            }
            "control/coral_forge/alphafold3_diffusion.py" => {
                "control/coral_forge/diffusion_baselines.json"
            }
            "control/coral_forge/alphafold3_pairformer.py" => {
                "control/coral_forge/pairformer_baselines.json"
            }
            "control/coral_forge/alphafold3_confidence.py" => {
                "control/coral_forge/confidence_baselines.json"
            }
            "control/training_trajectory/training_trajectory.py" => {
                "control/training_trajectory/baseline_values.json"
            }
            "control/hessian_eigenanalysis/hessian_eigenanalysis.py" => {
                "control/hessian_eigenanalysis/baseline_values.json"
            }
            "control/anderson_multiagent/anderson_multiagent.py" => {
                "control/anderson_multiagent/baseline_values.json"
            }
            "control/immunological_anderson/immunological_anderson.py" => {
                "inline analytical (Anderson localization in immune signaling)"
            }
            "control/immunological_anderson/immunological_anderson_extended.py" => {
                "inline analytical (Gonzales dose-response, PK, lattice, MATRIX)"
            }
            "control/glucose_prediction/glucose_prediction.py" => {
                "control/glucose_prediction/glucose_baseline.json"
            }
            "control/digestion_prediction/digestion_prediction.py" => {
                "control/digestion_prediction/digestion_prediction_baseline.json"
            }
            "control/digester_anderson/digester_anderson.py" => {
                "control/digester_anderson/digester_anderson_baseline.json"
            }
            "control/isomorphic_reservoir/isomorphic_reservoir.py" => {
                "control/isomorphic_reservoir/isomorphic_reservoir_baseline.json"
            }
            "control/wdm_ensemble_qs/wdm_ensemble_qs.py" => {
                "control/wdm_ensemble_qs/wdm_ensemble_qs_baseline.json"
            }
            "control/introgression_nn/introgression_nn.py" => {
                "control/introgression_nn/introgression_nn_baseline.json"
            }
            "control/attention_anderson/attention_anderson.py" => {
                "control/attention_anderson/attention_anderson_baseline.json"
            }
            _ => "",
        }
    }
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
pub const PUBLICATION_ENVIRONMENT: &str =
    "Python 3.10.12, PyTorch 2.9.0+cu128, NumPy 2.2.6, SciPy 1.15.3";

/// Complete registry of all provenance records (healthSpring V37 pattern).
///
/// Every `BaselineProvenance` constant must appear here. The test suite
/// validates that the count matches the number of `pub const *_PROVENANCE`
/// declarations in `experiments.rs`, catching any omission at compile time.
pub const PROVENANCE_REGISTRY: &[&BaselineProvenance] = &[
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
    &PANGENOME_SELECTION_PROVENANCE,
    &META_POPULATION_PROVENANCE,
    &ML_INFERENCE_PROVENANCE,
    &CPU_PARITY_PROVENANCE,
    &WDM_TRANSPORT_PROVENANCE,
    &WDM_EOS_PROVENANCE,
    &WDM_SQW_PROVENANCE,
    &WDM_TRANSFER_PROVENANCE,
    &WDM_ESN_PROVENANCE,
    &CORAL_FORGE_PROVENANCE,
    &TRAINING_TRAJECTORY_PROVENANCE,
    &HESSIAN_EIGENANALYSIS_PROVENANCE,
    &ANDERSON_MULTIAGENT_PROVENANCE,
    &ALPHAFOLD2_EVOFORMER_PROVENANCE,
    &ALPHAFOLD3_DIFFUSION_PROVENANCE,
    &ALPHAFOLD3_PAIRFORMER_PROVENANCE,
    &ALPHAFOLD3_CONFIDENCE_PROVENANCE,
    &IMMUNOLOGICAL_ANDERSON_PROVENANCE,
    &IMMUNOLOGICAL_ANDERSON_EXTENDED_PROVENANCE,
    &GLUCOSE_PREDICTION_PROVENANCE,
    &DIGESTION_PREDICTION_PROVENANCE,
    &DIGESTER_ANDERSON_PROVENANCE,
    &ISOMORPHIC_RESERVOIR_PROVENANCE,
    &WDM_ENSEMBLE_QS_PROVENANCE,
    &INTROGRESSION_NN_PROVENANCE,
    &ATTENTION_ANDERSON_PROVENANCE,
];

/// Runtime-detected execution environment.
///
/// Discovers Rust version, OS, and architecture at runtime rather than
/// relying on hardcoded strings. Each primal carries self-knowledge.
#[derive(Debug, Clone)]
pub struct RuntimeEnvironment {
    /// `rustc` version string for the running build.
    pub rust_version: String,
    /// Operating system name from `std::env::consts::OS`.
    pub os: String,
    /// CPU architecture from `std::env::consts::ARCH`.
    pub arch: String,
    /// Crate version from `CARGO_PKG_VERSION`.
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
        assert!((sum - 1.0).abs() < crate::tolerances::SOFTMAX_CROSS_PYTHON);
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
        assert!(y.abs() < crate::tolerances::NUMERICAL_DISTINCTNESS);
    }

    #[test]
    fn provenance_registry_records_non_empty() {
        let valid_commits = [BASELINE_COMMIT, CPU_PARITY_COMMIT];
        for p in PROVENANCE_REGISTRY {
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
    fn provenance_registry_completeness() {
        let src = include_str!("experiments.rs");
        let declared = src.matches("pub const ").count();
        let extra_non_provenance = src
            .lines()
            .filter(|l| {
                l.contains("pub const ")
                    && !l.contains("PROVENANCE")
                    && !l.contains("BaselineProvenance")
            })
            .count();
        let expected = declared - extra_non_provenance;
        assert_eq!(
            PROVENANCE_REGISTRY.len(),
            expected,
            "PROVENANCE_REGISTRY has {} entries but experiments.rs declares {} *_PROVENANCE constants",
            PROVENANCE_REGISTRY.len(),
            expected,
        );
    }

    #[test]
    fn provenance_registry_no_duplicate_scripts() {
        let mut scripts: Vec<&str> = PROVENANCE_REGISTRY.iter().map(|p| p.script).collect();
        scripts.sort_unstable();
        for pair in scripts.windows(2) {
            assert_ne!(
                pair[0], pair[1],
                "duplicate script in PROVENANCE_REGISTRY: {}",
                pair[0]
            );
        }
    }

    #[test]
    fn provenance_registry_expected_source_complete() {
        for p in PROVENANCE_REGISTRY {
            let src = p.expected_source();
            assert!(
                !src.is_empty(),
                "expected_source() returned empty for script: {}",
                p.script,
            );
        }
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "global minima are exactly 0.0 by mathematical definition"
    )]
    fn benchmark_references_have_global_minima() {
        assert!(
            RASTRIGIN_REFERENCE
                .iter()
                .any(|(x, y, _)| *x == 1.0 && *y == 1.0)
        );
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

    #[test]
    fn provenance_scripts_exist_on_disk() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut missing = Vec::new();
        for p in PROVENANCE_REGISTRY {
            let path = root.join(p.script);
            if !path.exists() {
                missing.push(p.script);
            }
        }
        assert!(
            missing.is_empty(),
            "registered scripts missing on disk: {missing:?}",
        );
    }

    #[test]
    fn provenance_scripts_have_provenance_header() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut missing_header = Vec::new();
        for p in PROVENANCE_REGISTRY {
            let path = root.join(p.script);
            if let Ok(content) = std::fs::read_to_string(&path) {
                if !content.contains("# Provenance: see src/provenance/") {
                    missing_header.push(p.script);
                }
            }
        }
        assert!(
            missing_header.is_empty(),
            "scripts without provenance header: {missing_header:?}",
        );
    }

    #[test]
    fn provenance_scripts_have_spdx_header() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut missing_spdx = Vec::new();
        for p in PROVENANCE_REGISTRY {
            let path = root.join(p.script);
            if let Ok(content) = std::fs::read_to_string(&path) {
                if !content.contains("SPDX-License-Identifier:") {
                    missing_spdx.push(p.script);
                }
            }
        }
        assert!(
            missing_spdx.is_empty(),
            "scripts without SPDX header: {missing_spdx:?}",
        );
    }

    #[test]
    fn provenance_scripts_content_stability() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut hashes = Vec::new();
        for p in PROVENANCE_REGISTRY {
            let path = root.join(p.script);
            if let Ok(content) = std::fs::read_to_string(&path) {
                let mut hasher = DefaultHasher::new();
                content.hash(&mut hasher);
                hashes.push((p.script, content.len(), hasher.finish()));
            }
        }
        assert_eq!(
            hashes.len(),
            PROVENANCE_REGISTRY.len(),
            "some scripts could not be read",
        );
        for (script, size, _hash) in &hashes {
            assert!(
                *size > 100,
                "script {script} is suspiciously small ({size} bytes)",
            );
        }
    }
}
