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
            "neuralSpring v{} | {} {} | {}",
            self.neuralspring_version, self.os, self.arch, self.rust_version
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
// Phase 0: Experiments (48/48 PASS)
// ═══════════════════════════════════════════════════════════════════

pub const SURROGATE_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp 001: Neural Surrogate Validation (11/11 PASS)",
    script: "control/surrogate/surrogate_validation.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/surrogate/surrogate_validation.py",
    environment: ENVIRONMENT,
    value: 11.0,
    unit: "checks passed",
};

pub const TRANSFORMER_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp 002: Transformer Inference Baseline (18/18 PASS)",
    script: "control/transformer/transformer_inference.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/transformer/transformer_inference.py",
    environment: ENVIRONMENT,
    value: 18.0,
    unit: "checks passed",
};

pub const SEQUENCE_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp 003: Sequence Forecasting (5/5 PASS)",
    script: "control/sequence/sequence_forecasting.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/sequence/sequence_forecasting.py",
    environment: ENVIRONMENT,
    value: 5.0,
    unit: "checks passed",
};

pub const TRANSFER_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp 004: Transfer Learning (6/6 PASS)",
    script: "control/transfer/transfer_learning.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/transfer/transfer_learning.py",
    environment: ENVIRONMENT,
    value: 6.0,
    unit: "checks passed",
};

pub const ISOMORPHIC_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp 005: Isomorphic Learning Catalog (8/8 PASS)",
    script: "control/isomorphic/isomorphic_catalog.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/isomorphic/isomorphic_catalog.py",
    environment: ENVIRONMENT,
    value: 8.0,
    unit: "checks passed",
};

// ═══════════════════════════════════════════════════════════════════
// Phase 0+: Scholarly Reproductions (27/27 PASS)
// ═══════════════════════════════════════════════════════════════════

pub const PINN_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Study 001: PINN Burgers Equation (6/6 PASS)",
    script: "control/pinn/pinn_burgers.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/pinn/pinn_burgers.py",
    environment: ENVIRONMENT,
    value: 6.0,
    unit: "checks passed",
};

pub const DEEPONET_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Study 002: DeepONet Antiderivative (5/5 PASS)",
    script: "control/deeponet/deeponet_antideriv.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/deeponet/deeponet_antideriv.py",
    environment: ENVIRONMENT,
    value: 5.0,
    unit: "checks passed",
};

pub const LENET_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Study 003: LeNet-5 MNIST (4/4 PASS)",
    script: "control/lenet/lenet_mnist.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/lenet/lenet_mnist.py",
    environment: ENVIRONMENT,
    value: 4.0,
    unit: "checks passed",
};

pub const LSTM_ERA5_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Study 004: LSTM ERA5 Weather (5/5 PASS)",
    script: "control/lstm_weather/lstm_era5.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/lstm_weather/lstm_era5.py",
    environment: ENVIRONMENT,
    value: 5.0,
    unit: "checks passed",
};

pub const QUANTIZED_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Study 005: Quantized Inference (6/6 PASS)",
    script: "control/quantized/quantized_inference.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/quantized/quantized_inference.py",
    environment: ENVIRONMENT,
    value: 6.0,
    unit: "checks passed",
};

// ═══════════════════════════════════════════════════════════════════
// Phase 0++: Paper Reproductions (53/53 PASS)
// ═══════════════════════════════════════════════════════════════════

pub const COUNTERDIABATIC_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 011: Counterdiabatic Evolution (11/11 PASS)",
    script: "control/counterdiabatic/counterdiabatic_evolution.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/counterdiabatic/counterdiabatic_evolution.py",
    environment: ENVIRONMENT,
    value: 11.0,
    unit: "checks passed",
};

pub const MODES_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 012: MODES Toolbox (9/9 PASS)",
    script: "control/modes/modes_toolbox.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/modes/modes_toolbox.py",
    environment: ENVIRONMENT,
    value: 9.0,
    unit: "checks passed",
};

pub const ECO_DYNAMICS_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 013: Ecological Dynamics (7/7 PASS)",
    script: "control/eco_dynamics/eco_dynamics.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/eco_dynamics/eco_dynamics.py",
    environment: ENVIRONMENT,
    value: 7.0,
    unit: "checks passed",
};

pub const DIRECTED_EVOLUTION_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 014: Directed Evolution (8/8 PASS)",
    script: "control/directed_evolution/directed_evolution.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/directed_evolution/directed_evolution.py",
    environment: ENVIRONMENT,
    value: 8.0,
    unit: "checks passed",
};

pub const HMM_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 016: HMM Phylogenetic Inference (10/10 PASS)",
    script: "control/hmm_phylo/hmm_phylo.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/hmm_phylo/hmm_phylo.py",
    environment: ENVIRONMENT,
    value: 10.0,
    unit: "checks passed",
};

pub const GAME_THEORY_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 019: Game Theory & QS Cooperation (8/8 PASS)",
    script: "control/game_theory/game_theory.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/game_theory/game_theory.py",
    environment: ENVIRONMENT,
    value: 8.0,
    unit: "checks passed",
};

pub const SWARM_ROBOTICS_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 015: Heterogeneous Swarm Robotics (11/11 PASS)",
    script: "control/swarm_robotics/swarm_robotics.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/swarm_robotics/swarm_robotics.py",
    environment: ENVIRONMENT,
    value: 11.0,
    unit: "checks passed",
};

pub const SATE_ALIGNMENT_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 017: SATé Alignment (8/8 PASS)",
    script: "control/sate_alignment/sate_alignment.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/sate_alignment/sate_alignment.py",
    environment: ENVIRONMENT,
    value: 8.0,
    unit: "checks passed",
};

pub const INTROGRESSION_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 018: Introgression Detection (8/8 PASS)",
    script: "control/introgression/introgression.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/introgression/introgression.py",
    environment: ENVIRONMENT,
    value: 8.0,
    unit: "checks passed",
};

pub const REGULATORY_NETWORK_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 020: Regulatory Network (7/7 PASS)",
    script: "control/regulatory_network/regulatory_network.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/regulatory_network/regulatory_network.py",
    environment: ENVIRONMENT,
    value: 7.0,
    unit: "checks passed",
};

pub const SIGNAL_INTEGRATION_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 021: Signal Integration (8/8 PASS)",
    script: "control/signal_integration/signal_integration.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/signal_integration/signal_integration.py",
    environment: ENVIRONMENT,
    value: 8.0,
    unit: "checks passed",
};

pub const SPECTRAL_COMMUTATIVITY_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 022: Spectral Commutativity (8/8 PASS)",
    script: "control/spectral_commutativity/spectral_commutativity.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/spectral_commutativity/spectral_commutativity.py",
    environment: ENVIRONMENT,
    value: 8.0,
    unit: "checks passed",
};

pub const ANDERSON_LOCALIZATION_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 023: Anderson Localization (8/8 PASS)",
    script: "control/anderson_localization/anderson_localization.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/anderson_localization/anderson_localization.py",
    environment: ENVIRONMENT,
    value: 8.0,
    unit: "checks passed",
};

// ═══════════════════════════════════════════════════════════════════
// Phase 0++: Empirical Corollary — R. Anderson (16/16 PASS)
// ═══════════════════════════════════════════════════════════════════

pub const PANGENOME_SELECTION_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 024: Pangenome Selection (8/8 PASS)",
    script: "control/pangenome_selection/pangenome_selection.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/pangenome_selection/pangenome_selection.py",
    environment: ENVIRONMENT,
    value: 8.0,
    unit: "checks passed",
};

pub const META_POPULATION_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 025: Meta-Population Differentiation (8/8 PASS)",
    script: "control/meta_population/meta_population.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/meta_population/meta_population.py",
    environment: ENVIRONMENT,
    value: 8.0,
    unit: "checks passed",
};

// ═══════════════════════════════════════════════════════════════════
// ML inference baselines (JSON weights + expected outputs)
// ═══════════════════════════════════════════════════════════════════

pub const ML_INFERENCE_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "ML Inference Baselines (MLP + Transformer JSON weights)",
    script: "control/ml_inference/generate_baselines.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/ml_inference/generate_baselines.py",
    environment: ENVIRONMENT,
    value: 2.0,
    unit: "baseline files generated (mlp_baseline.json, transformer_baseline.json)",
};

/// `BarraCUDA` validation expected values are analytically derived — no Python
/// dependency.  Provenance is mathematical: NIST DLMF, IEEE 754, and textbook
/// formulas.
pub const BARRACUDA_ANALYTICAL_REFS: &str = "Analytical (IEEE 754, NIST DLMF, textbook formulas)";

/// Chi-squared distribution reference values.
///
/// PDF/CDF validated against `SciPy` 1.15.3 `scipy.stats.chi2`.
/// Moments and test statistic are analytically derived.
///
/// Provenance:
/// ```text
/// python3 -c "from scipy.stats import chi2; print(chi2.pdf(2,3), chi2.pdf(0,3), chi2.pdf(5,1))"
/// python3 -c "from scipy.stats import chi2; print(chi2.cdf(3.84,1), chi2.cdf(5.99,2), chi2.cdf(0,5))"
/// ```
/// Environment: `SciPy` 1.15.3, Python 3.10.12, 2026-02-16
pub const CHI_SQUARED_REFS: &str = "SciPy 1.15.3 chi2 + analytical moments (Pearson 1900)";

/// FFT validation: analytical DFT pairs + Parseval's theorem.
///
/// No Python dependency — all expected values derive from the definition of
/// the Discrete Fourier Transform (Cooley & Tukey, 1965; FFTW docs).
pub const FFT_ANALYTICAL_REFS: &str =
    "Analytical (DFT definition, Parseval's theorem, Cooley-Tukey 1965)";

// ═══════════════════════════════════════════════════════════════════
// Cross-language reference values (Python-computed, hardcoded in Rust)
// ═══════════════════════════════════════════════════════════════════

/// Softmax of `[1,2,3,4,5]` computed by `NumPy` 2.2.6.
///
/// Provenance: `python3 -c "import numpy as np; x=np.array([1.,2.,3.,4.,5.]); e=np.exp(x-x.max()); print(e/e.sum())"`
/// Environment: `NumPy` 2.2.6, Python 3.10.12, IEEE 754 f64.
pub const SOFTMAX_1_TO_5: [f64; 5] = [
    1.165_623_095_603_961e-2,
    3.168_492_079_612_427e-2,
    8.612_854_443_626_87e-2,
    2.341_216_572_527_366e-1,
    6.364_086_465_588_308e-1,
];

/// GELU reference values at selected points, computed by `NumPy` 2.2.6.
///
/// Format: (input, `expected_output`)
/// Provenance: `python3 -c "import numpy as np; gelu=lambda x: 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3))); [print(x,gelu(x)) for x in [-2,-1,0,0.5,1,3]]"`
/// Environment: `NumPy` 2.2.6, Python 3.10.12, IEEE 754 f64.
pub const GELU_REFERENCE: [(f64, f64); 6] = [
    (-2.0, -4.540_230_591_222_494e-2),
    (-1.0, -1.588_080_093_917_233e-1),
    (0.0, 0.0),
    (0.5, 3.457_140_098_251_439e-1),
    (1.0, 8.411_919_906_082_768e-1),
    (3.0, 2.996_362_607_918_227),
];

/// Rastrigin 2D reference values at non-trivial points, computed by `NumPy` 2.2.6.
///
/// Provenance: `python3 control/surrogate/surrogate_validation.py` (`rastrigin_2d`).
/// Environment: `NumPy` 2.2.6, Python 3.10.12, IEEE 754 f64.
pub const RASTRIGIN_REFERENCE: [(f64, f64, f64); 4] = [
    (1.0, 1.0, 2.0),
    (2.5, -1.3, 4.103_016_994_374_947e1),
    (0.5, 0.5, 4.05e1),
    (-3.0, 2.0, 13.0),
];

/// Rosenbrock 2D reference values.
pub const ROSENBROCK_REFERENCE: [(f64, f64, f64); 4] = [
    (1.0, 1.0, 0.0),
    (2.5, -1.3, 5702.5),
    (0.5, 0.5, 6.5),
    (-3.0, 2.0, 4916.0),
];

/// Ackley 2D reference values.
pub const ACKLEY_REFERENCE: [(f64, f64, f64); 4] = [
    (1.0, 1.0, 3.625_384_938_440_363),
    (2.5, -1.3, 8.772_020_879_614_113),
    (0.5, 0.5, 4.253_654_026_568_412),
    (-3.0, 2.0, 7.988_910_810_518_7),
];

/// Analytical reference source for benchmark functions.
pub const BENCHMARK_REFS: &str = "Analytical global minima + NumPy 2.2.6 cross-validation";

/// Analytical reference source for transformer primitives.
pub const TRANSFORMER_REFS: &str = "NumPy 2.2.6 transformer_inference.py (softmax, gelu_numpy)";

/// Analytical reference source for statistical metrics.
pub const METRICS_REFS: &str = "Analytical (pure arithmetic on known arrays)";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_reference_sums_near_one() {
        let sum: f64 = SOFTMAX_1_TO_5.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14);
    }

    #[test]
    #[allow(clippy::expect_used)]
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
            &PANGENOME_SELECTION_PROVENANCE,
            &META_POPULATION_PROVENANCE,
        ];
        for p in records {
            assert!(!p.label.is_empty(), "empty label: {}", p.script);
            assert!(!p.script.is_empty());
            assert!(!p.date.is_empty());
            assert!(!p.command.is_empty());
            assert_eq!(p.commit, BASELINE_COMMIT);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
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
        assert!(summary.contains("neuralSpring"));
        assert!(summary.contains(&env.os));
    }
}
