// SPDX-License-Identifier: AGPL-3.0-or-later

//! Provenance records for Phase 0, 0+, 0++ experiments and related baselines.

use super::{
    ANDERSON_MULTIAGENT_ENVIRONMENT, BASELINE_COMMIT, BASELINE_DATE, BaselineProvenance,
    CPU_PARITY_COMMIT, CPU_PARITY_DATE, CPU_PARITY_ENVIRONMENT, ENVIRONMENT, NS06_BASELINE_DATE,
    PUBLICATION_BASELINE_DATE, PUBLICATION_ENVIRONMENT, WDM_ENVIRONMENT,
};

// ═══════════════════════════════════════════════════════════════════
// Phase 0: Experiments (48/48 PASS)
// ═══════════════════════════════════════════════════════════════════

/// Provenance for Exp 001: Neural Surrogate Validation.
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

/// Provenance for Exp 002: Transformer Inference Baseline.
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

/// Provenance for Exp 003: Sequence Forecasting.
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

/// Provenance for Exp 004: Transfer Learning.
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

/// Provenance for Exp 005: Isomorphic Learning Catalog.
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

/// Provenance for Study 001: PINN Burgers Equation.
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

/// Provenance for Study 002: `DeepONet` Antiderivative.
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

/// Provenance for Study 003: LeNet-5 MNIST.
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

/// Provenance for Study 004: LSTM ERA5 Weather.
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

/// Provenance for Study 005: Quantized Inference.
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

/// Provenance for Paper 011: Counterdiabatic Evolution.
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

/// Provenance for Paper 012: MODES Toolbox.
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

/// Provenance for Paper 013: Ecological Dynamics.
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

/// Provenance for Paper 014: Directed Evolution.
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

/// Provenance for Paper 016: HMM Phylogenetic Inference.
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

/// Provenance for Paper 019: Game Theory & QS Cooperation.
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

/// Provenance for Paper 015: Heterogeneous Swarm Robotics.
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

/// Provenance for Paper 017: `SATé` Alignment.
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

/// Provenance for Paper 018: Introgression Detection.
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

/// Provenance for Paper 020: Regulatory Network.
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

/// Provenance for Paper 021: Signal Integration.
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

/// Provenance for Paper 022: Spectral Commutativity.
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

/// Provenance for Paper 023: Anderson Localization.
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

/// Provenance for Paper 024: Pangenome Selection.
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

/// Provenance for Paper 025: Meta-Population Differentiation.
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

/// Provenance for ML Inference Baselines (MLP + Transformer).
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

/// Provenance for CPU math parity baselines (`validate_cpu_math_parity`).
///
/// Generated post-baseline-freeze with `NumPy` 2.1.3.  Values are within
/// `CROSS_LANGUAGE` tolerance of the Phase 0 baselines.
pub const CPU_PARITY_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "CPU Math Parity References (9 primitives + 9 paper kernels)",
    script: "control/generate_cpu_references.py",
    commit: CPU_PARITY_COMMIT,
    date: CPU_PARITY_DATE,
    command: "python3 control/generate_cpu_references.py > control/cpu_parity_references.json",
    environment: CPU_PARITY_ENVIRONMENT,
    value: 1.0,
    unit: "reference file generated (cpu_parity_references.json)",
};

/// Provenance for nW-01: WDM Transport Surrogate.
pub const WDM_TRANSPORT_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "nW-01: WDM Transport Surrogate (D*, η*, λ* baselines)",
    script: "control/wdm/transport_surrogate.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/wdm/transport_surrogate.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "surrogate trained → transport_surrogate_baseline.json",
};

/// Provenance for nW-02: WDM EOS Surrogate.
pub const WDM_EOS_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "nW-02: WDM EOS Surrogate (H, He, C pressure/energy baselines)",
    script: "control/wdm/eos_surrogate.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/wdm/eos_surrogate.py",
    environment: WDM_ENVIRONMENT,
    value: 3.0,
    unit: "elements trained (H, He, C) → eos_surrogate_baseline.json",
};

/// Provenance for nW-03: WDM S(q,ω) Peak Predictor.
pub const WDM_SQW_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "nW-03: WDM S(q,ω) Peak Predictor (LSTM reservoir baselines)",
    script: "control/wdm/sqw_peak_predictor.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/wdm/sqw_peak_predictor.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "predictor trained → sqw_peak_baseline.json",
};

/// Provenance for nW-04: WDM Classical→WDM Transfer Learning.
pub const WDM_TRANSFER_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "nW-04: WDM Classical→WDM Transfer Learning (R² baselines)",
    script: "control/wdm/transfer_classical_to_wdm.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/wdm/transfer_classical_to_wdm.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "transfer experiment → transfer_baseline.json",
};

/// Provenance for nW-05: WDM ESN Regime Classifier.
pub const WDM_ESN_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "nW-05: WDM ESN Regime Classifier (reservoir baselines)",
    script: "control/wdm/esn_regime_classifier.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/wdm/esn_regime_classifier.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "classifier trained → esn_regime_baseline.json",
};

/// Provenance for nF-01 Phase B: coralForge Evoformer primitives.
pub const CORAL_FORGE_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "nF-01 Phase B: Evoformer primitive baselines (GELU, LayerNorm, SDPA, TriMul, TriAttn)",
    script: "control/coral_forge/evoformer_primitives.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/coral_forge/evoformer_primitives.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated → evoformer_baselines.json",
};

// ═══════════════════════════════════════════════════════════════════
// Publication Experiments (Exp-050, Exp-052, Exp-053)
// ═══════════════════════════════════════════════════════════════════

/// Provenance for Exp-050: Training Trajectory Spectral Analysis.
pub const TRAINING_TRAJECTORY_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp-050: Training Trajectory Spectral Analysis (Paper A — ICML/NeurIPS)",
    script: "control/training_trajectory/training_trajectory.py",
    commit: BASELINE_COMMIT,
    date: PUBLICATION_BASELINE_DATE,
    command: "python3 control/training_trajectory/training_trajectory.py",
    environment: PUBLICATION_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated → baseline_values.json",
};

/// Provenance for Exp-052: Hessian Eigenanalysis at Trained Minima.
pub const HESSIAN_EIGENANALYSIS_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp-052: Hessian Eigenanalysis at Trained Minima (Paper D — Digital Discovery, RSC)",
    script: "control/hessian_eigenanalysis/hessian_eigenanalysis.py",
    commit: BASELINE_COMMIT,
    date: PUBLICATION_BASELINE_DATE,
    command: "python3 control/hessian_eigenanalysis/hessian_eigenanalysis.py",
    environment: PUBLICATION_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated → baseline_values.json",
};

/// Provenance for Exp-053: Anderson Multi-Agent Coordination.
pub const ANDERSON_MULTIAGENT_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp-053: Anderson Multi-Agent Coordination (Paper C — AAMAS/ICML)",
    script: "control/anderson_multiagent/anderson_multiagent.py",
    commit: BASELINE_COMMIT,
    date: PUBLICATION_BASELINE_DATE,
    command: "python3 control/anderson_multiagent/anderson_multiagent.py",
    environment: ANDERSON_MULTIAGENT_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated → baseline_values.json",
};

// ═══════════════════════════════════════════════════════════════════
// nF-02: AlphaFold2 Evoformer Block
// ═══════════════════════════════════════════════════════════════════

/// Provenance for nF-02: `AlphaFold2` Evoformer block (full Evoformer validation).
///
/// Jumper et al. "Highly accurate protein structure prediction with `AlphaFold`"
/// Nature 596:583-589 (2021)
pub const ALPHAFOLD2_EVOFORMER_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "nF-02: AlphaFold2 Evoformer block (SDPA, LayerNorm, matmul, eigh)",
    script: "control/coral_forge/alphafold2_evoformer_block.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/coral_forge/alphafold2_evoformer_block.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated → evoformer_block_baselines.json",
};

// ═══════════════════════════════════════════════════════════════════
// nF-03: AlphaFold3 Phase C — Diffusion, Pairformer, Confidence
// ═══════════════════════════════════════════════════════════════════

/// Provenance for nF-03a: `AlphaFold3` diffusion module.
///
/// Abramson et al. "Accurate structure prediction of biomolecular interactions
/// with `AlphaFold` 3" Nature 630:493-500 (2024), §5.4
pub const ALPHAFOLD3_DIFFUSION_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "nF-03a: AlphaFold3 diffusion module (noise schedule, denoising, loss)",
    script: "control/coral_forge/alphafold3_diffusion.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/coral_forge/alphafold3_diffusion.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated → diffusion_baselines.json",
};

/// Provenance for nF-03b: `AlphaFold3` Pairformer stack.
///
/// Abramson et al. Nature 630:493-500 (2024), §5.3
pub const ALPHAFOLD3_PAIRFORMER_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "nF-03b: AlphaFold3 Pairformer stack (TriMul, TriAttn, pair transition)",
    script: "control/coral_forge/alphafold3_pairformer.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/coral_forge/alphafold3_pairformer.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated → pairformer_baselines.json",
};

/// Provenance for nF-03c: `AlphaFold3` confidence heads.
///
/// Abramson et al. Nature 630:493-500 (2024), §5.9
pub const ALPHAFOLD3_CONFIDENCE_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "nF-03c: AlphaFold3 confidence heads (pLDDT, PAE, pDE, ranking)",
    script: "control/coral_forge/alphafold3_confidence.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/coral_forge/alphafold3_confidence.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated → confidence_baselines.json",
};

// ═══════════════════════════════════════════════════════════════════
// baseCamp Sub-thesis 06: Immunological Anderson (Paper 12)
// ═══════════════════════════════════════════════════════════════════

/// Provenance for nS-06: Anderson Localization in Immunological Signaling.
///
/// Gonzales AJ et al. (2013-2024), Fajgenbaum DC et al. (2019),
/// `McCandless` EE et al. (2014).
pub const IMMUNOLOGICAL_ANDERSON_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "nS-06: Immunological Anderson (baseCamp Paper 12, 20/20 PASS)",
    script: "control/immunological_anderson/immunological_anderson.py",
    commit: BASELINE_COMMIT,
    date: NS06_BASELINE_DATE,
    command: "python3 control/immunological_anderson/immunological_anderson.py",
    environment: WDM_ENVIRONMENT,
    value: 20.0,
    unit: "checks passed",
};

// ═══════════════════════════════════════════════════════════════════
// Paper 026 — Chuna: LSTM Blood Glucose Prediction
// ═══════════════════════════════════════════════════════════════════

/// Provenance for Paper 026: LSTM glucose prediction horizon analysis.
///
/// Chuna (2020) "Setting Limits on Neural Network's Predictive Capacity
/// in T1D Blood Glucose Concentration" (medRxiv 2020.08.04.20117812).
pub const GLUCOSE_PREDICTION_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 026: LSTM Blood Glucose Prediction (9/9 PASS)",
    script: "control/glucose_prediction/glucose_prediction.py",
    commit: BASELINE_COMMIT,
    date: crate::tolerances::GLUCOSE_BASELINE_DATE,
    command: "python3 control/glucose_prediction/glucose_prediction.py",
    environment: WDM_ENVIRONMENT,
    value: 9.0,
    unit: "checks passed",
};

// ═══════════════════════════════════════════════════════════════════
// Paper 027 — Digestion Prediction (Wang/Liao 2020)
// ═══════════════════════════════════════════════════════════════════

/// Provenance for Paper 027: Digestion Prediction ESN.
pub const DIGESTION_PREDICTION_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Paper 027: Digestion Prediction ESN",
    script: "control/digestion_prediction/digestion_prediction.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/digestion_prediction/digestion_prediction.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated",
};

// ═══════════════════════════════════════════════════════════════════
// Composition Experiments (Exp 096–100)
// ═══════════════════════════════════════════════════════════════════

/// Provenance for Exp 096: Digester Community–Performance Coupling via Anderson-ESN.
pub const DIGESTER_ANDERSON_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp 096: Digester-Anderson Coupling",
    script: "control/digester_anderson/digester_anderson.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/digester_anderson/digester_anderson.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated",
};

/// Provenance for Exp 097: Isomorphic Reservoir Ensemble.
pub const ISOMORPHIC_RESERVOIR_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp 097: Isomorphic Reservoir Ensemble",
    script: "control/isomorphic_reservoir/isomorphic_reservoir.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/isomorphic_reservoir/isomorphic_reservoir.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated",
};

/// Provenance for Exp 098: WDM Surrogate Ensemble Quorum Sensing.
pub const WDM_ENSEMBLE_QS_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp 098: WDM Ensemble QS",
    script: "control/wdm_ensemble_qs/wdm_ensemble_qs.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/wdm_ensemble_qs/wdm_ensemble_qs.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated",
};

/// Provenance for Exp 099: HMM Introgression on Neural Network Layers.
pub const INTROGRESSION_NN_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp 099: Introgression NN",
    script: "control/introgression_nn/introgression_nn.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/introgression_nn/introgression_nn.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated",
};

/// Provenance for Exp 100: Attention Anderson Spectral Analysis.
pub const ATTENTION_ANDERSON_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Exp 100: Attention Anderson Spectral",
    script: "control/attention_anderson/attention_anderson.py",
    commit: BASELINE_COMMIT,
    date: BASELINE_DATE,
    command: "python3 control/attention_anderson/attention_anderson.py",
    environment: WDM_ENVIRONMENT,
    value: 1.0,
    unit: "baselines generated",
};

/// Provenance for nS-06 extended: Gonzales dose-response, pruritus time-series,
/// lokivetmab PK, 3D tissue lattice, Fajgenbaum MATRIX scoring.
pub const IMMUNOLOGICAL_ANDERSON_EXTENDED_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "nS-06 extended: Gonzales/PK/Lattice/MATRIX (28/28 PASS)",
    script: "control/immunological_anderson/immunological_anderson_extended.py",
    commit: BASELINE_COMMIT,
    date: NS06_BASELINE_DATE,
    command: "python3 control/immunological_anderson/immunological_anderson_extended.py",
    environment: WDM_ENVIRONMENT,
    value: 28.0,
    unit: "checks passed",
};
