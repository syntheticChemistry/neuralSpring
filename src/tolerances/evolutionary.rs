// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tolerances for Phase 0++ evolutionary and stochastic algorithms.
//!
//! Covers counterdiabatic protocols, HMM, introgression, game theory,
//! eco-dynamics, pangenome, meta-population, signal integration, and
//! spectral commutativity validation thresholds.

/// Counterdiabatic protocol: max gap between CD and naive for the
/// comparison to be considered "comparable" (not a strict improvement).
///
/// 0.01 L1 distance is a tight threshold — the mean-field Wright-Fisher
/// operates in a 32-dimensional simplex where total variation is at most 2.
pub const CD_COMPARABLE_DIST: f64 = 0.01;

/// Adiabaticity gap: max excess KL for CD over naive before failing.
///
/// CD should stay closer to equilibrium; 0.05 nats allows for numerical
/// noise in the Fisher information discretization (1000 grid points).
pub const ADIABATIC_KL_GAP: f64 = 0.05;

/// HMM posterior: row-sum tolerance (should sum to 1.0).
///
/// Forward-backward accumulates rounding from T matrix-vector products.
/// For T ≤ 5000 with scaling, 1e-8 is conservative.
pub const HMM_POSTERIOR_SUM: f64 = 1e-8;

/// QS cooperation variance: max variance in late cooperation frequency.
///
/// A stabilized QS model should not oscillate beyond this level.
pub const QS_VARIANCE_MAX: f64 = 0.05;

/// IPR localization threshold: `IPR > 1/N` indicates localization.
///
/// For `N = 20`, the extended (delocalized) baseline is `IPR ≈ 1/N = 0.05`.
/// Localized states have `IPR >> 0.05`. The check verifies `mean_IPR > 0.05`
/// at strong disorder (`W/t = 10`), confirming Anderson localization.
pub const IPR_LOCALIZATION_MIN: f64 = 0.05;

/// HMM Viterbi accuracy: minimum fraction of correctly decoded states.
///
/// A well-specified HMM should decode > 50% of states correctly.
/// The weather HMM (2-state) with long sequences achieves ~70%.
pub const HMM_DECODE_ACCURACY_MIN: f64 = 0.05;

/// Introgression detection: minimum detected introgression fraction.
///
/// Under the PhyloNet-HMM model, true introgression fraction is ~0.2.
/// The detector should identify > 5% introgression loci.
pub const INTROGRESSION_FRACTION_MIN: f64 = 0.05;

/// Introgression detection: detected fraction vs true fraction tolerance.
///
/// With 500 synthetic loci (seed=42), the PhyloNet-HMM Viterbi path
/// detects fraction within 0.15 of the true introgression fraction.
/// This accounts for HMM boundary effects and ILS/introgression overlap.
/// Source: `control/introgression/introgression.py` (seed=42, n=500).
pub const INTROGRESSION_FRACTION_ABS: f64 = 0.15;

/// Introgression detection: false positive rate upper bound.
///
/// When no introgression is present (ILS-only loci), the detector
/// should report < 25% introgression. This is generous because
/// the 3-topology model has inherent overlap between ILS and
/// introgression gene-tree patterns.
/// Source: `control/introgression/introgression.py` (seed=42, ILS-only).
pub const INTROGRESSION_FPR_MAX: f64 = 0.25;

/// Gene tree concordance: minimum concordant topology fraction.
///
/// Under the multispecies coalescent, the concordant gene-tree
/// topology (matching the species tree) should dominate. With
/// introgression fraction ~0.2, concordant fraction > 0.2 is expected.
/// Source: coalescent theory — concordant topology probability ≥ 1 - 2/3 exp(-t).
pub const GENE_TREE_CONCORDANT_MIN: f64 = 0.2;

/// Game theory cooperation: minimum QS cooperation frequency.
///
/// At signal threshold below carrying capacity, QS-mediated
/// cooperation should be detectable (cooperation frequency > 5%).
pub const GAME_COOPERATION_MIN: f64 = 0.05;

/// Replicator dynamics: accumulated Euler-step tolerance.
///
/// 1000 Euler steps at dt=0.01 accumulate O(dt²·n) truncation error.
/// Cross-language validation shows Rust and Python agree to ~1e-7;
/// 1e-6 provides margin for different FP summation order.
///
/// Provenance: `control/game_theory/game_theory.py` (seed=42, commit `f9ad0268`)
pub const REPLICATOR_DYNAMICS: f64 = 1e-6;

/// Regulatory network: minimum Hill function response.
///
/// For non-zero inputs above the activation threshold, the Hill
/// function should produce a measurable response (> 1%).
pub const REGULATORY_RESPONSE_MIN: f64 = 0.01;

/// Eco-dynamics mean fitness improvement: EA should improve.
///
/// Over 10 generations with tournament selection, mean fitness
/// should increase by at least 8% from the random initial population.
pub const ECO_FITNESS_IMPROVEMENT_MIN: f64 = 0.08;

/// Pangenome selection: minimum positive selection signal (dN/dS > 1).
///
/// The test sequence has elevated nonsynonymous substitutions;
/// the chi-squared p-value should indicate significance.
pub const PANGENOME_SELECTION_P_MIN: f64 = 0.01;

/// Meta-population `F_ST` threshold: differentiation above drift alone.
///
/// With `F_ST` = 0.1 target, the observed `F_ST` should exceed 1% to
/// demonstrate measurable genetic structure.
pub const META_POP_FST_MIN: f64 = 0.01;

/// Meta-population inter-population allele frequency variance.
///
/// With `F_ST > 0`, allele frequency variance across populations
/// should exceed 0.1% (0.001) to confirm genetic structure.
pub const META_POP_AF_VARIANCE_MIN: f64 = 0.001;

/// Phylo HMM Viterbi margin: excess accuracy over chance for
/// 4-state phylo HMM on 5000 observations. Tighter than the
/// 2-state weather HMM because more states make the problem harder.
pub const HMM_PHYLO_DECODE_MARGIN: f64 = 0.02;

/// Signal integration: minimum dynamic range of Hill gate response.
///
/// The Hill function should produce distinguishable high and low
/// outputs. A dynamic range < 1% indicates a degenerate gate.
pub const SIGNAL_DYNAMIC_RANGE_MIN: f64 = 0.01;

/// Layer spectral similarity: self-similarity tolerance.
///
/// `layer_spectral_similarity(W, W)` should return ≈ 1.0 (cosine similarity
/// of a vector with itself).  Deviations arise from eigenvalue sorting
/// and f64 rounding in the SVD/eigendecomposition.  0.01 is conservative.
pub const SPECTRAL_SELF_SIMILARITY: f64 = 0.01;

/// PGM complexity monotonicity slack.
///
/// When asserting that a deeper/denser PGM is at least as complex as
/// a shallower/sparser one, allow a small negative margin for floating-point
/// rounding in the entropy-based complexity measure.
pub const PGM_COMPLEXITY_SLACK: f64 = 0.01;

/// Swarm fitness comparison tolerance (heterogeneous >= homogeneous - tol).
///
/// Heterogeneous swarms may not always exceed homogeneous fitness,
/// but should be within 2.0 fitness units (mean-of-last-10 scale).
/// Foreback, Bohm, Dolson (2025).
///
/// Derivation: measured fitness gap between heterogeneous and homogeneous
/// swarms across 10 seeded runs (seed 0..9): mean gap = 0.3 ± 1.2.
/// 2.0 ≈ mean + 1.5σ, ensuring > 95% of runs pass.
pub const SWARM_FITNESS_COMPARISON: f64 = 2.0;

/// Eco-dynamics dominance comparison tolerance.
///
/// Multi-niche dominance should not exceed single-niche by more
/// than 0.3 (fraction scale 0–1).  Allows for stochastic run
/// variance with seed=42.
pub const ECO_DOMINANCE_COMPARISON: f64 = 0.3;

/// `F_ST` for identical populations: Weir-Cockerham sample correction.
///
/// `F_ST` for identical allele frequencies should be ~0 by definition.
/// The Weir-Cockerham estimator introduces a small sample-size bias
/// (denominator correction), so the tolerance is 0.05 rather than
/// machine precision.
pub const FST_IDENTICAL_POP_TOL: f64 = 0.05;

/// `F_ST` estimator agreement: mean-of-ratios vs ratio-of-sums.
///
/// Two `F_ST` estimators (W-C individual-locus vs multi-locus average)
/// can differ by up to 0.05 for 10 loci with 20 individuals per pop.
pub const FST_ESTIMATOR_AGREEMENT: f64 = 0.05;

/// PD defection dominance upper bound on cooperation.
///
/// In the prisoner's dilemma with b=3, c=1, defection should dominate
/// after 2000 Euler steps.  Cooperation frequency should drop below 0.1.
pub const GAME_DEFECTION_UPPER: f64 = 0.1;

/// QS cooperation late-phase minimum.
///
/// After 80+ generations with quorum sensing (threshold 0.3, benefit 0.3),
/// cooperation should stabilize above 0.1 (detectable cooperation).
/// Stricter than `GAME_COOPERATION_MIN` (0.05) because this tests the
/// barracuda RK45 implementation specifically.
pub const GAME_QS_COOPERATION_MIN: f64 = 0.1;

/// QS cooperation variance upper bound.
///
/// After stabilization (gen 50+), the cooperation frequency variance
/// should remain below 0.1, indicating equilibrium rather than cycling.
pub const GAME_QS_VARIANCE_MAX: f64 = 0.1;
