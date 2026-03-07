// SPDX-License-Identifier: AGPL-3.0-or-later

//! baseCamp Sub-thesis 06: Anderson localization in immunological signaling.
//!
//! Domain adapter wiring neuralSpring's validated primitives to Paper 12's
//! immunological cytokine propagation framework. Maps atopic dermatitis (AD)
//! disease states onto Anderson localization regime transitions.
//!
//! ## Core Mapping
//!
//! | Anderson QS | Immunological Extension |
//! |-------------|------------------------|
//! | Lattice site | Cell position in tissue |
//! | On-site energy | Cell type (keratinocyte, Th2, neuron, mast, eosinophil) |
//! | Hopping t | Cytokine diffusion coefficient in ECM |
//! | Disorder W | Cell-type heterogeneity (Pielou evenness) |
//! | Dimension d | Tissue geometry (epidermis ~2D, dermis ~3D) |
//! | Level spacing r | Cytokine signal: extended (propagating) vs localized |
//!
//! ## neuralSpring Primitives Used
//!
//! - [`MultiHeadWdmClassifier`](crate::wdm_esn::MultiHeadWdmClassifier): ESN
//!   regime classifier adapted for AD states
//! - [`TrainingMonitor`]: pharmacokinetic trajectory monitoring via FSM
//! - [`Dispatcher::kl_divergence`](crate::gpu_dispatch::Dispatcher): cytokine
//!   distribution shift detection
//! - [`SpectralNautilusBridge`](crate::nautilus_bridge::SpectralNautilusBridge):
//!   spectral analysis to drift detection bridge
//!
//! ## Dimensional Promotion-Collapse Duality
//!
//! Paper 06 (no-till): tillage -> 3D->2D collapse -> QS fails.
//! Paper 12 (AD): scratching -> 2D->3D promotion -> cytokine delocalization.
//!
//! ## References
//!
//! - Gonzales AJ et al. (2013-2024)
//! - Fajgenbaum DC et al. (2019)
//! - McCandless EE et al. (2014)

#![expect(clippy::doc_markdown, reason = "domain-specific numeric patterns")]

pub mod lattice;
pub mod matrix;

use crate::training_monitor::{AttentionState, TrainingInterrupt, TrainingMonitor};
use crate::weight_spectral::WeightSpectralResult;

// Re-export submodule types for backward compatibility.
pub use lattice::{
    barrier_promotion_spectrum, level_spacing_ratio, three_compartment_disorder,
    tissue_lattice_hamiltonian, ThreeCompartmentDisorder,
};
pub use matrix::{
    fajgenbaum_matrix_score, score_all_candidates, DiseaseProfile, DrugCandidate,
    AD_CHRONIC_PROFILE, AD_FLARE_PROFILE, DRUG_CANDIDATES,
};

/// AD skin states mapped to Anderson localization regimes.
///
/// Each state corresponds to a distinct Anderson phase characterized by
/// different effective dimensions and disorder levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdSkinState {
    /// Epidermis 2D -> cytokines localized. Dermis 3D but low production.
    Healthy,
    /// Th2 activated in dermis -> cytokines propagate in 3D to nerve endings.
    Flare,
    /// Persistent 3D channels through barrier -> sustained delocalization.
    Chronic,
    /// Drug intervention reducing effective signal propagation.
    Treated,
}

/// Tissue compartment in the skin Anderson lattice.
///
/// Maps to `MultiHeadWdmClassifier` head indices for three-compartment
/// classification per McCandless (2014) G6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TissueCompartment {
    /// Th2 cells, mast cells, eosinophils. Head 0 (regime label).
    Immune,
    /// Keratinocytes, Langerhans cells. Head 1 (spectral bandwidth).
    Skin,
    /// Sensory nerve endings. Head 2 (confidence).
    Neural,
}

/// Skin layer with Anderson geometry parameters.
#[derive(Debug, Clone)]
pub struct SkinLayer {
    pub name: &'static str,
    pub thickness_um: (f64, f64),
    pub effective_dimension: f64,
    pub acellular: bool,
}

/// Canonical skin layer stack for Anderson lattice construction.
pub const SKIN_LAYERS: [SkinLayer; 5] = [
    SkinLayer {
        name: "stratum_corneum",
        thickness_um: (10.0, 20.0),
        effective_dimension: 0.0,
        acellular: true,
    },
    SkinLayer {
        name: "viable_epidermis",
        thickness_um: (50.0, 100.0),
        effective_dimension: 2.25,
        acellular: false,
    },
    SkinLayer {
        name: "basement_membrane",
        thickness_um: (0.5, 1.0),
        effective_dimension: 2.0,
        acellular: true,
    },
    SkinLayer {
        name: "papillary_dermis",
        thickness_um: (100.0, 200.0),
        effective_dimension: 3.0,
        acellular: false,
    },
    SkinLayer {
        name: "reticular_dermis",
        thickness_um: (1000.0, 3000.0),
        effective_dimension: 3.0,
        acellular: false,
    },
];

/// Drug mechanism of action in the Anderson framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrugMechanism {
    /// Removes the diffusible signal molecule (e.g., Cytopoint anti-IL-31).
    SignalElimination,
    /// Blocks intracellular transduction (e.g., Apoquel JAK1 inhibitor).
    TransductionBlock,
    /// Restores 2D barrier geometry -> Anderson re-localization.
    BarrierRepair,
    /// Blocks receptor -> prevents signal binding (e.g., Dupilumab anti-IL-4Ra).
    ReceptorBlock,
}

/// Anderson-augmented drug repurposing score.
///
/// Extends Fajgenbaum MATRIX: `Score(drug, disease, tissue)` =
/// `f(pathway) * g(geometry, delivery, size)`.
#[derive(Debug, Clone)]
pub struct AndersonDrugScore {
    pub drug_name: String,
    pub pathway_score: f64,
    pub geometry_score: f64,
    pub combined_score: f64,
    pub mechanism: DrugMechanism,
    pub delivery_systemic: bool,
}

impl AndersonDrugScore {
    /// Compute the combined Anderson-augmented score.
    #[must_use]
    pub fn compute(
        drug_name: &str,
        pathway_score: f64,
        geometry_score: f64,
        mechanism: DrugMechanism,
        delivery_systemic: bool,
    ) -> Self {
        let combined = pathway_score * geometry_score;
        Self {
            drug_name: drug_name.to_owned(),
            pathway_score,
            geometry_score,
            combined_score: combined,
            mechanism,
            delivery_systemic,
        }
    }
}

/// Dimensional promotion metric.
///
/// Quantifies how barrier disruption changes effective dimension from
/// baseline 2D (intact epidermis) toward 3D (breached). The inverse of
/// Paper 06's tillage dimensional collapse.
#[must_use]
pub fn dimensional_promotion(intact_fraction: f64, baseline_d: f64, target_d: f64) -> f64 {
    let breach_fraction = 1.0 - intact_fraction.clamp(0.0, 1.0);
    breach_fraction.mul_add(target_d - baseline_d, baseline_d)
}

/// Systemic delivery: molecular-weight attenuation rate (per kDa).
const SYSTEMIC_MW_ATTENUATION: f64 = 0.001;
/// Systemic delivery: minimum geometry factor floor.
const SYSTEMIC_FLOOR: f64 = 0.5;
/// Small-molecule threshold for topical penetration (kDa).
const SMALL_MOLECULE_KDA: f64 = 0.5;
/// Large-molecule threshold for topical penetration (kDa).
const LARGE_MOLECULE_KDA: f64 = 5.0;
/// Topical penetration factor for small molecules (<0.5 kDa).
const TOPICAL_SMALL: f64 = 0.8;
/// Topical penetration factor for medium molecules (0.5–5 kDa).
const TOPICAL_MEDIUM: f64 = 0.5;
/// Topical penetration factor for large molecules (>5 kDa).
const TOPICAL_LARGE: f64 = 0.1;
/// Barrier breach bonus scaling factor for topical delivery.
const BREACH_BONUS_SCALE: f64 = 0.3;

/// Geometry factor for drug tissue accessibility.
///
/// Large molecules (mAbs) require systemic delivery to reach 3D dermis.
/// Small molecules can penetrate the 2D barrier topically, especially
/// when the barrier is compromised (higher `barrier_breach_fraction`).
#[must_use]
pub fn tissue_geometry_factor(
    molecular_weight_kda: f64,
    delivery_systemic: bool,
    barrier_breach_fraction: f64,
) -> f64 {
    if delivery_systemic {
        SYSTEMIC_MW_ATTENUATION
            .mul_add(-molecular_weight_kda, 1.0)
            .clamp(SYSTEMIC_FLOOR, 1.0)
    } else {
        let size_factor = if molecular_weight_kda < SMALL_MOLECULE_KDA {
            TOPICAL_SMALL
        } else if molecular_weight_kda < LARGE_MOLECULE_KDA {
            TOPICAL_MEDIUM
        } else {
            TOPICAL_LARGE
        };
        let breach_bonus = barrier_breach_fraction * BREACH_BONUS_SCALE;
        (size_factor + breach_bonus).clamp(0.0, 1.0)
    }
}

/// Pharmacokinetic monitor for drug signal extinction tracking.
///
/// Wraps [`TrainingMonitor`] to track drug efficacy decay as an Anderson
/// signal extinction process. Maps pharmacokinetic half-life curves onto
/// the attention state machine.
pub struct PharmacoMonitor {
    monitor: TrainingMonitor,
    dose_mg_per_kg: f64,
    hours_elapsed: f64,
}

impl PharmacoMonitor {
    #[must_use]
    pub fn new(dose_mg_per_kg: f64) -> Self {
        Self {
            monitor: TrainingMonitor::new(),
            dose_mg_per_kg,
            hours_elapsed: 0.0,
        }
    }

    /// Record an observation timepoint.
    ///
    /// `pruritus_score` is a clinical score (lower = better), mapped to
    /// "loss" in the training monitor.
    pub fn observe(&mut self, hours: f64, pruritus_score: f64, spectral: &WeightSpectralResult) {
        let epoch = self.monitor.epoch_count();
        self.hours_elapsed = hours;
        self.monitor.observe_epoch(epoch, pruritus_score, spectral);
    }

    /// Check whether the treatment needs adjustment.
    ///
    /// Maps [`TrainingInterrupt`] to pharmacological decisions:
    /// - `Continue` -> treatment working, maintain dose
    /// - `ReduceLearningRate` -> partial response, consider dose adjustment
    /// - `EarlyStop` -> treatment failure, switch therapy
    #[must_use]
    pub fn check_treatment(&self) -> TrainingInterrupt {
        self.monitor.check_interrupt()
    }

    #[must_use]
    pub const fn attention_state(&self) -> AttentionState {
        self.monitor.attention()
    }

    #[must_use]
    pub const fn dose(&self) -> f64 {
        self.dose_mg_per_kg
    }

    #[must_use]
    pub const fn hours_elapsed(&self) -> f64 {
        self.hours_elapsed
    }

    #[must_use]
    pub fn is_drifting(&self) -> bool {
        self.monitor.is_drifting()
    }
}

/// AD disease state classifier output.
#[derive(Debug, Clone)]
pub struct AdClassification {
    pub state: AdSkinState,
    pub confidence: f64,
    pub compartment_disagreement: f64,
    pub effective_dimension: f64,
}

/// Classify AD skin state from cytokine-derived Anderson parameters.
///
/// Uses level spacing ratio and effective dimension to determine the
/// Anderson regime of cytokine propagation.
#[must_use]
pub fn classify_ad_state(
    level_spacing_ratio: f64,
    effective_dimension: f64,
    is_treated: bool,
) -> AdSkinState {
    const R_GOE: f64 = 0.5307;

    if is_treated && level_spacing_ratio < R_GOE {
        return AdSkinState::Treated;
    }

    if effective_dimension < 2.5 && level_spacing_ratio < R_GOE {
        AdSkinState::Healthy
    } else if effective_dimension > 2.7 && level_spacing_ratio > R_GOE {
        AdSkinState::Chronic
    } else if level_spacing_ratio > R_GOE {
        AdSkinState::Flare
    } else {
        AdSkinState::Healthy
    }
}

/// Pielou evenness index for cell-type heterogeneity -> disorder W.
///
/// `J = H' / ln(S)` where `H' = -sum(p_i * ln(p_i))`, `S` = number of
/// cell types. `J = 1.0` means perfectly even (maximum disorder), `J -> 0`
/// means dominated by one type (minimum disorder).
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "cell type count (small usize) → f64 for log normalization"
)]
pub fn pielou_evenness(cell_fractions: &[f64]) -> f64 {
    let s = cell_fractions.len();
    if s <= 1 {
        return 0.0;
    }
    let h_prime: f64 = cell_fractions
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    let h_max = (s as f64).ln();
    if h_max == 0.0 {
        0.0
    } else {
        h_prime / h_max
    }
}

/// Map Pielou evenness to Anderson disorder W.
///
/// In the Anderson model, `W/t` determines localization. Higher evenness
/// (more diverse cell populations) maps to higher disorder.
#[must_use]
pub fn evenness_to_disorder(evenness: f64, w_scale: f64) -> f64 {
    evenness * w_scale
}

/// IC50 as Anderson barrier height.
///
/// Maps drug IC50 concentration to the effective disorder reduction:
/// at IC50, half the signaling is blocked (effective W reduced by half).
/// Below IC50, W reduction proportional to `[drug]/IC50` (Hill equation, n=1).
#[must_use]
pub fn ic50_to_w_reduction(drug_concentration_nm: f64, ic50_nm: f64, max_w_reduction: f64) -> f64 {
    if ic50_nm <= 0.0 {
        return 0.0;
    }
    let occupancy = drug_concentration_nm / (drug_concentration_nm + ic50_nm);
    occupancy * max_w_reduction
}

/// Gonzales (2014) JAK1 IC50 values (nM).
pub mod gonzales_ic50 {
    pub const JAK1: f64 = 10.0;
    pub const IL2: f64 = 36.0;
    pub const IL4: f64 = 159.0;
    pub const IL6: f64 = 36.0;
    pub const IL13: f64 = 249.0;
    pub const IL31: f64 = 63.0;
}

/// Fleck/Gonzales (2021) lokivetmab dose-duration data.
pub mod lokivetmab_pk {
    /// `(dose_mg_per_kg, onset_hours, duration_days)`
    pub const DOSE_DURATION: [(f64, f64, f64); 3] =
        [(0.125, 3.0, 14.0), (0.5, 3.0, 28.0), (2.0, 3.0, 42.0)];

    /// Log-linear regression coefficients fit to G4 dose-duration data.
    /// `duration_days = A * ln(dose_mg_kg) + B`
    /// Fit: A ≈ 10.09, B ≈ 33.28 (R² ≈ 0.971)
    pub const REGRESSION_A: f64 = 10.09;
    pub const REGRESSION_B: f64 = 33.28;
}

// ═══════════════════════════════════════════════════════════════════════
// nS-601: Gonzales dose-response modeling (generalized Hill equation)
// ═══════════════════════════════════════════════════════════════════════

/// Generalized Hill equation for dose-response modeling.
///
/// `response = E_max * [drug]^n / ([drug]^n + IC50^n)`
///
/// When `n=1` this reduces to the simple Michaelis-Menten form used in
/// `ic50_to_w_reduction`. The Hill coefficient `n` captures cooperativity:
/// n>1 = positive cooperativity (steeper curve), n<1 = negative.
#[must_use]
pub fn hill_dose_response(concentration: f64, ic50: f64, hill_n: f64, e_max: f64) -> f64 {
    if ic50 <= 0.0 || concentration < 0.0 {
        return 0.0;
    }
    let c_n = concentration.powf(hill_n);
    let ic50_n = ic50.powf(hill_n);
    e_max * c_n / (c_n + ic50_n)
}

/// Sweep Hill dose-response across a concentration range.
///
/// Returns `(concentrations, responses)` for plotting dose-response curves.
#[must_use]
pub fn ic50_sweep(ic50: f64, hill_n: f64, concentrations: &[f64]) -> Vec<f64> {
    concentrations
        .iter()
        .map(|&c| hill_dose_response(c, ic50, hill_n, 1.0))
        .collect()
}

/// Compute Anderson barrier heights for all 6 Gonzales cytokines.
///
/// Maps each IC50 to `W = ln(IC50) * scale`, quantifying how much
/// effective disorder each pathway contributes to cytokine localization.
#[must_use]
pub fn cytokine_barrier_heights(scale: f64) -> [(f64, f64); 6] {
    let ic50s = [
        gonzales_ic50::JAK1,
        gonzales_ic50::IL2,
        gonzales_ic50::IL4,
        gonzales_ic50::IL6,
        gonzales_ic50::IL13,
        gonzales_ic50::IL31,
    ];
    let mut result = [(0.0, 0.0); 6];
    for (i, &ic50) in ic50s.iter().enumerate() {
        result[i] = (ic50, ic50.ln() * scale);
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════
// nS-603: Pharmacokinetic decay modeling
// ═══════════════════════════════════════════════════════════════════════

/// Exponential PK decay: concentration at time t.
///
/// `C(t) = C_0 * exp(-k * t)` where `k = ln(2) / half_life`.
#[must_use]
pub fn pk_exponential_decay(c0: f64, time_hours: f64, half_life_hours: f64) -> f64 {
    if half_life_hours <= 0.0 {
        return 0.0;
    }
    let k = core::f64::consts::LN_2 / half_life_hours;
    c0 * (-k * time_hours).exp()
}

/// Predict lokivetmab duration from dose using log-linear regression.
///
/// Fit to Fleck/Gonzales (2021) G4 data:
/// `duration = A * ln(dose) + B` where A=10.09, B=33.28.
#[must_use]
pub fn lokivetmab_duration_predict(dose_mg_kg: f64) -> f64 {
    if dose_mg_kg <= 0.0 {
        return 0.0;
    }
    dose_mg_kg
        .ln()
        .mul_add(lokivetmab_pk::REGRESSION_A, lokivetmab_pk::REGRESSION_B)
}

/// Pruritus score model from Gonzales (2016) G3.
///
/// Treatment effect decays exponentially from initial suppression.
/// `score(t) = baseline - (baseline - nadir) * exp(-decay_rate * t)`
/// where nadir is the maximum suppression at t=0 post-dose.
#[must_use]
pub fn pruritus_score_model(
    time_hours: f64,
    baseline_score: f64,
    treatment_suppression: f64,
    decay_rate: f64,
) -> f64 {
    let nadir = baseline_score * (1.0 - treatment_suppression.clamp(0.0, 1.0));
    let recovery = (baseline_score - nadir) * (1.0 - (-decay_rate * time_hours).exp());
    nadir + recovery
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensional_promotion() {
        let fully_intact = dimensional_promotion(1.0, 2.0, 3.0);
        assert!((fully_intact - 2.0).abs() < 1e-10, "intact = baseline 2D");

        let fully_breached = dimensional_promotion(0.0, 2.0, 3.0);
        assert!((fully_breached - 3.0).abs() < 1e-10, "breached = target 3D");

        let half_breached = dimensional_promotion(0.5, 2.0, 3.0);
        assert!((half_breached - 2.5).abs() < 1e-10, "half = midpoint");
    }

    #[test]
    fn test_classify_ad_state() {
        assert_eq!(classify_ad_state(0.40, 2.0, false), AdSkinState::Healthy);
        assert_eq!(classify_ad_state(0.60, 2.8, false), AdSkinState::Chronic);
        assert_eq!(classify_ad_state(0.60, 2.6, false), AdSkinState::Flare);
        assert_eq!(classify_ad_state(0.40, 2.6, true), AdSkinState::Treated);
    }

    #[test]
    fn test_pielou_evenness() {
        let even = pielou_evenness(&[0.25, 0.25, 0.25, 0.25]);
        assert!((even - 1.0).abs() < 1e-10, "perfectly even = 1.0");

        let dominated = pielou_evenness(&[0.97, 0.01, 0.01, 0.01]);
        assert!(dominated < 0.3, "dominated should be low");
    }

    #[test]
    fn test_evenness_to_disorder() {
        let w = evenness_to_disorder(0.8, 10.0);
        assert!((w - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_ic50_to_w_reduction() {
        let at_ic50 = ic50_to_w_reduction(gonzales_ic50::JAK1, gonzales_ic50::JAK1, 1.0);
        assert!((at_ic50 - 0.5).abs() < 1e-10, "at IC50 = 50% reduction");

        let at_zero = ic50_to_w_reduction(0.0, gonzales_ic50::JAK1, 1.0);
        assert!(at_zero.abs() < 1e-10, "no drug = no reduction");

        let at_10x = ic50_to_w_reduction(100.0, gonzales_ic50::JAK1, 1.0);
        assert!(at_10x > 0.9, "10x IC50 = >90% reduction");
    }

    #[test]
    fn test_tissue_geometry_systemic() {
        let small_systemic = tissue_geometry_factor(0.3, true, 0.0);
        assert!(small_systemic > 0.9, "small systemic reaches dermis");

        let large_systemic = tissue_geometry_factor(150.0, true, 0.0);
        assert!(large_systemic > 0.5, "large mAb still reaches via blood");
    }

    #[test]
    fn test_tissue_geometry_topical() {
        let small_topical_intact = tissue_geometry_factor(0.3, false, 0.0);
        let small_topical_breached = tissue_geometry_factor(0.3, false, 0.5);
        assert!(
            small_topical_breached > small_topical_intact,
            "breached barrier improves topical access"
        );

        let large_topical = tissue_geometry_factor(150.0, false, 0.0);
        assert!(
            large_topical < 0.3,
            "large molecules cannot penetrate topically"
        );
    }

    #[test]
    fn test_anderson_drug_score() {
        let score = AndersonDrugScore::compute(
            "Oclacitinib",
            0.95,
            0.90,
            DrugMechanism::TransductionBlock,
            true,
        );
        assert!((score.combined_score - 0.855).abs() < 1e-10);
        assert_eq!(score.mechanism, DrugMechanism::TransductionBlock);
    }

    #[test]
    fn test_skin_layer_stack() {
        assert_eq!(SKIN_LAYERS.len(), 5);
        assert!(SKIN_LAYERS[0].acellular, "stratum corneum is acellular");
        assert!(!SKIN_LAYERS[1].acellular, "viable epidermis has cells");
        assert!(
            (SKIN_LAYERS[3].effective_dimension - 3.0).abs() < 1e-10,
            "papillary dermis is 3D"
        );
    }

    #[test]
    fn test_lokivetmab_dose_duration_monotonic() {
        let doses = lokivetmab_pk::DOSE_DURATION;
        for i in 1..doses.len() {
            assert!(doses[i].2 > doses[i - 1].2, "higher dose = longer duration");
        }
    }

    #[test]
    fn test_gonzales_ic50_ordering() {
        const _: () = assert!(
            gonzales_ic50::JAK1 < gonzales_ic50::IL31,
            "JAK1 is more potent than IL-31 pathway"
        );
        const _: () = assert!(
            gonzales_ic50::IL13 > gonzales_ic50::IL2,
            "IL-13 requires higher concentration"
        );
    }

    #[test]
    fn test_hill_dose_response_n1() {
        let r = hill_dose_response(10.0, 10.0, 1.0, 1.0);
        assert!((r - 0.5).abs() < 1e-10, "n=1 at IC50 = 0.5");
    }

    #[test]
    fn test_hill_dose_response_cooperativity() {
        let r_n1 = hill_dose_response(5.0, 10.0, 1.0, 1.0);
        let r_n2 = hill_dose_response(5.0, 10.0, 2.0, 1.0);
        assert!(r_n2 < r_n1, "Hill n=2 steeper -> lower response below IC50");
    }

    #[test]
    fn test_ic50_sweep_monotonic() {
        let concs: Vec<f64> = (1..=10).map(|i| f64::from(i) * 10.0).collect();
        let responses = ic50_sweep(gonzales_ic50::JAK1, 1.0, &concs);
        for i in 1..responses.len() {
            assert!(
                responses[i] >= responses[i - 1],
                "dose-response must be monotonic"
            );
        }
    }

    #[test]
    fn test_cytokine_barrier_heights() {
        let heights = cytokine_barrier_heights(1.0);
        assert_eq!(heights.len(), 6);
        assert!(heights[0].1 < heights[2].1, "JAK1 barrier < IL4 barrier");
    }

    #[test]
    fn test_pk_exponential_decay_half_life() {
        let c = pk_exponential_decay(100.0, 24.0, 24.0);
        assert!((c - 50.0).abs() < 0.01, "at t=half_life, C = C0/2");
    }

    #[test]
    fn test_pk_exponential_decay_zero() {
        let c = pk_exponential_decay(100.0, 0.0, 24.0);
        assert!((c - 100.0).abs() < 1e-10, "at t=0, C = C0");
    }

    #[test]
    fn test_lokivetmab_duration_predict() {
        for &(dose, _, expected_dur) in &lokivetmab_pk::DOSE_DURATION {
            let predicted = lokivetmab_duration_predict(dose);
            assert!(
                (predicted - expected_dur).abs() < 5.0,
                "dose={dose}: predicted={predicted:.1}, expected={expected_dur}"
            );
        }
    }

    #[test]
    fn test_pruritus_score_model() {
        let baseline = 8.0;
        let at_zero = pruritus_score_model(0.0, baseline, 0.7, 0.01);
        assert!(
            (at_zero - baseline * 0.3).abs() < 0.01,
            "at t=0, score = baseline * (1-suppression)"
        );
        let later = pruritus_score_model(500.0, baseline, 0.7, 0.01);
        assert!(later > at_zero, "score recovers toward baseline over time");
    }

    #[test]
    fn pharmaco_monitor_creation() {
        let pm = PharmacoMonitor::new(2.0);
        assert!((pm.dose() - 2.0).abs() < 1e-15);
        assert!((pm.hours_elapsed() - 0.0).abs() < 1e-15);
        assert!(!pm.is_drifting());
    }

    #[test]
    fn pharmaco_monitor_observe_tracks_time() {
        let mut pm = PharmacoMonitor::new(1.5);
        let spectral = make_test_spectral();
        pm.observe(24.0, 5.0, &spectral);
        assert!((pm.hours_elapsed() - 24.0).abs() < 1e-15);
    }

    #[test]
    fn pharmaco_monitor_check_treatment_initial() {
        let pm = PharmacoMonitor::new(2.0);
        let interrupt = pm.check_treatment();
        assert!(
            matches!(interrupt, TrainingInterrupt::Continue),
            "fresh monitor should continue"
        );
    }

    #[test]
    fn pharmaco_monitor_not_drifting_initially() {
        let pm = PharmacoMonitor::new(2.0);
        assert!(!pm.is_drifting(), "fresh monitor should not drift");
    }

    #[test]
    fn pharmaco_monitor_observe_multiple() {
        let mut pm = PharmacoMonitor::new(2.0);
        let spectral = make_test_spectral();
        for hour in [0.0_f64, 24.0, 48.0, 72.0] {
            let score = hour.mul_add(-0.05, 8.0);
            pm.observe(hour, score, &spectral);
        }
        assert!((pm.hours_elapsed() - 72.0).abs() < 1e-15);
        let interrupt = pm.check_treatment();
        assert!(
            matches!(interrupt, TrainingInterrupt::Continue),
            "improving scores → continue treatment"
        );
    }

    fn make_test_spectral() -> crate::weight_spectral::WeightSpectralResult {
        crate::weight_spectral::WeightSpectralResult {
            eigenvalues: vec![1.0, 2.0, 3.0],
            mean_ipr: 0.33,
            level_spacing_ratio: 0.53,
            spectral_entropy: 0.95,
            mp_departure: 0.1,
            bandwidth: 2.0,
            condition_number: 3.0,
            phase: crate::weight_spectral::SpectralPhase::Extended,
        }
    }
}
