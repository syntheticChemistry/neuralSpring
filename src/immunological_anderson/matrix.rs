// SPDX-License-Identifier: AGPL-3.0-or-later

//! nS-605: Fajgenbaum MATRIX — Anderson-augmented drug repurposing.
//!
//! Extends Fajgenbaum's pathway×disease scoring with Anderson spatial
//! geometry for tissue accessibility. Scores drug candidates by pathway
//! fit, tissue penetration, and residual disorder reduction.

use super::{tissue_geometry_factor, AndersonDrugScore, DrugMechanism};

/// Drug candidate for Anderson-augmented MATRIX scoring.
#[derive(Debug, Clone)]
pub struct DrugCandidate {
    pub name: &'static str,
    pub original_indication: &'static str,
    pub mechanism: DrugMechanism,
    pub pathway_score: f64,
    pub molecular_weight_kda: f64,
    pub delivery_systemic: bool,
}

/// Disease profile for MATRIX tissue geometry evaluation.
#[derive(Debug, Clone)]
pub struct DiseaseProfile {
    pub name: &'static str,
    pub barrier_breach_fraction: f64,
    pub effective_dimension: f64,
    pub mean_disorder_w: f64,
}

/// Compute full MATRIX score: `pathway × geometry × (1 - W_residual)`.
///
/// Extends Fajgenbaum's pathway-only scoring with Anderson spatial geometry.
/// A drug must (a) target the right pathway AND (b) physically reach its
/// target through tissue geometry AND (c) reduce enough disorder to shift
/// the Anderson regime.
#[must_use]
pub fn fajgenbaum_matrix_score(
    drug: &DrugCandidate,
    disease: &DiseaseProfile,
) -> AndersonDrugScore {
    let geom = tissue_geometry_factor(
        drug.molecular_weight_kda,
        drug.delivery_systemic,
        disease.barrier_breach_fraction,
    );
    let w_factor = disease.mean_disorder_w.min(1.0).mul_add(-0.3, 1.0);
    let combined = drug.pathway_score * geom * w_factor;
    AndersonDrugScore {
        drug_name: drug.name.to_owned(),
        pathway_score: drug.pathway_score,
        geometry_score: geom * w_factor,
        combined_score: combined,
        mechanism: drug.mechanism,
        delivery_systemic: drug.delivery_systemic,
    }
}

/// Anderson-filtered repurposing candidates from Paper 12 §3.3.
pub const DRUG_CANDIDATES: [DrugCandidate; 6] = [
    DrugCandidate {
        name: "Rapamycin",
        original_indication: "Transplant rejection",
        mechanism: DrugMechanism::TransductionBlock,
        pathway_score: 0.85,
        molecular_weight_kda: 0.914,
        delivery_systemic: true,
    },
    DrugCandidate {
        name: "Tofacitinib",
        original_indication: "Rheumatoid arthritis",
        mechanism: DrugMechanism::TransductionBlock,
        pathway_score: 0.92,
        molecular_weight_kda: 0.312,
        delivery_systemic: true,
    },
    DrugCandidate {
        name: "Tanezumab",
        original_indication: "Osteoarthritis pain",
        mechanism: DrugMechanism::SignalElimination,
        pathway_score: 0.78,
        molecular_weight_kda: 148.0,
        delivery_systemic: true,
    },
    DrugCandidate {
        name: "Trametinib",
        original_indication: "Melanoma",
        mechanism: DrugMechanism::TransductionBlock,
        pathway_score: 0.65,
        molecular_weight_kda: 0.615,
        delivery_systemic: true,
    },
    DrugCandidate {
        name: "Crisaborole",
        original_indication: "Mild AD",
        mechanism: DrugMechanism::TransductionBlock,
        pathway_score: 0.70,
        molecular_weight_kda: 0.251,
        delivery_systemic: false,
    },
    DrugCandidate {
        name: "Nemolizumab",
        original_indication: "Prurigo nodularis",
        mechanism: DrugMechanism::ReceptorBlock,
        pathway_score: 0.90,
        molecular_weight_kda: 145.0,
        delivery_systemic: true,
    },
];

/// AD disease profile for MATRIX scoring.
pub const AD_FLARE_PROFILE: DiseaseProfile = DiseaseProfile {
    name: "Atopic dermatitis (flare)",
    barrier_breach_fraction: 0.4,
    effective_dimension: 2.7,
    mean_disorder_w: 0.75,
};

pub const AD_CHRONIC_PROFILE: DiseaseProfile = DiseaseProfile {
    name: "Atopic dermatitis (chronic)",
    barrier_breach_fraction: 0.6,
    effective_dimension: 2.9,
    mean_disorder_w: 0.85,
};

/// Score all drug candidates against a disease profile.
#[must_use]
pub fn score_all_candidates(disease: &DiseaseProfile) -> Vec<AndersonDrugScore> {
    DRUG_CANDIDATES
        .iter()
        .map(|drug| fajgenbaum_matrix_score(drug, disease))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fajgenbaum_matrix_score() {
        let score = fajgenbaum_matrix_score(&DRUG_CANDIDATES[1], &AD_FLARE_PROFILE);
        assert!(score.combined_score > 0.0);
        assert!(score.combined_score <= 1.0);
        assert_eq!(score.mechanism, DrugMechanism::TransductionBlock);
    }

    #[test]
    fn test_score_all_candidates() {
        let flare_scores = score_all_candidates(&AD_FLARE_PROFILE);
        assert_eq!(flare_scores.len(), 6);
        let chronic_scores = score_all_candidates(&AD_CHRONIC_PROFILE);
        assert_eq!(chronic_scores.len(), 6);
        for (f, c) in flare_scores.iter().zip(chronic_scores.iter()) {
            assert!(
                f.combined_score > 0.0 && c.combined_score > 0.0,
                "all scores must be positive"
            );
        }
    }

    #[test]
    fn test_systemic_mab_geometry_penalty() {
        let nemo = fajgenbaum_matrix_score(&DRUG_CANDIDATES[5], &AD_FLARE_PROFILE);
        let tofa = fajgenbaum_matrix_score(&DRUG_CANDIDATES[1], &AD_FLARE_PROFILE);
        assert!(
            tofa.geometry_score > nemo.geometry_score,
            "small molecule reaches tissue better than 145kDa mAb"
        );
    }

    #[test]
    fn test_topical_barrier_bonus() {
        let score_flare = fajgenbaum_matrix_score(&DRUG_CANDIDATES[4], &AD_FLARE_PROFILE);
        let score_chronic = fajgenbaum_matrix_score(&DRUG_CANDIDATES[4], &AD_CHRONIC_PROFILE);
        assert!(
            score_chronic.geometry_score >= score_flare.geometry_score - 0.1,
            "chronic barrier breach helps topical delivery"
        );
    }
}
