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
//! - [`TrainingMonitor`](crate::training_monitor::TrainingMonitor): pharmacokinetic trajectory monitoring via FSM
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

pub mod classification;
pub mod lattice;
pub mod matrix;
pub mod pharma;

pub use classification::*;
pub use pharma::*;

// Re-export submodule types for backward compatibility.
#[cfg(feature = "barracuda")]
pub use lattice::barrier_promotion_spectrum;
pub use lattice::{
    ThreeCompartmentDisorder, level_spacing_ratio, three_compartment_disorder,
    tissue_lattice_hamiltonian,
};
pub use matrix::{
    AD_CHRONIC_PROFILE, AD_FLARE_PROFILE, DRUG_CANDIDATES, DiseaseProfile, DrugCandidate,
    fajgenbaum_matrix_score, score_all_candidates,
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
    /// Tissue layer name (barrier identity).
    pub name: &'static str,
    /// Thickness range in micrometers (min, max).
    pub thickness_um: (f64, f64),
    /// Effective Anderson dimension for propagation in this layer.
    pub effective_dimension: f64,
    /// Whether the layer is acellular (no resident cells).
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
