# Sub-Thesis 06: Anderson Localization in Immunological Signaling

**Date:** March 2, 2026 (Sessions 105–107)
**Status:** COMPLETE — all nS-601..605 experiments implemented and validated.
Python 48/48, Rust 240/240, 27 unit tests. GPU 4, dispatch 3, mixed hardware 7.
**Modules:** `src/immunological_anderson.rs`, `src/wdm_esn.rs`, `src/training_monitor.rs`
**Validators:** `src/bin/validate_immunological_anderson.rs`, `src/bin/validate_immunological_anderson_extended.rs`, `src/bin/validate_multi_head_esn.rs`, `src/bin/validate_training_monitor.rs`
**Domain:** Immunology x condensed matter physics x pharmacology x drug repurposing
**Novelty:** No prior work applies Anderson localization to cytokine signal
propagation in tissue; no prior work adds spatial geometry to drug repurposing scoring
**Cross-Spring:** wetSpring (Anderson spectral) x neuralSpring (ESN regime classifier,
LSTM time series) x groundSpring (transport, uncertainty, spectral validation)
**baseCamp Paper:** 12

---

## Abstract

We extend the Anderson localization framework from microbial quorum sensing
(Papers 01, 05, 06) to immunological cytokine signaling in skin tissue. The
core observation: Th2 cytokines (IL-4, IL-13, IL-31) are diffusible signals
propagating through a disordered biological medium (heterogeneous skin tissue
with mixed cell populations). The same physics that governs autoinducer
propagation through microbial communities governs cytokine propagation through
inflamed tissue.

We map the atopic dermatitis (AD) disease cycle — allergen exposure -> Th2
activation -> cytokine release -> neuro-immune itch signaling -> barrier
disruption -> amplification — onto the Anderson framework and show that barrier
disruption constitutes a *dimensional promotion* (inverse of the tillage
dimensional collapse in Paper 06): scratching opens 3D diffusion channels
through normally 2D-barrier skin, enabling cytokine signal delocalization.

We then connect this to the Fajgenbaum drug repurposing paradigm (MATRIX,
ARPA-H $48.3M) by adding a spatial geometry dimension to pathway-based
drug-disease scoring: a drug must both (a) target the right pathway AND
(b) physically reach its target through tissue geometry. Anderson localization
quantifies condition (b).

---

## 1. Source Literature — The Gonzales Catalog

### 1.1 Publications to Ingest

All authored or co-authored by Andrea J. Gonzales (Zoetis -> MSU Pharmacology
& Toxicology, 2025-present). These constitute the experimental foundation
for the immunological Anderson extension.

| # | Citation | Key Data | Spring Target |
|---|----------|----------|---------------|
| G1 | Gonzales AJ et al. (2013) "Interleukin-31: its role in canine pruritus and naturally occurring canine atopic dermatitis." *Vet Dermatol* 24:48-53 | IL-31 elevated in AD dog serum; IV IL-31 induces pruritus in beagles; IL-31 activates peripheral nerves | wetSpring: IL-31 as diffusible signal, W mapping from tissue heterogeneity |
| G2 | Gonzales AJ et al. (2014) "Oclacitinib (APOQUEL) is a novel JAK inhibitor with activity against cytokines involved in allergy." *J Vet Pharmacol Ther* 37:317-324 | JAK1 IC50 = 10 nM; blocks IL-2, IL-4, IL-6, IL-13, IL-31 (IC50 36-249 nM); minimal off-target | neuralSpring: dose-response modeling, IC50 as Anderson barrier height |
| G3 | Gonzales AJ et al. (2016) "IL-31-induced pruritus in dogs: a novel experimental model." *Vet Dermatol* 27:34-e10 | Standardized IL-31 pruritus model in beagles; oclacitinib superior to prednisolone/dexamethasone at 1, 6, 11, 16 hr | wetSpring: time-series pruritus data for LSTM; model as controlled Anderson perturbation |
| G4 | Fleck TJ,...,Gonzales AJ (2021) "Onset and duration of action of lokivetmab in IL-31 induced pruritus." *Vet Dermatol* 32:681-e182 | Cytopoint: 3 hr onset, dose-dependent duration (14/28/42 days at 0.125/0.5/2.0 mg/kg); lab model correlates with clinical field trials | neuralSpring: pharmacokinetic decay as signal extinction; ESN classifier for regime transitions |
| G5 | Gonzales AJ et al. (2024) "Oclacitinib is a selective JAK1 inhibitor with efficacy in canine flea allergic dermatitis." *J Vet Pharmacol Ther* 47:447-453 | JAK1 selectivity confirmed in different allergic model | wetSpring: cross-disease validation of same Anderson pathway |
| G6 | McCandless EE, Rugg CA, Fici GJ et al. (2014) "Allergen-induced production of IL-31 by canine Th2 cells and identification of immune, skin, and neuronal target cells." *Vet Immunol Immunopathol* 157:42-48 | IL-31 produced by Th2 cells after allergen presentation by Langerhans cells; target cells = immune, skin, neuronal | wetSpring: cell-type heterogeneity -> disorder W; three-compartment Anderson lattice |

### 1.2 Companion Literature (Not Gonzales-Authored)

| # | Citation | Relevance |
|---|----------|-----------|
| F1 | Fajgenbaum DC et al. (2019) "Identifying and targeting pathogenic PI3K/AKT/mTOR signaling in IL-6 blockade-refractory iMCD." *J Clin Invest* | Proves pathway-based drug repurposing; mTOR cross-talks with JAK/STAT |
| F2 | Every Cure / MATRIX — ARPA-H $48.3M (2024) | 4,000 drugs x 18,000 diseases = 75M pairs scored. Open-source platform. |
| D1 | Simpson et al. (2020) "Dupilumab Phase 3 trials." *N Engl J Med* | Human anti-IL-4Ra for AD — blocks IL-4 + IL-13. Cross-species validation |
| D2 | Silverberg et al. (2023) "JAK inhibitors in AD." *J Am Acad Dermatol* | Upadacitinib, abrocitinib for human AD — human equivalents of Apoquel |
| N1 | Oetjen et al. (2023) "Sensory neurons co-opt immune cells for AD pathogenesis." *Cell* | IL-4/IL-13 directly sensitize sensory neurons — neuro-immune axis |
| N2 | Cohen et al. (2022) "Neuro-immune interactions in AD." *Sci Immunol* | Bidirectional neuron-immune cell communication in skin |

---

## 2. The Anderson Mapping

### 2.1 Tissue as Anderson Lattice

| Anderson QS (Paper 01) | Immunological Extension |
|------------------------|------------------------|
| Lattice site | Cell position in tissue |
| On-site energy e_i | Cell type identity (keratinocyte, Th2, neuron, mast cell, eosinophil) |
| Hopping parameter t | Cytokine diffusion coefficient in extracellular matrix |
| Disorder W | Cell-type heterogeneity (Pielou evenness of cell population) |
| Dimension d | Tissue geometry (epidermis ~ 2D barrier; dermis ~ 3D matrix) |
| Level spacing ratio r | Diagnostic: cytokine signal extended (propagating) vs localized (confined) |

### 2.2 Skin Layer Geometry

| Layer | Thickness | Geometry | Cell types | Anderson prediction |
|-------|-----------|----------|------------|---------------------|
| Stratum corneum | 10-20 um | 2D barrier, dead cells | None (acellular) | Impermeable — no signal propagation |
| Viable epidermis | 50-100 um | Quasi-2D (4-8 cell layers) | Keratinocytes, Langerhans cells, melanocytes | Low d_eff (2-2.5) -> signals localize -> contained |
| Basement membrane | <1 um | 2D boundary | Structural | Barrier between compartments |
| Papillary dermis | 100-200 um | 3D matrix (collagen + vessels + nerves) | Fibroblasts, Th2 cells, mast cells, eosinophils, dendritic cells, nerve endings | d = 3 -> signals propagate -> cytokine signaling active |
| Reticular dermis | 1-3 mm | 3D dense matrix | Fibroblasts, vessels | d = 3, low W -> deep extended regime |

### 2.3 The AD Disease Cycle as Anderson Phase Transitions

```
Healthy skin:
  Epidermis (2D) -> cytokines LOCALIZED (contained, homeostatic)
  Dermis (3D) -> cytokines EXTENDED (but low production = no pathology)

AD initiation:
  Allergen -> Langerhans -> Th2 -> IL-4, IL-13, IL-31 production in dermis
  Dermis (3D) -> cytokines PROPAGATE to sensory nerve endings -> ITCH

Barrier disruption (scratching):
  Epidermis physically breached -> NEW 3D channels through barrier
  d_eff of epidermal layer INCREASES (2D -> quasi-3D)
  Cytokines now propagate from dermis THROUGH barrier to surface
  External allergens now penetrate INTO dermis
  = DIMENSIONAL PROMOTION (inverse of Paper 06 tillage collapse)

Chronic AD:
  Persistent 3D channels -> persistent signal delocalization
  Th2 amplification loop -> increasing W (more immune cell types infiltrate)
  BUT still below W_c in 3D -> signals KEEP propagating -> chronic inflammation

Treatment:
  Cytopoint: removes IL-31 molecule -> no signal to propagate (signal elimination)
  Apoquel: blocks JAK1 receptor -> cells can't respond even if signal arrives (transduction block)
  Barrier repair: restores 2D epidermis -> Anderson localization re-confines signals (geometry intervention)
  Dupilumab: blocks IL-4Ra -> eliminates IL-4 + IL-13 simultaneously (receptor block)
```

### 2.4 The Dimensional Promotion-Collapse Duality

Paper 06 (no-till): Tillage is dimensional COLLAPSE (3D -> 2D) -> QS fails ->
soil ecosystem services collapse.

Paper 12 (AD): Scratching is dimensional PROMOTION (2D -> 3D) -> cytokine
signaling delocalizes -> inflammatory cascade amplifies.

**Same physics, opposite direction, opposite outcome:**
- In soil: losing 3D = losing coordination = BAD
- In AD skin: gaining 3D = gaining pathological propagation = BAD

The Anderson framework is agnostic — it predicts signal propagation. Whether
propagation is beneficial or pathological depends on the biological context.

---

## 3. The Fajgenbaum Bridge — Geometry-Aware Drug Repurposing

### 3.1 Standard MATRIX Score

Fajgenbaum's MATRIX: Score(drug, disease) = f(pathway overlap, literature
evidence, molecular similarity, clinical data)

This is pathway-only. It asks: "Does the drug hit a relevant target?"

### 3.2 Anderson-Augmented Score

Anderson extension: Score(drug, disease, tissue) = f(pathway overlap) x
g(tissue geometry, drug delivery route, molecular size)

The geometry factor g() encodes:
- Can the drug physically reach the target cell in the relevant tissue?
- What is the effective Anderson dimension of the target tissue?
- Does the drug need to cross a 2D barrier (epidermis) to reach a 3D
  compartment (dermis)?
- Large molecules (mAbs like Cytopoint): systemic delivery -> 3D dermal
  access -> good. Topical delivery -> 2D barrier blocks -> poor.
- Small molecules (oclacitinib): oral -> systemic -> 3D dermal access.
  Topical -> can penetrate barrier -> reaches both compartments.

### 3.3 Repurposing Targets for AD (Anderson-Filtered)

| Drug (Original Use) | Pathway | Anderson Geometry Score | Repurposing Logic |
|---------------------|---------|----------------------|-------------------|
| Rapamycin/sirolimus (transplant) | mTOR (cross-talks JAK/STAT via PI3K/AKT) | HIGH — small molecule, systemic, reaches 3D dermis | mTOR activated downstream of IL-4/IL-13 in keratinocytes; Fajgenbaum proved rapamycin works for cytokine storms |
| Tofacitinib (RA) | JAK1/JAK3 | HIGH — already confirmed in human AD trials | Direct pathway match — human equivalent of Apoquel |
| Tanezumab (OA pain, Phase 3) | Anti-NGF mAb | HIGH — systemic mAb reaches 3D dermis | NGF elevated in AD skin; Gonzales's team already proved anti-NGF works in OA (Librela/Solensia) |
| Trametinib (melanoma) | MEK/ERK (downstream IL-31RA) | MODERATE — systemic, but MEK inhibition has broad effects | ERK pathway activated by IL-31; could modulate keratinocyte dysfunction |
| Crisaborole (mild AD, topical) | PDE4 | LOW -> MODERATE — topical, must cross 2D barrier | Already approved for AD but limited by penetration; Anderson predicts better efficacy in barrier-compromised skin |
| Nemolizumab (prurigo nodularis) | Anti-IL-31RA mAb | HIGH — systemic, targets same receptor as Cytopoint | Direct IL-31 pathway; human equivalent of Cytopoint approach |

---

## 4. neuralSpring Integration

### 4.1 Existing Primitives (Ready)

All of these were implemented in Session 105 and are directly applicable to
Paper 12's neuralSpring connections without additional code.

| Primitive | Module | Application to Paper 12 |
|-----------|--------|------------------------|
| `MultiHeadWdmClassifier` | `wdm_esn.rs` | Multi-head ESN for AD skin state classification: regime label (healthy/flare/chronic/treated), spectral bandwidth (cytokine signal extent), confidence head |
| `HeadGroupDisagreement` | `wdm_esn.rs` (via barracuda) | Uncertainty quantification for AD regime classification — high disagreement at Anderson transition boundary |
| `TrainingMonitor` | `training_monitor.rs` | Brain-inspired FSM (GREEN/YELLOW/RED) for pharmacokinetic trajectory monitoring; DriftMonitor detects treatment regime transitions |
| `TrainingInterrupt` | `training_monitor.rs` | Adaptive control: ReduceLearningRate at YELLOW (dose adjustment), EarlyStop at RED (treatment failure) |
| `Dispatcher::kl_divergence` | `gpu_dispatch/dispatch_stats.rs` | Distribution shift detection in cytokine profiles — GPU-accelerated for high-throughput screening |
| `SpectralNautilusBridge` | `nautilus_bridge.rs` | Bridge spectral analysis (IPR, level spacing ratio) to DriftMonitor observations |
| `export_npu_weights` | `wdm_esn.rs` | AKD1000 NPU export for edge deployment of AD classifier (veterinary point-of-care) |

### 4.2 Reproduction Targets (neuralSpring-Specific)

| # | Source Paper | What to Reproduce | Method | Validates |
|---|-------------|-------------------|--------|-----------|
| nS-601 | Gonzales (2014) G2 | IC50 dose-response curves: JAK1 vs JAK2 vs JAK3 | `MultiHeadWdmClassifier` with concentration as input feature | IC50 as Anderson barrier height — drug concentration maps to effective W reduction |
| nS-602 | Gonzales (2016) G3 | Pruritus score time series: 1, 6, 11, 16 hr for oclacitinib vs steroids | LSTM (nW-03 architecture, R^2=0.98) | Pharmacodynamic prediction of treatment response |
| nS-603 | Fleck/Gonzales (2021) G4 | Dose-dependent duration curves: 0.125/0.5/2.0 mg/kg lokivetmab | ESN regime classifier with `TrainingMonitor` for decay tracking | Pharmacokinetic decay as signal extinction in Anderson model |
| nS-604 | McCandless (2014) G6 | Three-compartment classification: immune + skin + neural targets | `MultiHeadWdmClassifier` with 3 heads mapped to tissue compartments | Multi-compartment Anderson lattice from cell-type data |
| nS-605 | Fajgenbaum (2019) F1 | Anderson-augmented MATRIX score: pathway x geometry | `Dispatcher::kl_divergence` for distribution comparison across tissue compartments | Geometry-aware drug repurposing scoring |

### 4.3 Experiment Design Notes

**nS-601 (IC50 dose-response)**:
- Input: drug concentration (log-scale), cytokine panel (IL-2, IL-4, IL-6, IL-13, IL-31)
- Output head 0 (regime): effective/subtherapeutic/toxic
- Output head 1 (spectral): signal propagation bandwidth at given concentration
- Output head 2 (confidence): classification certainty
- Validation: compare predicted IC50 with Gonzales (2014) published values
  (JAK1 = 10 nM, IL-31 = 36-249 nM range)

**nS-602 (pruritus time series)**:
- Input: treatment type (oclacitinib/prednisolone/dexamethasone/vehicle), time post-dose
- Output: predicted pruritus score
- Architecture: LSTM from nW-03 with modified input embedding
- Validation: compare predicted vs observed at 1, 6, 11, 16 hr timepoints

**nS-603 (pharmacokinetic decay)**:
- Input: lokivetmab dose (mg/kg), time post-injection (days)
- ESN reservoir tracks signal extinction trajectory
- `TrainingMonitor` detects regime transitions: therapeutic -> subtherapeutic
- Validation: predicted duration vs observed (14/28/42 days at 0.125/0.5/2.0)

**nS-604 (three-compartment)**:
- Three heads of `MultiHeadWdmClassifier` assigned to:
  - Head 0: immune cell compartment (Th2, mast cells, eosinophils)
  - Head 1: skin cell compartment (keratinocytes, Langerhans)
  - Head 2: neural compartment (sensory nerve endings)
- `HeadGroupDisagreement` measures cross-compartment coherence
- High disagreement = signals localized within compartments (healthy)
- Low disagreement = signals propagating across compartments (AD flare)

**nS-605 (Anderson-augmented MATRIX)**:
- Compute pathway overlap score (standard MATRIX approach)
- Compute geometry factor via Anderson d_eff estimation
- `kl_divergence` between drug tissue distribution and target cell distribution
- Combined score = pathway_score * geometry_factor
- Validate against known AD drug efficacy rankings

---

## 5. Cross-Paper Connections

### Paper 01 -> Paper 12
Anderson QS in microbial communities -> Anderson cytokine signaling in tissue.
Same math, different biology. The level spacing ratio r, disorder W, dimension d,
and W_c all transfer directly.

### Paper 04 -> Paper 12
Sentinel microbes detect environmental perturbation via Anderson regime shift.
Paper 12 extends: immune cell populations detect disease perturbation (AD flare)
via the same Anderson regime shift. The ESN classifier (validated on AKD1000)
can classify AD tissue state from cytokine measurements.

### Paper 05 -> Paper 12
Cross-species signaling in symbiotic systems (lichen, coral, rhizobia).
Paper 12 extends: cross-cell-type signaling in immunological systems (Th2 -> neuron,
mast cell -> keratinocyte, eosinophil -> fibroblast). Same Anderson geometry
governs whether signals reach their cross-type targets.

### Paper 06 -> Paper 12
No-till = dimensional collapse -> QS fails -> ecosystem services lost.
AD scratching = dimensional promotion -> cytokine delocalization -> pathological cascade.
Same physics, opposite direction. Duality documented.

### Sub-thesis 01 -> Sub-thesis 06
Weight spectral analysis detects Anderson localization in neural networks.
Sub-thesis 06 applies the same spectral diagnostics (IPR, level spacing ratio)
to cytokine signal propagation in tissue lattices. `weight_spectral.rs` metrics
transfer directly to tissue lattice analysis.

### Sub-thesis 05 -> Sub-thesis 06
Multi-agent QS coordination uses Anderson framework to predict phase transitions
in decentralized AI systems. Sub-thesis 06 applies the same framework to
decentralized immune cell coordination in tissue — immune cells are the "agents,"
cytokines are the "signals," tissue geometry is the "interaction network."

---

## 6. Gonzales Lab Collaboration Potential

Whether or not the RA III position is obtained, the Gonzales lab is a natural
collaborator for ecoPrimals:

**What she has that we need:**
- Empirical cytokine data (IL-31 pruritus scores, dose-response curves)
- iPSC-derived skin models across species (canine, feline, human)
- Plate-based screening infrastructure
- Drug discovery pipeline and regulatory expertise
- 18 years of Zoetis data and institutional knowledge

**What we have that she needs:**
- Computational spatial modeling (Anderson framework — no one else has this)
- Statistical analysis and ML (data science for screening data)
- Bioinformatics (sequencing analysis, NCBI pipelines)
- Automated data workflows (Visual Basic -> Python/Rust)
- Drug repurposing framework connection (Fajgenbaum bridge)

---

## 7. Open Questions for Spring Evolution

1. What is W for inflamed vs healthy dermal tissue? (Need: single-cell
   transcriptomics data to compute Pielou evenness of cell populations)
2. What is the effective d_eff of barrier-disrupted epidermis? (Need:
   3D imaging of AD skin to quantify channel geometry)
3. Does the Anderson W_c hold for cytokine propagation as it does for
   QS autoinducers? (Need: diffusion coefficient data for IL-31 in ECM)
4. Can the ESN regime classifier distinguish AD flare from healthy skin
   using cytokine panel data? (Need: published cytokine profiling datasets)
5. Does rapamycin's efficacy in cytokine storms (Fajgenbaum) predict
   efficacy in AD via the mTOR/JAK cross-talk? (Testable with Gonzales's
   plate-based screening infrastructure)

---

## 8. wetSpring Experiments (Proposed)

These are cross-spring experiments proposed for wetSpring, documented here
for completeness and to define the neuralSpring data consumption interfaces.

| Exp | Description | neuralSpring Consumes |
|-----|-------------|----------------------|
| Exp 270 | Anderson lattice with skin-layer geometry: 2D epidermis (L=5-8) + 3D dermis (L=20) + barrier interface. Compute r for cytokine propagation across layers | Level spacing ratio r -> ESN input feature for regime classification |
| Exp 271 | Barrier disruption model: remove sites from 2D epidermal layer -> measure r transition as d_eff increases -> quantify "dimensional promotion" threshold | d_eff trajectory -> TrainingMonitor input for regime transition detection |
| Exp 272 | Cell-type heterogeneity sweep: vary W (immune cell diversity) in 3D dermal compartment -> confirm cytokine signals remain extended up to W_c | W vs r curve -> dose-response analog (W_reduction = drug effect) |
| Exp 273 | NCBI Protein search: IL-31RA, IL-4Ra, OSMR expression in skin tissue metagenomes -> map receptor distribution as lattice site occupancy | Gene expression -> lattice construction -> ESN input |
| Exp 274 | Cross-species skin comparison: canine (thin epidermis) vs human (thick epidermis) -> different d_eff barriers -> different Anderson predictions | Species-specific d_eff -> transfer learning validation |

---

## 9. groundSpring Connections

| Experiment | Application |
|------------|-------------|
| Exp 012 — Spin chain transport | Models cytokine signal propagation distance through linear tissue channels (nerve tracts, vessels) |
| Exp 008 — Anderson localization | Validates 2D/3D spectral diagnostics used for skin compartment classification |
| Exp 015 — Uncertainty bridge | Sensor noise -> cytokine measurement uncertainty -> Anderson regime classification confidence |
| Exp 018 — Band edge structure | Tissue periodicity (epidermal cell layers) creates band gaps for cytokine propagation — predicts frequency-dependent signal filtering |
