# neuralSpring baseCamp: Biophysical AI Interpretability

**Date**: May 19, 2026 (Sessions 49–213 — S213: lithoSpore Audit Absorption (39 capabilities with stability tiers). Python→Rust→Primal→Live Composition 4-tier validation of peer-reviewed science. V169 handoff. 754 workspace tests (IPC-first). **269 binaries**, zero clippy, 4,900+ checks. 10 validation scenarios. 6/6 GPU dispatch. barraCuda v0.4.0)
**Author**: Kevin Mok (BS Microbiology, MSU 2018; MS Data Science, MSU 2025)

---

## What This Is

These are independent scientific explorations that apply neuralSpring's
validated physics and biology primitives (spectral analysis, signal
propagation, dynamical systems, game theory) to understanding AI systems
as physical systems. They are companion papers, not thesis chapters.

Each document stands alone as a potential publication. Together they
demonstrate that the same 6 computational primitives (GEMM, Attention,
Normalization, Nonlinearity, Reduction, Gating) validated across 25
papers and 7 scientific domains also reveal the physical structure
hidden inside neural networks.

## The Thesis Angle

wetSpring took Anderson localization (condensed matter physics) and
applied it to quorum sensing (microbiology) — no prior work had done
this. The result: 3D geometry is necessary and sufficient for QS
(2,992+ checks, gen3 baseCamp Sub-thesis 01).

**neuralSpring takes the same spectral/dynamical-systems primitives
and applies them to understanding AI systems as physical systems.**

- The neural network IS the disordered medium.
- The weight matrix IS the Hamiltonian.
- Information propagation through layers IS wave propagation through
  a lattice.
- Loss landscapes ARE energy landscapes.
- Multi-agent AI coordination IS quorum sensing.

This is neuralSpring's niche: **Biophysical AI Interpretability** —
using validated physics and biology primitives to make black-box AI
into predictive, interpretable physical systems.

## How These Relate to the Main Thesis

The constrained evolution thesis argues that environmental constraints
reshape fitness landscapes and drive specialization. These baseCamp
explorations test that argument in the AI domain:

- **Sub-thesis 01**: Architectural constraints produce universal spectral
  signatures in weight matrices — convergent evolution at the matrix level.
- **Sub-thesis 02**: Gating mechanisms in LSTMs obey the same physics as
  signal propagation in disordered lattices — substrate-independent dynamics.
- **Sub-thesis 03**: Loss landscapes have the same topology as molecular
  energy landscapes — convergent landscape geometry.
- **Sub-thesis 04**: Neural networks approximate probabilistic graphical
  models — convergent inference algorithms.
- **Sub-thesis 05**: Multi-agent AI coordination exhibits QS phase
  transitions — convergent coordination strategies across biology and AI.
- **Sub-thesis 06**: Immunological cytokine signaling follows Anderson
  physics — cytokine propagation in tissue obeys the same dimensional
  rules as microbial QS, enabling geometry-aware drug repurposing.

```
                    Main Thesis
            (Constrained Evolution)
                    |
        +-----------+-----------+
        |           |           |
     Theory      System    Validation
   (Chs 3-4)   (Chs 5-6)  (Chs 7-12)
                    |
            --------+--------
            |               |
      Spring Papers    baseCamp Papers
      (reproduce)      (explore)
                            |
               +----+----+----+----+----+
               |    |    |    |    |    |
             nS01 nS02 nS03 nS04 nS05 nS06
```

Spring papers reproduce published work to validate the infrastructure.
baseCamp papers use that validated infrastructure to explore new science.

## The Papers

| # | Title | Domain | Grounding Papers | Experiments | Key Primitive |
|---|-------|--------|:----------------:|:-----------:|---------------|
| 01 | [Weight Matrices as Disordered Hamiltonians](sub01_weight_hamiltonians.md) | Random matrix theory x Deep learning | 3 | 6 | `eigh_f64`, `BatchIprGpu` |
| 02 | [Information Flow as Wave Propagation](sub02_information_propagation.md) | Statistical physics x Recurrent AI | 3 | 6 | `hmm.rs`, `stencil_cooperation.wgsl` |
| 03 | [Loss Landscapes as Energy Landscapes](sub03_loss_landscapes.md) | Chemical physics x Optimization | 3 | 5 | `rk45_adaptive.wgsl`, `eigh_f64` |
| 04 | [Neural Networks as Probabilistic Graphical Models](sub04_neural_pgm.md) | Bayesian inference x Interpretability | 3 | 6 | `hmm.rs`, `introgression.rs` |
| 05 | [Multi-Agent AI Coordination as Quorum Sensing](sub05_multiagent_qs.md) | Microbial ecology x Multi-agent AI | 3 | 5 | `anderson_localization.rs`, `game_theory.rs` |
| 06 | [Anderson Localization in Immunological Signaling](sub06_immunological_anderson.md) | Immunology x condensed matter x drug repurposing | 6 | 0 (proposed) | `wdm_esn.rs`, `training_monitor.rs` |

## What Makes This Novel

- **Nobody has applied Anderson localization IPR to neural network weight
  matrices** to predict generalization (Sub-thesis 01)
- **Nobody has modeled LSTM gating as stencil propagation on a disordered
  lattice** (Sub-thesis 02)
- **Nobody has used GPU-accelerated RK45 ODE integration for transition-state
  analysis of loss landscapes** in the EL4ML framework (Sub-thesis 03)
- **Nobody has combined HMM introgression detection with PGM extraction**
  to detect "knowledge transfer" between neural network layers (Sub-thesis 04)
- **Nobody has applied the Anderson QS framework to multi-agent AI
  coordination** (Sub-thesis 05)
- **Nobody has applied Anderson localization to cytokine signal
  propagation in tissue** or added spatial geometry to drug repurposing
  scoring (Sub-thesis 06)

All six use primitives neuralSpring has already validated at 4100+ checks
across 25 papers + 5 WDM surrogates + 3 publication experiments + sovereign
folding (nF-01, nF-02). The extensions require composition, not new math.

### gen3 baseCamp Cross-References

Each neuralSpring sub-thesis connects directly to gen3 baseCamp sub-theses:

| nS Sub-thesis | gen3 Connection | Shared Primitives |
|:-------------:|:---------------:|:-----------------:|
| 01 (Weight Hamiltonians) | gen3 01 (Anderson QS) | `eigh_f64`, `BatchIprGpu`, level spacing ratio |
| 02 (Information Flow) | gen3 04 (Sentinels), gen3 06 (No-till) | LSTM, HMM, stencil |
| 03 (Loss Landscapes) | gen3 07 (Sovereign WDM) | RK45, Hessian, Boltzmann |
| 04 (Neural PGM) | gen3 02 (LTEE), gen3 05 (Cross-species) | HMM, introgression, PhyloNet-HMM |
| 05 (Multi-Agent QS) | gen3 01 (Anderson QS), gen3 03 (Bioag) | Anderson, game theory, Wright-Fisher |
| 06 (Immunological Anderson) | gen3 01 (Anderson QS), gen3 05 (Cross-species), gen3 06 (No-till) | Anderson, ESN, LSTM, KL divergence |

### Implementation Status (Sessions 50–55, hardened 61–85, expanded S105)

All 5 sub-theses have core Rust modules implemented and validated at CPU,
GPU, and mixed-hardware tiers:

| # | Module | CPU Checks | GPU Checks | Status |
|---|--------|-----------|-----------|--------|
| 01 | `src/weight_spectral.rs` | 21/21 | 14/14 (shared) | **PASS** |
| 02 | `src/information_flow.rs` | 22/22 | — | **PASS** |
| 03 | `src/loss_landscape.rs` | 27/27 | — | **PASS** |
| 04 | `src/neural_pgm.rs` | 21/21 | — | **PASS** |
| 05 | `src/agent_coordination.rs` | 23/23 | — | **PASS** |
| 06 | `src/wdm_esn.rs`, `src/training_monitor.rs` | 0/0 | — | **PASS** |
| — | `validate_basecamp_gpu` | — | 14/14 | **PASS** |
| — | `validate_compute_dispatch` | 16/16 | — | **PASS** |
| — | `validate_mixed_hardware` | 14/14 | — | **PASS** |
| **Total** | **7 modules + 3 validators** | **114+30** | **14** | **128/128 PASS** + nS06 proposed |

Session 54 expanded experiments (82→114 CPU checks, nS-103..505).
Session 55 added CPU↔GPU dispatch parity and metalForge mixed-hardware routing.

## Faculty Anchors

| Sub-thesis | Faculty | Institution | Connection |
|:----------:|---------|------------|------------|
| 01 | Michael Mahoney | UC Berkeley (Statistics) | Weight matrix spectral analysis |
| 01 | Umut Simsekli | Inria/ENS Paris | Stochastic dynamics of SGD |
| 02 | Surya Ganguli | Stanford (Applied Physics) | Mean-field theory of deep networks |
| 03 | David Wales | Cambridge (Chemistry) | Energy Landscapes for ML program |
| 04 | Yee Whye Teh | Oxford/DeepMind | Probabilistic ML, Bayesian deep learning |
| 05 | Emily Dolson | MSU (Computer Science) | Already validated (Papers 011-015) |
| 06 | Andrea J. Gonzales | MSU (Pharmacology & Toxicology) | IL-31/JAK1 immunological signaling, drug repurposing |
| 06 | David Fajgenbaum | UPenn (Medicine) / Every Cure | MATRIX drug repurposing platform, mTOR/JAK cross-talk |

## Cross-Spring Connections

| Sub-thesis | hotSpring Primitive | wetSpring Primitive |
|:----------:|--------------------|--------------------|
| 01 | — | Anderson QS (IPR, level spacing) |
| 02 | — | QS signal propagation (stencil) |
| 03 | MD energy minimization (RK4/RK45), Boltzmann sampling | — |
| 04 | — | HMM phylogenetics (belief propagation) |
| 05 | — | Anderson QS dimensional analysis, Waters game theory |
| 06 | — | Anderson QS (IPR, level spacing, 2D/3D), cytokine diffusion |

## Validated Primitive Inventory

All baseCamp experiments build on primitives validated at 3162+ checks:

| Primitive | Papers Using It | GPU Status | baseCamp Use |
|-----------|:---------------:|:----------:|:------------:|
| `eigh_f64` | 022-023 | GPU | nS01, nS02, nS03, nS04 |
| `BatchIprGpu` | 022-023 | GPU | nS01, nS02, nS05 |
| `hmm.rs` (forward/backward) | 016-018 | GPU | nS02, nS04 |
| `stencil_cooperation.wgsl` | 019 | GPU | nS02, nS05 |
| `rk45_adaptive.wgsl` | 020-021 | GPU | nS03 |
| `game_theory.rs` / `SpatialPayoffGpu` | 019 | GPU | nS03, nS05 |
| `introgression.rs` | 018 | GPU | nS04 |
| `WrightFisherGpu` | 024-025 | GPU | nS05 |
| `swarm_robotics.rs` / `SwarmNnGpu` | 015 | GPU | nS05 |
| `anderson_localization.rs` | 022-023 | GPU | nS01, nS05 |
| `spectral_commutativity.rs` | 022 | GPU | nS01, nS04 |
| `signal_integration.rs` / `HillGateGpu` | 021 | GPU | nS02 |
| `wdm_esn.rs` / `MultiHeadWdmClassifier` | nW-05 | CPU | nS06 |
| `training_monitor.rs` / `TrainingMonitor` | — | CPU | nS06 |
| `gpu_dispatch::Dispatcher::kl_divergence` | all | GPU | nS06 |
| `gpu_dispatch::Dispatcher` | all | GPU | all |

## Reading Order

**For a physicist**: 01 (weight Hamiltonians) -> 02 (wave propagation)
-> 03 (energy landscapes)

**For an ML researcher**: 04 (PGM extraction) -> 01 (spectral analysis)
-> 02 (information flow)

**For a biologist**: 05 (multi-agent QS) -> 02 (biological gating
analogy) -> 06 (immunological Anderson)

**For an immunologist / pharmacologist**: 06 (immunological Anderson) ->
05 (multi-agent QS) -> 01 (spectral foundations)

**For a PhD committee**: 01 (novel contribution, strongest theoretical
grounding) -> 04 (practical interpretability value) -> 05 (cross-domain
bridge to wetSpring) -> 06 (translational bridge to drug repurposing)

## Data and Reproducibility

All experimental data is generated by neuralSpring binaries (AGPL-3.0)
using weights from our Phase 0/0+/0++ training runs. Deterministic
seed (42). No proprietary models. No external datasets beyond our
existing open baselines (ERA5 CC-BY-4.0, MNIST CC-BY-SA-3.0). No model
downloads.

```bash
cd neuralSpring
cargo run --release --bin validate_weight_spectral       # nS01 (15 checks)
cargo run --release --bin validate_information_flow       # nS02 (15 checks)
cargo run --release --bin validate_loss_landscape         # nS03 (19 checks)
cargo run --release --bin validate_neural_pgm             # nS04 (15 checks)
cargo run --release --bin validate_agent_coordination     # nS05 (18 checks)
```

## Priority Order

| Priority | Sub-thesis | Effort | Impact | Thesis Connection |
|:--------:|:----------:|:------:|:------:|:-----------------:|
| 1 | **01: Weight Hamiltonians** | Medium | Very High | Strongest novel claim, cleanest primitives |
| 2 | **04: Neural PGM** | Medium | Very High | Highest practical value (interpretability) |
| 3 | **02: Information Flow** | Medium | High | Deepest cross-domain connection |
| 4 | **05: Multi-Agent QS** | Low | High | Direct wetSpring bridge |
| 5 | **03: Loss Landscapes** | High | High | Strongest hotSpring bridge |
| 6 | **06: Immunological Anderson** | Medium | Very High | Translational: drug repurposing + Gonzales collaboration |

---

*neuralSpring baseCamp: Biophysical AI Interpretability + Translational
Immunology. 6 sub-theses, 21 grounding papers, 29 experiments (28 complete
+ 1 Session 61) + 5 proposed (nS-601..605), all built on 4100+ validated
checks across 26 papers + 5 WDM surrogates + 3 publication experiments
and 7+ scientific domains. Core primitives implemented in Sessions 50–55,
quality-hardened Sessions 61–90, extended Session 105, audited Session 139:
7 Rust modules, 8 validation binaries, 128/128 PASS (114 CPU + 14 GPU),
1048 lib tests, 0 clippy warnings (pedantic+nursery, all-features), 0 doc
warnings. 46 upstream rewires to upstream BarraCUDA. 92% line coverage.
Sub-thesis 06 (Session 105): Anderson localization in immunological
signaling — Gonzales catalog (6 papers), Fajgenbaum drug repurposing
bridge, dimensional promotion-collapse duality with Paper 06. neuralSpring
primitives ready: MultiHeadWdmClassifier, TrainingMonitor,
Dispatcher::kl\_divergence. gen3 baseCamp sub-theses 01–07 now
cross-referenced with neuralSpring connections. No new math — only novel
composition of validated primitives.*
