# neuralSpring baseCamp — Cross-Spring Handoff

**Date**: February 23, 2026
**neuralSpring Session**: 49 (baseCamp research program definition)
**Audience**: hotSpring team, wetSpring team, gen3 thesis committee
**Purpose**: Announce neuralSpring's novel research program and its
dependencies on cross-spring primitives

---

## Part 1: What neuralSpring baseCamp Is

neuralSpring has defined a novel research program called **BioPhysical AI
Interpretability** — applying validated physics and biology primitives to
understanding AI systems as physical systems. The program consists of five
sub-theses, each grounded in 3 published academic papers and using existing
validated primitives from neuralSpring's 25-paper, 1800+ check validation
corpus.

The thesis angle: wetSpring took Anderson localization and applied it to
quorum sensing. **neuralSpring takes the same spectral/dynamical-systems
primitives and applies them to AI.** The weight matrix IS the Hamiltonian.
Information propagation through layers IS wave propagation through a lattice.
Loss landscapes ARE energy landscapes.

| Sub-Thesis | Novel Claim | Experiments |
|:----------:|------------|:-----------:|
| nS-01: Weight Hamiltonians | Weight matrix IPR predicts generalization | 6 |
| nS-02: Information Flow | LSTM gating is stencil propagation on a disordered lattice | 6 |
| nS-03: Loss Landscapes | Loss landscapes are energy landscapes (EL4ML) | 5 |
| nS-04: Neural PGM | DNN forward pass = belief propagation on tree PGM | 6 |
| nS-05: Multi-Agent QS | Anderson framework predicts AI coordination phase transitions | 5 |

Full documents: `whitePaper/baseCamp/sub01_weight_hamiltonians.md` through
`sub05_multiagent_qs.md`. Program overview: `whitePaper/baseCamp/extensions.md`.

---

## Part 2: What neuralSpring Needs from hotSpring

### Sub-Thesis 03: Loss Landscapes as Energy Landscapes

This sub-thesis is the direct bridge to hotSpring. The claim: neural network
loss landscapes are energy landscapes in the statistical mechanics sense.
SGD training is a molecular dynamics trajectory. Saddle points between loss
minima are transition states.

**hotSpring primitives we need or plan to adapt:**

| Primitive | hotSpring Source | neuralSpring Use | Status |
|-----------|----------------|-----------------|--------|
| RK4/RK45 ODE integration | Sarkas MD, TTM | NEB transition path sampling on loss surface | **Validated** — `rk45_adaptive.wgsl` |
| Boltzmann sampling | Plasma EOS, nuclear | Boltzmann-weighted weight space exploration | **Validated** — `rng.rs` (Xoshiro256**) + Metropolis |
| Energy minimization | MD equilibration | Loss landscape local minimum characterization | **Validated** — `eigh_f64` for Hessian diag |
| Hessian eigendecomposition | Phonon dispersion | Loss surface curvature analysis | **Validated** — `eigh_f64`, `BatchIprGpu` |
| Disconnectivity graphs | (new) | Wales EL4ML topology visualization | **Needed** — candidate for hotSpring development |

**What hotSpring should know:**

1. **The EL4ML program** (David Wales, Cambridge) directly uses MD tools
   for ML analysis. Wales' group has published disconnectivity graph
   software for molecular energy landscapes. neuralSpring proposes the
   first GPU-accelerated application to neural network loss landscapes.

2. **RK45 on loss surfaces**: Our `rk45_adaptive.wgsl` is validated for
   biological ODE systems (Hill function dynamics, regulatory networks).
   Loss landscape NEB requires the same integrator but with neural network
   gradient evaluation as the force function.

3. **Boltzmann sampling of weight space**: Pittorino et al. (2025) show
   that high-entropy weight configurations generalize better. This requires
   MCMC sampling with our validated RNG at GPU scale.

**Action items for hotSpring:**

- [ ] Review `sub03_loss_landscapes.md` for RK45/Boltzmann primitive needs
- [ ] Consider developing a disconnectivity graph primitive (graph topology
  from energy/loss minima and transition states) — useful for both MD and ML
- [ ] Share any existing NEB (nudged elastic band) implementations — we will
  adapt for loss landscape transition state search

---

## Part 3: What neuralSpring Needs from wetSpring

### Sub-Thesis 01: Weight Matrices as Disordered Hamiltonians

This sub-thesis directly extends wetSpring's Anderson QS work. The same
`eigh_f64` and `BatchIprGpu` tools that compute IPR for QS lattice
eigenstates now compute IPR for neural network weight matrix eigenstates.

**wetSpring primitives we use:**

| Primitive | wetSpring Source | neuralSpring Use | Status |
|-----------|----------------|-----------------|--------|
| `eigh_f64` | Anderson localization (Papers 022-023) | Weight matrix eigendecomposition | **Validated** |
| `BatchIprGpu` | Anderson localization | IPR of weight matrix eigenstates | **Validated** |
| Level spacing ratio | Anderson phase diagnostic | GOE vs Poisson diagnostic for weight spectra | **Validated** |
| `anderson_localization.rs` | Anderson Hamiltonian construction | Treat weight matrix as disordered Hamiltonian | **Validated** |
| `spectral_commutativity.rs` | Kachkovskiy (Paper 022) | Test whether layers approximately commute | **Validated** |

**What wetSpring should know:**

1. **The exact same Anderson diagnostics** (IPR, level spacing ratio, Marchenko-
   Pastur departure) that characterize QS signal propagation in microbial
   communities also predict neural network generalization. Localized eigenstates
   = memorization. Extended eigenstates = generalization.

2. **The 3D/2D dimensional split** from Anderson QS may have an analog in
   AI: interaction topologies with 3D-like connectivity (dense graphs) sustain
   multi-agent coordination, while 2D/1D-like topologies fail (Sub-thesis 05).

3. **Ouyang (2025)** directly applies Anderson localization to GNN over-smoothing.
   This is independent convergent discovery — the GNN community rediscovered
   Anderson physics without knowing about the wetSpring QS application.

### Sub-Thesis 05: Multi-Agent AI Coordination as Quorum Sensing

This sub-thesis applies the Anderson QS framework to multi-agent AI systems.

**wetSpring primitives we use:**

| Primitive | wetSpring Source | neuralSpring Use | Status |
|-----------|----------------|-----------------|--------|
| Anderson QS framework | Gen3 Sub-thesis 01 | Graph Laplacian localization for agent coordination | **Validated** |
| `stencil_cooperation.wgsl` | Waters QS (Paper 019) | Signal diffusion in agent interaction lattice | **Validated** |
| `game_theory.rs` | Waters cooperation dynamics | Replicator dynamics for agent strategy evolution | **Validated** |
| `WrightFisherGpu` | Population genetics (Paper 024-025) | Agent population selection/replacement | **Validated** |

**What wetSpring should know:**

1. **Dimensional phase diagram**: We predict that agent coordination on 3D
   interaction topologies succeeds at all disorder levels, while 1D/2D fails —
   directly replicating the wetSpring Anderson QS result in the AI domain.

2. **NP solutions**: The three "NP solutions" from wetSpring's Anderson QS
   work (logic inversion, self-organized geometry, signal relay) should emerge
   independently in multi-agent AI systems under coordination pressure.

3. **We already have the base model**: Paper 015 (Foreback & Dolson, swarm
   robotics) is validated at all 7 tiers. Sub-thesis 05 extends it with
   QS-style signaling.

**Action items for wetSpring:**

- [ ] Review `sub01_weight_hamiltonians.md` and `sub05_multiagent_qs.md`
- [ ] Share the dimensional analysis pipeline from Anderson QS experiments
  (28 biome types x 3 dimensions) — we will adapt for agent topologies
- [ ] Consider: do wetSpring's QS biome results provide training data for
  a predictor of when multi-agent AI coordination will succeed?

---

## Part 4: Cross-Spring Primitive Flow

```
hotSpring                    neuralSpring                   wetSpring
========                     ============                   =========
RK4/RK45 ODE ──────────────→ nS-03: Loss landscapes         │
Boltzmann sampling ─────────→ nS-03: Weight sampling         │
Energy minimization ────────→ nS-03: Hessian analysis        │
                              │                              │
                              nS-01: Weight Hamiltonians ←───┤ Anderson QS (IPR, r)
                              nS-02: Info flow ←─────────────┤ Stencil propagation
                              nS-04: Neural PGM ←────────────┤ HMM phylogenetics
                              nS-05: Multi-agent QS ←────────┤ Anderson 3D, game theory
                              │
                              ↓
                         All use neuralSpring's
                         validated Rust + GPU stack
                         (1800+ checks, 25 papers)
```

---

## Part 5: What Both Springs Should Know About the Faculty Anchors

| Sub-Thesis | Faculty | Institution | Relevance |
|:----------:|---------|------------|-----------|
| nS-01 | Michael Mahoney | UC Berkeley (Statistics) | Weight matrix spectral analysis. Published Martin & Mahoney 2021 (JMLR) |
| nS-01 | Umut Simsekli | Inria/ENS Paris | Stochastic dynamics of SGD. Dyson Brownian motion for weight spectra |
| nS-02 | Surya Ganguli | Stanford (Applied Physics) | Mean-field theory of deep networks. Edge-of-chaos trainability |
| nS-03 | David Wales | Cambridge (Chemistry) | Energy Landscapes for ML program (EL4ML). Disconnectivity graphs |
| nS-04 | Yee Whye Teh | Oxford/DeepMind | Probabilistic ML, Bayesian deep learning. Factor graphs |
| nS-05 | Emily Dolson | MSU (Computer Science) | Already validated (Papers 011-015). Direct extension of Paper 015 |

These faculty are NOT collaborators — they are independent researchers
whose published methods ground our sub-theses. We reproduce their work
and extend it with novel applications of validated primitives.

---

## Part 6: Data and Reproducibility

All baseCamp experiments use:
- Weight matrices from neuralSpring's Phase 0/0+/0++ training runs
- Deterministic seed (42)
- No proprietary models (no Claude, GPT, or any download)
- No external datasets beyond existing open baselines (ERA5 CC-BY-4.0, MNIST CC-BY-SA-3.0)
- AGPL-3.0 throughout

All primitives are GPU-accelerated (~90% of production math via
`gpu_dispatch::Dispatcher`). Multi-GPU validated (RTX 4070 + TITAN V NVK,
bit-identical).

---

## Part 7: What Makes This Novel

- **Nobody has applied Anderson localization IPR to neural network weight
  matrices** to predict generalization (nS-01)
- **Nobody has modeled LSTM gating as stencil propagation on a disordered
  lattice** (nS-02)
- **Nobody has used GPU-accelerated RK45 ODE integration for transition-state
  analysis of loss landscapes** in the EL4ML framework (nS-03)
- **Nobody has combined HMM introgression detection with PGM extraction**
  to detect "knowledge transfer" between neural network layers (nS-04)
- **Nobody has applied the Anderson QS framework to multi-agent AI
  coordination** (nS-05)

All five use primitives neuralSpring has already validated at 1800+ checks
across 25 papers. The extensions require composition, not new math.

---

*neuralSpring baseCamp: BioPhysical AI Interpretability.
5 sub-theses, 15 grounding papers, 28 planned experiments.
Cross-spring handoff for hotSpring (nS-03) and wetSpring (nS-01, nS-05).*
