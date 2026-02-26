# neuralSpring baseCamp: Publication Candidate Outlines

**Date**: February 26, 2026 (Session 81)
**Author**: Kevin Mok (BS Microbiology, MSU 2018; MS Data Science, MSU 2025)
**Status**: DRAFT — outlines for the 4 strongest publication candidates

---

## Paper A: Weight Matrices as Disordered Hamiltonians — Anderson Localization Predicts Neural Network Generalization

**Target venue**: ICML / NeurIPS (main conference)
**Sub-thesis**: nS-01 (Weight Hamiltonians)
**Experiments**: Exp-050, Exp-051, nS-101 through nS-106
**Priority**: 1 (strongest novel claim, cleanest primitives)

### Abstract (draft)

We apply the Anderson localization framework from condensed matter physics to
neural network weight matrices, treating each layer's weights as a disordered
Hamiltonian. By computing inverse participation ratio (IPR) and level spacing
ratio across training checkpoints of MLP, LSTM, Transformer, and CNN
architectures, we show that the generalization→memorization transition
corresponds to an Anderson metal→insulator transition: generalizing networks
exhibit delocalized eigenstates (extended IPR, GOE-like level spacing) while
memorizing networks exhibit localized eigenstates (Poisson spacing). This
provides a physical interpretation of Martin & Mahoney's heavy-tailed weight
spectra and Gurbuzbalaban et al.'s Dyson Brownian motion dynamics, unifying
them within the Anderson framework. All analysis uses open-source GPU-accelerated
eigendecomposition validated against 25 published papers at 2250+ checks.

### Outline

1. **Introduction** (2 pages)
   - The generalization puzzle: why do overparameterized networks generalize?
   - Martin & Mahoney (2021): ESD reveals heavy-tailed structure
   - Gap: ESD analyzes eigenvalue *distribution*; we analyze eigenstate *structure*
   - Anderson localization: the physics of extended vs localized states
   - Thesis: weight matrix spectral structure predicts generalization

2. **Background** (1.5 pages)
   - Anderson localization: Hamiltonian, disorder, metal-insulator transition
   - IPR: measures eigenstate spread (1/N extended, 1 localized)
   - Level spacing ratio: r ~ 0.531 (GOE/extended) vs r ~ 0.386 (Poisson/localized)
   - Prior work: Ouyang (2025) on GNN over-smoothing as Anderson localization
   - Connection to Gurbuzbalaban et al. (2025): Dyson Brownian motion

3. **Method** (2 pages)
   - Weight matrix as Hamiltonian: H = (W + W^T)/2 for symmetrization
   - Eigendecomposition via validated `eigh_f64` (Householder + implicit QR)
   - IPR computation: Σ_i |ψ_k(i)|^4 for eigenstate k
   - Level spacing: r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
   - Spectral entropy via Shannon formula
   - Training checkpoint protocol: every 5 epochs, deterministic seed 42

4. **Experiments** (3 pages)
   - 4.1: Training trajectory spectral analysis (Exp-050)
     - 4 architectures × 20 checkpoints each
     - IPR trajectory tracks generalization, not training loss
   - 4.2: Cross-architecture spectral fingerprint (Exp-051)
     - Universal spectral signature at convergence
     - Fingerprint discriminates generalizing vs memorizing models
   - 4.3: Martin-Mahoney ESD reproduction (nS-101)
     - Confirm 5+1 phase progression
     - Show that phases correspond to Anderson transition stages
   - 4.4: Dyson Brownian motion validation (nS-104)
     - Track eigenvalue repulsion during SGD
     - Confirm Gurbuzbalaban et al. beta=1 dynamics

5. **Results** (2 pages)
   - IPR increases (delocalization) during effective learning
   - Level spacing shifts from Poisson → GOE during training
   - Cross-architecture: architectures at comparable generalization quality
     share quantitatively similar spectral fingerprints
   - The Anderson transition epoch predicts the epoch where test loss
     stops improving

6. **Discussion** (1.5 pages)
   - Physical interpretation: generalization = metallic state (information
     flows freely through weight matrix)
   - Connection to flat vs sharp minima: delocalized eigenstates ↔ flat minima
   - Practical value: spectral fingerprint as pre-deployment diagnostic
   - Connection to constrained evolution: architectural constraints produce
     universal spectral signatures

7. **Reproducibility** (0.5 pages)
   - Open-source code (AGPL-3.0), all Rust, deterministic seed 42
   - No proprietary models, no external datasets beyond MNIST/ERA5
   - 129+ named tolerance constants with documented justification

### Data requirements
- Own training checkpoints (~800MB)
- No external model downloads for Paper A core results

### Compute requirements
- ~8 hours training on Eastgate (RTX 4070)
- ~80 seconds eigendecomposition for all checkpoints

---

## Paper B: Spectral Circuit Discovery — Extracting Neural Network Circuits from Weight Matrix Eigenstructure

**Target venue**: NeurIPS (main conference) or ICLR
**Sub-thesis**: nS-04 (Neural PGM) + nS-01 (Weight Hamiltonians)
**Experiments**: nS-404, nS-401, Exp-051
**Priority**: 2 (highest practical value — interpretability)

### Abstract (draft)

We propose spectral circuit discovery: identifying computational circuits in
trained neural networks by eigendecomposing attention weight matrices, without
requiring activation patching, ablation studies, or forward passes. Each
significant eigenvector-eigenvalue pair of the attention matrix represents a
circuit — a specific computation performed by the attention head. We validate
this approach on our small Transformer by comparing spectrally-identified
circuits with circuits found by ACDC (Conmy et al. 2023), the established
ablation-based method. Spectral circuit discovery is orders of magnitude
faster than ACDC (no forward passes needed) and reveals circuit structure
from weights alone — enabling pre-deployment interpretability analysis of
any model with accessible weights.

### Outline

1. **Introduction** (2 pages)
   - The interpretability problem: neural networks as black boxes
   - Mechanistic interpretability: circuits as minimal computational subgraphs
   - ACDC (Conmy et al. 2023): ablation-based circuit discovery — effective but slow
   - Gap: circuit discovery requires many forward passes; we need weights-only methods
   - Thesis: eigendecomposition of attention matrices reveals circuits directly

2. **Background** (1.5 pages)
   - Li et al. (2023): DNNs as tree-structured PGMs
   - ACDC: iterative edge removal, task-specific ablation
   - PGM correspondence: forward pass ≈ belief propagation on tree PGM
   - Spectral decomposition: eigenvectors as basis functions of the linear operator

3. **Method** (2 pages)
   - Attention matrix A = softmax(QK^T/√d) as a linear operator
   - Eigendecomposition of A: eigenvectors are circuit directions
   - Circuit identification: significant eigenvalue = important computation
   - Spectral commutativity between heads: [A_i, A_j] ≈ 0 → independent circuits
   - Tree-PGM extraction from weight products: layer-by-layer conditional structure

4. **Experiments** (3 pages)
   - 4.1: Spectral circuit extraction from Phase 0+ Transformer (nS-404)
   - 4.2: ACDC comparison on same model and tasks (ACDC method, open-source)
   - 4.3: Agreement metric: fraction of ACDC circuits captured by spectral method
   - 4.4: Speed comparison: spectral (milliseconds) vs ACDC (minutes/hours)
   - 4.5: Introgression detection between layers (nS-403) — "knowledge transfer"
     circuits identified by PhyloNet-HMM

5. **Results** (2 pages)
   - Spectral circuits capture ≥80% of ACDC circuits (hypothesis)
   - Spectral method is 100-1000× faster than ACDC
   - Introgression-detected "skip circuits" correspond to residual connections
   - PGM extraction from weight matrices matches forward-pass behavior

6. **Discussion** (1.5 pages)
   - Practical value: pre-deployment interpretability without forward passes
   - Scaling: eigendecomposition of 768×768 takes ~1s; ACDC on GPT-2 takes hours
   - Connection to Anderson localization: localized eigenstates = specialized circuits;
     delocalized eigenstates = distributed computation
   - Limitations: spectral method captures linear circuits; nonlinear interactions
     require activation analysis

### Data requirements
- Own trained Transformer weights
- ACDC code (MIT license, ~100MB, Tier 2)

### Compute requirements
- Spectral analysis: seconds
- ACDC comparison: minutes to hours depending on task complexity

---

## Paper C: Anderson Localization Predicts Phase Transitions in Multi-Agent AI Coordination

**Target venue**: AAMAS (Autonomous Agents and Multi-Agent Systems) or ICML
**Sub-thesis**: nS-05 (Multi-Agent QS)
**Experiments**: Exp-053, nS-501 through nS-505
**Priority**: 3 (direct wetSpring bridge, strong novelty)

### Abstract (draft)

We apply the Anderson localization framework from condensed matter physics
to predict phase transitions in decentralized multi-agent AI coordination.
By treating the agent interaction network as a disordered lattice — where
agent heterogeneity is disorder and communication topology is the lattice
structure — we predict a critical heterogeneity threshold above which
coordination fails (localized signals) and below which coordination
emerges (extended signals). This threshold depends on the dimensionality
of the interaction topology: 3D-like topologies sustain coordination at
all disorder levels, while 1D and 2D topologies fail — replicating in
the AI domain the exact dimensional phase split that Anderson
localization produces in bacterial quorum sensing (QS). We validate these
predictions across agent populations of 64–512, confirming size
independence of the critical threshold.

### Outline

1. **Introduction** (2 pages)
   - Multi-agent coordination: when does emergent behavior appear?
   - SwarmSys (2025), Emergent Collective Memory (2025): empirical observations
   - Gap: no theoretical framework predicts *when* coordination emerges
   - Quorum sensing analogy: bacteria solved the same problem
   - Anderson localization: the physics that governs QS geometry dependence
   - Thesis: same physics governs multi-agent AI coordination

2. **Background** (2 pages)
   - Anderson localization: disorder, dimensionality, metal-insulator transition
   - QS in microbial communities: 3D biofilm sustains QS; 2D/1D fails
   - Experimental validation: 3,100+ checks across 37 wetSpring experiments
   - Graph Laplacian as Anderson Hamiltonian

3. **Method** (2 pages)
   - Agent interaction graph → weighted Laplacian L
   - Disorder from agent heterogeneity W (capability differences)
   - H = L + W·V (Anderson Hamiltonian on agent graph)
   - IPR and level spacing ratio as coordination diagnostics
   - Game-theoretic validation: replicator dynamics, Nash equilibrium

4. **Experiments** (3 pages)
   - 4.1: Anderson spectral analysis at 64/128/256/512 agents (Exp-053)
   - 4.2: Dimensional phase diagram: 1D chain, 2D grid, 3D cube (nS-503)
   - 4.3: QS-enhanced swarm coordination (nS-502)
   - 4.4: Wright-Fisher selection for interaction topology (nS-505)
   - 4.5: Replicator dynamics at each topology (nS-504)

5. **Results** (2 pages)
   - W_c is constant (±10%) across 64→512 agents (size independence)
   - 3D topologies sustain coordination at all tested disorder levels
   - 1D/2D topologies fail at W > 2 (replicating wetSpring QS result)
   - Wright-Fisher selects for higher-connectivity graphs
   - Replicator dynamics: cooperative equilibrium reachable only in 3D topology

6. **Discussion** (2 pages)
   - Convergent coordination strategies: biology and AI find same solutions
   - The three "NP solutions" from QS (geometry bootstrapping, signal relay,
     logic inversion) appear in agent systems
   - Practical value: predict coordination success from topology alone
   - Design implication: engineer 3D-like interaction topologies for robust
     multi-agent systems
   - Connection to constrained evolution thesis

### Data requirements
- Algorithmic (seed 42). No external data.

### Compute requirements
- ~4 hours on Eastgate for full disorder sweep (240 runs)

---

## Paper D: GPU-Accelerated Nudged Elastic Band for Neural Network Loss Landscape Analysis

**Target venue**: Digital Discovery (RSC) or Journal of Chemical Theory and Computation
**Sub-thesis**: nS-03 (Loss Landscapes as Energy Landscapes)
**Experiments**: Exp-052, nS-302, nS-301, nS-303
**Priority**: 4 (strongest hotSpring bridge, unique venue)

### Abstract (draft)

We present the first GPU-accelerated nudged elastic band (NEB) analysis of
neural network loss landscapes, applying the energy landscape framework
from chemical physics (Wales, Cambridge) to characterize transition states
between loss minima. Using GPU-accelerated RK45 ODE integration (validated
against molecular dynamics codes at 664 checks) and Hessian
eigendecomposition (validated at 129+ tolerances), we find minimum-loss
pathways between different trained configurations of the same architecture.
We characterize the activation barriers, correlate Hessian eigenvalue
flatness with generalization quality, and demonstrate that Boltzmann
ensemble averaging at optimal temperature improves predictions over
single-minimum inference. This extends the EL4ML (Energy Landscapes for
Machine Learning) program to real-scale networks using sovereign GPU
infrastructure.

### Outline

1. **Introduction** (2 pages)
   - Loss landscape non-convexity: exponentially many minima, saddle points
   - Wales' energy landscape framework: disconnectivity graphs, transition states
   - Ballard et al. (2024) EL4ML: limited to tiny networks (CPU-bound)
   - Gap: need GPU-accelerated tools for real-scale landscape analysis
   - Thesis: GPU NEB reveals loss landscape connectivity and predicts
     generalization from landscape topology

2. **Background** (1.5 pages)
   - NEB: elastic band of images connecting two minima, spring forces maintain spacing
   - Transition state theory: activation barrier determines escape rate
   - Boltzmann sampling: weight space at finite temperature
   - Pittorino et al. (2025): high-entropy network states generalize better

3. **Method** (2 pages)
   - Hessian via `numerical_hessian` (→ BarraCUDA upstream)
   - NEB band dynamics: RK45 integration via `rk45_adaptive.wgsl` on GPU
   - Loss evaluation: `gpu_dispatch::neural_forward` (WGSL pipeline)
   - Boltzmann sampling: Xoshiro256** + Metropolis acceptance
   - 15 training runs: 5 learning rates × 3 regularization settings

4. **Experiments** (3 pages)
   - 4.1: Hessian eigenanalysis at trained minima (Exp-052, nS-301)
     - Flat vs sharp minima: eigenvalue count near zero
   - 4.2: NEB transition paths (nS-302)
     - Minimum-loss pathways between best 3 minima
     - Activation barrier heights and widths
   - 4.3: Boltzmann ensemble (nS-303)
     - Sampling at T = {0.01, 0.1, 1.0, 10.0}
     - Ensemble predictions vs single-minimum
   - 4.4: Cross-architecture topology (nS-304)
     - Are Transformer landscapes smoother than MLP landscapes?

5. **Results** (2 pages)
   - Strong correlation (r > 0.7 hypothesis) between Hessian flatness and
     test loss across 15 runs
   - NEB identifies distinct transition pathways with measurable barriers
   - Boltzmann ensemble at optimal T improves test loss by ≥5%
   - Transformer landscapes have lower barriers (more connected) than MLP

6. **Discussion** (1.5 pages)
   - EL4ML at scale: GPU acceleration enables real-network analysis
   - Connection to hotSpring: same ODE integrators, same energy landscape tools
   - Practical value: landscape topology as architecture selection criterion
   - Connection to WDM simulation: same primitives serve both AI and plasma physics
   - Connection to constrained evolution: more constraint → smoother landscape

### Data requirements
- ERA5 Michigan data (already in pipeline)
- 15 training runs (~750MB checkpoints)

### Compute requirements
- ~7.5 hours training
- ~15 minutes Hessian analysis
- ~5 minutes NEB per pair

---

## Publication Timeline

| Paper | Data Ready | Analysis Complete | Draft | Target Submission |
|:-----:|:----------:|:-----------------:|:-----:|:-----------------:|
| A (Weight Hamiltonians) | After Exp-050 (Tier 1) | +1 session | +2 sessions | ICML 2027 or NeurIPS 2026 workshop |
| B (Spectral Circuits) | After ACDC comparison (Tier 2) | +2 sessions | +3 sessions | NeurIPS 2027 or ICLR 2027 |
| C (Anderson Multi-Agent) | After Exp-053 (Tier 1) | +1 session | +2 sessions | AAMAS 2027 |
| D (GPU NEB) | After Exp-052 (Tier 1) | +1 session | +2 sessions | Digital Discovery 2027 |

### Dependency Graph

```
Tier 1 experiments (Exp-050..053)
  ├── Paper A: Weight Hamiltonians (Exp-050 + Exp-051 data)
  ├── Paper C: Anderson Multi-Agent (Exp-053 data)
  └── Paper D: GPU NEB (Exp-052 data)

Tier 2 experiments (ACDC comparison)
  └── Paper B: Spectral Circuits (needs ACDC code, public model weights)
```

Papers A, C, and D can proceed with Tier 1 data only (no external dependencies).
Paper B requires Tier 2 (ACDC code download, HuggingFace model weights via NestGate).

---

## Shared Infrastructure

All four papers share:

- neuralSpring's 129+ named tolerance constants
- 39 functions rewired to upstream BarraCUDA
- Deterministic seed (42) with PyTorch and Rust RNG alignment
- AGPL-3.0 open-source code
- 604 unit tests + 9 integration tests at 93.5% coverage
- Three-tier hardware validation (CPU, GPU, metalForge)
- No proprietary models, no restricted datasets, no institutional dependencies
