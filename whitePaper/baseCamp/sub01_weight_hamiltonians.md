# Sub-Thesis 01: Weight Matrices as Disordered Hamiltonians

**Date:** February 23, 2026
**Status:** Core primitives implemented and validated (15/15 PASS)
**Module:** `src/weight_spectral.rs` | **Validator:** `src/bin/validate_weight_spectral.rs`
**Domain:** Random matrix theory applied to deep learning interpretability
**Novelty:** No prior work applies Anderson localization IPR to neural network
weight matrices to predict generalization (confirmed via literature search,
February 2026)

---

## Abstract

We apply the Anderson localization framework from condensed matter physics
to the weight matrices of trained neural networks. By treating each layer's
weight matrix as a disordered Hamiltonian and computing its spectral
properties — eigenvalue distribution, inverse participation ratio (IPR),
and level spacing ratio (r) — we predict generalization behavior, identify
phase transitions during training, and distinguish memorization from learning.

The key hypothesis: weight matrices that exhibit extended eigenstates
(high IPR, GOE-like level spacing) correspond to generalizing networks,
while localized eigenstates (low IPR, Poisson spacing) indicate memorization.
This maps exactly to the Anderson metal-insulator transition: generalizing
networks are "metallic" (information flows freely through the weight matrix),
while memorizing networks are "insulating" (information is trapped in specific
weight configurations).

We use the same `eigh_f64` and `BatchIprGpu` tools that characterize quorum
sensing in wetSpring's Anderson QS work (Sub-thesis 01, gen3 baseCamp),
applying them to a fundamentally different domain.

---

## 1. Introduction

### 1.1 The Problem: Why Do Some Networks Generalize?

Neural network generalization — the ability to perform well on unseen data —
remains poorly understood despite decades of research. Classical statistical
learning theory (VC dimension, Rademacher complexity) dramatically
overestimates the generalization gap for overparameterized networks. Modern
networks have more parameters than training samples yet generalize well,
violating classical bounds.

Recent work suggests that the answer lies in the spectral properties of
weight matrices. Martin & Mahoney (2021) demonstrated that the empirical
spectral density (ESD) of weight matrices exhibits heavy-tailed behavior
that correlates with generalization. But their analysis stops at spectral
density — they do not apply the full Anderson localization toolkit (IPR,
level spacing ratio, localization length) that condensed matter physics
provides.

### 1.2 The Anderson Localization Connection

Anderson localization (Anderson 1958) describes how eigenstates of a
Hamiltonian H = H_0 + W*V transition from extended (propagating) to
localized (trapped) as disorder W increases. The diagnostics are:

- **IPR** (inverse participation ratio): measures eigenstate spread.
  IPR ~ 1/N for extended states, IPR ~ 1 for localized states.
- **Level spacing ratio**: r ~ 0.531 (GOE) for extended, r ~ 0.386
  (Poisson) for localized.
- **Localization length**: ξ characterizes the exponential decay of
  localized eigenstates.

We propose that a trained weight matrix W is a disordered Hamiltonian.
The disorder comes from the specific training data and optimization path.
The spectral diagnostics predict whether information flows freely (generalizes)
or is trapped (memorizes).

---

## 2. Grounding Papers

### 2.1 Martin & Mahoney (2021) — Implicit Self-Regularization

**Citation**: Martin & Mahoney, "Implicit Self-Regularization in Deep Neural
Networks: Evidence from Random Matrix Theory and Implications for Learning",
JMLR 22(165):1-97, 2021.

**What they showed**: The ESD of DNN weight matrices progresses through 5+1
phases during training, from Marchenko-Pastur (random) to heavy-tailed
(self-regularized). Heavy-tailed spectra correlate with better generalization.

**What we reproduce**: ESD computation on our Phase 0 MLP and Transformer
weight matrices at training checkpoints.

**What we extend**: Apply IPR and level spacing ratio (Anderson diagnostics)
to the same weight matrices. Martin & Mahoney analyze the *distribution* of
eigenvalues; we analyze the *structure* of eigenstates.

### 2.2 Gurbuzbalaban, Hu, Simsekli, Zhu (2025) — SGD to Spectra

**Citation**: Gurbuzbalaban et al., "From SGD to Spectra: A Theory of
Neural Network Weight Dynamics", arXiv:2507.12709, 2025.

**What they showed**: Squared singular values of weight matrices follow
Dyson Brownian motion with eigenvalue repulsion (beta=1) during SGD.
The stationary distribution is a gamma-type density with power-law tail.

**What we reproduce**: Track singular value dynamics during training on
our MLP and Transformer models.

**What we extend**: Dyson Brownian motion is the *dynamics* of the Anderson
Hamiltonian under perturbation. We connect the Dyson dynamics to the
localization transition: as training progresses, do eigenvalues transition
from Poisson (localized, early training) to GOE (extended, generalized)?

### 2.3 Ouyang (2025) — Anderson Localization for GNNs

**Citation**: Ouyang, "Rethinking Over-Smoothing in Graph Neural Networks:
A Perspective from Anderson Localization", arXiv:2507.05263, 2025.

**What they showed**: GNN over-smoothing is Anderson localization in disguise.
Low-frequency modes expand (delocalize) while high-frequency modes localize
as depth increases.

**What we reproduce**: Participation degree computation on GNN feature matrices.

**What we extend**: Apply to non-GNN architectures (MLP, Transformer, LSTM).
Test whether over-smoothing equivalents exist in dense networks and whether
the same Anderson diagnostic (IPR) detects them.

---

## 3. Experiments

### Exp-nS-101: Martin-Mahoney ESD Reproduction

Compute empirical spectral density of weight matrices from our Phase 0
MLP (4->64->64->10) and Phase 0+ Transformer (d=32, h=4, seq=8) at
training checkpoints (epochs 1, 5, 10, 25, 50, 100).

**Primitives**: `eigh_f64` for eigendecomposition, `gpu_dispatch::variance`
for spectral statistics.

**Expected finding**: Progression from Marchenko-Pastur to heavy-tailed,
matching Martin & Mahoney.

### Exp-nS-102: Anderson IPR of Weight Matrices

Compute inverse participation ratio of weight matrix eigenstates at
each training checkpoint. Compare IPR trajectory for well-generalizing
vs overfitting training runs (controlled by regularization and dataset size).

**Primitives**: `BatchIprGpu`, `eigh_f64`.

**Novel prediction**: IPR increases (delocalization) during effective
learning and decreases (localization) during memorization/overfitting.

### Exp-nS-103: Level Spacing Ratio During Training

Compute the level spacing ratio r = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
for consecutive eigenvalue gaps of weight matrices during training.

**Primitives**: `eigh_f64`, sorted eigenvalue gap analysis.

**Novel prediction**: r transitions from ~0.386 (Poisson, random initialization)
toward ~0.531 (GOE, trained) for generalizing networks. Memorizing networks
remain near Poisson.

### Exp-nS-104: Spectral Dynamics — Dyson Brownian Motion

Track singular value trajectories during SGD training. Test for eigenvalue
repulsion (beta=1 Dyson) vs independent motion (beta=0).

**Primitives**: `eigh_f64` (per-epoch), `gpu_dispatch::pearson_correlation`
for eigenvalue spacing correlations.

**Expected finding**: Confirms Gurbuzbalaban et al. repulsion dynamics.

### Exp-nS-105: Cross-Architecture Spectral Comparison

Compare IPR, level spacing, and ESD between MLP, Transformer, LSTM, and
LeNet-5 weight matrices — all from our validated Phase 0/0+ models trained
on the same data.

**Primitives**: `eigh_f64`, `BatchIprGpu`.

**Novel question**: Do architectures that generalize better exhibit more
delocalized weight spectra? Is there a universal spectral signature of
generalization?

### Exp-nS-106: GNN Over-Smoothing via Anderson Framework

Construct a small GNN (message-passing on a graph) and compute participation
degree of feature matrices at increasing depth. Reproduce Ouyang's finding
that over-smoothing is Anderson localization.

**Primitives**: `eigh_f64`, `BatchIprGpu`, `StencilCooperationGpu` (graph
message passing as stencil).

**Extension**: Test whether adding skip connections (residual) shifts the
Anderson transition point — matching our Paper 022 finding that residual
layers improve spectral commutativity.

---

## 4. Connection to Constrained Evolution Thesis

The constrained evolution thesis predicts that under strong constraint,
solutions converge on the same structural elements. This sub-thesis tests
that prediction in the weight matrix domain:

- **Prediction 1**: All well-generalizing architectures exhibit the same
  spectral signature (GOE-like level spacing, delocalized IPR) regardless
  of architecture — MLP, Transformer, LSTM, CNN.

- **Prediction 2**: The Anderson transition point (where IPR crosses from
  localized to delocalized) corresponds to a critical training epoch that
  is predictable from network architecture and dataset properties.

- **Prediction 3**: Heavy-tailed weight spectra (Martin & Mahoney) are a
  *consequence* of the Anderson delocalization transition, not an independent
  phenomenon.

---

## 5. Reproducibility

All experiments use weight matrices from neuralSpring's Phase 0/0+/0++
training runs. Deterministic seed (42). Open data only.

```bash
cargo run --release --bin validate_weight_spectral   # 15/15 PASS (Session 50)
```

### Validated Primitives (Session 50)

| Function | What It Tests | Check Count |
|----------|--------------|-------------|
| `weight_to_hamiltonian` | Symmetry (`H == H^T`) | 1 |
| `empirical_spectral_density` | Bin normalization, eigenvalue coverage | 2 |
| `level_spacing_ratio` | GOE range (0.386–0.6), Poisson comparison | 3 |
| `marchenko_pastur_bounds` | Analytical MP bounds | 1 |
| `marchenko_pastur_departure` | Fraction outside MP bulk | 1 |
| `spectral_entropy` | Entropy positivity and finiteness | 1 |
| `weight_spectral_analysis` | Full pipeline (low-rank vs random comparison) | 3 |
| `activation_ipr` | IPR range and uniform baseline | 2 |
| Determinism | Identical results with same seed | 1 |

No proprietary models. No external datasets beyond our existing open baselines.
All experiments use deterministic seed (42) and in-code synthetic weight matrices.
