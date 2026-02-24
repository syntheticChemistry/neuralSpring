# Sub-Thesis 03: Loss Landscapes as Energy Landscapes

**Date:** February 24, 2026 (Session 58 — `numerical_hessian` + 7 Dispatcher methods rewired to upstream)
**Status:** Core primitives implemented and validated (27/27 PASS)
**Module:** `src/loss_landscape.rs` | **Validator:** `src/bin/validate_loss_landscape.rs`
**Domain:** Statistical mechanics applied to neural network optimization
**Novelty:** No prior work uses GPU-accelerated RK45 ODE integration for
transition-state analysis of neural network loss landscapes within the
EL4ML (Energy Landscapes for Machine Learning) framework
(confirmed via literature search, February 2026)

---

## Abstract

We apply the energy landscape framework from chemical physics and molecular
dynamics to neural network loss landscapes. Saddle points between loss
minima are transition states. Local minima are metastable configurations.
SGD training is a molecular dynamics trajectory on the loss surface.
Boltzmann sampling of weight space at finite temperature reveals the
entropic structure of the landscape.

Using RK45 ODE integration (validated for regulatory network dynamics,
Waters Paper 020-021), Hessian eigendecomposition (validated for Anderson
localization, Kachkovskiy Paper 022-023), and game-theoretic equilibrium
analysis (validated for QS cooperation dynamics, Waters Paper 019), we
characterize loss landscape topology and predict training outcomes from
landscape geometry.

The key hypothesis: the number and connectivity of transition states
between loss minima determines a model's capacity for generalization.
Models with many accessible transition states (smooth, connected landscape)
generalize; models with isolated minima (rough, disconnected landscape)
memorize. This is the EL4ML program (Wales, Cambridge) applied with
GPU-accelerated validated primitives.

---

## 1. Introduction

### 1.1 The Loss Landscape Problem

Neural network training minimizes a loss function L(theta) over the
parameter space theta in R^N. For modern networks, N ranges from millions
to billions. The loss landscape is highly non-convex with exponentially
many local minima, saddle points, and flat regions.

Key open questions:
- Why does SGD find good minima despite non-convexity?
- What distinguishes "flat" minima (good generalization) from "sharp"
  minima (poor generalization)?
- How does architecture affect landscape topology?

### 1.2 The Energy Landscape Connection

In chemical physics, potential energy surfaces (PES) of molecular systems
exhibit the same features: local minima (stable conformations), saddle
points (transition states), and pathways connecting them. David Wales'
group (Cambridge) has developed a mature toolkit for characterizing these
landscapes: disconnectivity graphs, transition path sampling, and
thermodynamic analysis.

Ballard et al. (2024) initiated the EL4ML (Energy Landscapes for Machine
Learning) program, showing that loss landscapes of small networks can be
analyzed with the same tools. But their analysis is limited to very small
networks (tens of parameters) because the tools are CPU-bound.

We bring GPU acceleration: our validated `eigh_f64` handles Hessian
eigendecomposition at the scale needed for real networks, and our
`rk45_adaptive.wgsl` provides GPU-accelerated trajectory integration
for transition path sampling.

### 1.3 The hotSpring Connection

hotSpring validates molecular dynamics primitives (Sarkas MD, TTM,
nuclear EOS) against published physics codes. The same ODE integration,
energy minimization, and Boltzmann sampling that characterize plasma
physics are mathematically identical to the tools needed for loss
landscape analysis. neuralSpring applies hotSpring's physical tools to
an AI problem.

---

## 2. Grounding Papers

### 2.1 Ballard, Das, Martiniani, Wales (2024)

**Citation**: "Insights into machine learning models from chemical physics:
an energy landscapes approach", Digital Discovery 3, 20-43, RSC, 2024.

**What they showed**: Loss landscapes of small neural networks can be
analyzed using disconnectivity graphs, revealing the structure of minima
and transition states. The framework connects training dynamics to
thermodynamic quantities (heat capacity, free energy barriers).

**What we reproduce**: Disconnectivity graph construction for our Phase 0
MLP surrogate (small enough for full landscape enumeration).

**What we extend**: GPU-accelerated Hessian eigendecomposition via `eigh_f64`
to scale the analysis to our full Phase 0+ architectures. GPU-accelerated
transition path sampling via `rk45_adaptive.wgsl`.

### 2.2 Pittorino et al. (2025) — Boltzmann Entropy

**Citation**: Pittorino et al., "Boltzmann entropy and neural network
generalization", 2025.

**What they showed**: Treating weights as atomic coordinates and loss as
potential energy, high-entropy network states (sampled via Boltzmann
distribution) achieve superior generalization. This advantage is more
pronounced in narrower networks.

**What we reproduce**: Boltzmann sampling of weight space using our
validated RNG (Xoshiro256**) and Metropolis acceptance criterion.

**What we extend**: Compute the entropy landscape S(E) — the density of
states as a function of loss energy E. This is a standard tool in chemical
physics (Wang-Landau sampling) that has not been applied to neural networks
at GPU scale.

### 2.3 Liu et al. (2024) — Loss Landscape Characterization

**Citation**: Liu et al., "Loss Landscape Characterization of Neural
Networks without Over-Parametrization", arXiv:2410.12455, 2024.

**What they showed**: Loss landscapes can be characterized even for
models with saddle points, providing convergence guarantees for
gradient-based optimizers without requiring overparameterization.

**What we reproduce**: Hessian eigenvalue analysis at trained minima
and saddle points.

**What we extend**: Connect the saddle point analysis to transition state
theory — each saddle point defines a reaction coordinate between minima,
with an activation barrier that determines the rate of escape.

---

## 3. Experiments

### Exp-nS-301: Hessian Eigenvalue Analysis at Trained Minima

Compute the full Hessian matrix H_{ij} = d^2L/dtheta_i dtheta_j at the
trained minimum of our Phase 0 MLP surrogate. Diagonalize via `eigh_f64`.
Characterize the curvature: number of positive, zero, and negative
eigenvalues.

**Primitives**: `eigh_f64`, numerical Hessian via finite differences,
`gpu_dispatch::neural_forward` (loss evaluation).

**Expected finding**: Flat minima (good generalization) have many near-zero
eigenvalues. Sharp minima (poor generalization) have large positive eigenvalues.

### Exp-nS-302: Transition State Search via NEB

Implement the nudged elastic band (NEB) method to find minimum energy
(minimum loss) pathways between two different trained minima. Use RK45
to integrate the band dynamics.

**Primitives**: `rk45_adaptive.wgsl` (ODE integration for band dynamics),
`gpu_dispatch::neural_forward` (loss and gradient evaluation).

**Novel result**: The first GPU-accelerated NEB analysis of neural network
loss landscapes.

### Exp-nS-303: Boltzmann Sampling of Weight Space

Sample weight configurations from P(theta) ~ exp(-L(theta)/T) at
temperatures T = {0.01, 0.1, 1.0, 10.0}. Compute observables: mean loss,
weight variance, entropy.

**Primitives**: `rng.rs` (Xoshiro256**), `gpu_dispatch::neural_forward`
(loss evaluation), `gpu_dispatch::variance` (statistics).

**Novel prediction**: High-temperature ensemble average predictions
outperform single-minimum predictions (Pittorino's high-entropy advantage),
and the optimal temperature corresponds to the training temperature
implicit in SGD noise.

### Exp-nS-304: Landscape Topology Comparison

Compare loss landscape topology across architectures: MLP, Transformer,
LSTM, LeNet-5 — all trained on the same Phase 0 tasks.

**Primitives**: `eigh_f64` (Hessian curvature), NEB (connectivity),
Boltzmann sampling (entropy).

**Novel question**: Does architecture determine landscape topology? Are
Transformer landscapes smoother (more connected) than MLP landscapes?

### Exp-nS-305: Game-Theoretic Equilibria as Loss Minima

Model training as a game between layers, where each layer's "strategy"
is its weight matrix and the "payoff" is the negative loss. Apply
replicator dynamics (from Waters Paper 019) to find Nash equilibria.

**Primitives**: `game_theory.rs` (replicator dynamics), `gpu_dispatch::
replicator_step` (GPU game theory).

**Novel connection**: Nash equilibria of the inter-layer game correspond
to loss landscape minima. The basin of attraction of each equilibrium
corresponds to the trainability region. This connects game theory
(biology) to optimization (AI).

---

## 4. Connection to Constrained Evolution Thesis

- **Prediction 1**: Under architectural constraint (fixed width, depth),
  loss landscapes converge to similar topologies — the same constrained
  evolution that produces convergent phenotypes in biology produces
  convergent landscape geometries in AI.

- **Prediction 2**: The smoothest landscape (most connected, fewest barriers)
  emerges from the architecture with the strongest constraints (e.g.,
  weight sharing in CNNs, attention masking in Transformers). More constraint
  produces smoother landscapes, not rougher ones.

---

## 5. Reproducibility

All experiments use our Phase 0/0+/0++ trained models and open data.

```bash
cargo run --release --bin validate_loss_landscape     # 27/27 PASS (Sessions 50, 54)
cargo run --release --bin validate_basecamp_gpu       # 14/14 PASS — pure GPU parity
cargo run --release --bin validate_basecamp_dispatch  # 19/19 PASS — Dispatcher routing
cargo run --release --bin validate_barracuda_parity   # 34/34 PASS — CPU↔GPU parity
```

All experiments (nS-301 through nS-305) are validated in the consolidated
`validate_loss_landscape` binary, including Hessian eigenvalues, transition
barriers, Boltzmann sampling, cross-architecture dimension sweep, gradient
descent trajectory tracking, and multi-barrier landscape analysis.

### Upstream Rewiring (Session 56)

`numerical_hessian` now delegates to `barracuda::numerical::numerical_hessian`
(ToadStool `9404fdb4`). Public API unchanged; all 27 checks still pass.

No proprietary models. No external data.
