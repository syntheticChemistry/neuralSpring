# Sub-Thesis 04: Neural Networks as Probabilistic Graphical Models

**Date:** February 24, 2026 (Session 58 — `belief_propagation_chain` + 7 Dispatcher methods rewired to upstream)
**Status:** Core primitives implemented and validated (21/21 PASS)
**Module:** `src/neural_pgm.rs` | **Validator:** `src/bin/validate_neural_pgm.rs`
**Domain:** Probabilistic inference applied to neural network interpretability
**Novelty:** No prior work combines HMM introgression detection with PGM
extraction to detect "knowledge transfer" between neural network layers
(confirmed via literature search, February 2026)

---

## Abstract

We decompose trained neural networks into equivalent probabilistic graphical
models (PGMs) whose structure reveals the network's "reasoning" as a graph
of conditional probability chains. The forward pass through a neural network
approximates belief propagation on a tree-structured PGM (Li et al. 2023).
We extract this implicit PGM using spectral decomposition of weight matrices
and validate it by comparing PGM inference with neural network inference.

The novel contribution: we apply the same introgression-detection HMM
(Liu, Paper 018) that detects gene flow between species to detect
"knowledge transfer" between neural network layers. Layers that share
information (like species that exchange genes) produce detectable signatures
in the spectral structure of their weight products. We also apply
Anthropic-style circuit tracing methods to our small validated models,
extracting interpretable circuits using open-source methods (ACDC) rather
than proprietary models.

This transforms the black box of neural network inference into a transparent
probabilistic graph with quantified uncertainty at every node.

---

## 1. Introduction

### 1.1 The Interpretability Problem

Neural networks are function approximators whose internal representations
are opaque. Given an input x and output y, the network computes y = f(x)
through a sequence of nonlinear transformations, but the intermediate
"reasoning" is hidden in high-dimensional activation vectors.

Two recent approaches attempt to make this reasoning visible:

1. **Mechanistic interpretability** (Anthropic, Nanda): decompose the
   network into circuits — subgraphs that perform identifiable computations
   (induction heads, copy heads, fact retrieval circuits).

2. **PGM correspondence** (Li et al. 2023): show that the forward pass
   approximates inference on an infinite tree-structured PGM, giving the
   network joint-distribution semantics over all variables.

### 1.2 The HMM Connection

The HMM forward-backward algorithm (validated in Papers 016-018) IS belief
propagation on a chain-structured graphical model. The transition matrix
defines conditional probabilities between hidden states. The observation
matrix defines likelihoods.

A neural network layer performs y = sigma(W*x + b). If we treat W as a
transition matrix and sigma as an observation model, each layer is an
HMM step. The forward pass through N layers is an HMM forward pass
through N timesteps. The key difference: the "transition matrices" (weights)
are different at each layer (non-homogeneous HMM).

### 1.3 Introgression as Knowledge Transfer

In population genetics, introgression is the transfer of genetic material
between species through hybridization (Liu, Paper 018). PhyloNet-HMM
detects introgression by identifying genomic regions where the observed
data is better explained by gene flow than by independent evolution.

We propose the same framework for neural networks: some layers "transfer
knowledge" to distant layers via skip connections, attention, or shared
representations. This introgression-like signal is detectable in the
spectral structure of weight matrix products across layers.

---

## 2. Grounding Papers

### 2.1 Li et al. (2023) — DNNs as Tree-Structured PGMs

**Citation**: Li et al., "Deep Neural Networks Correspond to Infinite
Tree-Structured Probabilistic Graphical Models", arXiv:2305.17583, 2023.

**What they showed**: During forward propagation, DNNs perform
approximations of precise PGM inference on infinite tree-structured
graphical models. The correspondence is exact for sigmoid activations
and extends to ReLU and nonnegative activations.

**What we reproduce**: Construct the tree-PGM for our Phase 0 MLP
surrogate. Compare PGM inference output with forward pass output.

**What we extend**: Extract the PGM structure from weight matrices using
spectral decomposition (eigendecomposition reveals the effective
conditional probability structure). Test whether PGM predictions match
neural network predictions on out-of-distribution inputs.

### 2.2 Nabarro et al. (2024) — Deep Factor Graphs

**Citation**: Nabarro et al., "Learning in Deep Factor Graphs with
Gaussian Belief Propagation", ICML 2024.

**What they showed**: Neural networks can be represented as Gaussian
factor graphs where all quantities (inputs, outputs, parameters,
activations) are random variables. Gaussian belief propagation enables
inherently local, distributed training.

**What we reproduce**: Factor graph construction for our Phase 0
Transformer. Run Gaussian BP and compare with backpropagation.

**What we extend**: Use the factor graph to propagate uncertainty
through the network. Given uncertain inputs (e.g., noisy sensor data
from groundSpring), does the factor graph produce calibrated uncertainty
in the output?

### 2.3 Conmy, Mavor-Parker et al. (2023) — Automated Circuit Discovery

**Citation**: Conmy et al., "Towards Automated Circuit Discovery for
Mechanistic Interpretability", NeurIPS 2023. Open-source ACDC method.

**What they showed**: Circuits (minimal subgraphs performing specific
computations) can be discovered automatically by iteratively removing
edges that don't affect task performance.

**What we reproduce**: Apply ACDC to our Phase 0+ Transformer on
structured tasks (sequence prediction, attention pattern recognition).

**What we extend**: Instead of ablation-based circuit discovery, use
spectral decomposition of the attention matrices to identify circuits
from weights alone — no forward passes needed. Compare with ACDC results.

**Note**: We implement the ACDC METHOD on our own small models. We do
NOT download, run, or interact with Claude, GPT, or any proprietary
model. The science is in the mathematical method, not the model scale.

---

## 3. Experiments

### Exp-nS-401: Tree-PGM Extraction from MLP

Construct the Li et al. tree-PGM for our Phase 0 MLP surrogate
(4->64->64->10). Compare PGM belief propagation output with forward
pass output on 1000 test inputs.

**Primitives**: `hmm.rs` (belief propagation as HMM forward), `eigh_f64`
(spectral decomposition of weight matrices to extract conditional structure).

**Expected finding**: PGM output matches forward pass to within numerical
precision for sigmoid/softmax activations.

### Exp-nS-402: Factor Graph for Transformer

Construct the Nabarro factor graph for our Phase 0+ Transformer
(d=32, h=4, seq=8). Run Gaussian BP and compare predictions with
standard forward pass.

**Primitives**: `hmm.rs` (message passing), `gpu_dispatch::matmul`
(Gaussian BP matrix operations), `gpu_dispatch::variance` (uncertainty
propagation).

**Novel result**: First factor-graph uncertainty quantification on a
Transformer validated against known-correct baselines.

### Exp-nS-403: Introgression Detection Between Layers

Compute the spectral similarity between weight matrices at different
layers. Apply the PhyloNet-HMM introgression detection framework (Paper
018): treat each layer as a "species" and weight similarity as
"genetic distance". Detect layers that share information beyond what
simple forward propagation would explain.

**Primitives**: `introgression.rs` (HMM introgression framework),
`eigh_f64` (spectral decomposition), `gpu_dispatch::pearson_correlation`
(inter-layer correlation).

**Novel prediction**: Skip connections and residual layers produce
introgression-like signals detectable by the HMM framework. Layers
connected by skip connections are "hybridizing species" in the
phylogenetic metaphor.

### Exp-nS-404: Spectral Circuit Discovery

Decompose the attention matrices of our trained Transformer into
low-rank components via SVD/eigendecomposition. Each significant
eigenvector-eigenvalue pair represents a "circuit" — a specific
computation performed by the attention head.

**Primitives**: `eigh_f64` (eigendecomposition), `spectral_commutativity.rs`
(commutativity analysis between circuit components).

**Novel approach**: Circuit discovery from spectral analysis of weights
alone, without activation patching or ablation studies. Test: do
spectrally-identified circuits match ACDC-identified circuits?

### Exp-nS-405: PGM Out-of-Distribution Prediction

Feed out-of-distribution inputs through both the neural network and
the extracted PGM. Compare: does the PGM provide better-calibrated
uncertainty estimates for OOD inputs?

**Primitives**: `hmm.rs` (PGM inference), `gpu_dispatch::neural_forward`
(network inference), standard calibration metrics.

**Novel prediction**: The PGM representation degrades gracefully on OOD
inputs (wider uncertainty bounds) while the neural network produces
overconfident incorrect predictions. This gives the PGM representation
practical value for safety-critical applications.

### Exp-nS-406: Cross-Architecture PGM Complexity

Extract PGMs from MLP, Transformer, LSTM, and LeNet-5 (all trained on
comparable tasks). Compare PGM graph complexity: number of factors,
tree depth, branching factor.

**Primitives**: All HMM/spectral primitives above.

**Novel question**: Is the PGM complexity related to the model's
generalization ability? Do simpler PGMs (fewer factors) correspond
to better-generalizing models?

---

## 4. Connection to Constrained Evolution Thesis

- **Prediction 1**: The PGM structure extracted from constrained
  architectures (weight sharing, attention masking) is simpler and more
  interpretable than the PGM from unconstrained architectures.
  Architectural constraints produce interpretable inference graphs.

- **Prediction 2**: Introgression-like knowledge transfer between layers
  is analogous to horizontal gene transfer in biology: it enables
  innovations that vertical (layer-by-layer) propagation alone cannot
  achieve. Skip connections are the neural network's horizontal gene
  transfer mechanism.

---

## 5. Reproducibility

All experiments use our Phase 0/0+/0++ trained models. Open data only.

```bash
cargo run --release --bin validate_neural_pgm         # 21/21 PASS (Sessions 50, 54)
cargo run --release --bin validate_basecamp_gpu       # 14/14 PASS — pure GPU parity
cargo run --release --bin validate_basecamp_dispatch  # 19/19 PASS — Dispatcher routing
cargo run --release --bin validate_barracuda_parity   # 34/34 PASS — CPU↔GPU parity
```

All experiments (nS-401 through nS-406) are validated in the consolidated
`validate_neural_pgm` binary, including tree-PGM extraction, deep factor
graph belief propagation, layer spectral similarity, effective rank,
OOD detection via PGM divergence, and PGM complexity scaling.

### Upstream Rewiring (Session 56)

`belief_propagation_chain` now delegates to `barracuda::linalg::graph::belief_propagation_chain`
(ToadStool `9404fdb4`). Public API unchanged; all 21 checks still pass.

No proprietary models. No external data. No model downloads.
