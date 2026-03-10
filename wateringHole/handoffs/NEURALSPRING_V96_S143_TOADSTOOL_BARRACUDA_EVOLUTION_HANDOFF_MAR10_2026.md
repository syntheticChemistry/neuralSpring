<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring V96 → toadStool/barraCuda Evolution Handoff

| Field | Value |
|-------|-------|
| **Date** | 2026-03-10 |
| **From** | neuralSpring S143 (1085 lib + 71 forge + 9 integration tests, 245 binaries, 0 clippy) |
| **To** | barraCuda team, toadStool team, coralReef team |
| **Supersedes** | V95 (S142 enable f64 fix + upstream rewire) |
| **Synced against** | barraCuda `83aa08a`, toadStool S142 (`a86bc546`), coralReef Iteration 29 (`2779c88`) |
| **License** | AGPL-3.0-or-later |

---

## Executive Summary

neuralSpring S143 completes the **Axis 2 Novel Compositions** phase — 5 new
experiments (Exp 096–100) that compose existing validated modules into novel
scientific applications without any new math. These experiments provide new
validation coverage for `barracuda::stats`, `barracuda::tensor::Tensor`,
and `barracuda::ops::linalg::eigh_householder_qr` in cross-domain contexts
that stress the composition and chaining of BarraCUDA primitives.

This handoff documents:

1. New BarraCUDA usage patterns discovered during composition experiments
2. GPU↔CPU parity observations across 5 novel domains
3. Upstream evolution opportunities for ToadStool compute dispatch
4. Lessons for metalForge mixed-hardware targeting

---

## Part 1: New Experiments and BarraCUDA Coverage

### Experiments Built (S143)

| Exp | Name | Modules Composed | BarraCUDA Features Exercised |
|-----|------|-----------------|------------------------------|
| 096 | Digester×Anderson coupling | ESN (Paper 027) + Anderson (Paper 023) | `stats::correlation`, `tensor::Tensor::matmul_ref/tanh/add`, `ops::linalg::eigh` |
| 097 | Isomorphic reservoir ensemble | ESN + LSTM glucose + LSTM weather | `tensor::Tensor::matmul_ref/transpose`, `stats::r_squared`, `ops::linalg::eigh` |
| 098 | WDM ensemble QS | game theory + WDM surrogates + Anderson | `stats::correlation::pearson_correlation/variance`, `tensor::Tensor::matmul_ref` |
| 099 | HMM introgression on NN layers | HMM (Paper 018) + weight statistics | `stats::r_squared/rmse`, `stats::correlation::pearson_correlation` |
| 100 | Attention Anderson spectral | attention + eigh + Anderson IPR | `tensor::Tensor::matmul_ref`, `stats::correlation::pearson_correlation`, `ops::linalg::eigh` |

### Validation Check Counts

| Tier | Exp 096 | Exp 097 | Exp 098 | Exp 099 | Exp 100 | Total |
|------|---------|---------|---------|---------|---------|-------|
| Python | 17 | 17 | 11 | 11 | 10 | 66 |
| Rust CPU | 55 | 35 | 81 | 15 | 34 | 220 |
| BarraCUDA CPU+GPU | 22 | 13 | 7 | 6 | 6 | 54 |
| **Total** | **94** | **65** | **99** | **32** | **50** | **340** |

---

## Part 2: GPU↔CPU Parity Observations

### Tensor Operations

| Experiment | Operation | Max Diff | Notes |
|-----------|-----------|----------|-------|
| 096 | ESN matmul (128×128) | ≤2.46e-4 | f32 GPU vs f64 CPU; expected precision gap |
| 097 | Weight matrix trace (128×128) | ≤1.6e-8 | Excellent f32 parity |
| 098 | Disorder sum (1×10 matmul) | 1.85e-5 | Small vector dot product |
| 100 | Attention trace (32×32) | 3.1e-9 | Near-exact parity |

**Observation**: GPU↔CPU parity is consistently excellent for medium-size matrices
(32–128). The f32→f64 precision gap is the dominant error source, not algorithmic
divergence. ToadStool's df64 core streaming should eliminate this gap entirely.

### Statistics Primitives

`barracuda::stats::correlation::pearson_correlation`, `variance`, and `r_squared`
produce bit-identical results to manual computation on the same f64 data in all 5
experiments. No numerical issues observed at any scale tested (3–100 element vectors).

---

## Part 3: Upstream Evolution Opportunities

### For BarraCUDA

1. **`eigh_householder_qr` batch mode**: Exp 097 and 100 call `eigh` on multiple
   matrices sequentially. A batched `eigh` (process N matrices in parallel on GPU)
   would accelerate spectral analysis across domains. Target: `BatchEighGpu` shader.

2. **`Tensor::eigh`**: Currently `eigh` is CPU-only via `barracuda::ops::linalg`.
   Promoting to a `Tensor` method with GPU-resident eigendecomposition would enable
   pure-GPU spectral workflows (Exp 097, 100, and the isomorphic thesis pipeline).

3. **HMM Viterbi GPU batch**: Exp 099 runs Viterbi on 100-step sequences. The
   existing `HmmBatchForwardF64` shader handles forward; a matching batch Viterbi
   would enable GPU-resident introgression detection at scale.

### For ToadStool Compute Dispatch

1. **Composition pipelines**: These 5 experiments demonstrate multi-stage compute
   where stage N's output feeds stage N+1 (e.g., eigendecomposition → IPR → Pearson
   correlation). ToadStool's `pipeline_graph` (absorbed from neuralSpring S139)
   could express these as DAGs with automatic device placement.

2. **Cross-domain spectral dispatch**: Exp 097 runs identical eigendecompositions
   on matrices from 3 different domains. A `ParallelEighDispatch` node in the
   pipeline graph could process all 3 concurrently on separate GPU streams.

3. **Mixed-hardware IPR**: Exp 096–100 all compute IPR from eigenvectors. The IPR
   kernel (`BatchIprGpu`) is already in BarraCUDA. Dispatching eigh→IPR as a fused
   pipeline (eigh on CPU/NPU, IPR on GPU) would demonstrate the NUCLEUS mixed
   dispatch pattern.

### For metalForge

The 5 composition experiments validate that neuralSpring's `metalForge/forge`
substrate model works correctly for cross-domain dispatch. No new shaders are
needed — all GPU work uses existing `Tensor` primitives and `barracuda::stats`.

---

## Part 4: Key Learnings for Evolution

### Spectral Universality (Exp 097)

The isomorphic reservoir ensemble proved that ESN and LSTM architectures from
3 unrelated domains (bioprocess, biomedical, meteorology) produce weight matrices
with nearly identical spectral properties:

- Effective dimension ratio CV = 0.003 (virtually zero variation)
- IPR CV = 0.003 (identical localization)
- Spacing ratios all in [0.48, 0.50] (Wigner-like)

**Implication for BarraCUDA**: The spectral analysis pipeline
(eigh → eigenvalues → IPR → effective dimension) is a universal diagnostic.
If BarraCUDA absorbs this as a first-class primitive (`spectral_profile(matrix)`),
every spring can use it for model interpretability.

### Disagreement as Phase Boundary Detector (Exp 098)

WDM surrogate ensemble disagreement maps to Anderson disorder, which then
predicts localization. This is a general pattern: **any ensemble of surrogates
can detect phase boundaries via disagreement → disorder → localization**.

**Implication for ToadStool**: `EnsembleDisagreement` as a streaming metric
(compute variance across N model outputs in real-time) would enable online
phase boundary detection in production deployments.

### HMM for Anomaly Detection in Weight Layers (Exp 099)

The PhyloNet-HMM originally designed for genomic introgression detection works
for detecting anomalous layers in neural networks (TPR=0.97, FPR=0). This
validates the isomorphic thesis at the algorithm level.

**Implication**: HMM infrastructure can be reused for model monitoring
(detecting weight drift, anomalous training updates, etc.).

---

## Part 5: Current neuralSpring State

| Metric | Value |
|--------|-------|
| **Lib tests** | 1085 |
| **Binaries** | 245 |
| **Modules** | 46 |
| **Papers reproduced** | 27/27 (queue closed) |
| **Composition experiments** | 5 (Exp 096–100) |
| **Clippy warnings** | 0 |
| **Unsafe code** | 0 |
| **BarraCUDA imports** | 209 files |
| **metalForge shaders** | 42 |
| **GPU dispatch ops** | 47 (~97%) |

### What's Next

- **Axis 1 (real data)**: P13–P16 (digester 16S, real operational data, QS regulons, CGM)
- **Axis 2 (remaining compositions)**: LSTM anomaly on MD trajectories (needs hotSpring),
  Digester→gut transfer (needs healthSpring)
- **Axis 3 (system scaling)**: LAN deployment via NUCLEUS, multi-tower compute

---

*neuralSpring V96 handoff — S143. 5 novel compositions complete. 1085 tests,
245 binaries, 0 clippy, 0 debt. Ready for ToadStool pipeline graph absorption
of composition workflows. Ready for BarraCUDA batch eigh and spectral profile
primitive. Ready for Axis 1 real data experiments.*
