# neuralSpring V169 — B3/B4 ML Surrogates for lithoSpore Modules 3+4

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**From:** neuralSpring (S213)
**To:** primalSpring, lithoSpore, groundSpring, barraCuda teams
**Date:** 2026-05-19
**Version:** V169

---

## Summary

Session **S213** implements B3 and B4 ML surrogates as **additive enrichment** for lithoSpore modules 3 (allele trajectories) and 4 (citrate structural). Both follow the established B1 pattern: Python control baseline → `expected_values.json` → Rust module → validator binary → NUCLEUS pipeline integration.

**39 capabilities** registered (37 → 39). **754 workspace tests** (750 pass, 4 pre-existing environment-dependent). Deploy graph `graphs/neuralspring_deploy.toml` aligned to **V169/S213**.

---

## B4: ESN Citrate Early-Warning Classifier (Blount et al. 2008)

**lithoSpore Module 4** — detects potentiating mutations before Cit+ innovation.

- **Architecture**: ESN(input=4, reservoir=256, output=2) — binary early-warning classifier with ridge regression readout
- **Input features**: mean fitness, fitness variance, allele frequency entropy, frequency change rate (per-generation window)
- **Validation**: 16/16 checks PASS. Test accuracy: 94.3%. Rust-Python score parity: 3.35e-14

### Files

| File | Purpose |
|------|---------|
| `control/ltee_citrate_esn/ltee_citrate_esn.py` | Python baseline (seed=42, 200 trajectories) |
| `control/ltee_citrate_esn/expected_values.json` | Frozen baseline (weights, metrics, first trajectory) |
| `src/ltee_citrate_esn.rs` | `CitrateEsnPredictor`, `early_warning_metrics()` |
| `src/bin/validate_ltee_b4_citrate_esn.rs` | Validator binary (16 checks) |

---

## B3: LSTM+HMM+ESN Allele Trajectory Classifier (Good et al. 2017)

**lithoSpore Module 3** — classifies allele frequency trajectories by fate (fixation / loss / polymorphic). Target: T06 (≥95% accuracy).

- **Architecture**: Three-model fusion
  1. **LSTM encoder** (hidden=32): temporal features from frequency series → pool [mean, std, last] → 96 features
  2. **HMM regime decoder** (3 states: sweep / interference / coexistence): posterior → 3 values
  3. **ESN classifier** (input=99, reservoir=128, output=3): multi-class allele fate
- **Validation**: 16/16 checks PASS. Test accuracy: 100% (exceeds T06 target). LSTM feature parity: 5.2e-18, HMM posterior parity: 6.9e-18

### Files

| File | Purpose |
|------|---------|
| `control/ltee_allele_trajectory/ltee_allele_trajectory.py` | Python baseline (seed=42, 300 alleles, 3-class balanced) |
| `control/ltee_allele_trajectory/expected_values.json` | Frozen baseline |
| `src/ltee_allele_trajectory.rs` | `AlleleFateClassifier` pipeline (LSTM+HMM+ESN) |
| `src/bin/validate_ltee_b3_allele_trajectory.rs` | Validator binary (16 checks) |

---

## Pipeline Integration

- **Graph**: 2 new `StageNode` entries in `metalForge/forge/src/graph.rs` — `ltee_allele_classifier` (GpuPreferred), `ltee_citrate_esn` (CpuOnly). Both depend on `introgression_nn`. Total: 8 stages, 8 edges.
- **Dispatch**: Match arms in `dispatch_capability()` and `dispatch_capability_gpu()` for both capabilities.
- **Capability Registry**: `science.ltee_allele_classifier` (evolving) and `science.ltee_citrate_esn` (evolving) added to `config/capability_registry.toml`, `src/config.rs`, `src/niche.rs`.
- **PRIMAL_GAPS**: Gap 28 RESOLVED. All 28 gaps now resolved.

---

## Primitives Used

All ML primitives are local to neuralSpring — no new barraCuda IPC surface was needed.

| Primitive | Source | GPU dispatch |
|-----------|--------|--------------|
| LSTM cell/forward | `src/sequence.rs` | `dispatcher.mat_mul()` for gate projections |
| HMM (forward/Viterbi) | `src/hmm.rs` | `dispatcher.hmm_chain()` |
| ESN reservoir | `src/digestion_prediction.rs` pattern | `dispatcher.mat_mul()` for recurrence |
| Ridge readout | barraCuda `linalg::ridge_regression` | CPU (small readout) |

---

## For lithoSpore

These surrogates are **additive enrichment** — lithoSpore modules 3 and 4 already pass on groundSpring statistics. neuralSpring's ML classifiers add:

- **Module 3**: T06 allele fate classification (≥95% accuracy on labeled trajectories)
- **Module 4**: T07 early-warning detection of potentiating mutations before Cit+

The `expected_values.json` files in each control directory define the frozen baselines that lithoSpore can reference for cross-spring parity checking.

---

## For Other Springs

The B1 pattern (Python baseline → JSON → Rust → validator) is now proven for 3 LTEE papers (B1, B3, B4). Springs implementing their own lithoSpore surrogates should follow this template:

1. Python control in `control/<name>/` with `expected_values.json`
2. Rust module in `src/<name>.rs` with JSON loading + science logic
3. Validator binary in `src/bin/validate_ltee_<name>.rs` using `ValidationHarness`
4. Pipeline integration in graph + dispatch + capability registry

---

## Metrics

| Metric | Value |
|--------|-------|
| Capabilities | 39 (37 → 39) |
| Workspace tests | 754 (750 pass, 4 environment-dependent) |
| Validation scenarios | 10 |
| B3 checks | 16/16 PASS |
| B4 checks | 16/16 PASS |
| PRIMAL_GAPS | 28/28 RESOLVED |
| Deploy graph | V169/S213 |
