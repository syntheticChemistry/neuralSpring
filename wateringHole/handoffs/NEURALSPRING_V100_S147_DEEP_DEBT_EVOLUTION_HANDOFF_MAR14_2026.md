# neuralSpring V100 S147 → barraCuda / toadStool: Deep Debt Execution + BarraCUDA Evolution

**Date:** March 14, 2026
**From:** neuralSpring S147 (1115 lib tests, 73 forge tests, 260 binaries, 0 clippy)
**To:** barraCuda, toadStool teams
**Scope:** Deep debt elimination, barracuda evolution, tolerance centralization, provenance completeness
**Supersedes:** V99 S146 Industry GPU Parity Handoff (Mar 12, 2026)
**License:** AGPL-3.0-or-later

---

## Executive Summary

- **Zero inline magic numbers** in production library code — all 13 sites across 8 files centralized to `tolerances::` named constants with doc justifications
- **Zero duplicate math** — `digester_anderson::shannon_diversity` rewired to delegate to `barracuda::stats::shannon_from_frequencies` via `primitives::shannon_entropy`
- **6 provenance records added** for composition experiments (Exp 096–100 + Paper 027) — all experiments now have documented Python baseline provenance
- **Capability-based discovery** — hardcoded `"petaltongue"` strings in IPC discovery evolved to `config::PETALTONGUE_SOCKET_DIR` / `config::PETALTONGUE_SOCKET_PREFIX`
- **All quality gates pass**: `cargo fmt`, `cargo clippy --workspace -- -W clippy::pedantic` (0 warnings), 1115/1115 lib tests, 0 doc warnings

---

## Part 1: What We Evolved (For BarraCUDA Awareness)

### Tolerance Centralization

All inline magic numbers in production code now reference `crate::tolerances::` constants:

| Constant | Value | Usage Sites |
|----------|-------|-------------|
| `LOG_ZERO_GUARD` | `1e-30` | Dirichlet normalization, Shannon diversity filter, spacing ratio guard, Boltzmann floor |
| `EXACT_F64` | `1e-12` | IPR threshold, evenness guard, disagreement range floor, effective dimension, CV denominator |
| `SPECIAL_FUNCTION_F64` | `1e-6` | Eigensolve pipeline acceptance |
| `NUMERICAL_DISTINCTNESS` | `1e-15` | Level spacing ratio filter, CD schedule interpolation |

**toadStool action:** If absorbing neuralSpring tolerance patterns into a shared tolerance crate, these 4 constants cover 80% of numerical guard use cases across all springs.

### Duplicate Math Elimination

| Before | After |
|--------|-------|
| `digester_anderson::shannon_diversity` — hand-rolled `-Σ(p*ln(p))` with filter | Delegates to `primitives::shannon_entropy` → `barracuda::stats::shannon_from_frequencies` |

The only remaining domain-specific `shannon_diversity` variants (`eco_dynamics`, `swarm_robotics`) take non-`&[f64]` inputs and compute frequencies from genotypes/types — these are not duplicates.

---

## Part 2: BarraCUDA Usage Snapshot (219 files, 45+ submodules)

### Fully Exercised Submodules

| Module | Functions Used | Notes |
|--------|---------------|-------|
| `stats` | pearson_correlation, variance, dot, shannon, shannon_from_frequencies, hill, r_squared, rmse | Core statistics |
| `linalg` | eigh_f64, solve_f64, cholesky_f64, graph_laplacian, effective_rank, nmf, ridge_regression | Linear algebra |
| `ops::bio` | HmmBatchForwardF64, hmm_viterbi, PairwiseL2Gpu, MultiObjFitnessGpu, SwarmNnGpu, HillGateGpu, BatchFitnessGpu | Bio GPU ops |
| `dispatch` | matmul, transpose, softmax, gelu, variance, mean, frobenius_norm, l2_distance, hmm_forward | CPU/GPU dispatch |
| `esn_v2` | MultiHeadEsn, ESNConfig, quantize_affine_i8_f64 | Reservoir computing |
| `nautilus` | DriftMonitor, NautilusBrain, BetaObservation | Adaptive control |
| `nn` | SimpleMlp, DenseLayer, Activation | Neural network primitives |

### Still Pending Upstream (Carried Forward from V99)

| Module | Use Case | Priority |
|--------|----------|----------|
| `ops::logsumexp` / `logsumexp_wgsl` | Replace manual logsumexp in HMM | High |
| `ops::pairwise_distance` | SATé (017) hand-rolled distance | High |
| `staging::StatefulPipeline` | HMM chain, ODE loops | Medium |
| `pipeline::ReduceScalarPipeline` | Log-likelihood, convergence | Medium |

---

## Part 3: Composition Experiments — Complete Provenance

All 5 composition experiments (Exp 096–100) + Paper 027 now have full provenance records in `src/provenance/experiments.rs`:

| Experiment | Label | Control Script |
|------------|-------|---------------|
| Paper 027 | Digestion Prediction ESN | `control/digestion_prediction/digestion_prediction.py` |
| Exp 096 | Digester-Anderson Coupling | `control/digester_anderson/digester_anderson.py` |
| Exp 097 | Isomorphic Reservoir Ensemble | `control/isomorphic_reservoir/isomorphic_reservoir.py` |
| Exp 098 | WDM Ensemble QS | `control/wdm_ensemble_qs/wdm_ensemble_qs.py` |
| Exp 099 | Introgression NN | `control/introgression_nn/introgression_nn.py` |
| Exp 100 | Attention Anderson Spectral | `control/attention_anderson/attention_anderson.py` |

---

## Part 4: Code Health Metrics

| Metric | Value |
|--------|-------|
| Library tests | 1115 pass, 0 fail |
| Forge tests | 73 pass |
| Integration tests | 9 pass |
| Validation binaries | 260 |
| Clippy warnings | 0 (pedantic + nursery) |
| Doc warnings | 0 |
| Unsafe code | 0 (`#![forbid(unsafe_code)]`) |
| Inline magic numbers in production | 0 |
| Mocks in production | 0 |
| Hardcoded primal names | 0 (all config-driven) |
| Max file size (library) | 812 LOC (`glucose_prediction.rs`) |
| External deps | All pure Rust |
| License | AGPL-3.0-or-later |

---

## Part 5: What toadStool/barraCuda Should Consider Absorbing

### From This Session

1. **Tolerance registry pattern**: neuralSpring's `tolerances/mod.rs` + `tolerances/registry.rs` pattern (80+ named constants, each with doc justification, centralized validation test) is mature enough to extract into a shared crate. All springs would benefit.

2. **Provenance struct pattern**: `BaselineProvenance` with script, commit, date, command, environment fields is a cross-spring validation pattern. Consider extracting to `barracuda::validation::BaselineProvenance`.

3. **`ValidationHarness`**: The `check_abs` / `check_rel` / `check_abs_or_rel` + exit code pattern is duplicated across springs. Could be a `barracuda::validation::Harness`.

### Carried Forward

All P0/P1/P2 items from V99 (softmax dispatch overhead, RFFT structural gap, MHA fused kernel) remain valid targets.
