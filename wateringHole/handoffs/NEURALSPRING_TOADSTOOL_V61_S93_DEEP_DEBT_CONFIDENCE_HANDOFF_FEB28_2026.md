# neuralSpring → ToadStool/BarraCUDA Handoff V61 — Deep Debt Evolution + Confidence Heads

**Date**: February 28, 2026
**From**: neuralSpring (ML/neuroevolution validation)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Sessions 92–93 — nF-03 AlphaFold3 Phases A+B+C, deep technical debt evolution
**Supersedes**: V60 (Dispatch Parity & Mixed-Hardware)

---

## Executive Summary

- **197 binaries**, **185/185 validate_all**, **685 lib tests**, **3200+ checks**
- **nF-03 AlphaFold3 Phase C complete**: pLDDT, PAE, pDE, ranking score — Py 62/62, Rs 55/55, 18 unit tests
- **Deep debt evolution**: `dispatch_ops.rs` split into 7 domain files, `gpu_ops/mod.rs` split into mod+tests, iterator modernization across 6 core modules
- **Self-identification**: `env!("CARGO_PKG_NAME")` — zero hardcoded primal names
- **Zero unsafe, zero production mocks, zero cross-primal logic**
- BarraCUDA CPU remains **83.6× faster** than Python/NumPy (geomean, 11 domains)
- **39 Python drift baselines**, all passing

---

## Part 1: Deep Debt Evolution (Session 93)

### 1a. Dispatcher Domain Split

`src/gpu_dispatch/dispatch_ops.rs` (842 lines, monolithic `impl Dispatcher`)
was split into 7 domain-specific files using Rust's multiple `impl` blocks:

| New File | Methods | Domain |
|----------|---------|--------|
| `dispatch_linalg.rs` | `mat_mul`, `frobenius_norm`, `transpose`, `commutator`, `distance_to_normal` | Linear algebra |
| `dispatch_activations.rs` | `softmax`, `softmax_row_wise`, `boltzmann`, `gelu`, `hill_activation_batch` | Activations/distributions |
| `dispatch_bio.rs` | `hill_gate`, `multi_obj_fitness`, `swarm_nn_forward` | Biology |
| `dispatch_stats.rs` | `l2_distance`, `shannon_entropy`, `mean`, `variance`, `pearson_correlation`, `chi_squared` | Statistics |
| `dispatch_hmm.rs` | `hmm_forward_step/chain`, `hmm_backward_step`, `hmm_viterbi_step/chain`, `detect_introgression` | HMM/phylogenetics |
| `dispatch_popgen.rs` | `allele_frequencies`, `nucleotide_diversity`, `pairwise_fst`, `global_fst`, 6 more | Population genetics |
| `dispatch_dynamics.rs` | `replicator_step`, `eigh`, `disorder_sweep`, `integrate_ode_batch`, `spectrum_chi_squared`, `selection_coefficient` | Dynamics/eigensolvers |

**ToadStool note**: The public `Dispatcher` API is unchanged. All 47 GPU-promoted
ops work identically. The split improves maintainability for Springs that compose
domain-specific dispatch subsets.

### 1b. GPU Ops Test Extraction

`src/gpu_ops/mod.rs` was 668 lines — ~630 lines were `#[cfg(test)]`. Extracted
to `src/gpu_ops/tests_ops.rs`, leaving `mod.rs` at 38 lines of declarations.

### 1c. Iterator Evolution

Manual `for i in 0..n` loops evolved to idiomatic Rust iterators:

| Module | Pattern | Before → After |
|--------|---------|----------------|
| `diffusion.rs` | Cosine schedule | `for` → `.map().collect()` + `.windows().unzip()` |
| `diffusion.rs` | COM removal | `for` → `.fold()` |
| `pairformer.rs` | Sinusoidal embedding | `for` → `.map().collect()` |
| `counterdiabatic.rs` | NK landscape | `for` → `.map().collect()` |
| `cpu_fallback.rs` | HMM steps | `for` → `.enumerate().map().sum()` |
| `meta_population.rs` | Allele freq, matrix corr | `for` → `.flat_map().unzip()` |
| `dispatch_bio.rs` | Hill gate CPU fallback | Nested `for` → `.flat_map().map().collect()` |

### 1d. Error Handling + Self-Identification

| Change | File | Detail |
|--------|------|--------|
| `.unwrap()` → `.unwrap_or()` | `validate_modern_cross_spring.rs` | NaN-safe comparison |
| `.unwrap()` → `.expect()` | `validate_gpu_ode_batch.rs` | Descriptive panic message |
| `"neuralSpring"` → `env!("CARGO_PKG_NAME")` | `provenance.rs` | Dynamic package name |

---

## Part 2: nF-03 AlphaFold3 — Phases A+B+C Complete

### Phase A+B: Diffusion + Pairformer (Session 92)

| Component | Python | Rust | Max Diff | Unit Tests |
|-----------|--------|------|----------|------------|
| Cosine/linear noise schedules | 7/7 | 6/6 | 1.24e-14 | 4 |
| Forward diffusion | 4/4 | 4/4 | 8.88e-16 | — |
| DDPM/DDIM reverse steps | 6/6 | 5/5 | 1.24e-14 | — |
| SE(3)-equivariant noise | 4/4 | 4/4 | 4.44e-16 | — |
| Sinusoidal embedding | 4/4 | 3/3 | 6.66e-16 | 2 |
| Pair conditioning + TriMul + FFN | 4/4 | 4/4 | 6.66e-16 | 5 |
| **Subtotal A+B** | **29+14=43** | **26+13=39** | — | **11** |

### Phase C: Confidence Heads (Session 93)

New module: `src/sovereign_folding/confidence.rs`

| Head | Algorithm | Python | Rust | Max Diff | Unit Tests |
|------|-----------|--------|------|----------|------------|
| pLDDT | Linear → sigmoid | 5/5 | 4/4 | 1.42e-14 | 2 |
| PAE | Pair → softmax → expected distance | 5/5 | 4/4 | 8.88e-16 | 2 |
| pDE | Pair → softmax → predicted distance error | 5/5 | 4/4 | 8.88e-16 | 2 |
| Ranking score | Weighted combination (mul_add) | 4/4 | 4/4 | 0.0 | 1 |
| **Subtotal C** | **19/19** | **16/16** | — | **7** |

**Totals**: Py 62/62, Rs 55/55, 18 unit tests across 3 phases.

### Absorption Opportunity: Confidence Heads

The confidence head pattern (`linear → activation → expected value`) is
reusable for any model quality estimation. `softmax_expected(logits, bins)`
computes `Σ softmax(logit_i) × bin_i` — a general-purpose weighted
expectation under softmax. Consider for `barracuda::ops::confidence` or
`barracuda::stats::softmax_expected`.

---

## Part 3: BarraCUDA Usage Inventory (Updated)

### Current Consumption

| Category | Count | Change from V60 |
|----------|-------|-----------------|
| Total binaries | 197 | +20 |
| validate_all | 185/185 PASS | +8 validators |
| Library tests | 685 | +17 |
| Total checks | 3200+ | +89 |
| Barracuda import sites | 130+ | +6 |
| Barracuda submodules | 20+ | — |
| Upstream rewires | 44 | +2 (matmul) |
| GPU-promoted Dispatcher ops | 47 | — |
| Python drift baselines | 39 | +1 (confidence) |
| Named tolerances | 131+ | — |

### New BarraCUDA Surface (nF-03)

nF-03 currently uses **pure Rust math** for all confidence/diffusion/pairformer
operations. When these mature, the following are GPU promotion candidates:

| Function | BarraCUDA Pattern | Priority |
|----------|-------------------|----------|
| `plddt_head` (n_res × d dot products) | `Tensor::matmul` or fused GEMV | P2 |
| `pae_head` (n² × bins softmax + expected) | `Tensor::softmax_dim` + reduce | P2 |
| `cosine_beta_schedule` (T steps) | Trivial — keep CPU | P3 |
| `forward_diffusion` (noise addition) | `Tensor::add` + `mul_scalar` | P2 |
| `ddpm_reverse_step` | Composition of existing ops | P2 |

### Dispatch Domain Split Impact

The 7-file dispatcher split means ToadStool can now reference domain-specific
dispatch logic without reading 842 lines. For absorption, the most relevant
files are:

- `dispatch_bio.rs` — CPU fallbacks already use `flat_map` iterators, ready
  for `barracuda::dispatch` domain expansion
- `dispatch_hmm.rs` — 7 HMM methods, all delegating to `barracuda::dispatch`
- `dispatch_popgen.rs` — 11 population genetics methods, heaviest file

---

## Part 4: Remaining Absorption Targets (Updated from V60)

### Still pending absorption

| Shader/API | Domain | Priority | Notes |
|-----------|--------|----------|-------|
| 15 sovereign folding df64 shaders | Protein structure | P1 | nF-01/02 GPU pipeline |
| `compile_shader_df64_streaming` | df64 pipeline | P1 | — |
| `nn::SimpleMLP` | WDM surrogates | P2 | JSON weight load + forward |
| Transcendental precision (degree-10+ Horner) | Core math | P2 | — |
| `head_split.wgsl`, `head_concat.wgsl` | MHA S-03b | P3 | Local MHA decomposition |
| `softmax_expected(logits, bins)` | Confidence/quality | P3 | New from nF-03 Phase C |

### Already absorbed (no action needed)

All S-01..S-17 shortcomings resolved. 44 upstream rewires complete.
21/21 WGSL shaders absorbed. 20+ submodules exercised.

---

## Part 5: Validation Matrix

| Metric | Count |
|--------|-------|
| Total binaries | 197 |
| validate_all | **185/185 PASS** |
| Library tests | 685 |
| Total checks | 3200+ |
| Python baselines | 39 scripts, 282 checks |
| CPU vs Python speedup | 83.6× geomean |
| GPU portability | 9/9 |
| Dispatch parity | 30/30 |
| Mixed-hardware dispatch | 47/47 |
| Multi-GPU (RTX 4070 + Titan V) | Bit-identical |
| nF-03 AlphaFold3 | Py 62/62, Rs 55/55, 18 unit tests |
| Clippy warnings | 5 (all pre-existing pedantic) |

---

## Part 6: Recommendations for ToadStool

1. **Confidence head abstraction**: `softmax_expected(logits, bins)` is a
   reusable primitive for any model quality estimation. Consider adding to
   `barracuda::stats` or `barracuda::ops::confidence`.

2. **Dispatcher domain split pattern**: neuralSpring proved that splitting a
   monolithic `impl` block across domain files works cleanly in Rust. Consider
   the same pattern for large `barracuda::dispatch` modules.

3. **Iterator evolution**: The `flat_map` + `chunks_exact` patterns in
   `dispatch_bio.rs` CPU fallbacks are candidates for `barracuda::cpu` reference
   implementations.

4. **AlphaFold3 GPU pipeline**: Phase D (multi-molecule tokenization) is next.
   When complete, the full AF3 pipeline will need 5+ new WGSL shaders for
   tokenization + attention + diffusion + confidence. Plan shader absorption.

5. **Existing V60 recommendations remain open**: Params struct docs, NPU
   bandwidth model, FST f64 precision, BLAS small-matrix fast-path.

---

## Part 7: Quality Gates (Session 93)

| Gate | Result |
|------|--------|
| `cargo check` | PASS |
| `cargo test --lib` | **685/685 PASS** |
| `cargo clippy --lib` | 5 warnings (all pre-existing pedantic) |
| `cargo run --release --bin validate_all` | **185/185 PASS** |
| `control/check_drift.sh` | **39/39 PASS** |

---

*AGPL-3.0-or-later*
