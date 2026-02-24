# neuralSpring → ToadStool/BarraCUDA Absorption Handoff (V18)

**Date**: February 24, 2026
**neuralSpring Session**: 50
**ToadStool HEAD**: `b41ee5f4`
**Audience**: ToadStool/BarraCUDA team
**Purpose**: Absorption roadmap for baseCamp primitives + lessons learned

---

## 1. Executive Summary

neuralSpring Session 50 added 5 library modules implementing "Biophysical AI
Interpretability" — novel analysis of AI systems using physics/biology
primitives. These introduce general-purpose numerical primitives that benefit
all Springs. This handoff documents what to absorb, what to evolve, and
what we learned that's relevant to BarraCUDA's evolution.

**Status**: 82/82 validation checks PASS. 412 unit tests. 0 clippy warnings.
No new shortcomings. No breaking API changes.

---

## 2. Primitives Ready for Absorption

### 2.1 High Priority — Universal Utility

| Primitive | Source | Signature | Why Absorb |
|-----------|--------|-----------|-----------|
| `graph_laplacian` | `agent_coordination.rs` | `fn graph_laplacian(adjacency: &[f64], n: usize) -> Vec<f64>` | Network analysis across ecology, genomics, physics — any adjacency matrix |
| `effective_rank` | `neural_pgm.rs` | `fn effective_rank(eigenvalues: &[f64]) -> f64` | Dimensionality measure for any eigenvalue spectrum |
| `numerical_hessian` | `loss_landscape.rs` | `fn numerical_hessian(f: impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64>` | General optimization and PES characterization |

### 2.2 Medium Priority — Spectral Diagnostics

| Primitive | Source | Signature | Why Absorb |
|-----------|--------|-----------|-----------|
| `level_spacing_ratio` | `weight_spectral.rs` | `fn level_spacing_ratio(eigenvalues: &[f64]) -> f64` | GOE/Poisson discriminator — used by 3 Springs now |
| `empirical_spectral_density` | `weight_spectral.rs` | `fn empirical_spectral_density(eigenvalues: &[f64], n_bins: usize) -> (Vec<f64>, Vec<f64>)` | Eigenvalue histogram — visual diagnostic |
| `marchenko_pastur_bounds` | `weight_spectral.rs` | `fn marchenko_pastur_bounds(gamma: f64) -> (f64, f64)` | Random matrix theory benchmark |

### 2.3 Low Priority — Domain-Specific

| Primitive | Source | Notes |
|-----------|--------|-------|
| `weight_to_hamiltonian` | `weight_spectral.rs` | Symmetrize any rectangular matrix — could generalize to `linalg::symmetrize` |
| `belief_propagation_chain` | `neural_pgm.rs` | Essentially HMM forward pass on a chain PGM — `hmm.rs` pattern |
| `boltzmann_sampling` (Metropolis) | `loss_landscape.rs` | Parallel MCMC — `WrightFisherGpu` pattern |
| `disordered_laplacian` | `agent_coordination.rs` | Laplacian with random heterogeneity — specialized |

---

## 3. GPU Shader Evolution

### 3.1 Shader Candidates

All shader patterns already exist in ToadStool. These are adaptation targets:

| Shader | Description | Template | Complexity |
|--------|-------------|----------|-----------|
| `symmetrize.wgsl` | `out[i,j] = (A[i,j] + A[j,i]) / 2` | `transpose.wgsl` | Trivial |
| `laplacian.wgsl` | Row-sum → diagonal, subtract adjacency | `spatial_payoff.wgsl` | Low |
| `hessian_column.wgsl` | Parallel `f(x+h_i) - 2f(x) + f(x-h_i)` per dimension | `batch_fitness_eval.wgsl` | Medium |
| `histogram.wgsl` | Atomic binning of eigenvalues | New pattern (workgroup atomics) | Medium |
| `metropolis.wgsl` | Parallel MCMC chains with acceptance/rejection | `wright_fisher_step.wgsl` | Medium |

### 3.2 No New Shortcomings

baseCamp modules do not encounter S-14/S-15/S-16. All matrices are
synthetic with controllable magnitude (≥ 0.5). The `eigh_f64` path
via Householder+QR works correctly for all baseCamp matrix sizes.

---

## 4. Lessons Learned — Relevant to BarraCUDA Evolution

### 4.1 Cross-Domain Primitive Reuse Validates the Architecture

The core insight: Anderson localization IPR, HMM forward/backward, and
graph Laplacian are used identically across wetSpring (biology), hotSpring
(physics), and neuralSpring (AI). BarraCUDA's architecture of providing
substrate-agnostic primitives is correct. These primitives should be:

1. **In `barracuda::ops`**, not in individual Springs
2. **GPU-capable** for scale (eigendecomposition already is)
3. **Documented with cross-domain examples** showing that `graph_laplacian`
   works equally well for ecological networks, molecular graphs, and
   neural network weight matrices

### 4.2 The `eigh_f64` Pathway Is Heavily Used

Session 50 added 5 new `eigh_f64` consumers, bringing the total to 9
modules across neuralSpring alone. The Householder+QR eigensolver at
`77f70b2e` is battle-tested. Consider:

- **Priority**: Tridiagonal Sturm-bisection GPU eigensolver for large matrices
  (neuralSpring needs n=512+ for real weight matrix analysis)
- **Batch eigensolve**: `BatchedEighGpu` would enable parallel spectral
  analysis of multiple layers simultaneously

### 4.3 Dispatch Patterns Scale

The `gpu_or_cpu` dispatch pattern from Session 49 seamlessly extends to
baseCamp. Adding 5 new module dispatchers would follow the existing
`Dispatcher` pattern with zero architectural changes needed.

### 4.4 The Testing Primitive Pattern Should Be Upstream

Three patterns now used across all Springs deserve absorption:

| Pattern | Current | Proposed |
|---------|---------|----------|
| `exit_no_gpu()` | Per-Spring copy | `barracuda::testing::require_gpu()` |
| `baseline_path()` | Per-Spring copy | `barracuda::testing::baseline_path()` |
| `gpu_or_cpu()` | Per-Spring dispatch | `barracuda::dispatch::gpu_or_cpu()` |

### 4.5 f64 Consistency

All baseCamp modules use f64 throughout. The Session 48 f32→f64 migration
for typed ops was prescient — baseCamp would have required it anyway.
All GPU promotion of baseCamp primitives should use f64 from the start.

---

## 5. Integration Checklist for ToadStool Team

- [ ] Review `graph_laplacian` and `effective_rank` for `ops::linalg` absorption
- [ ] Review `level_spacing_ratio` and `empirical_spectral_density` for `ops::stats` absorption
- [ ] Review `numerical_hessian` for `ops::numerical` absorption
- [ ] Evaluate `symmetrize.wgsl` and `laplacian.wgsl` shader candidates
- [ ] Consider `BatchedEighGpu` for batch eigendecomposition (9 consumers and growing)
- [ ] Evaluate `barracuda::testing` module for cross-Spring test utilities
- [ ] No breaking changes needed — all baseCamp uses existing stable API

---

## 6. Full neuralSpring → BarraCUDA Surface (Updated)

| Category | Count | Note |
|----------|-------|------|
| Typed GPU ops consumed | 12 | Unchanged from Session 49 |
| Tensor API methods consumed | 30+ | Unchanged |
| CPU primitives consumed | 18 + `eigh` (5 new consumers) | 9 total `eigh` consumers |
| Shaders consumed | 13 upstream + 8 local | Unchanged |
| Library modules | **36** (was 31) | +5 baseCamp |
| Validation binaries | **138** (was 133) | +5 baseCamp |
| Unit tests | **412** (was 374) | +38 baseCamp |
| Validation checks | **82 new** (82/82 PASS) | baseCamp only |

---

*neuralSpring → ToadStool absorption handoff V18 (Session 50). 5 baseCamp
modules with general-purpose primitives. Graph Laplacian, effective rank,
numerical Hessian, level spacing ratio, ESD histogram — all candidates for
`barracuda::ops`. GPU promotion uses existing shader patterns. No new
shortcomings. 9 modules now consume `eigh_f64`. Cross-domain reuse validates
the BarraCUDA architecture.*
