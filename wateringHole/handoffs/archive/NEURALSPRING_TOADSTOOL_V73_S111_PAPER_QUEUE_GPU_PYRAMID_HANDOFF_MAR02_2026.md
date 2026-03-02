<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring → ToadStool/BarraCUDA Handoff V73 — Paper Queue Validation & GPU Pyramid Complete

**Date**: March 2, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Sessions 110–111 — full control validation, CPU benchmark buildout (14 domains), 10-tier GPU pyramid validation
**Supersedes**: V72 (Deep Debt Resolution Complete)

---

## Executive Summary

- **207/207 validate_all PASS**, 861 lib tests, 0 clippy, 0 fmt diffs
- **CPU benchmark expanded**: 11 → 14 domains (31/31 PASS), all 15 Phase 0++ papers covered
- **Geometric mean speedup**: 38.6× Rust vs Python/NumPy (honest: includes 2 BLAS-bound domains)
- **Full 10-tier pyramid validated end-to-end**: Python → Rust → BarraCUDA CPU → GPU Tensor → WGSL → Pipeline → Cross-dispatch → Pure GPU → ToadStool Streaming → metalForge Cross-System
- **3 new Python bench scripts**: Papers 013 (eco batch fitness), 023 (Anderson IPR), 025 (global FST)
- **S110 bug fixes**: 3 bugs in validate_barracuda_basecamp (BP chain length, matmul orientation, eigenvalue variance tolerance), 1 in validate_multi_head_esn (error message mismatch)
- **S110 buildouts**: +11 dispatch parity checks, +6 ToadStool compute parity checks, +5 biomeOS graph coordination checks

---

## Part 1: BarraCUDA CPU — Pure Rust Math Proof

### 1.1 CPU Benchmark Results (14 domains)

| Domain | Paper | Python µs | Rust µs | Speedup | BarraCUDA Primitive |
|--------|-------|-----------|---------|---------|---------------------|
| HMM Forward | 016-018 | 12,082 | 84 | **143.9×** | `hmm::Hmm::forward` |
| NK Fitness | 011 | 14,439 | 18 | **807.6×** | `counterdiabatic::NkLandscape::fitness` |
| Pairwise L2 | 012 | 105 | 0.4 | **269.7×** | `modes::l2_distance` |
| Eco Batch Fitness | 013 | 49 | 22 | **2.3×** | `eco_dynamics::MultiNicheLandscape::batch_fitness` |
| Pairwise Hamming | 017 | 405 | 34 | **11.9×** | `sate_alignment::pairwise_distance_matrix` |
| Pairwise Jaccard | 024 | 2,035 | 141 | **14.5×** | `pangenome_selection::jaccard_distance_matrix` |
| Replicator Dynamics | 019 | 34,596 | 150 | **231.0×** | `game_theory::replicator_dynamics` |
| RK4 GRN | 020 | 24,500 | 374 | **65.5×** | `regulatory_network::rk4_step` |
| Commutator ‖[A,B]‖_F | 022 | 23 | 81 | **0.3×** | `spectral_commutativity::commutator` |
| Anderson IPR | 023 | 339 | 604 | **0.6×** | `anderson_localization::jacobi_eigh` + `mean_ipr` |
| Hill Gate | 021 | 499 | 4 | **115.2×** | `signal_integration::two_input_hill` |
| Multi-Obj Fitness | 014 | 2,813 | 3 | **1028.4×** | `directed_evolution::multi_objective_fitness` |
| Swarm NN Forward | 015 | 10,518 | 39 | **269.8×** | `swarm_robotics::neural_forward` |
| Global FST | 025 | 79 | 5 | **17.0×** | `meta_population::global_fst` |

**Geometric mean**: 38.6× across 14 domains.

### 1.2 Why 2 Domains Lose to Python

**Commutator (0.3×)** and **Anderson IPR (0.6×)** are BLAS-bound: NumPy calls LAPACK's
hand-tuned assembly for 64×64 matrix operations. Our pure Rust Jacobi eigensolve has
worse constants than LAPACK at this size. **This is exactly where GPU promotion pays off**
— BarraCUDA GPU eigensolve (`eigh_f64` via `BatchedEighGpu`) massively outperforms both.

### 1.3 Absorption Implications for ToadStool

ToadStool should consider:
1. **BLAS-competitive eigensolve**: The Jacobi rotation in `anderson_localization::jacobi_eigh`
   is correct but slow for n>32. ToadStool's NAK tridiagonal eigensolver (planned) would
   close this gap on CPU. On GPU, `BatchedEighGpu` already wins.
2. **Eco batch fitness vectorization**: The 2.3× speedup is modest because NumPy's broadcast
   + exp + max is efficient. A fused WGSL shader (`batch_niche_fitness.wgsl`) would provide
   the GPU-tier acceleration. Pattern: Gaussian kernel distances → exp → reduce_max.

---

## Part 2: BarraCUDA GPU — Math Portability Proof

### 2.1 CPU↔GPU Dispatch Parity (41/41 PASS)

Every CPU operation has a proven-equivalent GPU path via `gpu_dispatch::Dispatcher`:

| Category | Checks | Operations |
|----------|--------|------------|
| Core stats | 6 | variance, pearson, entropy, matmul, softmax, L2 |
| HMM | 3 | forward_step, alpha parity, scale parity |
| Bio activation | 2 | hill_gate (grid parity) |
| Population genetics | 6 | FST variance decomposition, pairwise FST (FST+FIS+FIT) |
| Information theory | 3 | KL divergence, softmax row-wise, thermal diversity |
| Spectral | 8 | eigensolve, IPR, commutator norm |
| Distance | 4 | L2, Hamming, Jaccard, geographic |
| Selection | 3 | selection coefficient, spectrum chi-squared |
| Misc | 6 | neural_forward, allele_freq, nucleotide_diversity |

### 2.2 Pure GPU Workload (34/34 PASS)

`validate_gpu_pure_workload_all` (10/10): All 15 Phase 0++ paper domains run through
typed BarraCUDA GPU ops with scalar-only readback. No CPU math in the hot path.

`validate_gpu_pure_wdm_coral` (24/24): WDM surrogates (nW-01..05) + coralForge
attention, TriMul, AF3 pLDDT/PAE, diffusion forward, Pairformer FFN/TriMul.

### 2.3 ToadStool Streaming (344/344 PASS)

`validate_toadstool_dispatch` (22/22): Substrate heuristics + 6 compute parity checks
(HMM chain, variance, eigh, matmul, entropy, allele_freq via `Dispatcher`).

`validate_streaming_spectral_pipeline` (28/28): Batch eigensolve→IPR→stats (8 Hamiltonians),
Anderson disorder sweep (6 W values), Dispatcher CPU↔GPU parity (1.6e-14).

`validate_toadstool_spectral_absorption` (294/294): CPU correctness + GPU parity +
batch scaling + mixed substrate routing.

---

## Part 3: metalForge Cross-System (271/271 PASS)

| Validator | Checks | Substrates |
|-----------|--------|------------|
| `validate_metalforge_pcie` | 23/23 | PCIe bandwidth + latency tiers |
| `validate_cross_system_dispatch` | 46/46 | Hardware discovery, domain heuristics, CPU↔GPU parity, transfer cost model, NPU routing, crossover sweep |
| `validate_metalforge_wdm_coral` | 47/47 | WDM + coralForge dispatch heuristics, PCIe bypass |
| `validate_mixed_hardware_dispatch` | 47/47 | Tower/Node/Nest atomics, triangle inequality |
| `validate_mixed_hardware` | 21/21 | nS-06 PCIe, AD classifier NPU export |
| `validate_nucleus_compute_dispatch` | 44/44 | NUCLEUS tower→node→nest + biomeOS graph coordination |
| `validate_publication_mixed_hardware` | 43/43 | Publication experiments mixed substrate |

---

## Part 4: BarraCUDA Usage Surface (for absorption planning)

### 4.1 Current Usage

| Metric | Count |
|--------|-------|
| Files with `barracuda::` imports | 205 |
| Unique submodules exercised | 25+ |
| Upstream function rewires | 44 |
| CPU→GPU dispatch ops | 47 (~97% of production math) |
| WGSL shaders (metalForge) | 42 |
| coralForge df64 shaders | 15 |

### 4.2 Most-Used Submodules

| Submodule | Usage Pattern |
|-----------|--------------|
| `barracuda::stats` | pearson_correlation, variance, entropy, dot, l2_norm, mae, rmse, r_squared |
| `barracuda::ops::linalg` | eigh_householder_qr, BatchedEighGpu, solve_f64, cholesky, SVD |
| `barracuda::ops::bio` | BatchFitnessGpu, HmmBatchForwardF64, PairwiseL2/Hamming/Jaccard, BatchIprGpu, WrightFisherGpu |
| `barracuda::tensor` | Tensor::from_data, matmul, add, tanh, sigmoid, softmax_dim |
| `barracuda::device` | WgpuDevice, GpuDriverProfile, Fp64Strategy |
| `barracuda::special` | chi_squared_statistic, gamma, erf, bessel, legendre |
| `barracuda::numerical` | rk45_solve, numerical_hessian |
| `barracuda::esn_v2` | MultiHeadEsn, quantize_affine_i8_f64 |
| `barracuda::dispatch` | matmul_dispatch, hmm_forward_dispatch, gelu_dispatch |
| `barracuda::spectral` | Lanczos, Anderson, level_spacing |

### 4.3 Absorption Targets for ToadStool

| Priority | Target | Why |
|----------|--------|-----|
| 1 | NAK tridiagonal eigensolver | Close CPU gap for Anderson/spectral (0.6× vs LAPACK) |
| 2 | Fused eco batch fitness shader | Close CPU gap for eco dynamics (2.3× vs NumPy) |
| 3 | Unidirectional streaming for spectral pipeline | 28/28 streaming checks already validate the pattern |
| 4 | BarraCUDA MultiHeadEsn NPU export | ESN → int8 quantize → NPU weights for edge deployment |

---

## Part 5: S110 Bug Fixes (for regression awareness)

| Validator | Bug | Root Cause | Fix |
|-----------|-----|------------|-----|
| validate_barracuda_basecamp | H² eigenvalue variance check | `check_abs` with 1e-8 on magnitude ~225 | Switch to `check_rel` |
| validate_barracuda_basecamp | BP chain length assertion | `belief_propagation_chain` returns input + N outputs | Assert `== 4` not `== 3` |
| validate_barracuda_basecamp | BP GPU matmul orientation | GPU was `T × v` (right), CPU was `v^T × T` (left) | Row-encode input, swap operand order |
| validate_multi_head_esn | NPU export error check | `"not been trained"` vs `"no trained heads"` | Match actual error string |

---

## Part 6: What neuralSpring Learned (for ToadStool evolution)

1. **Relative vs absolute tolerance**: Large-magnitude GPU results require relative tolerance
   checks. ToadStool's `ValidationHarness` could add a `check_auto` that selects rel/abs
   based on magnitude.

2. **Matmul orientation matters**: Left multiplication (`v^T × T`) vs right multiplication
   (`T × v`) produces different results. GPU matmul should document convention explicitly.

3. **14-domain benchmark is the honest story**: Including BLAS-bound domains (where LAPACK
   wins) gives a more defensible geometric mean (38.6×) than cherry-picking (77.3×).
   ToadStool's NAK eigensolver is the path to winning those 2 domains too.

4. **ToadStool streaming pattern works**: The 28/28 streaming spectral pipeline and
   294/294 absorption checks prove unidirectional dispatch preserves scientific conclusions.
   ToadStool can confidently absorb all neuralSpring spectral workloads.

5. **biomeOS graph coordination validated**: The AF→π→FST→entropy pipeline with CPU↔GPU
   parity per stage proves NUCLEUS atomics can orchestrate multi-stage science pipelines
   across mixed substrates.

---

## Appendix: Full Validation Pyramid

| Tier | Checks | Description |
|------|--------|-------------|
| 1. Python baselines | 41/41 | 25 papers + 5 WDM + 5 coralForge + 3 pub exp + 3 nS-06 |
| 2. Rust library | 861/861 | 41 modules + gpu_ops/ + gpu_dispatch/ |
| 3. BarraCUDA CPU primitives | 326/326 | 16 validators |
| 4. BarraCUDA CPU paper ports | 196/196 | 22 validators |
| 5. CPU benchmark (Rust vs Python) | 31/31 | 14 domains, 38.6× geomean |
| 6. CPU↔Python parity | 39/39 | 9 primitives + 9 kernels + 6 Dispatcher |
| 7. CPU↔GPU dispatch parity | 58/58 | dispatch_parity 41 + barracuda_parity 17 |
| 8. GPU promotion (A+B+C) | 65/65 | 27+20+18 checks |
| 9. Pure GPU workload | 34/34 | all-domains 10 + WDM+coral 24 |
| 10. ToadStool streaming | 344/344 | dispatch 22 + streaming 28 + absorption 294 |
| 11. metalForge cross-system | 271/271 | 7 validators |
| **Total** | **2276+** | **207/207 validate_all** |
