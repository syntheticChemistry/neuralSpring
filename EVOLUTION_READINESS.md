# neuralSpring — Evolution Readiness

**Date**: February 22, 2026 (post-Session 40 sync)
**ToadStool HEAD**: `d45fdfb3` (Session 39)
**Pattern**: Python baseline → Rust validation → BarraCUDA CPU → BarraCUDA GPU Tensor → metalForge WGSL → GPU Pipeline → Cross-dispatch → ToadStool absorption → lean on upstream

---

## Quick Status

31 Rust modules cover all 25 papers + 5 Phase 0/0+ studies. 115 validation binaries
span 7 tiers: Python (Py), Rust native (Rs), BarraCUDA CPU (bC), GPU Tensor (gT),
metalForge WGSL (mF), GPU Pipeline (gP), and Cross-dispatch (xD).

| Category | Count | Status |
|----------|-------|--------|
| Python baselines | 206/206 | **COMPLETE** |
| Rust native validation | 258 lib tests, 31 modules, 115 binaries | **COMPLETE** |
| BarraCUDA primitives | 272/272 | **COMPLETE** |
| BarraCUDA CPU (bC) | **24/25** papers (96%) | **ALL GREEN** |
| BarraCUDA GPU Tensor (gT) | **23/25** papers (92%) | **ALL GREEN** |
| metalForge WGSL (mF) | 14/25 papers (56%) | **ALL PASS** |
| GPU Pipeline (gP) | 7/25 papers (28%) | **ALL PASS** |
| Cross-dispatch (xD) | **15/15** Phase 0++ papers (100%) | **ALL GREEN** |
| GPU shader validation | 108/108 (17 WGSL shaders) | **COMPLETE** |
| GPU pipeline validation | 77/77 | **COMPLETE** |
| ToadStool shortcomings absorbed | 12/12 (S-01..S-12) | **ALL ABSORBED** |
| S-16 (transpose dispatch) | One-line fix | **FIXED** |
| S-15 (matmul hang) | Root-caused: magnitude ≤ 0.1 | **WORKAROUND** (≥ 0.5 data) |
| S-14 (naive matmul hang) | A×B^T pattern avoids | **WORKAROUND** |
| S-13 (PooledBuffer race) | Deferred return + device poll | **FIXED** upstream (Session 39) |
| TS-003 (trig precision) | 7-term Taylor + Cody-Waite | **FIXED** upstream (Session 36) |
| TS-001 (pow_f64 precision) | Extended exp/log polynomials | **FIXED** upstream (Session 36) |
| Shader absorption | 5 of 8 local shaders absorbed | **13/17 upstream** (Session 39) |
| Upstream wrapper validation | 6 bio ops + f64 HMM | **23/23 PASS** (new) |
| Upstream parity (dual-path) | 6 GPU validators | **6/6 PASS, 0.00e0 diff** (bit-identical) |
| ReduceScalarPipeline | f64 mean IPR via GPU reduce | **5.55e-17 diff** (machine ε) |
| Spectral theory stack | Lanczos, Anderson, Hofstadter, Lyapunov, eigh×Sturm | **17/17 PASS** (hotSpring lineage) |
| Capability-based dispatch | 12 validators + evolved HMM use `Gpu::dispatch_1d` | **Runtime-validated** (Session 40) |
| Upstream vs local benchmark | 6 kernels, RTX 4070 | **0.92–1.16×** overhead (negligible) |
| Evolved LOC | ~2,864 fossilized | Documented, bench migration complete |
| Grand total checks | **1604+** (206 Py + 1398+ Rust/GPU) | **ALL GREEN** |

---

## Tier A — Shader Absorption Status

### Absorbed Upstream (Session 39, `d45fdfb3` — generalized variants)

ToadStool absorbed 5 neuralSpring shaders, evolving them into generalized
upstream variants. Local copies retained for validation compatibility.

| Shader | Upstream | Binary | Checks | Key Differences |
|--------|----------|--------|--------|-----------------|
| `pairwise_l2.wgsl` | `barracuda::shaders::math::pairwise_l2` | `validate_gpu_modes` | 15/15 | Closed-form pair decoding vs linear search |
| `multi_obj_fitness.wgsl` | `barracuda::shaders::bio::multi_obj_fitness` | `validate_gpu_directed` | 6/6 | Bessel correction (n-1) vs population (n) |
| `hill_gate.wgsl` | `barracuda::shaders::bio::hill_gate` | `validate_gpu_signal` | 9/9 | Mode 0/1 generalization vs 2D-grid only |
| `swarm_nn_forward.wgsl` | `barracuda::shaders::bio::swarm_nn_forward` | `validate_gpu_swarm` | 9/9 | Generic MLP vs fixed 1→4→5, clamped sigmoid |
| `mean_reduce.wgsl` | `barracuda::shaders::reduce::mean_reduce` | `validate_gpu_pure_workload` | 7/7 | Effectively identical |

### Absorbed Upstream (Pre–Session 39, `77f70b2e` — identical copies)

| Shader | Upstream API | Binary | Checks |
|--------|-------------|--------|--------|
| `hmm_forward_log.wgsl` | `barracuda::ops::bio::hmm` | `validate_gpu_hmm_forward` | 13/13 |
| `batch_fitness_eval.wgsl` | `barracuda::ops::bio::batch_fitness` | `validate_gpu_batch_fitness` | 20/20 |
| `rk4_parallel.wgsl` | `barracuda::ops::rk_stage` | `validate_gpu_rk4` | 8/8 |
| `pairwise_jaccard.wgsl` | `barracuda::ops::bio::pairwise_jaccard` | `validate_gpu_pangenome` | 6/6 |
| `pairwise_hamming.wgsl` | `barracuda::ops::bio::pairwise_hamming` | `validate_gpu_sate` | 5/5 |
| `locus_variance.wgsl` | `barracuda::ops::bio::locus_variance` | `validate_gpu_meta_pop` | 7/7 |
| `spatial_payoff.wgsl` | `barracuda::ops::bio::spatial_payoff` | `validate_gpu_game_theory` | 5/5 |
| `batch_ipr.wgsl` | `barracuda::spectral::batch_ipr` | `validate_gpu_anderson` | 5/5 |

### Still Local (pending absorption)

| Shader | Domain | Binary | Checks | Absorption Target |
|--------|--------|--------|--------|-------------------|
| `head_split.wgsl` | MHA (attention) | `validate_mha_gpu` | 5/5 | `barracuda::ops::mha` (fix S-03b) |
| `head_concat.wgsl` | MHA (attention) | `validate_mha_gpu` | 5/5 | `barracuda::ops::mha` (fix S-03b) |
| `xoshiro128ss.wgsl` | Stochastic (PRNG) | `validate_gpu_prng` | 5/5 | `barracuda::ops::prng` |
| `swarm_nn_scores.wgsl` | Swarm (015) | `validate_gpu_pipeline_swarm` | PASS | No upstream equivalent |

### WGSL exports (forge crate — single source of truth)

All 17 WGSL shaders are centralized in `metalForge/forge/src/shaders.rs`.
13 have upstream equivalents (8 identical + 5 generalized variants); 4 are still local-only.
Library modules re-export for backward compatibility:

| Forge Constant | Library Re-Export |
|----------------|-------------------|
| `forge::shaders::HMM_FORWARD_LOG` | `hmm::WGSL_HMM_FORWARD_LOG` |
| `forge::shaders::PAIRWISE_JACCARD` | `pangenome_selection::WGSL_PAIRWISE_JACCARD` |
| `forge::shaders::LOCUS_VARIANCE` | `meta_population::WGSL_LOCUS_VARIANCE` |
| `forge::shaders::BATCH_FITNESS_EVAL` | `evolved::WGSL_BATCH_FITNESS_EVAL` |
| `forge::shaders::RK4_PARALLEL` | `evolved::WGSL_RK4_PARALLEL` |
| `forge::shaders::MEAN_REDUCE` | `evolved::WGSL_MEAN_REDUCE` |
| `forge::shaders::SPATIAL_PAYOFF` | `game_theory::WGSL_SPATIAL_PAYOFF` |
| `forge::shaders::BATCH_IPR` | `anderson_localization::WGSL_BATCH_IPR` |
| `forge::shaders::PAIRWISE_HAMMING` | `sate_alignment::WGSL_PAIRWISE_HAMMING` |
| `forge::shaders::PAIRWISE_L2` | `modes::WGSL_PAIRWISE_L2` |
| `forge::shaders::MULTI_OBJ_FITNESS` | `directed_evolution::WGSL_MULTI_OBJ_FITNESS` |
| `forge::shaders::SWARM_NN_FORWARD` | `swarm_robotics::WGSL_SWARM_NN_FORWARD` |
| `forge::shaders::HILL_GATE` | `signal_integration::WGSL_HILL_GATE` |
| `forge::shaders::HEAD_SPLIT` | `evolved::WGSL_HEAD_SPLIT` |
| `forge::shaders::HEAD_CONCAT` | `evolved::WGSL_HEAD_CONCAT` |
| `forge::shaders::XOSHIRO128SS` | `rng::WGSL_XOSHIRO128SS` |

Binding layouts and dispatch geometry documented in `forge::bindings`.

### Shader binding layouts (for ToadStool absorption)

**hmm_forward_log.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> trans: array<f32>` (N×N transition log-probs)
- `@group(0) @binding(1)` — `var<storage, read> emiss: array<f32>` (N emission log-probs for current obs)
- `@group(0) @binding(2)` — `var<storage, read> prev_alpha: array<f32>` (N forward log-probs at t-1)
- `@group(0) @binding(3)` — `var<storage, read_write> next_alpha: array<f32>` (N forward log-probs at t)
- `@group(0) @binding(4)` — `var<uniform> params: HmmParams` (`{n_states: u32}`)
- Dispatch: `(n_states.div_ceil(256), 1, 1)` — one thread per state

**batch_fitness_eval.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> pop: array<f32>` (pop_size × genome_len)
- `@group(0) @binding(1)` — `var<storage, read> weights: array<f32>` (genome_len)
- `@group(0) @binding(2)` — `var<storage, read_write> fitness: array<f32>` (pop_size)
- `@group(0) @binding(3)` — `var<uniform> params: FitnessParams` (`{pop_size, genome_len}`)
- Dispatch: `(pop_size.div_ceil(256), 1, 1)` — one thread per individual

**rk4_parallel.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read_write> state: array<f32>` (n_systems × 4)
- `@group(0) @binding(1)` — `var<uniform> params: Rk4Params` (`{n_systems, n_steps, dt, ...}`)
- Dispatch: `(n_systems.div_ceil(256), 1, 1)` — one thread per ODE system

**mean_reduce.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> values: array<f32>` (N values)
- `@group(0) @binding(1)` — `var<storage, read_write> result: array<f32>` (1 scalar)
- `@group(0) @binding(2)` — `var<uniform> params: ReduceParams` (`{n: u32}`)
- Dispatch: `(1, 1, 1)` — single workgroup (validation size)

**pairwise_jaccard.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> pa: array<f32>` (n_genes × n_genomes PA matrix)
- `@group(0) @binding(1)` — `var<storage, read_write> distances: array<f32>` (n_pairs)
- `@group(0) @binding(2)` — `var<uniform> params: JaccardParams` (`{n_genomes, n_genes}`)
- Dispatch: `(n_pairs.div_ceil(256), 1, 1)` — one thread per genome pair

**locus_variance.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> allele_freqs: array<f32>` (n_pops × n_loci)
- `@group(0) @binding(1)` — `var<storage, read_write> per_locus_var: array<f32>` (n_loci)
- `@group(0) @binding(2)` — `var<uniform> params: VarianceParams` (`{n_pops, n_loci}`)
- Dispatch: `(n_loci.div_ceil(256), 1, 1)` — one thread per locus

**spatial_payoff.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> grid: array<u32>` (grid_size² strategies)
- `@group(0) @binding(1)` — `var<storage, read_write> fitness: array<f32>` (grid_size²)
- `@group(0) @binding(2)` — `var<uniform> params: PayoffParams` (`{grid_size, b_x1000, c_x1000, _pad}`)
- Dispatch: `(grid_size².div_ceil(256), 1, 1)` — one thread per cell, Moore neighborhood

**batch_ipr.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> eigenvectors: array<f32>` (n_vectors × dim)
- `@group(0) @binding(1)` — `var<storage, read_write> ipr_out: array<f32>` (n_vectors)
- `@group(0) @binding(2)` — `var<uniform> params: IprParams` (`{dim, n_vectors}`)
- Dispatch: `(n_vectors.div_ceil(256), 1, 1)` — one thread per eigenvector

**pairwise_hamming.wgsl**:
- `@group(0) @binding(0)` — `var<storage, read> sequences: array<u32>` (n_seqs × seq_len)
- `@group(0) @binding(1)` — `var<storage, read_write> distances: array<f32>` (n_pairs)
- `@group(0) @binding(2)` — `var<uniform> params: HammingParams` (`{n_seqs, seq_len}`)
- Dispatch: `(n_pairs.div_ceil(256), 1, 1)` — one thread per sequence pair

---

## Tier B — Planned (needs design work)

| Shader | Domain | Priority | Blocker |
|--------|--------|----------|---------|
| `tridiag_eigensolver.wgsl` | Spectral (022–023) | P3 | Householder → bisection design |
| `pairwise_distance.wgsl` | Alignment (017) | P4 | O(N²) dispatch geometry |
| `stencil_cooperation.wgsl` | Game theory (019) | P5 | Neighborhood stencil pattern |
| `logsumexp_reduce.wgsl` | HMM/phylogenetics | P2 | Complements `hmm_forward_log.wgsl` |

---

## Tier C — New BarraCUDA Primitives Suggested

From our 25-paper analysis, these cross-cutting primitives would benefit
multiple Springs:

| Primitive | Use Case | Papers Served | Impact |
|-----------|----------|---------------|--------|
| `linalg::batch_matmul` | HMM forward/backward chain | 016–018 | Eliminate sequential dispatch |
| `ea::batch_fitness` | Population-parallel fitness | 011–015 | One dispatch per generation |
| `numerical::batch_rk45` | Multi-system ODE integration | 020–021 | Parallel biology simulation |
| `linalg::pairwise_distance` | O(N²) distance matrix | 017 | Alignment prerequisite |
| `ea::tournament_select` | GPU-parallel selection | 011–015 | Keep entire EA on GPU |
| `stencil::neighborhood_scan` | Spatial cooperation model | 019 | Reusable for any grid game |

---

## BarraCUDA APIs We Lean On

These are the native BarraCUDA APIs we depend on (via ToadStool absorption
or existing infrastructure):

| API | neuralSpring Use | Validated By |
|-----|------------------|-------------|
| `Tensor::from_data`, `to_vec` | All validation binaries | `validate_barracuda_tensor` (90/90) |
| `Tensor::layer_norm_wgsl` | ML inference validation | `validate_barracuda_tensor` (native, S-08 absorbed) |
| `Tensor::log_softmax_wgsl` | ML inference validation | `validate_barracuda_tensor` (native, S-09 absorbed) |
| `Tensor::leaky_relu_wgsl_with_slope` | Activation validation | `validate_barracuda_tensor` (S-05 absorbed) |
| `Tensor::elu_wgsl` | Activation validation | `validate_barracuda_tensor` (S-06 absorbed) |
| `ops::fft::{Fft1D, Ifft1D}` | f32 FFT validation | `validate_barracuda_fft` (12/12 f32) |
| `ops::fft::Fft1DF64` | f64 FFT (spectral, PPPM) | `validate_barracuda_fft` (8/8 f64, SHADER_F64) |
| `ops::fft::Rfft` | Real-to-complex FFT | `validate_barracuda_fft` (4/4 Rfft) |
| `ops::logsumexp::LogSumExp` | HMM log-domain | `validate_barracuda_logsumexp` (5/5) |
| `staging::StatefulPipeline` | Iterative GPU RK4 | `validate_gpu_stateful_pipeline` (10/10) |
| `dispatch::{dispatch_for, DispatchTarget}` | CPU/GPU parity | `validate_cross_dispatch` (8/8) |
| `WgpuDevice::new_cpu_relaxed` | CPU software adapter | `gpu.rs` (S-10 absorbed) |
| `stats::*`, `linalg::*`, `numerical::*`, `special::*` | 24 paper modules | 24 CPU port binaries (203/203) |

---

## BarraCUDA APIs — New in `d45fdfb3` (Sessions 25–39)

### Now leveraged by neuralSpring

| API | Use | Status |
|-----|-----|--------|
| `ops::linalg::eigh_householder_qr` | `src/eigh.rs` delegates to upstream | **Wired** (S-12 absorbed) |
| `ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32` | Shader source for HMM GPU forward | **Wired** (shader absorbed) |
| `ops::bio::{PairwiseHammingGpu, PairwiseJaccardGpu, LocusVarianceGpu, SpatialPayoffGpu, BatchFitnessGpu}` | Shader sources re-exported via forge | **Wired** (shaders absorbed) |
| `spectral::BatchIprGpu` | Shader source re-exported via forge | **Wired** (shader absorbed) |
| `ops::rk_stage::WGSL_RK4_PARALLEL` | Shader source re-exported via forge | **Wired** (shader absorbed) |
| S-13 PooledBuffer race fix | Deferred return + device poll — flows via path dep | **Automatic** (Session 39) |
| TS-003 trig precision | 7-term Taylor + Cody-Waite range reduction | **Automatic** (Session 36) |
| TS-001 pow_f64 precision | Extended exp/log polynomials | **Automatic** (Session 36) |
| TS-004 FusedMapReduceF64 fix | Single command encoder for both passes | **Automatic** (Session 36) |

### Available but not yet leveraged

| API | Potential Use | Status |
|-----|--------------|--------|
| `ops::bio::HmmBatchForwardF64` | Replace local `hmm_forward_gpu` dispatch entirely | Available (dispatch migration pending) |
| `spectral::{anderson_*, hofstadter_*, lanczos}` | Replace local model construction code | Available |
| `numerical::rk45_solve` | Adaptive ODE (Dormand-Prince) | Available |
| Native `Tensor::multi_head_attention` | Replace evolved MHA | **Blocked** (S-03b: projection shader hang) |
| `ops::bio::{FelsensteinGpu, GillespieGpu, SmithWatermanGpu}` | Future paper extensions | Available |
| `ops::bio::{RfBatchInferenceGpu, TreeInferenceGpu}` | Future ML forest workloads | Available |
| `ops::linalg::{InverseF64, LinSolveF64}` | GPU dense linear algebra | Available |
| `ReduceScalarPipeline::sum_f64` | Fitness aggregation | Available (local mean_reduce validated) |
| `BatchedRK4F64` | CPU-threaded ODE parameter sweeps | Available |
| `WGSL_BATCHED_EIGH_NAK_OPTIMIZED` | GPU-native eigensolve for Anderson | Available |
| `ops::nn::conv2d.wgsl` | Batched Conv2D — LeNet-5 conv layers | **New** (Session 39) — not yet wired to executor |
| `ops::nn::maxpool2d.wgsl` | MaxPool2D — LeNet-5 pooling | **New** (Session 39) — not yet wired to executor |
| `ops::nn::avgpool2d.wgsl` | AvgPool2D — alternative pooling | **New** (Session 39) — not yet wired to executor |
| `cpu_conv_pool::{conv2d, max_pool2d, avg_pool2d}` | CPU reference Conv2D/Pool | **New** (Session 39) |
| `esn_v2::export_weights/import_weights` | GPU-train → NPU-deploy pipeline | **New** (Session 39) |

---

## Phase 5b — Full-Stack Validation (23 Domains, ALL GREEN)

BarraCUDA `Tensor` operations validated against CPU f64 references across
23 papers spanning all 7 validation tiers. S-16 transpose dispatch **fixed**.
S-15 matmul hang **root-caused** (elements with magnitude ≤ 0.1 trigger
WGPU/Vulkan driver bug on RTX 4070). Workaround: generate data with
`rng.uniform() * 0.5 + 0.5` ensuring all elements ≥ 0.5.

| Validator | Domain | Papers | GPU Ops | Checks | Status |
|-----------|--------|--------|---------|--------|--------|
| `validate_barracuda_gpu_spectral` | Spectral commutativity | 022 | matmul | 10 | **PASS** |
| `validate_barracuda_gpu_eco` | Ecological dynamics | 013 | matmul, transpose | 6 | **PASS** |
| `validate_barracuda_gpu_hmm` | HMM phylogenetics | 016-018 | matmul, transpose | 5 | **PASS** |
| `validate_barracuda_gpu_fitness` | Evolutionary computation | 011-015 | matmul, transpose | 7 | **PASS** |
| `validate_barracuda_gpu_nn` | Neural nets | 015, 020-021 | matmul, transpose, tanh, add | 5 | **PASS** |
| `validate_barracuda_gpu_pairwise` | Pairwise distance | 017, 019, 024-025 | matmul, transpose | 5 | **PASS** (S-16 fixed) |
| `validate_barracuda_gpu_anderson` | Anderson localization | 023 | matmul, transpose | 7 | **PASS** (S-15 workaround) |
| `validate_barracuda_surrogate` | Surrogate MLP (Exp 001) | 001 | matmul, tanh | 7 | **PASS** |
| `validate_barracuda_transfer` | Transfer Learning (Exp 004) | 004 | matmul, tanh | 7 | **PASS** |
| `validate_barracuda_gpu_transformer` | Transformer (Exp 002) | 002 | matmul, transpose, tanh | 7 | **PASS** |
| `validate_barracuda_sequence` | Sequence (Exp 003) | 003 | matmul, tanh, sigmoid | 7 | **PASS** |
| `validate_barracuda_lenet` | LeNet-5 (Study 003) | S003 | matmul, tanh | 5 | **PASS** |
| `validate_barracuda_lstm` | LSTM (Study 004) | S004 | matmul, tanh, sigmoid | 6 | **PASS** |
| `validate_barracuda_bio_ops` | Upstream bio wrappers | 011-025 | BatchFitnessGpu, PairwiseHammingGpu, PairwiseJaccardGpu, LocusVarianceGpu, SpatialPayoffGpu, BatchIprGpu | 12 | **PASS** |
| `validate_barracuda_hmm_f64` | Upstream HMM f64 batch | 016-018 | HmmBatchForwardF64 (wetSpring) | 11 | **PASS** |

### Cross-dispatch Validators (xD — 15/15 Phase 0++ papers)

| Validator | Papers Covered | Checks | Status |
|-----------|---------------|--------|--------|
| `validate_cross_dispatch` | 011-015 | 8 | **PASS** |
| `validate_cross_dispatch_genomics` | 016-018 | 8 | **PASS** |
| `validate_cross_dispatch_extended` | 019-025 | 12 | **PASS** |
| `validate_cross_dispatch_phase4e` | PINN, DeepONet | 9 | **PASS** |
| `validate_cross_dispatch_hmm` | 016, 018 | 4 | **PASS** |
| `validate_cross_dispatch_ode` | 020 | 4 | **PASS** |

### Shortcoming Resolution

| # | Shortcoming | Severity | Root Cause | Resolution |
|---|-------------|----------|------------|------------|
| S-14 | Naive matmul hang (small square matrices) | Medium | Driver/binary complexity interaction | Workaround: A×B^T pattern (non-square shapes) |
| S-15 | Matmul hang when f32 elements ≤ 0.1 magnitude | Critical | WGPU/Vulkan driver bug (RTX 4070) | **Root-caused**: data ≥ 0.5 avoids hang |
| S-16 | 2D transpose dispatch uses divisor 256 vs tile 16 | High | `optimal_workgroup_size(ElementWise)` | **FIXED**: `const TILE: u32 = 16` |

Full details: `wateringHole/handoffs/`

---

## S-12: Absorbed Upstream (`77f70b2e`)

neuralSpring's Householder+QR eigensolver was absorbed by ToadStool as
`barracuda::ops::linalg::eigh_householder_qr`. `src/eigh.rs` now delegates
to upstream. The local fossil is preserved at `metalForge/fossils/evolved_s01_s11/eigh_local.rs`.

Validated by `validate_eigh_accuracy` (9/9 PASS). ToadStool also added
NAK-optimized GPU eigensolve shaders (`WGSL_BATCHED_EIGH_NAK_OPTIMIZED`).

---

## Code Quality (Post-Audit, February 21 2026)

| Aspect | Status |
|--------|--------|
| clippy pedantic + nursery | **0 warnings** (all `#[allow]` justified or removed) |
| `#[must_use]` | Applied to 24+ pure public functions across 5 modules |
| Centralized tolerances | All validation thresholds in `tolerances.rs` (no magic numbers) |
| GPU device init | Unified via `Gpu::new()` (removed ~800 LOC duplication) |
| Idiomatic Rust | HMM flat row-major layout, spectral flat layout, `NkLandscape.k` accessor |
| Consolidated math primitives | Shannon, Hill, sigmoid, RK4 centralized in `primitives.rs` — no duplicated math |
| GPU-ready flat layouts | HMM, spectral, anderson_localization, directed_evolution, sate_alignment use flat row-major `Vec<f64>` — direct GPU buffer upload |
| Graceful error handling | `require!` macro replaces `.expect()` in all validation binaries — no panic on GPU failure |
| Zero-copy genotype handling | `eco_dynamics.rs` uses `&[u8]` / `HashSet<&[u8]>` — avoids `Vec<u8>` clones |
| SPDX headers | All 40 Python/shell files have `AGPL-3.0-or-later` license identifier |
| Line coverage | **94.9%** line via `llvm-cov` |
| All files < 1000 LOC | `validate_barracuda_tensor.rs` reduced from 1053 → 864 lines |
| `unsafe` | Forbidden (`#![forbid(unsafe_code)]`) |

---

## Full Validation Stack (7 Tiers × 25 Papers)

The validation progression proves math portability at each level:

```
Tier 1 (Py)  → Open data + Python: reproducible science baseline
Tier 2 (Rs)  → Rust native: same math, type-safe, deterministic
Tier 3 (bC)  → BarraCUDA CPU: proves Rust math matches via barracuda primitives
Tier 4 (gT)  → BarraCUDA GPU Tensor: proves math is portable CPU → GPU
Tier 5 (mF)  → metalForge WGSL: domain-specific GPU kernels, validated vs CPU
Tier 6 (gP)  → GPU Pipeline: end-to-end multi-kernel GPU chains
Tier 7 (xD)  → Cross-dispatch: CPU ↔ GPU parity via dispatch routing
```

| Tier | Coverage | Status |
|------|----------|--------|
| Py (Python) | 25/25 (100%) | **ALL PASS** |
| Rs (Rust) | 25/25 (100%) | **ALL PASS** |
| bC (BarraCUDA CPU) | 24/25 (96%) | **ALL GREEN** |
| gT (GPU Tensor) | 23/25 (92%) | **ALL GREEN** |
| mF (metalForge WGSL) | 14/25 (56%) | **ALL PASS** |
| gP (GPU Pipeline) | 7/25 (28%) | **ALL PASS** |
| xD (Cross-dispatch) | 15/15 (100%) | **ALL GREEN** |

*Evolution readiness tracker — following the hotSpring pattern for ToadStool absorption.*
