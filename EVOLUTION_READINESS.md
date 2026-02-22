# neuralSpring — Evolution Readiness

**Date**: February 22, 2026 (post-absorption)
**ToadStool HEAD**: `77f70b2e` (Session 31h)
**Pattern**: Python baseline → Rust validation → WGSL shader → ToadStool absorption → lean on upstream

---

## Quick Status

Rust modules (29 total) include `pinn.rs` (Study 001 PINN Burgers) and `deeponet.rs` (Study 002 DeepONet antiderivative) alongside the 25-paper reproduction modules.

| Category | Count | Status |
|----------|-------|--------|
| Python baselines | 206/206 | **COMPLETE** |
| Rust native validation | 237 lib tests, 29 modules, 81 binaries, 94.9% coverage | **COMPLETE** |
| BarraCUDA primitives | 272/272 | **COMPLETE** |
| BarraCUDA CPU ports | 170/170 (17 modules) | **COMPLETE** |
| GPU shader validation | 85/85 (16 WGSL shaders) | **COMPLETE** |
| GPU pipeline validation | 77/77 | **COMPLETE** |
| ToadStool shortcomings | 12/12 absorbed (S-12 upstream) + S-03b locally workaround | **ALL RESOLVED** |
| Evolved LOC | ~2,864 fossilized | Documented, bench migration complete |
| Code quality audit | clippy pedantic+nursery clean, `#[must_use]`, centralized tolerances | **COMPLETE** |

---

## Tier A — Ready for ToadStool Absorption

These metalForge shaders are validated, documented, and ready for upstream
absorption. Each has a CPU reference implementation, a validation binary
with `ValidationHarness`, and documented binding layouts.

| Shader | Domain | Binary | Checks | Absorption Target |
|--------|--------|--------|--------|-------------------|
| `hmm_forward_log.wgsl` | Phylogenetics (016–018) | `validate_gpu_hmm_forward` | 13/13 | `barracuda::ops::hmm` |
| `batch_fitness_eval.wgsl` | Evolutionary computation (011–015) | `validate_gpu_batch_fitness` | 20/20 | `barracuda::ops::batch_gemm` |
| `rk4_parallel.wgsl` | Regulatory biology (020–021) | `validate_gpu_rk4` | 8/8 | `barracuda::ops::ode` |
| `mean_reduce.wgsl` | Fitness aggregation | `validate_gpu_pure_workload` | 7/7 | `barracuda::pipeline::ReduceScalarPipeline` |
| `pairwise_jaccard.wgsl` | Pangenome (024) | `validate_gpu_pangenome` | 6/6 | `barracuda::ops::pairwise_distance` |
| `locus_variance.wgsl` | Meta-population (025) | `validate_gpu_meta_pop` | 7/7 | `barracuda::ops::VarianceReduceF64` |
| `spatial_payoff.wgsl` | Game theory (019) | `validate_gpu_game_theory` | 5/5 | `barracuda::ops::stencil` |
| `batch_ipr.wgsl` | Spectral/Anderson (022–023) | `validate_gpu_anderson` | 5/5 | `barracuda::ops::batch_reduce` |
| `pairwise_hamming.wgsl` | Alignment (017) | `validate_gpu_sate` | 5/5 | `barracuda::ops::pairwise_distance` |
| `pairwise_l2.wgsl` | MODES novelty (012) | `validate_gpu_modes` | 15/15 | `barracuda::ops::pairwise_distance` |
| `multi_obj_fitness.wgsl` | Directed evolution (014) | `validate_gpu_directed` | 6/6 | `barracuda::ops::batch_gemm` |
| `swarm_nn_forward.wgsl` | Swarm robotics (015) | `validate_gpu_swarm` | 9/9 | `barracuda::ops` (NN inference) |
| `hill_gate.wgsl` | Signal integration (021) | `validate_gpu_signal` | 9/9 | `barracuda::ops` (Hill gate) |

### WGSL exports (forge crate — single source of truth)

All 16 WGSL shaders are now centralized in `metalForge/forge/src/shaders.rs`.
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
| `stats::*`, `linalg::*`, `numerical::*`, `special::*` | 17 paper modules | 17 CPU port binaries (170/170) |

---

## BarraCUDA APIs — New in `77f70b2e` (Sessions 25–31h)

### Now leveraged by neuralSpring

| API | Use | Status |
|-----|-----|--------|
| `ops::linalg::eigh_householder_qr` | `src/eigh.rs` delegates to upstream | **Wired** (S-12 absorbed) |
| `ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32` | Shader source for HMM GPU forward | **Wired** (shader absorbed) |
| `ops::bio::{PairwiseHammingGpu, PairwiseJaccardGpu, LocusVarianceGpu, SpatialPayoffGpu, BatchFitnessGpu}` | Shader sources re-exported via forge | **Wired** (shaders absorbed) |
| `spectral::BatchIprGpu` | Shader source re-exported via forge | **Wired** (shader absorbed) |
| `ops::rk_stage::WGSL_RK4_PARALLEL` | Shader source re-exported via forge | **Wired** (shader absorbed) |

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

*Evolution readiness tracker — following the hotSpring pattern for ToadStool absorption.*
