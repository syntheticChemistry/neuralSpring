# neuralSpring — Evolution Readiness

**Date**: February 21, 2026 (post-audit)
**ToadStool HEAD**: `dc540afd` (Session 25)
**Pattern**: Python baseline → Rust validation → WGSL shader → ToadStool absorption → lean on upstream

---

## Quick Status

Rust modules (29 total) include `pinn.rs` (Study 001 PINN Burgers) and `deeponet.rs` (Study 002 DeepONet antiderivative) alongside the 25-paper reproduction modules.

| Category | Count | Status |
|----------|-------|--------|
| Python baselines | 206/206 | **COMPLETE** |
| Rust native validation | 219 lib tests, 29 modules, 78 binaries, 90.55% coverage | **COMPLETE** |
| BarraCUDA primitives | 272/272 | **COMPLETE** |
| BarraCUDA CPU ports | 170/170 (17 modules) | **COMPLETE** |
| GPU shader validation | 85/85 (16 WGSL shaders) | **COMPLETE** |
| GPU pipeline validation | 77/77 | **COMPLETE** |
| ToadStool shortcomings | 11/11 absorbed + S-12/S-03b locally resolved | **ALL RESOLVED** |
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
| `pairwise_l2.wgsl` | MODES novelty (012) | `validate_gpu_modes` | 4/4 | `barracuda::ops::pairwise_distance` |
| `multi_obj_fitness.wgsl` | Directed evolution (014) | `validate_gpu_directed` | 4/4 | `barracuda::ops::batch_gemm` |
| `swarm_nn_forward.wgsl` | Swarm robotics (015) | `validate_gpu_swarm` | 4/4 | `barracuda::ops` (NN inference) |
| `hill_gate.wgsl` | Signal integration (021) | `validate_gpu_signal` | 4/4 | `barracuda::ops` (Hill gate) |

### WGSL exports (hotSpring pattern)

Following the hotSpring pattern, each shader is exported as a `pub const` from
its parent Rust library module, making absorption a single-import operation:

| Rust Export | Module |
|-------------|--------|
| `hmm::WGSL_HMM_FORWARD_LOG` | `src/hmm.rs` |
| `pangenome_selection::WGSL_PAIRWISE_JACCARD` | `src/pangenome_selection.rs` |
| `meta_population::WGSL_LOCUS_VARIANCE` | `src/meta_population.rs` |
| `evolved::WGSL_BATCH_FITNESS_EVAL` | `src/evolved/mod.rs` |
| `evolved::WGSL_RK4_PARALLEL` | `src/evolved/mod.rs` |
| `evolved::WGSL_MEAN_REDUCE` | `src/evolved/mod.rs` |
| `game_theory::WGSL_SPATIAL_PAYOFF` | `src/game_theory.rs` |
| `anderson_localization::WGSL_BATCH_IPR` | `src/anderson_localization.rs` |
| `sate_alignment::WGSL_PAIRWISE_HAMMING` | `src/sate_alignment.rs` |

*Additional shaders `pairwise_l2`, `multi_obj_fitness`, `swarm_nn_forward`, `hill_gate` validated by `validate_gpu_modes`, `validate_gpu_directed`, `validate_gpu_swarm`, `validate_gpu_signal` — export-from-module pending.*

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

## BarraCUDA APIs Available but Not Yet Leveraged

| API | Potential Use | Status |
|-----|--------------|--------|
| `TensorSession::{matmul, relu, gelu, softmax, layer_norm}` | Fused ML inference (evolved pipeline fossilized) | Available (S-01/S-11 absorbed) |
| `KernelRouter` 4-tier matmul | Tiled matmul (local shaders fossilized) | Available (S-02 absorbed) |
| Native `Tensor::multi_head_attention` | Replace evolved MHA | **Blocked** (S-03b: projection shader hang) |
| `ReduceScalarPipeline::sum_f64` | Fitness aggregation | Available (local mean_reduce validated) |
| `BatchedRK4F64` | CPU-threaded ODE parameter sweeps | New (Session 25) |
| `ops::filter::{Filter, FilterOperation}` | GPU stream compaction | New (Session 25) |
| `GemmF64::WGSL` | f64 GEMM shader source | New (wetSpring v4) |
| `Tensor::from_arc_buffer` / `try_arc_buffer` | Zero-copy buffer sharing | New (Session 18) |
| `NAK` eigensolve | Anderson localization GPU | Addresses S-12 accuracy gap |

---

## S-12: Locally Resolved (Householder+QR Eigensolver)

The Jacobi `eigh_f64` accuracy gap is addressed by `src/eigh.rs` — a
Householder+QR eigensolver achieving machine epsilon accuracy:

| n | Jacobi Reconstruction | Householder+QR | LAPACK Reference |
|---|----------------------|----------------|-----------------|
| 4 | ~1e-6 | **1.75e-14** | 1e-14 |
| 8 | ~1e-3 | **1.75e-14** | 1e-14 |
| 16 | ~0.01 | **1.75e-14** | 1e-14 |
| 32 | — | **1.75e-14** | 1e-14 |
| 64 | — | **1.75e-14** | 1e-14 |

Validated by `validate_eigh_accuracy` (9/9 PASS). ToadStool NAK eigensolver
remains the eventual GPU-native path.

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
| Line coverage | **90.55%** line / 92.55% region / 94.73% function |
| All files < 1000 LOC | `validate_barracuda_tensor.rs` reduced from 1053 → 864 lines |
| `unsafe` | Forbidden (`#![forbid(unsafe_code)]`) |

---

*Evolution readiness tracker — following the hotSpring pattern for ToadStool absorption.*
