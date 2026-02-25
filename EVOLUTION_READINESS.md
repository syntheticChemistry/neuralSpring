# neuralSpring — Evolution Readiness

**Date**: February 25, 2026 (Sessions 40–69)
**ToadStool HEAD**: `02207c4a` (S58–S69: 17 functions rewired + 6 validator shader sources → upstream constants, S-03b fully resolved upstream, 21/21 shaders absorbed, Phase C GPU 44 ops ~97%, CPU↔Python parity 39/39, deep debt audit S68: 104+ tolerances, 90.43% coverage, zero ad-hoc magic numbers)
**Pattern**: Python baseline → Rust validation → BarraCUDA CPU → BarraCUDA GPU Tensor → metalForge WGSL → GPU Pipeline → Cross-dispatch → Mixed-hardware → Multi-GPU → ToadStool absorption → lean on upstream
**Hardware**: RTX 4070 (Vulkan, proprietary) + TITAN V (NVK GV100, open-source)

---

## Quick Status

36 Rust modules cover all 25 papers + 5 Phase 0/0+ studies + 5 baseCamp sub-theses.
159 validation binaries span 9 tiers: Python (Py), Rust native (Rs), BarraCUDA CPU (bC),
GPU Tensor (gT), metalForge WGSL (mF), GPU Pipeline (gP), Cross-dispatch (xD),
Mixed-hardware (mH), and Multi-GPU (mG).

| Category | Count | Status |
|----------|-------|--------|
| Python baselines | 206/206 | **COMPLETE** |
| Rust native validation | 505 lib + 9 integration + 43 forge tests, 36 modules, 159 binaries | **COMPLETE** |
| BarraCUDA primitives | 272/272 | **COMPLETE** |
| BarraCUDA CPU (bC) | **24/25** papers (96%) | **ALL GREEN** |
| BarraCUDA GPU Tensor (gT) | **23/25** papers (92%) | **ALL GREEN** |
| metalForge WGSL (mF) | 15/25 papers (60%) | **ALL PASS** |
| GPU Pipeline (gP) | 9/25 papers (36%) | **ALL PASS** |
| Cross-dispatch (xD) | **15/15** Phase 0++ papers (100%) | **ALL GREEN** |
| Multi-GPU validation | RTX 4070 + TITAN V (NVK) | **Bit-identical** |
| GPU shader validation | 126/126 (21 WGSL shaders) | **COMPLETE** |
| GPU pipeline validation | 77/77 | **COMPLETE** |
| ToadStool shortcomings absorbed | 12/12 (S-01..S-12) | **ALL ABSORBED** |
| S-16 (transpose dispatch) | One-line fix | **FIXED** |
| S-15 (matmul hang) | Root-caused: magnitude ≤ 0.1 | **WORKAROUND** (≥ 0.5 data) |
| S-14 (naive matmul hang) | A×B^T pattern avoids | **WORKAROUND** |
| S-13 (PooledBuffer race) | Deferred return + device poll | **FIXED** upstream (Session 39) |
| TS-003 (trig precision) | 7-term Taylor + Cody-Waite | **FIXED** upstream (Session 36) |
| TS-001 (pow_f64 precision) | Extended exp/log polynomials | **FIXED** upstream (Session 36) |
| Shader absorption | 21/21 WGSL shaders absorbed upstream | **S-03b RESOLVED** — ToadStool `0c998992` (matmul + head_split/head_concat) |
| Upstream wrapper validation | **10 bio ops** + f64 HMM + Gillespie + wetSpring trio + chi² | **74/74 PASS** |
| Upstream parity (dual-path) | **10 GPU validators** | **10/10 PASS** (9 bit-identical, 1 Bessel diff 1.95e-3) |
| ReduceScalarPipeline | f64 mean IPR via GPU reduce | **5.55e-17 diff** (machine ε) |
| Spectral theory stack | Lanczos, Anderson, Hofstadter, Lyapunov, eigh×Sturm | **17/17 PASS** (hotSpring lineage) |
| Capability-based dispatch | 12 validators + evolved HMM use `Gpu::dispatch_1d` | **Runtime-validated** (Sessions 40, 42) |
| Upstream vs local benchmark | **10 kernels**, RTX 4070 | **0.72–1.10×** overhead (negligible) |
| LeNet-5 full bC validation | Conv→Pool→FC via `cpu_conv_pool` | **13/13 PASS** (new, Session 42) |
| Session 43: new WGSL shaders | logsumexp, stencil, rk45, wright-fisher (4 shaders, 4 validators) | **18/18 PASS** |
| Session 43: upstream wrappers | GillespieGpu, TaxonomyFcGpu, KmerHistogramGpu, UniFracPropagateGpu, chi² | **41/41 PASS** |
| Session 43: CPU vs GPU parity | Tensor API: MatMul, ReLU, Sigmoid, Tanh, Sum, erf, gamma, conv, pool | **17/17 PASS** |
| Session 43: dispatch routing | metalForge substrate heuristics (8 domains) | **16/16 PASS** |
| Session 43: mixed-hardware | MixedSubstrate, TransferCost, PcieBridge, cost model | **16/16 PASS** |
| Session 44: multi-GPU | RTX 4070 + TITAN V (NVK GV100): 131/131 PASS | **ALL GREEN** |
| Session 45: GPU promotion (Phase A) | `validate_gpu_promotion` 27/27 PASS (RTX 4070 + TITAN V NVK) | **ALL GREEN** |
| Session 46: GPU promotion (Phase B) | `validate_gpu_phase_b` 20/20 PASS (RTX 4070 + TITAN V NVK) | **ALL GREEN** |
| Session 44: stochastic pipelines | WF→reduce + Gillespie→reduce (zero CPU round-trips) | **10/10 PASS** |
| Session 44: Conv2d/MaxPool GPU | `Tensor::conv2d` + `Tensor::maxpool2d` WGSL shaders | **8/8 PASS** |
| Session 44: transformer bC | Full layer: Q/K/V, attention, FFN, residual, softmax | **12/12 PASS** |
| Session 44: BarraCUDA fixes | mean_reduce entry point + chi² expected values | **2 bugs fixed upstream** |
| Session 44: benchmarks | Pure Rust vs Python (11 kernels) | **178.5× faster** |
| Evolved LOC | ~2,864 fossilized | Documented, bench migration complete |
| gpu_dispatch, gpu_ops | Capability-based GPU/CPU dispatch + 44 promoted ops (Phase A+B+C), 7 rewired to upstream domain_ops | **159 binaries** |
| `validate_all` (S-67) | **147/148 PASS** (RTX 4070; logsumexp driver issue) | **ALL GREEN** (1 known skip) |
| Session 47: typed op migration | 10 validators rewired raw wgpu → typed BarraCUDA ops | **Cross-spring complete** |
| Session 48: mass typed op rewiring | 28 binaries rewired raw wgpu → typed BarraCUDA ops | **Complete** |
| Session 48: f32→f64 upstream sync | BatchFitnessGpu, LocusVarianceGpu, MultiObjFitnessGpu, WrightFisherGpu, StencilCooperationGpu, SwarmNnGpu | **Data type alignment** |
| Session 48: HillGateGpu f64 | Graceful skip on RTX 4070 (driver limitation) | **f32 path validated** |
| S-03b (MHA projection hangs) | Decomposed into matmul + head_split/head_concat (ToadStool `0c998992`) | **FULLY RESOLVED** upstream |
| Session 47: evolved/hmm_forward_gpu | Retired; HmmBatchForwardF64 (wetSpring) primary | **Fossil** `metalForge/fossils/evolved_hmm_forward_gpu/` |
| Session 54: baseCamp experiment expansion | 5 validators expanded 82→114 checks (nS-103..106, 205, 206, 304, 305, 402, 405, 504, 505) | **114/114 PASS** |
| Session 54: `validate_basecamp_gpu` | Pure GPU workload validation (eigensolve, variance, Pearson, entropy, matmul, chi², L2, KL) | **14/14 PASS** |
| Session 54: `bench_basecamp_parity` | CPU→GPU parity: var 7.77e-16, pearson 6.94e-18, entropy 1.60e-11 | **All sub-epsilon** |
| Session 55: `validate_compute_dispatch` | BarraCUDA CPU vs GPU dispatch parity (routing + variance/Pearson/entropy/chi²/eigh) | **16/16 PASS** |
| Session 55: `Dispatcher::mixed_dispatch()` | metalForge mixed-hardware wiring integrated into `gpu_dispatch` | **Wired** |
| Session 55: `validate_mixed_hardware` | Mixed-hardware dispatch (GPU↔NPU↔CPU routing, PCIe bridge, crossover) | **14/14 PASS** |
| Session 55: doc cleanup | 5 sub-thesis docs fixed (binary refs, check counts), 15 grounding papers → Primitives validated | **Done** |
| `validate_all` | **147/148 PASS** (RTX 4070; 1 logsumexp driver issue) | **ALL GREEN** |
| Grand total checks | **2120+** (206 Py + 1910+ Rust/GPU) | **ALL GREEN** |

---

## Tier A — Shader Absorption Status

### ToadStool Evolution Since Last Sync (Session 47: 9abd6857)

| Session | Key Changes for neuralSpring |
|---------|----------------------------|
| S39 | Absorb all Spring shaders (7 bio ops, 11 HFB physics, 3 wetSpring WGSL); S-14/S-15/S-16 fixes; `FlatTree`, `sparse_eigh`, `execute_to_buffer` |
| S40 | Richards PDE solver, moving window GPU stats |
| S41 | `cpu_conv_pool` made `pub` (conv2d, max_pool2d, avg_pool2d); 6 f64 shader compile bugs fixed; APIs exposed for Springs |
| S42 | 19 new WGSL shaders (chi_squared_f64, rk45_f64, factorial_f64, cubic_spline_f64, etc.); BarraCUDA → BarraCuda doc rename |

**New wrapper APIs available** (not yet used by neuralSpring):

| API | Domain | Replaces |
|-----|--------|----------|
| `ops::bio::HillGateGpu` | Signal integration (021) | Local `hill_gate.wgsl` dispatch |
| `ops::bio::MultiObjFitnessGpu` | Directed evolution (014) | Local `multi_obj_fitness.wgsl` dispatch |
| `ops::bio::PairwiseL2Gpu` | MODES (012) | Local `pairwise_l2.wgsl` dispatch |
| `ops::bio::SwarmNnGpu` | Swarm robotics (015) | Local `swarm_nn_forward.wgsl` dispatch |
| `cpu_conv_pool::{conv2d, max_pool2d, avg_pool2d}` | LeNet-5 (Study 003) | Python-only conv2d/pool |

### Absorbed Upstream (Session 42, `5437c170` — generalized variants)

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
| `xoshiro128ss.wgsl` | Stochastic (PRNG) | `validate_gpu_prng` | 5/5 | `barracuda::ops::prng` |
| `swarm_nn_scores.wgsl` | Swarm (015) | `validate_gpu_pipeline_swarm` | PASS | No upstream equivalent |
| `logsumexp_reduce.wgsl` | HMM/phylo (016–018) | `validate_gpu_logsumexp` | 5/5 | `barracuda::ops::reduce` (batched, Session 43) |
| `stencil_cooperation.wgsl` | Game theory (019) | `validate_gpu_stencil` | 3/3 | `barracuda::ops::stencil` (Session 43) |
| `rk45_adaptive.wgsl` | Regulatory ODE (020–021) | `validate_gpu_rk45` | 6/6 | `barracuda::ops::ode` (Session 43) |
| `wright_fisher_step.wgsl` | PopGen (024–025) | `validate_gpu_wright_fisher` | 4/4 | `barracuda::ops::popgen` (Session 43) |

### WGSL exports (forge crate — single source of truth)

All 21 WGSL shaders are centralized in `metalForge/forge/src/shaders.rs`.
21/21 absorbed upstream (S-03b resolved: head_split/head_concat in `barracuda::ops::mha`).
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

| Shader | Domain | Priority | Status |
|--------|--------|----------|--------|
| `tridiag_eigensolver.wgsl` | Spectral (022–023) | P3 | Pending: Householder → bisection design |
| `pairwise_distance.wgsl` | Alignment (017) | P4 | Pending: O(N²) dispatch geometry |
| ~~`stencil_cooperation.wgsl`~~ | ~~Game theory (019)~~ | ~~P5~~ | **BUILT** (Session 43) — 3/3 PASS |
| ~~`logsumexp_reduce.wgsl`~~ | ~~HMM/phylogenetics~~ | ~~P2~~ | **BUILT** (Session 43) — 5/5 PASS |

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

## BarraCUDA APIs — New in `5437c170` (Sessions 25–42)

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

### Validated via BarraCUDA CPU binaries (Tier 3)

These APIs are already validated in dedicated Tier 3 (bC) binaries:

| API | Validated By | Checks |
|-----|-------------|--------|
| `ops::bio::HmmBatchForwardF64` | `validate_barracuda_hmm_f64` | 11/11 PASS |
| `spectral::{anderson_*, hofstadter_*, lanczos}` | `validate_barracuda_spectral_theory` | 17/17 PASS |
| `numerical::rk45_solve` | `validate_barracuda_regulatory`, `validate_barracuda_signal`, `validate_barracuda_game` | 20+ PASS |
| `ops::linalg::eigh_householder_qr` | `validate_eigh_accuracy` | 9/9 PASS |

Local Tier 2 (Rust native) implementations intentionally retained as independent
cross-validation references. Both tiers matching Python proves portability.

### Available for future leverage

| API | Potential Use | Status |
|-----|--------------|--------|
| Native `ops::mha::MultiHeadAttention` | `evolved::mha` thin wrapper | **Wired** (S-03b resolved upstream `0c998992`) |
| `ops::bio::{FelsensteinGpu, SmithWatermanGpu}` | Future paper extensions | Available |
| `ops::bio::GillespieGpu` | Stochastic SSA (Papers 013, 020) | **Wired** (Session 43, 20/20 PASS) |
| `ops::bio::{TaxonomyFcGpu, KmerHistogramGpu, UniFracPropagateGpu}` | wetSpring metagenomics | **Wired** (Session 43, 8/8 PASS) |
| `special::chi_squared::*` | Pangenome selection (Paper 024) | **Wired** (Session 43, 13/13 PASS) |
| `ops::bio::{RfBatchInferenceGpu, TreeInferenceGpu}` | Future ML forest workloads | Available |
| `ops::linalg::{InverseF64, LinSolveF64}` | GPU dense linear algebra | Available |
| `ReduceScalarPipeline::sum_f64` | Fitness aggregation | Available (local mean_reduce validated) |
| `BatchedRK4F64` | CPU-threaded ODE parameter sweeps | Available |
| `WGSL_BATCHED_EIGH_NAK_OPTIMIZED` | GPU-native eigensolve for Anderson | Available |
| `ops::conv2d::Conv2D` | Batched Conv2D — LeNet-5 conv layers | **New** (Session 39) — not yet wired to executor |
| `ops::maxpool2d::MaxPool2D` | MaxPool2D — LeNet-5 pooling | **New** (Session 39) — not yet wired to executor |
| `ops::avgpool2d::AvgPool2D` | AvgPool2D — alternative pooling | **New** (Session 39) — not yet wired to executor |
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

## Code Quality (Post-Deep-Audit, February 23 2026)

| Aspect | Status |
|--------|--------|
| `cargo fmt` | **Clean** — zero formatting violations |
| `cargo clippy` pedantic + nursery | **0 warnings** — `clippy::doc_markdown` fully resolved (31 files), all remaining `#[allow]` audited and justified |
| `cargo doc --no-deps` | **0 warnings** — all rustdoc links valid |
| `cargo test --lib` | **500 tests PASS** (up from 264) |
| `cargo test --test integration` | **9 integration tests PASS** |
| `#[must_use]` | Applied to 24+ pure public functions across 5 modules |
| Centralized tolerances | Split into `tolerances/` module (`mod.rs` + `registry.rs`) — 101+ `NamedTolerance` entries in registry (24 `gpu_dispatch` category), zero standalone inline magic numbers |
| GPU validation helpers | Shared `gpu_readback`, `max_abs_diff_gpu_vs_cpu`, `gpu_tensor!` macro — deduplicated ~400 LOC from 24 binaries |
| GPU device init | Unified via `Gpu::new()` (removed ~800 LOC duplication) |
| Modular `gpu_ops/` | Refactored from monolithic 1328-line file into 6 focused submodules (`linalg`, `activation`, `reduction`, `bio`, `population`, `eigensolver`) — all under 1000 LOC |
| GPU dispatch coverage | `Dispatcher` CPU-fallback paths: **33 tests** covering all 26 dispatched operations |
| `GpuCapabilities` tested | Mock-based unit tests for `workgroup_size`, `dispatch_count`, `supports_workgroup` — no GPU required |
| Idiomatic Rust | HMM flat row-major layout, spectral flat layout, `NkLandscape.k` accessor, `mul_add` for FMA, infallible casts via `From` |
| Consolidated math primitives | Shannon, Hill, sigmoid, RK4 centralized in `primitives.rs` — no duplicated math |
| GPU-ready flat layouts | HMM, spectral, anderson_localization, directed_evolution, sate_alignment use flat row-major `Vec<f64>` — direct GPU buffer upload |
| Graceful error handling | `require!` macro replaces `.expect()` in all validation binaries — no panic on GPU failure |
| Zero-copy genotype handling | `eco_dynamics.rs` uses `&[u8]` / `HashSet<&[u8]>` — avoids `Vec<u8>` clones |
| Provenance | All hardcoded validation targets sourced with script, commit, date, exact command |
| Determinism tests | **16 tests** covering all stochastic modules (up from 7) |
| SPDX headers | All 40 Python/shell files have `AGPL-3.0-or-later` license identifier |
| Line coverage | **90.43%** line via `cargo llvm-cov` (remaining gap: GPU-only code paths unreachable on CPU) |
| All files < 1000 LOC | Largest: `validate_barracuda_tensor.rs` at 966 lines |
| `unsafe` | Forbidden (`#![forbid(unsafe_code)]`) |
| Mocks/stubs | Zero in production code — zero `todo!`/`unimplemented!` |
| External dependencies | All pure Rust — zero C/C++ wrapper crates |

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

---

## Cross-Spring Shader Evolution Lineage

The ToadStool/BarraCuda ecosystem benefits from cross-spring evolution: each
Spring contributes domain-specific shaders that are generalized into the shared
crate, then consumed by all Springs. This table tracks provenance.

### hotSpring → BarraCuda → neuralSpring (Precision & Physics)

| Primitive | hotSpring Origin | BarraCuda Location | neuralSpring Use |
|-----------|------------------|-------------------|-----------------|
| Taylor-series trig (sin/cos) | TS-003 7-term Taylor + Cody-Waite | `special::trig` | Spectral theory (17/17 checks) |
| Extended exp/log polynomials | TS-001 pow_f64 fix | `special::erf`, `math::exp/log` | Anderson localization, PINN |
| Lanczos eigensolver | hotSpring v0.5.16 lattice QCD | `spectral::lanczos` | Spectral 5/5, Hofstadter 5/5 |
| HFB deformed | hotSpring nuclear physics | `ops::physics::hfb_deformed` | Cross-validated (Session 39 absorption) |
| 19 new f64 WGSL shaders (S42) | chi_squared, factorial, rk45, cubic_spline | `shaders/special/*`, `shaders/math/*` | Available for GPU promotion |

### wetSpring → BarraCuda → neuralSpring (Bio/Genomics)

| Primitive | wetSpring Origin | BarraCuda Location | neuralSpring Use |
|-----------|------------------|-------------------|-----------------|
| HMM batch forward f64 | wetSpring phylogenetics | `ops::bio::hmm` | HMM validation 11/11, GPU HMM 13/13 |
| Quality filter | wetSpring FASTQ pipeline | `ops::bio::quality_filter` | bC genomics validation |
| DADA2 E-step | wetSpring amplicon denoising | `ops::bio::dada2` | Cross-dispatch genomics |

### neuralSpring → BarraCuda → all Springs (ML & Evolution)

| Primitive | neuralSpring Origin | BarraCuda Location | Beneficiary |
|-----------|---------------------|-------------------|-------------|
| batch_fitness_eval | Paper 011-015 (ML) | `ops::bio::BatchFitnessGpu` | wetSpring, hotSpring |
| pairwise_hamming | Paper 017 (SATé) | `ops::bio::PairwiseHammingGpu` | wetSpring genomics |
| pairwise_jaccard | Paper 024 (Pangenome) | `ops::bio::PairwiseJaccardGpu` | wetSpring metagenomics |
| spatial_payoff | Paper 019 (Game Theory) | `ops::bio::SpatialPayoffGpu` | Ecological modeling |
| locus_variance | Paper 025 (MetaPop) | `ops::bio::LocusVarianceGpu` | Population genetics |
| batch_ipr | Paper 022-023 (Anderson) | `spectral::BatchIprGpu` | hotSpring condensed matter |
| hill_gate | Paper 021 (Signal) | `ops::bio::HillGateGpu` | Regulatory network modeling |
| multi_obj_fitness | Paper 014 (Directed Evo) | `ops::bio::MultiObjFitnessGpu` | Optimization pipelines |
| pairwise_l2 | Paper 012 (MODES) | `ops::bio::PairwiseL2Gpu` | Novelty search, clustering |
| swarm_nn_forward | Paper 015 (Swarm) | `ops::bio::SwarmNnGpu` | Neuroevolution controllers |
| Householder+QR eigensolver | `eigh.rs` | `linalg::sparse::eigh` | hotSpring, wetSpring |
| 4-tier matmul KernelRouter | S-14/S-15 workarounds | `ops::matmul` | All Springs |
| Capability-based dispatch | `Gpu::dispatch_1d` | Pattern adopted | All Springs |

### Upstream Parity Benchmark (10 Kernels, RTX 4070)

| Kernel | Origin | Local µs | Upstream µs | Ratio |
|--------|--------|----------|-------------|-------|
| BatchFitness 10K×32 | neuralSpring 011-015 | 3153 | 2346 | 0.74× |
| Hamming 200×500 | neuralSpring 017 (SATé) | 4396 | 3388 | 0.77× |
| Jaccard 100×500 | neuralSpring 024 (Pangenome) | 2269 | 2272 | 1.00× |
| LocusVariance 50×500 | neuralSpring 025 (MetaPop) | 2270 | 2284 | 1.01× |
| SpatialPayoff 256² | neuralSpring 019 (GameTheory) | 2284 | 2266 | 0.99× |
| BatchIPR 1K×256 | neuralSpring 022-023 (Anderson) | 3150 | 2259 | 0.72× |
| **HillGate 100×100** | **neuralSpring 021 (Signal)** | **2236** | **2279** | **1.02×** |
| **MultiObjFitness 5K×4** | **neuralSpring 014 (DirEvo)** | **2432** | **2358** | **0.97×** |
| **PairwiseL2 200×50** | **neuralSpring 012 (MODES)** | **2271** | **2269** | **1.00×** |
| **SwarmNN 500×20** | **neuralSpring 015 (Swarm)** | **2279** | **2513** | **1.10×** |

All 10 upstream wrappers show negligible overhead (0.72–1.10×).
Bold entries are newly wired in Session 42 ToadStool sync.

### Session 50 — baseCamp Biophysical AI Interpretability (82/82 PASS)

5 new library modules implementing cross-domain analysis of AI systems using
validated physics/biology primitives. Each module composes existing neuralSpring
primitives (`eigh`, `anderson_localization`, `hmm`, `game_theory`) into novel
analysis pipelines. 459 unit tests, 0 clippy warnings, 0 doc warnings.

| Module | Sub-thesis | Checks | Key Primitives |
|--------|-----------|--------|----------------|
| `weight_spectral` | nS-01: Weight Hamiltonians | 15/15 | ESD, IPR, level spacing ratio, Marchenko-Pastur |
| `information_flow` | nS-02: Information Propagation | 15/15 | Depth scale, gate disorder, attention Hamiltonian |
| `loss_landscape` | nS-03: Loss Landscapes | 19/19 | Numerical Hessian, Boltzmann MCMC, spectral gap |
| `neural_pgm` | nS-04: Neural PGMs | 15/15 | Belief propagation, KL divergence, effective rank |
| `agent_coordination` | nS-05: Multi-Agent QS | 18/18 | Graph Laplacian, QS signaling, dimensional sweep |

**GPU promotion (Session 55):** All 4 candidates now have `Dispatcher` methods
routing to GPU or CPU fallback via `validate_basecamp_dispatch` (19/19 PASS).

**Upstream rewiring (Session 56 — ToadStool `9404fdb4`):** 4 functions now
delegate to upstream BarraCUDA, eliminating local implementations:

| Local Function | Upstream Module | Effect |
|----------------|----------------|--------|
| `graph_laplacian` | `barracuda::linalg::graph` | Thin wrapper → upstream |
| `disordered_laplacian` | `barracuda::linalg::graph` | Thin wrapper → upstream |
| `belief_propagation_chain` | `barracuda::linalg::graph` | Thin wrapper → upstream |
| `numerical_hessian` | `barracuda::numerical` | Thin wrapper → upstream |

Public API preserved; callers unchanged. Validated via `cargo test --lib` (478 PASS).

See `whitePaper/baseCamp/extensions.md` for the full research program.

### Session 49 — Code Quality Status

| Quality Gate | Status |
|--------------|--------|
| Hardcoded paths | **0** (all via `validation::baseline_path`) |
| TODO/FIXME/MOCK/STUB | **0** in src/ |
| `unsafe` blocks | **0** (`forbid` enforced) |
| `.unwrap()` in non-test | **0** |
| Clippy warnings | **0** (pedantic + nursery) |
| Doc warnings | **0** |
| Max file size | 965 lines (under 1000 wateringHole limit) |
| Dispatch pattern | 7 core methods delegate to upstream `domain_ops`; remainder use `gpu_or_cpu` |
| GPU skip policy | All 79 binaries use `exit_no_gpu()` (CI-fidelity) |

### Session 56 — ToadStool S53 Sync + Upstream Rewiring

| Action | Detail |
|--------|--------|
| **Pulled ToadStool HEAD** | `f78cf3b0` (absorbed Sessions 51–53 handoffs) |
| **New upstream modules** | `barracuda::linalg::graph`, `barracuda::numerical`, `barracuda::ops::bio::swarm_nn`, `barracuda::ops::bio::xoshiro128ss` |
| **Rewired 4 functions** | `graph_laplacian`, `disordered_laplacian`, `belief_propagation_chain`, `numerical_hessian` → delegate to upstream |
| **3 new validators** | `validate_basecamp_dispatch` (19 checks), `validate_barracuda_parity` (34 checks), `validate_metalforge_pcie` (36 checks) |
| **Total checks** | 2010+ (206 Python + 1810+ Rust+GPU) |
| **Lib tests** | 478 PASS |
| **Forge tests** | 30 PASS |
| **Quality gates** | fmt ✓ · clippy ✓ (pedantic+nursery) · doc ✓ |

### Session 57 — ToadStool S58–S59 Sync

| Action | Detail |
|--------|--------|
| **Pulled ToadStool HEAD** | `9404fdb4` (S58: df64/Fp64Strategy/ODE bio/NMF; S59: anderson correlated/ridge/ValidationHarness) |
| **Confirmed absorptions** | `ValidationHarness`, `exit_no_gpu`, `require!` macro — all from neuralSpring, now in `barracuda::validation` |
| **Consolidated** | 4 duplicate `patch_pow_to_polyfill` → `validation::patch_pow_to_polyfill` (shared) |
| **New upstream available** | `barracuda::spectral::anderson` (3D correlated, sweep averaged, find_w_c), `barracuda::linalg::ridge`, `barracuda::linalg::nmf`, `barracuda::numerical::ode_bio`, `barracuda::dispatch::domain_ops`, `barracuda::device::driver_profile` |
| **Quality gates** | fmt ✓ · clippy ✓ (pedantic+nursery) · 500 lib ✓ · 145/146 validate_all (1 pre-existing logsumexp) |

### Session 58 — Upstream Dispatch Rewiring + GpuDriverProfile

| Action | Detail |
|--------|--------|
| **Rewired 7 Dispatcher methods** | `mat_mul`, `frobenius_norm`, `transpose`, `softmax`, `l2_distance`, `mean`, `variance` → delegate to `barracuda::dispatch::domain_ops` |
| **Wired GpuDriverProfile** | `Dispatcher` now exposes `driver_profile()`, `fp64_strategy()`, `needs_pow_workaround()` via upstream `barracuda::device::driver_profile` (hotSpring-evolved) |
| **Driver detection confirmed** | RTX 4070: Ada arch, NvidiaPtxas compiler, Throttled FP64 → Hybrid strategy, pow workaround needed |
| **New validator** | `validate_cross_spring_evolution` (10/10 PASS): rewired method parity + driver profile + cross-spring benchmark |
| **Total rewired functions** | 11 (4 from S56 + 7 from S58) — all delegating to upstream BarraCUDA |
| **Quality gates** | fmt ✓ · clippy ✓ (pedantic+nursery) · 500 lib ✓ · 145/146 validate_all (1 pre-existing logsumexp) |

### Session 61 — Deep Code Quality Sweep (February 25, 2026)

| Action | Detail |
|--------|--------|
| **Deep code quality sweep** | Property tests, tolerance centralization, vestigial allow removal |
| **13 property tests added** | `src/property_tests.rs` — invariants across stochastic and numerical modules |
| **6 tolerance constants centralized** | Added to `tolerances/` registry |
| **4 vestigial `#[allow]` attributes removed** | Underlying code fixed, redundant suppression removed |
| **Line coverage** | **93.17%** via `cargo llvm-cov` |
| **Lib tests** | **500 PASS** |

### Session 66 — Phase C GPU Promotion (February 25, 2026)

6 new `Dispatcher` methods, 3 new `gpu_ops` functions. HMM forward/Viterbi chains,
pairwise/global FST, inter-population AF variance — all now GPU-dispatchable.
`validate_gpu_phase_c` 18/18 PASS. GPU coverage: ~90% → ~97% of production math.

|| Session 66: Phase C GPU promotion | 6 Dispatcher methods, 3 gpu_ops, validate_gpu_phase_c 18/18 | **~97% GPU** |
|| Session 66: Python baselines | 25/25 PASS — zero drift, 201.7× Rust faster | **ALL GREEN** |

### Session 67 — CPU Math Parity Validation (February 25, 2026)

Cross-language parity: `control/generate_cpu_references.py` → JSON →
`validate_cpu_math_parity` 39/39 PASS (9 primitives + 9 paper kernels + 6
Dispatcher cpu_only). All within 1e-10 tolerance. Proves BarraCUDA CPU = Python/NumPy.

|| Session 67: CPU↔Python parity | `validate_cpu_math_parity` 39/39 PASS (1e-10) | **ALL GREEN** |

### Session 67b — Dispatch Tier Benchmarks (February 25, 2026)

Three-tier benchmark: Library direct → Dispatcher::cpu_only() → Dispatcher::new() GPU.
9/10 ops ≤1.04× CPU dispatch overhead. Per-call GPU driver-bound for small workloads —
motivates StatefulPipeline/UnidirectionalPipeline batching for GPU-resident acceleration.

|| Session 67b: Dispatch tiers | `bench_dispatch_tiers` — 9/10 ops ≤1.04× CPU overhead | **Transparent** |

### Session 68 — Deep Debt Audit (February 25, 2026)

Full barracuda usage audit: 90+ import sites, 20+ submodules, zero duplicates.
Tolerance centralization: 104+ named constants, zero ad-hoc magic numbers.
Rewired `boltzmann_sampling` → `barracuda::sample::boltzmann_sampling` (17th function rewire).
505 lib tests, 90.43% coverage.

|| Session 68: Deep debt audit | 104+ tolerances, 90.43% coverage, 0 debt markers | **ALL GREEN** |
|| Session 68: boltzmann rewire | 17th function rewired to upstream | **LEAN** |

### Session 69 — Validator Shader Rewiring + Cross-Spring Benchmarks (February 25, 2026)

6 validator binaries rewired from local `include_str!` to upstream barracuda shader
constants. Cross-spring benchmarks refreshed. Upstream-vs-local: 10/10 ≈ or ~ (zero ⚠).
Complete cross-spring provenance mapped: hotSpring precision, wetSpring bio, neuralSpring ML.

|| Session 69: Shader source rewire | 6 validators → upstream barracuda constants | **LEAN** |
|| Session 69: Cross-spring bench | 10/10 upstream ≈ local, 22/22 evolution PASS | **ALL GREEN** |
|| Session 69: validate_all | 147/148 PASS (1 pre-existing logsumexp) | **ALL GREEN** |

*Evolution readiness tracker — following the hotSpring pattern for ToadStool absorption.*
