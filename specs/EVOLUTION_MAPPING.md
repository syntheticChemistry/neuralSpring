# neuralSpring — Evolution Mapping: Rust Module → WGSL Shader → Pipeline Stage

**Last Updated**: February 23, 2026 (Session 44: multi-GPU, Conv2d/MaxPool GPU, transformer bC)
**Purpose**: Concrete mapping from Phase 0 Python → Phase 1 Rust → Phase 2 GPU

---

## Tier Classification

| Tier | Meaning | Criteria |
|------|---------|----------|
| **A** (rewire) | Direct port — pure math, no framework dependencies | NumPy-only implementations, analytical known-values |
| **B** (adapt) | Needs adaptation — training loops, data dependencies | PyTorch training, real data, stochastic |
| **C** (new) | New implementation — no Python equivalent | GPU-specific (flash attention, fused kernels) |

---

## Module-by-Module Mapping

### Tier A — Direct Rewire (validated, ready for GPU promotion)

| Python Module | Rust Module | WGSL Shader | Pipeline Stage | Status |
|---------------|-------------|-------------|----------------|--------|
| `transformer/` softmax | `transformer::softmax` | `attention.wgsl` (softmax stage) | Inference | **VALIDATED** (18 checks) |
| `transformer/` GELU | `transformer::gelu` | elementwise | Inference | **VALIDATED** (18 checks) |
| `transformer/` LayerNorm | `Tensor::layer_norm_wgsl()` | `layer_norm.wgsl` | Inference | **ABSORBED** — native BarraCUDA (S-08) |
| `transformer/` SDPA | `TensorSession::attention()` | `attention.wgsl` | Inference | **ABSORBED** — native BarraCUDA (S-11) |
| `surrogate/` Rastrigin | `surrogate::rastrigin_2d` | N/A (test function) | Validation | **VALIDATED** (15 checks) |
| `surrogate/` Rosenbrock | `surrogate::rosenbrock_2d` | N/A (test function) | Validation | **VALIDATED** (15 checks) |
| `surrogate/` Ackley | `surrogate::ackley_2d` | N/A (test function) | Validation | **VALIDATED** (15 checks) |
| `surrogate/` R²/RMSE/MAE | `metrics::*` | `FusedMapReduceF64` | Validation | **VALIDATED** (10 checks) |

### Tier A — Phase 0++ Paper Reproductions (validated, ready for GPU promotion)

All Phase 0++ modules are pure math, deterministic (seed=42), and use no
external dependencies beyond `crate::rng::Rng`. They are ideal Tier A
candidates for BarraCUDA CPU port and subsequent GPU promotion.

| Python Module | Rust Module | Checks | WGSL Shader Target | Key Primitive |
|---------------|-------------|--------|--------------------|----|
| `counterdiabatic/` | `counterdiabatic.rs` | 19 | `gemm_f64` + `softmax.wgsl` | NK fitness, Boltzmann |
| `modes/` | `modes.rs` | 9 | `reduce_sum` + `elementwise` | Change/novelty/complexity |
| `eco_dynamics/` | `eco_dynamics.rs` | 7 | batch `gemm_f64` + `reduce_sum` | Multi-niche EA |
| `directed_evolution/` | `directed_evolution.rs` | 7 | batch `gemm_f64` + `reduce_max` | 5 selection algorithms |
| `swarm_robotics/` | `swarm_robotics.rs` | 7 | batch `gemm_f64` | Heterogeneous controllers |
| `hmm_phylo/` | `hmm.rs` | 17 | `gemm_f64` chain (log-domain) | Forward/backward/Viterbi |
| `sate_alignment/` | `sate_alignment.rs` | 8 | `gemm_f64` (distance matrix) | NJ tree + alignment |
| `introgression/` | `introgression.rs` | 13 | `gemm_f64` chain + log-sum-exp | PhyloNet-HMM + LRT |
| `game_theory/` | `game_theory.rs` | 8 | `gemm_f64` + `softmax.wgsl` | Replicator, QS spatial |
| `regulatory_network/` | `regulatory_network.rs` | 5 | `elementwise` | Hill ODE + RK4 |
| `signal_integration/` | `signal_integration.rs` | 8 | `elementwise` | Two-input Hill AND gate |
| `spectral_commutativity/` | `spectral_commutativity.rs` | 8 | `gemm_f64` | Commutator [A,B] |
| `anderson_localization/` | `anderson_localization.rs` | 8 | `tridiag` + `eigh_f64` | Aubry-André, IPR |
| `pangenome_selection/` | `pangenome_selection.rs` | 8 | sparse GEMM + chi-sq reduce | PA matrix, selection test |
| `meta_population/` | `meta_population.rs` | 8 | variance decomp + `pearson` | FST, Mantel, thermal corr |
| `pinn/` | `pinn.rs` | 16 | `barracuda::tensor` matmul + tanh | Cole-Hopf, MLP forward, PDE residual |
| `deeponet/` | `deeponet.rs` | 17 | `barracuda::tensor` matmul + dot | Branch-trunk, polynomial eval |

### Tier A+ — BarraCUDA GPU Primitives (validated 2026-02-20)

FFT validation pinned to ToadStool's Cooley-Tukey radix-2 WGSL implementation.

| BarraCUDA Module | Validation Binary | Checks | Status |
|------------------|-------------------|--------|--------|
| `ops::fft::{Fft1D, Ifft1D, Fft1DF64, Rfft}` | `validate_barracuda_fft` | 24 | **PASS** (RTX 4070 Vulkan) |

Validated properties: inverse round-trip (N=16, N=256), Parseval's theorem,
delta→constant, constant→delta, cosine energy concentration, multi-frequency
decomposition. All analytical (no Python baseline needed).

### Tier A+ — BarraCUDA CPU Primitives (validated 2026-02-19)

Direct `barracuda::*` calls validated against analytical / NIST DLMF baselines.

| BarraCUDA Module | Validation Binary | Checks | Status |
|------------------|-------------------|--------|--------|
| `stats::{variance, std_dev, pearson, covariance, spearman, norm_*}` | `validate_barracuda_stats` | 13 | **PASS** |
| `linalg::{solve_f64, lu_det, lu_solve, eigh_f64, cholesky_f64, tridiag}` | `validate_barracuda_linalg` | 17 | **PASS** |
| `special::{gamma, factorial, erf, bessel, legendre, hermite, laguerre}` | `validate_barracuda_special` | 26 | **PASS** |
| `optimize::{nelder_mead, bisect, brent}` | `validate_barracuda_optimize` | 10 | **PASS** |
| `shaders::precision::cpu` (add, mul, fma, dot, kahan\_sum) | `validate_barracuda_precision` | 12 | **PASS** |
| **Tensor API** (90 ops — native `layer_norm`, `log_softmax`, `leaky_relu`, `elu`) | `validate_barracuda_tensor` | 90 | **PASS** |
| **Tensor f64 API** (GPU reductions + fused maps) | `validate_barracuda_tensor_f64` | 35 | **PASS** |
| `shaders::quantized` (dequant Q4/Q8, GEMV) | `validate_barracuda_quantized` | 15 | **PASS** |
| `linalg::{svd\_\*, lu\_inverse, gen\_eigh}` | `validate_barracuda_linalg_ext` | 17 | **PASS** |
| **ML Inference** (MLP + Transformer end-to-end) | `validate_barracuda_ml_inference` | 13 | **PASS** |
| **FFT** (Fft1D/Ifft1D + Fft1DF64 + Rfft) | `validate_barracuda_fft` | 24 | **PASS** |
| **LogSumExp** (numerically stable log-probability summation) | `validate_barracuda_logsumexp` | 5 | **PASS** |
| **Total** | **12 binaries** | **272** | **ALL PASS** |

### Tier B — Adapt (needs training infrastructure)

| Python Module | Rust Module | WGSL Shader | Pipeline Stage | Blocker |
|---------------|-------------|-------------|----------------|---------|
| `surrogate/` MLP forward | `surrogate::mlp_forward` (stub) | `gemm_f64.wgsl` + `nn::ReLU` | Inference | BarraCUDA `nn::Layer` |
| `surrogate/` MLP training | `surrogate::mlp_train` (stub) | `gemm_f64.wgsl` + `nn::Optimizer::Adam` | Training | BarraCUDA autograd |
| `sequence/` LSTM cell | `sequence::lstm_cell` | `lstm_cell.wgsl` | Inference | **VALIDATED** (26 checks) |
| `sequence/` GRU cell | `sequence::gru_cell` | `gru_cell.wgsl` | Inference | **VALIDATED** (26 checks) |
| `pinn/` autograd | — | `fd_gradient_f64.wgsl` | Training | Reverse-mode AD in BarraCUDA |
| `lenet/` Conv2d | `Tensor::conv2d()` | `conv2d.wgsl` | Inference | **WIRED** — `validate_barracuda_gpu_lenet` (Session 44) |
| `lenet/` MaxPool | `Tensor::maxpool2d()` | `max_pool2d.wgsl` | Inference | **WIRED** — `validate_barracuda_gpu_lenet` (Session 44) |
| `deeponet/` Branch-Trunk | — | `gemm_f64.wgsl` × 2 | Inference | Compose from MLP |
| `quantized/` INT8 GEMV | `quantized::gemv_q8` | `gemv_q8.wgsl` | Deployment | **VALIDATED** (26 checks) |
| `quantized/` INT4 GEMV | `quantized::gemv_q4` | `gemv_q4.wgsl` | Deployment | **VALIDATED** (26 checks) |
| `transfer/` freeze+finetune | — | selective gradient | Training | BarraCUDA param freeze |

### Tier C — New (GPU-specific, no Python equivalent)

| Capability | WGSL Shader | Pipeline Stage | Blocker | ToadStool Leverage |
|------------|-------------|----------------|---------|-------------------|
| Flash attention | `flash_attention.wgsl` | Inference | Algorithm implementation | — |
| Fused LayerNorm+GELU | fused kernel | Inference | Kernel fusion framework | `TensorSession` extension |
| Batched GEMM | `gemm_f64.wgsl` (batched) | Training / EA | Batch dispatch | `KernelRouter` |
| Population fitness eval | `batch_gemv.wgsl` | Evolution (Dolson 011–015) | GA/ES framework | `StatefulPipeline` for gen loop |
| HMM forward (fused) | `hmm_forward_log.wgsl` | Genomics (Liu 016–018) | Log-domain matmul chain | `StatefulPipeline` for T steps |
| Pairwise distance | `pairwise_distance.wgsl` | Alignment (Liu 017) | One thread per pair | — (embarrassingly parallel) |
| GPU ODE integrator (RK4) | `rk4_batch.wgsl` | Biology (Waters 020–021) | Elementwise RHS | `StatefulPipeline` + `ReduceScalarPipeline` |
| Spatial stencil | `stencil_1d.wgsl` | Cooperation (Waters 019) | Neighbor averaging | — (reuse conv1d) |
| Tridiag eigensolver | `tridiag_eigh.wgsl` | Spectral (Kachkovskiy 022–023) | Bisection + inverse iteration | NAK-optimized eigh available |
| GPU PRNG (Xoshiro256**) | `xoshiro256ss.wgsl` | All stochastic algorithms | `jump()` for independent streams | — |
| Gillespie SSA | GPU PRNG + exp sampling | Biology (Waters) | New primitive | `StatefulPipeline` |

### ToadStool Infrastructure Available for GPU Promotion

ToadStool (reviewed `6ee71f07`, Feb 23, 2026 — all shortcomings through S-13 fixed)
provides infrastructure directly usable for Phase 0++ GPU promotion:

| Capability | API | Use Case |
|------------|-----|----------|
| `StatefulPipeline` | `staging::StatefulPipeline::run_iterations(chain, buf, n)` | EA gen loops, ODE integration, HMM chains — GPU-resident state, scalar-only readback |
| `ReduceScalarPipeline` | `pipeline::ReduceScalarPipeline::sum_f64(buf)` | Fitness aggregation, log-likelihood, convergence checks — 8 bytes readback |
| `KernelRouter` | `device::KernelRouter::route(workload)` | 4-tier matmul selection, device-aware kernel dispatch |
| `GpuDriverProfile` | `device::capabilities::GpuDriverProfile` | Per-driver shader specialization (NAK workarounds vs proprietary) |
| NAK eigensolve | `batched_eigh_nak_optimized_f64.wgsl` | Drop-in 2–4× faster eigensolve for Anderson localization (Paper 023) |

---

## GPU Promotion Priority

Based on cross-paper primitive usage and BarraCUDA impact:

| Priority | Primitive | Papers Served | Effort | Impact |
|----------|-----------|---------------|--------|--------|
| 1 | Batch GEMM/GEMV | 011–015 (5 papers) | Medium | Parallel population eval |
| 2 | Pairwise distance kernel | 017 | Low | Simple, high-value |
| 3 | GPU-parallel RK4 | 020–021 | Medium | Multi-system ODE |
| 4 | Fused HMM forward | 016–018 | Medium | Log-domain matmul chain |
| 5 | Tridiagonal eigensolver | 022–023 | High | Specialized for structure |
| 6 | Spatial stencil | 019 | Low | Reuse conv1d |
| 7 | GPU PRNG | All stochastic | Medium | Foundation for parallel EA |
| 8 | Binary matrix reduction | 024 | Low | PA matrix row/col sums |
| 9 | Parallel pairwise FST | 025 | Medium | ANOVA decomposition per-locus |

---

## Promotion Checklist

For each Rust module → GPU promotion:

- [ ] Python baseline passes with documented provenance
- [ ] Rust implementation matches Python to documented tolerance
- [ ] WGSL shader exists in BarraCUDA or is planned
- [ ] Validation binary follows hotSpring pattern (exit 0/1)
- [ ] Performance meets or exceeds Python baseline
- [ ] Test coverage ≥ 90% (analytical + round-trip + determinism)

---

## Current Status (February 22, 2026)

| Phase | Status | Coverage |
|-------|--------|----------|
| Phase 0 (Python baselines) | **206/206 PASS** | 25 experiments, drift detection via `control/check_drift.sh` |
| Phase 1a (neuralSpring Rust) | **264 lib + 9 integration PASS** | 31 modules (+3 evolved), 264 unit tests, 9 integration tests, 119 validation binaries |
| Phase 1b (BarraCUDA) | **272/272 PASS** | 12 validation binaries, incl. Tensor/WGSL (90), tensor_f64 (35), ml_inference (13), FFT (24), LogSumExp (5) |
| Phase 1c (Fused pipeline) | **46–78× speedup** | Single-encoder dispatch, GPU-resident ops |
| Phase 2 (BarraCUDA CPU ports) | **203/203 PASS** | 24/25 papers validated (96% bC coverage) |
| Phase 3a (FFT validation) | **24/24 PASS** | f32 Fft1D/Ifft1D + f64 Fft1DF64 + Rfft |
| Phase 3b (GPU streaming) | **COMPLETE** | `StatefulPipeline` validated (10/10 PASS) |
| Phase 3c (Shader evolution) | **COMPLETE** | 12 WGSL shaders (+4 new: pairwise_l2, multi_obj_fitness, swarm_nn_forward, hill_gate) |
| Phase 3d (Pure GPU + cross-dispatch) | **COMPLETE** | 58/58 PASS (SP 10 + chain 7 + xd 8 + xd-genomics 8 + xd-extended 12 + xd-phase4e 13) |
| Phase 4a (Performance benchmarks) | **COMPLETE** | 7 kernels, 71.8× overall speedup vs single-thread NumPy |
| Phase 4b (Pure GPU end-to-end pipelines) | **COMPLETE** | 7 pipelines, 32/32 PASS (+modes, directed, signal) |
| Phase 4c (GPU kernel benchmarks + PRNG) | **COMPLETE** | Crossover mapping (GPU wins at >1.5ms CPU work) + 5/5 PRNG PASS |
| Phase 4d (ToadStool S-12 + S-03b) | **COMPLETE** | eigh LAPACK (9/9 PASS) + head_split/head_concat (10/10 PASS) |
| Phase 4e (PINN/DeepONet + new GPU domains) | **COMPLETE** | PINN 16+14, DeepONet 17+9, GPU modes 15, directed 6, swarm 9, signal 9 |
| Phase 5a (BarraCUDA GPU Tensor) | **COMPLETE** | 14/14 PASS (spectral 8, eco 6) |
| Phase 4 (Sovereign pipeline) | **Active** | Cross-spring integration |

### Phase 3c — metalForge Shader Evolution

Following the hotSpring pattern (evolve → validate → handoff → absorb → retire),
twelve WGSL shaders (plus PRNG) are under development in `metalForge/shaders/`
with Rust orchestration in `src/evolved/`:

| Shader | Validation Binary | Papers | Status |
|--------|-------------------|--------|--------|
| `hmm_forward_log.wgsl` | `validate_gpu_hmm_forward` | 016–018 | **Compiled + validation binary** |
| `batch_fitness_eval.wgsl` | `validate_gpu_batch_fitness` | 011–015 | **Compiled + validation binary** |
| `rk4_parallel.wgsl` | `validate_gpu_rk4` | 020–021 | **Compiled + validation binary** |
| `pairwise_jaccard.wgsl` | `validate_gpu_pangenome` | 024 | **Compiled + validation binary** |
| `locus_variance.wgsl` | `validate_gpu_meta_pop` | 025 | **Compiled + validation binary** |
| `spatial_payoff.wgsl` | `validate_gpu_game_theory` | 019 | **Compiled + validation binary** |
| `batch_ipr.wgsl` | `validate_gpu_anderson` | 022-023 | **Compiled + validation binary** |
| `pairwise_hamming.wgsl` | `validate_gpu_sate` | 017 | **Compiled + validation binary** |
| `pairwise_l2.wgsl` | `validate_gpu_modes` | 012 | **15/15 PASS** (Feb 21) |
| `multi_obj_fitness.wgsl` | `validate_gpu_directed` | 014 | **6/6 PASS** (Feb 21) |
| `swarm_nn_forward.wgsl` | `validate_gpu_swarm` | 015 | **9/9 PASS** (Feb 21) |
| `hill_gate.wgsl` | `validate_gpu_signal` | 021 | **9/9 PASS** (Feb 21) |

See `metalForge/shaders/ABSORPTION_TRACKER.md` for the full lifecycle tracker.

### Phase 3d — Pure GPU Workload + Cross-Dispatch

| Validation Binary | BarraCUDA API | Checks | Status |
|-------------------|--------------|--------|--------|
| `validate_gpu_stateful_pipeline` | `StatefulPipeline` | 10 | **10/10 PASS** |
| `validate_gpu_pure_workload` | Multi-kernel chain | 7 | **7/7 PASS** |
| `validate_cross_dispatch` | `DispatchConfig` | 8 | **8/8 PASS** |
| `validate_cross_dispatch_genomics` | `DispatchConfig` + Jaccard/variance | 8 | **8/8 PASS** |
| `validate_cross_dispatch_phase4e` | `DispatchConfig` + pairwise_l2/multi_obj/swarm_nn/hill_gate | 13 | **13/13 PASS** |

### Phase 4a — Performance Benchmarks

The `bench_phase0pp_kernels` binary compares Rust pure math to Python NumPy at identical problem
sizes. Seven kernels, one per control script:

| Kernel | Paper | Rust µs | Python µs | Speedup |
|--------|-------|---------|-----------|---------|
| HMM forward (3×5000) | 016-018 | 330.0 | 12007.6 | 36.4× |
| Replicator dynamics (10k steps) | 019 | 150.0 | 34937.4 | 232.9× |
| Commutator ‖[A,B]‖_F (64×64) | 022 | 334.6 | 23.3 | 0.1× |
| NK fitness (N=10,K=2, 1000 genotypes) | 011 | 17.9 | 14087.2 | 787.1× |
| Pairwise Hamming (20×500) | 017 | 34.3 | 408.3 | 11.9× |
| Jaccard distance (30×500) | 024 | 142.3 | 2045.4 | 14.4× |
| RK4 GRN ODE (2000 steps) | 020-021 | 218.6 | 24659.8 | 112.8× |
| **TOTAL** | | **1227.8** | **88169.0** | **71.8×** |

Rust pure math is 71.8× faster than single-thread NumPy overall. GEMM-heavy operations
(commutator: 0.1×) show why GPU WGSL acceleration via BarraCUDA matters.

### Phase 4b — Pure GPU End-to-End Pipelines

Seven pure GPU pipelines, each kernel-chain → mean_reduce, GPU-resident with scalar-only readback.
Phase 3d+4b combined: **77/77 PASS** for pure GPU + cross-dispatch.

| Validation Binary | Pipeline | Papers | Checks | Status |
|-------------------|----------|--------|--------|--------|
| `validate_gpu_pipeline_hmm` | HMM forward → mean_reduce | 016–018 | 5 | **5/5 PASS** |
| `validate_gpu_pipeline_ecology` | spatial_payoff → mean_reduce | 019 | 5 | **5/5 PASS** |
| `validate_gpu_pipeline_spectral` | batch_ipr → mean_reduce | 022–023 | 5 | **5/5 PASS** |
| `validate_gpu_pipeline_genomics` | pairwise_jaccard → mean_reduce | 024 | 5 | **5/5 PASS** |
| `validate_gpu_pipeline_modes` | pairwise_l2 → mean_reduce | 012 | 4 | **4/4 PASS** (Feb 21) |
| `validate_gpu_pipeline_directed` | multi_obj_fitness → mean_reduce | 014 | 4 | **4/4 PASS** (Feb 21) |
| `validate_gpu_pipeline_signal` | hill_gate → mean_reduce | 021 | 4 | **4/4 PASS** (Feb 21) |

### Phase 4c — GPU WGSL Kernel Benchmarks + GPU PRNG

**bench_gpu_kernels**: Times WGSL shaders on RTX 4070 vs Rust CPU at small (paper-scale)
and large (production-scale) sizes, revealing the dispatch crossover point.

| Kernel | Scale | GPU µs | Rust CPU µs | Winner |
|--------|-------|--------|-------------|--------|
| Hamming | Small (20×500) | 1,589 | 34 | CPU 46× |
| Hamming | **Large (200×1000)** | **1,675** | **7,089** | **GPU 4.2×** |
| Jaccard | Small (30×500) | 1,659 | 142 | CPU 12× |
| Jaccard | **Large (100×2000)** | **1,464** | **8,246** | **GPU 5.6×** |

**Crossover**: GPU dispatch overhead ~1.5ms fixed. CPU wins below; GPU wins above.
This is exactly what `barracuda::dispatch` routes and what metalForge documents.

**validate_gpu_prng** (5/5 PASS): Xoshiro128** PRNG shader (`metalForge/shaders/xoshiro128ss.wgsl`).
Validates uniformity, range, determinism, independence, multi-call state advancement.
Exported as `rng::WGSL_XOSHIRO128SS` for ToadStool absorption.

| Validation Binary | Shader | Checks | Status |
|-------------------|--------|--------|--------|
| `validate_gpu_prng` | `xoshiro128ss.wgsl` | 5 | **5/5 PASS** |

### Phase 4d — ToadStool Issue Resolution (S-12 + S-03b)

| Shortcoming | Description | Checks | Status |
|-------------|-------------|--------|--------|
| **S-12** | Householder+QR eigensolver — LAPACK-level accuracy at all matrix sizes | 9/9 | **PASS** |
| **S-03b** | GPU head_split/head_concat WGSL shaders | 10/10 | **PASS** |

**New files:** `src/eigh.rs`, `metalForge/shaders/head_split.wgsl`, `metalForge/shaders/head_concat.wgsl`  
**New binaries:** `validate_eigh_accuracy`, `validate_mha_gpu`

### Phase 4e — PINN/DeepONet + New GPU Domain Validators (Feb 21, 2026)

Two new Rust domain modules (`pinn.rs`, `deeponet.rs`) implementing the pure math
components of Studies 001–002 (Raissi PINN, Lu DeepONet). BarraCUDA CPU validation
proves Tensor API (matmul, tanh, dot) reproduces hand-rolled Rust. Four new GPU
domain shaders expand coverage to Papers 012, 014, 015, 021.

| Validation Binary | Domain | Checks | Status |
|-------------------|--------|--------|--------|
| `validate_pinn` | Burgers' PDE: Cole-Hopf, MLP forward, FD residual | 16 | **16/16 PASS** |
| `validate_deeponet` | Antiderivative operator: polynomials, branch-trunk | 17 | **17/17 PASS** |
| `validate_barracuda_pinn` | BarraCUDA Tensor: MLP matmul+tanh, Cole-Hopf cross-val | 14 | **14/14 PASS** (RTX 4070) |
| `validate_barracuda_deeponet` | BarraCUDA Tensor: branch-trunk dot, antideriv cross-val | 9 | **9/9 PASS** (RTX 4070) |
| `validate_gpu_modes` | WGSL `pairwise_l2` — novelty metric (Paper 012) | 15 | **15/15 PASS** (RTX 4070) |
| `validate_gpu_directed` | WGSL `multi_obj_fitness` — chunk mean+std (Paper 014) | 6 | **6/6 PASS** (RTX 4070) |
| `validate_gpu_swarm` | WGSL `swarm_nn_forward` — batch MLP inference (Paper 015) | 9 | **9/9 PASS** (RTX 4070) |
| `validate_gpu_signal` | WGSL `hill_gate` — 2D Hill function grid (Paper 021) | 9 | **9/9 PASS** (RTX 4070) |

### Phase 5a — BarraCUDA GPU Tensor Validation

| Validation Binary | Domain | Checks | Status |
|-------------------|--------|--------|--------|
| `validate_barracuda_gpu_spectral` | GPU Tensor matmul for commutator (Paper 022) | 8 | **8/8 PASS** |
| `validate_barracuda_gpu_eco` | GPU Tensor matmul for eco dynamics (Paper 013) | 6 | **6/6 PASS** |

GPU-ready layout evolution completed for `anderson_localization.rs` (flat N×N Hamiltonians),
`directed_evolution.rs` (flat pop×genome), and `sate_alignment.rs` (flat n×len sequences).

### ToadStool Shortcoming Status

**Reviewed:** ToadStool commit `6ee71f07` (Session 42 + bug fixes, Feb 23, 2026).
**Result:** **All shortcomings through S-13 FIXED/ABSORBED.** Key absorption
commit: `fbedd222` (`TensorSession` ML ops). Validation binary
`validate_barracuda_tensor` rewired from evolved ops to native BarraCUDA
APIs — 90/90 PASS. `src/evolved/` workarounds documented for retirement.
See `specs/TOADSTOOL_HANDOFF.md` for full details and migration plan.

---

## Deep Evolution: GPU-Ready Layout Status (February 21, 2026)

Library modules that have been evolved to flat row-major layouts for direct
GPU buffer upload:

| Module | Layout | GPU Ready | Absorption Target |
|--------|--------|-----------|-------------------|
| `hmm.rs` | Flat `Vec<f64>` (N×N, N×M, T×N) | **Yes** — `Hmm::from_flat()` | `barracuda::ops::hmm` |
| `spectral_commutativity.rs` | Flat `Vec<f64>` (N×N) | **Yes** — `mat_mul(a, b, n)` | `barracuda::ops::matmul` f64 |
| `primitives.rs` | Centralized math + constants | **Yes** — no layout barrier | `barracuda::numerical`, `barracuda::stats` |
| `anderson_localization.rs` | Flat `Vec<f64>` (N×N) | **Yes** — flat Hamiltonians + eigenvectors | `barracuda::linalg::eigh_gpu` |
| `directed_evolution.rs` | Flat `Vec<f64>` (pop×genome, pop×obj) | **Yes** — flat population + fitness | `barracuda::ops::batch_gemm` |
| `sate_alignment.rs` | Flat `Vec<u8>` (n×len), `Vec<f64>` (n×n) | **Yes** — flat sequences + distance matrix | `barracuda::ops::pairwise_distance` |
| `pinn.rs` | Scalar + grid ops | **Yes** — Cole-Hopf, MLP, PDE residual | `barracuda::tensor` matmul+tanh |
| `deeponet.rs` | Scalar + polynomial | **Yes** — branch-trunk dot product | `barracuda::tensor` matmul |

### New Module: `primitives.rs`

Consolidates 6× Shannon entropy, 3× Hill kinetics, 2× sigmoid, 2× RK4
across 8 library modules. Numerical constants `LOG_GUARD`, `HILL_EPS`,
`DIVISION_GUARD` replace all module-local magic numbers. Generic RK4
uses `const N: usize` + `FnMut` closure.

### Validation Robustness: `require!` Macro

All validation binaries use `require!(h, result, label)` for GPU operations.
No `.expect()` calls remain in validation code. Enables graceful CI runs
on machines without GPU adapters.
