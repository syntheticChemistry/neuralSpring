# neuralSpring — Pure GPU Roadmap

**Date**: February 23, 2026 (Sessions 40–48)
**Goal**: All math runs on GPU. Even a Raspberry Pi is a science platform.
**Philosophy**: Prove math is entirely portable on GPU first, then reverse-engineer
for CPU efficiency and older hardware. Mixed workloads come after pure GPU validation.

---

## Why Pure GPU

1. **Universal portability**: WGSL compiles to Vulkan (desktop), Metal (Apple), WebGPU
   (browser), and llvmpipe (CPU fallback). One source, every platform.
2. **Scale**: GPU parallelism enables population-scale science (10k genomes, 1M particles,
   4k-state HMMs) that is infeasible on CPU within interactive timescales.
3. **Proof of concept**: If neuralSpring runs purely on GPU, it runs on any device with
   a Vulkan-capable GPU — from TITAN V to Raspberry Pi 5 (VideoCore VII).
4. **Reverse pipeline**: GPU-validated math → optimize for CPU using BLAS techniques →
   optimize for older GPU → mixed hardware as an efficiency choice, not a requirement.

---

## Current State: What's Already on GPU

### Fully GPU-resident (21 WGSL shaders + BarraCUDA ops)

| Domain | WGSL Shader / bC Op | Pipeline | Status |
|--------|---------------------|----------|--------|
| Batch fitness | `batch_fitness_eval.wgsl` | fitness → mean_reduce | **GPU** |
| Pairwise L2 | `pairwise_l2.wgsl` | distance → mean_reduce | **GPU** |
| Pairwise Hamming | `pairwise_hamming.wgsl` | distance → mean_reduce | **GPU** |
| Pairwise Jaccard | `pairwise_jaccard.wgsl` | distance → mean_reduce | **GPU** |
| Multi-obj fitness | `multi_obj_fitness.wgsl` | fitness → mean_reduce | **GPU** |
| Spatial payoff | `spatial_payoff.wgsl` | payoff → mean_reduce | **GPU** |
| Locus variance | `locus_variance.wgsl` | variance → mean_reduce | **GPU** |
| Batch IPR | `batch_ipr.wgsl` | ipr → mean_reduce | **GPU** |
| Hill AND gate | `hill_gate.wgsl` | elementwise | **GPU** |
| Swarm NN | `swarm_nn_forward.wgsl` | forward → scores | **GPU** |
| HMM forward | `hmm_forward_log.wgsl` | forward chain | **GPU** |
| RK4 parallel | `rk4_parallel.wgsl` | ODE integration | **GPU** |
| RK45 adaptive | `rk45_adaptive.wgsl` | Dormand-Prince | **GPU** |
| Stencil cooperation | `stencil_cooperation.wgsl` | Fermi imitation | **GPU** |
| Logsumexp | `logsumexp_reduce.wgsl` | batched reduction | **GPU** |
| Wright-Fisher | `wright_fisher_step.wgsl` | stochastic drift | **GPU** |
| PRNG (xoshiro128**) | `xoshiro128ss.wgsl` | random seeds | **GPU** |
| Mean reduce | `mean_reduce.wgsl` | scalar reduction | **GPU** |
| Head split/concat | `head_split.wgsl` / `head_concat.wgsl` | MHA reshape | **GPU** |
| MatMul/Transpose | `Tensor::matmul()` / `Tensor::transpose()` | GEMM | **GPU** |
| Activations | `Tensor::relu/gelu/sigmoid/tanh/softmax()` | elementwise | **GPU** |
| Conv2d/MaxPool | `Tensor::conv2d()` / `Tensor::maxpool2d()` | CNN | **GPU** |
| FFT | `Fft1D` / `Ifft1D` | spectral | **GPU** |

### GPU Pipeline Chains (single CommandEncoder, zero CPU round-trips)

| Pipeline | Shaders Chained | Domain |
|----------|-----------------|--------|
| HMM → reduce | hmm_forward_log × T → mean_reduce | Phylogenetics |
| Fitness → reduce | batch_fitness → mean_reduce | Ecology |
| Spectral → reduce | batch_ipr → mean_reduce | Anderson |
| Genomics → reduce | pairwise_jaccard → mean_reduce | Pangenome |
| WF → reduce | wright_fisher_step → mean_reduce | Pop genetics |
| Gillespie → reduce | GillespieGpu → mean_reduce | Stochastic |

---

## CPU-Bound Operations: Promotion Plan

### Phase A — COMPLETE (Session 45)

Phase A (wire existing ops) is now **complete**. The `gpu_ops` module provides GPU
paths for: matmul, transpose, frobenius_norm, commutator, distance_to_normal,
softmax, boltzmann, gelu, l2_distance, mean, sum, max, variance, shannon_entropy,
pearson_correlation, chi_squared, kl_divergence, hmm_forward_step, neural_forward.
The `gpu_dispatch` module provides runtime capability-based GPU/CPU dispatch.
`validate_gpu_promotion` validates all 27 promotions (27/27 PASS, RTX 4070 + TITAN V NVK).

### Tier 1 — High Impact, GPU Path Exists (wire it) — DONE

These operations had BarraCUDA GPU equivalents; production now routes through `gpu_dispatch`.

| Module | CPU Operation | GPU Replacement | Effort | Papers |
|--------|---------------|-----------------|--------|--------|
| `spectral_commutativity.rs` | `mat_mul()` (triple-nested loop) | `Tensor::matmul()` | **Low** | 022 |
| `spectral_commutativity.rs` | `transpose()` (double loop) | `Tensor::transpose()` | **Low** | 022 |
| `spectral_commutativity.rs` | `frobenius_norm()` (iter sum) | `NormReduceF64` | **Low** | 022 |
| `counterdiabatic.rs` | `boltzmann_distribution()` (max+sum) | `Tensor::softmax()` or `logsumexp.wgsl` | **Low** | 011 |
| `eco_dynamics.rs` | `batch_fitness()` (Hamming+max) | `BatchFitnessGpu` | **Low** | 013 |
| `modes.rs` | `l2_distance()` (iter sum) | `PairwiseL2Gpu` | **Low** | 012 |
| `modes.rs` | `novelty_metric()` (L2 batch) | `PairwiseL2Gpu` → reduce | **Low** | 012 |
| `directed_evolution.rs` | `lexicase_selection()` (fold/max) | Batch GEMM + reduce_max | **Medium** | 014 |
| `swarm_robotics.rs` | `neural_forward()` (matmul chain) | `SwarmNnGpu` | **Low** | 015 |
| `game_theory.rs` | `replicator_dynamics()` (ODE loop) | `rk4_parallel.wgsl` | **Medium** | 019 |
| `transformer.rs` | `softmax()` (fold+sum) | `Tensor::softmax()` | **Low** | Exp 002 |

### Tier 2 — Medium Impact, GPU Path Exists (adapt it) — PARTIAL (Session 46)

Phase B (Session 46) promoted the HMM backward/Viterbi, meta-population statistics,
game theory replicator dynamics, and Hill activation to GPU. New `gpu_ops` functions:
`hmm_backward_step_gpu`, `hmm_viterbi_step_gpu`, `allele_frequencies_gpu`,
`nucleotide_diversity_gpu`, `matrix_correlation_gpu`, `geographic_distance_matrix_gpu`,
`thermal_diversity_correlation_gpu`, `inter_population_af_variance_gpu`,
`replicator_step_gpu`. Also fixed `hill_activation_batch_gpu` from pseudo-GPU to
genuine GPU computation via log→scale→exp→div pipeline.
`validate_gpu_phase_b` (20/20 PASS on RTX 4070 + TITAN V NVK).

| Module | CPU Operation | GPU Replacement | Effort | Status |
|--------|---------------|-----------------|--------|--------|
| `hmm.rs` | `backward()` | `hmm_backward_step_gpu` (GEMV per step) | **Medium** | **GPU (S46)** |
| `hmm.rs` | `viterbi()` | `hmm_viterbi_step_gpu` (broadcast+max+argmax) | **Medium** | **GPU (S46)** |
| `signal_integration.rs` | `two_input_hill()` batch | `hill_activation_batch_gpu` (genuine) | **Low** | **GPU (S46)** |
| `regulatory_network.rs` | Hill activation/repression | `hill_activation_batch_gpu` | **Low** | **GPU (S46)** |
| `meta_population.rs` | `allele_frequencies()` | `allele_frequencies_gpu` (column-sum) | **Low** | **GPU (S46)** |
| `meta_population.rs` | `nucleotide_diversity()` | `nucleotide_diversity_gpu` (elementwise+mean) | **Low** | **GPU (S46)** |
| `meta_population.rs` | `matrix_correlation()` | `matrix_correlation_gpu` (Pearson) | **Medium** | **GPU (S46)** |
| `meta_population.rs` | `geographic_distance_matrix()` | `geographic_distance_matrix_gpu` (L2) | **Low** | **GPU (S46)** |
| `meta_population.rs` | `thermal_diversity_correlation()` | `thermal_diversity_correlation_gpu` | **Low** | **GPU (S46)** |
| `meta_population.rs` | `inter_population_af_variance()` | `inter_population_af_variance_gpu` | **Medium** | **GPU (S46)** |
| `game_theory.rs` | `replicator_dynamics()` step | `replicator_step_gpu` (matmul) | **Low** | **GPU (S46)** |
| `signal_integration.rs` | `integrate_ode()` (full RK4 loop) | Batch ODE on GPU (needs PRNG) | **Medium** | Pending |
| `regulatory_network.rs` | `integrate_grn()` (full RK4 loop) | Batch ODE on GPU | **Medium** | Pending |
| `meta_population.rs` | `global_fst()` / `pairwise_fst()` | Variance decomposition shader | **Medium** | Pending |
| `introgression.rs` | `detect_introgression()` → HMM chain | `HmmBatchForwardF64` + Viterbi | **Medium** | Pending |
| `pangenome_selection.rs` | `spectrum_chi_squared()` | `spectrum_chi_squared_gpu` | **Medium** | **GPU (S47)** |
| `pangenome_selection.rs` | `selection_coefficient()` | `selection_coefficient_gpu` | **Low** | **GPU (S47)** |

### Tier 3 — Eigensolvers — COMPLETE (Session 47)

| Module | CPU Operation | GPU Replacement | Status |
|--------|---------------|-----------------|--------|
| `anderson_localization.rs` | `jacobi_eigh()` | `BatchedEighGpu` / NAK eigensolve | **GPU (S47)** |
| `anderson_localization.rs` | `disorder_sweep()` | `disorder_sweep_gpu` (batch eigensolve + mean IPR) | **GPU (S47)** |
| `eigh.rs` | `eigh_householder_qr()` | `eigh_gpu` via BatchedEighGpu (n≤32) | **GPU (S47)** |

### Tier 4 — New Shaders Needed

| Module | CPU Operation | New Shader | Effort | Papers |
|--------|---------------|------------|--------|--------|
| `hmm.rs` | `backward()` | `hmm_backward_log.wgsl` | **Medium** | 016-018 |
| `hmm.rs` | `viterbi()` | `hmm_viterbi.wgsl` (max-reduce) | **Medium** | 016, 018 |
| `pangenome_selection.rs` | `chi_squared_test()` | `chi_squared_reduce.wgsl` | **Medium** | 024 |
| `meta_population.rs` | `mantel_test()` | `matrix_correlation.wgsl` + permutation | **High** | 025 |
| `modes.rs` | `complexity_metric()` regression | `linear_regression.wgsl` | **Medium** | 012 |
| `primitives.rs` | `shannon_entropy()` | `FusedMapReduceF64::shannon_entropy` (exists) | **Low** | multiple |

### Tier 5 — Stays on CPU (by design)

| Category | Why | Impact |
|----------|-----|--------|
| Validator reference implementations | CPU reference for parity checking | Zero — not production |
| `ValidationHarness` diff checks | Orchestration, not math | Zero |
| RNG seeding / parameter setup | One-time, negligible cost | Zero |
| Python subprocess benchmarks | Infrastructure, not science | Zero |
| Tolerance constants / registry | Metadata, not computation | Zero |

---

## hotSpring Streaming Patterns to Adopt

hotSpring eliminates CPU round-trips using these ToadStool patterns:

### Pattern 1: Encoder Batching
```rust
let mut encoder = gpu.begin_encoder("md_step");
for _ in 0..batch_size {
    gpu.encode_pass(&mut encoder, &force_pipeline, &force_bg, wg);
    gpu.encode_pass(&mut encoder, &kick_pipeline, &kick_bg, wg);
}
gpu.submit_encoder(encoder);
```
**neuralSpring equivalent**: Chain RK4 steps, HMM timesteps, or fitness evaluations
in a single encoder. Already demonstrated for WF→reduce and Gillespie→reduce pipelines.

### Pattern 2: GPU→GPU Copy in Encoder
```rust
encoder.copy_buffer_to_buffer(vel_buf, 0, &ring.flat_buf, offset, size);
```
**neuralSpring equivalent**: Copy HMM alpha matrix to next-timestep input buffer
without CPU readback. Copy fitness results to selection buffer without readback.

### Pattern 3: Scalar-Only Readback via ReduceScalarPipeline
```rust
let total_ke = reducer.sum_f64(&ke_buf)?;  // N values → 8 bytes
```
**neuralSpring equivalent**: Already used in pipeline validators. Extend to all
production paths — read back only the final statistic (mean fitness, log-likelihood,
IPR, etc.), not intermediate buffers.

### Pattern 4: GPU-Resident Ring Buffer
hotSpring stores velocity snapshots on GPU via `GpuVelocityRing` — never read back,
only consumed by subsequent GPU shaders.
**neuralSpring equivalent**: Store HMM alpha/beta matrices, population state, and
eigenvector data on GPU between pipeline stages.

---

## Promotion Strategy (ordered by impact)

### Phase A — Wire existing GPU ops into production modules (Tier 1) — COMPLETE (Session 45)

**Completed**: Session 45. `gpu_ops` and `gpu_dispatch` modules route all 27 CPU-bound
ops through GPU when capability is available. `validate_gpu_promotion` (27/27 PASS)
confirms correctness on RTX 4070 and TITAN V NVK.

### Phase B — Adapt HMM, meta-pop, game theory to GPU (Tier 2) — IN PROGRESS (Session 46)

Session 46 completed the first wave of Phase B:
- **HMM backward**: GPU GEMV per step via `hmm_backward_step_gpu`
- **HMM Viterbi**: GPU broadcast + max_dim + CPU argmax via `hmm_viterbi_step_gpu`
- **Meta-population**: 6 functions promoted (allele_frequencies, nucleotide_diversity,
  matrix_correlation, geographic_distances, thermal_diversity_correlation,
  inter_population_af_variance)
- **Game theory**: `replicator_step_gpu` — 2×2 payoff GEMV on GPU
- **Hill activation**: `hill_activation_batch_gpu` now genuinely GPU-computed
  (log→scale→exp→div pipeline, not pseudo-GPU)
- **Validator**: `validate_gpu_phase_b` — 20/20 PASS on RTX 4070 + TITAN V NVK

**Remaining Phase B work** (future sessions):
- Full ODE loops (integrate_ode, integrate_grn) → encoder batching with GPU PRNG
- FST variance decomposition → custom shader
- Introgression HMM chain → compose forward + Viterbi steps

### Phase C — New shaders for remaining gaps (Tier 4)

**Estimated effort**: 2-3 sessions. Write `hmm_backward_log.wgsl`,
`hmm_viterbi.wgsl`, `chi_squared_reduce.wgsl`, `matrix_correlation.wgsl`.

Target: Complete the HMM GPU suite (forward + backward + Viterbi), chi-squared
hypothesis testing on GPU, Mantel test matrix correlation on GPU.

### Phase D — Batch eigensolve on GPU (Tier 3)

**Estimated effort**: 1-2 sessions. Wire `BatchedEighGpu` for Anderson localization
disorder sweep. This is the most compute-intensive CPU path.

Target: `disorder_sweep()` dispatches all W values as a batch eigensolve on GPU.

---

## Raspberry Pi Viability

VideoCore VII (Raspberry Pi 5) supports Vulkan 1.2 via the `v3dv` driver.
WGSL → SPIR-V → Vulkan → VideoCore VII is the same path as RTX 4070 / TITAN V.

| Constraint | RPi 5 | Impact |
|------------|-------|--------|
| VRAM | 8 GB shared | Limits population size; 10k genomes × 100 loci = 8 MB (fine) |
| Compute units | 12 QPUs | ~100× slower than RTX 4070; still faster than Python on ARM CPU |
| f64 support | Likely absent | Use f32 path (already validated — all validators have f32 GPU checks) |
| Workgroup size | May be < 256 | `Gpu::dispatch_1d()` already handles via `GpuCapabilities` |
| Vulkan features | 1.2 (v3dv) | Subset of 1.3 — may need feature gating for some shaders |

**Key insight**: The `GpuCapabilities` infrastructure (Session 40) already validates
workgroup compatibility at runtime. Shaders that need `@workgroup_size(256)` will
fail gracefully on hardware with smaller limits, and `dispatch_1d()` clamps workgroup
counts to hardware limits.

---

## Success Criteria

1. **All 25 papers' production math runs on GPU** (no CPU math in production path)
2. **Scalar-only readback** for all validation checks (no intermediate buffer reads)
3. **Bit-identical results** on RPi 5 (VideoCore VII) vs RTX 4070 vs TITAN V
4. **No CPU fallback in dispatch heuristics** for pure GPU mode
5. **hotSpring streaming patterns** adopted for iterative workloads (ODE, HMM, MD)

---

## Coverage Gap Summary (updated Session 66)

| Category | Currently GPU | Total | Gap | Priority |
|----------|-------------|-------|-----|----------|
| MatMul / GEMM | 17/17 modules | 17 | None | **Done** |
| Reductions (sum/mean/max) | 16/16 modules | 16 | None | **Done** |
| ODE integration (RK4) | 3/5 modules | 5 | Full loop batching (parallel trajectories) | **P2** |
| HMM (fwd/bwd/Viterbi + chains) | **5/5 ops** | 5 | None — chains compose step ops (S66) | **Done** |
| Eigensolvers | **2/2 modules** | 2 | None | **Done** |
| Statistics (variance, correlation) | 6/6 modules | 6 | None | **Done** |
| Special functions (chi²) | 1/1 module | 1 | None | **Done** |
| Meta-population | **8/8 ops** | 8 | None — FST + AF variance wired (S66) | **Done** |
| Game theory | 2/3 ops | 3 | Spatial stencil (shader exists) | **P2** |
| Introgression | **1/1 chain** | 1 | None — Viterbi chain (S66) | **Done** |

**Bottom line**: ~97% of production math has a GPU path through dispatch.
Session 66: HMM chains, FST (pairwise + global), introgression, inter-pop AF variance.
Remaining: ODE full-loop batching, spatial stencil cooperation.

### Session 48: Raw wgpu Coverage — Most Eliminated

Session 48 rewired 28 binaries from raw wgpu to typed BarraCUDA ops. **Most raw wgpu
usage is now eliminated.**

**Remaining raw wgpu** (intentional or no clean typed op mapping):

| Binary | Reason |
|--------|--------|
| `bench_upstream_vs_local` | Intentional — compares local vs upstream dispatch |
| `validate_gpu_pipeline_swarm` | No upstream equivalent for scores variant |
| `validate_gpu_pipeline_regulatory` | ODE structure mismatch with upstream |
| `validate_cross_dispatch_ode` | Same ODE structure mismatch |

---

*Pure GPU is the proof. Mixed hardware is the optimization. CPU is the fallback.*
