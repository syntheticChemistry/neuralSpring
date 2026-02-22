# neuralSpring → ToadStool: Absorption Request v5

**Date:** February 22, 2026
**From:** neuralSpring (ML validation & evolutionary computation biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**ToadStool HEAD:** `77f70b2e` (Session 31h)
**neuralSpring HEAD:** current `main`

---

## Executive Summary

neuralSpring has 8 validated WGSL shaders and 2 bug reports ready for ToadStool
absorption. The shaders span evolutionary computation, population genetics,
regulatory biology, swarm robotics, and attention mechanisms — all validated
against Python baselines and CPU references. This handoff includes shader
sources, binding layouts, validation counts, cross-spring context, and a
recommended absorption sequence.

The cross-spring pattern continues to pay off: neuralSpring's S-12 eigensolver
fix (Householder+QR) was absorbed at `77f70b2e` and now benefits hotSpring's
nuclear physics and wetSpring's genomics. The 8 pending shaders here follow the
same lifecycle — absorb once, benefit all Springs.

---

## 1. Shaders Ready for Absorption (Priority Order)

### Priority 1: Distance Kernels (3 shaders, 26 checks)

These complete BarraCUDA's `ops::bio::pairwise_*` family. All three use
identical buffer binding patterns — `N×D` flat row-major input → `N×N`
output matrix.

#### 1a. `pairwise_l2.wgsl` — L2 Distance Matrix

| Field | Value |
|-------|-------|
| Source | `metalForge/shaders/pairwise_l2.wgsl` |
| Domain | Open-ended evolution (MODES), novelty search |
| Papers | 012 (Dolson et al. 2019) |
| Validation | `validate_gpu_modes` — **15/15 PASS** |
| Suggested target | `barracuda::ops::bio::pairwise_l2` |

**Buffer layout:**
```
@group(0) @binding(0) input:   array<f32>  [N × D]  (row-major)
@group(0) @binding(1) output:  array<f32>  [N × N]  (pairwise L2 distances)
@group(0) @binding(2) params:  struct { n: u32, dim: u32 }
```

**Cross-spring note:** L2 distance is a general-purpose primitive. hotSpring
could use it for particle trajectory analysis; wetSpring for sequence embedding
distances. Already has siblings `pairwise_jaccard` and `pairwise_hamming`
absorbed at `77f70b2e`.

#### 1b. `multi_obj_fitness.wgsl` — Multi-Objective Fitness Evaluation

| Field | Value |
|-------|-------|
| Source | `metalForge/shaders/multi_obj_fitness.wgsl` |
| Domain | Directed evolution, multi-objective optimization |
| Papers | 014 (Dolson et al. 2022 eLife) |
| Validation | `validate_gpu_directed` — **6/6 PASS** |
| Suggested target | `barracuda::ops::bio::multi_obj_fitness` |

**Buffer layout:**
```
@group(0) @binding(0) genotypes: array<f32>  [pop × genome_len]
@group(0) @binding(1) weights:   array<f32>  [n_obj × genome_len]
@group(0) @binding(2) fitness:   array<f32>  [pop × n_obj]
@group(0) @binding(3) params:    struct { pop: u32, genome_len: u32, n_obj: u32 }
```

**Cross-spring note:** Batch GEMM pattern — identical to `batch_fitness_eval`
but with multi-objective output. Could generalize to `ops::batch_gemm_2d`.

#### 1c. `swarm_nn_forward.wgsl` — Swarm Neural Network Inference

| Field | Value |
|-------|-------|
| Source | `metalForge/shaders/swarm_nn_forward.wgsl` |
| Domain | Heterogeneous swarm robotics, population-parallel NN evaluation |
| Papers | 015 (Foreback/Dolson 2025 IEEE) |
| Validation | `validate_gpu_swarm` — **9/9 PASS** |
| Suggested target | `barracuda::ops::bio::swarm_nn` |

**Buffer layout:**
```
@group(0) @binding(0) inputs:   array<f32>  [n_agents × input_dim]
@group(0) @binding(1) w1:       array<f32>  [input_dim × hidden_dim]
@group(0) @binding(2) w2:       array<f32>  [hidden_dim × output_dim]
@group(0) @binding(3) outputs:  array<f32>  [n_agents × output_dim]
@group(0) @binding(4) params:   struct { n_agents, input_dim, hidden_dim, output_dim: u32 }
```

**Cross-spring note:** Population-parallel MLP forward pass. Generalizes to any
per-individual NN evaluation (evolutionary strategies, quality-diversity).

### Priority 2: Regulatory / Signal Shaders (1 shader, 9 checks)

#### 2a. `hill_gate.wgsl` — Two-Input Hill Function AND Gate

| Field | Value |
|-------|-------|
| Source | `metalForge/shaders/hill_gate.wgsl` |
| Domain | Regulatory networks, signal integration |
| Papers | 021 (Srivastava et al. 2011 J Bacteriol) |
| Validation | `validate_gpu_signal` — **9/9 PASS** |
| Suggested target | `barracuda::ops::bio::hill_gate` |

**Buffer layout:**
```
@group(0) @binding(0) input_a: array<f32>  [N]
@group(0) @binding(1) input_b: array<f32>  [N]
@group(0) @binding(2) output:  array<f32>  [N]
@group(0) @binding(3) params:  struct { n: u32, k_a: f32, k_b: f32, n_a: f32, n_b: f32 }
```

**Cross-spring note:** Elementwise biological gate. Pattern: `H(a) × H(b)`
where `H(x) = x^n / (K^n + x^n)`. Useful for wetSpring's gene regulatory
network modeling.

### Priority 3: Infrastructure Shaders (2 shaders)

#### 3a. `mean_reduce.wgsl` — Scalar Mean Reduction

| Field | Value |
|-------|-------|
| Source | `metalForge/shaders/mean_reduce.wgsl` |
| Domain | Pipeline aggregation (chains with any domain shader) |
| Validation | `validate_gpu_pure_workload` — **7/7 PASS** |
| Suggested target | `barracuda::pipeline::ReduceScalarPipeline` (may already overlap) |

**Note:** Check if `BarraCUDA` already has equivalent reduce functionality.
This shader exists to enable zero-readback pipeline chaining (domain shader →
mean_reduce → single scalar output). If `ReduceScalarPipeline::sum_f64()`
covers this, the shader can retire.

#### 3b. `xoshiro128ss.wgsl` — GPU Xoshiro128** PRNG

| Field | Value |
|-------|-------|
| Source | `metalForge/shaders/xoshiro128ss.wgsl` |
| Domain | GPU-parallel stochastic algorithms |
| Validation | `validate_gpu_prng` — **5/5 PASS** (uniformity, range, determinism, independence, state) |
| Suggested target | `barracuda::ops::prng` |

**Buffer layout:**
```
@group(0) @binding(0) state: array<vec4<u32>>  [N]  (per-thread 128-bit state)
@group(0) @binding(1) output: array<f32>       [N]  (uniform [0,1) floats)
@group(0) @binding(2) params: struct { n: u32 }
```

**Cross-spring note:** Foundation for all stochastic GPU algorithms:
Wright-Fisher drift, Gillespie SSA, parallel EA mutation, MCMC sampling.
hotSpring needs this for HMC trajectories; wetSpring for stochastic
simulation of birth-death processes.

### Priority 4: MHA Shaders (2 shaders, requires S-03b fix)

#### 4a/4b. `head_split.wgsl` + `head_concat.wgsl` — MHA Reshape

| Field | Value |
|-------|-------|
| Sources | `metalForge/shaders/head_split.wgsl`, `metalForge/shaders/head_concat.wgsl` |
| Domain | Multi-Head Attention data movement |
| Validation | `validate_mha_gpu` — **10/10 PASS** (B=4, S=128, H=8, d=512) |
| Suggested target | `barracuda::ops::mha` |

**head_split:** `[B, S, D] → [B, H, S, D/H]`
**head_concat:** `[B, H, S, D/H] → [B, S, D]`

**Absorption recommendation:** Replace `mha_projection.wgsl` with
`matmul + head_split`; replace `mha_output.wgsl` with `head_concat + matmul`.
The current fused projection shaders cause GPU watchdog timeouts on Vulkan
(S-03b). Decomposing into matmul + reshape avoids heavy per-thread nested loops.

---

## 2. Bug Reports

### S-03b: MHA Projection GPU Hang (High Priority)

**Symptom:** `Tensor::multi_head_attention` hangs during GPU execution on
RTX 4070 (Vulkan). CPU backend works correctly.

**Root cause (suspected):** `project_with_head_split` and
`concat_and_project` shaders fuse matmul into the projection loop, creating
heavy per-thread nested loops that trigger GPU watchdog timeout.

**Fix recommendation:** Decompose MHA into:
1. `matmul` (projection) — already validated, uses `KernelRouter`
2. `head_split.wgsl` — pure data movement (validated 10/10)
3. Batched attention — existing `attention.wgsl`
4. `head_concat.wgsl` — pure data movement (validated 10/10)
5. `matmul` (output projection) — same as step 1

This decomposition matches PyTorch's approach and avoids the GPU hang entirely.
neuralSpring's `evolved::mha` uses this pattern successfully.

**Validation:** `validate_mha_gpu` proves correct output at production sizes
(B=4, S=128, H=8, d_head=64).

### S-13: PooledBuffer Drop-Before-Completion Race (Medium Priority)

**Symptom:** `PooledBuffer::drop` returns the buffer to the pool before GPU
commands referencing it complete. Intermittent corruption under heavy workloads.

**Fix recommendation:** Add `device.poll(wgpu::Maintain::Wait)` in
`PooledBuffer::drop` before returning to pool. Alternative: track in-flight
`SubmissionIndex` and `poll(WaitForSubmissionIndex)`.

**neuralSpring workaround:** `evolved::tensor_sync::{gpu_fence, fenced_matmul, materialize}`

---

## 3. Cross-Spring Evolution Learnings

### What neuralSpring contributed to the ecosystem

| Contribution | Impact | Absorbed |
|-------------|--------|----------|
| Householder+QR eigensolver | Trillion-fold accuracy improvement at n≥8 | Yes (`77f70b2e`) |
| 9 WGSL shaders | Bio-compute primitives (HMM, pairwise distance, fitness, ODE, IPR) | 8 yes, 1 (batch_ipr) yes |
| `TensorSession` API design pressure | ML ops added to session (matmul, relu, gelu, softmax, LN) | Yes (S-01/S-11) |
| 4-tier `KernelRouter` | Size-aware matmul dispatch | Yes (S-02) |
| GPU head_split/head_concat | Decomposed MHA without GPU hang | Pending (S-03b) |
| GPU Xoshiro128** PRNG | Foundation for all stochastic GPU algorithms | Pending |

### What neuralSpring learned from other Springs

| Learning | Source | How we applied it |
|----------|--------|-------------------|
| Single-encoder batch pattern | hotSpring MD simulation | Fused pipeline (46–78× speedup) |
| `SHADER_F64` adapter detection | hotSpring GPU characterization | Dual CPU/GPU tensor validation |
| `log_f64` precision fix | wetSpring genomics | All f64 shader operations |
| Ada Lovelace f64 workaround | wetSpring RTX support | RTX 4070 GPU validation |
| Evolve → validate → handoff cycle | hotSpring methodology | All 16 WGSL shader evolutions |
| `SubstrateCapability` dispatch | hotSpring device probing | Cross-dispatch routing |

### Patterns that benefit all Springs

1. **GPU dispatch crossover at ~1.5ms CPU work.** Below: route to CPU.
   Above: route to GPU. `dispatch_for()` codifies this automatically.

2. **f64 on consumer GPUs is viable.** RTX 4070 supports `SHADER_F64`
   with the Ada Lovelace workaround. Precision: 1.75e-14 for eigensolve,
   machine-epsilon for statistics.

3. **Flat row-major layouts eliminate GPU→CPU→GPU round-trips.** All 16
   neuralSpring shaders use `Vec<f64>` / `Vec<f32>` / `Vec<u8>` flat
   layouts that map directly to GPU `array<T>` bindings.

4. **Pipeline chaining (domain → reduce → scalar) eliminates readback.**
   7 validated GPU pipelines prove the pattern: chain domain shader with
   `mean_reduce`, read back only the final scalar.

5. **Stale counts propagate.** We discovered 4 shader validation counts
   were stale across 2 documents (4 vs actual 15/6/9/9). Lesson: keep
   check counts in one authoritative location and derive from there.

---

## 4. BarraCUDA API Usage — Full Inventory

### 4.1 APIs wired and validated

| API | neuralSpring consumer | Checks |
|-----|----------------------|--------|
| `ops::linalg::eigh_householder_qr` | `src/eigh.rs` (delegation) | 9 |
| `ops::bio::hmm::WGSL_HMM_FORWARD_LOG_F32` | forge `HMM_FORWARD_LOG`, `evolved::hmm_forward_gpu` | 13 |
| `ops::bio::batch_fitness::WGSL_BATCH_FITNESS_EVAL` | forge `BATCH_FITNESS_EVAL` | 20 |
| `ops::rk_stage::WGSL_RK4_PARALLEL` | forge `RK4_PARALLEL` | 8 |
| `ops::bio::pairwise_jaccard::WGSL_PAIRWISE_JACCARD` | forge `PAIRWISE_JACCARD` | 6 |
| `ops::bio::pairwise_hamming::WGSL_PAIRWISE_HAMMING` | forge `PAIRWISE_HAMMING` | 5 |
| `ops::bio::locus_variance::WGSL_LOCUS_VARIANCE` | forge `LOCUS_VARIANCE` | 7 |
| `ops::bio::spatial_payoff::WGSL_SPATIAL_PAYOFF` | forge `SPATIAL_PAYOFF` | 5 |
| `spectral::batch_ipr::WGSL_BATCH_IPR` | forge `BATCH_IPR` | 5 |
| `Tensor::from_data`, `to_vec`, `matmul`, `relu`, `gelu`, etc. | All tensor validation | 90 |
| `Tensor::layer_norm_wgsl`, `log_softmax_wgsl` | ML inference validation | 13 |
| `session::TensorSession` | `evolved::mha`, bench binaries | — |
| `ops::fft::{Fft1D, Ifft1D, Fft1DF64, Rfft}` | FFT validation | 24 |
| `staging::StatefulPipeline` | GPU-resident iteration | 10 |
| `dispatch::{dispatch_for, DispatchTarget}` | Cross-dispatch parity | 41 |
| `stats::*`, `linalg::*`, `numerical::*`, `special::*` | 17 CPU port binaries | 170 |
| `shaders::quantized::{dequant_q4/q8, gemv_q4/q8}` | Quantized inference | 15 |
| `shaders::precision::cpu::*` | f64 precision validation | 12 |

### 4.2 APIs available but not yet leveraged

These are available in `77f70b2e` and could extend neuralSpring's coverage:

| API | Potential neuralSpring Use |
|-----|---------------------------|
| `ops::bio::felsenstein` | Phylogenetic likelihood (Paper 016 extension) |
| `ops::bio::gillespie` | Stochastic GRN simulation (Paper 020 extension) |
| `ops::bio::smith_waterman` | Sequence alignment (Paper 017 extension) |
| `ops::bio::ani` | Average nucleotide identity (pangenome, Paper 024) |
| `ops::bio::dnds` | Molecular evolution rates |
| `ops::bio::snp` | Variant calling |
| `spectral::{anderson_*, hofstadter_*, lanczos}` | Extended spectral analysis (Papers 022–023) |
| `numerical::rk45_solve` (GPU) | GPU-parallel adaptive ODE (Papers 020–021) |
| `session::TensorSession` full ML ops | Replace `evolved::mha` when S-03b is fixed |

---

## 5. Recommended Absorption Sequence

```text
Phase 1 (low-hanging fruit — same binding family as absorbed shaders):
  pairwise_l2.wgsl   → ops::bio::pairwise_l2        (sibling of jaccard/hamming)
  hill_gate.wgsl      → ops::bio::hill_gate           (elementwise, trivial)

Phase 2 (batch compute patterns):
  multi_obj_fitness.wgsl → ops::bio::multi_obj_fitness (batch GEMM variant)
  swarm_nn_forward.wgsl  → ops::bio::swarm_nn          (batch MLP forward)
  mean_reduce.wgsl       → pipeline::reduce or retire   (check overlap)

Phase 3 (infrastructure):
  xoshiro128ss.wgsl   → ops::prng                     (all stochastic algos)

Phase 4 (requires S-03b fix):
  head_split.wgsl     → ops::mha (decomposed)
  head_concat.wgsl    → ops::mha (decomposed)
```

---

## 6. Benchmark Data for Prioritization

### GPU vs CPU crossover (RTX 4070)

| Kernel | Scale | GPU µs | CPU µs | Winner |
|--------|-------|--------|--------|--------|
| Pairwise L2 | 200×1000 | ~1,700 | ~7,000 | **GPU 4×** |
| Multi-obj fitness | 50k×64 | ~1,500 | — | GPU (no CPU baseline at scale) |
| Hill gate | 50×50 | ~1,600 | 3 | CPU (below crossover) |
| Swarm NN | 1000×50 | ~1,700 | — | GPU at scale |
| Mean reduce | 100k | ~1,500 | — | Pipeline component |

### Pure Rust vs Python (single-thread NumPy)

| Kernel | Rust µs | Python µs | Speedup |
|--------|---------|-----------|---------|
| NK fitness (1k genotypes) | 17.9 | 14,087 | **787×** |
| Replicator dynamics (10k) | 150 | 34,937 | **233×** |
| RK4 GRN ODE (2k steps) | 219 | 24,660 | **113×** |
| HMM forward (3×5000) | 330 | 12,008 | **36×** |
| Jaccard (30×500) | 142 | 2,045 | **14×** |
| Hamming (20×500) | 34 | 408 | **12×** |
| **Total** | **1,228** | **88,169** | **72×** |

GEMM-heavy operations (commutator 64×64: Python 23µs vs Rust 335µs) show
where GPU acceleration matters most — NumPy's BLAS backend dominates small
dense GEMM. This validates the GPU crossover strategy.

---

## 7. Files Changed in This Session

| File | Change |
|------|--------|
| `README.md` | SHA `77f70b2e`, 12 shortcomings, S-12 in table, eigh note updated |
| `CONTROL_EXPERIMENT_STATUS.md` | SHA `77f70b2e`, 12 shortcomings, shader check counts corrected |
| `EVOLUTION_READINESS.md` | Shader check counts corrected (modes/directed/swarm/signal) |
| `metalForge/ABSORPTION_MANIFEST.md` | SHA `77f70b2e`, S-12 added, eigh.rs delegation noted |
| `whitePaper/README.md` | Date, SHA, 12 shortcomings, eigh_f64 discovery updated |
| `whitePaper/STUDY.md` | 12 shortcomings |
| `whitePaper/BARRACUDA_EVOLUTION.md` | Date, SHA, S-12 ABSORBED, eigendecomp updated, planned shaders |
| `wateringHole/handoffs/` | V4 archived, V5 evolution + absorption request created |

---

*neuralSpring → ToadStool absorption request v5 — 8 shaders, 2 bugs, 78 checks.*
*Lifecycle: evolve → validate → handoff → absorb → lean.*
