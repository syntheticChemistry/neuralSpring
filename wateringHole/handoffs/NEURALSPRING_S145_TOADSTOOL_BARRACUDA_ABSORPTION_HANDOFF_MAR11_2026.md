# neuralSpring S145 → toadStool / barraCuda Absorption Handoff

**Date:** March 11, 2026
**From:** neuralSpring S145 (1115 lib tests, 73 forge tests, 9 integration tests, 258 binaries, 0 clippy, 47 modules)
**To:** barraCuda, toadStool, coralReef teams
**Scope:** Absorption targets, GPU dispatch patterns, NUCLEUS pipeline lessons, cross-spring provenance
**Supersedes:** V98 GPU Dispatch Evolution Handoff (same date, operational focus)
**License:** AGPL-3.0-only

---

## Executive Summary

neuralSpring has reached full lean status for all math primitives — 25 workloads
absorbed into barraCuda, 1 local workload remaining (MHA `head_split`/`head_concat`
with param struct mismatch). This handoff documents:

1. **What toadStool/barraCuda should absorb** from neuralSpring's evolution
2. **Patterns proven at scale** that should become shared infrastructure
3. **NUCLEUS pipeline lessons** from GPU dispatch integration
4. **Cross-spring evolution data** for the learning system

---

## Part 1: Absorption Targets for barraCuda

### P0 — Immediate Absorption (proven, validated, ready)

| Pattern | Source | What It Provides | Validation |
|---------|--------|-----------------|------------|
| `composition_pipeline()` DAG | `metalForge/forge/src/graph.rs` | 6-stage composition DAG with topological execution, substrate-aware routing | 9/9 nucleus_pipeline tests |
| `dispatch_capability()` | `src/nucleus_pipeline.rs` | Runtime GPU capability detection → stage routing (GpuPreferred vs CpuOnly) | Exp 103–106 |
| `StageResult` provenance | `src/nucleus_pipeline.rs` | Records actual substrate (GPU/CPU) used per stage in pipeline metadata | Mixed-hardware Exp 106 |
| Mixed-substrate DAG | `metalForge/forge/src/mixed.rs` | GPU+CPU stages in single pipeline with transfer cost measurement | 47/47 mixed-hardware |
| `enable f64;` PTXAS workaround | `src/pipeline_cache.rs` | Strip `enable f64;` before compilation on Ada Lovelace — prevents silent-zero regression | 55/55 dispatch parity |

### P1 — High-Value Absorption (validated patterns)

| Pattern | Source | What It Provides | Impact |
|---------|--------|-----------------|--------|
| Streaming spectral pipeline | `src/spectral_commutativity.rs` | Eigensolve → Anderson → IPR → classify in single GPU dispatch chain | 28/28 streaming checks |
| Composition scenario builders | `src/visualization/scenarios/` | 21 petalTongue scenario builders covering 8 DataChannel types | Reusable for any spring |
| Isomorphic reservoir ensemble | Exp 099 | Same ESN architecture across 3 domains (digester/glucose/weather) | Cross-domain transfer proof |
| `gpu_or_exit()` helper | `src/lib.rs` | Async GPU init with graceful exit — eliminates 5-line boilerplate across 75+ binaries | Developer ergonomics |
| Tolerance registry pattern | `src/tolerances.rs` | 80+ named tolerances with justification comments, `tolerance_registry!` macro | Adopted by 5 springs |

### P2 — Future Absorption (needs upstream work)

| Pattern | Source | Blocker | When Ready |
|---------|--------|---------|------------|
| MHA head_split/head_concat | `metalForge/shaders/` | Param struct mismatch (`HeadSplitParams` vs `Params`) | After barraCuda unifies MHA projection params |
| DF64 core streaming pipeline | `metalForge/forge/src/coralreef_bridge.rs` | coralReef IPC stabilization | After coralReef Iter 35+ |
| BLAST-like seed-extend search | `src/search/` | CPU-only, needs GPU kmer index | After barraCuda `ops::search` |

---

## Part 2: NUCLEUS Pipeline GPU Dispatch Lessons

### What We Learned (Exp 103–106)

**Eigensolve on GPU (Exp 103):** `tridiag_eigh_gpu` on RTX 4070 produces correct
eigenvalues for composition analysis. The crossover point is ~512×512 matrices —
below this, CPU `eigh_f64` is faster due to GPU dispatch overhead (~1.5ms).
Recommendation: barraCuda should expose a `size_threshold` parameter in
`tridiag_eigh_gpu` for automatic CPU fallback.

**Batched spectral dispatch (Exp 104):** `BatchedComputeDispatch` successfully
processes 5 composition experiments in a single GPU submission. Key finding:
batching eliminates per-experiment dispatch overhead, achieving ~4× throughput
vs sequential dispatch for small-to-medium matrices. This pattern should be
the default for multi-experiment analysis.

**Sovereign compile (Exp 105):** `ComputeDispatch<CoralReefDevice>` compiles and
dispatches via coralReef on RTX 4070 Ada (GSP firmware). The sovereign path
produces bit-identical results to the wgpu Vulkan path for all tested workloads.
This validates the entire sovereign compute pipeline for Ada/GSP hardware.

**Mixed-hardware composition (Exp 106):** GPU stages (eigensolve, attention_anderson)
interleave with CPU stages (data preparation, provenance recording) with measured
transfer costs. Key finding: PCIe transfer cost is negligible (<0.1ms) for
matrices up to 4096×4096 — the real cost is GPU dispatch initialization, not data
movement. toadStool's `transfer_buffer_strategy()` should account for this.

### Dispatcher Integration Pattern

neuralSpring's NUCLEUS pipeline uses this dispatch pattern:

```rust
fn dispatch_capability(stage: &StageNode) -> SubstratePreference {
    match stage.name.as_str() {
        "eigensolve" | "attention_anderson" => SubstratePreference::GpuPreferred,
        _ => SubstratePreference::CpuOnly,
    }
}
```

This should evolve into toadStool's `SubstrateRouter` with per-stage capability
queries rather than name-matching. The `SubstratePreference::GpuPreferred` tier
includes automatic CPU fallback when GPU is unavailable.

---

## Part 3: Cross-Spring Provenance Data

### neuralSpring's Contribution to the Shader Ecosystem

neuralSpring originated these patterns that were absorbed upstream:

| Original Pattern | Origin Session | Upstream Location | Consuming Springs |
|-----------------|---------------|-------------------|-------------------|
| Chi-squared f64 fused reduce | S64 | `barracuda::ops::fused_chi_squared_f64` | wetSpring, groundSpring, airSpring |
| KL divergence f64 fused reduce | S64 | `barracuda::ops::fused_kl_divergence_f64` | wetSpring, neuralSpring |
| Pipeline graph DAG | S133 | `toadstool::orchestration::PipelineGraph` | All springs via NUCLEUS |
| Spectral streaming pipeline | S88 | 15 coralForge df64 shaders → `barracuda::shaders` | hotSpring, groundSpring |
| Mixed-hardware substrate model | S115 | `metalForge::mixed` → `toadstool::dispatch` | All springs |
| `tolerance_registry!` macro | S70 | Pattern adopted by 5 springs | Ecosystem-wide |
| Cross-spring evolution validator | S60 | Pattern → `barracuda::shaders::provenance` | All springs |

### Round-Trip Provenance Examples

These show the full evolution cycle — neuralSpring domain concept → upstream
absorption → all springs benefit:

1. **Chi-squared**: neuralSpring wrote `chi_squared_f64.wgsl` (S64) → toadStool
   absorbed as `FusedChiSquaredGpu` → barraCuda v0.3.5 exposes via
   `ReduceScalarPipeline` → neuralSpring consumes back at higher precision

2. **Pipeline graph**: neuralSpring wrote `PipelineGraph` for biomeOS DAG (S133)
   → toadStool absorbed into `orchestration` module (S139) → available to all
   springs as shared infrastructure

3. **SimpleMlp rewire**: neuralSpring hand-rolled MLP inference for WDM surrogates
   → toadStool S83 absorbed as `barracuda::nn::SimpleMlp` → neuralSpring rewired
   to upstream, eliminating ~300 LOC (S121)

---

## Part 4: 42 metalForge WGSL Shaders — Status Matrix

| Shader | Domain | Status | Upstream Equivalent |
|--------|--------|--------|-------------------|
| `attention_apply_f64.wgsl` | Protein structure | Local | coralForge corpus |
| `backbone_update_f64.wgsl` | Protein structure | Local | coralForge corpus |
| `batch_fitness_eval.wgsl` | Evolution | Absorbed | `barracuda::ops::bio::batch_fitness` |
| `batch_ipr.wgsl` | Spectral | Absorbed | `barracuda::ops::spectral::batch_ipr` |
| `chi_squared_f64.wgsl` | ML validation | **Absorbed S145** | `FusedChiSquaredGpu` |
| `gelu_f64.wgsl` | ML activation | Absorbed | `barracuda::ops::nn::gelu_f64` |
| `head_concat.wgsl` | MHA | **Local** (param mismatch) | `barracuda::ops::mha` |
| `head_split.wgsl` | MHA | **Local** (param mismatch) | `barracuda::ops::mha` |
| `hill_gate.wgsl` | Biology | Absorbed | `barracuda::ops::bio::hill_gate` |
| `hmm_backward_log.wgsl` | HMM | **Absorbed S145** | `barracuda::ops::bio::hmm_backward` |
| `hmm_viterbi.wgsl` | HMM | **Absorbed S145** | `barracuda::ops::bio::hmm_viterbi` |
| `ipa_scores_f64.wgsl` | Protein structure | Local | coralForge corpus |
| `kl_divergence_f64.wgsl` | ML validation | **Absorbed S145** | `FusedKlDivergenceGpu` |
| `layer_norm_f64.wgsl` | ML normalization | Absorbed | `barracuda::ops::nn::layer_norm_f64` |
| `linear_regression.wgsl` | Statistics | Absorbed | `barracuda::stats::fit_linear` |
| `logsumexp_reduce.wgsl` | HMM numerics | Absorbed | `barracuda::ops::logsumexp` |
| `locus_variance.wgsl` | Population genetics | Absorbed | `barracuda::ops::bio::locus_variance` |
| `matrix_correlation.wgsl` | Statistics | Absorbed | `barracuda::stats::pearson_correlation` |
| `mean_reduce.wgsl` | Statistics | Absorbed | `barracuda::shaders::reduce::mean_reduce` |
| `msa_col_attention_scores_f64.wgsl` | Protein MSA | Local | coralForge corpus |
| `msa_row_attention_scores_f64.wgsl` | Protein MSA | Local | coralForge corpus |
| `multi_obj_fitness.wgsl` | Evolution | Absorbed | `barracuda::ops::bio::multi_obj_fitness` |
| `outer_product_mean_f64.wgsl` | Protein structure | Local | coralForge corpus |
| `pairwise_hamming.wgsl` | Phylogenetics | Absorbed | `barracuda::ops::distance::pairwise_hamming` |
| `pairwise_jaccard.wgsl` | Ecology | Absorbed | `barracuda::ops::distance::pairwise_jaccard` |
| `pairwise_l2.wgsl` | Distance | **Absorbed S145** | `barracuda::ops::distance::PairwiseL2Gpu` |
| `rk45_adaptive.wgsl` | ODE integration | Absorbed | `barracuda::ops::rk45_adaptive` |
| `rk4_parallel.wgsl` | ODE integration | Fossil | `barracuda::ops::rk4_parallel_f64` |
| `sdpa_scores_f64.wgsl` | Attention | Local | coralForge corpus |
| `sigmoid_f64.wgsl` | ML activation | Absorbed | `barracuda::ops::nn::sigmoid_f64` |
| `softmax_f64.wgsl` | ML normalization | Absorbed | `barracuda::ops::nn::softmax_f64` |
| `spatial_payoff.wgsl` | Game theory | Absorbed | `barracuda::ops::bio::spatial_payoff` |
| `stencil_cooperation.wgsl` | Game theory | Absorbed | `barracuda::ops::bio::stencil_cooperation` |
| `swarm_nn_forward.wgsl` | Swarm robotics | Absorbed | `barracuda::ops::bio::swarm_nn_forward` |
| `swarm_nn_scores.wgsl` | Swarm robotics | Absorbed | `barracuda::ops::bio::swarm_nn_scores` |
| `torsion_angles_f64.wgsl` | Protein structure | Local | coralForge corpus |
| `triangle_attention_f64.wgsl` | Protein structure | Local | coralForge corpus |
| `triangle_mul_incoming_f64.wgsl` | Protein structure | Local | coralForge corpus |
| `triangle_mul_outgoing_f64.wgsl` | Protein structure | Local | coralForge corpus |
| `wright_fisher_step.wgsl` | Population genetics | Absorbed | `barracuda::ops::bio::wright_fisher` |
| `xoshiro128ss.wgsl` | PRNG | Absorbed | `barracuda::ops::prng_xoshiro` |

**Summary**: 28/42 absorbed, 2 local (param mismatch), 10 coralForge corpus, 1 fossil, 1 absorbed S145.

---

## Part 5: What toadStool Should Evolve

### Hardware Discovery Enrichment

neuralSpring's Exp 103–106 proved that GPU dispatch decisions need:

1. **Matrix size threshold** — CPU wins below ~512×512 for eigensolve
2. **Batch dispatch awareness** — `BatchedComputeDispatch` is 4× faster than sequential
3. **Transfer cost reality** — PCIe cost is negligible vs dispatch init cost
4. **Sovereign compile parity** — coralReef produces bit-identical results on Ada GSP

toadStool should add these to `GpuDevice` capability metadata:
- `eigensolve_crossover_size: usize` (discovered via probe)
- `batch_dispatch_available: bool` (barraCuda v0.3.5+)
- `sovereign_compile_verified: bool` (coralReef validation result)
- `dispatch_init_overhead_us: f64` (measured)

### SpringDomain Evolution

neuralSpring exercises these domains that should inform `SpringDomain` routing:

| Domain | Key Workloads | Precision | Substrate Preference |
|--------|--------------|-----------|---------------------|
| `NeuralSpring` | eigensolve, HMM, attention, GEMM | f64 | GPU for N>512 |
| `Spectral` | Anderson, IPR, Lanczos | f64 | GPU always |
| `Evolution` | fitness eval, Wright-Fisher | f32 | GPU for pop>1000 |
| `Phylogenetics` | HMM forward/backward/Viterbi | f64 | GPU for T×N > 10K |
| `Composition` | mixed DAG (eigensolve + attention + stats) | f64 | Mixed GPU+CPU |

### Absorption from Other Springs (Relevant to toadStool)

From the V98 handoff and cross-spring reviews:

- **hotSpring** (v0.6.29): NVVM-safe DF64 exp (Taylor + Cody-Waite) is the single
  biggest performance unlock — 4-8× on Ampere. Exp 053 data available.
- **wetSpring** (V113): EMP real data loader (`barracuda::io::biom`), Gompertz batch
  fitting, dynamic W(t) model, QS gene regulon database, VRAM-aware batch sizing.
- **healthSpring** (V20): NLME population Monte Carlo, Hill dose-response GPU,
  Michaelis-Menten nonlinear PK — all validated and ready for absorption.

---

## Part 6: Metrics

| Metric | S142 | S145 |
|--------|------|------|
| barraCuda | v0.3.3 (`83aa08a`) | v0.3.5 (`0649cd0`) |
| toadStool | S142 (`a86bc546`) | S146 (`751b3849`) |
| coralReef | Iter 29 (`2779c88`) | Iter 33 (`b783217`) |
| Lib tests | 1048 | 1115 |
| Forge tests | 71 | 73 |
| Integration tests | 9 | 9 |
| Binaries | 233 | 258 |
| Modules | 47 | 47 |
| Absorbed workloads | 20 | 25 |
| Local workloads | 6 | 1 |
| Sovereign compile | 45/46 | 46/46 |
| Dispatch parity | 55/55 | 55/55 |
| Clippy warnings | 0 | 0 |

---

## Next Steps

1. **barraCuda**: Unify MHA projection params → absorb `head_split`/`head_concat` → neuralSpring reaches zero local shaders
2. **toadStool**: Add eigensolve crossover size to `GpuDevice` capability → enable auto-routing in NUCLEUS
3. **toadStool**: Absorb `composition_pipeline()` DAG pattern into `orchestration` module
4. **coralReef**: Nouveau E2E validation on Titan V (kernel 6.17) → unblock sovereign dispatch on Volta
5. **barraCuda**: NVVM-safe DF64 exp (hotSpring P1) → 4-8× Ampere performance unlock

---

*neuralSpring is now fully lean for all non-coralForge math. The remaining local
shaders are coralForge protein structure operations (10 shaders) and 2 MHA
projections awaiting param unification. The evolution cycle is complete:
Python → Rust → WGSL → absorb → lean → compose.*
