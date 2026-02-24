# neuralSpring → ToadStool/BarraCUDA Absorption Briefing — V23

**Date**: February 24, 2026 (Session 58)
**From**: neuralSpring (ecoPrimals/neuralSpring)
**To**: ToadStool/BarraCUDA team
**ToadStool HEAD**: `9404fdb4`

---

## Executive Summary

neuralSpring has completed its upstream dispatch rewiring. 11 functions now
delegate to BarraCUDA (4 baseCamp S56 + 7 domain_ops S58). GpuDriverProfile
is wired in for hardware-adaptive f64 strategy. The handoff → absorb → rewire →
validate cycle is proven at scale. neuralSpring is now a lean validation layer
that proves BarraCUDA produces correct science.

## Confirmed Absorptions (neuralSpring → BarraCUDA)

### Already Absorbed and Rewired

| Component | BarraCUDA Module | Session | Status |
|-----------|-----------------|---------|--------|
| `graph_laplacian` | `barracuda::linalg::graph` | S54 | Rewired S56 |
| `disordered_laplacian` | `barracuda::linalg::graph` | S54 | Rewired S56 |
| `belief_propagation_chain` | `barracuda::linalg::graph` | S56 | Rewired S56 |
| `numerical_hessian` | `barracuda::numerical` | S54 | Rewired S56 |
| `ValidationHarness` | `barracuda::validation` | S59 | Confirmed; local retained (different env var) |
| `exit_no_gpu` / `gpu_required` | `barracuda::validation` | S59 | Confirmed |
| `require!` macro | `barracuda::validation` | S59 | Confirmed |
| `patch_pow_to_polyfill` | `barracuda::shaders::precision` | S58 | Consolidated locally |
| `mat_mul` dispatch | `barracuda::dispatch::matmul_dispatch` | S58 | Rewired S58 |
| `frobenius_norm` dispatch | `barracuda::dispatch::frobenius_norm_dispatch` | S58 | Rewired S58 |
| `transpose` dispatch | `barracuda::dispatch::transpose_dispatch` | S58 | Rewired S58 |
| `softmax` dispatch | `barracuda::dispatch::softmax_dispatch` | S58 | Rewired S58 |
| `l2_distance` dispatch | `barracuda::dispatch::l2_distance_dispatch` | S58 | Rewired S58 |
| `mean` dispatch | `barracuda::dispatch::mean_dispatch` | S58 | Rewired S58 |
| `variance` dispatch | `barracuda::dispatch::variance_dispatch` | S58 | Rewired S58 |

### Shaders Absorbed (18/20 local shaders)

| Shader | BarraCUDA Location | Session |
|--------|-------------------|---------|
| batch_fitness_eval.wgsl | `shaders/ml/` | S27 |
| batch_ipr.wgsl | `shaders/spectral/` | S27 |
| hill_gate.wgsl | `shaders/bio/` | S27 (generalized) |
| locus_variance.wgsl | `shaders/bio/` | S27 |
| logsumexp_reduce.wgsl | `shaders/reduce/` | S51 |
| mean_reduce.wgsl | `shaders/reduce/` | S27 (generalized) |
| multi_obj_fitness.wgsl | `shaders/bio/` | S27 (generalized) |
| pairwise_hamming.wgsl | `shaders/math/` | S27 |
| pairwise_jaccard.wgsl | `shaders/math/` | S27 |
| pairwise_l2.wgsl | `shaders/math/` | S27 (generalized) |
| rk4_parallel.wgsl | `shaders/numerical/` | S27 |
| rk45_adaptive.wgsl | `shaders/numerical/` | S51 |
| spatial_payoff.wgsl | `shaders/math/` | S27 |
| stencil_cooperation.wgsl | `shaders/bio/` | S52 |
| swarm_nn_forward.wgsl | `shaders/bio/` | S27 (generalized) |
| swarm_nn_scores.wgsl | `shaders/bio/` | S52 |
| wright_fisher_step.wgsl | `shaders/bio/` | S52 |
| xoshiro128ss.wgsl | `shaders/misc/` | S51 |

### Still Local (2 shaders)

| Shader | Reason |
|--------|--------|
| `head_split.wgsl` | Different param structs; upstream MHA projection hangs on RTX 4070 |
| `head_concat.wgsl` | Same — needs upstream MHA stability fix |

## Remaining Local Items (Potential Absorption Targets)

### Dispatcher Methods Without Upstream Equivalents

| Method | Domain | Upstream Candidate |
|--------|--------|--------------------|
| `boltzmann` | Temperature-scaled softmax | `boltzmann_dispatch` |
| `hill_activation_batch` | Regulatory biology | `hill_dispatch` |
| `shannon_entropy` | Information theory | `shannon_entropy_dispatch` |
| `pearson_correlation` | Statistics | `pearson_dispatch` |
| `chi_squared` | Statistics | `chi_squared_dispatch` |
| `hmm_backward_step` | Phylogenetics | `hmm_backward_dispatch` |
| `hmm_viterbi_step` | Phylogenetics | `hmm_viterbi_dispatch` |
| `allele_frequencies` | Population genetics | domain-specific |
| `nucleotide_diversity` | Population genetics | domain-specific |
| `replicator_step` | Game theory | domain-specific |
| `eigh` | Spectral analysis | `eigh_dispatch` (with GpuDriverProfile strategy) |
| `disorder_sweep` | Anderson localization | GPU-only batch eigensolve |

### baseCamp Extensions

| Method | Domain | Notes |
|--------|--------|-------|
| `weight_spectral_analysis` | RMT | Uses eigh internally |
| `belief_propagation` | PGM | Multi-layer GEMV chain |
| `agent_interaction_graph` | Multi-agent | Pairwise L2 adjacency |

## Cross-Spring Learnings for BarraCUDA Evolution

### From hotSpring

1. **GpuDriverProfile works**: neuralSpring confirms RTX 4070 correctly detected
   as Ada/NvidiaPtxas/Throttled/Hybrid. The f64 strategy distinction between
   compute-class (Native) and consumer (Hybrid) GPUs is validated.

2. **pow(f64) workaround is essential**: Every WGSL shader using `pow(f64,f64)`
   needs the polyfill on Ada GPUs. The `needs_pow_f64_workaround()` check is used.

3. **LeNet-5 unblocked**: `cpu_conv_pool` is now `pub` — neuralSpring can complete
   GPU path for Paper 008 (LeNet MNIST).

### From wetSpring

1. **Import modernization**: wetSpring moved to crate-root re-exports
   (`barracuda::BatchFitnessGpu` instead of `barracuda::ops::bio::BatchFitnessGpu`).
   neuralSpring could mirror this pattern.

2. **Clippy Rust 1.93**: pedantic+nursery lints tightened — both Springs are clean.

3. **Anderson 3D correlated**: Available but not yet consumed by neuralSpring.
   Could enhance Paper 023 Anderson localization experiments.

### From neuralSpring (for BarraCUDA)

1. **domain_ops size thresholds work well**: Small inputs (n < ~256) correctly
   route to CPU even when GPU is available, avoiding dispatch overhead.

2. **GPU matmul parity expectation**: max diff 2.3e-4 for 64x64 matmul.
   This is expected (accumulation order) and should be documented as the
   expected tolerance for GPU↔CPU parity in matmul operations.

3. **Dispatcher pattern**: The "try upstream, fall back to local CPU" pattern
   is robust and easy to extend. When new domain_ops are added, neuralSpring
   can rewire immediately.

4. **Three-tier validation**: The Python → BarraCUDA CPU → BarraCUDA GPU
   validation chain is proven across 25 papers and 5 baseCamp sub-theses.
   Every tolerance is named, documented, and minimal.

## Validation Summary

| Metric | Value |
|--------|-------|
| Python baselines | 206/206 PASS |
| Rust lib tests | 478 PASS |
| Forge tests | 30 PASS |
| Validation binaries | 145/146 PASS (1 logsumexp driver) |
| Clippy (pedantic+nursery) | 0 warnings |
| Format | clean |
| TODO/FIXME in src/ | 0 |
| unsafe blocks | 0 (forbid enforced) |
| Functions delegating upstream | 11 |
| WGSL shaders absorbed | 18/20 |
| Total checks | 2020+ |

---

*neuralSpring V23 absorption briefing — 11 functions rewired, GpuDriverProfile confirmed, cross-spring cycle validated end-to-end. The Spring is lean and the upstream is strong.*
