# neuralSpring V105 → ToadStool/BarraCUDA Absorption + Niche Architecture Handoff

**Date:** March 15, 2026
**From:** neuralSpring S154 (1297 tests, 4500+ checks, 260 binaries, 47 modules)
**To:** ToadStool/BarraCUDA team
**Authority:** wateringHole (ecoPrimals Core Standards)
**Supersedes:** V104 S153 Ecosystem Audit Handoff (Mar 15)
**Pins:** barraCuda v0.3.5 (`0649cd0`), toadStool S146+ (`751b3849`), coralReef Phase 10
**License:** AGPL-3.0-or-later

---

## Executive Summary

- neuralSpring now has **niche deployment architecture** (Steps 1–4 of 7):
  `src/niche.rs` (self-knowledge), `graphs/neuralspring_deploy.toml` (biomeOS
  deploy graph), capability-based discovery (zero hardcoded primal names)
- Consumes **100+ barraCuda files**, **45+ submodules** across device, tensor,
  dispatch, stats, activations, ops::bio, spectral, nn, nautilus, staging
- **4 local reimplementations** identified as delegation candidates for upstream
- **22 capabilities** registered (science + provenance + cross-primal + compute)
- **Zero production panics, zero unsafe, zero clippy warnings (pedantic+nursery)**
- All 3 crate roots `#![forbid(unsafe_code)]`

---

## Part 1: Niche Deployment Architecture

neuralSpring now follows the airSpring/groundSpring niche deployment pattern:

| Artifact | Status | Description |
|----------|--------|-------------|
| `src/niche.rs` | **NEW** | Self-knowledge: NICHE_NAME, 22 CAPABILITIES, operation_dependencies(), cost_estimates(), science_semantic_mappings() |
| `graphs/neuralspring_deploy.toml` | **NEW** | 5-phase biomeOS deploy graph: Tower Atomic → ToadStool/NestGate (optional) → neuralSpring → health check → provenance |
| `src/config.rs` | **EVOLVED** | BIOMEOS_SOCKET_SUBDIR, BIOMEOS_ORCHESTRATOR_SOCKET, TOADSTOOL_NAME_HINT, CORALREEF_NAME_HINT, SQUIRREL_NAME_HINT |
| Primal binary | **EVOLVED** | 5-tier socket resolution (env → XDG_RUNTIME_DIR → temp_dir), zero hardcoded "biomeOS.sock" |

Deploy graph uses `by_capability` resolution for all primals — no hardcoded
primal names. ToadStool and NestGate are `fallback = "skip"` (graceful
degradation when GPU or data primals are unavailable).

### Capabilities Registered

```
science.spectral_analysis        science.anderson_localization
science.hessian_eigen            science.agent_coordination
science.ipr                      science.disorder_sweep
science.training_trajectory      science.evoformer_block
science.structure_module         science.folding_health
science.gpu_dispatch             science.cross_spring_provenance
science.cross_spring_benchmark   science.precision_routing
provenance.begin                 provenance.record
provenance.complete              provenance.status
primal.forward                   primal.discover
capability.list                  compute.offload
```

### Niche Deployment Completion

| Step | Status | Description |
|------|--------|-------------|
| 1. UniBin | DONE | `neuralspring_primal` binary exists |
| 2. Capabilities | DONE | 22 capabilities via `niche::CAPABILITIES` |
| 3. Deploy Graph | DONE | `graphs/neuralspring_deploy.toml` |
| 4. Niche Module | DONE | `src/niche.rs` with full self-knowledge |
| 5. Provenance Trio | PENDING | Wire `rhizoCrypt`/`loamSpine`/`sweetGrass` IPC |
| 6. Cross-Spring Time Series | PENDING | Cross-niche data exchange graphs |
| 7. Workflow Graphs | PENDING | Experiment-specific pipelines as TOML graphs |

---

## Part 2: Primitives Consumed by Domain

### device
`WgpuDevice`, `WgpuDevice::new_cpu_relaxed`, `capabilities::WORKGROUP_SIZE_1D`,
`driver_profile::{Fp64Strategy, GpuDriverProfile, PrecisionRoutingAdvice}`

### tensor
`Tensor`, `Tensor::from_data`, `Tensor::softmax`, `Tensor::softmax_dim`,
`Tensor::layer_norm_wgsl`, `Tensor::gelu_wgsl`, `SessionTensor`

### dispatch
`matmul_dispatch`, `softmax_dispatch`, `gelu_dispatch`, `variance_dispatch`,
`dispatch_for`, `DispatchTarget`

### stats
`mean`, `dot`, `l2_norm`, `r_squared`, `correlation::{variance, pearson_correlation}`,
`shannon`, `shannon_from_frequencies`, `simpson`, `chao1_classic`, `pielou_evenness`,
`bray_curtis`, `marchenko_pastur_bounds`, `bootstrap_ci`, `jackknife`,
`kimura_fixation_prob`, `norm_cdf`, `norm_ppf`,
`hargreaves_et0`, `thornthwaite_et0`, `hamon_et0`, `makkink_et0`, `turc_et0`,
`hargreaves_et0_batch`

### activations
`sigmoid`, `gelu`, `relu`, `relu_batch`

### linalg
`eigh_f64`

### spectral
`BatchIprGpu`, `level_spacing_ratio`, `spectral_bandwidth`,
`spectral_condition_number`, `classify_spectral_phase`, `tridiag_eigenvectors`

### ops::bio
`PairwiseL2Gpu`, `MultiObjFitnessGpu`, `SwarmNnGpu`, `HillGateGpu`,
`BatchFitnessGpu`, `PairwiseJaccardGpu`, `PairwiseHammingGpu`,
`LocusVarianceGpu`, `SpatialPayoffGpu`, `HmmBatchForwardF64`,
`GillespieGpu`, `WrightFisherGpu`, `StencilCooperationGpu`,
`KmerHistogramGpu`, `UniFracPropagateGpu`, `DiversityFusionGpu`

### ops (other)
`logsumexp::LogSumExp`, `pairwise_distance::PairwiseDistance`,
`mha::MultiHeadAttention`, `fft::{Fft1D, Fft1DF64, Ifft1D, Rfft}`,
`fused_map_reduce_f64::FusedMapReduceF64`, `variance_f64_wgsl::VarianceF64`,
`linalg::BatchedEighGpu`, `cosine_similarity_f64`, `max_abs_diff_f64`,
`norm_reduce_f64`, `sum_reduce_f64`, `weighted_dot_f64`,
`correlation_f64_wgsl::CorrelationF64`

### nn
`SimpleMlp`, `simple_mlp::{Activation, DenseLayer}`

### nautilus
`NautilusBrain`, `NautilusBrainConfig`, `BetaObservation`, `DriftMonitor`,
`GenerationRecord`, `InstanceId`

### staging
`StatefulPipeline`, `KernelDispatch`, `StatefulConfig`

### pipeline
`ReduceScalarPipeline`

### unified_hardware
`BandwidthTier`, `ComputeExecutor`

### shaders::provenance
`evolution_report`, `cross_spring_shaders`, `cross_spring_matrix`

---

## Part 3: Local Reimplementations — Absorption Candidates

These are local implementations that could delegate to upstream barraCuda:

| Location | Function | Current | barraCuda Alternative | Priority |
|----------|----------|---------|----------------------|----------|
| `src/primitives.rs` | `softmax(x: &[f64])` | max-exp-normalize loop | `barracuda::dispatch::softmax_dispatch` or `Tensor::softmax` | P3 |
| `src/coral_forge/activation.rs` | `layer_norm(...)` | mean/var per row + gamma/beta | `Tensor::layer_norm_wgsl` | P3 |
| `src/coral_forge/activation.rs` | `softmax_rows(x, rows, cols)` | row-wise softmax loop | `Tensor::softmax_dim(1)` | P3 |
| `src/lenet.rs` | `relu(x: &[f64])` | slice max(0, x) | `barracuda::activations::relu_batch` | P3 |

**Note**: These are all CPU-only convenience functions used in validation paths.
They exist so validation binaries can run without GPU. Delegation would tighten
the "zero duplicate math" goal but adds a GPU dependency to CPU-only validation.
The right pattern is likely: keep CPU fallback, add barraCuda delegation behind
`Dispatcher` for GPU-available paths (which `gpu_dispatch` already does for
softmax and GELU).

---

## Part 4: Hardcoding Elimination (S154)

| Before | After | Pattern |
|--------|-------|---------|
| `"biomeOS.sock"` inline fallback | 5-tier resolution: `$BIOMEOS_ORCHESTRATOR_SOCKET` → `$XDG_RUNTIME_DIR/biomeos/biomeos.sock` → `temp_dir()` | Environment → XDG → fallback |
| `"toadstool"` inline name hint | `config::TOADSTOOL_NAME_HINT` | Single constant |
| `"coralreef"` inline name hint | `config::CORALREEF_NAME_HINT` | Single constant |
| `"squirrel"` inline name hint | `config::SQUIRREL_NAME_HINT` | Single constant |
| Duplicated `ALL_CAPABILITIES` | `config::ALL_CAPABILITIES` (S153) | Single source of truth |

---

## Part 5: Cross-Spring Patterns Learned

### From airSpring v0.8.2
- **`src/niche.rs`** — niche self-knowledge pattern (capabilities, deps, costs, semantic mappings)
- **Edition 2024** — migration pattern for Rust edition evolution
- **Zero `panic!()` in library** — all errors through `Result<T, E>`

### From groundSpring V105
- **`#![deny(clippy::expect_used, clippy::unwrap_used)]`** — panic-free production code
- **Typed `tarpc` IPC** — runtime socket discovery with type-safe RPC
- **Shared Python tolerance module** — cross-language tolerance consistency

### From wetSpring V118
- **200+ named tolerances** with `{DOMAIN}_{METRIC}_{QUALIFIER}` naming convention
- **ESN urgency thresholds** — neuromorphic decision making pattern
- **`BatchedOdeRK4<S>::generate_shader()`** — ODE shader generation pattern

### From hotSpring v0.6.31
- **`sovereign_resolves_poisoning()`** — enables DF64 transcendentals on consumer GPUs
- **SM86/Ampere prioritized dispatch** — sovereign dispatch probe order
- **PRIMAL_NAMESPACE** — structured primal identity

### From healthSpring V25
- **`TissueContext` uniform buffer** — species as dispatch parameter
- **Comparative medicine track** — cross-species validation methodology

---

## Part 6: Quality Gates (at handoff time)

| Gate | Status |
|------|--------|
| `cargo fmt --all --check` | 0 diffs |
| `cargo clippy --workspace -- -W pedantic -W nursery` | 0 warnings |
| `cargo test --workspace` | 1297 passed, 0 failed |
| Production `panic!()` | 0 (structural invariants in `nucleus_pipeline` annotated with `#[expect]`) |
| `unsafe` blocks | 0 (`#![forbid(unsafe_code)]` on all 3 crate roots) |
| TODO/FIXME/HACK in production | 0 |
| Local WGSL shaders to absorb | 6 (xoshiro128ss, swarm_nn_scores, logsumexp_reduce, stencil_cooperation, rk45_adaptive, wright_fisher_step) |
| Named tolerances | 80+ (centralized registry) |
| barraCuda import files | 219 |
| Deploy graph | `graphs/neuralspring_deploy.toml` (5-phase, capability-based) |

---

## Recommended ToadStool/BarraCUDA Actions

1. **Review deploy graph** (`graphs/neuralspring_deploy.toml`) for ToadStool
   integration points — neuralSpring declares ToadStool as `by_capability = "compute"`
   with `features = ["gpu", "f64"]` and `fallback = "skip"`
2. **Absorb 6 remaining local WGSL shaders** (xoshiro128ss, swarm_nn_scores,
   logsumexp_reduce, stencil_cooperation, rk45_adaptive, wright_fisher_step)
3. **Consider `sovereign_resolves_poisoning()` integration** for neuralSpring's
   spectral workloads on consumer GPUs (hotSpring v0.6.31 pattern)
4. **Multi-GPU init API** (`compute.hardware.auto_init_all` from toadStool S155b)
   — neuralSpring has multi-GPU validation on RTX 4070 + TITAN V, ready for dispatch
5. **Provenance trio integration** — neuralSpring declares provenance capabilities
   but needs IPC wiring to `rhizoCrypt`/`loamSpine`/`sweetGrass`
6. **Review 4 CPU-only reimplementations** (Part 3) — candidate for `Dispatcher`
   routing if upstream wants to eliminate all local math
