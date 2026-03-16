# neuralSpring V107 S156 — Audit + IPC Discovery Fixes Handoff

**Date**: March 16, 2026
**From**: neuralSpring S156 (V107)
**To**: barraCuda, toadStool, coralReef
**Supersedes**: V106 S155 (Cross-Spring Absorption)
**License**: AGPL-3.0-or-later

## Pins

| Component | Version | Commit |
|-----------|---------|--------|
| barraCuda | v0.3.5 | `0649cd0` |
| toadStool | S146 | `751b3849` |
| coralReef | Iter 49 | (latest) |
| neuralSpring | S156 | V107 |

## Executive Summary

- **2 critical IPC discovery bugs fixed** — capability-based discovery now works
  end-to-end for the first time across all client types
- **Typed BiomeOsClient** created — canonical pattern for `nucleus.*` and
  `capability.*` IPC calls with consistent request ID generation
- **3 validators** converted from ad-hoc assert/println to `ValidationHarness`
  (hotSpring pattern) — all 260 binaries now use structured pass/fail
- **All ecosystem standards met** — zero warnings, zero unsafe, zero mocks in
  production, all files < 1000 LOC, AGPL-3.0-or-later throughout
- **1301 tests**, 0 clippy (pedantic+nursery), 0 fmt diffs, 0 doc warnings

## Part 1 — Critical IPC Discovery Fixes

### P1: `probe_capabilities` response format mismatch

**File**: `playGround/src/ipc_client.rs:218`
**Bug**: `probe_capabilities()` deserialized the `capability.list` result as
`Vec<String>`, but neuralSpring's primal (and others following the same pattern)
returns `{"primal": "neuralspring", "capabilities": ["science.ipr", ...]}`.
This caused `discover_by_capability()` to silently fail for every primal using
the object format, falling through to name-based discovery every time.

**Fix**: New `parse_capability_list()` handles both formats:
- Array: `["cap1", "cap2"]` — deserialized directly
- Object: `{"capabilities": [...]}` — extracts the array from the object

**Impact**: Capability-first discovery now actually works. Previously, all
`discover_by_capability()` calls silently degraded to `discover_socket()`.

### P2: coralReef bridge returns manifest path instead of socket

**File**: `metalForge/forge/src/coralreef_bridge.rs:182`
**Bug**: `discover_by_capability()` scanned manifest JSONs for `shader.compile`
capability, but returned the `.json` manifest file path itself — callers would
try to open it as a Unix socket and fail immediately.

**Fix**: Now parses manifest JSON for `socket_path`/`socket`/`name` fields and
derives the actual socket path. Fallback socket scan widened from hardcoded
`"coralreef"` to hint set `["coralreef", "coral-reef", "shader"]`.

### P3: Squirrel client name-only discovery

**File**: `playGround/src/squirrel_client.rs`
**Was**: `discover_socket(SQUIRREL_NAME_HINT)` — name-only, no capability probe
**Now**: `discover_by_capability("ai.query", SQUIRREL_NAME_HINT)` — matches
the pattern used by ToadStool and coralReef clients.

## Part 2 — IPC Infrastructure Evolution

### Typed BiomeOsClient

**File**: `playGround/src/biomeos_client.rs` (NEW)
Typed client for the biomeOS orchestrator with methods for:
- `register(primal_name, socket)` → `nucleus.register`
- `deregister(primal_name, socket)` → `nucleus.deregister`
- `heartbeat(primal_name, socket, status)` → `nucleus.heartbeat`
- `register_capability(primal, cap, socket)` → `capability.register`
- `register_all_capabilities(primal, caps, socket)` — batch with logging
- `resolve_capability(cap)` → `capability.resolve`

Uses `ipc_client::call()` for consistent request IDs and timeouts.

### Atomic Request IDs in discovery.rs

**File**: `src/bin/neuralspring_primal/discovery.rs`
`forward_to_primal` and `forward_to_primal_raw` used hardcoded `"id": 1` for
all JSON-RPC requests. Now uses `AtomicU64` counter via `next_id()`, matching
`ipc_client`'s pattern. Enables request tracing in multi-request sessions.

## Part 3 — Validation Harness Migration

Three validators converted from ad-hoc `assert!`/`println!` to
`ValidationHarness` with structured `check_bool`/`check_abs` and proper
exit 0/1:

| Binary | Before | After |
|--------|--------|-------|
| `validate_sovereign_compile` | Manual println PASS/FAIL | 34+ checks via harness |
| `validate_mixed_composition_pipeline` | `assert!` + println | Per-stage check_bool |
| `validate_batched_spectral` | `assert!` + println | Pipeline + spectral checks |

## Part 4 — barraCuda Primitives Consumed (current)

neuralSpring uses **45+ barraCuda submodules**, **80+ functions**, across
**219+ import sites** with **46 upstream rewires**:

| Category | Modules |
|----------|---------|
| Device & GPU | `WgpuDevice`, `GpuDriverProfile`, `Fp64Strategy`, `PrecisionRoutingAdvice`, `WORKGROUP_SIZE_1D` |
| Tensor & Session | `Tensor`, `TensorSession`, `GpuSession` |
| Bio ops | `BatchFitnessGpu`, `PairwiseHammingGpu`, `PairwiseJaccardGpu`, `PairwiseL2Gpu`, `LocusVarianceGpu`, `SpatialPayoffGpu`, `MultiObjFitnessGpu`, `BatchIprGpu`, `SwarmNnGpu`, `HillGateGpu`, `HmmBatchForwardF64`, `WrightFisherGpu`, `StencilCooperationGpu`, `GillespieGpu`, `KmerHistogramGpu`, `UniFracPropagateGpu` |
| Fused GPU | `FusedMapReduceF64`, `FusedChiSquaredGpu`, `FusedKlDivergenceGpu`, `VarianceF64`, `CorrelationF64`, `CosineSimilarityF64`, `MaxAbsDiffF64`, `NormReduceF64`, `SumReduceF64`, `WeightedDotF64` |
| Linalg | `BatchedEighGpu`, `tridiag_eigenvectors` |
| FFT | `Fft1D`, `Fft1DF64`, `Ifft1D`, `Rfft` |
| MHA | `MultiHeadAttention` |
| ODE | `Rk45AdaptiveGpu` |
| NN | `SimpleMlp`, `DenseLayer`, `Activation` |
| ESN | `esn_v2` (reservoir computing) |
| Stats | `variance`, `pearson`, `shannon`, `r_squared`, `rmse` |
| Staging | `StatefulPipeline`, `KernelDispatch` |
| Dispatch | `dispatch_for`, `DispatchTarget` |
| Provenance | `cross_spring_shaders`, `SpringDomain`, `evolution_report` |

## Part 5 — Recommended Actions

### For barraCuda team

1. **Expose `WGSL_MEAN_REDUCE` as public constant** — neuralSpring carries a
   local `mean_reduce.wgsl` that's functionally identical to upstream's private
   `reduce::mean_reduce`. Exposing the constant lets us drop the local copy.

2. **`head_split` / `head_concat` unification** — neuralSpring still carries
   local shaders for MHA head reshaping (S-03b workaround). Upstream MHA has
   the ops but param structs differ. Unifying would retire 2 local WGSL files.

3. **`StatefulPipeline` adoption path** — neuralSpring's HMM chains and ODE
   loops still use per-step CPU dispatch. `barracuda::staging::StatefulPipeline`
   could fuse these into GPU-resident pipelines. Document migration pattern.

4. **coralForge Evoformer shaders** — 16+ local WGSL shaders for
   AlphaFold2/3 (gelu_f64, sigmoid_f64, layer_norm_f64, sdpa_scores_f64,
   triangle_attention_f64, ipa_scores_f64, etc.). Evaluate for upstream
   absorption vs keeping in coralReef domain.

### For toadStool team

5. **`capability.list` response format standardization** — Some primals return
   `["cap1", "cap2"]`, others return `{"primal": "...", "capabilities": [...]}`.
   neuralSpring now handles both, but ecosystem should standardize on one
   format. Recommend the object format for extensibility.

6. **Manifest-based discovery protocol** — coralReef bridge scans
   `$XDG_RUNTIME_DIR/ecoPrimals/*.json` for capability manifests. This path
   needs formal specification: what fields are required (`name`, `socket_path`,
   `capabilities`), where manifests live, who writes them.

7. **biomeOS orchestrator socket naming** — Currently `biomeos.sock` in the
   socket directory. Consider standardizing as `biomeos-{family_id}.sock` for
   consistency with primal socket naming.

## Quality Gates (S156)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets --all-features -W pedantic -W nursery` | 0 warnings |
| `cargo test --lib` | 1128/1128 |
| `cargo test -p neural-spring-forge` | 73/73 |
| `cargo test -p neuralspring-playground` | 65 (2 active + 63 unit, 11 ignored) |
| `cargo doc --no-deps --all` | 0 warnings, 265 files |
| `#![forbid(unsafe_code)]` | 3/3 crate roots |
| Files > 1000 LOC | 0 (largest: 949) |
| `#[allow()]` in production | 1 (wildcard_imports in tolerance registry, justified) |
| Mocks outside `#[cfg(test)]` | 0 |
