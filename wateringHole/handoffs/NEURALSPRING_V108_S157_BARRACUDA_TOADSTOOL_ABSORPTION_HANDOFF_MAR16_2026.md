# neuralSpring V108 S157 — barraCuda/toadStool Absorption Handoff

**Date**: March 16, 2026
**From**: neuralSpring S157 (V108)
**To**: barraCuda team, toadStool team
**License**: AGPL-3.0-or-later
**Pins**: barraCuda v0.3.5 (`0649cd0`), toadStool S146 (`751b3849`), coralReef Iter 49

## Executive Summary

neuralSpring S157 completed deep debt elimination and idiomatic Rust evolution.
Key findings and absorption opportunities for the barraCuda/toadStool team:

- **Tower Atomic pattern validated** — neuralSpring eliminated reqwest/ring by routing
  all HTTP through Songbird IPC. The pattern works cleanly for model downloads and
  API calls. Recommend all primals adopt this for external HTTP.
- **2 local WGSL shaders remain** — `head_split.wgsl` and `head_concat.wgsl` still
  local due to upstream MHA param struct mismatch and RTX 4070 hang.
- **Pairwise Hamming f32 regression** — upstream f64 path is 20.85x slower than
  local f32 for 200x500 workloads. Size-based f32/f64 routing would help.
- **39 shaders absorbed**, 46 upstream rewires, 260 validation binaries — neuralSpring
  is the most complete validation suite for barraCuda bio/ML ops.

## Part 1: Absorption Candidates

### 1.1 Local Shaders Still Needing Upstream Home

| Shader | Domain | Issue | Recommended Action |
|--------|--------|-------|--------------------|
| `head_split.wgsl` | MHA | Upstream MHA uses different `Uniforms` struct layout; local shader uses `(seq_len, num_heads, head_dim)` | **barraCuda action**: Unify param structs in `ops::mha`, or expose a `head_split` dispatch that accepts our layout |
| `head_concat.wgsl` | MHA | Same struct mismatch as `head_split` | **barraCuda action**: Same as above — paired shader |

### 1.2 Upstream Regressions Found

| Issue | Benchmark | Local | Upstream | Regression |
|-------|-----------|-------|----------|------------|
| Pairwise Hamming f32→f64 | `bench_upstream_vs_local` (200×500) | f32 local | f64 upstream | **20.85×** slower |

**barraCuda action**: Consider size-based f32/f64 routing for pairwise ops, or expose a
public f32 constant (`WGSL_PAIRWISE_HAMMING_F32`) for validation-scale workloads where
f64 precision is unnecessary.

### 1.3 Private Constants Needing Public Export

neuralSpring's `metalForge/forge/src/shaders.rs` re-exports several constants from
barraCuda that are only accessible via `LazyLock<String>` or private modules:

| Constant | Current Access | Recommended |
|----------|---------------|-------------|
| `WGSL_MEAN_REDUCE` | Not exposed | **barraCuda action**: `pub const` in `shaders::reduce` |
| `WGSL_PAIRWISE_JACCARD` | `LazyLock<String>` | Consider `pub const &str` |
| `WGSL_SPATIAL_PAYOFF` | `LazyLock<String>` | Consider `pub const &str` |
| `WGSL_PAIRWISE_HAMMING` | `LazyLock<String>` | Consider `pub const &str` |
| `WGSL_BATCH_IPR` | `LazyLock<String>` | Consider `pub const &str` |

### 1.4 AlphaFold Shaders (Future Absorption)

10 local WGSL shaders for coralForge (AlphaFold2/3 sovereign structure prediction):

| Shader | Domain |
|--------|--------|
| `torsion_angles_f64.wgsl` | Backbone geometry |
| `backbone_update_f64.wgsl` | IPA backbone update |
| `ipa_scores_f64.wgsl` | Invariant Point Attention |
| `msa_col_attention_scores_f64.wgsl` | MSA column attention |
| `msa_row_attention_scores_f64.wgsl` | MSA row attention |
| `outer_product_mean_f64.wgsl` | Pair representation update |
| `triangle_attention_f64.wgsl` | Triangle attention |
| `triangle_mul_incoming_f64.wgsl` | Triangle multiplication |
| `triangle_mul_outgoing_f64.wgsl` | Triangle multiplication |
| `sdpa_scores_f64.wgsl` | Scaled dot-product attention (f64) |

These are candidates for a `barracuda::ops::structural_bio` module if/when
structure prediction becomes an ecosystem-wide capability.

## Part 2: Tower Atomic Pattern (For All Primals)

neuralSpring validated the Tower Atomic pattern for external HTTP:

```
Before: primal → reqwest → rustls-tls → ring (C assembly)
After:  primal → songbird_http → IPC → Songbird → BearDog+Songbird (pure Rust)
```

### Implementation Pattern

```rust
// Discover Songbird at runtime (capability-based)
let http = SongbirdHttp::discover()?;

// JSON API calls
let info: ModelInfo = http.get_json(&url).await?;

// File downloads (Songbird writes directly to disk)
let bytes = http.download_to_file(&url, &dest).await?;
```

**toadStool action**: Consider adding `http.request` to the compute dispatch
taxonomy so primals can discover HTTP capability alongside GPU compute.

**barraCuda action**: No HTTP dependencies — barraCuda is already pure Rust.
This is informational for the ecosystem.

## Part 3: Validation Coverage Summary

### Current barraCuda Ops Validated

| Category | Ops | Status |
|----------|-----|--------|
| Bio/evolution | `BatchFitnessGpu`, `MultiObjFitnessGpu`, `WrightFisherGpu`, `SwarmNnGpu` | All PASS |
| Bio/distance | `PairwiseL2Gpu`, `PairwiseHammingGpu`, `PairwiseJaccardGpu` | All PASS |
| Bio/signal | `HillGateGpu`, `SpatialPayoffGpu`, `StencilCooperationGpu` | All PASS |
| Bio/population | `LocusVarianceGpu`, `GillespieGpu` | All PASS |
| Spectral | `BatchIprGpu`, `BatchedEighGpu` | All PASS |
| HMM | `HmmBatchForwardF64` | All PASS |
| Stats | `FusedChiSquaredGpu`, `FusedKlDivergenceGpu`, `CorrelationF64`, `VarianceF64` | All PASS |
| ODE | `Rk45AdaptiveGpu` | All PASS |
| MHA | `MultiHeadAttention` | All PASS |
| FFT | `Fft1D`, `Rfft`, `Ifft1D`, `Fft1DF64` | All PASS |
| Metagenomics | `KmerHistogramGpu`, `UniFracPropagateGpu`, `TaxonomyFcGpu` | All PASS |
| Tensor | `Tensor::matmul`, `::add`, `::sub`, `::mul`, `::transpose`, `::layer_norm_wgsl` | All PASS |
| Dispatch | `gelu_dispatch`, `softmax_dispatch`, `matmul_dispatch`, `hmm_forward_dispatch` | All PASS |
| Precision | `Fp64Strategy`, `PrecisionRoutingAdvice` | All PASS |

### Quality Gates

| Check | Result |
|-------|--------|
| `cargo test --lib` | 1128 pass, 0 fail |
| `cargo clippy (pedantic+nursery, -D warnings)` | 0 warnings |
| `cargo fmt --check` | 0 diffs |
| `#![forbid(unsafe_code)]` | enforced |
| C dependencies in workspace | **0** (Tower Atomic) |
| barraCuda pin | v0.3.5 (`0649cd0`) |

## Part 4: Evolution Recommendations

### For barraCuda v0.4.0

1. **Unify MHA param structs** — enable `head_split`/`head_concat` absorption
2. **f32 pairwise path** — size-based routing for validation-scale workloads
3. **Expose shader constants** — `WGSL_MEAN_REDUCE` and pairwise ops as `pub const`
4. **StatefulPipeline** for iterative ops — HMM chains, ODE loops, EA generations
5. **ReduceScalarPipeline** — scalar-only readback for convergence checks

### For toadStool

1. **`http.request` in capability taxonomy** — Tower Atomic is ecosystem-wide
2. **data.hf_fetch** — HuggingFace model capability for NestGate or dedicated primal
3. **RTX 4070 MHA hang** — upstream projection shaders hang; investigate dispatch ordering
