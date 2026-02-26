# neuralSpring → ToadStool/BarraCUDA Absorption Request V44

**Session 79 | February 26, 2026**
**Previous**: V43 (Session 79 — complete rewiring inventory, cross-spring provenance)
**Type**: Absorption request — what ToadStool should absorb from neuralSpring

---

## Executive Summary

neuralSpring has reached a stable plateau: 38 functions + 6 shader sources rewired
to upstream BarraCUDA, 52/52 cross-spring evolution checks PASS, 19/19 benchmarks
PASS, 581 lib tests, 166 binaries, zero debt. This handoff formally requests
ToadStool absorption of 9 sovereign folding shaders, documents the WDM surrogate
pattern as a reusable template, and captures cross-spring evolution lessons.

---

## 1. Absorption Request: 9 Sovereign Folding Shaders

These 9 WGSL shaders were developed in `metalForge/shaders/` and already
conform to ToadStool's `compile_shader_df64` convention. They use `Df64`
structs, `df64_add`, `two_prod`, `df64_zero`, `df64_from_f32` — all
auto-injected by `compile_shader_df64`.

### Activations (3 shaders)

| Shader | Purpose | Convention | Recommended Location |
|--------|---------|------------|---------------------|
| `layer_norm_f64.wgsl` | f64-emulated LayerNorm via df64 reduction | `compile_shader_df64` | `shaders/nn/` |
| `gelu_f64.wgsl` | GELU activation with FMA precision | `compile_shader_df64` | `shaders/nn/` |
| `sigmoid_f64.wgsl` | Numerically stable sigmoid (split formula) | `compile_shader_df64` | `shaders/nn/` |

### Scaled Dot-Product Attention (3 shaders — f64 3-pass SDPA)

| Shader | Purpose | Convention | Recommended Location |
|--------|---------|------------|---------------------|
| `sdpa_scores_f64.wgsl` | QK^T / sqrt(d_k) with df64 accumulation | `compile_shader_df64` | `shaders/nn/` |
| `softmax_f64.wgsl` | Row-wise softmax with df64 sum | `compile_shader_df64` | `shaders/nn/` |
| `attention_apply_f64.wgsl` | Weighted sum of values with df64 | `compile_shader_df64` | `shaders/nn/` |

### AlphaFold2-Inspired (3 shaders — sovereign folding)

| Shader | Purpose | Convention | Recommended Location |
|--------|---------|------------|---------------------|
| `triangle_mul_outgoing_f64.wgsl` | Triangle multiplicative update — outgoing (Alg 11) | `compile_shader_df64` | `shaders/folding/` |
| `triangle_mul_incoming_f64.wgsl` | Triangle multiplicative update — incoming (Alg 12) | `compile_shader_df64` | `shaders/folding/` |
| `triangle_attention_f64.wgsl` | Triangle self-attention with pair bias (Alg 13/14) | `compile_shader_df64` | `shaders/folding/` |

### Absorption Notes

- All 9 shaders use workgroup size 256 (triangle attention uses 16×16×1)
- All use `@builtin(global_invocation_id)` for thread indexing
- All activations are pointwise (no shared memory needed)
- SDPA and triangle shaders use `var<workgroup>` shared memory for reductions
- These compose into a full f64 protein folding forward pass on consumer GPUs
- hotSpring's `df64_core.wgsl` + `df64_transcendentals.wgsl` are prerequisites
- Recommended new directory: `barracuda/src/shaders/folding/`

---

## 2. WDM Surrogate Pattern (Reusable Template)

neuralSpring's WDM (Warm Dense Matter) surrogates demonstrate a complete
Python→Rust→GPU pipeline for MLP-based scientific surrogates. The pattern
is reusable for any domain needing learned function approximation.

### Pattern Summary

```
1. Python baseline (control/wdm/*.py)
   └─ SimpleMLP: numpy-only MLP (no PyTorch dependency)
   └─ Z-score normalization of inputs/outputs
   └─ JSON export: weights, biases, normalization stats

2. Rust CPU validator (src/wdm_surrogate.rs)
   └─ Load JSON weights → EosSurrogate::predict()
   └─ Forward pass: for each layer, matmul + bias + ReLU
   └─ Denormalize outputs

3. BarraCUDA GPU validator (src/bin/validate_barracuda_wdm_eos.rs)
   └─ Same weights loaded into Tensor objects
   └─ Forward pass via Tensor::matmul() + relu()
   └─ Compare GPU vs CPU outputs (f32-level parity)
```

### What ToadStool Could Absorb

| Component | Description |
|-----------|-------------|
| `barracuda::nn::SimpleMLP` | Weight-loading + forward pass from JSON |
| `barracuda::nn::Normalization` | Z-score normalize/denormalize with stored stats |
| `barracuda::nn::Linear` | Single layer: matmul + bias |
| Fused MLP dispatch | `matmul + bias + activation` in one encoder submission |

This pattern has been validated with:
- **nW-01**: Stanton-Murillo transport coefficients (viscosity, conductivity)
- **nW-02**: Militzer FPEOS equation of state (pressure, energy vs density/temperature)
- **nW-04**: Classical-to-WDM transfer learning (pretraining advantage demonstration)

---

## 3. API Gaps We Discovered

### High Priority (blocking or frequently worked around)

| # | Gap | Impact | Workaround |
|---|-----|--------|------------|
| 1 | **Fused MLP dispatch** | Every MLP forward pass makes N encoder submissions (N = layers). A `TensorSession::fused_mlp(weights, activations)` would reduce to 1 submission. | Individual matmul + relu calls |
| 2 | **`stats::hill` with amplitude** | All neuralSpring callers wrap `amplitude * hill(x, k, n)`. A `hill_scaled(x, amplitude, k, n)` variant eliminates the wrapper. | `amplitude * barracuda::stats::hill(x, k, n)` |
| 3 | **`l2_distance_cpu` shortcut** | `l2_distance_dispatch(a, b, None)` forces callers to pass `None` device. A CPU-only shortcut would be cleaner. | `l2_distance_dispatch(a, b, None)` |
| 4 | **f64 SDPA pipeline** | 3-pass f64 attention (scores → softmax → apply) as a single pipeline submission. | 3 separate shader dispatches |

### Medium Priority (nice-to-have, documented workarounds)

| # | Gap | Impact |
|---|-----|--------|
| 5 | Re-export `WGSL_RK4_PARALLEL` constant | metalForge references ToadStool shader source by string |
| 6 | `pearson_correlation_or(default)` convenience | Callers guard NaN returns manually |
| 7 | Cross-spring benchmark standardization as `barracuda::bench` module | Each spring reimplements `bench()` timing utility |

---

## 4. Cross-Spring Evolution Lessons

### What Worked

1. **Write → Absorb → Lean cycle**: neuralSpring develops locally in `metalForge/`,
   validates against Python controls, hands off to ToadStool, then rewires to upstream
   once absorbed. This cycle ran 9 times for production shaders and 38 times for functions.

2. **`compile_shader_df64` convention**: hotSpring pioneered this, ToadStool absorbed it,
   and neuralSpring was able to align all 9 sovereign folding shaders without upstream
   changes. The convention (auto-inject `df64_core.wgsl`, use `Df64` struct) is a
   strong cross-spring standard.

3. **Cross-spring stat absorption**: airSpring's regression (`fit_linear`),
   wetSpring's diversity metrics (`shannon_from_frequencies`, `hill`), and
   hotSpring's precision infrastructure (`df64_core`, `Fp64Strategy`) all flow
   through ToadStool and benefit every spring equally.

4. **Population vs sample variance clarity**: ToadStool S66 converged
   `variance_dispatch` to population variance (ddof=0), matching GPU kernel
   convention. `barracuda::stats::correlation::variance` remains sample variance
   (N-1). This distinction is now well-documented across all springs.

### What We Learned for ToadStool

1. **Activations need f64 variants**: Consumer GPUs lack native f64, but scientific
   workloads (protein folding, WDM physics) need >f32 precision for stable training.
   The `compile_shader_df64` path with `Df64` struct provides 48-bit mantissa at
   9.9× throughput vs native f64. We validated this for layer norm, GELU, sigmoid.

2. **MLP forward pass is the next fusion target**: After shader-level fusion
   (single encoder), the next optimization is operation-level fusion (matmul +
   bias + activation in one dispatch). neuralSpring's WDM surrogates showed
   that MLP inference dominates at scale.

3. **Triangle operations are memory-bound**: The AlphaFold2 triangle
   multiplicative updates and triangle attention shaders are O(N³) in the
   contraction dimension. Shared memory tiling (like the matmul BLAS evolution)
   would significantly improve performance.

---

## 5. Complete neuralSpring → ToadStool Inventory

### Functions Rewired (38)

| Session | Count | Functions |
|---------|-------|-----------|
| S58 | 7 | matmul, frobenius, transpose, softmax, l2\_distance, mean, variance |
| S59 | 5 | gelu, hmm\_forward, ESD, MP bounds, effective\_rank |
| S72 | 4 | softmax\_row\_wise, fst\_single\_locus, pairwise\_fst\_full, argmax\_dim |
| S75 | 8 | r\_squared, rmse, nse, dot, l2\_norm, shannon, pearson × 2 |
| S76 | 2 | pearson\_correlation × 2 (meta\_population) |
| S78 | 6 | mae, shannon\_from\_frequencies, hill × 2, l2\_distance (modes), fit\_linear |
| S79 | 6 | (included in S78 count — validation/benchmark session) |

### Shader Sources Rewired (6)

| Validator | Shader Constant | Origin |
|-----------|----------------|--------|
| `validate_barracuda_gpu_anderson` | `WGSL_ANDERSON` | ToadStool |
| `validate_barracuda_gpu_spectral` | `WGSL_COMMUTATOR` | ToadStool |
| `validate_barracuda_gpu_fitness` | `WGSL_FITNESS` | ToadStool |
| `validate_barracuda_gpu_directed` | `WGSL_DIRECTED` | ToadStool |
| `validate_barracuda_gpu_eco` | `WGSL_ECO` | ToadStool |
| `validate_barracuda_gpu_game` | `WGSL_GAME` | ToadStool |

### Shaders Absorbed (21 production + 9 pending folding)

| Category | Count | Status |
|----------|-------|--------|
| Production WGSL (math, bio, ml, spectral, numerical) | 21 | **Absorbed upstream** |
| Sovereign folding f64 (nn, attention, triangle) | 9 | **Pending absorption** |
| Local diagnostic/utility | 0 | N/A |

### Cross-Spring Provenance

```
hotSpring  → ToadStool: df64_core, Fp64Strategy, lattice QCD, spectral
wetSpring  → ToadStool: Shannon, Hill, HMM, FST, diversity metrics
airSpring  → ToadStool: mae, rmse, r², nse, fit_linear, hydrology
neuralSpring → ToadStool: batch_fitness, pairwise_l2, eigh, matmul tiers
groundSpring → ToadStool: multinomial, MC propagation
```

---

## 6. Current State

| Metric | Value |
|--------|-------|
| Library tests | **581/581** PASS |
| Forge tests | **43/43** PASS |
| Integration tests | **9/9** PASS |
| Doc tests | **9/9** PASS |
| Validation binaries | **166** |
| Cross-spring validator | **52/52** PASS |
| Cross-spring benchmark | **19/19** PASS |
| Upstream rewires | **38 functions + 6 shader sources** |
| metalForge shaders | **30** (21 absorbed + 9 folding) |
| WDM Python baselines | **3** (nW-01, nW-02, nW-04) |
| WGSL convention | **100%** `compile_shader_df64` |
| Quality gates | fmt ✓ clippy ✓ test ✓ SPDX ✓ |

---

*AGPL-3.0-or-later | neuralSpring → ToadStool V44 | Session 79 | February 26, 2026*
