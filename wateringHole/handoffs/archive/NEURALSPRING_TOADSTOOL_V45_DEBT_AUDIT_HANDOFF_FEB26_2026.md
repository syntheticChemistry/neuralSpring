# neuralSpring → ToadStool/BarraCUDA Handoff V45: Debt Audit, Evolution Insights & Absorption Guide

**Session 80 | February 26, 2026**
**Previous**: V44 (Session 79 — absorption request, 9 sovereign folding shaders)
**Type**: Debt audit results + barracuda evolution recommendations + absorption guide
**License**: AGPL-3.0-or-later

---

## Executive Summary

neuralSpring completed a comprehensive codebase audit (Session 80) covering completeness,
provenance, code quality, validation fidelity, barracuda dependency health, evolution
readiness, and test coverage. All identified debt has been resolved:

- **604 lib tests**, 93.5% coverage (target 90%), zero inline magic numbers
- `wdm_surrogate.rs` coverage: 43% → 98%, `basecamp.rs`: 49% → 91%
- 16 `unwrap()` calls eliminated in validation binaries → graceful `Result` flow
- All `1e-30` guards promoted to `tolerances::LOG_ZERO_GUARD`
- WDM EOS provenance complete with script, commit, date, command, environment
- CI now runs Python↔Rust cross-validation and archives baseline artifacts

This handoff documents what the ToadStool/BarraCUDA team should absorb, adapt, or
be aware of from neuralSpring's evolution journey.

---

## 1. BarraCUDA Primitive Usage Report

neuralSpring consumes barracuda across **16 submodules** with **90+ import sites**.
No duplicate math — every barracuda primitive that exists is used where applicable.

### Category A: Heavy Consumption (core to neuralSpring's function)

| Module | Primitives | Call Sites | Purpose |
|--------|-----------|------------|---------|
| `stats` | `r_squared`, `rmse`, `mae`, `nash_sutcliffe`, `shannon`, `shannon_from_frequencies`, `hill`, `dot`, `l2_norm`, `pearson_correlation`, `fit_linear`, `empirical_spectral_density`, `marchenko_pastur_bounds` | 30+ | Validation metrics, diversity, spectral analysis |
| `dispatch` | `matmul_dispatch`, `frobenius_norm_dispatch`, `transpose_dispatch`, `softmax_dispatch`, `gelu_dispatch`, `l2_distance_dispatch`, `mean_dispatch`, `variance_dispatch`, `hmm_forward_dispatch` | 20+ | GPU/CPU transparent routing |
| `tensor` | `Tensor::from_data`, `matmul`, `sub`, `norm`, `mean`, `sum`, `max`, `softmax_dim`, `relu`, `sigmoid`, `tanh`, `transpose`, `add`, `mul` | 25+ | GPU tensor operations |
| `linalg` | `eigh_f64`, `solve_f64`, `cholesky_f64`, `lu_det`, `lu_solve`, `tridiagonal_solve`, `effective_rank` | 10+ | Eigendecomposition, linear systems |

### Category B: Domain-Specific Integration

| Module | Primitives | Purpose |
|--------|-----------|---------|
| `linalg::graph` | `belief_propagation_chain`, `graph_laplacian`, `disordered_laplacian` | Neural PGM, agent coordination |
| `numerical` | `numerical_hessian`, `rk45_solve` | Loss landscapes, ODE evolution |
| `ops::bio` | `fst_variance_decomposition`, `BatchFitnessGpu`, `PairwiseL2Gpu`, `SpatialPayoffGpu` | Population genetics, evolution |
| `spectral` | `level_spacing_ratio`, `BatchIprGpu` | Anderson localization, weight spectral |
| `sample` | `boltzmann_sampling` | Loss landscape energy sampling |

### Category C: Infrastructure / Pipeline

| Module | Primitives | Purpose |
|--------|-----------|---------|
| `device` | `WgpuDevice`, `GpuDriverProfile`, `Fp64Strategy` | GPU initialization, driver detection |
| `ops::fused_map_reduce_f64` | `FusedMapReduceF64` | GPU Shannon entropy |
| `ops::variance_reduce_f64` | `VarianceReduceF64` | GPU population variance |
| `ops::correlation_f64_wgsl` | `CorrelationF64` | GPU Pearson correlation |
| `staging` | `KernelDispatch`, `StatefulPipeline` | GPU pipeline orchestration |
| `validation` | `ValidationHarness`, `exit_no_gpu`, `require!` | Validation infrastructure |

### Category D: Not Used (Available in BarraCUDA)

| Module | Reason Not Used |
|--------|----------------|
| `optimize` (`nelder_mead`, `bisect`, `brent`) | Validated but not needed in science modules |
| `linalg::ridge_regression`, `linalg::nmf` | No current use case in ML validation |
| `nn` (Layer, LossFunction, Optimizer) | neuralSpring validates primitives, not training loops |
| `interpolate` (cubic spline) | No interpolation use cases |
| `timeseries`, `vision` | Domain-specific, not needed |
| `ops::batch_gemm` | Mentioned in docs but direct matmul_dispatch suffices |

---

## 2. Patterns Discovered for ToadStool Absorption

### 2.1 WDM Surrogate Pattern — Reusable MLP Template

neuralSpring's WDM (Warm Dense Matter) work revealed a reusable pattern for
scientific MLP surrogates. This is the third most requested pattern across springs
(after eigendecomposition and diversity metrics).

**toadStool action**: Consider a `barracuda::nn::SimpleMLP` that:
1. Loads weights/biases from JSON (the common interchange format)
2. Applies Z-score normalization with stored mean/std
3. Forward pass: `for layer in layers { x = activation(W @ x + b) }`
4. Returns denormalized output

This would eliminate ~150 lines of boilerplate per spring that implements surrogates.
neuralSpring's `wdm_surrogate.rs` is the reference implementation.

### 2.2 Validation Helper Pattern — Shared Tensor Operations

Session 80 extracted `validate_tensor_unary` and `validate_tensor_reduction` as
shared helpers in `validation.rs`. These generalize the pattern:

```
for each test case:
    create Tensor from data
    apply operation (unary or reduction)
    compare against expected value
    record pass/fail in harness
```

**toadStool action**: Consider adding these to `barracuda::validation` alongside
the existing `ValidationHarness`. Every spring that validates tensor ops would benefit.

### 2.3 Population vs Sample Variance

neuralSpring uses population variance (ddof=0) for GPU kernels but
`barracuda::stats::correlation::variance` provides sample variance (N-1).
This distinction is well-documented but caused confusion during early integration.

**toadStool action**: Consider `barracuda::stats::variance_population` as an
explicit function, or a `ddof` parameter on `variance`. Currently springs must
use `VarianceReduceF64` (GPU) or implement their own CPU population variance.

### 2.4 Error Handling Evolution

neuralSpring evolved all validation binaries from `unwrap()` to `Result`-based
error handling. The pattern: extract GPU operations into helper functions that
return `Result<T, String>`, then match in the validation flow:

```rust
match gpu_operation(&device, &data) {
    Ok(result) => harness.check_abs("name", result, expected, tol),
    Err(e) => harness.check_bool(&format!("name: {e}"), false),
}
```

**toadStool action**: This pattern should be standardized in the `ValidationHarness`
docs. Consider `harness.check_result("name", result, expected, tol)` that handles
the `Result` unwrapping internally.

---

## 3. API Gaps Update (from V44 + Session 80 findings)

### Still Open (from V44)

| # | Gap | Priority | Notes |
|---|-----|----------|-------|
| 1 | **Fused MLP dispatch** | High | N encoder submissions per MLP forward. `TensorSession::fused_mlp` would reduce to 1. |
| 2 | **`stats::hill` with amplitude** | Medium | All neuralSpring callers wrap `amplitude * hill(x, k, n)`. |
| 3 | **`l2_distance_cpu` shortcut** | Low | `l2_distance_dispatch(a, b, None)` works but verbose. |
| 4 | **f64 SDPA pipeline** | Medium | 3 separate shader dispatches; should be 1 pipeline submission. |

### New from Session 80

| # | Gap | Priority | Notes |
|---|-----|----------|-------|
| 5 | **`validate_tensor_unary` / `validate_tensor_reduction`** | Medium | Currently in neuralSpring's `validation.rs`; should live in `barracuda::validation`. |
| 6 | **`Result`-aware `check_*` methods** | Low | `harness.check_abs_result("name", result, expected, tol)` — handles `Err` gracefully. |
| 7 | **Tolerance registry convention** | Low | neuralSpring has 107+ named tolerances in `tolerances/mod.rs` with derivation annotations. Consider a cross-spring tolerance standard. |

---

## 4. Tolerance Provenance (What ToadStool Should Know)

neuralSpring maintains 107+ named tolerances. Key patterns for ToadStool:

| Tolerance | Value | Derivation |
|-----------|-------|------------|
| `LOG_ZERO_GUARD` | `1e-30` | f64 subnormal range; `ln(1e-30) ≈ -69.1` is safely within f64 exponent range. Used in Shannon entropy, chi-squared, FST — any operation that takes `ln(x)` where `x` might be zero. |
| `SWARM_FITNESS_COMPARISON` | `1e-2` | Observed mean gap 3.8e-3 ± 2.1e-3 across 10 seeded runs. Threshold = mean + 3σ. |
| `KAPPUS_WEGNER_REL` | `0.15` | Statistical variance proportional to `1/N_realizations`. With 500 realizations, expected relative deviation ~10-15%. |
| `F32_GPU_ABS` | `1e-3` | f32 mantissa (23 bits) → ~7 decimal digits. After GPU round-trips, 1e-3 captures accumulated error. |
| `F64_CROSS_LANGUAGE` | `1e-10` | Python (IEEE 754 f64) vs Rust (IEEE 754 f64). Differences arise from operation ordering and FMA availability. |

**Convention**: Every tolerance has a `// Derivation:` comment explaining why that
specific value was chosen. This should be a cross-spring standard.

---

## 5. Cross-Spring Lessons for ToadStool Evolution

### What neuralSpring Validated That Benefits All Springs

1. **Dispatch overhead is negligible for CPU**: 9/10 ops show ≤1.04× overhead when routing
   through `barracuda::dispatch` vs direct library calls. Springs should always use dispatch.

2. **GPU crossover at ~1.5ms**: Below this threshold, CPU is faster due to dispatch overhead.
   Above it, GPU wins dramatically (up to 201.7×). This informs the `metalForge` cost model.

3. **`compile_shader_df64` works for ML**: neuralSpring's 9 sovereign folding shaders
   (layer_norm, GELU, sigmoid, SDPA, triangle ops) all use `Df64` successfully. The
   48-bit mantissa is sufficient for protein folding and physics surrogates.

4. **Population variance vs sample variance matters**: GPU kernels should offer both.
   Currently `VarianceReduceF64` does population, `stats::correlation::variance` does sample.

5. **Validation binary modernization**: The `Result`-based pattern (not `unwrap()`) makes
   GPU failures informative rather than panicking. All springs should adopt this.

### What neuralSpring Still Needs from ToadStool

1. **9 sovereign folding shaders absorbed** (layer_norm_f64, gelu_f64, sigmoid_f64,
   sdpa_scores_f64, softmax_f64, attention_apply_f64, triangle_mul_outgoing/incoming_f64,
   triangle_attention_f64) — these compose into a full f64 protein folding forward pass.

2. **Fused MLP dispatch** — single encoder submission for multi-layer forward pass.

3. **`SimpleMLP` struct** — JSON weight loading + forward pass as a BarraCUDA primitive.

---

## 6. Current neuralSpring State (Post-Audit)

| Metric | Value |
|--------|-------|
| Library tests | **604/604** PASS |
| Forge tests | **43/43** PASS |
| Integration tests | **9/9** PASS |
| Line coverage | **93.5%** (target 90%) |
| Validation binaries | **166** |
| Cross-spring validator | **52/52** PASS |
| Upstream rewires | **38 functions + 6 shader sources** |
| Named tolerances | **107+** (all with derivation annotations) |
| Inline magic numbers | **0** |
| `unwrap()` in non-test | **0** |
| Clippy warnings | **0** (pedantic + nursery) |
| Doc warnings | **0** |
| SPDX compliance | **100%** |
| WDM baselines | **3** (nW-01, nW-02, nW-04) |
| Sovereign folding shaders | **9** (pending absorption) |

---

## 7. BarraCUDA Evolution Recommendations

### For ToadStool Core Team

1. **Absorb `validate_tensor_unary` and `validate_tensor_reduction`** into
   `barracuda::validation`. These reduce ~40 lines of boilerplate per validation
   function and standardize the tensor validation pattern across all springs.

2. **Add `barracuda::nn::SimpleMLP`** — the WDM surrogate pattern is reusable:
   JSON weight loading, Z-score normalization, layer-by-layer forward pass.
   neuralSpring's `wdm_surrogate.rs` is the reference (97.6% test coverage).

3. **Standardize tolerance conventions** — `// Derivation:` comments on all
   tolerance constants. neuralSpring's `tolerances/mod.rs` is the reference.

4. **Consider `variance(data, ddof)` API** — population (ddof=0) vs sample (ddof=1)
   as a single function with explicit parameter.

5. **Absorb the 9 sovereign folding shaders** — these compose into a complete
   f64 protein folding forward pass on consumer GPUs. All use `compile_shader_df64`.

### For Future Springs

1. Use `barracuda::dispatch::*` everywhere — the overhead is negligible for CPU
   and the GPU path provides automatic acceleration.

2. Follow the `tolerances/mod.rs` pattern — centralized named constants with
   derivation annotations, not inline magic numbers.

3. Validation binaries should use `Result`-based error handling, not `unwrap()`.
   The `gpu_mlp_forward` helper pattern (return `Result`, match in harness)
   is the recommended approach.

4. Run `cargo llvm-cov` and target 90% library coverage. Low-coverage modules
   indicate untested edge cases that will bite during GPU promotion.

---

## 8. Files Changed in Session 80

| File | Change |
|------|--------|
| `src/provenance.rs` | Added `WDM_EOS_PROVENANCE` record |
| `src/tolerances/mod.rs` | Derivation annotations for LOG_ZERO_GUARD, SWARM_FITNESS_COMPARISON, KAPPUS_WEGNER_REL |
| `src/gpu_ops/reduction.rs` | Inline `1e-30` → `tolerances::LOG_ZERO_GUARD` |
| `src/gpu_ops/population.rs` | Inline `1e-30` → `tolerances::LOG_ZERO_GUARD` |
| `src/wdm_surrogate.rs` | Inline `1e-30` → LOG_ZERO_GUARD + 14 new tests (43→98% coverage) |
| `src/gpu_dispatch/basecamp.rs` | 12 new tests (49→91% coverage) |
| `src/gpu_dispatch/tests_cpu.rs` | 12 new tests + mul_add fixes |
| `src/bin/validate_barracuda_wdm_eos.rs` | 16 unwrap → Result via gpu_mlp_forward |
| `src/validation.rs` | Added validate_tensor_unary + validate_tensor_reduction |
| `src/bin/validate_barracuda_tensor.rs` | Refactored using shared helpers (966→911 lines) |
| `scripts/run_all_baselines.sh` | Added WDM EOS + ML inference + enhanced provenance |
| `.github/workflows/baselines.yml` | Artifact upload for baseline results |
| `.github/workflows/rust.yml` | Cross-validation job (Python + Rust parity) |

---

*AGPL-3.0-or-later | neuralSpring → ToadStool V45 | Session 80 | February 26, 2026*
