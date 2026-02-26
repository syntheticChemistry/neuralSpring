# neuralSpring → ToadStool/BarraCUDA Handoff V42 — S66 Absorption + Shader Convention Alignment

**Session 78 | February 26, 2026**
**Previous**: V41 (Session 77 — WDM surrogates, baseCamp GPU promotion, 9 new shaders)

---

## Part 1: Executive Summary

Session 78 absorbs ToadStool S66 (Waves 1–5) into neuralSpring. Three new
upstream capabilities enabled rewiring:

1. **`barracuda::stats::mae`** — retired local MAE implementation
2. **`barracuda::stats::shannon_from_frequencies`** — retired local Shannon entropy
3. **`variance_dispatch` now uses population variance (ddof=0)** — documented convergence

Additionally, all 9 neuralSpring metalForge shaders were updated to use
ToadStool's `compile_shader_df64` convention (`Df64` struct, `two_prod`,
`df64_zero`, `df64_from_f32`), eliminating duplicated df64 helper functions
and enabling direct absorption via `compile_shader_df64()`.

### What's New in V42 (Session 78)

| Action | Details |
|--------|---------|
| **+2 functions rewired** | `metrics::mae` → `barracuda::stats::mae`, `primitives::shannon_entropy` → `barracuda::stats::shannon_from_frequencies` |
| **Variance convergence** | `cpu_fallback::variance` docs updated — `variance_dispatch` now agrees (ddof=0) |
| **9 shaders aligned** | All df64 shaders use ToadStool `Df64` struct + `compile_shader_df64` convention |
| **ToadStool baseline** | Updated from S65 (`17932267`) to S66 Wave 5 (`045103a7`) |
| **Total rewires** | **34 functions + 6 shader sources** (was 32) |

---

## Part 2: ToadStool S66 Review

### Commits Since V41 Baseline (S65 `17932267`)

| Commit | Session | Scope |
|--------|---------|-------|
| `bd62d939` | S66 | Cross-spring absorption + deep debt + dependency evolution |
| `95eaad92` | S66 W4 | Cross-spring evolution — API gaps, sovereign compiler fix |
| `045103a7` | S66 W5 | Multi-precision expansion — `compile_shader_df64`, universal DF64 math |

### Key S66 Capabilities Absorbed

| Capability | Module | Provenance |
|------------|--------|-----------|
| `mae(observed, simulated)` | `barracuda::stats::metrics` | airSpring → ToadStool S64–S66 |
| `shannon_from_frequencies(freqs)` | `barracuda::stats::diversity` | wetSpring → ToadStool S64 |
| `compile_shader_df64()` | `barracuda::device::wgpu_device` | hotSpring biomeGate → S58–S66 |
| `df64_core.wgsl` | Shader preamble | `Df64` struct, `df64_add`, `df64_mul`, `two_prod`, `df64_from_f32` |
| `df64_transcendentals.wgsl` | Shader preamble | `sqrt_df64`, `exp_df64`, `tanh_df64`, etc. |
| `stats::regression` | `barracuda::stats::regression` | airSpring V009 — `fit_linear`, `fit_quadratic`, etc. |
| `stats::hydrology` | `barracuda::stats::hydrology` | airSpring V009 — `hargreaves_et0`, `soil_water_balance` |
| `stats::moving_window_f64` | `barracuda::stats::moving_window_f64` | S62 infrastructure |

### S66 Capabilities NOT Absorbed (No Current Need)

| Capability | Reason |
|------------|--------|
| `stats::regression::fit_*` | WDM uses MLP surrogates, not curve fitting |
| `stats::hydrology` | Domain-specific to airSpring/groundSpring |
| `stats::moving_window_f64` | No time-series analysis in current neuralSpring scope |

---

## Part 3: Rewiring Details

### 3a: MAE (metrics.rs)

```rust
// Before (V41): local implementation
pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    let n = y_true.len() as f64;
    y_true.iter().zip(y_pred).map(|(t, p)| (t - p).abs()).sum::<f64>() / n
}

// After (V42): delegates to upstream
pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    barracuda::stats::mae(y_true, y_pred)
}
```

### 3b: Shannon Entropy from Frequencies (primitives.rs)

```rust
// Before (V41): local implementation
pub fn shannon_entropy(frequencies: &[f64]) -> f64 {
    let mut h = 0.0;
    for &p in frequencies { if p > DIVISION_GUARD { h -= p * p.ln(); } }
    h
}

// After (V42): delegates to upstream
pub fn shannon_entropy(frequencies: &[f64]) -> f64 {
    barracuda::stats::shannon_from_frequencies(frequencies)
}
```

### 3c: Variance Convention Documentation (cpu_fallback.rs)

Updated module docs to reflect that `variance_dispatch` now uses population
variance (ddof=0), matching neuralSpring's convention. The local fallback
implementation is retained as an independent reference, but the note warning
against rewiring was replaced with documentation of the convergence.

Note: `barracuda::stats::correlation::variance` still uses sample variance
(N-1) — only the dispatch path converged.

---

## Part 4: Shader Convention Alignment

### Before: Inline df64 Helpers

Each of our 9 metalForge shaders inlined their own df64 functions:

```wgsl
// Old (V41 convention)
fn df64_add(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> vec2<f32> { ... }
fn df64_mul(a: f32, b: f32) -> vec2<f32> { ... }
var acc = vec2<f32>(0.0, 0.0);
acc = df64_add(acc.x, acc.y, val, 0.0);
```

### After: ToadStool `compile_shader_df64` Convention

```wgsl
// New (V42 convention — compatible with compile_shader_df64)
// Requires: df64_core.wgsl (auto-injected by compile_shader_df64)
var acc = df64_zero();
acc = df64_add(acc, df64_from_f32(val));
let prod = two_prod(a, b);
acc = df64_add(acc, prod);
output[idx] = acc.hi;
```

### Shaders Updated

| Shader | df64 Usage |
|--------|-----------|
| `layer_norm_f64.wgsl` | Mean + variance reduction via `df64_add`, `two_prod` |
| `gelu_f64.wgsl` | No df64 (pointwise op, uses FMA directly) |
| `sigmoid_f64.wgsl` | No df64 (pointwise op, uses exp branch) |
| `sdpa_scores_f64.wgsl` | Dot product via `two_prod` + `df64_add` |
| `softmax_f64.wgsl` | Exp-sum reduction via `df64_add`, `df64_from_f32` |
| `attention_apply_f64.wgsl` | Weighted sum via `two_prod` + `df64_add` |
| `triangle_mul_outgoing_f64.wgsl` | Contraction via `two_prod` + `df64_add` |
| `triangle_mul_incoming_f64.wgsl` | Contraction via `two_prod` + `df64_add` |
| `triangle_attention_f64.wgsl` | Biased scores via `two_prod` + `df64_add` |

---

## Part 5: Remaining Local Implementations (By Design)

| Module | Function | Reason |
|--------|----------|--------|
| `primitives.rs` | `sigmoid`, `rk4_step`, `hill_activation`, etc. | CPU validation references |
| `spectral_commutativity.rs` | `frobenius_norm`, `mat_mul`, `commutator` | GPU validation references |
| `pangenome_selection.rs` | `spectrum_chi_squared`, `env_association_chi2` | Domain-specific interface |
| `sequence.rs` | LSTM gate dot products | Inline in closure, compiler-inlined |
| `cpu_fallback.rs` | `variance` | Independent reference (now documents ddof=0 convergence) |
| `primitives.rs` | `shannon_equitability` | Computes H/H_max ratio — no upstream equivalent |

---

## Part 6: Recommendations for ToadStool Team

### Resolved from V41

| # | Item | Status |
|---|------|--------|
| 1 | Add `stats::mae` | **RESOLVED** — now in `barracuda::stats::metrics::mae` |
| 3 | `shannon(frequencies)` variant | **RESOLVED** — `shannon_from_frequencies` in S66 |
| 5 | Population vs sample variance | **PARTIALLY RESOLVED** — `variance_dispatch` is ddof=0, `stats::correlation::variance` still ddof=1 |

### Still Open from V40/V41

| # | Item |
|---|------|
| 2 | Re-export `WGSL_RK4_PARALLEL` constant |
| 4 | `pearson_correlation_or(default)` convenience |
| 6 | Cross-spring benchmark standardization as `barracuda::bench` module |

### New from V42

| # | Item |
|---|------|
| 7 | **Absorb 9 f64 shaders** — all use `compile_shader_df64` convention now (see Part 4) |
| 8 | **Create `barracuda/src/shaders/folding/` directory** for triangle ops |
| 9 | **df64 utility deduplication** — shaders no longer inline df64 helpers |
| 10 | **f64 SDPA as alternative pipeline** — drop-in replacement for f32 3-pass |
| 11 | **Fused MLP dispatch** — `matmul_dispatch` + ReLU pattern is common in WDM |
| 12 | **`barracuda::nn::Linear`/`MLP` abstraction** for reusable NN building blocks |

---

## Part 7: Current State

| Metric | Value |
|--------|-------|
| Papers reproduced | 25 + 5 baseCamp + 3 WDM surrogates |
| Python baselines | 209/209 PASS |
| Rust+GPU checks | 2010+ PASS |
| Library tests | **581/581** PASS |
| Forge tests | **43/43** PASS |
| Doc tests | **9/9** PASS |
| metalForge shaders | **30** (9 new f64, all aligned to `compile_shader_df64`) |
| Upstream rewires | **34 functions + 6 shader sources** |
| ToadStool HEAD | `045103a7` (S66 Wave 5) |

### Quality Gates (all green)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | **PASS** |
| `cargo clippy --workspace -- -D warnings` | **0 warnings** |
| `cargo test --workspace` | **581 lib + 43 forge + 9 doc PASS** |
| SPDX compliance | **100%** |

---

*AGPL-3.0-or-later | neuralSpring → ToadStool V42 | Session 78 | February 26, 2026*
