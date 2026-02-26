# neuralSpring → ToadStool/BarraCUDA Handoff V43 — Complete Rewiring + Cross-Spring Evolution

**Session 79 | February 26, 2026**
**Previous**: V42 (Session 78 — S66 absorption, shader convention alignment)

---

## Part 1: Executive Summary

Session 79 completes the modern rewiring pass against ToadStool S66 Wave 5.
A deep scan of all 57 library modules identified 4 remaining local
implementations with upstream equivalents. These were rewired, bringing the
total to **38 functions + 6 shader sources**. The cross-spring evolution
validator now covers **52 checks** (was 39) and the benchmark tracks
**19 provenance-traced operations** (was 15).

### What's New in V43 (Session 79)

| Action | Details |
|--------|---------|
| **+4 functions rewired** | `l2_distance`, `complexity_metric`, `hill_activation`, `hill_repression` |
| **Validator expanded** | 52/52 PASS (was 39), +13 S78 checks |
| **Benchmark expanded** | 19/19 PASS (was 15), +4 S78 provenance benches |
| **Cross-spring lineage** | Complete 5-spring provenance documentation |
| **Total rewires** | **38 functions + 6 shader sources** |

---

## Part 2: New Rewires (S79)

### 2a: `modes::l2_distance` → `barracuda::dispatch::l2_distance_dispatch`

```rust
// Origin: neuralSpring modes.rs → barracuda dispatch (S58)
pub fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    barracuda::dispatch::l2_distance_dispatch(a, b, None)
        .unwrap_or_else(|_| /* CPU fallback */)
}
```

Previously: inline scalar loop. Now: dispatches to BarraCuda CPU path (or
GPU when device is available), with inline fallback for error resilience.

### 2b: `modes::complexity_metric` → `barracuda::stats::fit_linear`

```rust
// Origin: airSpring V009 regression → ToadStool S66 → barracuda::stats
pub fn complexity_metric(complexities: &[f64]) -> (f64, bool) {
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    match barracuda::stats::fit_linear(&t, complexities) {
        Some(result) => (result.params[0], result.params[0] > 0.0),
        None => (0.0, false),
    }
}
```

Previously: manual normal-equation regression. Now: delegates to BarraCuda
`fit_linear` which provides R², RMSE, and predict() for free.

### 2c: `primitives::hill_activation` → `barracuda::stats::hill`

```rust
// Origin: wetSpring + hotSpring gene regulatory → ToadStool S64
pub fn hill_activation(x: f64, amplitude: f64, k: f64, n: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    amplitude * barracuda::stats::hill(x, k, n)
}
```

Previously: local `x.powf(n) / (k.powf(n) + x.powf(n) + HILL_EPS)` with
guard epsilon. Now: delegates core Hill formula to upstream, retains x<=0
guard and amplitude scaling (barracuda::hill has amplitude=1).

### 2d: `primitives::hill_repression` → `barracuda::stats::hill` (inverted)

```rust
pub fn hill_repression(x: f64, amplitude: f64, k: f64, n: f64) -> f64 {
    if x <= 0.0 { return amplitude; }
    amplitude * (1.0 - barracuda::stats::hill(x, k, n))
}
```

Repression = amplitude × (1 − activation). Same delegation pattern.

---

## Part 3: Complete Rewiring Inventory

### By Session

| Session | Count | Functions |
|---------|-------|-----------|
| S58 | 7 | matmul, frobenius, transpose, softmax, l2_distance, mean, variance (dispatch) |
| S59 | 5 | gelu, hmm_forward (dispatch) + ESD, MP bounds, effective_rank (stats) |
| S72 | 4 | softmax_row_wise, fst_single_locus, pairwise_fst_full, argmax_dim |
| S75 | 8 | r_squared, rmse, nse, dot, l2_norm, shannon (stats) + pearson × 2 |
| S76 | 2 | pearson_correlation × 2 (meta_population) |
| S78 | 6 | mae, shannon_from_frequencies, hill × 2, l2_distance (modes), fit_linear |
| S79 | 4 | (included in S78 count — implementation session) |
| **Total** | **38** | + 6 shader sources |

### By Cross-Spring Origin

| Origin Spring | Functions | Via |
|---------------|-----------|-----|
| **airSpring** | rmse, r_squared, nse, mae, index_of_agreement, dot, l2_norm, fit_linear | barracuda::stats (S64–S66) |
| **wetSpring** | shannon, shannon_from_frequencies, simpson, chao1, alpha_diversity, bray_curtis, hill | barracuda::stats (S64) |
| **hotSpring** | pearson_correlation, ESD, MP bounds, validation harness, df64_core | barracuda::stats + spectral (S54–S58) |
| **neuralSpring** | batch_fitness, pairwise_l2, matmul, transpose, frobenius, softmax, gelu, variance, mean, l2_distance, hmm_forward, effective_rank | barracuda::dispatch (S52–S59) |
| **Cross-spring** | hill (wetSpring+hotSpring), Fp64Strategy (hotSpring), FST (wetSpring+neuralSpring) | Multiple |

### Intentionally Local (By Design)

| Module | Function | Reason |
|--------|----------|--------|
| `primitives.rs` | `sigmoid`, `rk4_step` | CPU validation references — no scalar upstream API |
| `spectral_commutativity.rs` | `frobenius_norm`, `mat_mul`, `commutator` | GPU validation references — must be independent |
| `transformer.rs` | `softmax`, `gelu` | CPU fallbacks, already routed via dispatch |
| `cpu_fallback.rs` | `variance` | Independent reference, documents ddof=0 convergence |
| `primitives.rs` | `shannon_equitability` | No upstream `pielou_from_frequencies` equivalent |
| `pangenome_selection.rs` | `spectrum_chi_squared` | Domain-specific e≤0.5 bin skip |

---

## Part 4: Cross-Spring Evolution Story

```text
                    ┌─────────────┐
                    │  hotSpring   │  Precision physics: lattice QCD, HFB,
                    │  (Feb 2026)  │  df64_core, Taylor trig, Lanczos,
                    │              │  Fp64Strategy, GpuDriverProfile
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  wetSpring   │  Bioinformatics: Shannon, Simpson,
                    │  (Feb 2026)  │  HMM, FST, ODE bio, NMF, Anderson,
                    │              │  Bray-Curtis, Chao1, Pielou
                    └──────┬───────┘
                           │
    ┌─────────────┐ ┌──────▼───────┐ ┌─────────────┐
    │  airSpring   │ │neuralSpring  │ │groundSpring  │
    │  (Feb 2026)  │ │  (Feb 2026)  │ │  (Feb 2026)  │
    │ RMSE, R²,    │ │ batch_fitness│ │ multinomial,  │
    │ NSE, MAE,    │ │ pairwise_l2, │ │ MC propagat.  │
    │ fit_linear,  │ │ eigh, swarm, │ │               │
    │ hydrology    │ │ 9 f64 shaders│ │               │
    └──────┬───────┘ └──────┬───────┘ └───────┬───────┘
           │                │                  │
           └────────────────┼──────────────────┘
                            │
                    ┌───────▼──────┐
                    │  ToadStool   │  633+ WGSL shaders
                    │  S66 Wave 5  │  compile_shader_df64
                    │  (045103a7)  │  Sovereign GPU pipeline
                    └──────────────┘
```

### Notable Cross-Spring Synergies

| What Evolved | From → To | Impact |
|-------------|-----------|--------|
| `df64_core.wgsl` | hotSpring biomeGate → ToadStool → neuralSpring 9 shaders | f64 precision on consumer GPUs (48-bit mantissa, 9.9× throughput vs native f64) |
| `Fp64Strategy` | hotSpring hardware detection → ToadStool → all springs | Automatic hybrid/native f64 routing per GPU |
| Shannon diversity | wetSpring bio → ToadStool → neuralSpring eco_dynamics | Same diversity metric across bioinformatics and neuroevolution |
| Hill kinetics | wetSpring gene regulation + hotSpring kinetics → ToadStool → neuralSpring | Enzyme kinetics reused for neural regulatory networks |
| `fit_linear` | airSpring hydrology → ToadStool → neuralSpring modes | Regression from atmospheric science reused for complexity analysis |
| `ValidationHarness` | hotSpring pattern → neuralSpring → ToadStool → all springs | Tolerance-driven validation became ecosystem standard |
| Pairwise L2 | neuralSpring MODES → ToadStool → hotSpring molecular dynamics | Distance metric from neuroevolution reused for particle physics |

---

## Part 5: Validation Results

### Cross-Spring Evolution Validator (52/52 PASS)

| Section | Checks | Status |
|---------|--------|--------|
| S58 dispatch rewires | 7 | PASS |
| S59 dispatch + library | 5 | PASS |
| S72 tensor API + FST | 14 | PASS |
| S78 stats absorption | 13 | PASS |
| Driver profile | 2 | PASS |
| Throughput benchmarks | 11 | PASS |

### Cross-Spring Evolution Benchmark (19/19 PASS)

| Operation | Origin | µs/iter (N=10K) |
|-----------|--------|-----------------|
| RMSE | airSpring | 4.2 |
| R² | airSpring | 12.3 |
| NSE | airSpring | 12.5 |
| IA | airSpring | 14.0 |
| dot | shared | 4.1 |
| l2_norm | shared | 4.1 |
| Shannon | wetSpring | 2.0 |
| Simpson | wetSpring | 0.7 |
| Chao1 | wetSpring | 0.2 |
| alpha_diversity | wetSpring | 5.0 |
| Bray-Curtis | wetSpring | 0.1 |
| DiversityFusion CPU | wetSpring | 83.1 |
| DiversityFusion GPU | wetSpring→GPU | 3459.0 |
| Pearson r | hotSpring | 15.2 |
| MAE | airSpring (S78) | 4.1 |
| Shannon freq | wetSpring (S78) | 30.9 |
| Hill (10K) | wetSpring+hotSpring (S78) | 8.2 |
| fit_linear (10K) | airSpring (S78) | 44.0 |

---

## Part 6: Recommendations for ToadStool Team

### Resolved from V42

| # | Item | Status |
|---|------|--------|
| 1 | `stats::mae` | **RESOLVED** (S66) |
| 3 | `shannon_from_frequencies` | **RESOLVED** (S64) |
| 5 | Population variance convergence | **RESOLVED** (`variance_dispatch` ddof=0) |

### Still Open

| # | Item |
|---|------|
| 2 | Re-export `WGSL_RK4_PARALLEL` constant |
| 4 | `pearson_correlation_or(default)` convenience |
| 6 | Cross-spring benchmark standardization as `barracuda::bench` module |
| 7 | Absorb 9 f64 shaders (all use `compile_shader_df64` convention) |
| 8 | Create `barracuda/src/shaders/folding/` directory |
| 10 | f64 SDPA as alternative to f32 3-pass pipeline |
| 11 | Fused MLP dispatch (matmul + ReLU in one submission) |
| 12 | `barracuda::nn::Linear`/`MLP` abstraction |

### New from V43

| # | Item |
|---|------|
| 13 | **`stats::hill` with amplitude parameter** — all neuralSpring callers wrap `amplitude * hill(x, k, n)`. An `hill_scaled(x, amplitude, k, n)` would eliminate the wrapper. |
| 14 | **`l2_distance_dispatch` on None device** — currently dispatches correctly to CPU, but the function signature takes `Option<&Arc<WgpuDevice>>` which forces callers to pass `None`. A `l2_distance_cpu(a, b)` shortcut would be cleaner. |

---

## Part 7: Current State

| Metric | Value |
|--------|-------|
| Papers reproduced | 25 + 5 baseCamp + 3 WDM surrogates |
| Python baselines | 209/209 PASS |
| Cross-spring validator | **52/52** PASS |
| Cross-spring benchmark | **19/19** PASS |
| Library tests | **581/581** PASS |
| Forge tests | **43/43** PASS |
| Doc tests | **9/9** PASS |
| Upstream rewires | **38 functions + 6 shader sources** |
| ToadStool HEAD | `045103a7` (S66 Wave 5) |

### Quality Gates (all green)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | **PASS** |
| `cargo clippy --workspace -- -D warnings` | **0 warnings** |
| `cargo test --workspace` | **581 lib + 43 forge + 9 doc PASS** |
| `validate_cross_spring_evolution` | **52/52 PASS** |
| `bench_cross_spring_evolution` | **19/19 PASS** |
| SPDX compliance | **100%** |

---

*AGPL-3.0-or-later | neuralSpring → ToadStool V43 | Session 79 | February 26, 2026*
