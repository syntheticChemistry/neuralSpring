# neuralSpring → ToadStool/BarraCUDA Handoff V48

**Session**: 83 (February 26, 2026)
**neuralSpring HEAD**: `main` (post-S83 commit)
**ToadStool HEAD**: `f0feb226` (S68)
**Previous**: V47 — Titan V Pure Rust Pipeline Validation (archived)

---

## Summary

ToadStool S66–S68 (22 commits since `17932267`) completed the universal precision
evolution: all 700 WGSL shaders are now f64 canonical with runtime downcast via
`LazyLock<String>`. neuralSpring synced to this new HEAD, fixed 5 broken shader
imports, revalidated the full stack, and updated all documentation.

## What Changed in ToadStool (S66–S68)

### Universal Precision Architecture (S67–S68)

- **f64 canonical**: All shaders authored as f64; f32 via `downcast_f64_to_f32()`
  at runtime using `LazyLock<String>`.
- **700 WGSL shaders**: 497 f32 (LazyLock downcast), 182 f64, 21 DF64. Zero
  f32-only shaders remain.
- **Dual-layer precision**: (1) `op_preamble` abstract ops; (2) naga `df64_rewrite`
  for infix → bridge.
- **11 waves** of f32→f64 evolution (Waves 1–11, Session 68).

### Cross-Spring Absorption (S66)

- New stats: `mae`, `hill`, `monod`, `spearman_correlation`, `regression::*`,
  `hydrology::*`, `moving_window_f64::*`, `bootstrap::rawr_mean`.
- `variance_ddof(data, ddof)` — resolves our API gap #3 (population vs sample
  variance in one API).
- `compile_shader_df64` and universal DF64 math.

### Impact on neuralSpring

The precision evolution privatized several shader constants that neuralSpring
re-exported. Changes made:

| Broken Import | Fix |
|---------------|-----|
| `WGSL_PAIRWISE_JACCARD` (now private `LazyLock`) | Local shader copy |
| `WGSL_SPATIAL_PAYOFF` (now private `LazyLock`) | Local shader copy |
| `WGSL_PAIRWISE_HAMMING` (now private `LazyLock`) | Local shader copy |
| `WGSL_LOCUS_VARIANCE` (removed) | Switched to `WGSL_LOCUS_VARIANCE_F64` |
| `rk4_parallel.wgsl` (renamed to `_f64.wgsl`) | Local f32 copy (f64 requires Sovereign polyfill) |
| `WGSL_BATCH_IPR` (`pub const` → `pub static LazyLock`) | Local shader copy |
| `WGSL_SWARM_NN_SCORES` (privatized) | Rewired to `forge::shaders::SWARM_NN_SCORES` |
| `LogSumExp::WGSL_LOGSUMEXP_REDUCE` (renamed) | Rewired to `forge::shaders::LOGSUMEXP_REDUCE` |

## Validation Results

| Gate | Result |
|------|--------|
| `cargo test --lib` | **604/604 PASS** |
| `cargo test -p neural-spring-forge --lib` | **43/43 PASS** |
| `cargo clippy --all-targets -D warnings` | **0 warnings** |
| `validate_all` | **150/150 PASS** |
| `validate_basecamp_gpu` | **14/14 PASS** |
| `validate_gpu_rk4` | **8/8 PASS** |
| `validate_gpu_logsumexp` | **5/5 PASS** |
| `validate_gpu_pipeline_swarm` | **5/5 PASS** |

Pre-existing: `validate_wdm_eos` 35/36 (monotonicity check, known since S77).

## API Gap Status

| # | Gap | Status |
|---|-----|--------|
| 1 | `barracuda::nn::SimpleMLP` | **OPEN** — JSON weight loading + forward pass |
| 2 | `validate_tensor_unary` / `validate_tensor_reduction` in `barracuda::validation` | **OPEN** — shared validation helpers |
| 3 | `variance(data, ddof)` | **CLOSED** — `variance_ddof(data, ddof)` at ToadStool S66 |
| 4 | `harness.check_abs_result()` | **OPEN** — Result-aware validation check |

## Upstream API Sync

| ToadStool S66–S68 API | neuralSpring Status |
|----------------------|---------------------|
| `stats::mae` | Already rewired (S75) |
| `stats::hill`, `stats::monod` | Already rewired (S75) |
| `stats::spearman_correlation` | Already used (S79) |
| `stats::regression::fit_linear` | Already used (S79) |
| `stats::regression::{fit_quadratic, fit_exponential, fit_logarithmic}` | Available, not needed |
| `stats::hydrology::*` | Available, not needed |
| `stats::moving_window_f64::*` | Available, not needed |
| `stats::bootstrap::rawr_mean` | Available, not needed |
| `barracuda::validation::ValidationHarness` | Upstream absorbed from us; keeping local for resilience |

## Documents Updated

14 ToadStool HEAD references updated from `17932267` (S65) to `f0feb226` (S68):
README, CONTROL_EXPERIMENT_STATUS, EVOLUTION_READINESS, DEPRECATION_MIGRATION,
experiments/README, specs/EVOLUTION_MAPPING, specs/TOADSTOOL_HANDOFF,
specs/CROSS_SPRING_EVOLUTION, specs/BARRACUDA_USAGE, metalForge/ABSORPTION_MANIFEST,
metalForge/shaders/ABSORPTION_TRACKER, metalForge/fossils/FOSSIL_RECORD,
whitePaper/CROSS_SPRING_SHADER_LINEAGE, src/evolved/{mha,mod}.rs.

## For ToadStool

### Immediate Actions

1. **Consider re-exporting shader `&str` constants** alongside `LazyLock<String>`.
   Downstream consumers that compile WGSL directly via wgpu (not through the
   Sovereign pipeline) need `&str`. neuralSpring now uses 6 local shader copies
   because the upstream constants were privatized.

2. **`SimpleMLP` remains the top gap** — JSON weight loading + forward pass
   would eliminate the last manual pipeline in WDM surrogate validation.

### Evolution Notes

- The universal precision architecture means all neuralSpring GPU paths now
  automatically benefit from f64 canonical shaders via the barracuda path dep.
  Our local f32 shaders (used for raw wgpu validation) are the only precision
  outliers.

- neuralSpring's `ValidationHarness` is identical to upstream's — could be
  unified via re-export when API stability is confirmed.

## Pending Items from V47

All V47 pending items addressed or carried forward:
- [x] Audit all f64 WGSL shaders for bare `fma()` calls — done (S82)
- [x] Document abstract-float coercion rules — done (S82)
- [x] Sync to ToadStool HEAD post-S68 — done (this handoff)
- [ ] NVK pipeline cache warming strategy — still pending
- [ ] `SimpleMLP` absorption — still pending

---

*V48 handoff — ToadStool S68 universal precision sync. neuralSpring Session 83.*
