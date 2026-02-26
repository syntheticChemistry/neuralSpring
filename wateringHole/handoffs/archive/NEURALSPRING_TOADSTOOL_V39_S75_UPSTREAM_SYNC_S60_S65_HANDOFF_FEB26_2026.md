# neuralSpring → ToadStool/BarraCUDA Handoff V39 — Upstream Sync S60–S65

**Session 75 | February 26, 2026**
**Previous**: V38 (Session 74 — Pure GPU all-domains, cross-system dispatch)

---

## Part 1: Executive Summary

Session 75 syncs neuralSpring to ToadStool commits S60–S65 (4 commits,
234 files changed, ~23K lines). The upstream crate evolved significantly:
DF64 transcendentals, SovereignCompiler, stats absorption, smart refactoring,
and 8 new lattice shaders. This session rewires 9 neuralSpring functions to
upstream `barracuda::stats`, fixes 4 validators broken by API changes, and
achieves **150/150 validate_all** (up from 149/150).

### ToadStool S60–S65 Changes Reviewed

| Session | Commit | Summary |
|---------|--------|---------|
| S60 | `93a61bb5` | DF64 FMA + transcendentals + polyfill hardening + deep debt |
| S61–63 | `86bfe0f5` | SovereignCompiler (naga FMA fusion) + deep debt + archive cleanup |
| S64 | `80f5a707` | Cross-spring stats absorption + lattice shaders + 2,490 tests |
| S65 | `17932267` | Smart refactoring (5 large files) + doc cleanup + test dead code |

### What's New in V39 (Session 75)

| Action | Details |
|--------|---------|
| **9 functions rewired** to `barracuda::stats` | `r_squared`, `rmse`, `nse`, `branch_trunk_dot`, `rmse` (deeponet) → upstream delegates; `shannon_entropy_from_counts` → `barracuda::stats::shannon`; `l2_relative_error` → `barracuda::stats::l2_norm`; `dot` in counterdiabatic, meta\_population, neural\_pgm |
| **logsumexp validator** fixed | Upstream evolved to f64-only; validator updated from f32 to f64 tensors |
| **3 RK4 validators** fixed | `barracuda::ops::rk_stage::WGSL_RK4_PARALLEL` removed upstream; rewired to `neural_spring_forge::shaders::RK4_PARALLEL` |
| **validate_all** | **150/150 PASS** (was 149/150 — logsumexp was the 1 pre-existing failure) |

---

## Part 2: Upstream API Evolution

### New `barracuda::stats` Module (S64)

Absorbed from airSpring/groundSpring/wetSpring:

| Category | Functions |
|----------|-----------|
| Agreement metrics | `rmse`, `mbe`, `nash_sutcliffe`, `r_squared`, `index_of_agreement`, `hit_rate` |
| Descriptive stats | `mean`, `percentile` |
| Vector ops | `dot`, `l2_norm` |
| Diversity (CPU) | `shannon`, `simpson`, `chao1`, `pielou_evenness`, `alpha_diversity`, `bray_curtis`, `rarefaction_curve` |
| Diversity (GPU) | `DiversityFusionGpu::compute`, `diversity_fusion_cpu` |
| Sampling | `BatchedMultinomialGpu::sample`, `multinomial_sample_cpu` |
| Correlation | `pearson_correlation`, `spearman_correlation`, `variance`, `std_dev`, `covariance` |

### New Device Capabilities (S60)

| Method | Purpose |
|--------|---------|
| `WgpuDevice::from_env()` | Deterministic adapter via `BARRACUDA_GPU_ADAPTER` |
| `is_nvk()`, `is_radv()` | Driver detection |
| `needs_f64_exp_log_workaround()` | f64 polyfill guard |
| `probe_f64_exp_capable()` | Runtime f64 exp probe |

### SovereignCompiler (S61)

`barracuda::shaders::sovereign::SovereignCompiler` — naga-IR FMA fusion +
dead expression elimination. Used internally by `compile_shader_f64()` when
SPIR-V passthrough is available. neuralSpring does not call it directly.

### DF64 Transcendentals (S60)

New `df64_transcendentals.wgsl`: `sqrt_df64`, `exp_df64`, `log_df64`,
`sin_df64`/`cos_df64`, `pow_df64`, `tanh_df64`. All at FP32 core speed.

### Upstream Shader Count

645+ → **694** WGSL shaders (49 new, mostly lattice absorption from hotSpring).

### Upstream Test Count

2,440 → **2,490** barracuda tests (stats + diversity absorption).

---

## Part 3: Rewiring Details

### 3.1 `metrics.rs` → `barracuda::stats`

| Local Function | Upstream Equivalent | Status |
|----------------|---------------------|--------|
| `r_squared(y_true, y_pred)` | `barracuda::stats::r_squared(obs, sim)` | **Rewired** — thin delegate |
| `rmse(y_true, y_pred)` | `barracuda::stats::rmse(obs, sim)` | **Rewired** — thin delegate |
| `nse(y_true, y_pred)` | `barracuda::stats::nash_sutcliffe(obs, sim)` | **Rewired** — thin delegate |
| `mae(y_true, y_pred)` | No upstream CPU equivalent | **Kept local** |

### 3.2 `deeponet.rs` → `barracuda::stats`

| Local Function | Upstream Equivalent | Status |
|----------------|---------------------|--------|
| `branch_trunk_dot(b, t, bias)` | `barracuda::stats::dot(b, t) + bias` | **Rewired** |
| `rmse(pred, exact)` | `barracuda::stats::rmse(pred, exact)` | **Rewired** — thin delegate |

### 3.3 `primitives.rs` → `barracuda::stats`

| Local Function | Upstream Equivalent | Status |
|----------------|---------------------|--------|
| `shannon_entropy_from_counts(counts)` | `barracuda::stats::shannon(counts)` | **Rewired** |
| `shannon_entropy(frequencies)` | No equivalent (takes frequencies, not counts) | **Kept local** |
| `shannon_equitability(frequencies)` | No equivalent | **Kept local** |

### 3.4 `neural_pgm.rs` → `barracuda::stats`

| Local Code | Upstream Equivalent | Status |
|------------|---------------------|--------|
| `ev1.iter().zip(ev2).map(\|(&a,&b)\| a*b).sum()` | `barracuda::stats::dot(ev1, ev2)` | **Rewired** |
| `ev.iter().map(\|&x\| x*x).sum().sqrt()` | `barracuda::stats::l2_norm(ev)` | **Rewired** |

### 3.5 `counterdiabatic.rs` → `barracuda::stats`

| Local Code | Upstream Equivalent | Status |
|------------|---------------------|--------|
| `p_s.iter().zip(f_s).map(\|(p,f)\| p*f).sum()` | `barracuda::stats::dot(&p_s, &f_s)` | **Rewired** |

### 3.6 `meta_population.rs` → `barracuda::stats`

| Local Code | Upstream Equivalent | Status |
|------------|---------------------|--------|
| `ns.iter().zip(p_i).map(\|(n,p)\| n*p).sum()` | `barracuda::stats::dot(&ns, &p_i)` | **Rewired** |

### 3.7 Not Rewired (Upstream Gaps)

| neuralSpring | Reason |
|-------------|--------|
| `mae()` | No `barracuda::stats::mae` — only `Tensor::mae_loss` (GPU) |
| `shannon_entropy(frequencies)` | Upstream `shannon()` takes counts; different input type |
| Simpson/Chao1/Bray-Curtis/rarefaction | Not used in neuralSpring |

---

## Part 4: Validator Fixes

### 4.1 `validate_barracuda_logsumexp` (1/5 → 5/5)

**Root cause**: Upstream `LogSumExp::execute()` evolved to f64-only
(`compile_shader_f64`, `create_buffer_f64`). The validator was feeding f32
tensors via `Tensor::from_data`, causing the f64 shader to misinterpret
4-byte floats as 8-byte doubles.

**Fix**: Changed to `Tensor::from_f64_data`, `to_f64_vec()`, and f64
CPU reference. Updated tolerances to `GPU_F64_TRANSCENDENTAL`.

### 4.2 `validate_gpu_stateful_pipeline`, `validate_gpu_pipeline_regulatory`, `validate_cross_dispatch_ode`

**Root cause**: `barracuda::ops::rk_stage::WGSL_RK4_PARALLEL` was removed
upstream in S65 refactoring (RK45 test extraction). The WGSL shader file
still exists but is no longer re-exported as a constant.

**Fix**: Rewired all three to `neural_spring_forge::shaders::RK4_PARALLEL`
(which `include_str!`s the shader directly from the upstream path).

---

## Part 5: Upstream Items NOT Requiring neuralSpring Changes

| Item | Why No Change Needed |
|------|---------------------|
| Cholesky SPD validation | Internal to `cholesky_f64()`; callers see better errors |
| `BARRACUDA_GPU_ADAPTER` env var | neuralSpring uses `NEURALSPRING_BACKEND`; no conflict |
| kernel_router named constants | Private constants; neuralSpring doesn't use `KernelRouter` |
| SovereignCompiler | Internal to `compile_shader_f64()`; automatic |
| Deep debt fixes (Crank-Nicolson, cross-attention, etc.) | Internal upstream improvements |

---

## Part 6: Cumulative Upstream Rewire Count

| Session | Rewires | Running Total |
|---------|---------|---------------|
| S50–S59 | 16 functions | 16 |
| S60–S69 | 5 functions + 6 shader sources | 21 + 6 shaders |
| **S75** | **9 functions** (stats + dot + l2\_norm) | **30 + 6 shaders** |

---

## Part 7: Current State

| Metric | Value |
|--------|-------|
| Papers reproduced | 25 + 5 baseCamp sub-theses |
| Python baselines | 206/206 PASS |
| Rust+GPU checks | 1970+ PASS |
| Total validation | **2180+** checks |
| Library tests | **580/580** PASS |
| Forge tests | **43/43** PASS |
| Integration tests | 9/9 PASS |
| Validation binaries | **163** |
| Named tolerances | **107+** |
| Upstream rewires | **30 functions + 6 shader sources** |
| ToadStool HEAD | `17932267` (S65) |

### Quality Gates (all green)

| Gate | Result |
|------|--------|
| `cargo clippy --lib` | **0 warnings** |
| `cargo test --lib` | **580/580 PASS** |
| `cargo test --test integration` | **9/9 PASS** |
| `validate_cross_spring_evolution` | **39/39 PASS** |
| `validate_gpu_pure_workload_all` | **10/10 PASS** |
| `validate_cross_system_dispatch` | **46/46 PASS** |
| `validate_all` | **150/150 PASS** |
| `cargo doc --no-deps` | **0 warnings** |
| Python baselines | **206/206 PASS** |
| CPU↔Python parity | **39/39 PASS** (1e-10) |
| Coverage | **94.53%** |
| SPDX compliance | **100%** |

---

## Part 8: Recommendations for ToadStool Team

1. **Add `stats::mae`**: neuralSpring keeps a local MAE because upstream only
   has `Tensor::mae_loss` (GPU). A CPU `barracuda::stats::mae(obs, sim)` would
   let us retire the local version.

2. **Re-export `WGSL_RK4_PARALLEL`**: The S65 refactoring moved RK45 tests to
   `rk45_tests.rs` and stopped re-exporting the WGSL constant. neuralSpring
   works around this via `include_str!` but a public constant would be cleaner.

3. **`shannon(frequencies)` variant**: `barracuda::stats::shannon(counts)` only
   accepts count data. A `shannon_from_frequencies()` variant would let
   neuralSpring retire `primitives::shannon_entropy()`.

---

## Part 9: Document Index

| Document | Location | Purpose |
|----------|----------|---------|
| This handoff | `wateringHole/handoffs/` | V39 upstream sync S60–S65 |
| BARRACUDA_USAGE | `specs/BARRACUDA_USAGE.md` | Module-level usage inventory |
| CROSS_SPRING_EVOLUTION | `specs/CROSS_SPRING_EVOLUTION.md` | Shader/primitive provenance |
| TOADSTOOL_HANDOFF | `specs/TOADSTOOL_HANDOFF.md` | Shortcoming tracking (all resolved) |
| EVOLUTION_READINESS | `EVOLUTION_READINESS.md` | Module → WGSL → pipeline mapping |
| Experiment 043 | `experiments/README.md` | S75 upstream sync journal |
| Cross-spring bench | `bench_cross_spring_evolution` | Provenance-traced benchmark (15/15 PASS) |
| V38 (archived) | `wateringHole/handoffs/archive/` | S74 pure GPU all-domains handoff |

---

*AGPL-3.0-or-later | neuralSpring → ToadStool V39 | Session 75 | February 26, 2026*
