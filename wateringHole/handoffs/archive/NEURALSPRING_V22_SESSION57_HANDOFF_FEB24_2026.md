# neuralSpring V22 — Session 57: ToadStool S58–S59 Sync + pow Consolidation

**Date**: February 24, 2026
**From**: neuralSpring (ML surrogates, scholarly reproduction, isomorphic learning)
**To**: ToadStool / BarraCUDA core team
**Session**: 57
**ToadStool HEAD**: `9404fdb4` (S58: df64/ODE bio/NMF; S59: anderson correlated/ridge/ValidationHarness)
**Previous**: [V21 Session 56](archive/NEURALSPRING_V21_SESSION56_HANDOFF_FEB24_2026.md)
**License**: AGPL-3.0-or-later

---

## Executive Summary

| Metric | V21 (S56) | V22 (S57) | Delta |
|--------|-----------|-----------|-------|
| ToadStool HEAD | `f78cf3b0` | `9404fdb4` | +2 sessions (S58–S59) |
| Duplicate code eliminated | 0 | 4 `patch_pow_to_polyfill` → 1 shared | -60 lines |
| New upstream modules available | 4 | 13 | +9 (from S58–S59) |
| Validation | 144/145 | 144/145 | Same (pre-existing logsumexp) |
| Quality gates | All pass | All pass | — |

---

## Part 1: Confirmed Absorptions from neuralSpring

ToadStool S59 absorbed our `ValidationHarness` infrastructure:

| Component | Origin | BarraCUDA Location | Notes |
|-----------|--------|-------------------|-------|
| `ValidationHarness` struct | `src/validation.rs` | `barracuda::validation` | Core API identical |
| `exit_no_gpu()` | `src/validation.rs` | `barracuda::validation` | Uses `BARRACUDA_REQUIRE_GPU` (ours: `NEURALSPRING_REQUIRE_GPU`) |
| `gpu_required()` | `src/validation.rs` | `barracuda::validation` | Same |
| `require!` macro | `src/validation.rs` | `barracuda::validation` | Same |

**We keep our local copy** because it adds neuralSpring-specific extensions:
`check_abs_or_rel`, `baseline_path`, `gpu_readback`, `max_abs_diff_*`,
`check_gpu_points`, `gpu_tensor!`, and uses `NEURALSPRING_REQUIRE_GPU`.

---

## Part 2: Code Cleanup — pow Polyfill Consolidation

4 identical `patch_pow_to_polyfill` functions across validation binaries were
consolidated into a single shared function in `validation::patch_pow_to_polyfill`:

| Binary | Before | After |
|--------|--------|-------|
| `validate_cross_dispatch_phase4e` | Local 20-line fn | `use validation::patch_pow_to_polyfill` |
| `validate_gpu_pipeline_signal` | Local 20-line fn | `use validation::patch_pow_to_polyfill` |
| `validate_hillgate_f64_fix` | Local 25-line fn | `use validation::patch_pow_to_polyfill` |
| `validate_gpu_signal` | Local 22-line fn | `use validation::patch_pow_to_polyfill` |

**~60 lines of duplicated code eliminated.** All 4 validators confirmed passing.

---

## Part 3: New Upstream Capabilities (Available, Not Yet Consumed)

ToadStool S58–S59 absorbed from wetSpring and hotSpring. These are now available
to neuralSpring via the BarraCUDA path dep but not yet consumed:

| Module | Origin | Potential neuralSpring Use |
|--------|--------|--------------------------|
| `spectral::anderson::anderson_3d_correlated` | wetSpring | baseCamp Sub-01/05: 3D correlated disorder |
| `spectral::anderson::anderson_sweep_averaged` | wetSpring | baseCamp Sub-01: disorder-averaged ⟨r⟩(W) |
| `spectral::anderson::find_w_c` | wetSpring | baseCamp Sub-01: critical disorder crossing |
| `linalg::ridge::ridge_regression` | wetSpring ESN | Future: ESN readout layer |
| `linalg::nmf` | wetSpring | Future: topic decomposition of weight matrices |
| `numerical::ode_bio` | wetSpring | Future: replace local GRN ODE with upstream |
| `device::driver_profile::Fp64Strategy` | hotSpring | Dispatcher f64 routing optimization |
| `device::driver_profile::GpuDriverProfile` | hotSpring | Per-driver workaround detection |
| `dispatch::domain_ops` | cross-spring | Upstream matmul/variance/softmax dispatch |

---

## Part 4: What ToadStool Should Know

### 4.1 We Kept Our ValidationHarness

Our local `validation.rs` (750 lines, 30 tests) extends the core harness with
GPU tensor helpers that are neuralSpring-specific. The upstream copy (295 lines)
correctly contains only the shared core. This is the right pattern: upstream has
the shared infrastructure, Springs add domain-specific extensions.

### 4.2 pow Workaround Is Still Needed

Even with `needs_pow_f64_workaround()` in `GpuDriverProfile`, the shader
compilation pipeline doesn't auto-patch `pow(` → `pow_f64(` yet. neuralSpring
still applies the workaround manually via `validation::patch_pow_to_polyfill`.
When ToadStool adds auto-patching to `compile_shader_f64`, we can remove our
polyfill caller entirely.

### 4.3 dispatch::domain_ops Overlaps with Our Dispatcher

The upstream `matmul_dispatch`, `variance_dispatch`, `softmax_dispatch` etc.
overlap with our local `Dispatcher` methods. Long-term convergence: neuralSpring's
`Dispatcher` should delegate to upstream `domain_ops` once the APIs stabilize.
No action now — both work correctly.

---

## Appendix: Files Modified in Session 57

| File | Change |
|------|--------|
| `src/validation.rs` | Added `patch_pow_to_polyfill` (shared), backtick fix |
| `src/bin/validate_cross_dispatch_phase4e.rs` | Removed local polyfill, import shared |
| `src/bin/validate_gpu_pipeline_signal.rs` | Removed local polyfill, import shared |
| `src/bin/validate_hillgate_f64_fix.rs` | Removed local polyfill, import shared |
| `src/bin/validate_gpu_signal.rs` | Removed local polyfill, import shared |
| 15 `.md` files | ToadStool HEAD `f78cf3b0` → `9404fdb4` |
| `metalForge/ABSORPTION_MANIFEST.md` | S58–S59 absorption table added |
| `EVOLUTION_READINESS.md` | Session 57 section added |
| `specs/TOADSTOOL_HANDOFF.md` | Session 57 sync notes |
