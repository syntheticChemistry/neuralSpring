# neuralSpring → ToadStool/BarraCUDA Handoff V68 — S71 GPU Stats Parity + Upstream Shader Bugs

**Date**: March 1, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Session 101 — ToadStool pin bump `1dd7e338`→`8dc01a37` (S71+++), GPU stats parity validation (KimuraGpu, HistogramGpu PASS; JackknifeMeanGpu, HargreavesBatchGpu blocked by shader bugs), upstream shader bug reports, +1 validation binary (219 total)
**Supersedes**: V67 (Deep Debt + Cross-Spring Evolution)

---

## Executive Summary

- **ToadStool pin advanced** 6 commits (`1dd7e338`→`8dc01a37`): S71 ComputeDispatch migration, DF64 transcendentals (gamma, erf, trig, hyperbolic), pure math shaders, ~9000 lines of boilerplate removed
- **GPU stats parity validated**: `KimuraGpu` CPU↔GPU max diff = 1.11e-16 (essentially zero); `HistogramGpu` correct bins, counts, distribution
- **Two upstream shader bugs discovered** (blocking 2 of 4 GPU stats ops):
  - `jackknife_mean_f64.wgsl`: `bitcast<f64>(vec2<u32>())` fails naga validation after `ComputeDispatch::f64()` DF64 emulation transform
  - `hargreaves_batch_f64.wgsl`: `enable f64;` directive not supported by naga parser
- **All existing tests pass**: 746 lib tests, 0 clippy warnings (pedantic+nursery), 0 regressions from pin bump
- **New binary**: `validate_toadstool_s71_gpu_stats` (11/11 PASS)

---

## Part 1: ToadStool S71 Changes Reviewed

### ComputeDispatch Migration

ToadStool S71 introduced a `ComputeDispatch` builder that replaces manual BGL/BG/pipeline/encoder boilerplate:

```rust
// ~5-10 lines instead of ~80
ComputeDispatch::new(&device, "op_name")
    .shader(SHADER_SOURCE, "main")
    .f64()
    .storage_read(0, &input_buf)
    .storage_rw(1, &output_buf)
    .uniform(2, &params_buf)
    .dispatch(wg_count, 1, 1)
    .submit();
```

- 66 of ~250 ops migrated (~184 remaining)
- ~9,000 lines removed
- API is transparent to consumers — neuralSpring required zero changes

### DF64 Transcendentals

`df64_transcendentals.wgsl` now provides f64 precision on f32-native GPUs:

| Function | Technique |
|----------|-----------|
| `sqrt_df64` | Newton–Raphson |
| `exp_df64` | Cody–Waite + Horner degree-6 |
| `log_df64` | atanh + Horner degree-5 |
| `sin_df64` / `cos_df64` | Cody–Waite π/2 reduction + minimax |
| `gamma_df64` | Lanczos g=7 + reflection |
| `erf_df64` | Abramowitz & Stegun 7.1.26 rational |
| `tanh_df64`, `sinh_df64`, `cosh_df64` | exp-based |
| `atan_df64`, `asin_df64`, `acos_df64`, `atan2_df64` | Argument reduction |
| `pow_df64` | `exp(b * log(a))` |

### New GPU Stats Ops

| Op | Shader | Status from neuralSpring |
|----|--------|--------------------------|
| `KimuraGpu::dispatch()` | `kimura_fixation_f64.wgsl` | PASS — CPU↔GPU parity 1.11e-16 |
| `JackknifeMeanGpu::dispatch()` | `jackknife_mean_f64.wgsl` | BLOCKED — `bitcast<f64>` naga bug |
| `HargreavesBatchGpu::dispatch()` | `hargreaves_batch_f64.wgsl` | BLOCKED — `enable f64` naga bug |
| `HistogramGpu::dispatch()` | `histogram_f64.wgsl` | PASS — correct bins/counts/balance |
| `BootstrapMeanGpu::dispatch()` | `bootstrap_mean_f64.wgsl` | Not yet exercised (likely `bitcast<f64>` same pattern) |

### Cleanup

- Hardcoded primal names → `primals::*` constants
- `jsonrpc_server.rs`: 904 → 628 lines
- `network_config/types.rs`: 859 → 7 domain submodules
- `layer_adaptation.rs`: 842 → module directory
- `runtime_discovery.rs`: 849 → module directory
- 4 reducible unsafe blocks addressed (45 remain, all documented FFI/hardware)

---

## Part 2: Upstream Shader Bugs

### Bug 1: `bitcast<f64>(vec2<u32>())` fails after DF64 emulation

**File**: `crates/barracuda/src/shaders/stats/jackknife_mean_f64.wgsl`

```wgsl
let full_sum = bitcast<f64>(vec2<u32>(params.full_sum_lo, params.full_sum_hi));
let leave_mean = (full_sum - data[idx]) / (n_f - 1.0);
//               ^^^^^^^^^^^^^^^^^^^^^ naga: Operation Subtract can't work
```

**Root cause**: When `ComputeDispatch::f64()` transforms the shader for DF64 emulation, `bitcast<f64>` produces a type incompatible with the transformed `data[idx]` type.

**Fix suggestion**: Pass `full_sum` through a storage buffer instead of packing into uniform `u32` halves, or add a DF64-aware bitcast transform to the ComputeDispatch pipeline.

**Affected ops**: `JackknifeMeanGpu`, likely `BootstrapMeanGpu` and any other shader using `bitcast<f64>`.

### Bug 2: `enable f64;` directive not supported by naga

**File**: `crates/barracuda/src/shaders/science/hargreaves_batch_f64.wgsl`

```wgsl
enable f64;  // naga parser: "expected global item, found 'enable'"
```

**Root cause**: naga's WGSL frontend does not implement the `enable` directive from the WGSL spec extensions. The Kimura shader works because it omits `enable f64;` — naga infers f64 support from `array<f64>` type declarations.

**Fix**: Remove the `enable f64;` line. If `array<f64>` and `f64(...)` work without it (as proven by `kimura_fixation_f64.wgsl`), the directive is unnecessary.

---

## Part 3: neuralSpring Quality Metrics

| Metric | Value |
|--------|-------|
| Lib tests | **746** |
| Integration tests | 9 |
| Forge tests | 43 |
| Validation binaries | **219** (+1 S101) |
| validate_all | 200/200 |
| Clippy warnings | **0** (pedantic + nursery) |
| Unsafe code | **0** |
| Bare unwrap | **0** in library |
| Mocks in production | **0** |
| Files > 1000 LOC | **0** |
| SPDX headers | **100%** |
| Named tolerances | 139+ |
| External deps | 9 (all pure Rust) |
| ToadStool pin | **`8dc01a37`** |

---

## Part 4: BarraCUDA Usage Summary (S101)

| Category | Count |
|----------|-------|
| Import sites | 130+ across 208 files |
| Submodules used | 20+ |
| Function rewires | 44 upstream |
| WGSL shaders absorbed | 21 + 15 coralForge df64 |
| CPU→GPU dispatch ops | 47 (~97% of production math) |
| **New GPU stats ops** | 2 validated (Kimura, Histogram), 2 blocked (Jackknife, Hargreaves) |

---

## Part 5: Recommended Actions for ToadStool

### Priority 1: Fix `bitcast<f64>` in DF64 pipeline

The `ComputeDispatch::f64()` transform path needs to handle `bitcast<f64>(vec2<u32>())`. Options:
1. Transform `bitcast<f64>(vec2<u32>(lo, hi))` → DF64 pair construction
2. Provide a helper function `df64_from_u32_pair(lo, hi)` in the DF64 library

### Priority 2: Remove `enable f64;` from shaders

The `enable f64;` directive is unnecessary — naga handles `array<f64>` and `f64()` without it. Simply strip the line from `hargreaves_batch_f64.wgsl` and any other shaders that use it.

### Priority 3: Continue ComputeDispatch migration

66/250 ops migrated. The remaining ~184 ops represent ~14,000+ lines of boilerplate that can be replaced.

---

*neuralSpring Session 101 — ToadStool S71 GPU stats parity validation + upstream shader bug reports.*
*V68 supersedes V67. Archive V67.*
