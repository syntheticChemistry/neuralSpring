# neuralSpring → barraCuda: `enable f64;` PTXAS Silent-Zero Fix

**Date**: 2026-03-10
**Spring**: neuralSpring
**Target**: barraCuda `pipeline_cache.rs`
**Severity**: Critical — silently produces zeros for ALL fused f64 GPU ops on Ada Lovelace

---

## Summary

On NVIDIA Ada Lovelace (SM89, RTX 40xx) with the proprietary driver via Vulkan,
WGSL shaders containing `enable f64;` that are compiled through the
`get_or_compile_shader_f64_native` path produce **silently broken** GPU code
that returns **zeros for all outputs**. The shader compiles without error, the
pipeline creates successfully, the dispatch executes, but every output buffer
value is 0.0.

Stripping `enable f64;` from the source before compilation fixes the issue
completely. naga resolves f64 support from device capability flags (`SHADER_F64`),
not from WGSL directives — the directive is unnecessary and harmful.

## Root Cause

```
WGSL source → naga WGSL parser → naga IR → SPIR-V backend → NVIDIA Vulkan driver → PTXAS
                                                                                      ↑
                                                                        `enable f64;` triggers
                                                                        buggy compilation path
                                                                        in complex shaders
```

1. `get_or_compile_shader_f64_native` in `pipeline_cache.rs` compiles shaders
   **raw** — no preprocessing, no directive stripping
2. The `fused_shader_for_device()` functions in variance/correlation ops prepend
   `enable f64;\n` to DF64 combined shaders
3. On Ada Lovelace + PTXAS, the `enable f64;` directive causes the compiled
   shader to silently return zeros for **all** outputs
4. Simple shaders (probes, 1-function) are not affected — only multi-function
   shaders (df64_core + reduction = ~20 functions) trigger the bug

## Affected Operations

| Operation | Symptom | After Fix |
|-----------|---------|-----------|
| `VarianceF64` (fused) | Returns 0.0 | Correct (mean=3.0, var=2.0) |
| `CorrelationF64` (fused 5-acc) | Returns 0.0 | Correct (r=0.9987) |
| `MatrixCorrelationF64` | Returns 0.0 | Correct (1.0) |
| `HmmBatchForwardF64` log_lik | Returns 0.0 | N/A (separate shader/binding mismatch) |
| `InterPopAfVariance` | Returns 1.34e8 (garbage) | Correct (0.00879) |
| `ThermalDiversityCorr` | Returns 0.0 | Correct (0.867) |

All ops routed through `create_f64_data_pipeline` → `get_or_compile_shader_f64_native`
with `source_is_f64()` == true.

## Diagnostic Evidence

```
=== Test 1: Minimal f64 echo shader ===
  echo WITH enable f64:    [0.000000, 0.000000]  ← BROKEN
  echo WITHOUT enable f64: [3.000000, 12.000000]  ← CORRECT

=== Test 2: DF64 sum shader ===
  df64_sum WITH enable f64:    0.000000  ← BROKEN
  df64_sum WITHOUT enable f64: 10.000000 ← CORRECT

=== Test 3: DF64 variance shader ===
  Raw (enable f64 present):  mean=0.0, var=0.0     ← BROKEN
  compile_shader_f64 path:   mean=3.0, var=2.0     ← CORRECT
  Manually stripped:         mean=3.0, var=2.0     ← CORRECT
```

Hardware: NVIDIA GeForce RTX 4070, Vulkan, proprietary driver, wgpu 28.

## Fix (1 line change + comment)

### `barraCuda/crates/barracuda/src/device/pipeline_cache.rs`

In `get_or_compile_shader_f64_native`, strip `enable f64;` before passing
the source to `create_shader_module`:

```rust
pub fn get_or_compile_shader_f64_native(
    &self, device: &Device, adapter_info: &wgpu::AdapterInfo,
    source: &str, label: Option<&str>,
) -> Arc<ShaderModule> {
    // ... cache lookup ...

    // Strip `enable f64;` — naga resolves f64 support from device capability
    // flags, not WGSL directives.  Leaving the directive in causes NVIDIA
    // PTXAS (Ada Lovelace / SM89) to silently produce broken shaders that
    // return zeros for all outputs.
    let stripped: Cow<'_, str> = if source.contains("enable f64;") {
        source.lines()
            .filter(|l| l.trim() != "enable f64;")
            .collect::<Vec<_>>()
            .join("\n")
            .into()
    } else {
        source.into()
    };
    let module = Arc::new(device.create_shader_module(ShaderModuleDescriptor {
        label,
        source: ShaderSource::Wgsl((&*stripped).into()),
    }));
    // ... cache insert ...
}
```

This matches the behavior already present in:
- `compile_shader_f64()` (compilation.rs, line 87)
- `compile_shader_df64()` (compilation.rs, line 134)
- `ShaderTemplate::for_driver_auto()` (precision/mod.rs, line 284)
- `ShaderTemplate::for_driver_profile()` (precision/mod.rs, line 313)

## Validation Results (post-fix)

```
validate_barracuda_dispatch_parity: 55/55 PASS, 0 FAIL  (was 48/55)
validate_toadstool_s79_rewire:     19/19 PASS, 0 FAIL
validate_modern_cross_spring:      68/68 PASS, 0 FAIL
cargo test --lib:                  1048 passed, 0 failed
fused_ops_healthy:                 true (was false)
```

## Separate Issues Found (not fixed by this patch)

### 1. `HmmBatchForwardF64` shader/binding mismatch

`hmm_forward_f64.wgsl` is a per-step shader (5 bindings) but
`HmmBatchForwardF64::dispatch()` passes 7 bindings for a batch API.
The mismatched dispatch silently produces zeros because `log_lik_out`
(binding 6) is never written. neuralSpring works around this by detecting
0.0 results from the fused path and falling through to the per-step path.

### 2. DF64 precision for large-N correlation

DF64 correlation (~48-bit mantissa) on 1008 elements diverges from CPU f64
by ~1.7e-5. This is expected for DF64 and documented via
`tolerances::GPU_DF64_TRANSCENDENTAL` (5e-4).

### 3. Bio op buffer binding issues

`PairwiseHammingGpu`, `PairwiseJaccardGpu`, and `SpatialPayoffGpu` produce
garbage values (1e35+). These use raw `create_shader_module` without
`enable f64;` — a separate buffer binding or data layout issue.

---

## Diagnostic Binary

`neuralSpring/src/bin/diagnose_f64_regression.rs` — standalone diagnostic
that reproduces the issue and validates the fix. Can be run on any hardware
to test the `enable f64;` compilation path.
