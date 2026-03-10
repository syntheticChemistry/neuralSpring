<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# neuralSpring V95 → toadStool/barraCuda Evolution Handoff

| Field | Value |
|-------|-------|
| **Date** | 2026-03-10 |
| **From** | neuralSpring S142 (1048 lib + 71 forge + 9 integration tests, 233 binaries, 55/55 dispatch parity) |
| **To** | barraCuda team, toadStool team |
| **Supersedes** | V94 (S142 upstream rewire), V93 (S139 absorption) |
| **Synced against** | barraCuda `83aa08a`, toadStool S142 (`a86bc546`), coralReef Iteration 29 (`2779c88`) |
| **License** | AGPL-3.0-or-later |

---

## Executive Summary

neuralSpring S142 completes the upstream rewire to modern barraCuda/toadStool/coralReef
and includes a **critical bug discovery and local fix** for the `enable f64;` PTXAS
silent-zero regression on Ada Lovelace. This handoff documents:

1. The `enable f64;` fix (ready for upstream absorption)
2. A separate `HmmBatchForwardF64` shader/binding mismatch (upstream bug report)
3. DF64 precision characterization for large-N correlation
4. Cross-spring shader evolution observations
5. Remaining absorption opportunities

---

## Part 1: `enable f64;` PTXAS Silent-Zero Regression

### Root Cause

On NVIDIA Ada Lovelace (SM89, RTX 40xx) with the proprietary driver via Vulkan,
WGSL shaders containing `enable f64;` that are compiled through
`get_or_compile_shader_f64_native` in `pipeline_cache.rs` produce **silently
broken** GPU code: the shader compiles, the pipeline creates, the dispatch
executes, but every output buffer is 0.0.

naga resolves f64 support from device capability flags (`SHADER_F64`), not from
WGSL directives. The directive is unnecessary and harmful on this hardware.

### Affected Operations (pre-fix)

| Operation | Symptom | After Fix |
|-----------|---------|-----------|
| `VarianceF64` (fused) | 0.0 | Correct (mean=3.0, var=2.0) |
| `CorrelationF64` (fused 5-acc) | 0.0 | Correct (r=0.9987) |
| `MatrixCorrelationF64` | 0.0 | Correct (1.0) |
| `InterPopAfVariance` | 1.34e8 (garbage) | Correct (0.00879) |
| `ThermalDiversityCorr` | 0.0 | Correct (0.867) |

### Fix (implemented locally, ready for upstream)

In `pipeline_cache.rs::get_or_compile_shader_f64_native`, strip `enable f64;`
before passing the source to `create_shader_module`:

```rust
let stripped: Cow<'_, str> = if source.contains("enable f64;") {
    source.lines()
        .filter(|l| l.trim() != "enable f64;")
        .collect::<Vec<_>>()
        .join("\n")
        .into()
} else {
    source.into()
};
```

This matches the behavior already present in `compile_shader_f64()`,
`compile_shader_df64()`, `ShaderTemplate::for_driver_auto()`, and
`ShaderTemplate::for_driver_profile()`.

### Diagnostic Binary

`neuralSpring/src/bin/diagnose_f64_regression.rs` — reproduces the issue and
validates the fix. Shows WITH vs WITHOUT `enable f64;` on echo, DF64 sum, and
variance shaders. Can be run on any hardware.

### Validation Results (post-fix)

```
validate_barracuda_dispatch_parity: 55/55 PASS (was 48/55)
validate_toadstool_s79_rewire:     19/19 PASS
validate_modern_cross_spring:      68/68 PASS
fused_ops_healthy:                 true (was false)
cargo test --lib:                  1048 passed, 0 failed
```

See also: `NEURALSPRING_ENABLE_F64_FIX_HANDOFF_MAR10_2026.md` for full
diagnostic evidence.

---

## Part 2: `HmmBatchForwardF64` Shader/Binding Mismatch (Upstream Bug)

`hmm_forward_f64.wgsl` is a **per-step** shader (5 bindings: params, initial,
transition, emission, alpha). But `HmmBatchForwardF64::dispatch()` passes **7
bindings** for a batch API (adding observation_indices and log_lik_out). The
mismatched dispatch silently produces 0.0 because `log_lik_out` (binding 6) is
never written by the shader.

**neuralSpring workaround**: `hmm_forward_chain_gpu` in `src/gpu_ops/bio/hmm.rs`
detects `Ok(0.0)` from the fused path for non-empty sequences and falls back to
the per-step Tensor-based implementation, which works correctly.

**Recommended upstream fix**: Either update the shader to accept 7 bindings and
perform batch processing, or update the Rust dispatch to match the per-step
shader's 5-binding interface.

---

## Part 3: DF64 Precision Characterization

DF64 correlation (~48-bit mantissa via f32-pair emulation) on 1008 elements
diverges from CPU f64 by ~1.7e-5. This is **expected behavior** for the
5-accumulator DF64 reduction:

| Metric | CPU f64 | GPU DF64 | Diff |
|--------|---------|----------|------|
| Pearson r (1008 pts) | -0.005714 | -0.005697 | 1.7e-5 |

neuralSpring documents this via `tolerances::GPU_DF64_TRANSCENDENTAL` (5e-4).
No upstream action needed — this is inherent to DF64 arithmetic.

---

## Part 4: Cross-Spring Shader Evolution Observations

### What neuralSpring Benefits From

| Source Spring | Shader/Primitive | neuralSpring Benefit |
|--------------|-----------------|---------------------|
| **hotSpring** | `fused_welford_f64` (precision Welford) | Numerically stable variance in spectral analysis, Anderson IPR |
| **hotSpring** | `df64_core.wgsl` (double-float library) | All DF64 ops — correlation, chi², KL divergence |
| **hotSpring** | `batched_eigh_nak_optimized_f64.wgsl` | GPU eigendecomposition for Anderson, spectral commutativity |
| **wetSpring** | `fused_entropy_f64` (Shannon entropy) | Diversity indices, ecological dynamics, agent coordination |
| **wetSpring** | `hmm_forward_f64.wgsl` (HMM forward) | Phylogenetics (Paper 016), introgression (Paper 018) |
| **wetSpring** | `fst_variance_decomposition` | Meta-population genetics (Paper 025) |
| **neuralSpring** → upstream | `xoshiro128ss.wgsl` (GPU PRNG) | Stochastic simulations — ready for absorption |
| **neuralSpring** → upstream | `logsumexp_reduce.wgsl` | HMM/softmax numerical stability — ready for absorption |

### Cross-Spring Evolution Timeline

The precision shaders evolved across springs in a clear progression:

1. **hotSpring** pioneered `df64_core.wgsl` for plasma physics (lattice QCD
   precision requirements)
2. **wetSpring** adapted DF64 for biological statistics (Shannon entropy, HMM)
3. **neuralSpring** validated the full stack across 26 papers and 6 baseCamp
   sub-theses, discovering the `enable f64;` regression in the process

This cross-spring evolution pattern validates the ecoPrimals design: each Spring
independently discovers the same primitives need the same precision engineering.

---

## Part 5: Remaining Absorption Opportunities

### For barraCuda Team

| Priority | Item | Description |
|----------|------|-------------|
| **P0** | `enable f64;` stripping | Absorb the `pipeline_cache.rs` fix (see Part 1) |
| **P0** | `HmmBatchForwardF64` binding fix | Fix shader/dispatch mismatch (see Part 2) |
| **P1** | `xoshiro128ss.wgsl` | GPU PRNG — validated 5/5, enables stochastic GPU |
| **P1** | `logsumexp_reduce.wgsl` | Numerical stability for HMM/softmax — validated 5/5 |
| **P2** | `stencil_cooperation.wgsl` | QS spatial modeling — validated 3/3 |
| **P2** | `rk45_adaptive.wgsl` | ODE integration — validated 6/6 |
| **P2** | `wright_fisher_step.wgsl` | Population genetics — validated 4/4 |

### For toadStool Team

| Priority | Item | Description |
|----------|------|-------------|
| **P1** | GPU training infrastructure | No GPU autograd, no composable layer trait, CPU-only optimizers |
| **P2** | Kokkos/cuBLAS benchmark data | `bench_kokkos_parity` harness exists (9 ops), needs comparison data |
| **P2** | StatefulPipeline for HMM chains | Reduce CPU→GPU round-trips in multi-step algorithms |

---

## Part 6: Metrics Snapshot

| Metric | Value |
|--------|-------|
| Library tests | 1048 |
| Forge tests | 71 |
| Integration tests | 9 |
| Validation binaries | 233 |
| validate\_all | 220/220 PASS |
| Dispatch parity | 55/55 PASS (was 48/55 pre-fix) |
| Cross-spring modern | 68/68 PASS |
| fused\_ops\_healthy | true (was false) |
| Line coverage (llvm-cov) | 92% |
| Clippy warnings | 0 (pedantic + nursery) |
| Doc warnings | 0 |
| Named tolerances | 80+ |
| Upstream rewires | 46 |
| WGSL shaders (metalForge) | 42 |
| CPU→GPU dispatch ops | 47 (~97%) |
| Files > 1000 LOC | 0 |
| Unsafe code | 0 |
| TODO/FIXME/MOCK/STUB | 0 |

---

## Quality Gates (Reproducible)

```bash
cargo fmt --check                                          # PASS
cargo clippy --all-targets --all-features -- -D warnings   # 0 warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps             # 0 warnings
cargo test --lib                                           # 1048/1048 PASS
cargo test --test integration                              # 9/9 PASS
cargo run --release --bin validate_all                     # 220/220 PASS
cargo run --release --bin validate_barracuda_dispatch_parity  # 55/55 PASS
cargo run --release --bin validate_modern_cross_spring        # 68/68 PASS
```

---

*This handoff is unidirectional: neuralSpring → ecosystem. No response expected.*
*License: AGPL-3.0-or-later*
