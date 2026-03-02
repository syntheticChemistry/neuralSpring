# neuralSpring → ToadStool/BarraCUDA Handoff V70 — FFT Fix + Full Green + Shader Compat

**Date**: March 2, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Session 104 — Full validation chain 202/202, 3 barracuda fixes evolved locally, NUCLEUS Tower fix, V70 handoff
**Supersedes**: V69 (Nautilus Bridge + BarraCUDA Review)

---

## Executive Summary

- **202/202 validate_all PASS** (0 FAIL) — up from 197/202. Full green across all 9 tiers (Py → Rs → bC → gT → mF → gP → xD → mH → mG)
- **3 barracuda fixes evolved locally** for ToadStool absorption: FFT buffer selection, `enable f64;` naga strip, `asin_df64` iterative confirmation
- **Python baseline drift check**: 39/39 PASS — zero drift detected across all control experiments
- **90.49% line coverage** (llvm-cov, target 90%), 753 lib tests, 0 clippy (pedantic+nursery), 0 unsafe
- **NUCLEUS Tower validated**: 22/22 + 29/29 PASS — JSON-RPC primal with 11 capabilities, evoformer, structure module, GPU dispatch

---

## Part 1: BarraCUDA Fixes for Absorption

### Fix 1: FFT Ping-Pong Buffer Selection (P0 — correctness)

**File**: `crates/barracuda/src/ops/fft/fft_1d.rs`

**Bug**: After the Cooley-Tukey butterfly loop, `current_input` and `current_output` are swapped each stage. The final buffer selection used `is_multiple_of(2)` to choose between them — this was wrong for odd-stage FFTs (e.g., N=8, 3 stages), reading from the stale intermediate buffer.

**Symptoms**: Parseval's theorem off by 2×, delta→constant wrong, cosine energy leaking. 5 of 24 FFT checks failing.

**Fix**: Always read from `current_input` after the swap loop, since each swap puts the last-written buffer into `current_input`.

```rust
// Before (wrong for odd num_stages):
let final_buffer = if num_stages.is_multiple_of(2) {
    current_input
} else {
    current_output
};

// After (correct — swap always puts result in current_input):
let final_buffer = current_input;
```

**Validation**: barracuda_fft 24/24 PASS (was 19/24).

**ToadStool action**: Absorb this one-line fix into `crates/barracuda/src/ops/fft/fft_1d.rs`.

---

### Fix 2: `enable f64;` Naga Compatibility (P0 — shader compilation)

**File**: `crates/barracuda/src/shaders/precision/mod.rs` (`ShaderTemplate::for_driver_auto`)

**Bug**: ~30 WGSL shaders use `enable f64;` at the top (WGSL extension directive). Naga's WGSL parser doesn't support this directive — it handles f64 via capability flags instead. When the SovereignCompiler SPIR-V path fails for a given shader, the fallback to raw WGSL compilation panics on `enable f64;`.

**Symptoms**: `validate_gpu_pipeline_wright_fisher` panicked during `WrightFisherGpu::new()` → `compile_shader_f64`. Also affected `hargreaves_batch_f64.wgsl` (reported in V68).

**Fix**: Strip `enable f64;` lines in `for_driver_auto` before any compilation path.

```rust
pub fn for_driver_auto(shader_body: &str, needs_exp_log_workaround: bool) -> String {
    // Strip `enable f64;` — naga handles f64 via capability flags, not directives.
    let stripped = shader_body
        .lines()
        .filter(|l| l.trim() != "enable f64;")
        .collect::<Vec<_>>()
        .join("\n");
    let substituted = Self::substitute_fossil_f64(&stripped);
    // ... rest unchanged
}
```

**Validation**: gpu_pipeline_wright_fisher 4/4 PASS (was panic). Also unblocks `hargreaves_batch_f64.wgsl`.

**ToadStool action**: Absorb into `crates/barracuda/src/shaders/precision/mod.rs`. Consider also stripping in `compile_shader_df64` for defense-in-depth.

---

### Fix 3: `asin_df64` Recursive → Iterative (confirmed)

**File**: `crates/barracuda/src/shaders/math/df64_transcendentals.wgsl`

**Status**: Already fixed in working tree (iterative form with `x_in` parameter, negate flag, explicit large-reduction branch). Confirmed that coral forge GPU pipeline compiles and validates: 16/16 PASS (SDPA scores, attention apply, IPA scores, backbone update, torsion angles).

**Note**: The `bitcast<f64>(vec2<u32>())` issue in `jackknife_mean_f64.wgsl` (from V68) remains open — that's a different class of naga DF64 incompatibility.

**ToadStool action**: Commit the iterative `asin_df64` if not already committed. Review other `df64_transcendentals.wgsl` functions for similar recursion patterns.

---

## Part 2: Validation Chain Results

### Full Tier Validation

| Tier | Description | Status |
|------|-------------|--------|
| Py | Python baselines (39 experiments) | 39/39 PASS, zero drift |
| Rs | Rust native (25 papers) | ALL PASS |
| bC | BarraCUDA CPU (24 papers + primitives) | ALL PASS |
| gT | GPU Tensor (RTX 4070 Vulkan) | 86/86 PASS |
| mF | metalForge WGSL (42 shaders) | ALL PASS |
| gP | GPU Pipeline (14 validators) | ALL PASS |
| xD | Cross-dispatch (CPU↔GPU parity) | ALL PASS |
| mH | Mixed-hardware (NPU↔GPU↔CPU) | 47/47 + 43/43 PASS |

### Key Validators

| Binary | Checks | Result |
|--------|--------|--------|
| `validate_all` | 202 binaries | **202/202 PASS** |
| `validate_barracuda_fft` | 24 | 24/24 PASS (was 19/24) |
| `validate_gpu_pipeline_wright_fisher` | 4 | 4/4 PASS (was panic) |
| `validate_coral_forge_gpu_pipeline` | 16 | 16/16 PASS (was panic) |
| `validate_nucleus_tower` | 22 | 22/22 PASS |
| `validate_biomeos_spectral` | 29 | 29/29 PASS |
| `validate_mixed_hardware_dispatch` | 47 | 47/47 PASS |
| `validate_publication_mixed_hardware` | 43 | 43/43 PASS |

### Quality Gates

| Check | Result |
|-------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets --all-features -D warnings` | 0 warnings |
| `cargo doc --no-deps` | 218 files generated |
| `cargo test --lib` | 753 passed, 0 failed |
| `cargo llvm-cov --lib` | 90.49% line coverage |

---

## Part 3: Lessons for BarraCUDA Evolution

### 3.1 Naga WGSL Compatibility

The `enable f64;` issue affects ~30 shaders. The fix (stripping in `for_driver_auto`) is surgical but the root cause is naga's incomplete WGSL extension support. Options for upstream:

1. **Current fix** (strip in Rust before compilation) — works, zero shader source changes needed
2. **Shader source migration** — remove `enable f64;` from all `.wgsl` files entirely (they're already compiled via `compile_shader_f64` which enables f64 at the capability level)
3. **Naga contribution** — teach naga to parse and ignore `enable` directives it doesn't need

Recommendation: Option 2 (remove from shader sources) + Option 1 (keep strip as safety net).

### 3.2 FFT Buffer Ping-Pong

The Cooley-Tukey FFT implementation uses a classic ping-pong pattern but the final buffer selection was fragile. Consider:

- Adding a debug assertion: `debug_assert_eq!(final_buffer.id(), last_written_buffer.id())`
- Adding an explicit `current` variable that always tracks the last-written buffer without relying on swap parity
- Unit tests for N=4 (2 stages, even), N=8 (3 stages, odd), N=16 (4 stages, even) to catch both paths

### 3.3 DF64 Recursion

WGSL forbids recursion entirely — no recursive function calls, even depth-1. Any math library function that uses recursion (like `asin_df64`'s negation/reduction pattern) must be manually inlined. Audit all `df64_transcendentals.wgsl` functions for this pattern.

### 3.4 NUCLEUS Tower Integration

The neuralSpring primal is fully validated as a biomeOS provider with 11 capabilities. The socket path uses `CARGO_PKG_NAME` (`neural-spring`) which includes a hyphen — validators and other primals must use the hyphenated form.

---

## Part 4: Open Items (unchanged from V69)

- `jackknife_mean_f64.wgsl`: `bitcast<f64>(vec2<u32>())` incompatible with DF64 transform — needs redesign
- L-BFGS optimizer: not yet in barracuda (neuralSpring uses Nelder-Mead)
- `memmap2` for safetensors: blocked by `forbid(unsafe_code)` — awaiting safe mmap abstraction

---

## Reproduction

```bash
cd neuralSpring

# Full validation chain
cargo build --release
cargo run --release --bin validate_all          # 202/202 PASS

# Specific fixes
cargo run --release --bin validate_barracuda_fft              # 24/24 (FFT fix)
cargo run --release --bin validate_gpu_pipeline_wright_fisher  # 4/4 (enable f64 fix)
cargo run --release --bin validate_coral_forge_gpu_pipeline    # 16/16 (asin_df64)

# NUCLEUS Tower
cargo run --release --features primal --bin validate_nucleus_tower      # 22/22
cargo run --release --features primal --bin validate_biomeos_spectral   # 29/29

# Quality gates
cargo clippy --all-targets --all-features -- -D warnings  # 0 warnings
cargo test --lib                                          # 753 PASS
cargo llvm-cov --lib --summary-only                       # 90.49%

# Python drift check
bash control/check_drift.sh                               # 39/39 PASS
```

---

Unidirectional handoff — no response expected.
