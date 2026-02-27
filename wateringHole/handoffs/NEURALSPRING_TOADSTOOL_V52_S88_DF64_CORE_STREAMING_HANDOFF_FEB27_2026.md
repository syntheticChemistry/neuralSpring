# neuralSpring → ToadStool/BarraCUDA Handoff: V52 df64 Core Streaming

**Date:** February 27, 2026
**From:** neuralSpring Session 88
**To:** ToadStool/BarraCUDA team
**ToadStool pin:** S68 (`f0feb226`)
**neuralSpring:** 623 lib + 43 forge + 9 integration tests, 172 binaries, 158/158 PASS

---

## Executive Summary

- **All 15 sovereign folding WGSL shaders evolved** to hotSpring/ToadStool df64
  core streaming pattern: f64 buffer I/O → df64 compute on FP32 cores → f64 output
- `Fp64Strategy::Hybrid` auto-detected on RTX 4070 (1:64 FP64:FP32 ratio)
- Two precision tiers validated: arithmetic 3.6e-8 to 5.6e-7, transcendental
  1.7e-4 to 3.4e-4 (vs native f64 CPU reference)
- GPU wrapper: `create_buffer_f64`, `upload_f64`, `compile_shader_f64_hybrid`
- All quality gates green: 158/158 validate_all, 37/37 sovereign folding GPU,
  0 clippy, 0 doc warnings

---

## Part 1: How neuralSpring Uses BarraCUDA

### Upstream Dependencies (lean — we use barracuda for this)

| barracuda Module | neuralSpring Usage | Validation |
|------------------|-------------------|------------|
| `device::WgpuDevice` | GPU adapter, shader compilation, buffer ops | All GPU binaries |
| `ops::lattice::su3::{WGSL_DF64_CORE, WGSL_DF64_TRANSCENDENTALS}` | df64 preamble for `compile_shader_f64_hybrid` | 37/37 sovereign GPU |
| `device::driver_profile::{GpuDriverProfile, Fp64Strategy}` | Hardware-aware precision strategy | Auto-detect in GPU validator |
| `WgpuDevice::compile_shader_f64()` | f64 shader compilation with polyfills | All f64 shaders |
| `WgpuDevice::create_buffer_f64()`, `read_buffer_f64()` | f64 GPU buffer allocation/readback | All f64 I/O paths |
| `tensor::Tensor` | Tensor API (matmul, softmax, activations) | 90+ checks |
| `stats::*` | variance, pearson, shannon, hill, mae, fit_* | 39+ function rewires |
| `linalg::*` | eigh_f64, solve_f64, cholesky, SVD, graph | 17 linalg checks |
| `special::*` | gamma, erf, bessel, chi_squared | 26 special checks |
| `numerical::*` | rk45_solve, numerical_hessian | ODE/optimization |
| `ops::bio::*` | DiversityFusion, BatchFitness, Pairwise*, HMM, Gillespie | 74+ upstream checks |
| `staging::StatefulPipeline` | GPU pipeline batching | 10/10 stateful PASS |
| `dispatch::*` | CPU↔GPU routing | ~97% GPU promotion |
| `validation::ValidationHarness` | Structured pass/fail reporting | All 158 binaries |

### Local Implementation (neuralSpring owns the domain, barracuda owns the compute)

| neuralSpring Module | Why Local | Absorption Candidate? |
|--------------------|-----------|:---------------------:|
| `sovereign_folding.rs` | AlphaFold2 Evoformer CPU reference | No — domain physics |
| `structure_module.rs` | AlphaFold2 structure module CPU reference | No — domain physics |
| 15 sovereign folding WGSL shaders | df64 core streaming shaders for protein folding | **Yes** — see Part 2 |
| `wdm_*.rs` (5 modules) | WDM surrogate domain models | Pending `barracuda::nn` |
| `weight_spectral.rs`, `information_flow.rs`, etc. | baseCamp research modules | No — domain research |
| `evolved/mha.rs` | Thin wrapper — can retire when using MHA directly | Low priority |

### Evolution Principle

> neuralSpring owns domain science (ML, protein folding, WDM surrogates).
> barracuda owns compute primitives (df64, tensors, dispatch, pipelines).
> Local code that becomes a reusable compute primitive gets absorbed.
> Local code that encodes domain-specific science stays local.

---

## Part 2: What to Absorb — df64 Core Streaming Shaders

### 2.1 Sovereign Folding Shaders (15 — Ready for Absorption)

All 15 shaders follow the identical three-zone df64 core streaming pattern:

```
Zone 1 (Load):    let val = df64_from_f64(input[idx]);
Zone 2 (Compute): let result = df64_mul(a, b); / df64_add / sqrt_df64 / exp_df64
Zone 3 (Store):   output[idx] = df64_to_f64(result);
```

| Shader | Algorithm | Precision Tier | Max GPU-CPU Diff |
|--------|-----------|----------------|------------------|
| `gelu_f64.wgsl` | Pointwise GELU | Transcendental | 3.41e-4 |
| `layer_norm_f64.wgsl` | LayerNorm (mean/var/normalize) | Arithmetic | 5.58e-7 |
| `softmax_f64.wgsl` | Row-wise softmax (3-pass) | Transcendental | 2.92e-4 |
| `sdpa_scores_f64.wgsl` | QKᵀ/√d (attention pass 1) | Arithmetic | 6.76e-8 |
| `attention_apply_f64.wgsl` | Σ weights × V (attention pass 3) | Arithmetic | 6.89e-8 |
| `triangle_mul_outgoing_f64.wgsl` | Algorithm 11 | Arithmetic | 3.10e-7 |
| `triangle_mul_incoming_f64.wgsl` | Algorithm 12 | Arithmetic | 4.66e-7 |
| `triangle_attention_f64.wgsl` | Algorithms 13-14 | Arithmetic | 1.54e-7 |
| `outer_product_mean_f64.wgsl` | MSA → pair (OPM) | Arithmetic | 6.43e-8 |
| `msa_row_attention_scores_f64.wgsl` | Row attn + pair bias | Arithmetic | 1.06e-7 |
| `msa_col_attention_scores_f64.wgsl` | Col attn (no bias) | Arithmetic | 9.57e-8 |
| `sigmoid_f64.wgsl` | Sigmoid gate (sign-branch stable) | Transcendental | (CPU validated) |
| `ipa_scores_f64.wgsl` | IPA (SE(3)-equivariant, 3-term) | Arithmetic | 3.40e-7 |
| `backbone_update_f64.wgsl` | Frame composition (quat→rot) | Arithmetic | 3.59e-8 |
| `torsion_angles_f64.wgsl` | Fused ResNet + unit circle norm | Arithmetic | 1.10e-7 |

**Compilation path**: All shaders compile via `compile_shader_f64_hybrid()`:
```rust
let combined = format!("{WGSL_DF64_CORE}\n{WGSL_DF64_TRANSCENDENTALS}\n{source}");
self.wgpu_device.compile_shader_f64(&combined, Some(label))
```

**Absorption suggestion**: These 15 shaders could become
`barracuda::ops::folding::*` or `barracuda::ops::attention::*` — the
primitives (GELU, LayerNorm, softmax, SDPA, triangle updates) are universal
ML building blocks, not specific to AlphaFold2.

### 2.2 `compile_shader_f64_hybrid` Pattern

neuralSpring manually prepends `WGSL_DF64_CORE` + `WGSL_DF64_TRANSCENDENTALS`
before calling `compile_shader_f64`. This is the same pattern hotSpring uses.

**Suggestion**: Add `WgpuDevice::compile_shader_df64_streaming(source, label)`
that encapsulates the preamble concatenation. This would eliminate duplicate
code across Springs.

### 2.3 Outstanding from V51 (Still Pending)

1. **`barracuda::nn::SimpleMLP`** — JSON weight loading + forward pass (3 WDM users)
2. **`barracuda::nn::LstmReservoir`** — LSTM reservoir with pooled readout (nW-03)
3. **`barracuda::nn::EsnClassifier`** — ESN reservoir classifier (nW-05)
4. **Hamming 20.85× regression** — `PairwiseHammingGpu` 200×500 f64 path
5. **Public f32 shader constants** for integer-distance ops
6. **`barracuda::testing::GpuTestHarness`** — shared device + mutex pattern
7. **Variance convention docs** (`stats::variance` ÷(N-1) vs `dispatch` ÷N)

---

## Part 3: Evolution Discoveries — df64 Precision Hierarchy

### 3.1 Two-Tier Tolerance Structure

df64 core streaming creates two distinct precision tiers:

| Tier | Operations | Tolerance | Observed Range | Bottleneck |
|------|-----------|-----------|----------------|------------|
| **Arithmetic** | dot products, matmul, accumulate, `sqrt_df64` | 1e-6 | 3.6e-8 to 5.6e-7 | f32 FMA error tracking in `two_prod`/`two_sum` |
| **Transcendental** | `exp_df64`, `tanh_df64` (degree-6 Horner) | 5e-4 | 1.7e-4 to 3.4e-4 | Polynomial approximation truncation |

Both tiers are 100-10000x better than pure f32 (~1e-3 to 1e-2).

### 3.2 Full Precision Ladder

| Level | Mantissa Bits | Decimal Digits | Source | Use Case |
|-------|--------------|----------------|--------|----------|
| fp16 | 10 | ~3 | Native | Inference |
| bf16 | 7 | ~2 | Native | Training dynamic range |
| f32 | 23 | ~7 | Native | Standard GPU compute |
| **df64 (fp48)** | **~48** | **~14** | **Emulated (f32 pairs)** | **Scientific validation** |
| f64 | 52 | ~15.9 | Native (data-center) | Gold standard |

### 3.3 WGSL Gotcha: No Ternary If

WGSL does not support Rust-style ternary `if` expressions:
```wgsl
// INVALID: let x = if cond { a } else { b };
// VALID:
var x = b;
if cond { x = a; }
```
This tripped `torsion_angles_f64.wgsl` during development.

### 3.4 Transcendental Precision Opportunity

The `exp_df64` and `tanh_df64` implementations in `df64_transcendentals.wgsl`
use degree-6 Horner polynomials. Higher-degree polynomials (degree-10 or
degree-12) or range-reduction + Padé approximants could close the gap between
transcendental and arithmetic precision tiers. This is a ToadStool-level
optimization — all downstream Springs would benefit.

---

## Part 4: Updated Metrics

| Metric | V51 | V52 | Delta |
|--------|-----|-----|-------|
| validate_all | 156/156 | 158/158 | +2 |
| Sovereign folding GPU | 37/37 (f32 buffers) | 37/37 (f64 buffers, df64 compute) | Architecture evolved |
| Shader architecture | f32 buffer + ad-hoc df64 | f64 buffer + three-zone df64 streaming | Aligned to hotSpring |
| Max GPU-CPU diff (arithmetic) | — | 3.6e-8 to 5.6e-7 | New metric |
| Max GPU-CPU diff (transcendental) | — | 1.7e-4 to 3.4e-4 | New metric |
| Total checks | 2450+ | 2480+ | +30 |

---

## Part 5: Validation

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --workspace -- -D warnings` | 0 warnings |
| `cargo test --workspace` | **675/675 PASS** |
| `validate_all` | **158/158 PASS** |

---

## Part 6: Verification Commands

```bash
cd /home/eastgate/Development/ecoPrimals/neuralSpring
cargo test --workspace                     # 675/675 PASS
cargo clippy --workspace -- -D warnings   # 0 warnings
cargo run --release --bin validate_all    # 158/158 PASS
cargo run --release --bin validate_sovereign_folding_gpu  # 37/37 (df64)
```

---

## Part 7: Barracuda API Evolution Suggestions

### 7.1 `compile_shader_df64_streaming` (High Priority)

Both neuralSpring and hotSpring manually prepend `WGSL_DF64_CORE` +
`WGSL_DF64_TRANSCENDENTALS` before calling `compile_shader_f64`. This should
be a first-class API: `WgpuDevice::compile_shader_df64_streaming(source, label)`.

### 7.2 Typed df64 Ops (Medium Priority)

The 15 sovereign folding shaders demonstrate a set of universal df64 building
blocks that could be typed ops in barracuda:

- `barracuda::ops::gelu_df64` — pointwise GELU on f64 buffers
- `barracuda::ops::layer_norm_df64` — layer normalization on f64 buffers
- `barracuda::ops::softmax_df64` — row-wise softmax on f64 buffers
- `barracuda::ops::sdpa_df64` — scaled dot-product attention pipeline
- `barracuda::ops::matmul_df64` — general df64 matrix multiply

These are not domain-specific — they are universal ML primitives that happen
to run at fp48 precision on consumer GPUs.

### 7.3 Transcendental Precision Improvement (Low Priority)

Upgrade `exp_df64` and `tanh_df64` from degree-6 to degree-10+ Horner
polynomials. Current ~3.4e-4 max error could potentially reach ~1e-8,
closing the gap with arithmetic precision. Would benefit all Springs using
df64 transcendentals.

---

*neuralSpring V52 handoff — February 27, 2026, Session 88. AGPL-3.0-or-later.*
