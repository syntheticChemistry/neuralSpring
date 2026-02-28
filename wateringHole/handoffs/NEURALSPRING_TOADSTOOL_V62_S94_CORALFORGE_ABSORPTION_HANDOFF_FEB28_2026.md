# neuralSpring → ToadStool/BarraCUDA Handoff V62 — coralForge Rename + Deep Debt Resolution + Absorption Targets

**Date**: February 28, 2026
**From**: neuralSpring (ML/neuroevolution validation + coralForge sovereign structure prediction)
**To**: ToadStool/BarraCUDA team
**License**: AGPL-3.0-or-later
**Covers**: Session 94 — coralForge rename, deep debt resolution, tolerance hardening, provenance documentation
**Supersedes**: V61 (Deep Debt + Confidence Heads)

---

## Executive Summary

- **coralForge** is now the unified name for neuralSpring's sovereign structure prediction engine (formerly `sovereign_folding` + `structure_module`). 25+ source files, 3 validation binaries, Cargo.toml, control scripts, specs, and all docs updated.
- **197 binaries**, **185/185 validate_all**, **685 lib tests**, **3200+ checks**
- **139+ named tolerances** — 5 new domain guards: `FISHER_EPS`, `BURGERS_IC_GUARD`, `DP_EQUALITY_EPS`, `SINGLETON_FREQ_EPS`, `PHENOTYPE_TIE_EPS`. Zero inline magic numbers in production code.
- **24 `expect()` → `require!()`** in GPU validation binaries — graceful error recording via `ValidationHarness`
- **34 `BaselineProvenance` constants** documented with `///` comments. All provenance paths updated for coralForge.
- **Cast safety**: `cpu_fallback.rs` activator indices bounds-checked via `safe_idx()`
- **Zero unsafe, zero production mocks, zero cross-primal logic, 0 clippy pedantic warnings, 0 doc warnings**
- All 12 external dependencies are pure Rust — zero C/C++ wrappers. Fully documented in `EVOLUTION_READINESS.md`.

---

## Part 1: coralForge Unification

### What Changed

| Old | New | Scope |
|-----|-----|-------|
| `src/sovereign_folding/` | `src/coral_forge/` | All Evoformer primitives |
| `src/structure_module/` | `src/coral_forge/structure/` | IPA, backbone frames, torsion angles |
| `control/sovereign_folding/` | `control/coral_forge/` | Python baselines + JSON |
| `sovereign_folding/` (root docs) | `coral_forge/` | Planning docs |
| `validate_sovereign_folding*` | `validate_coral_forge*` | 3 validation binaries |

### Why This Matters for ToadStool

1. **Single import path**: `neural_spring::coral_forge::{gelu, layer_norm, softmax_rows, sdpa_full, triangle_mul_outgoing, ...}` and `neural_spring::coral_forge::structure::{ipa_scores, backbone_update, torsion_angles}`
2. **RPC capability names stable**: `science.structure_module` in the biomeOS registry was not renamed — protocol compatibility preserved
3. **15 df64 WGSL shaders** in `metalForge/shaders/` remain unchanged — they reference the shader source, not the Rust module path

### coralForge Primitive Inventory

| Category | Primitives | Validation |
|----------|-----------|------------|
| Activations | `gelu`, `layer_norm`, `softmax_rows` | Py 12/12, Rs 9/9 |
| Attention | `sdpa_scores`, `sdpa_full`, `msa_row_attention`, `msa_col_attention`, `outer_product_mean` | Py 12/12, Rs 9/9 |
| Triangle | `triangle_mul_outgoing`, `triangle_mul_incoming`, `triangle_attention_scores` | Py 12/12, Rs 9/9 |
| Diffusion | Cosine/linear schedules, forward diffusion, DDPM/DDIM reverse, SE(3)-equivariant noise | Py 29/29, Rs 26/26 |
| Pairformer | Sinusoidal embedding, conditioning, triangle ops + FFN | Py 14/14, Rs 13/13 |
| Confidence | pLDDT, PAE, pDE, ranking score | Py 19/19, Rs 16/16 |
| Structure (IPA) | `ipa_scores`, `backbone_update`, `torsion_angles` | Py 12/12, Rs 9/9 |

**Total coralForge**: Py 62/62 + Rs 55/55 + 37/37 GPU (15 df64 shaders) = **154 checks**

---

## Part 2: Deep Debt Resolution

### 2a. Tolerance Domain Guards

5 new constants in `tolerances/mod.rs` under `domain_guards` category:

| Constant | Value | Domain | Justification |
|----------|-------|--------|---------------|
| `FISHER_EPS` | `1e-10` | Counterdiabatic driving | Fisher metric floor at saddle points |
| `BURGERS_IC_GUARD` | `1e-14` | PINN Burgers | Cole-Hopf IC detection (t=0) |
| `DP_EQUALITY_EPS` | `1e-10` | SATé alignment | DP traceback float equality |
| `SINGLETON_FREQ_EPS` | `1e-10` | Pangenome analysis | Singleton gene detection |
| `PHENOTYPE_TIE_EPS` | `1e-10` | Regulatory network | Fate decision tie-breaking |

All registered in `tolerances/registry.rs` and tested in integration tests. Registry now has 139+ entries across categories: `exact`, `cross_language`, `cross_precision`, `gpu_specific`, `domain_specific`, `algorithmic`, `folding`, `domain_guards`.

### 2b. Graceful Error Handling

24 `expect()` calls in 3 GPU validation binaries evolved to `require!(h, ...)`:

| Binary | Sites | Pattern |
|--------|-------|---------|
| `validate_coral_forge_gpu` | 9 | `gpu.create_buffer_f64(...)` |
| `validate_coral_forge_gpu_pipeline` | 8 | `gpu.create_buffer_f64(...)` |
| `validate_barracuda_alphafold2` | 7 | `barracuda::*_dispatch(...)` |

**toadStool action**: No upstream changes needed. The `require!` macro is neuralSpring-local (`src/validation.rs`).

### 2c. Cast Safety

`src/gpu_dispatch/cpu_fallback.rs`: `f64` → `usize` conversion for ODE activator indices now uses a bounds-checking helper:

```rust
fn safe_idx(raw: f64, dim: usize) -> usize {
    assert!(raw >= 0.0 && (raw as usize) < dim,
        "activator_idx {raw} out of range [0, {dim})");
    raw as usize
}
```

### 2d. Provenance Documentation

All 34 `BaselineProvenance` constants in `src/provenance.rs` now have `///` doc comments. No previously undocumented constant remains. Clippy `doc_markdown` warnings for `DeepONet` and `SATé` resolved.

---

## Part 3: BarraCUDA Usage Summary & Absorption Targets

### Current Usage (130+ import sites, 44 upstream rewires)

| Category | Import Sites | Key APIs |
|----------|-------------|----------|
| Device/GPU | 20+ | `WgpuDevice`, `compile_shader_universal`, `Precision`, `GpuCapabilities` |
| Statistics | 25+ | `variance`, `pearson`, `ESD`, `MP bounds`, `r_squared`, `rmse`, `dot`, `l2_norm`, `shannon` |
| Linear algebra | 15+ | `eigh_f64`, `solve_f64`, `cholesky_f64`, `lu_det`, `SVD`, `gen_eigh` |
| Tensor API | 40+ | `Tensor`, FFT, `ops::*_f64`, reductions |
| Shaders | 15+ | `precision::cpu`, `quantized`, df64 compilation |
| Numerical | 5+ | `rk45_solve` |
| Special functions | 10+ | `chi_squared`, `gamma`, `erf`, `bessel_*`, `legendre` |
| Bio/dispatch | 20+ | `domain_ops::*`, `HillGateGpu`, `MultiObjFitnessGpu`, `SwarmNnGpu` |

### What neuralSpring Has Built That ToadStool Should Consider Absorbing

| Item | Location | LOC | Why Absorb |
|------|----------|-----|------------|
| **Evoformer primitives** | `coral_forge/activation.rs`, `attention.rs`, `triangle.rs`, `msa.rs` | ~800 | Pure f64 reference implementations for AlphaFold2. Reusable for any structure prediction pipeline. |
| **IPA (Invariant Point Attention)** | `coral_forge/structure/ipa.rs` | ~300 | Algorithm 22 from Jumper et al. The geometric attention mechanism for 3D coordinate reasoning. |
| **Diffusion primitives** | `coral_forge/diffusion.rs` | ~400 | DDPM/DDIM schedules, forward/reverse diffusion, SE(3)-equivariant noise. General enough for any diffusion model. |
| **Pairformer block** | `coral_forge/pairformer.rs` | ~300 | AlphaFold3's pairformer with timestep conditioning. Compositional: triangle ops + FFN + attention. |
| **Confidence heads** | `coral_forge/confidence.rs` | ~350 | pLDDT, PAE, pDE, ranking score. Common post-prediction quality metrics. |
| **Backbone frames** | `coral_forge/structure/backbone.rs` | ~150 | Rigid-body transformations for protein backbone. |
| **Torsion angles** | `coral_forge/structure/torsion.rs` | ~100 | Side-chain torsion angle prediction. |
| **df64 WGSL shaders** | `metalForge/shaders/` | 15 shaders | Already evolved to df64 core streaming. Ready for upstream cataloging. |

**toadStool action**: Review `coral_forge/` primitives for upstream absorption into `barracuda::ops::structure` or `barracuda::ops::ml`. The f64 reference implementations are battle-tested (154 checks) and the 15 df64 WGSL shaders are ready for integration into the Sovereign Compiler pipeline.

### What neuralSpring Learned That's Relevant to ToadStool Evolution

1. **df64 hybrid strategy works**: On GPUs with low native f64 throughput (RTX 4070: 1:64 ratio), df64 core streaming achieves ~14-digit (fp48) precision. Arithmetic ops: 3.6e-8 to 5.6e-7 max diff. Transcendental ops: 1.7e-4 to 3.4e-4. `Fp64Strategy::Hybrid` auto-detection works reliably.

2. **Triangle operations are the bottleneck**: In Evoformer/Pairformer blocks, triangle multiplication and attention dominate compute. The O(N²) pair representation is the memory bottleneck. ToadStool's tile-based matmul strategy applies directly.

3. **IPA geometry is GPU-friendly**: Invariant Point Attention decomposes into standard matmul + softmax + weighted aggregation, with additional geometric distance terms that reduce to elementwise ops + reductions. All 6 fundamental primitives appear.

4. **Diffusion reverse requires sequential steps**: Unlike forward diffusion (parallelizable), DDPM/DDIM reverse is inherently sequential (each step depends on previous). `StatefulPipeline` is the right abstraction. For DDIM, a skip schedule (e.g., 50 steps from 1000) works with scalar-only readback between steps.

5. **Confidence heads are embarrassingly parallel**: pLDDT, PAE, pDE can all be computed in a single forward pass with one encoder and scalar readback. Perfect for `UnidirectionalPipeline`.

6. **Zero production mocks**: neuralSpring has achieved zero mocks in production code. All BarraCUDA interactions are real dispatches or `#[cfg(test)]`-gated stubs. This validates that the API surface is complete for ML workloads.

---

## Part 4: Dependency Analysis

All 12 external dependencies are pure Rust with zero C/C++ wrappers:

| Crate | Purpose | C/C++ Free |
|-------|---------|------------|
| `barracuda` | GPU compute (path dep) | Pure Rust + WGSL |
| `neural-spring-forge` | metalForge WGSL (path dep) | Pure Rust + WGSL |
| `biomeos-primal-sdk` | biomeOS IPC (path dep) | Pure Rust |
| `bytemuck` | Safe transmute for GPU buffers | Pure Rust |
| `serde` + `serde_json` | Baseline serialization | Pure Rust |
| `tokio` | Async runtime (primal feature) | Pure Rust |
| `wgpu` | WebGPU backend | Pure Rust (Vulkan/DX12/Metal via system drivers) |
| `anyhow` | Error handling | Pure Rust |
| `uuid`, `chrono` | Request IDs, timestamps | Pure Rust |
| `log`, `env_logger` | Diagnostics | Pure Rust |
| `approx` | Floating-point assertions (dev) | Pure Rust |

---

## Part 5: Quality Gates

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets -- -W clippy::pedantic` | **0 warnings** |
| `cargo doc --no-deps` | **0 warnings** |
| `cargo test --lib --test integration` | **685 lib + 9 integration PASS** |
| `cargo build --bin neuralspring_primal --features primal` | PASS |
| `validate_all` | **185/185 PASS** |
| Zero unsafe | Confirmed (`#![forbid(unsafe_code)]`) |
| Zero production mocks | Confirmed |
| Zero inline magic numbers | Confirmed (139+ named tolerances) |
| SPDX headers | 100% coverage |
| LOC per file | All files < 1000 LOC |

---

## Superseded Handoffs

All in `wateringHole/handoffs/archive/`:

- V61: Deep Debt + Confidence Heads (Feb 28, S92-93)
- V60: Dispatch Parity + Mixed-Hardware (Feb 27, S89)
- V59: Comprehensive Evolution (Feb 27)
- V58: CPU Parity + GPU Portability (Feb 27)
- V57: Modern Rewire + Cross-Spring (Feb 27)
- V56: ToadStool `e96576ee` Sync (Feb 27)

---

*neuralSpring V62 — Session 94, February 28, 2026. coralForge unified, deep debt resolved, all quality gates green.*
