# neuralSpring → ToadStool/BarraCUDA Handoff: V53 Publication Experiments & Absorption Roadmap

**Date:** February 27, 2026
**From:** neuralSpring Session 88+
**To:** ToadStool/BarraCUDA team
**ToadStool pin:** S68 (`f0feb226`)
**neuralSpring:** 668 lib + 43 forge + 9 integration tests, 176 binaries, 163/163 PASS
**Supersedes:** V52 (df64 core streaming)

---

## Executive Summary

- **3 publication experiments completed**: Exp-050 (training trajectory spectral
  analysis), Exp-052 (Hessian eigenanalysis), Exp-053 (Anderson multi-agent QS)
- **Papers A, C, D data-ready** — Tier 1 experiments done; analysis/drafting next
- **Full control matrix verified**: every paper has controls at open data (Py),
  BarraCUDA CPU (Rs), BarraCUDA GPU (Tensor), and metalForge mixed hardware tiers
- **ToadStool absorption targets crystallized**: 15 sovereign folding df64 shaders,
  `compile_shader_df64_streaming` API, 5 typed df64 ops, transcendental precision
  improvement, `nn::SimpleMLP` for WDM surrogates
- **Cross-spring alignment**: wetSpring V60 (NPU live), hotSpring V0614 (debt
  reduction), neuralSpring V53 — all on ToadStool `f0feb226`

---

## Part 1: Publication Experiment Results

### Exp-050 — Training Trajectory Spectral Analysis (Sub-thesis 01: Paper A)

**Target venue**: ICML 2027 or NeurIPS 2026 workshop

IPR and level-spacing ratio computed across training epochs of a 3-layer MLP.
The weight matrix transitions from Wigner-Dyson random-matrix statistics to
structured spectrum as training progresses — a measurable spectral fingerprint
of learning.

| Metric | Python | Rust | Primitives Used |
|--------|--------|------|-----------------|
| IPR across epochs | 11/11 PASS | 12/12 PASS | `eigh_f64`, `BatchIprGpu`, `level_spacing_ratio` |

**BarraCUDA primitives exercised**: `linalg::eigh_f64` (eigendecomposition),
`spectral::BatchIprGpu` (GPU batch IPR), `spectral::level_spacing_ratio` (GOE
statistics). All already upstream — no new absorption needed for this experiment.

### Exp-052 — Hessian Eigenanalysis (Sub-thesis 03: Paper D)

**Target venue**: Digital Discovery 2027

GPU-accelerated Hessian eigendecomposition at trained minima detects saddle-point
vs true-minimum character via negative eigenvalue count. NEB-style path
interpolation between minima validates energy landscape topology.

| Metric | Python | Rust | Primitives Used |
|--------|--------|------|-----------------|
| Hessian eigensystem at minima | 8/8 PASS | 14/14 PASS | `eigh_f64`, `numerical_hessian`, `rk45_solve` |

**BarraCUDA primitives exercised**: `linalg::eigh_f64`, `numerical::rk45_solve`,
`numerical::numerical_hessian`. All upstream. NEB path interpolation uses
`stats::l2_norm`, `stats::dot`.

### Exp-053 — Anderson Multi-Agent QS (Sub-thesis 05: Paper C)

**Target venue**: AAMAS 2027

Anderson localization metrics on multi-agent communication graphs. Cooperation
phase transitions detected via IPR threshold on the agent interaction Laplacian.
Strong disorder (selfish) → localization; weak disorder (cooperative) →
delocalization.

| Metric | Python | Rust | Primitives Used |
|--------|--------|------|-----------------|
| Cooperation phase transitions | 11/11 PASS | 18/18 PASS | `eigh_f64`, `graph_laplacian`, `BatchIprGpu`, `stencil_cooperation` |

**BarraCUDA primitives exercised**: `linalg::eigh_f64`, `linalg::graph_laplacian`,
`linalg::disordered_laplacian`, `spectral::BatchIprGpu`,
`ops::bio::StencilCooperationGpu`. All upstream.

---

## Part 2: What to Absorb — ToadStool Evolution Targets

### 2.1 From V52 (Still Pending)

These items from the V52 handoff remain the primary absorption targets:

#### Priority 1: `compile_shader_df64_streaming` (API consolidation)

Both neuralSpring and hotSpring manually prepend `WGSL_DF64_CORE` +
`WGSL_DF64_TRANSCENDENTALS` before calling `compile_shader_f64`. Three Springs
doing the same concatenation = ready for first-class API.

```rust
// Current pattern (duplicated in neuralSpring + hotSpring):
let combined = format!("{WGSL_DF64_CORE}\n{WGSL_DF64_TRANSCENDENTALS}\n{source}");
device.compile_shader_f64(&combined, Some(label))

// Proposed barracuda API:
device.compile_shader_df64_streaming(source, label)
```

#### Priority 2: 15 Sovereign Folding df64 Shaders

All 15 shaders use the identical three-zone df64 pattern. The primitives are
universal ML building blocks (GELU, LayerNorm, softmax, SDPA, etc.), not
protein-specific. Suggested namespace: `barracuda::ops::attention::*` or
`barracuda::ops::folding::*`.

| Shader | Algorithm | Precision Tier | Max GPU-CPU Diff |
|--------|-----------|----------------|------------------|
| `gelu_f64.wgsl` | Pointwise GELU | Transcendental | 3.41e-4 |
| `layer_norm_f64.wgsl` | LayerNorm | Arithmetic | 5.58e-7 |
| `softmax_f64.wgsl` | Row-wise softmax (3-pass) | Transcendental | 2.92e-4 |
| `sdpa_scores_f64.wgsl` | QKᵀ/√d | Arithmetic | 6.76e-8 |
| `attention_apply_f64.wgsl` | Σ weights × V | Arithmetic | 6.89e-8 |
| `triangle_mul_outgoing_f64.wgsl` | Algorithm 11 | Arithmetic | 3.10e-7 |
| `triangle_mul_incoming_f64.wgsl` | Algorithm 12 | Arithmetic | 4.66e-7 |
| `triangle_attention_f64.wgsl` | Algorithms 13-14 | Arithmetic | 1.54e-7 |
| `outer_product_mean_f64.wgsl` | MSA → pair (OPM) | Arithmetic | 6.43e-8 |
| `msa_row_attention_scores_f64.wgsl` | Row attn + pair bias | Arithmetic | 1.06e-7 |
| `msa_col_attention_scores_f64.wgsl` | Col attn (no bias) | Arithmetic | 9.57e-8 |
| `sigmoid_f64.wgsl` | Sign-branch stable sigmoid | Transcendental | (CPU validated) |
| `ipa_scores_f64.wgsl` | IPA (SE(3)-equivariant) | Arithmetic | 3.40e-7 |
| `backbone_update_f64.wgsl` | Frame composition (quat→rot) | Arithmetic | 3.59e-8 |
| `torsion_angles_f64.wgsl` | Fused ResNet + unit circle | Arithmetic | 1.10e-7 |

#### Priority 3: 5 Typed df64 Ops

Universal ML primitives at fp48 precision on consumer GPUs:

- `barracuda::ops::gelu_df64` — pointwise GELU on f64 buffers
- `barracuda::ops::layer_norm_df64` — layer normalization on f64 buffers
- `barracuda::ops::softmax_df64` — row-wise softmax on f64 buffers
- `barracuda::ops::sdpa_df64` — scaled dot-product attention pipeline
- `barracuda::ops::matmul_df64` — general df64 matrix multiply

#### Priority 4: `barracuda::nn::SimpleMLP`

JSON weight loading + forward pass. Three WDM surrogate users in neuralSpring
(`wdm_surrogate.rs`, `wdm_transport.rs`, `wdm_sqw.rs`). Would also serve
hotSpring's surrogate use cases.

#### Priority 5: Transcendental Precision Improvement

Upgrade `exp_df64` and `tanh_df64` from degree-6 to degree-10+ Horner
polynomials. Current ~3.4e-4 max error could reach ~1e-8, closing the gap with
arithmetic precision. All Springs using df64 transcendentals benefit.

### 2.2 New from S88+ (Publication Experiments)

No new absorption targets from the publication experiments — all primitives used
(eigh, IPR, level_spacing, Hessian, graph Laplacian, stencil cooperation) are
already upstream in BarraCUDA. This validates that the absorption work from
S39–S88 was comprehensive.

### 2.3 Items That Stay Local

| neuralSpring Module | Why Local | Absorption? |
|--------------------|-----------|:-----------:|
| `sovereign_folding.rs` | AlphaFold2 CPU reference (domain physics) | No |
| `structure_module.rs` | AlphaFold2 structure module (domain physics) | No |
| `weight_spectral.rs` | baseCamp research: spectral analysis of weight matrices | No |
| `information_flow.rs` | baseCamp research: signal propagation metrics | No |
| `loss_landscape.rs` | baseCamp research: energy landscape analysis | No |
| `neural_pgm.rs` | baseCamp research: PGM extraction from DNNs | No |
| `agent_coordination.rs` | baseCamp research: multi-agent QS metrics | No |
| `wdm_*.rs` (5 modules) | WDM surrogate domain models (pending `nn::SimpleMLP`) | Partial |

---

## Part 3: Cross-Spring Evolution Learnings

### What neuralSpring Learned from wetSpring

- **Fused entropy kernel**: wetSpring's fused Shannon entropy (2.59× speedup)
  informed neuralSpring's spectral_entropy rewire (S81)
- **Tolerance registry pattern**: wetSpring gen3's centralized tolerance approach
  adopted in neuralSpring (129+ named tolerances)
- **NPU live patterns**: wetSpring V60 demonstrates AKD1000 NPU integration;
  neuralSpring's `Dispatcher::mixed_dispatch()` is structurally ready
- **Sub-thesis cross-referencing**: wetSpring baseCamp Sub-thesis 01 (Anderson
  localization in QS) directly validates neuralSpring Sub-thesis 05 (Anderson
  in multi-agent AI) — same physics, different domains

### What neuralSpring Learned from hotSpring

- **df64 core streaming**: The three-zone pattern (load → compute → store) was
  invented in hotSpring for Yukawa/Coulomb physics, then adopted in neuralSpring
  for ML primitives (Session 88). Proves the pattern is domain-agnostic.
- **Variance convention**: hotSpring's Welford online variance (÷N) vs
  BarraCUDA `stats::variance` (÷(N-1)) documented through V49-V52
- **NVK validation**: hotSpring pioneered Titan V NVK testing; neuralSpring
  adopted the same `NEURALSPRING_BACKEND=titan` pattern (384/384 PASS)
- **PRNG composition**: hotSpring's `xoshiro128ss.wgsl` pattern for GPU-side
  random number generation reused across all stochastic GPU pipelines

### What ToadStool/BarraCUDA Should Absorb from neuralSpring

1. **df64 streaming shader pattern** — domain-agnostic; proven across physics
   (hotSpring), biology (wetSpring), and ML (neuralSpring)
2. **Two-tier tolerance structure** — arithmetic vs transcendental precision
   classes are fundamental to df64, not domain-specific
3. **Publication experiment validation** — the spectral analysis, Hessian, and
   multi-agent experiments provide additional stress tests for `eigh_f64`,
   `BatchIprGpu`, `graph_laplacian`, and `numerical_hessian`
4. **WDM surrogate architecture** — the SimpleMLP/LSTM/ESN pattern is reusable
   across any domain that trains a surrogate model

---

## Part 4: Paper Queue Control Matrix Verification

All papers verified across the hardware progression:

### Open Data (Python) → BarraCUDA CPU → BarraCUDA GPU → metalForge

| Tier | Papers | Checks | Coverage |
|------|--------|--------|----------|
| Python control (open data) | 25/25 + 5 WDM + 3 pub exp | 263 | **100%** |
| Rust CPU | 25/25 + baseCamp + WDM + pub exp | 668 lib + 114 baseCamp | **100%** |
| BarraCUDA CPU (bC) | 24/25 | 203 | **96%** |
| BarraCUDA GPU Tensor (gT) | 23/25 | 98+ | **92%** |
| BarraCUDA GPU baseCamp | 5/5 sub-theses | 14 | **100%** |
| metalForge WGSL (mF) | 15/25 | 108 | **100%** of applicable |
| GPU Pipeline (gP) | 15/25 | 94 | **100%** of applicable |
| Cross-dispatch (xD) | 15/15 | 49 | **100%** |
| CPU↔GPU dispatch | 25 + baseCamp | 16 | **100%** |
| Mixed hardware (mH) | baseCamp | 14 | **100%** |
| Multi-GPU | RTX 4070 + Titan V | 384+133 | **100%** bit-identical |

**Gap analysis**: Only Exp 005 (analytical) and Study 005 (integer Q4/Q8) lack
bC/gT coverage — neither has numerical computation amenable to GPU validation.

---

## Part 5: BarraCUDA Evolution History (neuralSpring's Perspective)

| Session | Milestone | Functions Rewired |
|---------|-----------|:-----------------:|
| S39 | S-14/S-15/S-16 resolved upstream | 0 |
| S56 | First batch rewire (stats, linalg) | 4 |
| S58 | S-17 pow polyfill; stats rewire | 7 |
| S59 | ESD, Marchenko-Pastur, effective_rank | +5 |
| S66 | variance_ddof gap closed | 0 |
| S68 | Universal precision sync (22 commits) | +1 |
| S69 | 6 validator shader sources → upstream | 0 (shaders) |
| S73 | graph, belief_propagation, Boltzmann | +4 |
| S75 | r_squared, rmse, NSE, dot, l2_norm | +9 |
| S76 | mae, l2_distance | +2 |
| S78 | hill, shannon, fit_linear | +6 |
| S81 | spectral_entropy | +1 |
| S88 | df64 core streaming (15 shaders) | 0 (pattern) |
| **Total** | **39 functions + 6 shader sources + 15 df64 shaders** | **39** |

---

## Part 6: Updated Metrics

| Metric | V52 | V53 | Delta |
|--------|-----|-----|-------|
| validate_all | 158/158 | 163/163 | +5 |
| Python checks | 233 | 263 | +30 |
| Rust+GPU checks | 2250+ | 2290+ | +40 |
| Total checks | 2480+ | 2550+ | +70 |
| lib tests | 623 | 668 | +45 |
| binaries | 172 | 176 | +4 |
| Papers data-ready | 0 | 3 (A, C, D) | +3 |
| Absorption gaps | 7 | 7 | 0 (unchanged) |

---

## Part 7: Verification Commands

```bash
cd /home/eastgate/Development/ecoPrimals/neuralSpring
cargo test --workspace                     # 720/720 PASS
cargo clippy --workspace -- -D warnings   # 0 warnings
cargo run --release --bin validate_all    # 163/163 PASS
cargo run --release --bin validate_sovereign_folding_gpu  # 37/37 (df64)
```

---

## Part 8: Next Steps

### For neuralSpring
1. **Paper B** (Spectral Circuits): Awaits ACDC comparison (Tier 2 experiment)
2. **Paper drafting**: A, C, D analysis and writing sessions
3. **Exp-050/052/053 GPU promotion**: Push publication experiments through
   Tier 3+ (BarraCUDA GPU → metalForge → cross-dispatch)

### For ToadStool/BarraCUDA
1. **`compile_shader_df64_streaming`**: First-class API (eliminates 3× dup)
2. **15 sovereign folding shaders → `ops::attention`**: Universal ML building blocks
3. **`nn::SimpleMLP`**: JSON weights + forward (unblocks WDM surrogates)
4. **Transcendental degree-10+**: Close arithmetic↔transcendental precision gap
5. **Variance convention docs**: `stats::variance` ÷(N-1) vs `dispatch` ÷N

---

*neuralSpring V53 handoff — February 27, 2026, Session 88+. AGPL-3.0-or-later.*
