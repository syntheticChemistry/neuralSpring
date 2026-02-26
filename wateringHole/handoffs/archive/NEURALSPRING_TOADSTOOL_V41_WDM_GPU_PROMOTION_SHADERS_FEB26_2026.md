# neuralSpring → ToadStool/BarraCUDA Handoff V41 — WDM Surrogates + baseCamp GPU Promotion + Folding Shaders

**Session 77 | February 26, 2026**
**Previous**: V40 (Session 76 — Modern rewiring + benchmark validation)

---

## Part 1: Executive Summary

Session 77 delivers three workstreams:

1. **baseCamp GPU Promotion** — All 5 sub-theses now run on pure GPU via BarraCUDA
   dispatch (eigensolve, matmul, transpose, pairwise L2), with CPU-only fallbacks.
   New validator (`validate_basecamp_gpu_pure`) and benchmark
   (`bench_basecamp_gpu_pure`) binaries prove GPU parity and document speedups.

2. **WDM Surrogate Extensions** — Three new ML surrogates for Warm Dense Matter
   physics (nW-01 transport, nW-02 EOS, nW-04 transfer learning), with Python
   baselines, Rust CPU validation, and BarraCUDA GPU validation. Militzer FPEOS
   data acquisition automated.

3. **Sovereign Folding Shaders** — 9 new WGSL shaders for f64-precision transformer
   operations and AlphaFold2 triangle updates, ready for ToadStool absorption.

### What's New in V41 (Session 77)

| Action | Details |
|--------|---------|
| **baseCamp GPU pure** | 5/5 sub-theses on GPU: eigensolve, matmul, transpose, pairwise L2 |
| **+3 dispatcher methods** | `landscape_analysis`, `attention_spectral_analysis`, `mlp_signal_propagation` |
| **Sub-04 f64 fix** | Belief propagation rewired from f32 Tensor to f64 `matmul_dispatch` + `transpose_dispatch` |
| **Sub-01 gamma fix** | `spectral_result_from_decomposition` now accepts computed gamma (was hardcoded 1.0) |
| **WDM surrogates** | nW-01 transport, nW-02 EOS (H/He/C), nW-04 transfer learning — Python + Rust + GPU |
| **+1 Rust module** | `wdm_surrogate.rs` — MLP inference, JSON weight loading, z-score normalization |
| **+3 Rust validators** | `validate_wdm_eos`, `validate_barracuda_wdm_eos`, `validate_basecamp_gpu_pure` |
| **+1 Rust benchmark** | `bench_basecamp_gpu_pure` — CPU vs GPU per sub-thesis |
| **+9 WGSL shaders** | f64 activations (3), f64 SDPA (3), triangle ops (3) |

---

## Part 2: New Shaders for ToadStool Absorption

### 2a: f64 Activation Shaders (General Purpose)

These use df64 (double-float) emulation for f64 precision on consumer GPUs.
All three are building blocks for any transformer architecture.

| Shader | Entry Point | Location | Cross-Spring Use |
|--------|-------------|----------|------------------|
| `layer_norm_f64.wgsl` | `layer_norm` | `metalForge/shaders/` | baseCamp Sub-02, WDM, folding Evoformer |
| `gelu_f64.wgsl` | `gelu_f64` | `metalForge/shaders/` | FFN blocks, baseCamp, WDM |
| `sigmoid_f64.wgsl` | `sigmoid_f64` | `metalForge/shaders/` | Gating, folding pair bias, WDM output |

**Key design decisions**:
- `layer_norm_f64` uses workgroup-cooperative reduction (256 threads per row).
  Mean and variance accumulation via df64 prevents catastrophic cancellation at
  hidden_dim > 512. Eps passed as uniform (hi/lo split for future true f64 eps).
- `gelu_f64` uses `fma()` for the cubic term to preserve precision in the
  `0.044715 * x^3` coefficient. Polynomial tanh approximation.
- `sigmoid_f64` uses sign-branch for numerical stability: avoids `exp(large)`
  overflow by computing `exp(x)/(1+exp(x))` for x < 0.

**Suggested absorption path**: `barracuda::ops::layer_norm_f64`,
`barracuda::ops::gelu_f64`, `barracuda::ops::sigmoid_f64`. These complement
existing f32 variants in `barracuda/src/shaders/activation/`.

### 2b: f64 Scaled Dot-Product Attention (3-Pass)

Mirrors ToadStool's existing 3-pass SDPA pipeline but with df64 accumulation:

| Shader | Entry Point | Mirrors |
|--------|-------------|---------|
| `sdpa_scores_f64.wgsl` | `main` | `sdpa_scores.wgsl` |
| `softmax_f64.wgsl` | `main` | `attention_softmax.wgsl` |
| `attention_apply_f64.wgsl` | `main` | `attention_apply.wgsl` |

**Key differences from f32 pipeline**:
- Score accumulation uses `df64_mul` + `df64_add` for the Q·K dot product,
  preventing cancellation at head_dim > 64.
- Softmax denominator uses df64 tree reduction for the exp-sum, critical when
  kv_seq_len > 256.
- Value application uses df64 accumulation for the weighted sum.
- Layout and `AttentionParams` struct are identical to existing f32 shaders
  for drop-in replacement.

**Suggested absorption path**: `barracuda::ops::sdpa_scores_f64`,
`barracuda::ops::softmax_f64`, `barracuda::ops::attention_apply_f64`.
These extend the existing `barracuda/src/shaders/attention/` directory.

### 2c: Triangle Operations (Folding-Specific)

AlphaFold2 Evoformer building blocks (Jumper et al. 2021, Algorithms 11-14):

| Shader | Algorithm | Contraction |
|--------|-----------|-------------|
| `triangle_mul_outgoing_f64.wgsl` | Alg 11 | `sum_k a[i,k] * b[j,k]` |
| `triangle_mul_incoming_f64.wgsl` | Alg 12 | `sum_k a[k,i] * b[k,j]` |
| `triangle_attention_f64.wgsl` | Alg 13/14 | Row-wise SDPA + pair bias |

**Data layout**: `[N, N, C]` for pair representations, `[H, N, N]` for bias.
All use df64 accumulation over the shared dimension k.

**For column-wise attention** (Algorithm 14): transpose the pair representation
`z[j,i,c]` before dispatch rather than duplicating the shader.

**Suggested absorption path**: New `barracuda/src/shaders/folding/` directory.
These are the first folding-specific shaders in the ecosystem and should be
tagged as such for discovery by other springs working on structural biology.

---

## Part 3: baseCamp GPU Promotion Details

### Sub-thesis GPU Paths

| Sub-thesis | CPU Step | GPU Step | BarraCUDA API |
|------------|----------|----------|---------------|
| Sub-01: Weight Spectral | Hamiltonian construction O(mn) | Eigensolve O(n³) | `eigh_gpu` |
| Sub-02: Information Flow | Attention Hamiltonian O(n²) | Eigensolve + matmul | `eigh_gpu`, `matmul_dispatch` |
| Sub-03: Loss Landscape | Numerical Hessian (closure) | Eigensolve O(n³) | `eigh_gpu` |
| Sub-04: Neural PGM | — | Full GEMV chain f64 | `transpose_dispatch`, `matmul_dispatch` |
| Sub-05: Agent Coordination | — | Pairwise L2 matrix | `pairwise_l2_matrix_gpu` |

### Bug Fixes

1. **Sub-01 gamma parameter**: `spectral_result_from_decomposition` was hardcoding
   `gamma = 1.0` for the Marchenko-Pastur departure calculation. Now accepts
   the computed aspect ratio `rows / cols` from the dispatcher. This affects
   non-square weight matrices (which are common in practice).

2. **Sub-04 f32 precision loss**: Belief propagation was casting f64 transition
   matrices to f32 for `Tensor::matmul`, then casting back. Now uses
   `barracuda::dispatch::transpose_dispatch` and `matmul_dispatch` which
   operate at f64 natively.

### New Dispatcher Methods

```rust
// Sub-03: CPU Hessian → GPU eigensolve → scalar metrics
pub fn landscape_analysis(
    &self, loss_fn: &dyn Fn(&[f64]) -> f64,
    params: &[f64], epsilon: f64, flatness_threshold: f64,
) -> LandscapeResult

// Sub-02: Attention Hamiltonian → GPU eigensolve → spectral metrics
pub fn attention_spectral_analysis(
    &self, attention: &[f64], n: usize,
) -> AttentionSpectralResult

// Sub-02: GPU matmul chain with ReLU activation
pub fn mlp_signal_propagation(
    &self, input: &[f64],
    weight_matrices: &[&[f64]], layer_dims: &[usize],
) -> Vec<f64>
```

---

## Part 4: WDM Surrogate Extensions

### nW-01: Transport Coefficients (Stanton-Murillo)

Predicts viscosity and thermal conductivity for WDM using synthetic
Stanton-Murillo data. MLP architecture: `[4, 128, 128, 2]`.

| Metric | Value |
|--------|-------|
| Python R² (viscosity) | > 0.90 |
| Python R² (conductivity) | > 0.90 |
| Baseline | `control/wdm/transport_baseline.json` |

### nW-02: Equation of State (Militzer FPEOS)

Predicts log-pressure and log-energy from (log-density, log-temperature)
for H, He, C from first-principles EOS tables.

| Metric | Value |
|--------|-------|
| Python R² (pressure) | > 0.95 |
| Python R² (energy) | > 0.70 |
| Rust CPU parity | finite + deterministic + monotonic |
| BarraCUDA GPU parity | < 1e-3 vs Rust CPU (f32 representation) |
| Baseline | `control/wdm/eos_baseline.json` |

Data source: Militzer et al., PRE 103, 013203 (2021).
Automated acquisition: `control/wdm/download_fpeos.sh`.

### nW-04: Transfer Learning (Classical → WDM)

Demonstrates that pretraining on classical plasma data improves WDM prediction
with scarce WDM samples (30 points).

| Metric | Value |
|--------|-------|
| Classical pretraining R² | > 0.90 |
| Transfer R² | 0.94 |
| From-scratch R² | 0.67 |
| Transfer improvement | +0.27 |
| Baseline | `control/wdm/transfer_baseline.json` |

---

## Part 5: Recommendations for ToadStool Team

### Carried from V40 (still relevant)

1. **Add `stats::mae`** — neuralSpring has local MAE.
2. **Re-export `WGSL_RK4_PARALLEL`** constant.
3. **`shannon(frequencies)` variant** for pre-normalized inputs.
4. **`pearson_correlation_or(default)`** convenience.
5. **`variance_population()` or `ddof` parameter**.
6. **Cross-spring benchmark standardization** as `barracuda::bench` module.

### New from V41

7. **Absorb f64 activation shaders**: `layer_norm_f64`, `gelu_f64`, `sigmoid_f64`
   are general-purpose and cross-spring. They should live alongside existing f32
   activations. The df64 helper functions (`df64_add`, `df64_mul`) could be
   factored into a shared WGSL include or utility module.

8. **df64 utility module**: The three df64 helper functions are duplicated across
   9 shaders. A `df64_utils.wgsl` include (or inline snippet approach) would
   reduce maintenance burden. Consider a WGSL preprocessor include mechanism.

9. **Create `barracuda/src/shaders/folding/` directory**: Triangle multiplication
   and triangle attention are the first structural biology primitives. As more
   springs contribute folding-adjacent shaders, a dedicated directory prevents
   sprawl in the existing attention/ and math/ directories.

10. **f64 SDPA as alternative pipeline**: The 3-pass f64 SDPA should be selectable
    alongside existing f32 SDPA via a precision parameter in `TensorSession::attention()`.
    Same `AttentionParams` struct, same binding layout — the only difference is
    df64 accumulation internally.

11. **MLP dispatch primitive**: `matmul_dispatch` + ReLU is a common pattern across
    WDM surrogates. Consider a fused `mlp_forward_dispatch` that chains matmul →
    activation in a single submission, reducing dispatch overhead for small layers.

12. **WDM surrogate GPU Tensor path**: `validate_barracuda_wdm_eos.rs` manually
    constructs `Tensor` objects for each MLP layer. A `barracuda::nn::Linear` or
    `barracuda::nn::MLP` abstraction would make this pattern reusable across springs.

---

## Part 6: New Files Index

| File | Type | Purpose |
|------|------|---------|
| `metalForge/shaders/layer_norm_f64.wgsl` | WGSL | f64 layer normalization |
| `metalForge/shaders/gelu_f64.wgsl` | WGSL | f64 GELU activation |
| `metalForge/shaders/sigmoid_f64.wgsl` | WGSL | f64 sigmoid activation |
| `metalForge/shaders/sdpa_scores_f64.wgsl` | WGSL | f64 SDPA pass 1 (scores) |
| `metalForge/shaders/softmax_f64.wgsl` | WGSL | f64 SDPA pass 2 (softmax) |
| `metalForge/shaders/attention_apply_f64.wgsl` | WGSL | f64 SDPA pass 3 (apply) |
| `metalForge/shaders/triangle_mul_outgoing_f64.wgsl` | WGSL | Triangle mul outgoing (Alg 11) |
| `metalForge/shaders/triangle_mul_incoming_f64.wgsl` | WGSL | Triangle mul incoming (Alg 12) |
| `metalForge/shaders/triangle_attention_f64.wgsl` | WGSL | Triangle attention + bias (Alg 13/14) |
| `src/wdm_surrogate.rs` | Rust lib | WDM EOS surrogate MLP |
| `src/gpu_dispatch/basecamp.rs` | Rust lib | Updated: +3 methods, 2 bug fixes |
| `src/weight_spectral.rs` | Rust lib | Updated: gamma parameter |
| `src/bin/validate_basecamp_gpu_pure.rs` | Rust bin | baseCamp GPU validator (20+ checks) |
| `src/bin/bench_basecamp_gpu_pure.rs` | Rust bin | baseCamp GPU benchmark |
| `src/bin/validate_wdm_eos.rs` | Rust bin | WDM EOS CPU validator |
| `src/bin/validate_barracuda_wdm_eos.rs` | Rust bin | WDM EOS GPU validator |
| `control/wdm/eos_surrogate.py` | Python | nW-02 baseline |
| `control/wdm/transport_surrogate.py` | Python | nW-01 baseline |
| `control/wdm/transfer_classical_to_wdm.py` | Python | nW-04 baseline |
| `control/wdm/download_fpeos.sh` | Shell | FPEOS data acquisition |

---

## Part 7: Current State

| Metric | Value |
|--------|-------|
| Papers reproduced | 25 + 5 baseCamp sub-theses + 3 WDM surrogates |
| Python baselines | 209/209 PASS |
| Rust+GPU checks | 2010+ PASS |
| Total validation | **2220+** checks |
| Library tests | **581/581** PASS |
| New WGSL shaders | **9** (ready for ToadStool absorption) |
| baseCamp GPU pure | **5/5** sub-theses validated |
| WDM surrogates | **3/3** validated (Python + Rust + GPU) |

### Quality Gates (all green)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | **PASS** |
| `cargo clippy --workspace -- -D warnings` | **0 warnings** |
| `cargo test --workspace` | **581+ PASS** |
| SPDX compliance | **100%** |

---

*AGPL-3.0-or-later | neuralSpring → ToadStool V41 | Session 77 | February 26, 2026*
