# neuralSpring → ToadStool/BarraCUDA Handoff: V51 WDM Queue Closed

**Date:** February 26, 2026
**From:** neuralSpring Session 87
**To:** ToadStool/BarraCUDA team
**ToadStool pin:** S68 (`f0feb226`)
**neuralSpring:** 623 lib + 43 forge + 9 integration tests, 172 binaries, 156/156 PASS

---

## Executive Summary

- **WDM surrogate queue fully closed**: nW-01 through nW-05 all complete
- 6 Rust validators (186 total WDM checks) — CPU + GPU tiers
- 2 new Rust modules: `wdm_sqw.rs` (LSTM reservoir) + `wdm_esn.rs` (ESN classifier)
- New absorption targets: **LSTM reservoir inference** + **ESN reservoir inference** + `SimpleMLP`
- Reservoir computing pattern validated: fixed random weights + ridge regression readout
- Cross-language parity: Python NumPy ↔ Rust f64 bit-exact for ESN (< 1e-10)
- All quality gates green: 675/675 tests, 0 clippy, 0 doc warnings

---

## Part 1: New WDM Surrogates (Session 87)

Two new Warm Dense Matter surrogates close the paper queue:

| Item | Scope | Py | Rs | GPU | Module |
|------|-------|----|----|-----|--------|
| nW-03 | S(q,ω) LSTM peak predictor | 5/5 | 27/27 | — | `wdm_sqw.rs` (NEW) |
| nW-05 | ESN WDM regime classifier | 5/5 | 39/39 | — | `wdm_esn.rs` (NEW) |

### Full WDM Status (All 5 Complete)

| Item | Paper | Py | Rs | GPU | Key Primitive |
|------|-------|----|----|-----|---------------|
| nW-01 | Stanton-Murillo transport (D*, η*, λ*) | 4/4 | 30/30 | — | MLP 3→H→3, log-space normalization |
| nW-02 | EOS surrogate P(ρ,T), E(ρ,T) | 9/9 | 36/36 | 15/15 | MLP 2→H→2, signed-log, `Tensor::matmul` |
| nW-03 | S(q,ω) LSTM peak predictor | 5/5 | 27/27 | — | LSTM reservoir, pooled readout, R²=0.98 |
| nW-04 | Classical→WDM transfer learning | 4/4 | 6/6 | — | Pre-train MLP, fine-tune WDM |
| nW-05 | ESN WDM regime classifier | 5/5 | 39/39 | — | ESN classifier, 96.5% accuracy |

---

## Part 2: Learnings for BarraCUDA Evolution

### 2.1 LSTM Reservoir Computing — New Absorption Target

`wdm_sqw.rs` implements LSTM reservoir computing with **pooled readout**:

```
Input time series → LSTM cell (fixed weights) → collect hidden states
→ washout(4) → pool(mean, std, last) → linear readout → (ω, γ)
```

Key design: LSTM weights are randomly initialized and **never trained**.
Only the linear readout layer is trained via ridge regression. This makes
the inference pipeline deterministic and lightweight:

- `lstm_cell(x, h, c)` — standard LSTM gates (forget, input, cell, output)
- `pool_hidden_states(all_h)` — mean + std + last hidden state → 3×H features
- `linear_readout(features, W_out, b_out)` — dense matmul → predictions

**Absorption target**: `barracuda::nn::LstmReservoir` with:
- Fixed weight initialization from JSON
- Configurable `hidden_size`, `input_scale`, `spectral_radius`, `forget_bias`
- Pooled readout (mean/std/last) — not just final hidden state
- GPU `Tensor` path for batch inference on time series

### 2.2 Echo State Network (ESN) — New Absorption Target

`wdm_esn.rs` implements a minimal ESN classifier:

```
Input (log_ρ, log_T) → normalize → 2-step reservoir (tanh activation)
→ linear readout → argmax → regime label
```

Key design: `W_res` (reservoir-to-reservoir) has fixed spectral radius 0.9.
`W_in` (input-to-reservoir) has fixed input scale 0.5. Only `W_out` is trained.

**Absorption target**: `barracuda::nn::EsnClassifier` with:
- Fixed reservoir weights (`W_in`, `W_res`, `b_res`)
- Configurable `reservoir_size`, `spectral_radius`, `input_scale`, `n_steps`
- tanh reservoir activation
- Linear readout + argmax classification
- GPU reservoir step could batch across inputs

### 2.3 `SimpleMLP` Remains #1 Priority

Three independent MLP implementations (nW-01, nW-02, nW-04) plus the new
reservoir models all share JSON weight loading. A unified `barracuda::nn`
module with:

1. `SimpleMLP` — matmul + activation stack (3 WDM surrogates + hotSpring)
2. `LstmReservoir` — fixed-weight LSTM with pooled readout (nW-03)
3. `EsnClassifier` — fixed-weight reservoir with linear readout (nW-05)

...would replace ~800 LOC across 5 modules and enable GPU inference for all.

### 2.4 Reservoir Computing as Validation Pattern

Both nW-03 and nW-05 demonstrate that reservoir computing (fixed random
weights + linear readout) is an effective validation pattern for recurrent
architectures. It avoids backpropagation complexity while still exercising:

- LSTM gate mechanics (forget, input, cell, output)
- Reservoir dynamics (spectral radius, echo state property)
- Sequence processing (washout, pooling)
- Ridge regression readout (closed-form, deterministic)

This pattern generalizes: any Spring needing time-series or sequence
classification can use reservoir computing for fast validation without
training infrastructure.

---

## Part 3: Updated Metrics

| Metric | V50 | V51 | Delta |
|--------|-----|-----|-------|
| Python baselines | 223/223 | 233/233 | +10 |
| Rust lib tests | 611 | 623 | +12 |
| Validation binaries | 170 | 172 | +2 |
| validate_all | 154/154 | 156/156 | +2 |
| Modules | 38 | 40 | +2 |
| WDM validators | 4 | 6 | +2 |
| check_drift.sh baselines | 29 | 31 | +2 |
| Total checks | 2350+ | 2450+ | +100 |

---

## Part 4: Outstanding from V50

Items still pending ToadStool action:

1. **Hamming 20.85× regression** — `PairwiseHammingGpu` 200×500 f64 path
2. **Public f32 shader constants** for integer-distance ops
3. **`wgsl_source()` methods** on typed ops for downstream validation
4. **`barracuda::nn::SimpleMLP`** with JSON weight loading (now #1 — 3 WDM users)
5. **`barracuda::nn::LstmReservoir`** — LSTM reservoir with pooled readout (NEW)
6. **`barracuda::nn::EsnClassifier`** — ESN reservoir classifier (NEW)
7. **`barracuda::testing::GpuTestHarness`** — shared device + mutex pattern
8. **Variance convention docs** (`stats::variance` ÷(N-1) vs `dispatch` ÷N)

---

## Part 5: Validation

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --workspace -- -D warnings` | 0 warnings |
| `cargo doc --workspace --no-deps` | 0 warnings |
| `cargo test --workspace` | **675/675 PASS** |
| `validate_all` | **156/156 PASS** |

---

## Part 6: Verification Commands

```bash
cd /home/eastgate/Development/ecoPrimals/neuralSpring
cargo test --workspace                     # 675/675 PASS
cargo clippy --workspace -- -D warnings   # 0 warnings
cargo run --release --bin validate_all    # 156/156 PASS
cargo run --release --bin validate_wdm_sqw       # 27/27
cargo run --release --bin validate_wdm_esn       # 39/39
```

---

## Part 7: WDM Evolution Path (Updated)

```
nW-01 transport:  Python ✓ → Rust CPU ✓ → BarraCUDA CPU (pending SimpleMLP) → GPU Tensor
nW-02 EOS:        Python ✓ → Rust CPU ✓ → BarraCUDA GPU ✓ → Pipeline (pending TensorSession)
nW-03 S(q,ω):    Python ✓ → Rust CPU ✓ → BarraCUDA CPU (pending LstmReservoir) → GPU Tensor
nW-04 transfer:   Python ✓ → Rust CPU ✓ → BarraCUDA CPU (pending SimpleMLP with training)
nW-05 ESN:        Python ✓ → Rust CPU ✓ → BarraCUDA CPU (pending EsnClassifier) → GPU Tensor
```

### GPU Promotion Path for WDM Surrogates

Once `barracuda::nn` absorbs the three model types:

1. **SimpleMLP GPU**: `Tensor::matmul` chain already works (nW-02 proves it).
   Promote nW-01 and nW-04 by swapping CPU MLP for GPU MLP.
2. **LstmReservoir GPU**: LSTM cell is `4×(W_i·x + W_h·h + b)` — four matmuls
   per timestep, fully parallelizable via `TensorSession`. Pooling is a
   `ReduceScalarPipeline`.
3. **EsnClassifier GPU**: Reservoir step is `tanh(W_in·x + W_res·h + b)` —
   two matmuls + activation. 2 steps = 4 matmuls total. Trivial GPU port.

---

## Part 8: Cross-Spring Learnings for ToadStool Absorption

### 8.1 Reservoir Computing Generality

The reservoir computing pattern (fixed random weights + trained linear readout)
appeared independently in two different WDM surrogates:

- **nW-03**: LSTM reservoir for time-series regression (ω, γ prediction)
- **nW-05**: ESN reservoir for classification (3 WDM regimes)

Both use identical training: ridge regression on pooled/final hidden states.
This suggests a `barracuda::nn::Reservoir` trait with:

```rust
trait Reservoir {
    fn step(&self, input: &[f64], state: &mut [f64]) -> Vec<f64>;
    fn readout(&self, features: &[f64]) -> Vec<f64>;
}
```

### 8.2 JSON Weight Interchange

All 5 WDM surrogates use JSON for weight interchange (Python → Rust).
The pattern is universal:

```json
{
  "weights_0": [[...], ...],  // layer 0 weight matrix
  "bias_0": [...],            // layer 0 bias
  "normalization": { "mean": [...], "std": [...] },
  "reference_predictions": { "input": [...], "output": [...] }
}
```

A `barracuda::nn::WeightLoader` that standardizes this format would
benefit all Springs doing ML validation.

### 8.3 Pooled Readout vs Final-State Readout

nW-03 showed that pooling all hidden states (mean + std + last) dramatically
outperforms using only the final hidden state (R² 0.98 vs < 0.10). This is
because the reservoir's temporal dynamics encode information across the
full sequence. The pooling pattern should be the default for reservoir
computing in `barracuda::nn`.

---

*neuralSpring V51 handoff — February 26, 2026, Session 87. AGPL-3.0-or-later.*
