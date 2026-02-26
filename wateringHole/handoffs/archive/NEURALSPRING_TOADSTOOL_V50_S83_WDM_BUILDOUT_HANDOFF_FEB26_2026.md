# neuralSpring → ToadStool/BarraCUDA Handoff: V50 WDM Surrogate Buildout

**Date:** February 26, 2026
**From:** neuralSpring Session 86
**To:** ToadStool/BarraCUDA team
**ToadStool pin:** S68 (`f0feb226`)
**neuralSpring:** 611 lib + 43 forge + 9 integration tests, 170 binaries, 154/154 PASS

---

## Executive Summary

- WDM surrogate paper queue buildout complete: nW-01, nW-02, nW-04 all wired
- 4 new Rust validators (87 total WDM checks) — CPU + GPU tiers
- New `wdm_transport.rs` module: MLP 3→H→3 with log-space normalization
- `SimpleMLP` JSON weight loading remains the #1 absorption target
- Cross-language RNG divergence documented (xoshiro256++ vs Mersenne Twister)
- Hamming 20.85× regression from V49 still outstanding
- All quality gates green: 663/663 tests, 0 clippy, 0 doc warnings

---

## Part 1: WDM Surrogate Buildout

Three Warm Dense Matter (WDM) machine learning surrogates, extending
hotSpring's MD/DFT physics into ML territory:

| Item | Scope | Py | Rs | GPU | Module |
|------|-------|----|----|-----|--------|
| nW-01 | Stanton-Murillo transport (D*, η*, λ*) | 4/4 | 30/30 | — | `wdm_transport.rs` (NEW) |
| nW-02 | EOS surrogate P(ρ,T), E(ρ,T) | 9/9 | 36/36 | 15/15 | `wdm_surrogate.rs` |
| nW-04 | Classical→WDM transfer learning | 4/4 | 6/6 | — | `validate_wdm_transfer.rs` |

**Python baselines**: All 3 generate JSON with weights, normalization params,
and reference predictions. Baselines added to `check_drift.sh` (29 total).

**Rust validators**: All 4 added to `validate_all.rs` (154 binaries total).

### nW-01: Transport Surrogate (NEW)

New module `src/wdm_transport.rs` implements:

- `TransportSurrogate` struct: MLP with 3→H→3 architecture
- `Normalization3`: input/output normalization (mean/std for 3-component vectors)
- `load_transport_from_json()`: JSON weight loading + validation
- Forward pass: log-space input → normalize → MLP → denormalize → exp output
- Stanton-Murillo transport coefficients: diffusivity, viscosity, thermal conductivity

Validation (`validate_wdm_transport`): 30 checks covering loaded, finite,
positive, deterministic, monotonic (diffusivity, viscosity, thermal).

### nW-04: Transfer Learning (NEW)

New validator `src/bin/validate_wdm_transfer.rs` implements:

- `SimpleMlp` struct: configurable layer sizes, forward/backward/train
- Classical pretraining + WDM fine-tuning pipeline
- R² threshold validation: classical > 0.85, transfer > 0.40
- Cross-language RNG documented (Python Mersenne Twister vs Rust xoshiro256++)
- Python baseline provenance check: `py_improvement > 0.0` from JSON

---

## Part 2: Learnings for BarraCUDA Evolution

### 2.1 `barracuda::nn::SimpleMLP` — #1 Absorption Target

neuralSpring now has **3 independent MLP implementations** across WDM surrogates
(EOS, transport, transfer). All share the same pattern:

```
load JSON weights → normalize input → layered matmul+relu → denormalize output
```

A `barracuda::nn::SimpleMLP` with:
- JSON weight loading (`serde_json`)
- Configurable activation (ReLU, tanh, identity)
- Optional input/output normalization
- GPU `Tensor::matmul` forward pass

...would replace ~400 LOC across 3 modules and enable GPU-accelerated
surrogate inference for all Springs (hotSpring has similar surrogate patterns).

### 2.2 GPU Surrogate Inference Pipeline

`validate_barracuda_wdm_eos.rs` demonstrates the pattern:
1. Load JSON weights → `Vec<f32>`
2. Upload to `Tensor::from_data()`
3. Chain `matmul → relu → matmul → relu → matmul`
4. Readback scalar predictions

This is exactly the `TensorSession` pattern from Section 1 of
`BARRACUDA_EVOLUTION.md`. A fused surrogate pipeline would:
- Pre-compile all matmul shaders
- Pre-allocate weight/intermediate buffers
- Single `CommandEncoder` for full forward pass
- Readback only final predictions

### 2.3 Log-Space Normalization

WDM transport uses log-space input/output (`log10(ρ)`, `log10(T)` inputs;
`exp(output)` predictions). This is common in scientific surrogates where
quantities span many orders of magnitude. A `LogNormalization` trait in
`barracuda::nn` would standardize this pattern.

### 2.4 Cross-Language RNG Divergence

Rust's xoshiro256++ produces different sequences than Python's Mersenne
Twister, making bit-exact parity impossible for stochastic training
(initialization, batch shuffling). The solution: validate structural
properties (R² thresholds, monotonicity) rather than chasing exact values.

This is a general lesson for all Springs doing ML training validation.

---

## Part 3: Updated Metrics

| Metric | V49 | V50 | Delta |
|--------|-----|-----|-------|
| Python baselines | 206/206 | 223/223 | +17 |
| Rust lib tests | 604 | 611 | +7 |
| Validation binaries | 166 | 170 | +4 |
| validate_all | 150/150 | 154/154 | +4 |
| Modules | 37 | 38 | +1 |
| WDM validators | 2 | 4 | +2 |
| check_drift.sh baselines | 27 | 29 | +2 |
| Total checks | 2250+ | 2350+ | +100 |

---

## Part 4: Outstanding from V49

Items from V49 still pending ToadStool action:

1. **Hamming 20.85× regression** — `PairwiseHammingGpu` 200×500 f64 path
2. **Public f32 shader constants** for integer-distance ops
3. **`wgsl_source()` methods** on typed ops for downstream validation
4. **`barracuda::nn::SimpleMLP`** with JSON weight loading (now higher priority)
5. **`barracuda::testing::GpuTestHarness`** — shared device + mutex pattern
6. **Variance convention docs** (`stats::variance` ÷(N-1) vs `dispatch` ÷N)

---

## Part 5: Validation

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --workspace -- -D warnings` | 0 warnings |
| `cargo doc --workspace --no-deps` | 0 warnings |
| `cargo test --workspace` | **663/663 PASS** |
| `validate_all` | **154/154 PASS** |

---

## Part 6: Verification Commands

```bash
cd /home/eastgate/Development/ecoPrimals/neuralSpring
cargo test --workspace                     # 663/663 PASS
cargo clippy --workspace -- -D warnings   # 0 warnings
cargo run --release --bin validate_all    # 154/154 PASS
cargo run --release --bin validate_wdm_transport  # 30/30
cargo run --release --bin validate_wdm_transfer   # 6/6
```

---

## Part 7: WDM Evolution Path

```
nW-01 transport:  Python ✓ → Rust CPU ✓ → BarraCUDA CPU (pending SimpleMLP) → GPU Tensor
nW-02 EOS:        Python ✓ → Rust CPU ✓ → BarraCUDA GPU ✓ → Pipeline (pending TensorSession)
nW-04 transfer:   Python ✓ → Rust CPU ✓ → BarraCUDA CPU (pending SimpleMLP with training)
nW-03 S(q,ω):    Queued — LSTM on MD-generated S(q,ω) time series
nW-05 NPU phase:  Queued — ESN classifier for WDM regime detection
```

The next step for ToadStool: `SimpleMLP` JSON loading + forward pass would
immediately promote nW-01 and nW-02 to full BarraCUDA CPU tier, and nW-02's
existing GPU path would generalize to all WDM surrogates.

---

*neuralSpring V50 handoff — February 26, 2026, Session 86. AGPL-3.0-or-later.*
