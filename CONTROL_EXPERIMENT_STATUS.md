# neuralSpring — Control Experiment Status

**Last updated**: February 16, 2026
**Gate**: Eastgate (i9-12900K, 32 GB DDR5, RTX 4070 12GB, Pop!_OS 22.04)
**Python**: 3.10, PyTorch 2.9.0+cu128
**Phase 0 Status**: 48/48 quantitative checks PASS

---

## Experiment Register

| ID | Title | Domain | Tests | Status |
|----|-------|--------|-------|--------|
| 001 | Neural Surrogate Validation | MLP vs RBF + FAO-56 | 11/11 | **PASS** |
| 002 | Transformer Inference Baseline | Self-attention from scratch | 18/18 | **PASS** |
| 003 | Sequence Forecasting | LSTM/GRU weather time series | 5/5 | **PASS** |
| 004 | Transfer Learning | Michigan → NM/CA adaptation | 6/6 | **PASS** |
| 005 | Isomorphic Pattern Catalog | Cross-domain op mapping | 8/8 | **PASS** |
| **TOTAL** | | | **48/48** | **ALL PASS** |

---

## Experiment 001: Neural Surrogate Validation (11/11)

**Key Result**: MLP (4,673 params) achieves R²=0.999 on FAO-56 ET₀

| Check | Result | Status |
|-------|--------|--------|
| Rastrigin RBF R² ≥ 0.40 | 0.62 | PASS |
| Rastrigin MLP R² ≥ 0.40 | 0.47 | PASS |
| Rosenbrock RBF R² ≥ 0.95 | 1.00 | PASS |
| Rosenbrock MLP R² ≥ 0.95 | 1.00 | PASS |
| Ackley RBF R² ≥ 0.90 | 0.96 | PASS |
| Ackley MLP R² ≥ 0.90 | 0.94 | PASS |
| FAO-56 RBF R² ≥ 0.95 | 0.999 | PASS |
| FAO-56 MLP R² ≥ 0.95 | 0.999 | PASS |
| FAO-56 MLP RMSE ≤ 0.15 | 0.07 | PASS |
| Training efficiency | Done | PASS |
| Op count analysis | Done | PASS |

---

## Experiment 002: Transformer Inference (18/18)

**Key Result**: NumPy self-attention matches PyTorch to <1e-10

| Check | Status |
|-------|--------|
| Softmax sums to 1.0 | PASS |
| Softmax non-negative | PASS |
| Softmax vs PyTorch | PASS |
| SDPA weights sum to 1.0 | PASS |
| SDPA output shape | PASS |
| SDPA vs PyTorch | PASS |
| Causal mask blocks future | PASS |
| First token self-attention = 1.0 | PASS |
| MHA output shape | PASS |
| MHA weights shape | PASS |
| All heads valid distributions | PASS |
| LayerNorm mean ≈ 0 | PASS |
| LayerNorm variance ≈ 1 | PASS |
| GELU(0) ≈ 0 | PASS |
| FFN output shape | PASS |
| Full block output shape | PASS |
| Block produces finite output | PASS |
| Isomorphic catalog | PASS |

---

## Experiment 003: Sequence Forecasting (5/5)

**Key Result**: LSTM R²≈0.93, competitive with persistence baseline

| Check | Status |
|-------|--------|
| LSTM competitive with persistence | PASS |
| LSTM R² > 0.80 | PASS |
| GRU R² > 0.80 | PASS |
| Horizon sweep completed | PASS |
| Op analysis completed | PASS |

---

## Experiment 004: Transfer Learning (6/6)

**Key Result**: Michigan→NM domain gap = 0.33 R²; fine-tuning recovers it

| Check | Status |
|-------|--------|
| Source model R² > 0.95 | PASS (0.999) |
| Domain gap NM detected | PASS (ΔR²=0.33) |
| Domain gap CA detected | PASS (ΔR²=0.07) |
| Fine-tuning NM improves | PASS |
| Fine-tuning CA improves | PASS |
| From-scratch baseline | PASS |

---

## Experiment 005: Isomorphic Pattern Catalog (8/8)

**Key Result**: 6 primitives explain ALL architectures; BarraCUDA covers all 6

| Check | Status |
|-------|--------|
| Architecture survey (6 archs) | PASS |
| GEMM universality | PASS |
| Nonlinearity universality | PASS |
| Cross-domain sharing | PASS |
| BarraCUDA coverage | PASS |
| Quantization path | PASS |
| Isomorphism theorem | PASS |
| PyTorch op trace | PASS |

---

## Evolution Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Python/PyTorch baselines | **48/48 PASS** |
| 1 | BarraCUDA Rust port | Planned |
| 2 | Quantized inference (Q4/Q8) | Planned |
| 3 | Cross-spring integration | Planned |
