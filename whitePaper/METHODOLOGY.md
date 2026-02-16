# neuralSpring — Methodology

## Phase 0: Python/PyTorch Baselines

### Validation Framework

Each experiment follows the same structure:
1. **Benchmark definition** — expected results, acceptance criteria
2. **Implementation** — Python/PyTorch reference code
3. **Validation** — automated PASS/FAIL checks against criteria
4. **Isomorphic analysis** — map ops to BarraCUDA WGSL shaders

### Experiment Details

#### Experiment 001: Neural Surrogate Validation (11 checks)

**Objective**: Compare MLP neural surrogates against RBF interpolation.

| Check | Target |
|-------|--------|
| Rastrigin RBF R² | ≥ 0.40 (multimodal — intentionally hard) |
| Rastrigin MLP R² | ≥ 0.40 |
| Rosenbrock RBF R² | ≥ 0.95 |
| Rosenbrock MLP R² | ≥ 0.95 |
| Ackley RBF R² | ≥ 0.90 |
| Ackley MLP R² | ≥ 0.90 |
| FAO-56 RBF R² | ≥ 0.95 |
| FAO-56 MLP R² | ≥ 0.95 |
| FAO-56 MLP RMSE | ≤ 0.15 mm/day |
| Training efficiency | Completed |
| Op count analysis | Completed |

**Key finding**: MLP (4,673 params) achieves R²=0.999 on FAO-56, matching RBF.
Rastrigin exposes the limitation of random sampling for multimodal functions —
exactly why hotSpring's SparsitySampler matters.

#### Experiment 002: Transformer Inference Baseline (18 checks)

**Objective**: Implement self-attention from scratch, validate against PyTorch.

| Check Category | Count |
|----------------|-------|
| Softmax properties | 3 |
| SDPA correctness | 3 |
| Causal attention | 2 |
| Multi-head attention | 3 |
| LayerNorm + GELU + FFN | 4 |
| Full transformer block | 2 |
| Isomorphic catalog | 1 |

**Key finding**: NumPy SDPA matches PyTorch to <1e-10. The transformer block
(8 MatMuls + Softmax + LayerNorm + GELU) is identical across llama.cpp,
OpenFold, and ViT — confirmed computationally.

#### Experiment 003: Sequence Forecasting (5 checks)

**Objective**: Train LSTM/GRU on Michigan weather time series.

| Check | Target |
|-------|--------|
| LSTM competitive with persistence | Within 0.10 R² |
| LSTM R² | > 0.80 |
| GRU R² | > 0.80 |
| Horizon sweep | Completed |
| Op analysis | Completed |

**Key finding**: LSTM and GRU both achieve R²≈0.93 on 1-day Tmax forecasts.
Persistence is a strong baseline for autocorrelated weather data — the real
advantage of neural models shows at longer horizons.

#### Experiment 004: Transfer Learning (6 checks)

**Objective**: Transfer Michigan ET₀ model to New Mexico and California.

| Check | Target |
|-------|--------|
| Source model R² | > 0.95 |
| Domain gap NM | Detected |
| Domain gap CA | Detected |
| Fine-tuning NM | Improves |
| Fine-tuning CA | Improves |
| From-scratch baseline | Completed |

**Key finding**: Michigan→NM domain gap is 0.33 R² (different climate regime).
Fine-tuning just the head layer with 200 NM samples bridges most of the gap.
Same pattern as vision transfer (freeze backbone, retrain head).

#### Experiment 005: Isomorphic Pattern Catalog (8 checks)

**Objective**: Map shared primitives across language, protein, vision, physics, time series.

| Check | Target |
|-------|--------|
| Architecture survey | 6 architectures |
| GEMM universality | All architectures |
| Nonlinearity universality | All architectures |
| Cross-domain sharing | > 0 pairs |
| BarraCUDA coverage | ≥ 70% |
| Quantization path | Analyzed |
| Isomorphism theorem | Stated |
| PyTorch op trace | Correct output |

**Key finding**: Six fundamental primitives explain all architectures.
BarraCUDA has WGSL shaders for all six. The Rust evolution team needs
to optimize 6 ops, not 600.

### Grand Total: 48 checks across 5 experiments

## Evolution Roadmap

| Phase | Focus | Validates |
|-------|-------|-----------|
| 0 | Python/PyTorch baselines | Science correctness |
| 1 | BarraCUDA Rust port | WGSL shader correctness |
| 2 | Quantized inference | Q4/Q8 deployment |
| 3 | Cross-spring integration | Live surrogates |
