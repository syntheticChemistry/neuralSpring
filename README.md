# neuralSpring — Learning, Surrogates, and Isomorphic Patterns

**The learning layer: ML surrogates, transfer learning, and the shared computational DNA across domains.**

neuralSpring is where models learn. Where airSpring validates clean equations, groundSpring quantifies measurement noise, and hotSpring benchmarks physics simulations, neuralSpring asks: **"can we learn a model that adapts, predicts, and generalizes?"**

```
groundSpring (noise labels) → neuralSpring (learn + adapt) → adapted models for new domains
hotSpring (physics surrogates) → neuralSpring (neural surrogates) → faster-than-simulation predictions
```

## The Core Thesis: Isomorphic Learning Patterns

Across seemingly different domains, the same computational primitives appear:

| Domain | Architecture | Key Ops |
|--------|-------------|---------|
| **Language** (llama.cpp) | Transformer | Embed → Attn → FFN → Norm |
| **Protein** (OpenFold) | Evoformer | MSA Attn → Pair Attn → Structure |
| **Vision** (ResNet/ViT) | CNN/ViT | Conv → Pool → FC / Patch → Attn |
| **Physics Surrogate** | MLP/RBF | Sample → Interpolate → Predict |
| **Time Series** (weather) | LSTM/GRU | Embed → Recur → Decode |

The **isomorphic pattern**: at the primitive level, all of these are compositions of:
- **MatMul** (GEMM/GEMV) — the universal workhorse
- **Attention** (scaled dot-product) — weighted information routing
- **Normalization** (LayerNorm, BatchNorm) — scale stabilization
- **Nonlinearity** (ReLU, GELU, SiLU) — feature carving
- **Reduction** (sum, mean, max) — aggregation
- **Quantization** (Q4, Q8, FP16) — deployment compression

neuralSpring validates these primitives in Python, then hands off to the BarraCUDA team for Rust/WGSL evolution. BarraCUDA already has ~100+ WGSL shaders covering most of these — neuralSpring provides the **test harness** that proves they produce correct learning.

## Current Status

| Experiment | Domain | Status | Key Question |
|------------|--------|--------|--------------|
| 001: Neural Surrogate | Function approximation | Phase 0 | MLP vs RBF on benchmark + FAO-56 |
| 002: Transformer Inference | Language/Protein foundation | Phase 0 | Can we reproduce self-attention from scratch? |
| 003: Sequence Forecasting | Time series (weather) | Phase 0 | LSTM/GRU on real Michigan weather data |
| 004: Transfer Learning | Domain adaptation | Phase 0 | Michigan ET0 model → different climates |
| 005: Isomorphic Catalog | Cross-domain analysis | Phase 0 | Map shared primitives to BarraCUDA ops |

## Quick Start

```bash
pip install -r control/requirements.txt
bash scripts/run_all_baselines.sh
```

## How neuralSpring Relates to Other Springs

| Spring | What It Provides | What neuralSpring Adds |
|--------|------------------|------------------------|
| hotSpring | Physics surrogates (RBF, SparsitySampler) | Neural surrogates (MLP, attention-based) |
| airSpring | FAO-56 ET0, water balance models | Learned ET0 predictor, transfer to new locations |
| wetSpring | Taxonomy pipelines, PFAS screening | Learned classifiers for noisy spectra |
| groundSpring | Noise characterization, uncertainty labels | Uses noise labels for robust training + adaptation |

## BarraCUDA Connection

BarraCUDA already has the infrastructure; neuralSpring proves it works for learning:

| BarraCUDA Module | neuralSpring Validation |
|------------------|------------------------|
| `nn::Layer` (Linear, Conv2D, etc.) | Exp 001: MLP surrogate training |
| `attention`, `mha`, `flash_attention` | Exp 002: Transformer inference |
| `lstm_cell`, `gru_cell`, `bi_lstm` | Exp 003: Sequence forecasting |
| `gemm_f64`, `gemv_q4/q8` | Exp 005: Isomorphic GEMM patterns |
| `esn_v2`, `snn` | Future: reservoir computing |

## Evolution Roadmap

- **Phase 0**: Python/PyTorch baselines (current) — validate the science
- **Phase 1**: BarraCUDA Rust port — prove WGSL shaders produce correct gradients
- **Phase 2**: Quantized inference — Q4/Q8 models on consumer GPU
- **Phase 3**: Cross-spring integration — live surrogate for Penny Irrigation

## Directory Structure

```
neuralSpring/
├── control/
│   ├── surrogate/          # Exp 001: MLP vs RBF surrogates
│   ├── transformer/        # Exp 002: Self-attention from scratch
│   ├── sequence/           # Exp 003: LSTM/GRU weather forecasting
│   ├── transfer/           # Exp 004: Domain adaptation
│   └── isomorphic/         # Exp 005: Cross-domain pattern catalog
├── scripts/
│   └── run_all_baselines.sh
├── whitePaper/
├── data/
├── CONTROL_EXPERIMENT_STATUS.md
├── README.md
└── LICENSE
```

## License

AGPL-3.0-or-later

---

*Initialized: February 16, 2026*
