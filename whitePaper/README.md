# neuralSpring White Paper

## The Isomorphic Learning Engine

**Status**: Phase 0 complete — 48/48 quantitative checks pass

### Key Results

| Experiment | Domain | Tests | Key Finding |
|------------|--------|-------|-------------|
| 001 Neural Surrogate | Function approx + FAO-56 | 11/11 | MLP (4,673 params) replaces FAO-56 chain at R²=0.999, RMSE=0.07 mm/day |
| 002 Transformer | Self-attention mechanics | 18/18 | NumPy SDPA matches PyTorch to <1e-10. Same ops as llama.cpp/OpenFold/ViT |
| 003 Sequence | LSTM/GRU weather forecast | 5/5 | LSTM R²≈0.93 on Michigan Tmax, competitive with persistence baseline |
| 004 Transfer | Michigan→NM/CA adaptation | 6/6 | Domain gap 0.33 R² (NM); fine-tuning with 200 samples bridges it |
| 005 Isomorphic | Cross-domain op catalog | 8/8 | 6 primitives explain ALL architectures. BarraCUDA covers all 6 |

### The Isomorphism Theorem

All neural architectures decompose into compositions of six fundamental primitives:

1. **GEMM** (matrix multiply) — 60-90% of all FLOPs
2. **Attention** (scaled dot-product) — learned routing
3. **Normalization** (LN/BN/RMS) — scale stabilization
4. **Nonlinearity** (ReLU/GELU/SiLU) — feature carving
5. **Reduction** (sum/mean/max) — aggregation
6. **Gating** (sigmoid × value) — information filtering

A single engine optimizing these 6 ops in WGSL serves every domain.

### Key Research Questions Answered

1. **Can neural surrogates replace equation chains?** Yes — MLP surrogate for FAO-56 achieves R²>0.999 with 2000 training samples
2. **Is self-attention correct from scratch?** Yes — NumPy matches PyTorch to machine precision
3. **Can LSTM learn weather patterns?** Yes — R²≈0.93 for 1-day Tmax forecasts
4. **Does transfer learning work across climates?** Yes — fine-tuning with 200 NM samples recovers most of the domain gap
5. **Are architectures isomorphic?** Yes — 6 primitives, all in BarraCUDA

### Cross-Spring Connection

| Spring | Provides | neuralSpring Uses |
|--------|----------|-------------------|
| airSpring | FAO-56 ET₀ model | Surrogate target, real weather data |
| groundSpring | Noise labels, uncertainty | Training robustness, domain gap quantification |
| hotSpring | Physics surrogates (RBF) | Neural surrogate comparison (MLP vs RBF) |
| wetSpring | Taxonomy pipelines | Future: learned classifiers |
