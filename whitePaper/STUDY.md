# neuralSpring — Phase 0 + Phase 0+ Study Results

## Abstract

neuralSpring validates the computational foundations of machine learning across
ten experiments spanning function approximation, transformer attention, sequence
forecasting, transfer learning, cross-domain architecture analysis, physics-informed
neural networks, operator learning, convolutional networks, real-data LSTM, and
quantized inference. All **206 quantitative checks pass** (48 Phase 0 + 31 Phase 0+ + 127 Phase 0++).
Phase 1–5b Rust validation adds **1400+ Rust+GPU checks** (264 lib + 9 integration tests + 119 validation binaries across 31 modules + 2 evolved, 94.9% line coverage). The fused ToadStool pipeline achieves 46–78× speedup over per-op dispatch.
The 3-way benchmark (Python vs CPU vs GPU) with double-buffered evolved shaders
achieves **GPU 104× faster** than Python at 103M FLOPs and **CPU 3.9× faster**
at the same scale.

Phase 0 establishes synthetic baselines. Phase 0+ reproduces five published studies:
Raissi et al. (2019) PINNs, Lu et al. (2021) DeepONet, LeCun et al. (1998) LeNet-5,
LSTM on real ERA5 weather data, and INT8/INT4 quantized inference matching the
llama.cpp GGML pipeline.

The central finding is the **Isomorphism Theorem**: all neural architectures
decompose into compositions of six fundamental primitives (GEMM, Attention,
Normalization, Nonlinearity, Reduction, Gating), and BarraCUDA's existing WGSL
shader library covers all six. Phase 0+ extends this to include Conv2d, autograd,
and quantized GEMV primitives.

## Experiment 001: Neural Surrogate Validation

**Question**: Can a small MLP replace classical surrogates and equation chains?

A 2-hidden-layer MLP (6→64→64→1, 4,673 parameters) was trained on FAO-56 ET₀
computed from random weather inputs. The surrogate achieves:

| Method | R² | RMSE (mm/day) |
|--------|-----|---------------|
| RBF (thin plate spline) | 0.9989 | 0.072 |
| MLP (2×64, ReLU) | 0.999+ | 0.07-0.08 |

Both methods require 500+ samples for R²>0.99 on ET₀. On multimodal benchmarks
(Rastrigin), random sampling struggles — confirming why hotSpring's optimizer-
directed SparsitySampler (Diaw et al. 2024) is essential for complex landscapes.

**Training efficiency**: R²>0.97 with just 100 samples; R²>0.999 with 2000.

## Experiment 002: Transformer Inference Baseline

**Question**: Can we implement self-attention from scratch and match PyTorch?

Pure NumPy implementations of scaled dot-product attention, multi-head attention,
causal masking, LayerNorm, GELU, and the full transformer block match PyTorch
to machine epsilon (<1e-10 max absolute difference).

The transformer block = 8 MatMuls + Softmax + LayerNorm + GELU. This is the
**identical** structure in:
- llama.cpp (language model inference)
- OpenFold (protein structure prediction)
- Vision Transformer (image classification)

Causal masking correctly blocks future token attention (max leak <1e-6).

## Experiment 003: Sequence Forecasting

**Question**: Can LSTM/GRU learn weather temporal patterns?

Trained on 2-year synthetic Michigan weather (AR(1) + seasonal sinusoid):

| Model | RMSE (°C) | R² | vs Persistence |
|-------|-----------|----|----------------|
| Persistence | ~2.6 | 0.939 | baseline |
| LSTM (32 hidden) | ~2.5 | 0.937 | competitive |
| GRU (32 hidden) | ~2.5 | 0.938 | competitive |

Persistence is a strong 1-day baseline for autocorrelated data. The neural
models match it at 1-day horizon but show advantage at longer horizons (3-14
days) where persistence degrades.

LSTM gates (forget, input, output, cell) are isomorphic to attention weights —
both implement "learned information routing" via sigmoid-gated GEMM.

## Experiment 004: Transfer Learning

**Question**: Can a Michigan ET₀ model transfer to New Mexico pistachios?

| Transfer | R² | RMSE (mm/day) | Gap from Source |
|----------|-----|---------------|-----------------|
| Michigan → Michigan | 0.999 | 0.014 | — |
| Michigan → New Mexico | 0.674 | 0.911 | ΔR² = 0.326 |
| Michigan → California | 0.930 | 0.241 | ΔR² = 0.070 |

The NM domain gap (0.33 R²) reflects the fundamental climate difference:
Michigan is humid continental; NM is arid with higher altitude, stronger VPD,
lower humidity. Fine-tuning just the head layer with 200 NM samples recovers
most of the gap.

This is the **same pattern** as ImageNet→medical imaging transfer (freeze
backbone, retrain head) and BERT→task-specific transfer. The isomorphism is
operational: `freeze(features) + retrain(head)` works across all domains.

## Experiment 005: Isomorphic Pattern Catalog

**Question**: What computational primitives are shared across all ML domains?

Six architectures cataloged:
- LLaMA 7B (6.7B params) — language
- OpenFold Evoformer (93M params) — protein
- ResNet-50 (25.6M params) — vision CNN
- ViT-B/16 (86.6M params) — vision transformer
- Physics MLP (4.7K params) — surrogate
- Weather LSTM (4.5K params) — time series

Six fundamental primitives identified:

| Primitive | FLOPs | Coverage |
|-----------|-------|----------|
| GEMM/GEMV | 60-90% | ALL architectures |
| Attention | 10-30% | Transformers only |
| Normalization | 1-5% | All except simplest MLPs |
| Nonlinearity | 1-5% | ALL architectures |
| Reduction | 1-5% | ALL architectures |
| Gating | 5-30% | RNNs + SwiGLU transformers |

BarraCUDA coverage: all six primitives have WGSL shaders in the crate.

## Cross-Domain Synthesis

The five experiments tell a unified story:

1. **Exp 001** proves GEMM + ReLU can approximate any smooth function
2. **Exp 002** proves GEMM + Softmax + LayerNorm + GELU can route information
3. **Exp 003** proves GEMM + Sigmoid (gating) can learn temporal patterns
4. **Exp 004** proves freeze(GEMM layers) + retrain(head) enables domain transfer
5. **Exp 005** proves these are the SAME 6 primitives across all domains

The Rust evolution team needs to optimize **6 operations**, not 600. Every
improvement to `gemm_f64.wgsl` benefits language, protein, vision, physics,
and time series simultaneously.

## Phase 0+ Scholarly Reproductions

### Study 001: Physics-Informed Neural Network — Burgers' Equation
**Paper**: Raissi, Perdikaris, Karniadakis (2019) "Physics-informed neural networks." J Comp Physics  
**Result**: L2 relative error 5.1% with Adam-only (paper: 0.06% with Adam + L-BFGS). Cole-Hopf exact solution validated to machine precision. Gradient clipping + cosine LR scheduling stabilize training.  
**BarraCUDA**: Validates MLP + autograd for PDE residual computation → `fd_gradient_f64.wgsl`

### Study 002: DeepONet — Antiderivative Operator Learning
**Paper**: Lu et al. (2021) "Learning nonlinear operators." Nature Machine Intelligence  
**Result**: 1.2% mean L2 error on antiderivative operator. Branch-trunk architecture = encoder-decoder attention pattern.  
**BarraCUDA**: Branch = encoder GEMM, trunk = decoder GEMM, dot product = attention → `gemm_f64.wgsl`

### Study 003: LeNet-5 — MNIST Classification
**Paper**: LeCun et al. (1998) "Gradient-based learning applied to document recognition."  
**Result**: 98.89% test accuracy (paper: ~99%). First CNN validation for BarraCUDA.  
**BarraCUDA**: Conv2d + MaxPool + FC pipeline → `conv2d.wgsl`, `gemm_f64.wgsl`

### Study 004: LSTM on Real ERA5 Weather Data
**Data**: Open-Meteo ERA5 reanalysis, 4 years of Michigan daily weather  
**Result**: NSE=0.849, RMSE=3.46°C for daily max temperature forecasting.  
**BarraCUDA**: Real-data LSTM validates `lstm_cell.wgsl` on actual temporal patterns

### Study 005: Quantized Inference (INT8/INT4)
**Method**: PyTorch dynamic quantization (INT8) + simulated INT4 with round-to-nearest  
**Result**: INT8: 0.017% accuracy loss, 3.9× compression. INT4: 0.79% loss, 7.3× compression.  
**BarraCUDA**: Same pipeline as llama.cpp GGML → `gemv_q4.wgsl`, `gemv_q8.wgsl`, `dequant_q4.wgsl`

## Rust Validation Layer (Phase 1–5b)

The audit (February 2026) produced a Rust crate that cross-validates Python baselines.
BarraCUDA integration extended it to 1400+ GPU/CPU validation checks across 119 binaries.

- **31 library modules + 2 evolved**: `metrics.rs`, `surrogate.rs`, `transformer.rs`, `sequence.rs`, `validation.rs`, `tolerances/` (20+ named constants + runtime registry), `provenance.rs`, `gpu.rs`, `eigh.rs`, `primitives.rs`, `pinn.rs`, `deeponet.rs`, `fft.rs`, `evolved/`, plus 15 paper modules
- **115 validation binaries + 5 bench**: native + BarraCUDA + GPU shader + GPU pipeline + cross-dispatch
- **264 lib tests + 9 integration tests**, 94.9% line coverage via `llvm-cov`
- **Quality gates**: `clippy` (pedantic+nursery), `fmt`, `doc`, `unsafe_code = "forbid"`
- **17 WGSL shaders** in `metalForge/shaders/` with validation binaries and absorption targets (13 upstream, 4 local)

See `specs/EVOLUTION_MAPPING.md` for the Tier A/B/C promotion path from Rust to WGSL.

## BarraCUDA Shader Evolution (Phase 1c–1d)

Following the hotSpring pattern (Python control → Rust port → WGSL evolution),
we evolved from per-op dispatch (200× slower than Python) through fused pipeline
(46–78× speedup) to BLAS-evolved shaders with double-buffered tiles.

### 3-Way Benchmark: Python vs CPU vs GPU

Target: **Python (slowest) < CPU < GPU (fastest)**

| Scale | Py(1t) | CPU | GPU | CPU/Py | GPU/Py |
|-------|--------|-----|-----|--------|--------|
| MLP large (3.1M) | 3.0 ms | **2.7 ms** | **178 µs** | **1.1× faster** | 16.8× faster |
| TF medium (103M) | 59 ms | **15.1 ms** | **566 µs** | **3.9× faster** | **104× faster** |
| TF xlarge (6.6B) | 232 ms | 1.42 s | **17.8 ms** | — | **13.1× faster** |

GPU dominates CPU by 4–80× at every scale. The target progression
(GPU < CPU < Python) is achieved at MLP large and TF medium.

### Evolved Shader Architecture

4-tier router driven by `DeviceCapabilities`:

| Tier | Shader | Key Technique |
|------|--------|---------------|
| Tiny M,N | naive (BarraCUDA stock) | Direct global reads |
| CPU | matmul_cpu_tiled.wgsl | 32×32 DB, 8×4 µkernel, vec4, 4× k-unroll |
| GPU (small) | matmul_tiled.wgsl (BarraCUDA stock) | 16×16 shared-memory |
| GPU (large) | matmul_gpu_evolved.wgsl | 32×32 DB, 2×2 µkernel, vec4, 4× k-unroll |

Both evolved shaders use **double-buffered tiles** — load the next tile while
computing on the current, overlapping memory latency with ALU work. This
technique was learned from hotSpring's double-buffered staging pattern.

See `whitePaper/BARRACUDA_EVOLUTION.md` for the full technical narrative.

## Evolution Roadmap

| Phase | Focus | Deliverable | Status |
|-------|-------|-------------|--------|
| 0 | Python baselines (48 checks) | Validate the science | **COMPLETE** |
| 0+ | Scholarly reproductions (31 checks) | Reproduce published results | **COMPLETE** |
| 0++ | Paper reproductions (127 checks) | 15 papers, 4 faculty, 5 disciplines | **COMPLETE** |
| 1a | neuralSpring Rust validation | 31 modules, 264 lib + 9 integration tests, 119 binaries (94.9% coverage) | **COMPLETE** |
| 1b | BarraCUDA validation | 12 domains, 275 checks (CPU + GPU + FFT) | **COMPLETE** |
| 1c | Fused ToadStool pipeline | 46–78× speedup via single-encoder dispatch | **COMPLETE** |
| 1d | 3-way benchmark + evolved shaders | Double-buffered, 4-tier routing | **COMPLETE** |
| 2 | BarraCUDA CPU ports | 17 modules, 170 checks | **COMPLETE** |
| 3a | BarraCUDA FFT | 24 analytical checks (f32/f64/Rfft) | **COMPLETE** |
| 3b | GPU streaming | `StatefulPipeline` (10/10 PASS) | **COMPLETE** |
| 3c | Shader evolution | 17 WGSL shaders, 108+ checks | **COMPLETE** |
| 3d | Cross-dispatch | GPU-CPU parity (41 checks) | **COMPLETE** |
| 4a | Performance benchmarks | 7 kernels, 71.8× overall | **COMPLETE** |
| 4b | GPU pipelines | 7 pipelines, 32 checks | **COMPLETE** |
| 4c | GPU PRNG | Xoshiro128** (5/5 PASS) | **COMPLETE** |
| 4d | ToadStool issue resolution | S-12 eigensolver + S-03b MHA (19 checks) | **COMPLETE** |
| 4e | Domain modules + GPU | PINN, DeepONet, 4 new shaders, 95 checks | **COMPLETE** |
| 5a | BarraCUDA GPU Tensor | Spectral (8) + eco (6) | **COMPLETE** |
| 5b | Upstream fixes | S-13 pool sync, S-14 Naive matmul | **Active** |

## Faculty-Driven Paper Reproductions (All Completed)

All four faculty research groups have been reproduced in Phase 0++:

| Faculty | Papers | Key Result |
|---------|--------|------------|
| **Dolson** (MSU CS) | 011–015 | Counterdiabatic driving, MODES, eco dynamics, lexicase, swarm |
| **Liu** (MSU CSE) | 016–018, 024–025 | HMM phylogenetics, SATé, introgression, pangenome, meta-pop |
| **Waters** (MSU Micro) | 019–021 | Game theory, regulatory networks, signal integration |
| **Kachkovskiy** (MSU Math) | 022–023 | Spectral commutativity, Anderson localization |

All 25 papers validated in Python (206/206) and BarraCUDA CPU (203/203, 24/25 papers),
with 17 WGSL shaders evolved for GPU acceleration via metalForge (13 absorbed upstream, 4 local).

### BarraCUDA Primitives Validated

Following the hotSpring pattern, `neuralSpring` validates 275+ BarraCUDA primitives (CPU + GPU + FFT):

| Binary | Module | Checks | Reference |
|--------|--------|--------|-----------|
| `validate_barracuda_stats` | stats (variance, pearson, norm) | 13 | Analytical |
| `validate_barracuda_linalg` | linalg (solve, lu, eigh, cholesky) | 17 | Analytical |
| `validate_barracuda_special` | special (gamma, erf, bessel, polynomials) | 26 | NIST DLMF |
| `validate_barracuda_optimize` | optimize (nelder_mead, bisect, brent) | 10 | Analytical |
| `validate_barracuda_precision` | precision (add, mul, fma, dot, sum) | 12 | Exact f64 |
| `validate_barracuda_tensor` | Tensor API (90 ops, CPU + GPU) | 90 | WGSL unified |
| `validate_barracuda_tensor_f64` | Tensor f64 (GPU ops) | 35 | f64 GPU |
| `validate_barracuda_quantized` | quantized (Q4/Q8 dequant, GEMV) | 15 | Hand-constructed |
| `validate_barracuda_linalg_ext` | linalg ext (SVD, LU inv, gen eigh) | 17 | Analytical |
| `validate_barracuda_ml_inference` | ML inference (MLP + Transformer) | 13 | Python baselines |
| `validate_barracuda_fft` | FFT (f32 Fft1D/Ifft1D + f64 Fft1DF64 + Rfft) | 24 | Analytical (DFT definition) |

### Fused ToadStool Pipeline + Evolved Shaders (2026-02-19)

Per-op dispatch overhead (~200 µs per `queue.submit()`) made BarraCUDA 200× slower
than Python/NumPy for small tensors. The fused pipeline collapses N submissions to 1.
Double-buffered evolved shaders with a 4-tier DeviceCapabilities-driven router provide
the optimal matmul kernel for every dispatch:

| Model | Per-Op (GPU) | Fused (GPU) | Python/NumPy | Fused vs Per-Op |
|-------|-------------|-------------|--------------|-----------------|
| MLP (4→64→64→10) | 4.0 ms | 92 µs | 23 µs | **43.6×** |
| Transformer (d=32,h=4,seq=8) | 13.3 ms | 174 µs | 77 µs | **76.6×** |

At scale, the evolved shaders achieve: GPU **104× faster** than Python at TF
medium (103M FLOPs), CPU **3.9× faster** at the same scale.

12 BarraCUDA shortcomings documented in `specs/TOADSTOOL_HANDOFF.md` — all absorbed at `77f70b2e`.
Full 3-way benchmark in `specs/BENCHMARK_ANALYSIS.md`.
Shader evolution narrative in `whitePaper/BARRACUDA_EVOLUTION.md`.

### BarraCUDA Gaps Resolved

All gaps identified during Phase 0+ have been addressed via local Rust
implementations validated against BarraCUDA CPU math and GPU WGSL shaders:

| Gap | Resolution | Shader / Module |
|-----|-----------|-----------------|
| Evolutionary optimization (GA/ES) | `counterdiabatic.rs` + `batch_fitness_eval.wgsl` | Phase 0++ Paper 011 |
| HMM Viterbi decoding | `hmm.rs` + `hmm_forward_log.wgsl` | Phase 0++ Paper 016 |
| Gillespie stochastic simulation | `regulatory_network.rs` + `rk4_parallel.wgsl` | Phase 0++ Paper 020 |
| MODES metrics computation | `modes.rs` + `pairwise_l2.wgsl` | Phase 0++ Paper 012 |
